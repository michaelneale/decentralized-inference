use super::{Node, NodeRole, current_time_unix_ms};
use crate::crypto::verify_control_plane_target_node;
use crate::mesh::node_requirements::preflight_pushed_config_for_current_node;
use crate::mesh::owner_control_response;
use crate::mesh::owner_lifecycle_cache::OwnerLifecycleResponseReservation;
use crate::protocol::{
    ControlFrameError, NODE_PROTOCOL_GENERATION, ValidateControlFrame, ensure_control_frame_size,
    read_len_prefixed, write_len_prefixed,
};
use anyhow::Result;
use iroh::EndpointId;
use prost::Message;
use std::future::Future;
use std::sync::Arc;

mod commands;

use commands::{OwnedNodeCommand, OwnedNodeCommandDeadline, OwnedNodeCommandExecutionShape};

const OWNER_CONTROL_SERVER_HANDSHAKE_TIMEOUT_SECS: u64 = 2;
const OWNER_CONTROL_SERVER_REQUEST_TIMEOUT_SECS: u64 = 5;
const OWNER_CONTROL_SERVER_RESPONSE_WRITE_TIMEOUT_SECS: u64 = 2;
const OWNER_CONTROL_STREAM_RESET_ERROR_CODE: u32 = 0;

pub(crate) fn endpoint_id_hex(id: EndpointId) -> String {
    hex::encode(id.as_bytes())
}

pub(crate) fn new_plugin_message_id(source_peer_id: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{source_peer_id}:{nanos}:{}", rand::random::<u64>())
}

pub(crate) fn node_role_label(role: &NodeRole) -> String {
    match role {
        NodeRole::Worker => "worker".into(),
        NodeRole::Host { .. } => "host".into(),
        NodeRole::Client => "client".into(),
    }
}

pub(crate) fn owner_control_error_envelope(
    code: crate::proto::node::OwnerControlErrorCode,
    request_id: Option<u64>,
    current_revision: Option<u64>,
    message: impl Into<String>,
) -> crate::proto::node::OwnerControlEnvelope {
    crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: None,
        error: Some(crate::proto::node::OwnerControlError {
            code: code as i32,
            message: message.into(),
            request_id,
            current_revision,
        }),
    }
}

pub(crate) fn owner_control_rejection_envelope(
    data: &[u8],
    request_id: Option<u64>,
    err: &ControlFrameError,
) -> crate::proto::node::OwnerControlEnvelope {
    let code = if matches!(err, ControlFrameError::MissingControlCommand) {
        crate::proto::node::OwnerControlErrorCode::UnknownCommand
    } else if serde_json::from_slice::<serde_json::Value>(data).is_ok() {
        crate::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
    } else {
        crate::proto::node::OwnerControlErrorCode::BadRequest
    };
    owner_control_error_envelope(code, request_id, None, err.to_string())
}

fn bound_owner_control_envelope(
    envelope: crate::proto::node::OwnerControlEnvelope,
) -> crate::proto::node::OwnerControlEnvelope {
    if ensure_control_frame_size(&envelope.encode_to_vec()).is_ok() {
        return envelope;
    }
    let request_id = envelope
        .response
        .as_ref()
        .map(|response| response.request_id)
        .or_else(|| envelope.error.as_ref().and_then(|error| error.request_id));
    owner_control_error_envelope(
        crate::proto::node::OwnerControlErrorCode::ControlUnavailable,
        request_id,
        None,
        "owner-control response exceeds the maximum frame size",
    )
}

fn owner_control_command_timeout_envelope(
    request_id: u64,
    deadline: OwnedNodeCommandDeadline,
) -> crate::proto::node::OwnerControlEnvelope {
    owner_control_error_envelope(
        crate::proto::node::OwnerControlErrorCode::ControlUnavailable,
        Some(request_id),
        None,
        deadline.timeout_message(),
    )
}

async fn await_owner_control_response_write<F>(future: F) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<()>>,
{
    tokio::time::timeout(
        std::time::Duration::from_secs(OWNER_CONTROL_SERVER_RESPONSE_WRITE_TIMEOUT_SECS),
        future,
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "owner-control response write timed out after {OWNER_CONTROL_SERVER_RESPONSE_WRITE_TIMEOUT_SECS}s"
        )
    })?
}

fn apply_owner_control_config_with_persistence<F>(
    config_state: Arc<tokio::sync::Mutex<crate::runtime::config_state::ConfigState>>,
    mesh_config: crate::plugin::MeshConfig,
    expected_revision: u64,
    persist: F,
) -> (crate::runtime::config_state::ApplyResult, u64, [u8; 32])
where
    F: FnOnce(
        &crate::runtime::config_state::PendingConfigApply,
    ) -> crate::runtime::config_state::ConfigPersistence,
{
    use crate::runtime::config_state::apply_prepared_config_with_live_logging;

    // Applies must remain serialized from revision preflight through commit so
    // a stale request cannot persist after a newer revision wins. The lock is
    // independent from `config_state`, letting snapshots and watches proceed
    // while the synchronous config and sidecar writes run.
    let apply_serialization_lock = {
        let state = config_state.blocking_lock();
        state.apply_serialization_lock()
    };
    let _apply_serialization_guard = apply_serialization_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let preparation = {
        let state = config_state.blocking_lock();
        state.prepare_apply(mesh_config, expected_revision)
    };
    let result =
        apply_prepared_config_with_live_logging(preparation, persist, |pending, persistence| {
            let mut state = config_state.blocking_lock();
            state.finish_apply(pending, persistence)
        });
    let state = config_state.blocking_lock();
    let current_revision = state.revision();
    let current_hash = *state.config_hash();
    (result, current_revision, current_hash)
}

async fn await_owner_control_terminal_completion<F, E, T>(future: F) -> anyhow::Result<()>
where
    F: Future<Output = Result<Option<T>, E>>,
    E: std::fmt::Display,
    T: std::fmt::Display,
{
    match tokio::time::timeout(
        std::time::Duration::from_secs(OWNER_CONTROL_SERVER_RESPONSE_WRITE_TIMEOUT_SECS),
        future,
    )
    .await
    {
        Ok(Ok(None)) => Ok(()),
        Ok(Ok(Some(error_code))) => {
            anyhow::bail!("owner-control terminal stream stopped by peer with code {error_code}")
        }
        Ok(Err(error)) => anyhow::bail!("owner-control terminal stream completion failed: {error}"),
        Err(_) => anyhow::bail!(
            "owner-control terminal stream completion timed out after {OWNER_CONTROL_SERVER_RESPONSE_WRITE_TIMEOUT_SECS}s"
        ),
    }
}

fn reset_owner_control_send_stream(send: &mut iroh::endpoint::SendStream) {
    let _ = send.reset(OWNER_CONTROL_STREAM_RESET_ERROR_CODE.into());
}

impl Node {
    pub(crate) async fn read_owner_control_handshake(
        &self,
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Result<Option<crate::proto::node::OwnerControlHandshake>> {
        let handshake_bytes = match tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_SERVER_HANDSHAKE_TIMEOUT_SECS),
            read_len_prefixed(recv),
        )
        .await
        {
            Err(_) => {
                let _ = self
                    .send_owner_control_terminal_envelope(
                        send,
                        owner_control_error_envelope(
                            crate::proto::node::OwnerControlErrorCode::InvalidHandshake,
                            None,
                            None,
                            format!(
                                "owner-control handshake timed out after {OWNER_CONTROL_SERVER_HANDSHAKE_TIMEOUT_SECS}s"
                            ),
                        ),
                    )
                    .await;
                return Ok(None);
            }
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => {
                tracing::debug!(
                    "control handshake read failed from {}: {error}",
                    remote.fmt_short()
                );
                return Ok(None);
            }
        };

        let handshake_envelope =
            match crate::proto::node::OwnerControlEnvelope::decode(handshake_bytes.as_slice()) {
                Ok(envelope) => envelope,
                Err(error) => {
                    let code =
                        if serde_json::from_slice::<serde_json::Value>(&handshake_bytes).is_ok() {
                            crate::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
                        } else {
                            crate::proto::node::OwnerControlErrorCode::InvalidHandshake
                        };
                    let _ = self
                        .send_owner_control_terminal_envelope(
                            send,
                            owner_control_error_envelope(code, None, None, error.to_string()),
                        )
                        .await;
                    return Ok(None);
                }
            };
        if let Err(error) = handshake_envelope.validate_frame() {
            let _ = self
                .send_owner_control_terminal_envelope(
                    send,
                    owner_control_error_envelope(
                        crate::proto::node::OwnerControlErrorCode::InvalidHandshake,
                        None,
                        None,
                        error.to_string(),
                    ),
                )
                .await;
            return Ok(None);
        }
        let Some(handshake) = handshake_envelope.handshake else {
            let _ = self
                .send_owner_control_terminal_envelope(
                    send,
                    owner_control_error_envelope(
                        crate::proto::node::OwnerControlErrorCode::InvalidHandshake,
                        None,
                        None,
                        "first owner-control envelope must be a handshake",
                    ),
                )
                .await;
            return Ok(None);
        };
        Ok(Some(handshake))
    }

    pub(crate) async fn read_owner_control_request(
        &self,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Result<Option<crate::proto::node::OwnerControlRequest>> {
        let request_bytes = match tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_SERVER_REQUEST_TIMEOUT_SECS),
            read_len_prefixed(recv),
        )
        .await
        {
            Err(_) => {
                let _ = self
                    .send_owner_control_terminal_envelope(
                        send,
                        owner_control_error_envelope(
                            crate::proto::node::OwnerControlErrorCode::BadRequest,
                            None,
                            None,
                            format!(
                                "owner-control request timed out after {OWNER_CONTROL_SERVER_REQUEST_TIMEOUT_SECS}s"
                            ),
                        ),
                    )
                    .await;
                return Ok(None);
            }
            Ok(Ok(bytes)) => bytes,
            Ok(Err(_)) => return Ok(None),
        };
        let envelope =
            match crate::proto::node::OwnerControlEnvelope::decode(request_bytes.as_slice()) {
                Ok(envelope) => envelope,
                Err(error) => {
                    let code =
                        if serde_json::from_slice::<serde_json::Value>(&request_bytes).is_ok() {
                            crate::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
                        } else {
                            crate::proto::node::OwnerControlErrorCode::BadRequest
                        };
                    let _ = self
                        .send_owner_control_terminal_envelope(
                            send,
                            owner_control_error_envelope(code, None, None, error.to_string()),
                        )
                        .await;
                    return Ok(None);
                }
            };
        if let Err(error) = envelope.validate_frame() {
            let request_id = envelope.request.as_ref().map(|request| request.request_id);
            let _ = self
                .send_owner_control_terminal_envelope(
                    send,
                    owner_control_rejection_envelope(&request_bytes, request_id, &error),
                )
                .await;
            return Ok(None);
        }
        let Some(request) = envelope.request else {
            let _ = self
                .send_owner_control_terminal_envelope(
                    send,
                    owner_control_error_envelope(
                        crate::proto::node::OwnerControlErrorCode::BadRequest,
                        None,
                        None,
                        "owner-control envelope must contain a request after handshake",
                    ),
                )
                .await;
            return Ok(None);
        };
        Ok(Some(request))
    }

    pub(crate) async fn handle_control_stream(
        &self,
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let Some(handshake) = self
            .read_owner_control_handshake(remote, send, recv)
            .await?
        else {
            return Ok(());
        };

        let local_owner = self.owner_summary.lock().await.clone();
        let trust_store = self.trust_store.lock().await.clone();
        if let Err(error) = crate::crypto::verify_control_plane_peer_ownership(
            &local_owner,
            handshake.ownership.as_ref(),
            remote.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        ) {
            let _ = self
                .send_owner_control_terminal_envelope(
                    send,
                    self.owner_control_auth_error_envelope(&error),
                )
                .await;
            return Ok(());
        }

        loop {
            let Some(request) = self.read_owner_control_request(send, recv).await? else {
                break;
            };
            let execution_shape = self
                .handle_owner_control_request(remote, send, recv, request)
                .await?;
            if execution_shape == OwnedNodeCommandExecutionShape::Watch {
                break;
            }
        }
        Ok(())
    }
}
impl Node {
    pub(crate) fn owner_control_snapshot_from_state(
        &self,
        state: &crate::runtime::config_state::ConfigState,
    ) -> crate::proto::node::OwnerControlConfigSnapshot {
        crate::proto::node::OwnerControlConfigSnapshot {
            node_id: self.endpoint.id().as_bytes().to_vec(),
            revision: state.revision(),
            config_hash: state.config_hash().to_vec(),
            config: Some(crate::protocol::convert::mesh_config_to_proto(
                state.config(),
            )),
            hostname: self.hostname.clone(),
        }
    }

    pub(crate) fn owner_control_update_from_state(
        &self,
        state: &crate::runtime::config_state::ConfigState,
    ) -> crate::proto::node::OwnerControlConfigUpdate {
        crate::proto::node::OwnerControlConfigUpdate {
            node_id: self.endpoint.id().as_bytes().to_vec(),
            revision: state.revision(),
            config_hash: state.config_hash().to_vec(),
            config: Some(crate::protocol::convert::mesh_config_to_proto(
                state.config(),
            )),
        }
    }

    pub(crate) async fn send_owner_control_envelope(
        &self,
        send: &mut iroh::endpoint::SendStream,
        envelope: crate::proto::node::OwnerControlEnvelope,
    ) -> anyhow::Result<()> {
        let envelope = bound_owner_control_envelope(envelope);
        let bytes = envelope.encode_to_vec();
        if let Err(error) =
            await_owner_control_response_write(write_len_prefixed(send, &bytes)).await
        {
            reset_owner_control_send_stream(send);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn send_owner_control_terminal_envelope(
        &self,
        send: &mut iroh::endpoint::SendStream,
        envelope: crate::proto::node::OwnerControlEnvelope,
    ) -> anyhow::Result<()> {
        self.send_owner_control_envelope(send, envelope).await?;
        if let Err(error) = send.finish() {
            reset_owner_control_send_stream(send);
            anyhow::bail!("owner-control terminal stream finish failed: {error}");
        }
        if let Err(error) = await_owner_control_terminal_completion(send.stopped()).await {
            reset_owner_control_send_stream(send);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn refresh_local_inventory_snapshot(
        &self,
    ) -> anyhow::Result<crate::runtime_data::InventoryScanOutcome> {
        self.refresh_local_inventory_snapshot_with(|| {
            Ok(crate::models::scan_local_inventory_snapshot_with_progress(
                |_| {},
            ))
        })
        .await
    }

    pub(crate) async fn refresh_local_inventory_snapshot_with<F>(
        &self,
        load: F,
    ) -> anyhow::Result<crate::runtime_data::InventoryScanOutcome>
    where
        F: FnOnce() -> crate::runtime_data::InventoryScanResult + Send + 'static,
    {
        let node = self.clone();
        tokio::spawn(async move {
            let outcome = node
                .runtime_data_collector()
                .coalesce_local_inventory_scan_outcome(load)
                .await
                .map_err(|error| anyhow::anyhow!(error))?;
            let mut models = outcome
                .snapshot
                .model_names
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            models.sort();
            node.set_available_models(models).await;
            Ok(outcome)
        })
        .await
        .map_err(|error| anyhow::anyhow!("inventory reconciliation task failed: {error}"))?
    }

    pub(crate) fn owner_control_auth_error_envelope(
        &self,
        err: &crate::crypto::ControlPlaneAuthError,
    ) -> crate::proto::node::OwnerControlEnvelope {
        let code = match err {
            crate::crypto::ControlPlaneAuthError::MissingRemoteOwnerAttestation
            | crate::crypto::ControlPlaneAuthError::RemoteOwnershipInvalid { .. } => {
                crate::proto::node::OwnerControlErrorCode::InvalidHandshake
            }
            crate::crypto::ControlPlaneAuthError::TargetNodeMismatch { .. } => {
                crate::proto::node::OwnerControlErrorCode::TargetNodeMismatch
            }
            crate::crypto::ControlPlaneAuthError::MissingLocalOwnerIdentity { .. }
            | crate::crypto::ControlPlaneAuthError::RemoteOwnerMismatch { .. }
            | crate::crypto::ControlPlaneAuthError::UnsupportedTrustPolicy { .. } => {
                crate::proto::node::OwnerControlErrorCode::Unauthorized
            }
        };
        owner_control_error_envelope(code, None, None, err.to_string())
    }

    pub(crate) fn verify_owner_control_request_ids(
        &self,
        remote: EndpointId,
        requester_node_id: &[u8],
        target_node_id: &[u8],
        request_id: u64,
    ) -> Result<(), Box<crate::proto::node::OwnerControlEnvelope>> {
        if requester_node_id != remote.as_bytes() {
            return Err(Box::new(owner_control_error_envelope(
                crate::proto::node::OwnerControlErrorCode::BadRequest,
                Some(request_id),
                None,
                "requester_node_id does not match connection identity",
            )));
        }
        if let Err(err) =
            verify_control_plane_target_node(target_node_id, self.endpoint.id().as_bytes())
        {
            return Err(Box::new(owner_control_error_envelope(
                crate::proto::node::OwnerControlErrorCode::TargetNodeMismatch,
                Some(request_id),
                None,
                err.to_string(),
            )));
        }
        Ok(())
    }

    pub(crate) async fn send_owner_control_request_id_error(
        &self,
        send: &mut iroh::endpoint::SendStream,
        verification: Result<(), Box<crate::proto::node::OwnerControlEnvelope>>,
    ) -> Option<anyhow::Result<()>> {
        match verification {
            Ok(()) => None,
            Err(envelope) => Some(self.send_owner_control_envelope(send, *envelope).await),
        }
    }

    pub(crate) async fn current_owner_control_snapshot(
        &self,
    ) -> crate::proto::node::OwnerControlConfigSnapshot {
        let state = self.config_state.lock().await;
        self.owner_control_snapshot_from_state(&state)
    }

    pub(crate) async fn current_owner_control_update(
        &self,
    ) -> crate::proto::node::OwnerControlConfigUpdate {
        let state = self.config_state.lock().await;
        self.owner_control_update_from_state(&state)
    }

    #[cfg(test)]
    pub(crate) async fn replace_config_state_for_test(
        &self,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let state = crate::runtime::config_state::ConfigState::load(path)?;
        let revision = state.revision();
        *self.config_state.lock().await = state;
        let _ = self.config_revision_tx.send(revision);
        Ok(())
    }

    pub(crate) fn owner_control_watch_response(
        &self,
        include_snapshot: bool,
        snapshot: Option<crate::proto::node::OwnerControlConfigSnapshot>,
        update: Option<crate::proto::node::OwnerControlConfigUpdate>,
    ) -> crate::proto::node::OwnerControlWatchConfigResponse {
        crate::proto::node::OwnerControlWatchConfigResponse {
            accepted: (!include_snapshot && update.is_none()).then(|| {
                crate::proto::node::OwnerControlWatchAccepted {
                    target_node_id: self.endpoint.id().as_bytes().to_vec(),
                }
            }),
            snapshot,
            update,
        }
    }

    pub(crate) fn owner_control_watch_envelope(
        &self,
        request_id: u64,
        watch_response: crate::proto::node::OwnerControlWatchConfigResponse,
    ) -> crate::proto::node::OwnerControlEnvelope {
        crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(crate::proto::node::OwnerControlResponse {
                request_id,
                get_config: None,
                watch_config: Some(watch_response),
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        }
    }

    pub(crate) async fn send_owner_control_watch_update(
        &self,
        send: &mut iroh::endpoint::SendStream,
        request_id: u64,
        update: crate::proto::node::OwnerControlConfigUpdate,
    ) -> anyhow::Result<()> {
        self.send_owner_control_envelope(
            send,
            self.owner_control_watch_envelope(
                request_id,
                self.owner_control_watch_response(false, None, Some(update)),
            ),
        )
        .await
    }

    pub(crate) async fn handle_owner_control_get_config(
        &self,
        send: &mut iroh::endpoint::SendStream,
        request_id: u64,
        _get: crate::proto::node::OwnerControlGetConfigRequest,
    ) -> anyhow::Result<()> {
        let snapshot = self.current_owner_control_snapshot().await;
        self.send_owner_control_envelope(
            send,
            crate::proto::node::OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: Some(crate::proto::node::OwnerControlResponse {
                    request_id,
                    get_config: Some(crate::proto::node::OwnerControlGetConfigResponse {
                        snapshot: Some(snapshot),
                    }),
                    watch_config: None,
                    apply_config: None,
                    refresh_inventory: None,
                    load_model: None,
                    unload_model: None,
                    ensure_model: None,
                    drain_model: None,
                    kv_cache: None,
                }),
                error: None,
            },
        )
        .await
    }

    pub(crate) async fn handle_owner_control_watch_config(
        &self,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
        remote: EndpointId,
        request_id: u64,
        watch: crate::proto::node::OwnerControlWatchConfigRequest,
    ) -> anyhow::Result<()> {
        let mut rev_rx = self.config_revision_tx.subscribe();
        self.send_owner_control_watch_start(send, request_id, watch.include_snapshot)
            .await?;

        self.stream_owner_control_watch_updates(send, recv, remote, request_id, &mut rev_rx)
            .await;

        Ok(())
    }

    pub(crate) async fn send_owner_control_watch_start(
        &self,
        send: &mut iroh::endpoint::SendStream,
        request_id: u64,
        include_snapshot: bool,
    ) -> anyhow::Result<()> {
        let watch_response = self.owner_control_watch_response(
            include_snapshot,
            if include_snapshot {
                Some(self.current_owner_control_snapshot().await)
            } else {
                None
            },
            None,
        );
        self.send_owner_control_envelope(
            send,
            self.owner_control_watch_envelope(request_id, watch_response),
        )
        .await?;

        Ok(())
    }

    pub(crate) async fn stream_owner_control_watch_updates(
        &self,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
        remote: EndpointId,
        request_id: u64,
        rev_rx: &mut tokio::sync::watch::Receiver<u64>,
    ) {
        loop {
            tokio::select! {
                changed = rev_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let update = self.current_owner_control_update().await;
                    if self
                        .send_owner_control_watch_update(send, request_id, update)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                inbound = read_len_prefixed(recv) => {
                    if inbound.is_ok() {
                        tracing::debug!(
                            "owner-control watch from {} sent unexpected extra frame; closing stream",
                            remote.fmt_short()
                        );
                    }
                    break;
                }
            }
        }
    }

    pub(crate) async fn handle_owner_control_apply_config(
        &self,
        send: &mut iroh::endpoint::SendStream,
        request_id: u64,
        apply: crate::proto::node::OwnerControlApplyConfigRequest,
    ) -> anyhow::Result<()> {
        use crate::runtime::config_state::{ApplyResult, ConfigApplyMode};

        let Some(config_snapshot) = apply.config.clone() else {
            return self
                .send_owner_control_envelope(
                    send,
                    owner_control_error_envelope(
                        crate::proto::node::OwnerControlErrorCode::BadRequest,
                        Some(request_id),
                        None,
                        "missing config payload",
                    ),
                )
                .await;
        };

        let mesh_config =
            match crate::protocol::convert::proto_config_to_mesh_strict(&config_snapshot) {
                Ok(config) => config,
                Err(error) => {
                    return self
                        .send_owner_control_envelope(
                            send,
                            owner_control_error_envelope(
                                crate::proto::node::OwnerControlErrorCode::BadRequest,
                                Some(request_id),
                                None,
                                error.to_string(),
                            ),
                        )
                        .await;
                }
            };
        let config_state = Arc::clone(&self.config_state);
        let expected_revision = apply.expected_revision;
        let apply_result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            preflight_pushed_config_for_current_node(&mesh_config)?;
            Ok(apply_owner_control_config_with_persistence(
                config_state,
                mesh_config,
                expected_revision,
                crate::runtime::config_state::PendingConfigApply::persist,
            ))
        })
        .await
        .map_err(|e| anyhow::anyhow!("config apply task panicked: {e}"))?;

        let (result, current_revision, current_hash) = match apply_result {
            Ok(values) => values,
            Err(error) => {
                return self
                    .send_owner_control_envelope(
                        send,
                        owner_control_error_envelope(
                            crate::proto::node::OwnerControlErrorCode::BadRequest,
                            Some(request_id),
                            None,
                            error.to_string(),
                        ),
                    )
                    .await;
            }
        };

        let envelope = match result {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
                diagnostics,
            } => {
                if apply_mode != ConfigApplyMode::Noop {
                    let _ = self.config_revision_tx.send(revision);
                }
                owner_control_response::apply_response_envelope(
                    request_id,
                    crate::proto::node::OwnerControlApplyConfigResponse {
                        success: true,
                        current_revision: revision,
                        config_hash: hash.to_vec(),
                        error: None,
                        apply_mode: owner_control_response::proto_apply_mode(apply_mode),
                        diagnostics: owner_control_response::config_diagnostics_to_proto(
                            &diagnostics,
                        ),
                    },
                )
            }
            ApplyResult::AppliedWithRestartRequired {
                revision,
                hash,
                diagnostics,
            } => {
                let _ = self.config_revision_tx.send(revision);
                owner_control_response::apply_response_envelope(
                    request_id,
                    crate::proto::node::OwnerControlApplyConfigResponse {
                        success: true,
                        current_revision: revision,
                        config_hash: hash.to_vec(),
                        error: None,
                        apply_mode: owner_control_response::proto_apply_mode(
                            ConfigApplyMode::Staged,
                        ),
                        diagnostics: owner_control_response::config_diagnostics_to_proto(
                            &diagnostics,
                        ),
                    },
                )
            }
            ApplyResult::RevisionConflict { current_revision } => owner_control_error_envelope(
                crate::proto::node::OwnerControlErrorCode::RevisionConflict,
                Some(request_id),
                Some(current_revision),
                "revision conflict: expected_revision does not match current",
            ),
            ApplyResult::PersistedWithRevisionTrackingError {
                revision,
                hash,
                error,
                diagnostics,
            } => {
                let _ = self.config_revision_tx.send(revision);
                owner_control_response::apply_response_envelope(
                    request_id,
                    crate::proto::node::OwnerControlApplyConfigResponse {
                        success: false,
                        current_revision: revision,
                        config_hash: hash.to_vec(),
                        error: Some(error),
                        apply_mode: crate::proto::node::ConfigApplyMode::Staged as i32,
                        diagnostics: owner_control_response::config_diagnostics_to_proto(
                            &diagnostics,
                        ),
                    },
                )
            }
            ApplyResult::ValidationError { error, diagnostics } => {
                owner_control_response::apply_response_envelope(
                    request_id,
                    crate::proto::node::OwnerControlApplyConfigResponse {
                        success: false,
                        current_revision,
                        config_hash: current_hash.to_vec(),
                        error: Some(error),
                        apply_mode: crate::proto::node::ConfigApplyMode::Unspecified as i32,
                        diagnostics: owner_control_response::config_diagnostics_to_proto(
                            &diagnostics,
                        ),
                    },
                )
            }
            ApplyResult::PersistError(error) => owner_control_response::apply_response_envelope(
                request_id,
                crate::proto::node::OwnerControlApplyConfigResponse {
                    success: false,
                    current_revision,
                    config_hash: current_hash.to_vec(),
                    error: Some(error),
                    apply_mode: crate::proto::node::ConfigApplyMode::Unspecified as i32,
                    diagnostics: Vec::new(),
                },
            ),
        };
        self.send_owner_control_envelope(send, envelope).await
    }

    pub(crate) async fn handle_owner_control_request(
        &self,
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        recv: &mut iroh::endpoint::RecvStream,
        request: crate::proto::node::OwnerControlRequest,
    ) -> anyhow::Result<OwnedNodeCommandExecutionShape> {
        let request_id = request.request_id;
        let Some(command) = OwnedNodeCommand::decode(request) else {
            self.send_owner_control_envelope(
                send,
                owner_control_error_envelope(
                    crate::proto::node::OwnerControlErrorCode::UnknownCommand,
                    Some(request_id),
                    None,
                    "unknown owner-control command",
                ),
            )
            .await?;
            return Ok(OwnedNodeCommandExecutionShape::Unary);
        };
        let execution_shape = command.execution_shape();
        let deadline = command.deadline();
        let command_request_id = command.request_id();
        let is_model_lifecycle = command.is_model_lifecycle();
        if let Some(result) = self
            .send_owner_control_request_id_error(
                send,
                self.verify_owner_control_request_ids(
                    remote,
                    command.requester_node_id(),
                    command.target_node_id(),
                    command.request_id(),
                ),
            )
            .await
        {
            result?;
            return Ok(execution_shape);
        }
        let mut lifecycle_leader = if is_model_lifecycle {
            let cache_duration = match deadline {
                commands::OwnedNodeCommandDeadline::Unary(duration) => duration,
                commands::OwnedNodeCommandDeadline::Scan(_)
                | commands::OwnedNodeCommandDeadline::Watch => {
                    anyhow::bail!("model lifecycle command must use a unary deadline")
                }
            };
            match self.owner_lifecycle_response_cache.reserve_with_fallback(
                remote,
                command_request_id,
                owner_control_command_timeout_envelope(command_request_id, deadline),
                cache_duration,
            ) {
                OwnerLifecycleResponseReservation::Ready(envelope) => {
                    self.send_owner_control_envelope(send, envelope).await?;
                    return Ok(execution_shape);
                }
                OwnerLifecycleResponseReservation::Follower(follower) => {
                    let envelope = follower.wait().await;
                    self.send_owner_control_envelope(send, envelope).await?;
                    return Ok(execution_shape);
                }
                OwnerLifecycleResponseReservation::Leader(leader) => Some(leader),
            }
        } else {
            None
        };

        let execution = async {
            match command {
                OwnedNodeCommand::GetConfig {
                    request_id,
                    request,
                } => {
                    self.handle_owner_control_get_config(send, request_id, request)
                        .await?
                }
                OwnedNodeCommand::WatchConfig {
                    request_id,
                    request,
                } => {
                    self.handle_owner_control_watch_config(send, recv, remote, request_id, request)
                        .await?
                }
                OwnedNodeCommand::ApplyConfig {
                    request_id,
                    request,
                } => {
                    self.handle_owner_control_apply_config(send, request_id, request)
                        .await?
                }
                OwnedNodeCommand::ScanRefresh { request_id, .. } => {
                    let envelope = commands::scan_refresh::execute(self, request_id).await;
                    self.send_owner_control_envelope(send, envelope).await?;
                }
                OwnedNodeCommand::LoadModel {
                    request_id,
                    request,
                } => {
                    let envelope =
                        commands::model_lifecycle::execute_load(self, request_id, request).await;
                    if let Some(leader) = lifecycle_leader.take() {
                        leader.publish(envelope.clone());
                    }
                    self.send_owner_control_envelope(send, envelope).await?;
                }
                OwnedNodeCommand::UnloadModel {
                    request_id,
                    request,
                } => {
                    let envelope =
                        commands::model_lifecycle::execute_unload(self, request_id, request).await;
                    if let Some(leader) = lifecycle_leader.take() {
                        leader.publish(envelope.clone());
                    }
                    self.send_owner_control_envelope(send, envelope).await?;
                }
                OwnedNodeCommand::EnsureModel {
                    request_id,
                    request,
                } => {
                    let envelope =
                        commands::model_lifecycle::execute_ensure(self, request_id, request).await;
                    if let Some(leader) = lifecycle_leader.take() {
                        leader.publish(envelope.clone());
                    }
                    self.send_owner_control_envelope(send, envelope).await?;
                }
                OwnedNodeCommand::DrainModel {
                    request_id,
                    request,
                } => {
                    let envelope =
                        commands::model_lifecycle::execute_drain(self, request_id, request).await;
                    if let Some(leader) = lifecycle_leader.take() {
                        leader.publish(envelope.clone());
                    }
                    self.send_owner_control_envelope(send, envelope).await?;
                }
                OwnedNodeCommand::KvCache {
                    request_id,
                    request,
                } => {
                    let envelope = commands::kv_cache::execute(request_id, request).await;
                    self.send_owner_control_envelope(send, envelope).await?;
                }
            }
            anyhow::Ok(())
        };
        match commands::await_command_deadline(deadline, execution).await {
            Ok(result) => result?,
            Err(expired) => {
                self.send_owner_control_envelope(
                    send,
                    owner_control_command_timeout_envelope(command_request_id, expired),
                )
                .await?;
            }
        }
        Ok(execution_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn config_persistence_releases_config_state_mutex() {
        let directory = std::env::temp_dir().join(format!(
            "mesh-llm-owner-control-config-lock-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&directory).expect("create test directory");
        let config_path = directory.join("config.toml");
        let config_state = Arc::new(tokio::sync::Mutex::new(
            crate::runtime::config_state::ConfigState::load(&config_path)
                .expect("load config state"),
        ));
        let persistence_lock_check = Arc::clone(&config_state);
        let apply_state = Arc::clone(&config_state);
        let config = crate::plugin::MeshConfig {
            version: Some(1),
            ..Default::default()
        };

        let (result, revision, _hash) = tokio::task::spawn_blocking(move || {
            apply_owner_control_config_with_persistence(apply_state, config, 0, |pending| {
                assert!(
                    persistence_lock_check.try_lock().is_ok(),
                    "config_state mutex must be released while config persistence runs"
                );
                pending.persist()
            })
        })
        .await
        .expect("config apply task should not panic");

        assert!(
            matches!(
                result,
                crate::runtime::config_state::ApplyResult::Applied { revision: 1, .. }
            ),
            "config apply should commit after persistence: {result:?}"
        );
        assert_eq!(revision, 1);
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test(start_paused = true)]
    async fn response_write_timeout_is_bounded() {
        let result = await_owner_control_response_write(std::future::pending()).await;

        assert_eq!(
            result
                .expect_err("pending write should time out")
                .to_string(),
            "owner-control response write timed out after 2s"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn terminal_completion_timeout_is_bounded() {
        let result = await_owner_control_terminal_completion(std::future::pending::<
            Result<Option<u32>, anyhow::Error>,
        >())
        .await;

        assert_eq!(
            result
                .expect_err("pending terminal completion should time out")
                .to_string(),
            "owner-control terminal stream completion timed out after 2s"
        );
    }

    #[tokio::test]
    async fn terminal_completion_reports_peer_stop_code() {
        let result =
            await_owner_control_terminal_completion(async { Ok::<_, anyhow::Error>(Some(7u32)) })
                .await;

        assert_eq!(
            result
                .expect_err("peer stop code should be surfaced")
                .to_string(),
            "owner-control terminal stream stopped by peer with code 7"
        );
    }
}
