use crate::client::builder::MeshClient;
use crate::crypto::OwnerKeypair;
use crate::proto::node::{
    NodeConfigSnapshot, OwnerControlApplyConfigRequest, OwnerControlApplyConfigResponse,
    OwnerControlConfigSnapshot, OwnerControlConfigUpdate, OwnerControlDrainModelRequest,
    OwnerControlDrainModelResponse, OwnerControlEnsureModelRequest,
    OwnerControlEnsureModelResponse, OwnerControlEnvelope, OwnerControlError,
    OwnerControlErrorCode, OwnerControlGetConfigRequest, OwnerControlHandshake,
    OwnerControlKvCacheOperation, OwnerControlKvCacheRequest, OwnerControlKvCacheResponse,
    OwnerControlLoadModelRequest, OwnerControlLoadModelResponse, OwnerControlModelRef,
    OwnerControlRefreshInventory, OwnerControlRefreshInventoryRequest, OwnerControlRequest,
    OwnerControlResponse, OwnerControlUnloadModelRequest, OwnerControlUnloadModelResponse,
    OwnerControlWatchAccepted, OwnerControlWatchConfigRequest, OwnerControlWatchConfigResponse,
    SignedNodeOwnership,
};
use crate::protocol::{
    ALPN_CONTROL_V1, ALPN_V1, NODE_PROTOCOL_GENERATION, decode_owner_control_envelope,
    write_len_prefixed,
};
use anyhow::Context;
use base64::Engine;
use iroh::{Endpoint, EndpointAddr};
use prost::Message;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

const DEFAULT_NODE_CERT_LIFETIME_SECS: u64 = 7 * 24 * 60 * 60;
const NODE_OWNERSHIP_VERSION: u32 = 1;
const SIGNING_DOMAIN_TAG: &[u8] = b"mesh-llm-node-ownership-v1:";
const OWNER_CONTROL_CONNECT_TIMEOUT_SECS: u64 = 8;
const OWNER_CONTROL_OPEN_TIMEOUT_SECS: u64 = 2;
const OWNER_CONTROL_HANDSHAKE_TIMEOUT_SECS: u64 = 2;
const OWNER_CONTROL_REQUEST_WRITE_TIMEOUT_SECS: u64 = 2;
const OWNER_CONTROL_SERVER_UNARY_DEADLINE_SECS_FOR_CLIENT_MARGIN: u64 = 5;
const OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS: u64 =
    OWNER_CONTROL_SERVER_UNARY_DEADLINE_SECS_FOR_CLIENT_MARGIN + 5;
const OWNER_CONTROL_SERVER_KV_CACHE_DEADLINE_SECS_FOR_CLIENT_MARGIN: u64 = 30;
const OWNER_CONTROL_KV_CACHE_RESPONSE_TIMEOUT_SECS: u64 =
    OWNER_CONTROL_SERVER_KV_CACHE_DEADLINE_SECS_FOR_CLIENT_MARGIN + 5;
const OWNER_CONTROL_SERVER_SCAN_DEADLINE_SECS_FOR_CLIENT_MARGIN: u64 = 30;
const OWNER_CONTROL_INVENTORY_RESPONSE_TIMEOUT_SECS: u64 =
    OWNER_CONTROL_SERVER_SCAN_DEADLINE_SECS_FOR_CLIENT_MARGIN + 5;
const OWNER_CONTROL_WATCH_ACCEPT_TIMEOUT_SECS: u64 = 5;
const FAILED_BOOTSTRAP_CLOSE_TIMEOUT_MILLIS: u64 = 250;

fn owner_control_client_bind_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([0, 0, 0, 0], 0))
}

/// Explicit owner-control bootstrap policy for new config clients.
///
/// Negotiation matrix:
/// - new client + explicit control endpoint -> use `mesh-llm-control/1`; configured control
///   failures stay on the control lane and return structured errors.
/// - new client + no control endpoint -> return `ControlEndpointRequired`.
///
/// Config and inventory mutation is intentionally exclusive to `mesh-llm-control/1`.
/// The legacy mesh-plane config stream IDs remain reserved, but no client bootstrap path
/// falls back to them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlPlaneBootstrapOptions {
    control_endpoint: Option<String>,
    connect_timeout: std::time::Duration,
}

impl Default for ControlPlaneBootstrapOptions {
    fn default() -> Self {
        Self {
            control_endpoint: None,
            connect_timeout: std::time::Duration::from_secs(OWNER_CONTROL_CONNECT_TIMEOUT_SECS),
        }
    }
}

impl ControlPlaneBootstrapOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_control_endpoint(mut self, control_endpoint: impl Into<String>) -> Self {
        self.control_endpoint = Some(control_endpoint.into());
        self
    }

    pub fn control_endpoint(&self) -> Option<&str> {
        self.control_endpoint.as_deref()
    }

    /// Bound owner-control endpoint bootstrap without changing request deadlines.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn select_transport(
        &self,
    ) -> Result<ConfigTransportSelection, ControlPlaneNegotiationError> {
        match self.control_endpoint() {
            Some(endpoint) => Ok(ConfigTransportSelection::OwnerControl {
                endpoint: endpoint.to_string(),
                retry_policy: ControlPlaneRetryPolicy::NoSilentLegacyDowngrade,
            }),
            None => Err(ControlPlaneNegotiationError::endpoint_required()),
        }
    }

    pub fn configured_endpoint_failure(
        &self,
        code: OwnerControlErrorCode,
        message: impl Into<String>,
    ) -> ControlPlaneNegotiationError {
        debug_assert!(self.control_endpoint.is_some());
        ControlPlaneNegotiationError::structured(code, message, false)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigTransportSelection {
    OwnerControl {
        endpoint: String,
        retry_policy: ControlPlaneRetryPolicy,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlPlaneRetryPolicy {
    NoSilentLegacyDowngrade,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlPlaneNegotiationError {
    pub code: OwnerControlErrorCode,
    pub message: String,
    pub legacy_retry_allowed: bool,
}

impl fmt::Display for ControlPlaneNegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ControlPlaneNegotiationError {}

impl ControlPlaneNegotiationError {
    pub fn endpoint_required() -> Self {
        Self {
            code: OwnerControlErrorCode::ControlEndpointRequired,
            message: "owner-control endpoint must be provided explicitly".to_string(),
            legacy_retry_allowed: false,
        }
    }

    pub fn structured(
        code: OwnerControlErrorCode,
        message: impl Into<String>,
        legacy_retry_allowed: bool,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            legacy_retry_allowed,
        }
    }
}

#[derive(Debug, Error)]
pub enum ControlPlaneClientError {
    #[error(transparent)]
    Negotiation(#[from] ControlPlaneNegotiationError),
    #[error(transparent)]
    Remote(#[from] OwnerControlRemoteError),
    #[error("control transport error: {0}")]
    Transport(String),
    #[error("control protocol error: {0}")]
    Protocol(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerControlRemoteError {
    pub code: OwnerControlErrorCode,
    pub message: String,
    pub request_id: Option<u64>,
    pub current_revision: Option<u64>,
}

impl fmt::Display for OwnerControlRemoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for OwnerControlRemoteError {}

impl From<OwnerControlError> for OwnerControlRemoteError {
    fn from(error: OwnerControlError) -> Self {
        Self {
            code: OwnerControlErrorCode::try_from(error.code)
                .unwrap_or(OwnerControlErrorCode::BadRequest),
            message: error.message,
            request_id: error.request_id,
            current_revision: error.current_revision,
        }
    }
}

/// Control-plane bootstrap is explicit and out-of-band.
///
/// Callers either receive an owner-control session bound to a configured endpoint,
/// or a structured error. The client never performs a silent downgrade.
pub enum ControlPlaneConnection {
    OwnerControl(Box<OwnerControlClient>),
}

pub struct OwnerControlClient {
    endpoint_token: String,
    endpoint: Endpoint,
    connection: iroh::endpoint::Connection,
    owner_keypair: OwnerKeypair,
    next_request_id: AtomicU64,
}

pub struct OwnerControlWatchStream {
    send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
    request_id: u64,
    pending: Option<OwnerControlWatchEvent>,
    closed: bool,
}

pub enum OwnerControlWatchEvent {
    Accepted(OwnerControlWatchAccepted),
    Snapshot(OwnerControlConfigSnapshot),
    Update(OwnerControlConfigUpdate),
}

/// Result of a completed owner-control inventory scan.
///
/// Older servers return only the refreshed config snapshot. In that compatibility
/// case, `inventory` is `None` while the command itself still succeeds.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerControlScanRefreshResult {
    pub snapshot: OwnerControlConfigSnapshot,
    pub inventory: Option<OwnerControlRefreshInventory>,
}

impl MeshClient {
    /// Bootstrap config transport using the explicit owner-control endpoint policy.
    ///
    /// Owner-control endpoints are not discovered through gossip or status APIs;
    /// callers must provide them explicitly through out-of-band bootstrap.
    pub async fn connect_control_plane(
        &self,
        options: ControlPlaneBootstrapOptions,
    ) -> Result<ControlPlaneConnection, ControlPlaneClientError> {
        match options.select_transport()? {
            ConfigTransportSelection::OwnerControl { endpoint, .. } => {
                OwnerControlClient::connect(endpoint, self.config.owner_keypair.clone(), &options)
                    .await
                    .map(Box::new)
                    .map(ControlPlaneConnection::OwnerControl)
            }
        }
    }
}

fn validate_lifecycle_acceptance(
    operation: &str,
    intent_id: &str,
    accepted_state: &str,
    expected_state: &str,
    target: Option<&crate::proto::node::OwnerControlModelRef>,
    expected_model_ref: &str,
    expected_instance_id: Option<&str>,
) -> Result<(), ControlPlaneClientError> {
    if intent_id.is_empty() {
        return Err(ControlPlaneClientError::Protocol(format!(
            "owner-control {operation} response missing intent id"
        )));
    }
    if accepted_state != expected_state {
        return Err(ControlPlaneClientError::Protocol(format!(
            "owner-control {operation} response has invalid accepted state"
        )));
    }
    let target = target.ok_or_else(|| {
        ControlPlaneClientError::Protocol(format!(
            "owner-control {operation} response missing target"
        ))
    })?;
    if target.canonical_model_ref != expected_model_ref
        || target.instance_id.as_deref() != expected_instance_id
    {
        return Err(ControlPlaneClientError::Protocol(format!(
            "owner-control {operation} response target does not match request"
        )));
    }
    Ok(())
}

fn map_legacy_lifecycle_unsupported(
    operation: &str,
    error: ControlPlaneClientError,
) -> ControlPlaneClientError {
    const LEGACY_UNKNOWN_COMMAND_MESSAGE: &str =
        "owner control request requires exactly one command variant";
    match error {
        ControlPlaneClientError::Remote(mut remote)
            if matches!(
                remote.code,
                OwnerControlErrorCode::BadRequest | OwnerControlErrorCode::UnknownCommand
            ) && remote.message == LEGACY_UNKNOWN_COMMAND_MESSAGE =>
        {
            remote.code = OwnerControlErrorCode::ControlUnsupported;
            remote.message = format!("remote owner-control endpoint does not support {operation}");
            ControlPlaneClientError::Remote(remote)
        }
        other => other,
    }
}

fn map_kv_cache_unsupported(error: ControlPlaneClientError) -> ControlPlaneClientError {
    const LEGACY_UNKNOWN_COMMAND_MESSAGE: &str =
        "owner control request requires exactly one command variant";
    match error {
        // Only a server that does not know the command at all. A supported
        // server answers a malformed request with BadRequest, and reporting
        // that as "unsupported" would be a false capability result.
        ControlPlaneClientError::Remote(mut remote)
            if remote.code == OwnerControlErrorCode::UnknownCommand
                || (remote.code == OwnerControlErrorCode::BadRequest
                    && remote.message == LEGACY_UNKNOWN_COMMAND_MESSAGE) =>
        {
            remote.code = OwnerControlErrorCode::ControlUnsupported;
            remote.message =
                "remote owner-control endpoint does not support kv-cache operations".to_string();
            ControlPlaneClientError::Remote(remote)
        }
        other => other,
    }
}

enum LifecycleCommand {
    Load(OwnerControlLoadModelRequest),
    Unload(OwnerControlUnloadModelRequest),
    Ensure(OwnerControlEnsureModelRequest),
    Drain(OwnerControlDrainModelRequest),
}

impl LifecycleCommand {
    fn operation(&self) -> &'static str {
        match self {
            Self::Load(_) => "load_model",
            Self::Unload(_) => "unload_model",
            Self::Ensure(_) => "ensure_model",
            Self::Drain(_) => "drain_model",
        }
    }

    fn into_request(self, request_id: u64) -> OwnerControlRequest {
        let mut request = OwnerControlRequest {
            request_id,
            ..Default::default()
        };
        match self {
            Self::Load(command) => request.load_model = Some(command),
            Self::Unload(command) => request.unload_model = Some(command),
            Self::Ensure(command) => request.ensure_model = Some(command),
            Self::Drain(command) => request.drain_model = Some(command),
        }
        request
    }
}

impl OwnerControlClient {
    async fn connect(
        endpoint_token: String,
        owner_keypair: OwnerKeypair,
        options: &ControlPlaneBootstrapOptions,
    ) -> Result<Self, ControlPlaneClientError> {
        let control_addr = decode_endpoint_addr_token(&endpoint_token).map_err(|error| {
            ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(
                OwnerControlErrorCode::ControlUnavailable,
                format!("invalid owner-control endpoint token: {error}"),
            ))
        })?;
        let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .alpns(vec![ALPN_CONTROL_V1.to_vec()])
            .bind_addr(owner_control_client_bind_addr())
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        builder = builder.relay_mode(relay_mode_from_endpoint_addr(&control_addr));
        let endpoint = builder
            .bind()
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        if control_addr.relay_urls().next().is_some() {
            let _ = tokio::time::timeout(options.connect_timeout, endpoint.online()).await;
        }
        let connection = match tokio::time::timeout(
            options.connect_timeout,
            endpoint.connect(control_addr.clone(), ALPN_CONTROL_V1),
        )
        .await
        {
            Ok(Ok(connection)) => connection,
            Ok(Err(error)) => {
                let error =
                    configured_endpoint_connect_error(&endpoint, control_addr, options, error)
                        .await;
                close_failed_bootstrap_endpoint(&endpoint).await;
                return Err(error);
            }
            Err(_) => {
                close_failed_bootstrap_endpoint(&endpoint).await;
                return Err(ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(
                    OwnerControlErrorCode::ControlUnavailable,
                    format!(
                        "remote owner-control endpoint is unavailable or unreachable: connect timed out after {:.3}s",
                        options.connect_timeout.as_secs_f64()
                    ),
                )));
            }
        };
        Ok(Self {
            endpoint_token,
            endpoint,
            connection,
            owner_keypair,
            next_request_id: AtomicU64::new(1),
        })
    }

    pub fn endpoint_token(&self) -> &str {
        &self.endpoint_token
    }

    pub fn local_node_id(&self) -> [u8; 32] {
        *self.endpoint.id().as_bytes()
    }

    pub fn target_node_id(&self) -> [u8; 32] {
        *self.connection.remote_id().as_bytes()
    }

    pub async fn close(&self) {
        self.connection
            .close(0u32.into(), b"owner-control-client-close");
        self.endpoint.close().await;
    }

    pub async fn get_config(&self) -> Result<OwnerControlConfigSnapshot, ControlPlaneClientError> {
        let response = self
            .send_unary_request(
                std::time::Duration::from_secs(OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS),
                |request_id, requester_node_id, target_node_id| OwnerControlRequest {
                    request_id,
                    get_config: Some(OwnerControlGetConfigRequest {
                        requester_node_id,
                        target_node_id,
                    }),
                    watch_config: None,
                    apply_config: None,
                    refresh_inventory: None,
                    load_model: None,
                    unload_model: None,
                    ensure_model: None,
                    drain_model: None,
                    kv_cache: None,
                },
            )
            .await?;
        response
            .get_config
            .and_then(|response| response.snapshot)
            .ok_or_else(|| {
                ControlPlaneClientError::Protocol(
                    "owner-control get_config response missing snapshot payload".to_string(),
                )
            })
    }

    pub async fn apply_config(
        &self,
        expected_revision: u64,
        config: NodeConfigSnapshot,
    ) -> Result<OwnerControlApplyConfigResponse, ControlPlaneClientError> {
        let response = self
            .send_unary_request(
                std::time::Duration::from_secs(OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS),
                |request_id, requester_node_id, target_node_id| OwnerControlRequest {
                    request_id,
                    get_config: None,
                    watch_config: None,
                    apply_config: Some(OwnerControlApplyConfigRequest {
                        requester_node_id,
                        target_node_id,
                        expected_revision,
                        config: Some(config),
                    }),
                    refresh_inventory: None,
                    load_model: None,
                    unload_model: None,
                    ensure_model: None,
                    drain_model: None,
                    kv_cache: None,
                },
            )
            .await?;
        response.apply_config.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control apply_config response missing apply payload".to_string(),
            )
        })
    }

    pub async fn refresh_inventory(
        &self,
    ) -> Result<OwnerControlConfigSnapshot, ControlPlaneClientError> {
        self.scan_refresh().await.map(|result| result.snapshot)
    }

    pub async fn scan_refresh(
        &self,
    ) -> Result<OwnerControlScanRefreshResult, ControlPlaneClientError> {
        let response = self
            .send_unary_request(
                std::time::Duration::from_secs(OWNER_CONTROL_INVENTORY_RESPONSE_TIMEOUT_SECS),
                |request_id, requester_node_id, target_node_id| OwnerControlRequest {
                    request_id,
                    get_config: None,
                    watch_config: None,
                    apply_config: None,
                    refresh_inventory: Some(OwnerControlRefreshInventoryRequest {
                        requester_node_id,
                        target_node_id,
                    }),
                    load_model: None,
                    unload_model: None,
                    ensure_model: None,
                    drain_model: None,
                    kv_cache: None,
                },
            )
            .await?;
        let response = response.refresh_inventory.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control refresh_inventory response missing refresh payload".to_string(),
            )
        })?;
        let snapshot = response.snapshot.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control refresh_inventory response missing snapshot payload".to_string(),
            )
        })?;
        Ok(OwnerControlScanRefreshResult {
            snapshot,
            inventory: response.inventory,
        })
    }

    pub async fn kv_cache(
        &self,
        operation: OwnerControlKvCacheOperation,
        target_bytes: Option<u64>,
        model_identity: Option<String>,
    ) -> Result<OwnerControlKvCacheResponse, ControlPlaneClientError> {
        let response = self
            .send_unary_request(
                std::time::Duration::from_secs(OWNER_CONTROL_KV_CACHE_RESPONSE_TIMEOUT_SECS),
                |request_id, requester_node_id, target_node_id| OwnerControlRequest {
                    request_id,
                    kv_cache: Some(OwnerControlKvCacheRequest {
                        requester_node_id,
                        target_node_id,
                        operation: operation as i32,
                        target_bytes,
                        model_identity,
                    }),
                    ..Default::default()
                },
            )
            .await
            .map_err(map_kv_cache_unsupported)?;
        let response = response.kv_cache.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control kv_cache response missing payload".to_string(),
            )
        })?;
        serde_json::from_slice::<serde_json::Value>(&response.status_json).map_err(|error| {
            ControlPlaneClientError::Protocol(format!(
                "owner-control kv_cache response has invalid status JSON: {error}"
            ))
        })?;
        Ok(response)
    }

    pub async fn load_model(
        &self,
        model_ref: String,
        profile: Option<String>,
    ) -> Result<OwnerControlLoadModelResponse, ControlPlaneClientError> {
        let expected_model_ref = model_ref.clone();
        let response = self
            .send_lifecycle_request(LifecycleCommand::Load(OwnerControlLoadModelRequest {
                requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                model: Some(OwnerControlModelRef {
                    canonical_model_ref: model_ref,
                    instance_id: None,
                }),
                profile,
            }))
            .await?;
        let response = response.load_model.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control load_model response missing payload".to_string(),
            )
        })?;
        validate_lifecycle_acceptance(
            "load_model",
            &response.intent_id,
            &response.accepted_state,
            "present",
            response.target.as_ref(),
            &expected_model_ref,
            None,
        )?;
        Ok(response)
    }

    pub async fn unload_model(
        &self,
        model_ref: String,
        instance_id: Option<String>,
    ) -> Result<OwnerControlUnloadModelResponse, ControlPlaneClientError> {
        let (expected_model_ref, expected_instance_id) =
            validate_absent_model_target(model_ref, instance_id)?;
        let response = self
            .send_lifecycle_request(LifecycleCommand::Unload(OwnerControlUnloadModelRequest {
                requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                model: Some(OwnerControlModelRef {
                    canonical_model_ref: expected_model_ref.clone(),
                    instance_id: expected_instance_id.clone(),
                }),
            }))
            .await?;
        let response = response.unload_model.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control unload_model response missing payload".to_string(),
            )
        })?;
        validate_lifecycle_acceptance(
            "unload_model",
            &response.intent_id,
            &response.accepted_state,
            "absent",
            response.target.as_ref(),
            &expected_model_ref,
            expected_instance_id.as_deref(),
        )?;
        Ok(response)
    }

    pub async fn ensure_model(
        &self,
        model_ref: String,
        profile: Option<String>,
    ) -> Result<OwnerControlEnsureModelResponse, ControlPlaneClientError> {
        let expected_model_ref = model_ref.clone();
        let response = self
            .send_lifecycle_request(LifecycleCommand::Ensure(OwnerControlEnsureModelRequest {
                requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                model: Some(OwnerControlModelRef {
                    canonical_model_ref: model_ref,
                    instance_id: None,
                }),
                profile,
            }))
            .await?;
        let response = response.ensure_model.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control ensure_model response missing payload".to_string(),
            )
        })?;
        validate_lifecycle_acceptance(
            "ensure_model",
            &response.intent_id,
            &response.accepted_state,
            "present",
            response.target.as_ref(),
            &expected_model_ref,
            None,
        )?;
        Ok(response)
    }

    pub async fn drain_model(
        &self,
        model_ref: String,
        instance_id: Option<String>,
    ) -> Result<OwnerControlDrainModelResponse, ControlPlaneClientError> {
        let (expected_model_ref, expected_instance_id) =
            validate_absent_model_target(model_ref, instance_id)?;
        let response = self
            .send_lifecycle_request(LifecycleCommand::Drain(OwnerControlDrainModelRequest {
                requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                model: Some(OwnerControlModelRef {
                    canonical_model_ref: expected_model_ref.clone(),
                    instance_id: expected_instance_id.clone(),
                }),
                drain_timeout_secs: None,
            }))
            .await?;
        let response = response.drain_model.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control drain_model response missing payload".to_string(),
            )
        })?;
        validate_lifecycle_acceptance(
            "drain_model",
            &response.intent_id,
            &response.accepted_state,
            "draining",
            response.target.as_ref(),
            &expected_model_ref,
            expected_instance_id.as_deref(),
        )?;
        Ok(response)
    }

    async fn send_lifecycle_request(
        &self,
        command: LifecycleCommand,
    ) -> Result<OwnerControlResponse, ControlPlaneClientError> {
        let operation = command.operation();
        self.send_unary_request(
            std::time::Duration::from_secs(OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS),
            move |request_id, _, _| command.into_request(request_id),
        )
        .await
        .map_err(|error| map_legacy_lifecycle_unsupported(operation, error))
    }

    pub async fn watch_config(
        &self,
        include_snapshot: bool,
    ) -> Result<OwnerControlWatchStream, ControlPlaneClientError> {
        let request_id = self.next_request_id();
        let (mut send, recv) = self.open_authenticated_stream().await?;
        let envelope = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id,
                get_config: None,
                watch_config: Some(OwnerControlWatchConfigRequest {
                    requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                    target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                    include_snapshot,
                }),
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        };
        write_owner_control_request(&mut send, &envelope).await?;
        let mut stream = OwnerControlWatchStream {
            send,
            recv,
            request_id,
            pending: None,
            closed: false,
        };
        let accepted = tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_WATCH_ACCEPT_TIMEOUT_SECS),
            stream.next(),
        )
        .await
        .map_err(|_| {
            ControlPlaneClientError::Transport(format!(
                "owner-control watch accept timed out after {OWNER_CONTROL_WATCH_ACCEPT_TIMEOUT_SECS}s"
            ))
        })??;
        stream.pending = Some(accepted);
        Ok(stream)
    }

    async fn send_unary_request<F>(
        &self,
        response_timeout: std::time::Duration,
        build_request: F,
    ) -> Result<OwnerControlResponse, ControlPlaneClientError>
    where
        F: FnOnce(u64, Vec<u8>, Vec<u8>) -> OwnerControlRequest,
    {
        let request_id = self.next_request_id();
        let (mut send, mut recv) = self.open_authenticated_stream().await?;
        let envelope = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(build_request(
                request_id,
                self.endpoint.id().as_bytes().to_vec(),
                self.connection.remote_id().as_bytes().to_vec(),
            )),
            response: None,
            error: None,
        };
        write_owner_control_request(&mut send, &envelope).await?;
        let envelope =
            tokio::time::timeout(response_timeout, read_owner_control_message(&mut recv))
                .await
                .map_err(|_| {
                    ControlPlaneClientError::Transport(format!(
                        "owner-control unary response timed out after {}s",
                        response_timeout.as_secs()
                    ))
                })??;
        let _ = send.finish();
        decode_response_envelope(request_id, envelope)
    }

    fn next_request_id(&self) -> u64 {
        next_nonzero_request_id(&self.next_request_id)
    }

    async fn open_authenticated_stream(
        &self,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream), ControlPlaneClientError>
    {
        let (mut send, recv) = tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_OPEN_TIMEOUT_SECS),
            self.connection.open_bi(),
        )
        .await
        .map_err(|_| {
            ControlPlaneClientError::Transport(format!(
                "owner-control stream open timed out after {OWNER_CONTROL_OPEN_TIMEOUT_SECS}s"
            ))
        })?
        .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        let handshake = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: Some(OwnerControlHandshake {
                ownership: Some(sign_node_ownership_proto(
                    &self.owner_keypair,
                    self.endpoint.id().as_bytes(),
                )),
            }),
            request: None,
            response: None,
            error: None,
        };
        tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_HANDSHAKE_TIMEOUT_SECS),
            write_len_prefixed(&mut send, &handshake.encode_to_vec()),
        )
        .await
        .map_err(|_| {
            ControlPlaneClientError::Transport(format!(
                "owner-control handshake timed out after {OWNER_CONTROL_HANDSHAKE_TIMEOUT_SECS}s"
            ))
        })?
        .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        Ok((send, recv))
    }
}

fn validate_absent_model_target(
    model_ref: String,
    instance_id: Option<String>,
) -> Result<(String, Option<String>), ControlPlaneClientError> {
    let has_model_ref = !model_ref.trim().is_empty();
    let has_instance_id = instance_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());

    match (has_model_ref, has_instance_id) {
        (true, false) => Ok((model_ref, None)),
        (false, true) => Ok((String::new(), instance_id)),
        _ => Err(ControlPlaneClientError::Protocol(
            "unload_model and drain_model require exactly one model reference or instance id"
                .to_string(),
        )),
    }
}

async fn close_failed_bootstrap_endpoint(endpoint: &Endpoint) {
    let _ = tokio::time::timeout(
        std::time::Duration::from_millis(FAILED_BOOTSTRAP_CLOSE_TIMEOUT_MILLIS),
        endpoint.close(),
    )
    .await;
}

async fn write_owner_control_request(
    send: &mut iroh::endpoint::SendStream,
    envelope: &OwnerControlEnvelope,
) -> Result<(), ControlPlaneClientError> {
    tokio::time::timeout(
        std::time::Duration::from_secs(OWNER_CONTROL_REQUEST_WRITE_TIMEOUT_SECS),
        write_len_prefixed(send, &envelope.encode_to_vec()),
    )
    .await
    .map_err(|_| {
        ControlPlaneClientError::Transport(format!(
            "owner-control request write timed out after {OWNER_CONTROL_REQUEST_WRITE_TIMEOUT_SECS}s"
        ))
    })?
    .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))
}

impl OwnerControlWatchStream {
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    pub async fn next(&mut self) -> Result<OwnerControlWatchEvent, ControlPlaneClientError> {
        if let Some(event) = self.pending.take() {
            return Ok(event);
        }
        let envelope = read_owner_control_message(&mut self.recv).await?;
        let response = decode_response_envelope(self.request_id, envelope)?;
        let watch = response.watch_config.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control watch response missing watch_config payload".to_string(),
            )
        })?;
        decode_watch_event(watch)
    }

    pub async fn close(&mut self) -> Result<(), ControlPlaneClientError> {
        if self.closed {
            return Ok(());
        }
        self.send
            .finish()
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        self.closed = true;
        Ok(())
    }

    pub async fn cancel(&mut self) -> Result<(), ControlPlaneClientError> {
        self.close().await
    }
}

fn next_nonzero_request_id(counter: &AtomicU64) -> u64 {
    loop {
        let request_id = counter.fetch_add(1, Ordering::Relaxed);
        if request_id != 0 {
            return request_id;
        }
    }
}

impl Drop for OwnerControlWatchStream {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.send.finish();
            self.closed = true;
        }
    }
}

fn decode_watch_event(
    watch: OwnerControlWatchConfigResponse,
) -> Result<OwnerControlWatchEvent, ControlPlaneClientError> {
    if let Some(accepted) = watch.accepted {
        return Ok(OwnerControlWatchEvent::Accepted(accepted));
    }
    if let Some(snapshot) = watch.snapshot {
        return Ok(OwnerControlWatchEvent::Snapshot(snapshot));
    }
    if let Some(update) = watch.update {
        return Ok(OwnerControlWatchEvent::Update(update));
    }
    Err(ControlPlaneClientError::Protocol(
        "owner-control watch response missing accepted/snapshot/update payload".to_string(),
    ))
}

fn decode_response_envelope(
    expected_request_id: u64,
    envelope: OwnerControlEnvelope,
) -> Result<OwnerControlResponse, ControlPlaneClientError> {
    if let Some(error) = envelope.error {
        return Err(ControlPlaneClientError::Remote(error.into()));
    }
    let response = envelope.response.ok_or_else(|| {
        ControlPlaneClientError::Protocol(
            "owner-control response envelope missing response payload".to_string(),
        )
    })?;
    if response.request_id != expected_request_id {
        return Err(ControlPlaneClientError::Protocol(format!(
            "owner-control response request_id mismatch: expected {expected_request_id}, got {}",
            response.request_id
        )));
    }
    Ok(response)
}

async fn read_owner_control_message(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<OwnerControlEnvelope, ControlPlaneClientError> {
    let bytes = crate::protocol::read_len_prefixed(recv)
        .await
        .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
    decode_owner_control_envelope(&bytes)
        .map_err(|error| ControlPlaneClientError::Protocol(error.to_string()))
}

async fn configured_endpoint_connect_error(
    endpoint: &Endpoint,
    control_addr: EndpointAddr,
    options: &ControlPlaneBootstrapOptions,
    error: iroh::endpoint::ConnectError,
) -> ControlPlaneClientError {
    let message = error.to_string();
    let disposition = connect_error_probe_disposition(&error);
    let legacy_mesh_reachable = match disposition {
        ConnectProbeDisposition::ProbeLegacyMesh => legacy_mesh_probe(endpoint, control_addr).await,
        ConnectProbeDisposition::SkipUnavailable => false,
        ConnectProbeDisposition::Unsupported => true,
    };
    let (code, rendered) = if legacy_mesh_reachable {
        control_unsupported_message(&message)
    } else {
        control_unavailable_message(&message)
    };
    ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(code, rendered))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectProbeDisposition {
    SkipUnavailable,
    ProbeLegacyMesh,
    Unsupported,
}

fn connect_error_probe_disposition(
    error: &iroh::endpoint::ConnectError,
) -> ConnectProbeDisposition {
    match error {
        iroh::endpoint::ConnectError::Connect { source, .. } => match source {
            iroh::endpoint::ConnectWithOptsError::SelfConnect { .. }
            | iroh::endpoint::ConnectWithOptsError::NoAddress { .. }
            | iroh::endpoint::ConnectWithOptsError::Noq { .. }
            | iroh::endpoint::ConnectWithOptsError::InternalConsistencyError { .. }
            | iroh::endpoint::ConnectWithOptsError::LocallyRejected { .. }
            | iroh::endpoint::ConnectWithOptsError::EndpointClosed { .. } => {
                ConnectProbeDisposition::SkipUnavailable
            }
            _ => fallback_probe_disposition(&source.to_string()),
        },
        iroh::endpoint::ConnectError::Connecting { source, .. } => match source {
            iroh::endpoint::ConnectingError::ConnectionError { source, .. } => {
                connection_error_probe_disposition(&source.to_string())
            }
            iroh::endpoint::ConnectingError::HandshakeFailure { source, .. } => match source {
                iroh::endpoint::AuthenticationError::NoAlpn { .. } => {
                    ConnectProbeDisposition::Unsupported
                }
                iroh::endpoint::AuthenticationError::RemoteId { .. } => {
                    ConnectProbeDisposition::SkipUnavailable
                }
                _ => fallback_probe_disposition(&source.to_string()),
            },
            iroh::endpoint::ConnectingError::InternalConsistencyError { .. }
            | iroh::endpoint::ConnectingError::LocallyRejected { .. } => {
                ConnectProbeDisposition::SkipUnavailable
            }
            _ => fallback_probe_disposition(&source.to_string()),
        },
        iroh::endpoint::ConnectError::Connection { source, .. } => {
            connection_error_probe_disposition(&source.to_string())
        }
        _ => fallback_probe_disposition(&error.to_string()),
    }
}

fn connection_error_probe_disposition(message: &str) -> ConnectProbeDisposition {
    if is_alpn_mismatch_message(message) {
        ConnectProbeDisposition::Unsupported
    } else {
        ConnectProbeDisposition::SkipUnavailable
    }
}

fn fallback_probe_disposition(message: &str) -> ConnectProbeDisposition {
    if is_alpn_mismatch_message(message) {
        ConnectProbeDisposition::ProbeLegacyMesh
    } else {
        ConnectProbeDisposition::SkipUnavailable
    }
}

fn control_unsupported_message(message: &str) -> (OwnerControlErrorCode, String) {
    (
        OwnerControlErrorCode::ControlUnsupported,
        format!("remote endpoint did not negotiate mesh-llm-control/1: {message}"),
    )
}

fn control_unavailable_message(message: &str) -> (OwnerControlErrorCode, String) {
    (
        OwnerControlErrorCode::ControlUnavailable,
        format!("remote owner-control endpoint is unavailable or unreachable: {message}"),
    )
}

async fn legacy_mesh_probe(_endpoint: &Endpoint, control_addr: EndpointAddr) -> bool {
    let Ok(probe_endpoint) = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .relay_mode(relay_mode_from_endpoint_addr(&control_addr))
        .bind_addr(owner_control_client_bind_addr())
    else {
        return false;
    };
    let Ok(probe_endpoint) = probe_endpoint.bind().await else {
        return false;
    };
    if control_addr.relay_urls().next().is_some() {
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(3), probe_endpoint.online()).await;
    }
    let reachable = match tokio::time::timeout(
        std::time::Duration::from_secs(3),
        probe_endpoint.connect(control_addr, ALPN_V1),
    )
    .await
    {
        Ok(Ok(connection)) => {
            connection.close(0u32.into(), b"owner-control-legacy-probe-complete");
            true
        }
        _ => false,
    };
    probe_endpoint.close().await;
    reachable
}

fn relay_mode_from_endpoint_addr(addr: &EndpointAddr) -> iroh::endpoint::RelayMode {
    match relay_map_from_endpoint_addr(addr) {
        Some(relay_map) => iroh::endpoint::RelayMode::Custom(relay_map),
        None => iroh::endpoint::RelayMode::Disabled,
    }
}

fn is_alpn_mismatch_message(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("alpn mismatch")
        || lowered.contains("no application protocol")
        || lowered.contains("application protocol selected")
}

fn decode_endpoint_addr_token(invite_token: &str) -> anyhow::Result<EndpointAddr> {
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(invite_token)
        .context("invalid endpoint encoding")?;
    serde_json::from_slice(&json).context("invalid endpoint JSON")
}

fn relay_map_from_endpoint_addr(addr: &EndpointAddr) -> Option<iroh::RelayMap> {
    let configs: Vec<_> = addr
        .relay_urls()
        .cloned()
        // Preserve iroh's default QUIC Address Discovery (QAD). `new(url, None)`
        // disables it, preventing reflexive candidate discovery and direct-path
        // upgrades across NAT (see issue #1065). `RelayUrl::into()` keeps QAD on.
        .map(|url| -> iroh::RelayConfig { url.into() })
        .collect();
    if configs.is_empty() {
        None
    } else {
        Some(iroh::RelayMap::from_iter(configs))
    }
}

fn sign_node_ownership_proto(
    owner: &OwnerKeypair,
    node_endpoint_id: &[u8; 32],
) -> SignedNodeOwnership {
    let issued_at_unix_ms = current_time_unix_ms();
    let expires_at_unix_ms =
        issued_at_unix_ms + DEFAULT_NODE_CERT_LIFETIME_SECS.saturating_mul(1000);
    let cert_id = uuid::Uuid::new_v4().simple().to_string();
    let owner_sign_public_key = owner.verifying_key().as_bytes().to_vec();
    let owner_id = owner.owner_id();
    let signature_payload = canonical_claim_bytes(CanonicalClaim {
        version: NODE_OWNERSHIP_VERSION,
        cert_id: &cert_id,
        owner_id: &owner_id,
        owner_sign_public_key: &owner_sign_public_key,
        node_endpoint_id,
        issued_at_unix_ms,
        expires_at_unix_ms,
        node_label: None,
        hostname_hint: None,
    });
    SignedNodeOwnership {
        version: NODE_OWNERSHIP_VERSION,
        cert_id,
        owner_id,
        owner_sign_public_key,
        node_endpoint_id: node_endpoint_id.to_vec(),
        issued_at_unix_ms,
        expires_at_unix_ms,
        node_label: None,
        hostname_hint: None,
        signature: owner.sign_bytes(&signature_payload).to_vec(),
    }
}

struct CanonicalClaim<'a> {
    version: u32,
    cert_id: &'a str,
    owner_id: &'a str,
    owner_sign_public_key: &'a [u8],
    node_endpoint_id: &'a [u8; 32],
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    node_label: Option<&'a str>,
    hostname_hint: Option<&'a str>,
}

fn canonical_claim_bytes(claim: CanonicalClaim<'_>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(SIGNING_DOMAIN_TAG);
    buf.extend_from_slice(&claim.version.to_le_bytes());
    write_string(&mut buf, claim.cert_id);
    write_string(&mut buf, claim.owner_id);
    buf.extend_from_slice(claim.owner_sign_public_key);
    buf.extend_from_slice(claim.node_endpoint_id);
    buf.extend_from_slice(&claim.issued_at_unix_ms.to_le_bytes());
    buf.extend_from_slice(&claim.expires_at_unix_ms.to_le_bytes());
    write_optional_string(&mut buf, claim.node_label);
    write_optional_string(&mut buf, claim.hostname_hint);
    buf
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn owner_control_client_binds_wildcard_for_direct_remote_endpoints() {
        let bind_addr = owner_control_client_bind_addr();

        assert_eq!(bind_addr.port(), 0);
        assert!(
            bind_addr.ip().is_unspecified(),
            "owner-control clients must not be loopback-bound when dialing explicit remote endpoints"
        );
    }

    #[test]
    fn relay_mode_uses_custom_relays_from_endpoint_addr() {
        let addr = EndpointAddr::new(iroh::SecretKey::generate().public()).with_relay_url(
            iroh::RelayUrl::from_str("https://relay.example.com").expect("relay URL parses"),
        );

        assert!(matches!(
            relay_mode_from_endpoint_addr(&addr),
            iroh::endpoint::RelayMode::Custom(_)
        ));
    }

    #[test]
    fn endpoint_addr_relays_preserve_default_qad() {
        let addr = EndpointAddr::new(iroh::SecretKey::generate().public())
            .with_relay_url(
                iroh::RelayUrl::from_str("https://relay-a.example.com").expect("relay URL parses"),
            )
            .with_relay_url(
                iroh::RelayUrl::from_str("https://relay-b.example.com").expect("relay URL parses"),
            );

        let map = relay_map_from_endpoint_addr(&addr).expect("relay map should be enabled");
        let configs = map.relays::<Vec<_>>();

        assert_eq!(configs.len(), 2);
        assert!(
            configs
                .iter()
                .all(|config| { config.quic.as_ref().is_some_and(|quic| quic.port == 7842) })
        );
    }

    #[test]
    fn relay_mode_is_disabled_without_endpoint_relays() {
        let addr = EndpointAddr::new(iroh::SecretKey::generate().public());

        assert!(matches!(
            relay_mode_from_endpoint_addr(&addr),
            iroh::endpoint::RelayMode::Disabled
        ));
    }

    #[test]
    fn request_id_generator_skips_zero_after_wraparound() {
        let counter = AtomicU64::new(u64::MAX);

        assert_eq!(next_nonzero_request_id(&counter), u64::MAX);
        assert_eq!(next_nonzero_request_id(&counter), 1);
    }

    #[test]
    fn inventory_response_timeout_exceeds_server_scan_deadline() {
        const {
            assert!(
                OWNER_CONTROL_INVENTORY_RESPONSE_TIMEOUT_SECS
                    > OWNER_CONTROL_SERVER_SCAN_DEADLINE_SECS_FOR_CLIENT_MARGIN
            );
        }
        assert_eq!(
            OWNER_CONTROL_INVENTORY_RESPONSE_TIMEOUT_SECS
                - OWNER_CONTROL_SERVER_SCAN_DEADLINE_SECS_FOR_CLIENT_MARGIN,
            5
        );
    }

    #[test]
    fn unary_response_timeout_exceeds_server_command_deadline() {
        const {
            assert!(
                OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS
                    > OWNER_CONTROL_SERVER_UNARY_DEADLINE_SECS_FOR_CLIENT_MARGIN
            );
        }
        assert_eq!(
            OWNER_CONTROL_UNARY_RESPONSE_TIMEOUT_SECS
                - OWNER_CONTROL_SERVER_UNARY_DEADLINE_SECS_FOR_CLIENT_MARGIN,
            5
        );
    }

    #[test]
    fn lifecycle_acceptance_requires_intent_id_and_exact_state() {
        let target = crate::proto::node::OwnerControlModelRef {
            canonical_model_ref: "model/test".to_string(),
            instance_id: None,
        };
        assert!(
            validate_lifecycle_acceptance(
                "load_model",
                "owner-1",
                "present",
                "present",
                Some(&target),
                "model/test",
                None,
            )
            .is_ok()
        );
        assert!(
            validate_lifecycle_acceptance(
                "load_model",
                "",
                "present",
                "present",
                Some(&target),
                "model/test",
                None,
            )
            .is_err()
        );
        assert!(
            validate_lifecycle_acceptance(
                "load_model",
                "owner-1",
                "absent",
                "present",
                Some(&target),
                "model/test",
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn absent_model_target_requires_exactly_one_reference() {
        assert_eq!(
            validate_absent_model_target("model/test".to_string(), None)
                .expect("model-only target"),
            ("model/test".to_string(), None)
        );
        assert_eq!(
            validate_absent_model_target(String::new(), Some("runtime-2".to_string()))
                .expect("instance-only target"),
            (String::new(), Some("runtime-2".to_string()))
        );
        assert!(matches!(
            validate_absent_model_target("model/test".to_string(), Some("runtime-2".to_string())),
            Err(ControlPlaneClientError::Protocol(_))
        ));
        assert!(matches!(
            validate_absent_model_target(String::new(), None),
            Err(ControlPlaneClientError::Protocol(_))
        ));
    }

    #[test]
    fn legacy_unknown_lifecycle_command_maps_to_control_unsupported() {
        for code in [
            OwnerControlErrorCode::BadRequest,
            OwnerControlErrorCode::UnknownCommand,
        ] {
            let legacy = ControlPlaneClientError::Remote(OwnerControlRemoteError {
                code,
                message: "owner control request requires exactly one command variant".to_string(),
                request_id: Some(7),
                current_revision: None,
            });
            let mapped = map_legacy_lifecycle_unsupported("load_model", legacy);
            let ControlPlaneClientError::Remote(mapped) = mapped else {
                panic!("legacy response should remain a structured remote error");
            };
            assert_eq!(mapped.code, OwnerControlErrorCode::ControlUnsupported);
            assert_eq!(
                mapped.message,
                "remote owner-control endpoint does not support load_model"
            );
            assert_eq!(mapped.request_id, Some(7));
        }

        let unrelated = ControlPlaneClientError::Remote(OwnerControlRemoteError {
            code: OwnerControlErrorCode::BadRequest,
            message: "invalid model target".to_string(),
            request_id: Some(8),
            current_revision: None,
        });
        let ControlPlaneClientError::Remote(unrelated) =
            map_legacy_lifecycle_unsupported("load_model", unrelated)
        else {
            panic!("unrelated response should remain a structured remote error");
        };
        assert_eq!(unrelated.code, OwnerControlErrorCode::BadRequest);
        assert_eq!(unrelated.message, "invalid model target");
    }

    #[test]
    fn connect_error_fallback_is_narrow_to_alpn_negotiation() {
        assert!(is_alpn_mismatch_message("no application protocol selected"));
        assert!(is_alpn_mismatch_message("ALPN mismatch"));
        assert!(!is_alpn_mismatch_message("connection refused"));
        assert!(!is_alpn_mismatch_message("endpoint is closed"));
        assert!(!is_alpn_mismatch_message(
            "ALPN configuration is unavailable locally"
        ));
        assert_eq!(
            fallback_probe_disposition("connection refused"),
            ConnectProbeDisposition::SkipUnavailable
        );
        assert_eq!(
            fallback_probe_disposition("no application protocol selected"),
            ConnectProbeDisposition::ProbeLegacyMesh
        );
    }
}
