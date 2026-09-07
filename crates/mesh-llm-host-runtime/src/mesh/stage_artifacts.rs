use super::{Node, PeerInfo};
use crate::crypto::{OwnershipStatus, OwnershipSummary};
use crate::mesh::artifact_transfer_io::{
    PartialArtifactGuard, append_artifact_transfer_body, select_partial_artifact,
    write_artifact_transfer_chunk,
};
use crate::mesh::stage_proto::{
    stage_control_request_from_proto, stage_control_response_to_proto,
    stage_control_unavailable_response, stage_status_from_load,
};
use crate::mesh::stage_transport::{
    ARTIFACT_TRANSFER_BUFFER_BYTES, ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR,
    ARTIFACT_TRANSFER_OPEN_TIMEOUT, ARTIFACT_TRANSFER_READ_IDLE_TIMEOUT, StageTopologyInstance,
    artifact_transfer_allowed_by_topology, wait_local_stage_control_response,
    write_artifact_transfer_response,
};
use crate::protocol::{ValidateControlFrame, read_len_prefixed, write_len_prefixed};
use anyhow::{Context, Result};
use iroh::EndpointId;
use prost::Message;
use skippy_protocol::proto::stage as skippy_stage_proto;

impl Node {
    pub(crate) async fn handle_mesh_subprotocol_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        use prost::Message as _;

        let buf = read_len_prefixed(&mut recv).await?;
        let open = crate::proto::node::MeshSubprotocolOpen::decode(buf.as_slice())
            .map_err(|error| anyhow::anyhow!("MeshSubprotocolOpen decode error: {error}"))?;
        open.validate_frame()
            .map_err(|error| anyhow::anyhow!("MeshSubprotocolOpen validation error: {error}"))?;
        match (open.name.as_str(), open.major) {
            (skippy_protocol::STAGE_SUBPROTOCOL_NAME, skippy_protocol::STAGE_SUBPROTOCOL_MAJOR) => {
                self.handle_skippy_stage_subprotocol_stream(remote, send, recv)
                    .await
            }
            _ => anyhow::bail!(
                "unsupported mesh subprotocol {}/{} from {}",
                open.name,
                open.major,
                remote.fmt_short()
            ),
        }
    }

    pub(crate) async fn handle_skippy_stage_subprotocol_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut type_buf = [0u8; 1];
        recv.read_exact(&mut type_buf).await?;
        match type_buf[0] {
            skippy_protocol::STAGE_STREAM_CONTROL => {
                self.handle_stage_control(remote, send, recv).await
            }
            skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER => {
                self.handle_artifact_transfer_stream(remote, send, recv)
                    .await
            }
            skippy_protocol::STAGE_STREAM_TRANSPORT => {
                anyhow::bail!("skippy activation transport stays on skippy-stage/2")
            }
            other => anyhow::bail!("unknown skippy stage subprotocol stream kind {other:#04x}"),
        }
    }

    pub(crate) async fn decode_stage_control_request(
        &self,
        remote: EndpointId,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> anyhow::Result<skippy_stage_proto::StageControlRequest> {
        let buf = read_len_prefixed(recv).await.map_err(|e| {
            tracing::warn!(
                "handle_stage_control: read_len_prefixed failed from {}: {e}",
                remote.fmt_short()
            );
            e
        })?;
        let frame = skippy_protocol::proto::stage::StageControlRequest::decode(buf.as_slice())
            .map_err(|e| {
                tracing::warn!(
                    "handle_stage_control: decode failed from {}: {e}",
                    remote.fmt_short()
                );
                anyhow::anyhow!("StageControlRequest decode error: {e}")
            })?;
        skippy_protocol::validate_stage_control_request(&frame).map_err(|e| {
            tracing::warn!(
                "handle_stage_control: validation failed from {}: {e}",
                remote.fmt_short()
            );
            anyhow::anyhow!("StageControlRequest validation error: {e}")
        })?;
        anyhow::ensure!(
            frame.requester_id.as_slice() == remote.as_bytes(),
            "stage control requester_id does not match QUIC peer identity"
        );
        Ok(frame)
    }

    pub(crate) fn stage_control_request_kind(
        frame: &skippy_stage_proto::StageControlRequest,
    ) -> &'static str {
        match &frame.command {
            Some(skippy_stage_proto::stage_control_request::Command::ClaimCoordinator(_)) => {
                "claim"
            }
            Some(skippy_stage_proto::stage_control_request::Command::LoadStage(_)) => "load",
            Some(skippy_stage_proto::stage_control_request::Command::LoadLocalStage(_)) => {
                "load-local"
            }
            Some(skippy_stage_proto::stage_control_request::Command::StopStage(_)) => "stop",
            Some(skippy_stage_proto::stage_control_request::Command::PrepareStage(_)) => "prepare",
            _ => "other",
        }
    }

    pub(crate) async fn record_stage_control_response(
        &self,
        response: &crate::inference::skippy::StageControlResponse,
    ) {
        match response {
            crate::inference::skippy::StageControlResponse::Ready(ready) => {
                self.record_stage_status(Some(self.endpoint.id()), ready.status.clone())
                    .await;
            }
            crate::inference::skippy::StageControlResponse::Status(statuses) => {
                for status in statuses {
                    self.record_stage_status(Some(self.endpoint.id()), status.clone())
                        .await;
                }
            }
            _ => {}
        }
    }

    pub(crate) async fn execute_stage_control_request(
        &self,
        request: crate::inference::skippy::StageControlRequest,
    ) -> anyhow::Result<crate::inference::skippy::StageControlResponse> {
        // Load/Prepare can take minutes on large stages; use the same
        // per-request budget the remote sender uses instead of the short
        // local default, otherwise the executing node rejects its own load.
        let timeout = Self::stage_control_request_timeout(&request);
        let control_tx = self.stage_control_tx.lock().await.clone();
        match control_tx {
            Some(tx) => {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                tx.send(crate::inference::skippy::StageControlCommand {
                    request,
                    resp: resp_tx,
                })
                .map_err(|_| anyhow::anyhow!("stage control loop is unavailable"))?;
                wait_local_stage_control_response(resp_rx, timeout).await
            }
            None => Ok(stage_control_unavailable_response(request)),
        }
    }

    pub(crate) async fn execute_stage_control_request_for_peer(
        &self,
        remote: EndpointId,
        request: crate::inference::skippy::StageControlRequest,
    ) -> anyhow::Result<crate::inference::skippy::StageControlResponse> {
        match self.execute_stage_control_request(request.clone()).await {
            Ok(response) => Ok(response),
            Err(error) => Self::stage_control_load_failure_response(remote, request, error),
        }
    }

    pub(crate) fn stage_control_load_failure_response(
        remote: EndpointId,
        request: crate::inference::skippy::StageControlRequest,
        error: anyhow::Error,
    ) -> anyhow::Result<crate::inference::skippy::StageControlResponse> {
        let load = match request {
            crate::inference::skippy::StageControlRequest::Load(load)
            | crate::inference::skippy::StageControlRequest::LoadLocal(load) => load,
            _ => return Err(error),
        };
        let error_message = format!("{error:#}");
        tracing::warn!(
            peer = %remote.fmt_short(),
            stage_id = %load.stage_id,
            "stage load failed: {error_message}"
        );
        let mut status =
            stage_status_from_load(&load, crate::inference::skippy::StageRuntimeState::Failed);
        status.error = Some(error_message.clone());
        Ok(crate::inference::skippy::StageControlResponse::Ready(
            crate::inference::skippy::StageReadyResponse {
                accepted: false,
                status,
                error: Some(error_message),
            },
        ))
    }

    pub(crate) async fn handle_stage_control(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> anyhow::Result<()> {
        use prost::Message as _;

        let frame = self.decode_stage_control_request(remote, &mut recv).await?;
        self.ensure_current_stage_control_peer(remote).await?;
        let request_kind = Self::stage_control_request_kind(&frame);
        tracing::debug!(
            "handle_stage_control: received {request_kind} from {}",
            remote.fmt_short()
        );

        let mut request = stage_control_request_from_proto(frame)?;
        self.prepare_stage_control_request(&mut request)
            .await
            .map_err(|e| {
                tracing::warn!(
                    "handle_stage_control: prepare failed for {request_kind} from {}: {e}",
                    remote.fmt_short()
                );
                e
            })?;
        if let crate::inference::skippy::StageControlRequest::Load(load)
        | crate::inference::skippy::StageControlRequest::LoadLocal(load) = &request
        {
            self.record_stage_load_topology(load).await;
        }
        let response = self
            .execute_stage_control_request_for_peer(remote, request)
            .await?;
        self.record_stage_control_response(&response).await;
        let proto_response = stage_control_response_to_proto(response);
        write_len_prefixed(&mut send, &proto_response.encode_to_vec()).await?;
        let _ = send.finish();
        Ok(())
    }

    pub(crate) async fn prepare_stage_control_request(
        &self,
        request: &mut crate::inference::skippy::StageControlRequest,
    ) -> anyhow::Result<()> {
        if let crate::inference::skippy::StageControlRequest::LoadLocal(load) = request {
            // Enforce the command-level policy before any resolver is allowed
            // to inspect the request. StageControlState repeats this at the
            // execution boundary as defense in depth.
            load.local_source_required = true;
        }
        match request {
            crate::inference::skippy::StageControlRequest::Claim(_) => {}
            crate::inference::skippy::StageControlRequest::Load(load)
            | crate::inference::skippy::StageControlRequest::LoadLocal(load) => {
                let mut effective_load = load.clone();
                let verified = tokio::task::spawn_blocking(move || {
                    let verified =
                        crate::inference::skippy::apply_verified_local_source(&mut effective_load)?;
                    anyhow::Ok((effective_load, verified))
                })
                .await
                .context("join verify local-required stage load source task")??;
                *load = verified.0;
                if !verified.1
                    && load.load_mode == skippy_protocol::LoadMode::RuntimeSlice
                    && load
                        .model_path
                        .as_deref()
                        .is_none_or(|path| !std::path::Path::new(path).exists())
                {
                    for candidate in [
                        load.model_id.as_str(),
                        load.package_ref.strip_prefix("gguf://").unwrap_or_default(),
                    ]
                    .into_iter()
                    .filter(|candidate| !candidate.is_empty())
                    {
                        if let Ok(path) =
                            crate::models::resolve_model_spec(std::path::Path::new(candidate)).await
                            && path.exists()
                        {
                            load.model_path = Some(path.to_string_lossy().to_string());
                            break;
                        }
                    }
                }
                let topology_id = load.topology_id.clone();
                let run_id = load.run_id.clone();
                if let Some(upstream) = load.upstream.as_mut() {
                    self.prepare_stage_peer_endpoint(&topology_id, &run_id, upstream)
                        .await?;
                }
                if let Some(downstream) = load.downstream.as_mut() {
                    self.prepare_stage_peer_endpoint(&topology_id, &run_id, downstream)
                        .await?;
                }
            }
            crate::inference::skippy::StageControlRequest::Prepare(_) => {}
            crate::inference::skippy::StageControlRequest::Stop(stop) => {
                self.stop_stage_transport_bridge(&stop.topology_id, &stop.run_id, &stop.stage_id)
                    .await;
            }
            crate::inference::skippy::StageControlRequest::Status(_)
            | crate::inference::skippy::StageControlRequest::Inventory(_)
            | crate::inference::skippy::StageControlRequest::CancelPrepare(_)
            | crate::inference::skippy::StageControlRequest::StatusUpdate(_) => {}
        }
        Ok(())
    }

    pub(crate) async fn prepare_stage_peer_endpoint(
        &self,
        topology_id: &str,
        run_id: &str,
        peer: &mut crate::inference::skippy::StagePeerDescriptor,
    ) -> anyhow::Result<()> {
        let Some(peer_node) = peer.node_id else {
            return Ok(());
        };
        if peer_node == self.endpoint.id() {
            return Ok(());
        }
        let bridge_addr = self
            .ensure_stage_transport_bridge(peer_node, topology_id, run_id, peer.stage_id.clone())
            .await?;
        peer.endpoint = bridge_addr;
        Ok(())
    }

    pub(crate) async fn prefetch_stage_package_from_coordinator(
        &self,
        prepare: &crate::inference::skippy::StagePrepareRequest,
    ) -> Result<()> {
        let load = &prepare.load;
        if load.load_mode != skippy_protocol::LoadMode::LayerPackage {
            return Ok(());
        }
        if !crate::models::artifact_transfer::artifact_transfer_enabled() {
            return Ok(());
        }
        let Some(coordinator_id) = prepare.coordinator_id else {
            return Ok(());
        };
        if coordinator_id == self.endpoint.id() {
            return Ok(());
        }
        if !self
            .peer_supports_skippy_subprotocol_feature(
                coordinator_id,
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER,
            )
            .await
        {
            return Ok(());
        }
        self.fetch_stage_package_artifacts_from_peer(coordinator_id, load)
            .await
    }

    pub(crate) async fn peer_supports_skippy_subprotocol_feature(
        &self,
        peer_id: EndpointId,
        feature: &str,
    ) -> bool {
        let peer = {
            let state = self.state.lock().await;
            state.peers.get(&peer_id).cloned()
        };
        let Some(peer) = peer else {
            return false;
        };
        match feature {
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL => {
                peer.stage_protocol_generation_supported
            }
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER => {
                self.artifact_transfer_allowed_for_peer(&peer).await
            }
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST => {
                peer.stage_status_list_supported
            }
            _ => false,
        }
    }

    pub(crate) async fn ensure_current_stage_control_peer(
        &self,
        peer_id: EndpointId,
    ) -> Result<()> {
        anyhow::ensure!(
            self.peer_supports_skippy_subprotocol_feature(
                peer_id,
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL,
            )
            .await,
            "stage peer {} does not advertise the required generation-{} control bundle",
            peer_id.fmt_short(),
            skippy_protocol::STAGE_PROTOCOL_GENERATION
        );
        Ok(())
    }

    pub(crate) async fn fetch_stage_package_artifacts_from_peer(
        &self,
        peer_id: EndpointId,
        load: &crate::inference::skippy::StageLoadRequest,
    ) -> Result<()> {
        let package_dir =
            crate::models::artifact_transfer::package_cache_dir_for_ref(&load.package_ref)?;
        let manifest_request = crate::models::artifact_transfer::manifest_artifact_request(
            &load.package_ref,
            &load.manifest_sha256,
        )?;
        let manifest_path =
            crate::models::artifact_transfer::local_artifact_path(&package_dir, &manifest_request);
        if !crate::models::artifact_transfer::local_artifact_satisfies(
            &package_dir,
            &manifest_request,
            true,
        )? {
            self.fetch_artifact_from_peer(peer_id, load, &manifest_request, &manifest_path)
                .await
                .context("fetch package manifest from peer")?;
        }

        let artifacts = crate::models::artifact_transfer::required_stage_package_artifacts(
            &package_dir,
            &load.package_ref,
            &load.manifest_sha256,
            crate::models::artifact_transfer::StageArtifactSelection {
                layer_start: load.layer_start,
                layer_end: load.layer_end,
                include_embeddings: load.layer_start == 0,
                include_output: load.downstream.is_none(),
                include_projectors: load.layer_start == 0,
            },
        )?;
        for artifact in artifacts {
            if crate::models::artifact_transfer::local_artifact_satisfies(
                &package_dir,
                &artifact,
                true,
            )? {
                continue;
            }
            let destination =
                crate::models::artifact_transfer::local_artifact_path(&package_dir, &artifact);
            self.fetch_artifact_from_peer(peer_id, load, &artifact, &destination)
                .await
                .with_context(|| {
                    format!(
                        "fetch package artifact {} from peer",
                        artifact.relative_path.display()
                    )
                })?;
        }
        Ok(())
    }

    pub(crate) async fn fetch_artifact_from_peer(
        &self,
        peer_id: EndpointId,
        load: &crate::inference::skippy::StageLoadRequest,
        artifact: &crate::models::artifact_transfer::PackageArtifactRequest,
        destination: &std::path::Path,
    ) -> Result<()> {
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("create package artifact directory")?;
        }
        crate::models::artifact_transfer::ensure_local_artifact_install_parent(
            &artifact.package_ref,
            destination,
        )?;
        let resume_limit = Self::artifact_transfer_resume_limit(artifact)?;
        let partial = select_partial_artifact(destination, resume_limit)?;
        let temp_path = partial.path;
        let offset = partial.offset;
        let mut partial_guard = PartialArtifactGuard::preserve_on_error(temp_path.clone());

        let frame = skippy_stage_proto::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: self.endpoint.id().as_bytes().to_vec(),
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            stage_id: load.stage_id.clone(),
            package_ref: artifact.package_ref.clone(),
            manifest_sha256: artifact.manifest_sha256.clone(),
            relative_path: artifact.relative_path.to_string_lossy().to_string(),
            offset,
            expected_size: artifact.expected_size,
            expected_sha256: artifact.expected_sha256.clone(),
        };
        skippy_protocol::validate_stage_artifact_transfer_request(&frame)
            .map_err(|error| anyhow::anyhow!("invalid artifact transfer request: {error}"))?;

        let response = tokio::time::timeout(ARTIFACT_TRANSFER_OPEN_TIMEOUT, async {
            let (mut send, mut recv) = self
                .open_skippy_stage_mesh_stream(
                    peer_id,
                    skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER,
                )
                .await?;
            write_len_prefixed(&mut send, &frame.encode_to_vec()).await?;
            let _ = send.finish();
            let response_buf = read_len_prefixed(&mut recv).await?;
            let response =
                skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())
                    .map_err(|error| {
                        anyhow::anyhow!("StageArtifactTransferResponse decode error: {error}")
                    })?;
            skippy_protocol::validate_stage_artifact_transfer_response(&response).map_err(
                |error| anyhow::anyhow!("StageArtifactTransferResponse validation error: {error}"),
            )?;
            Ok::<_, anyhow::Error>((recv, response))
        })
        .await
        .map_err(|_| anyhow::anyhow!("timeout opening artifact transfer stream"))??;
        let (mut recv, response) = response;
        Self::remove_invalid_resume_partial(&mut partial_guard, offset, &response);
        if !response.accepted {
            anyhow::bail!(
                "peer artifact transfer rejected: {}",
                response
                    .error
                    .unwrap_or_else(|| "artifact unavailable".to_string())
            );
        }
        if let Some(expected_size) = artifact.expected_size {
            anyhow::ensure!(
                response.total_size == expected_size,
                "peer artifact size mismatch"
            );
        } else if artifact.relative_path.as_path()
            == std::path::Path::new(crate::models::artifact_transfer::PACKAGE_MANIFEST_FILE)
        {
            anyhow::ensure!(
                response.total_size <= crate::models::artifact_transfer::MAX_PACKAGE_MANIFEST_BYTES,
                "peer package manifest exceeds transfer limit"
            );
        } else {
            anyhow::bail!("peer artifact response missing expected size");
        }
        if let Some(expected_sha) = artifact.expected_sha256.as_deref() {
            anyhow::ensure!(
                response
                    .sha256
                    .as_deref()
                    .is_some_and(|sha| sha.eq_ignore_ascii_case(expected_sha)),
                "peer artifact sha256 mismatch"
            );
        }
        anyhow::ensure!(
            offset <= response.total_size,
            "peer artifact response is smaller than resume offset"
        );

        let transfer_result = async {
            append_artifact_transfer_body(
                &mut recv,
                &temp_path,
                offset,
                response.total_size,
                ARTIFACT_TRANSFER_BUFFER_BYTES,
                ARTIFACT_TRANSFER_READ_IDLE_TIMEOUT,
            )
            .await?;

            let actual_size = tokio::fs::metadata(&temp_path)
                .await
                .context("stat partial artifact")?
                .len();
            anyhow::ensure!(
                actual_size == response.total_size,
                "partial artifact size mismatch after transfer"
            );
            let temp_for_hash = temp_path.clone();
            let actual_sha = tokio::task::spawn_blocking(move || {
                crate::models::artifact_transfer::file_sha256_hex(&temp_for_hash)
            })
            .await
            .context("join artifact sha256 task")??;
            let expected_sha = artifact
                .expected_sha256
                .as_deref()
                .or(response.sha256.as_deref())
                .context("peer artifact response missing sha256")?;
            anyhow::ensure!(
                actual_sha.eq_ignore_ascii_case(expected_sha),
                "transferred artifact sha256 mismatch"
            );
            if destination.exists() {
                let _ = tokio::fs::remove_file(destination).await;
            }
            tokio::fs::rename(&temp_path, destination)
                .await
                .context("install transferred artifact")?;
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if let Err(error) = transfer_result {
            let error_message = error.to_string();
            if error_message.contains("transferred artifact sha256 mismatch")
                || error_message.contains("partial artifact size mismatch after transfer")
            {
                partial_guard.remove_now();
            }
            return Err(error);
        }
        partial_guard.disarm();
        Ok(())
    }

    pub(crate) fn remove_invalid_resume_partial(
        partial_guard: &mut PartialArtifactGuard,
        offset: u64,
        response: &skippy_stage_proto::StageArtifactTransferResponse,
    ) {
        if Self::artifact_transfer_response_invalidates_resume_offset(offset, response) {
            partial_guard.remove_now();
        }
    }

    pub(crate) fn artifact_transfer_response_invalidates_resume_offset(
        offset: u64,
        response: &skippy_stage_proto::StageArtifactTransferResponse,
    ) -> bool {
        if offset == 0 {
            return false;
        }
        if response.accepted {
            return offset > response.total_size;
        }
        response.error.as_deref() == Some(ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR)
    }

    pub(crate) fn artifact_transfer_resume_limit(
        artifact: &crate::models::artifact_transfer::PackageArtifactRequest,
    ) -> Result<u64> {
        if let Some(expected_size) = artifact.expected_size {
            return Ok(expected_size);
        }
        if artifact.relative_path.as_path()
            == std::path::Path::new(crate::models::artifact_transfer::PACKAGE_MANIFEST_FILE)
        {
            return Ok(crate::models::artifact_transfer::MAX_PACKAGE_MANIFEST_BYTES);
        }
        anyhow::bail!("artifact transfer resume requires an expected artifact size")
    }

    pub(crate) async fn artifact_transfer_rejected(
        send: &mut iroh::endpoint::SendStream,
        total_size: u64,
        sha256: Option<&str>,
        error: &'static str,
    ) -> anyhow::Result<()> {
        write_artifact_transfer_response(send, false, total_size, sha256, Some(error)).await
    }

    pub(crate) async fn authorize_artifact_transfer_request(
        &self,
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        request: &skippy_stage_proto::StageArtifactTransferRequest,
    ) -> anyhow::Result<Option<std::path::PathBuf>> {
        if !self
            .artifact_transfer_serving_allowed_for_remote(remote)
            .await
        {
            Self::artifact_transfer_rejected(send, 0, None, "artifact transfer disabled").await?;
            return Ok(None);
        }
        let Some(package_dir) = Self::artifact_transfer_package_dir(remote, send, request).await?
        else {
            return Ok(None);
        };
        let topologies = self
            .stage_topologies
            .lock()
            .await
            .topologies
            .values()
            .cloned()
            .collect::<Vec<_>>();
        if !Self::artifact_transfer_topology_allows(
            remote,
            send,
            request,
            &package_dir,
            &topologies,
        )
        .await?
        {
            return Ok(None);
        }
        Ok(Some(package_dir))
    }

    pub(crate) async fn artifact_transfer_package_dir(
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        request: &skippy_stage_proto::StageArtifactTransferRequest,
    ) -> anyhow::Result<Option<std::path::PathBuf>> {
        match crate::models::artifact_transfer::package_cache_dir_for_ref(&request.package_ref) {
            Ok(path) => Ok(Some(path)),
            Err(error) => {
                tracing::debug!(
                    peer = %remote.fmt_short(),
                    "artifact transfer request has unsupported package ref: {error}"
                );
                Self::artifact_transfer_rejected(send, 0, None, "artifact unavailable").await?;
                Ok(None)
            }
        }
    }

    pub(crate) async fn artifact_transfer_topology_allows(
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        request: &skippy_stage_proto::StageArtifactTransferRequest,
        package_dir: &std::path::Path,
        topologies: &[StageTopologyInstance],
    ) -> anyhow::Result<bool> {
        let allowed =
            match artifact_transfer_allowed_by_topology(topologies, remote, package_dir, request) {
                Ok(allowed) => allowed,
                Err(error) => {
                    tracing::debug!(
                        peer = %remote.fmt_short(),
                        path = %request.relative_path,
                        "artifact transfer authorization failed: {error}"
                    );
                    Self::artifact_transfer_rejected(send, 0, None, "artifact unavailable").await?;
                    return Ok(false);
                }
            };
        if !allowed {
            tracing::debug!(
                peer = %remote.fmt_short(),
                path = %request.relative_path,
                "artifact transfer request is not authorized for this stage assignment"
            );
            Self::artifact_transfer_rejected(send, 0, None, "artifact unavailable").await?;
        }
        Ok(allowed)
    }

    pub(crate) async fn resolve_artifact_transfer_request(
        &self,
        remote: EndpointId,
        send: &mut iroh::endpoint::SendStream,
        request: &skippy_stage_proto::StageArtifactTransferRequest,
    ) -> anyhow::Result<Option<crate::models::artifact_transfer::ServableArtifact>> {
        let request_for_resolution = request.clone();
        let artifact = match tokio::task::spawn_blocking(move || {
            crate::models::artifact_transfer::servable_artifact_from_request(
                &request_for_resolution,
            )
        })
        .await
        .context("join artifact transfer resolution task")?
        {
            Ok(artifact) => artifact,
            Err(error) => {
                tracing::debug!(
                    peer = %remote.fmt_short(),
                    path = %request.relative_path,
                    "artifact transfer request cannot be served: {error}"
                );
                Self::artifact_transfer_rejected(send, 0, None, "artifact unavailable").await?;
                return Ok(None);
            }
        };
        if request.offset > artifact.size {
            Self::artifact_transfer_rejected(
                send,
                artifact.size,
                Some(&artifact.sha256),
                ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR,
            )
            .await?;
            return Ok(None);
        }
        Ok(Some(artifact))
    }

    pub(crate) async fn handle_artifact_transfer_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> anyhow::Result<()> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let buf = read_len_prefixed(&mut recv).await?;
        let request = skippy_stage_proto::StageArtifactTransferRequest::decode(buf.as_slice())
            .map_err(|error| {
                anyhow::anyhow!("StageArtifactTransferRequest decode error: {error}")
            })?;
        skippy_protocol::validate_stage_artifact_transfer_request(&request).map_err(|error| {
            anyhow::anyhow!("StageArtifactTransferRequest validation error: {error}")
        })?;
        if request.requester_id.as_slice() != remote.as_bytes() {
            anyhow::bail!("artifact transfer requester_id does not match QUIC peer identity");
        }
        let Some(_package_dir) = self
            .authorize_artifact_transfer_request(remote, &mut send, &request)
            .await?
        else {
            return Ok(());
        };
        let Some(artifact) = self
            .resolve_artifact_transfer_request(remote, &mut send, &request)
            .await?
        else {
            return Ok(());
        };

        write_artifact_transfer_response(
            &mut send,
            true,
            artifact.size,
            Some(&artifact.sha256),
            None,
        )
        .await?;
        let mut file = tokio::fs::File::open(&artifact.path)
            .await
            .context("open artifact for transfer")?;
        file.seek(std::io::SeekFrom::Start(request.offset))
            .await
            .context("seek artifact for transfer")?;
        let mut buffer = vec![0u8; ARTIFACT_TRANSFER_BUFFER_BYTES];
        let mut remaining = artifact.size.saturating_sub(request.offset);
        while remaining > 0 {
            let limit = buffer.len().min(remaining as usize);
            let read = file
                .read(&mut buffer[..limit])
                .await
                .context("read artifact for transfer")?;
            anyhow::ensure!(read > 0, "artifact file ended before expected byte count");
            write_artifact_transfer_chunk(
                &mut send,
                &buffer[..read],
                ARTIFACT_TRANSFER_READ_IDLE_TIMEOUT,
            )
            .await?;
            remaining -= read as u64;
        }
        let _ = send.finish();
        Ok(())
    }

    pub(crate) async fn local_verified_owner_id(&self) -> Option<String> {
        let summary = self.owner_summary.lock().await.clone();
        if summary.status == OwnershipStatus::Verified {
            summary.owner_id
        } else {
            None
        }
    }

    pub(crate) async fn artifact_transfer_allowed_for_peer(&self, peer: &PeerInfo) -> bool {
        peer.artifact_transfer_supported
            && self
                .artifact_transfer_policy_allows_peer_owner(&peer.owner_summary)
                .await
    }

    pub(crate) async fn artifact_transfer_serving_allowed_for_remote(
        &self,
        remote: EndpointId,
    ) -> bool {
        let peer_owner = {
            let state = self.state.lock().await;
            state
                .peers
                .get(&remote)
                .map(|peer| peer.owner_summary.clone())
        };
        let Some(peer_owner) = peer_owner else {
            return false;
        };
        self.artifact_transfer_policy_allows_peer_owner(&peer_owner)
            .await
    }

    pub(crate) async fn artifact_transfer_policy_allows_peer_owner(
        &self,
        peer_owner: &OwnershipSummary,
    ) -> bool {
        let local_owner = self.owner_summary.lock().await.clone();
        let trust_store = self.trust_store.lock().await.clone();
        crate::models::artifact_transfer::artifact_transfer_allowed_between(
            &local_owner,
            peer_owner,
            &trust_store,
        )
    }
}
