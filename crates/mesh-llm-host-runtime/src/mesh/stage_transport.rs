use super::{Node, PeerInfo, StageTransportBridgeLabel};
use crate::mesh::node::LocalRequestMetricsSampler;
use crate::mesh::stage_proto::{
    stage_control_request_to_proto, stage_control_response_from_proto,
    stage_runtime_status_from_snapshot, stage_runtime_status_key,
    stage_snapshot_from_runtime_status, stage_topology_from_load, stage_topology_key,
};
use crate::protocol::{ControlProtocol, read_len_prefixed, write_len_prefixed};
use anyhow::{Context, Result};
use iroh::endpoint::Connection;
use iroh::{EndpointId, TransportAddr};
use prost::Message;
use serde_json::json;
use skippy_protocol::proto::stage as skippy_stage_proto;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectedPathObservation {
    pub(crate) path_type: &'static str,
    pub(crate) rtt_ms: Option<u32>,
    pub(crate) observed_direct_remote_addr: Option<SocketAddr>,
}

pub(crate) struct ConnectionCaptureEvent<'a> {
    pub(crate) event: &'a str,
    pub(crate) remote: EndpointId,
    pub(crate) direction: &'a str,
    pub(crate) phase: &'a str,
    pub(crate) protocol: Option<ControlProtocol>,
    pub(crate) path_type: Option<&'a str>,
    pub(crate) rtt_ms: Option<u32>,
    pub(crate) admitted_peer: Option<bool>,
    pub(crate) reason: Option<&'a str>,
}

pub(crate) struct PeerLifecycleCaptureEvent<'a> {
    pub(crate) event: &'a str,
    pub(crate) peer: EndpointId,
    pub(crate) reason: &'a str,
    pub(crate) reporter: Option<EndpointId>,
    pub(crate) last_seen_age_ms: Option<u64>,
    pub(crate) last_mentioned_age_ms: Option<u64>,
    pub(crate) had_connection: Option<bool>,
    pub(crate) bridge_id: Option<EndpointId>,
}

pub(crate) struct HttpCaptureEvent<'a> {
    pub(crate) event: &'a str,
    pub(crate) source_addr: Option<SocketAddr>,
    pub(crate) method: &'a str,
    pub(crate) path: &'a str,
    pub(crate) body_len_bytes: usize,
    pub(crate) model_name: Option<&'a str>,
    pub(crate) completion_tokens: Option<u32>,
    pub(crate) stream: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SplitStagePathKind {
    Direct,
    Relay,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitStagePathSnapshot {
    pub(crate) kind: SplitStagePathKind,
    pub(crate) rtt_ms: Option<u32>,
}

impl SplitStagePathSnapshot {
    pub(crate) const fn direct(rtt_ms: Option<u32>) -> Self {
        Self {
            kind: SplitStagePathKind::Direct,
            rtt_ms,
        }
    }

    pub(crate) const fn relay(rtt_ms: Option<u32>) -> Self {
        Self {
            kind: SplitStagePathKind::Relay,
            rtt_ms,
        }
    }

    pub(crate) const fn unknown() -> Self {
        Self {
            kind: SplitStagePathKind::Unknown,
            rtt_ms: None,
        }
    }

    pub(crate) const fn with_direct_rtt_fallback(self, fallback_rtt_ms: Option<u32>) -> Self {
        match (self.kind, self.rtt_ms, fallback_rtt_ms) {
            (SplitStagePathKind::Direct, None, Some(rtt_ms)) => Self::direct(Some(rtt_ms)),
            _ => self,
        }
    }

    pub(crate) fn with_peer_path_fallback(self, fallback: Option<SelectedPathObservation>) -> Self {
        match (self.kind, fallback) {
            (SplitStagePathKind::Direct, Some(observation)) => {
                self.with_direct_rtt_fallback(observation.rtt_ms)
            }
            (SplitStagePathKind::Unknown, Some(observation)) => {
                split_stage_path_snapshot_from_observation(observation)
            }
            _ => self,
        }
    }
}

pub(crate) fn selected_path_observation(conn: &Connection) -> Option<SelectedPathObservation> {
    let path_list = conn.paths();
    for path_info in &path_list {
        if !path_info.is_selected() {
            continue;
        }

        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
        let rtt = path_info.rtt();
        let rtt_ms = if rtt.is_zero() {
            None
        } else {
            Some(rtt.as_millis().min(u128::from(u32::MAX)) as u32)
        };
        let observed_direct_remote_addr = match path_info.remote_addr() {
            TransportAddr::Ip(addr) => Some(*addr),
            _ => None,
        };

        return Some(SelectedPathObservation {
            path_type,
            rtt_ms,
            observed_direct_remote_addr,
        });
    }

    None
}

pub(crate) fn split_stage_path_snapshot_from_observation(
    observation: SelectedPathObservation,
) -> SplitStagePathSnapshot {
    match observation.path_type {
        "direct" => SplitStagePathSnapshot::direct(observation.rtt_ms),
        "relay" => SplitStagePathSnapshot::relay(observation.rtt_ms),
        _ => SplitStagePathSnapshot::unknown(),
    }
}

pub(crate) fn split_stage_path_snapshot_from_connection(
    conn: &Connection,
) -> SplitStagePathSnapshot {
    let Some(observation) = selected_path_observation(conn) else {
        return SplitStagePathSnapshot::unknown();
    };
    split_stage_path_snapshot_from_observation(observation)
}

pub(crate) fn endpoint_id_capture_fields(id: EndpointId) -> serde_json::Value {
    json!({
        "short": id.fmt_short().to_string(),
        "hex": hex::encode(id.as_bytes()),
    })
}

pub(crate) fn peer_capture_fields(
    peer: &PeerInfo,
    source: &str,
    bridge_id: Option<EndpointId>,
) -> serde_json::Value {
    let direct_rtt_ms = peer
        .display_rtt
        .as_ref()
        .map(|observation| observation.rtt_ms);
    let propagated_latency = peer.propagated_latency.as_ref().map(|observation| {
        json!({
            "latency_ms": observation.latency_ms,
            "age_ms_at_received": observation.age_ms_at_received,
            "observer": observation.observer_id.map(endpoint_id_capture_fields),
        })
    });

    json!({
        "peer": endpoint_id_capture_fields(peer.id),
        "source": source,
        "bridge": bridge_id.map(endpoint_id_capture_fields),
        "role": &peer.role,
        "version": &peer.version,
        "hostname": &peer.hostname,
        "models": &peer.models,
        "serving_models": &peer.serving_models,
        "hosted_models": &peer.hosted_models,
        "hosted_models_known": peer.hosted_models_known,
        "available_models": &peer.available_models,
        "requested_models": &peer.requested_models,
        "explicit_model_interests": &peer.explicit_model_interests,
        "model_source": &peer.model_source,
        "gpu_name": &peer.gpu_name,
        "is_soc": peer.is_soc,
        "vram_bytes": peer.vram_bytes,
        "gpu_vram": &peer.gpu_vram,
        "gpu_reserved_bytes": &peer.gpu_reserved_bytes,
        "gpu_mem_bandwidth_gbps": &peer.gpu_mem_bandwidth_gbps,
        "gpu_compute_tflops_fp32": &peer.gpu_compute_tflops_fp32,
        "gpu_compute_tflops_fp16": &peer.gpu_compute_tflops_fp16,
        "direct_rtt_ms": direct_rtt_ms.or(peer.rtt_ms),
        "propagated_latency": propagated_latency,
        "owner": &peer.owner_summary,
        "artifact_transfer_supported": peer.artifact_transfer_supported,
        "stage_status_list_supported": peer.stage_status_list_supported,
        "first_joined_mesh_ts": peer.first_joined_mesh_ts,
    })
}

pub(super) const PEER_CONNECT_AND_GOSSIP_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
pub(crate) const ARTIFACT_TRANSFER_OPEN_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
pub(crate) const ARTIFACT_TRANSFER_READ_IDLE_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
pub(crate) const STAGE_STREAM_TYPE_READ_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(10);
pub(crate) const LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
pub(crate) const ARTIFACT_TRANSFER_BUFFER_BYTES: usize = 1024 * 1024;
pub(crate) const ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR: &str = "invalid transfer offset";

pub(crate) type MeshBiStream = (iroh::endpoint::SendStream, iroh::endpoint::RecvStream);

pub(crate) enum StageBiAccept {
    Streams(MeshBiStream),
    Continue,
    Closed,
}

pub(crate) enum StageStreamAccept {
    Dispatch(MeshBiStream, u8),
    Continue,
    Closed,
}

pub(crate) async fn write_artifact_transfer_response(
    send: &mut iroh::endpoint::SendStream,
    accepted: bool,
    total_size: u64,
    sha256: Option<&str>,
    error: Option<&str>,
) -> Result<()> {
    let response = skippy_stage_proto::StageArtifactTransferResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        accepted,
        total_size,
        sha256: sha256.map(str::to_string),
        error: error.map(str::to_string),
    };
    skippy_protocol::validate_stage_artifact_transfer_response(&response)
        .map_err(|error| anyhow::anyhow!("invalid artifact transfer response: {error}"))?;
    write_len_prefixed(send, &response.encode_to_vec()).await?;
    if !accepted {
        let _ = send.finish();
    }
    Ok(())
}

pub(crate) async fn read_stage_stream_type<R>(reader: &mut R) -> Result<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut type_buf = [0u8; 1];
    tokio::time::timeout(
        STAGE_STREAM_TYPE_READ_TIMEOUT,
        tokio::io::AsyncReadExt::read_exact(reader, &mut type_buf),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "timeout reading skippy stage stream type after {STAGE_STREAM_TYPE_READ_TIMEOUT:?}"
        )
    })?
    .context("read skippy stage stream type")?;
    Ok(type_buf[0])
}

pub(crate) async fn wait_local_stage_control_response(
    response: tokio::sync::oneshot::Receiver<
        anyhow::Result<crate::inference::skippy::StageControlResponse>,
    >,
    timeout: std::time::Duration,
) -> Result<crate::inference::skippy::StageControlResponse> {
    tokio::time::timeout(timeout, response)
        .await
        .map_err(|_| {
            anyhow::anyhow!("timeout waiting for local stage control response after {timeout:?}")
        })?
        .map_err(|_| anyhow::anyhow!("stage control response dropped"))?
}

pub(crate) fn artifact_transfer_allowed_by_topology(
    topologies: &[StageTopologyInstance],
    remote: EndpointId,
    package_dir: &std::path::Path,
    request: &skippy_stage_proto::StageArtifactTransferRequest,
) -> Result<bool> {
    let relative_path =
        crate::models::artifact_transfer::safe_relative_artifact_path(&request.relative_path)?;
    let manifest_path =
        std::path::PathBuf::from(crate::models::artifact_transfer::PACKAGE_MANIFEST_FILE);
    for topology in topologies {
        if topology.topology_id != request.topology_id
            || topology.run_id != request.run_id
            || topology.package_ref != request.package_ref
            || !topology
                .manifest_sha256
                .eq_ignore_ascii_case(&request.manifest_sha256)
        {
            continue;
        }
        let final_stage_index = topology.stages.iter().map(|stage| stage.stage_index).max();
        for assignment in topology
            .stages
            .iter()
            .filter(|stage| stage.node_id == remote && stage.stage_id == request.stage_id)
        {
            if relative_path == manifest_path {
                return Ok(true);
            }
            let include_output = final_stage_index == Some(assignment.stage_index);
            let allowed =
                if crate::models::artifact_transfer::package_manifest_schema_version(package_dir)?
                    == u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION)
                {
                    let admission = topology
                        .admissions
                        .get(&assignment.stage_id)
                        .context("package-v2 topology is missing stage admission")?;
                    crate::models::artifact_transfer::required_admitted_stage_package_artifacts(
                        package_dir,
                        &topology.package_ref,
                        &topology.manifest_sha256,
                        admission,
                    )?
                } else {
                    crate::models::artifact_transfer::required_stage_package_artifacts(
                        package_dir,
                        &topology.package_ref,
                        &topology.manifest_sha256,
                        crate::models::artifact_transfer::StageArtifactSelection {
                            layer_start: assignment.layer_start,
                            layer_end: assignment.layer_end,
                            include_embeddings: assignment.layer_start == 0,
                            include_output,
                            include_projectors: assignment.layer_start == 0,
                        },
                    )?
                };
            if allowed.iter().any(|artifact| {
                artifact.relative_path == relative_path
                    && request
                        .expected_size
                        .is_none_or(|expected_size| Some(expected_size) == artifact.expected_size)
                    && request
                        .expected_sha256
                        .as_deref()
                        .is_none_or(|expected_sha| {
                            artifact
                                .expected_sha256
                                .as_deref()
                                .is_some_and(|sha| sha.eq_ignore_ascii_case(expected_sha))
                        })
            }) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Channels returned by Node::start for inbound tunnel streams.
pub struct TunnelChannels {
    pub rpc: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    pub http: tokio::sync::mpsc::Receiver<(
        EndpointId,
        iroh::endpoint::SendStream,
        iroh::endpoint::RecvStream,
    )>,
    pub stage: tokio::sync::mpsc::Receiver<(
        EndpointId,
        iroh::endpoint::SendStream,
        iroh::endpoint::RecvStream,
    )>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageTopologyInstance {
    pub topology_id: String,
    pub run_id: String,
    pub model_id: String,
    pub package_ref: String,
    pub manifest_sha256: String,
    pub admissions: std::collections::BTreeMap<String, skippy_protocol::StageAdmissionDescriptor>,
    pub stages: Vec<StageAssignment>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageAssignment {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: EndpointId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub endpoint: StageEndpoint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageEndpoint {
    pub bind_addr: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageRuntimeStatus {
    pub topology_id: String,
    pub run_id: String,
    pub model_id: String,
    pub backend: String,
    pub package_ref: Option<String>,
    pub manifest_sha256: Option<String>,
    pub source_model_path: Option<String>,
    pub source_model_sha256: Option<String>,
    pub source_model_bytes: Option<u64>,
    pub materialized_path: Option<String>,
    pub materialized_pinned: bool,
    pub projector_path: Option<String>,
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: Option<EndpointId>,
    pub layer_start: u32,
    pub layer_end: u32,
    pub admission: Option<skippy_protocol::StageAdmissionDescriptor>,
    pub activation_codec: skippy_protocol::StageActivationCodec,
    pub activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
    pub state: crate::inference::skippy::StageRuntimeState,
    pub bind_addr: String,
    pub input_activation_boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
    pub output_activation_boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
    pub selected_device: Option<skippy_protocol::StageDevice>,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub flash_attn_type: skippy_protocol::FlashAttentionType,
    pub error: Option<String>,
    pub shutdown_generation: u64,
}

/// Classifies a stage status-refresh failure so that only a definitive
/// "peer answered but has no such stage" outcome marks the stage Failed. A
/// transient failure (unreachable peer, timeout) retains the last-known status
/// instead of tearing down a healthy split on a momentary blip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageStatusRefreshFailure {
    MissingFromRuntime,
    Transient,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StageTopologyState {
    pub(crate) topologies: HashMap<String, StageTopologyInstance>,
    pub(crate) statuses: HashMap<String, StageRuntimeStatus>,
}

impl StageTopologyState {
    pub(crate) fn record_topology(&mut self, topology: StageTopologyInstance) {
        self.topologies.insert(
            stage_topology_key(&topology.topology_id, &topology.run_id),
            topology,
        );
    }

    pub(crate) fn activate_topology(&mut self, topology: StageTopologyInstance) {
        let active_key = stage_topology_key(&topology.topology_id, &topology.run_id);
        let model_id = topology.model_id.clone();
        self.topologies
            .retain(|key, existing| existing.model_id != model_id || key == &active_key);
        self.statuses.retain(|_, status| {
            status.model_id != model_id
                || (status.topology_id == topology.topology_id && status.run_id == topology.run_id)
        });
        self.record_topology(topology);
    }

    pub(crate) fn withdraw_topology(&mut self, topology_id: &str, run_id: &str) -> bool {
        let topology_key = stage_topology_key(topology_id, run_id);
        let removed_topology = self.topologies.remove(&topology_key).is_some();
        let old_status_count = self.statuses.len();
        self.statuses
            .retain(|_, status| status.topology_id != topology_id || status.run_id != run_id);
        removed_topology || self.statuses.len() != old_status_count
    }

    pub(crate) fn visible_topologies(&self) -> Vec<StageTopologyInstance> {
        self.topologies
            .values()
            .filter(|topology| {
                topology.stages.len() > 1
                    || !self.statuses.values().any(|status| {
                        status.topology_id == topology.topology_id
                            && status.run_id == topology.run_id
                    })
            })
            .cloned()
            .collect()
    }

    pub(crate) fn runtime_statuses(&self) -> Vec<StageRuntimeStatus> {
        self.statuses
            .values()
            .filter(|status| {
                !status.topology_id.is_empty()
                    && !status.run_id.is_empty()
                    && !status.stage_id.is_empty()
            })
            .cloned()
            .collect()
    }

    pub(crate) fn record_status(&mut self, runtime_status: StageRuntimeStatus) {
        if runtime_status.topology_id.is_empty()
            || runtime_status.run_id.is_empty()
            || runtime_status.stage_id.is_empty()
        {
            return;
        }
        if !runtime_status.bind_addr.is_empty() && !runtime_status.bind_addr.ends_with(":0") {
            let topology_key =
                stage_topology_key(&runtime_status.topology_id, &runtime_status.run_id);
            if let Some(topology) = self.topologies.get_mut(&topology_key)
                && let Some(stage) = topology
                    .stages
                    .iter_mut()
                    .find(|stage| stage.stage_id == runtime_status.stage_id)
            {
                stage.endpoint.bind_addr = runtime_status.bind_addr.clone();
            }
        }
        self.statuses.insert(
            stage_runtime_status_key(
                &runtime_status.topology_id,
                &runtime_status.run_id,
                &runtime_status.stage_id,
            ),
            runtime_status,
        );
    }

    pub(crate) fn record_status_refresh_failure(
        &mut self,
        status: &StageRuntimeStatus,
        failure: StageStatusRefreshFailure,
    ) {
        // A transient refresh failure (peer briefly unreachable, request
        // timeout) must NOT mark the stage Failed. A newly published stage can
        // also be absent from the responder until its embedded runtime records
        // the first status, so retain Starting and retry. Once a stage has been
        // observed Ready, a definitive "missing from runtime" response still
        // marks it Failed.
        if failure == StageStatusRefreshFailure::Transient
            || status.state == crate::inference::skippy::StageRuntimeState::Starting
        {
            return;
        }
        self.record_status(stage_runtime_status_from_snapshot(
            status.node_id,
            stage_snapshot_from_runtime_status(
                status,
                crate::inference::skippy::StageRuntimeState::Failed,
                Some("stage status missing from runtime".to_string()),
            ),
        ));
    }

    pub(crate) fn active_statuses(&self) -> Vec<StageRuntimeStatus> {
        self.statuses
            .values()
            .filter(|status| {
                matches!(
                    status.state,
                    crate::inference::skippy::StageRuntimeState::Starting
                        | crate::inference::skippy::StageRuntimeState::Ready
                )
            })
            .cloned()
            .collect()
    }
}

pub struct InflightRequestGuard {
    pub(crate) inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) inflight_change_tx: watch::Sender<u64>,
    pub(crate) local_request_metrics: Arc<LocalRequestMetricsSampler>,
    pub(crate) started_at: std::time::Instant,
    pub(crate) routing_metrics: crate::network::metrics::RoutingMetrics,
    pub(crate) routing_telemetry: Option<Arc<dyn crate::network::metrics::RoutingTelemetrySink>>,
    pub(crate) runtime_data_producer: crate::runtime_data::RuntimeDataProducer,
}

impl Drop for InflightRequestGuard {
    fn drop(&mut self) {
        let _ = self.inflight_requests.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
        let _ = self.inflight_change_tx.send(
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64,
        );
        self.local_request_metrics
            .record_request_completed(self.started_at);
        let current_inflight_requests =
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64;
        if let Some(routing_telemetry) = &self.routing_telemetry {
            routing_telemetry.observe_inflight_requests(current_inflight_requests);
        }
        self.runtime_data_producer.publish_routing_snapshot(
            self.routing_metrics
                .collector_snapshot(current_inflight_requests),
        );
    }
}

#[async_trait::async_trait]
impl crate::inference::skippy::StagePackagePrefetcher for Node {
    async fn prefetch_stage_package(
        &self,
        request: &crate::inference::skippy::StagePrepareRequest,
    ) -> Result<()> {
        self.prefetch_stage_package_from_coordinator(request).await
    }
}

impl Node {
    pub(crate) fn set_routing_telemetry_sink(
        &self,
        sink: Option<Arc<dyn crate::network::metrics::RoutingTelemetrySink>>,
    ) {
        *self
            .routing_telemetry
            .lock()
            .expect("routing telemetry sink lock poisoned") = sink;
    }

    pub(crate) fn routing_telemetry_sink(
        &self,
    ) -> Option<Arc<dyn crate::network::metrics::RoutingTelemetrySink>> {
        self.routing_telemetry
            .lock()
            .expect("routing telemetry sink lock poisoned")
            .clone()
    }

    pub(crate) fn publish_routing_runtime_snapshot(&self) {
        self.runtime_data_producer.publish_routing_snapshot(
            self.routing_metrics
                .collector_snapshot(self.inflight_requests()),
        );
    }

    pub fn begin_inflight_request(&self) -> InflightRequestGuard {
        self.local_request_metrics.record_request_accepted();
        self.inflight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let current = self
            .inflight_requests
            .load(std::sync::atomic::Ordering::Relaxed) as u64;
        let _ = self.inflight_change_tx.send(current);
        self.routing_metrics.observe_inflight(current);
        let routing_telemetry = self.routing_telemetry_sink();
        if let Some(sink) = &routing_telemetry {
            sink.observe_inflight_requests(current);
        }
        self.publish_routing_runtime_snapshot();
        InflightRequestGuard {
            inflight_requests: self.inflight_requests.clone(),
            inflight_change_tx: self.inflight_change_tx.clone(),
            local_request_metrics: self.local_request_metrics.clone(),
            started_at: std::time::Instant::now(),
            routing_metrics: self.routing_metrics.clone(),
            routing_telemetry,
            runtime_data_producer: self.runtime_data_producer.clone(),
        }
    }

    pub fn inflight_requests(&self) -> u64 {
        self.inflight_requests
            .load(std::sync::atomic::Ordering::Relaxed) as u64
    }

    /// Locally observed routing metrics, used by the auto-router to score
    /// models by their measured throughput from this node's perspective.
    pub fn routing_metrics(&self) -> &crate::network::metrics::RoutingMetrics {
        &self.routing_metrics
    }

    pub fn inflight_change_rx(&self) -> watch::Receiver<u64> {
        self.inflight_change_tx.subscribe()
    }

    pub(crate) async fn set_stage_control_handle(
        &self,
        lifecycle: crate::inference::skippy::StageControlHandle,
    ) {
        *self.stage_control_tx.lock().await = Some(lifecycle.sender());
        *self.stage_control_lifecycle.lock().await = Some(lifecycle);
    }

    #[cfg(test)]
    pub(crate) async fn set_stage_control_sender(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<crate::inference::skippy::StageControlCommand>,
    ) {
        *self.stage_control_tx.lock().await = Some(tx);
    }

    pub async fn record_stage_topology(&self, topology: StageTopologyInstance) {
        self.stage_topologies.lock().await.record_topology(topology);
    }

    pub(crate) async fn record_stage_load_topology(
        &self,
        load: &crate::inference::skippy::StageLoadRequest,
    ) {
        let topology = stage_topology_from_load(self.endpoint.id(), load);
        let mut state = self.stage_topologies.lock().await;
        state.record_topology(topology.clone());
        for stage in topology.stages {
            let mut snapshot = crate::mesh::stage_status_from_load(
                load,
                crate::inference::skippy::StageRuntimeState::Starting,
            );
            snapshot.stage_id = stage.stage_id;
            snapshot.stage_index = stage.stage_index;
            snapshot.layer_start = stage.layer_start;
            snapshot.layer_end = stage.layer_end;
            snapshot.bind_addr = stage.endpoint.bind_addr;
            snapshot.admission =
                (snapshot.stage_id == load.stage_id).then(|| load.admission.clone());
            snapshot.error = None;
            state.record_status(stage_runtime_status_from_snapshot(
                Some(stage.node_id),
                snapshot,
            ));
        }
    }

    pub async fn activate_stage_topology(&self, topology: StageTopologyInstance) {
        self.stage_topologies
            .lock()
            .await
            .activate_topology(topology);
    }

    pub async fn withdraw_stage_topology(&self, topology_id: &str, run_id: &str) -> bool {
        self.stage_topologies
            .lock()
            .await
            .withdraw_topology(topology_id, run_id)
    }

    pub async fn stage_topologies(&self) -> Vec<StageTopologyInstance> {
        self.stage_topologies.lock().await.visible_topologies()
    }

    pub async fn stage_runtime_statuses(&self) -> Vec<StageRuntimeStatus> {
        self.stage_topologies.lock().await.runtime_statuses()
    }

    pub(crate) async fn locally_executing_stage_statuses(
        &self,
        filter: &crate::inference::skippy::StageStatusFilter,
    ) -> Vec<crate::inference::skippy::StageStatusSnapshot> {
        let local_node = self.endpoint.id();
        self.stage_topologies
            .lock()
            .await
            .runtime_statuses()
            .into_iter()
            .filter(|status| {
                status.node_id == Some(local_node)
                    && filter
                        .topology_id
                        .as_ref()
                        .is_none_or(|value| value == &status.topology_id)
                    && filter
                        .run_id
                        .as_ref()
                        .is_none_or(|value| value == &status.run_id)
                    && filter
                        .stage_id
                        .as_ref()
                        .is_none_or(|value| value == &status.stage_id)
            })
            .map(|status| {
                stage_snapshot_from_runtime_status(&status, status.state, status.error.clone())
            })
            .collect()
    }

    pub async fn refresh_stage_runtime_statuses(&self, timeout: std::time::Duration) {
        let active_statuses = self.stage_topologies.lock().await.active_statuses();
        for status in active_statuses {
            self.refresh_stage_runtime_status(status, timeout).await;
        }
    }

    async fn refresh_stage_runtime_status(
        &self,
        status: StageRuntimeStatus,
        timeout: std::time::Duration,
    ) {
        let Some(peer_id) = status.node_id else {
            return;
        };
        if peer_id == self.endpoint.id() {
            return;
        }
        let filter = crate::inference::skippy::StageStatusFilter {
            topology_id: Some(status.topology_id.clone()),
            run_id: Some(status.run_id.clone()),
            stage_id: Some(status.stage_id.clone()),
        };
        let refresh = async {
            if peer_id == self.endpoint.id() {
                self.query_local_stage_status(filter)
                    .await
                    .map(crate::inference::skippy::StageControlResponse::Status)
            } else {
                self.send_stage_control(
                    peer_id,
                    crate::inference::skippy::StageControlRequest::Status(filter),
                )
                .await
            }
        };
        match tokio::time::timeout(timeout, refresh).await {
            Ok(Ok(response)) => {
                self.record_stage_status_refresh(&status, peer_id, response)
                    .await;
            }
            Ok(Err(error)) => {
                self.record_transient_stage_status_refresh_failure(&status, peer_id, Some(&error))
                    .await;
            }
            Err(_) => {
                self.record_transient_stage_status_refresh_failure(&status, peer_id, None)
                    .await;
            }
        }
    }

    async fn record_stage_status_refresh(
        &self,
        status: &StageRuntimeStatus,
        peer_id: EndpointId,
        response: crate::inference::skippy::StageControlResponse,
    ) {
        match response {
            crate::inference::skippy::StageControlResponse::Status(statuses) => {
                if statuses.is_empty() {
                    self.stage_topologies
                        .lock()
                        .await
                        .record_status_refresh_failure(
                            status,
                            StageStatusRefreshFailure::MissingFromRuntime,
                        );
                    return;
                }
                for status in statuses {
                    self.record_stage_status(Some(peer_id), status).await;
                }
            }
            crate::inference::skippy::StageControlResponse::Ready(ready) => {
                self.record_stage_status(Some(peer_id), ready.status).await;
            }
            _ => {}
        }
    }

    async fn record_transient_stage_status_refresh_failure(
        &self,
        status: &StageRuntimeStatus,
        peer_id: EndpointId,
        error: Option<&anyhow::Error>,
    ) {
        self.stage_topologies
            .lock()
            .await
            .record_status_refresh_failure(status, StageStatusRefreshFailure::Transient);
        match error {
            Some(error) => tracing::debug!(
                topology_id = %status.topology_id,
                run_id = %status.run_id,
                stage_id = %status.stage_id,
                peer = %peer_id.fmt_short(),
                error = %error,
                "stage status refresh failed; retaining last stage status"
            ),
            None => tracing::debug!(
                topology_id = %status.topology_id,
                run_id = %status.run_id,
                stage_id = %status.stage_id,
                peer = %peer_id.fmt_short(),
                "stage status refresh timed out; retaining last stage status"
            ),
        }
    }

    pub(crate) async fn record_stage_status(
        &self,
        node_id: Option<EndpointId>,
        status: crate::inference::skippy::StageStatusSnapshot,
    ) {
        let runtime_status = stage_runtime_status_from_snapshot(node_id, status);
        self.stage_topologies
            .lock()
            .await
            .record_status(runtime_status);
    }

    pub(crate) async fn query_local_stage_status(
        &self,
        filter: crate::inference::skippy::StageStatusFilter,
    ) -> Result<Vec<crate::inference::skippy::StageStatusSnapshot>> {
        let control_tx = self.stage_control_tx.lock().await.clone();
        let Some(tx) = control_tx else {
            anyhow::bail!("stage control is not available");
        };
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        tx.send(crate::inference::skippy::StageControlCommand {
            request: crate::inference::skippy::StageControlRequest::Status(filter),
            resp: resp_tx,
        })
        .map_err(|_| anyhow::anyhow!("stage control loop is unavailable"))?;
        match wait_local_stage_control_response(resp_rx, LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT)
            .await?
        {
            crate::inference::skippy::StageControlResponse::Status(statuses) => Ok(statuses),
            crate::inference::skippy::StageControlResponse::Ready(_) => {
                anyhow::bail!("unexpected ready response for stage status request")
            }
            _ => anyhow::bail!("unexpected response for stage status request"),
        }
    }

    pub(crate) async fn send_local_stage_control(
        &self,
        mut request: crate::inference::skippy::StageControlRequest,
    ) -> Result<crate::inference::skippy::StageControlResponse> {
        self.prepare_stage_control_request(&mut request).await?;
        if let crate::inference::skippy::StageControlRequest::Load(load)
        | crate::inference::skippy::StageControlRequest::LoadLocal(load) = &request
        {
            self.record_stage_load_topology(load).await;
        }
        // Load/Prepare can take minutes on large stages; use the same
        // per-request budget remote control uses instead of the short default.
        let timeout = Self::stage_control_request_timeout(&request);
        let control_tx = self.stage_control_tx.lock().await.clone();
        let Some(tx) = control_tx else {
            anyhow::bail!("stage control is not available");
        };
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        tx.send(crate::inference::skippy::StageControlCommand {
            request,
            resp: resp_tx,
        })
        .map_err(|_| anyhow::anyhow!("stage control loop is unavailable"))?;
        let response = wait_local_stage_control_response(resp_rx, timeout).await?;
        match &response {
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
        Ok(response)
    }

    pub async fn send_stage_control(
        &self,
        peer_id: EndpointId,
        request: crate::inference::skippy::StageControlRequest,
    ) -> Result<crate::inference::skippy::StageControlResponse> {
        use prost::Message as _;

        let timeout = Self::stage_control_request_timeout(&request);
        if let crate::inference::skippy::StageControlRequest::Load(load)
        | crate::inference::skippy::StageControlRequest::LoadLocal(load) = &request
        {
            self.record_stage_load_topology(load).await;
        }
        let frame = stage_control_request_to_proto(self.endpoint.id(), request)?;
        let response = tokio::time::timeout(timeout, async {
            self.ensure_current_stage_control_peer(peer_id).await?;
            let (mut send, mut recv) = self
                .open_skippy_stage_mesh_stream(peer_id, skippy_protocol::STAGE_STREAM_CONTROL)
                .await?;
            write_len_prefixed(&mut send, &frame.encode_to_vec()).await?;
            let buf = read_len_prefixed(&mut recv).await?;
            let response =
                skippy_protocol::proto::stage::StageControlResponse::decode(buf.as_slice())
                    .map_err(|e| anyhow::anyhow!("StageControlResponse decode error: {e}"))?;
            skippy_protocol::validate_stage_control_response(&response)
                .map_err(|e| anyhow::anyhow!("StageControlResponse validation error: {e}"))?;
            let _ = send.finish();
            stage_control_response_from_proto(response)
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!("timeout waiting for stage control response after {timeout:?}")
        })??;

        match &response {
            crate::inference::skippy::StageControlResponse::Ready(ready) => {
                self.record_stage_status(Some(peer_id), ready.status.clone())
                    .await;
            }
            crate::inference::skippy::StageControlResponse::Status(statuses) => {
                for status in statuses {
                    self.record_stage_status(Some(peer_id), status.clone())
                        .await;
                }
            }
            _ => {}
        }
        Ok(response)
    }

    pub(crate) fn stage_control_request_timeout(
        request: &crate::inference::skippy::StageControlRequest,
    ) -> std::time::Duration {
        match request {
            crate::inference::skippy::StageControlRequest::Claim(_)
            | crate::inference::skippy::StageControlRequest::Stop(_)
            | crate::inference::skippy::StageControlRequest::Status(_)
            | crate::inference::skippy::StageControlRequest::Inventory(_)
            | crate::inference::skippy::StageControlRequest::CancelPrepare(_)
            | crate::inference::skippy::StageControlRequest::StatusUpdate(_) => {
                std::time::Duration::from_secs(30)
            }
            crate::inference::skippy::StageControlRequest::Load(load) => {
                crate::inference::skippy::stage_load_timeout(load)
            }
            crate::inference::skippy::StageControlRequest::LoadLocal(load) => {
                crate::inference::skippy::stage_load_timeout(load)
            }
            crate::inference::skippy::StageControlRequest::Prepare(prepare) => {
                crate::inference::skippy::stage_load_timeout(&prepare.load)
            }
        }
    }

    pub async fn open_stage_transport_stream(
        &self,
        peer_id: EndpointId,
        topology_id: impl Into<String>,
        run_id: impl Into<String>,
        stage_id: impl Into<String>,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        use prost::Message as _;

        let open = skippy_protocol::proto::stage::StageTransportOpen {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: self.endpoint.id().as_bytes().to_vec(),
            topology_id: topology_id.into(),
            run_id: run_id.into(),
            stage_id: stage_id.into(),
        };
        skippy_protocol::validate_stage_transport_open(&open)
            .map_err(|e| anyhow::anyhow!("StageTransportOpen validation error: {e}"))?;
        let conn = self.stage_connection_to_peer(peer_id).await?;
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[skippy_protocol::STAGE_STREAM_TRANSPORT])
            .await?;
        write_len_prefixed(&mut send, &open.encode_to_vec()).await?;
        Ok((send, recv))
    }

    pub async fn ensure_stage_transport_bridge(
        &self,
        peer_id: EndpointId,
        topology_id: impl Into<String>,
        run_id: impl Into<String>,
        stage_id: impl Into<String>,
    ) -> Result<String> {
        let topology_id = topology_id.into();
        let run_id = run_id.into();
        let stage_id = stage_id.into();
        let key = stage_runtime_status_key(&topology_id, &run_id, &stage_id);
        let label = StageTransportBridgeLabel::new(&topology_id, &run_id, &stage_id);
        let owner = self
            .reserve_stage_transport_bridge(key.clone(), &label)
            .await?;

        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) => {
                self.remove_stage_transport_bridge_if_owner(&key, &owner)
                    .await;
                return Err(error.into());
            }
        };
        let bind_addr = match listener.local_addr() {
            Ok(addr) => addr.to_string(),
            Err(error) => {
                self.remove_stage_transport_bridge_if_owner(&key, &owner)
                    .await;
                return Err(error.into());
            }
        };
        let node = self.clone();
        let cleanup_node = self.clone();
        let cleanup_key = key.clone();
        let cleanup_owner = owner.clone();
        let topology_for_task = topology_id.clone();
        let run_for_task = run_id.clone();
        let stage_for_task = stage_id.clone();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((tcp_stream, _)) = listener.accept().await else {
                    break;
                };
                let node = node.clone();
                let topology_id = topology_for_task.clone();
                let run_id = run_for_task.clone();
                let stage_id = stage_for_task.clone();
                tokio::spawn(async move {
                    if let Err(err) = async {
                        tcp_stream.set_nodelay(true)?;
                        let (send, recv) = node
                            .open_stage_transport_stream(peer_id, topology_id, run_id, stage_id)
                            .await?;
                        let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
                        crate::network::tunnel::relay_bidirectional(tcp_read, tcp_write, send, recv)
                            .await
                    }
                    .await
                    {
                        tracing::warn!(
                            "stage transport bridge to {} ended: {err}",
                            peer_id.fmt_short()
                        );
                    }
                });
            }
            cleanup_node
                .remove_stage_transport_bridge_if_owner(&cleanup_key, &cleanup_owner)
                .await;
        });
        if self
            .publish_stage_transport_bridge(key, owner, handle)
            .await
        {
            return Ok(bind_addr);
        }
        Err(label.cancelled_error())
    }

    pub(crate) async fn register_stage_transport_alias(
        &self,
        topology_id: &str,
        run_id: &str,
        stage_id: &str,
        bind_addr: impl Into<String>,
    ) {
        let key = stage_runtime_status_key(topology_id, run_id, stage_id);
        self.stage_transport_aliases
            .lock()
            .await
            .insert(key, bind_addr.into());
    }

    pub(crate) async fn stage_transport_alias(
        &self,
        topology_id: &str,
        run_id: &str,
        stage_id: &str,
    ) -> Option<String> {
        let key = stage_runtime_status_key(topology_id, run_id, stage_id);
        self.stage_transport_aliases.lock().await.get(&key).cloned()
    }

    pub(crate) async fn unregister_stage_transport_alias(
        &self,
        topology_id: &str,
        run_id: &str,
        stage_id: &str,
    ) {
        let key = stage_runtime_status_key(topology_id, run_id, stage_id);
        self.stage_transport_aliases.lock().await.remove(&key);
    }
}
impl Node {
    pub(crate) async fn stage_stream_admitted(&self, remote: EndpointId) -> bool {
        let state = self.state.lock().await;
        state.peers.get(&remote).is_some_and(PeerInfo::is_admitted)
    }

    pub(crate) async fn dispatch_stage_stream_kind(
        &self,
        remote: EndpointId,
        stream_type: u8,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        match dedicated_stage_stream_kind(stream_type) {
            Some(DedicatedStageStreamKind::ActivationTransport) => {
                if self
                    .stage_transport_tx
                    .send((remote, send, recv))
                    .await
                    .is_err()
                {
                    tracing::warn!("Stage transport channel closed, dropping stream");
                }
            }
            None => {
                tracing::warn!(
                    "unsupported dedicated skippy stage stream type {stream_type:#04x} from {}; control and artifact streams require mesh subprotocol transport",
                    remote.fmt_short()
                );
            }
        }
    }

    pub(crate) async fn dispatch_stage_streams(&self, conn: Connection, remote: EndpointId) {
        loop {
            match self.accept_stage_stream(&conn, remote).await {
                StageStreamAccept::Dispatch((send, recv), stream_type) => {
                    self.dispatch_stage_stream_kind(remote, stream_type, send, recv)
                        .await;
                }
                StageStreamAccept::Continue => continue,
                StageStreamAccept::Closed => break,
            }
        }
    }

    pub(crate) async fn accept_admitted_stage_bi(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> StageBiAccept {
        let (send, recv) = match conn.accept_bi().await {
            Ok(streams) => streams,
            Err(e) => {
                tracing::info!(
                    "Skippy stage connection to {} closed: {e}",
                    remote.fmt_short()
                );
                return StageBiAccept::Closed;
            }
        };
        if !self.stage_stream_admitted(remote).await {
            tracing::warn!(
                "Quarantine: skippy stage stream from unadmitted peer {} rejected",
                remote.fmt_short()
            );
            drop((send, recv));
            return StageBiAccept::Continue;
        }
        StageBiAccept::Streams((send, recv))
    }

    pub(crate) async fn accept_stage_stream(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> StageStreamAccept {
        let (send, mut recv) = match self.accept_admitted_stage_bi(conn, remote).await {
            StageBiAccept::Streams(streams) => streams,
            StageBiAccept::Continue => return StageStreamAccept::Continue,
            StageBiAccept::Closed => return StageStreamAccept::Closed,
        };
        let stream_type = match read_stage_stream_type(&mut recv).await {
            Ok(stream_type) => stream_type,
            Err(error) => {
                tracing::debug!(
                    "skippy stage stream from {} did not provide stream type: {error}",
                    remote.fmt_short()
                );
                return StageStreamAccept::Continue;
            }
        };
        StageStreamAccept::Dispatch((send, recv), stream_type)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DedicatedStageStreamKind {
    ActivationTransport,
}

fn dedicated_stage_stream_kind(stream_type: u8) -> Option<DedicatedStageStreamKind> {
    (stream_type == skippy_protocol::STAGE_STREAM_TRANSPORT)
        .then_some(DedicatedStageStreamKind::ActivationTransport)
}

#[cfg(test)]
mod dedicated_stream_tests {
    use super::{DedicatedStageStreamKind, dedicated_stage_stream_kind};

    #[test]
    fn dedicated_stage_alpn_accepts_activation_transport_only() {
        assert_eq!(
            dedicated_stage_stream_kind(skippy_protocol::STAGE_STREAM_TRANSPORT),
            Some(DedicatedStageStreamKind::ActivationTransport)
        );
        assert_eq!(
            dedicated_stage_stream_kind(skippy_protocol::STAGE_STREAM_CONTROL),
            None
        );
        assert_eq!(
            dedicated_stage_stream_kind(skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER),
            None
        );
    }
}
