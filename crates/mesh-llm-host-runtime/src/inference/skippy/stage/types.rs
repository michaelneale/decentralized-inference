use anyhow::Result;
use skippy_protocol::{FlashAttentionType, LoadMode, SplitMode, StageDevice};
use tokio::sync::oneshot;

#[derive(Debug)]
pub(crate) struct StageControlCommand {
    pub(crate) request: StageControlRequest,
    pub(crate) resp: oneshot::Sender<Result<StageControlResponse>>,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum StageControlRequest {
    Claim(StageCoordinatorClaim),
    Load(StageLoadRequest),
    /// Fail-closed content-addressed load. Older protobuf decoders do not
    /// recognize its oneof arm and reject it as a missing command.
    LoadLocal(StageLoadRequest),
    Stop(StageStopRequest),
    Status(StageStatusFilter),
    Inventory(StageInventoryRequest),
    Prepare(StagePrepareRequest),
    CancelPrepare(StageCancelPrepareRequest),
    StatusUpdate(StagePreparationStatus),
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum StageControlResponse {
    ClaimAccepted(StageCoordinatorClaimAck),
    Ready(StageReadyResponse),
    Status(Vec<StageStatusSnapshot>),
    Inventory(StageLayerInventory),
    PrepareAccepted(StagePrepareAcceptedResponse),
    PreparationStatus(StagePreparationStatus),
    StatusAck(StageStatusAck),
}

pub(crate) type StageCoordinatorClaim = skippy_coordinator::CoordinatorClaim;

#[derive(Clone, Debug)]
pub(crate) struct StageCoordinatorClaimAck {
    pub(crate) accepted: bool,
    pub(crate) claim: StageCoordinatorClaim,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageLoadRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    /// Present for current peers even when the exact profile is the empty
    /// default. `None` identifies an older/profile-unaware sender.
    pub(crate) runtime_profile: Option<String>,
    pub(crate) backend: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) admission: skippy_protocol::StageAdmissionDescriptor,
    pub(crate) participant_set_hash: String,
    pub(crate) topology_hash: String,
    pub(crate) activation_codec: skippy_protocol::StageActivationCodec,
    pub(crate) activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
    /// Canonical generation-wide stage list. Every participant receives the
    /// same identities, ownership, and ranges; readiness updates replace any
    /// provisional `:0` endpoint with the observed bound address.
    pub(crate) topology_stages: Vec<StageTopologyStageDescriptor>,
    pub(crate) model_path: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) local_source_required: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) projector_use_gpu: Option<bool>,
    pub(crate) media_marker: Option<String>,
    pub(crate) image_min_tokens: Option<u32>,
    pub(crate) image_max_tokens: Option<u32>,
    pub(crate) batch_max_tokens: Option<u32>,
    pub(crate) glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub(crate) generation_signal_window: Option<u32>,
    pub(crate) selected_device: Option<StageDevice>,
    pub(crate) bind_addr: String,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) continuous_batching: bool,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) mmap: Option<bool>,
    pub(crate) mlock: bool,
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) runtime_settings: StageLoadRuntimeSettings,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) shutdown_generation: u64,
    pub(crate) coordinator_term: u64,
    pub(crate) coordinator_id: Option<iroh::EndpointId>,
    pub(crate) lease_until_unix_ms: u64,
    pub(crate) load_mode: LoadMode,
    pub(crate) upstream: Option<StagePeerDescriptor>,
    pub(crate) downstream: Option<StagePeerDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StageTopologyStageDescriptor {
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) node_id: iroh::EndpointId,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) bind_addr: String,
}

#[cfg(test)]
pub(crate) fn test_stage_admission(
    layer_start: u32,
    layer_end: u32,
) -> skippy_protocol::StageAdmissionDescriptor {
    skippy_protocol::StageAdmissionDescriptor {
        version: skippy_protocol::STAGE_ADMISSION_DESCRIPTOR_VERSION,
        package_id: format!("sha256:{}", "a5".repeat(32)),
        plan_id: format!("skippy-plan:v1:{}", "b6".repeat(32)),
        layer_start,
        layer_end,
        resident_tensor_ids: vec!["tensor-0".to_string()],
        sidecars: Vec::new(),
        profiles: vec![skippy_protocol::StageAdmissionProfile {
            profile_id: "default".to_string(),
            graph_identity: "graph".to_string(),
            profile_identity: "profile".to_string(),
            slice_identity: "slice".to_string(),
            source_snapshot_identity: "snapshot".to_string(),
            graph_configuration_id: "graph-config".to_string(),
            backend_id: "backend".to_string(),
        }],
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StageLoadRuntimeSettings {
    pub(crate) repack: bool,
    pub(crate) op_offload: Option<bool>,
    pub(crate) no_host_buffer: bool,
    pub(crate) check_tensors: bool,
    pub(crate) direct_io: bool,
    pub(crate) main_gpu: Option<u32>,
    pub(crate) split_mode: SplitMode,
    pub(crate) kv_offload: Option<bool>,
    pub(crate) kv_unified: Option<bool>,
    pub(crate) swa_full: Option<bool>,
    pub(crate) cache_idle_slots: Option<u32>,
    pub(crate) activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStopRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) stage_id: String,
    pub(crate) shutdown_generation: u64,
    pub(crate) coordinator_term: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StageStatusFilter {
    pub(crate) topology_id: Option<String>,
    pub(crate) run_id: Option<String>,
    pub(crate) stage_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageInventoryRequest {
    pub(crate) model_id: String,
    /// Present for current peers even when the exact profile is the empty
    /// default. `None` identifies an older/profile-unaware sender.
    pub(crate) runtime_profile: Option<String>,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) expected_source_model_sha256: Option<String>,
    pub(crate) local_source_required: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePrepareRequest {
    pub(crate) load: StageLoadRequest,
    pub(crate) coordinator_id: Option<iroh::EndpointId>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageCancelPrepareRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) stage_id: String,
    pub(crate) shutdown_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerRange {
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct StageLayerInventory {
    pub(crate) model_id: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) layer_count: u32,
    pub(crate) ready_ranges: Vec<LayerRange>,
    pub(crate) available_ranges: Vec<LayerRange>,
    pub(crate) missing_ranges: Vec<LayerRange>,
    pub(crate) preparing_ranges: Vec<StagePreparationStatus>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) content_addressed_local_source: Option<bool>,
    pub(crate) source_model_kind: SourceModelKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceModelKind {
    Unknown,
    LayerPackage,
    PlainGguf,
    SplitGguf,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePeerDescriptor {
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) endpoint: String,
    pub(crate) node_id: Option<iroh::EndpointId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageRuntimeState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StagePreparationState {
    Assigned,
    Downloading,
    Available,
    Resolving,
    Loading,
    Ready,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub(crate) struct StageReadyResponse {
    pub(crate) accepted: bool,
    pub(crate) status: StageStatusSnapshot,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStatusSnapshot {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) package_ref: Option<String>,
    pub(crate) manifest_sha256: Option<String>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) materialized_path: Option<String>,
    pub(crate) materialized_pinned: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) admission: Option<skippy_protocol::StageAdmissionDescriptor>,
    pub(crate) activation_codec: skippy_protocol::StageActivationCodec,
    pub(crate) activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
    pub(crate) state: StageRuntimeState,
    pub(crate) bind_addr: String,
    pub(crate) input_activation_boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
    pub(crate) output_activation_boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
    pub(crate) selected_device: Option<StageDevice>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) error: Option<String>,
    pub(crate) shutdown_generation: u64,
    pub(crate) coordinator_term: u64,
    pub(crate) coordinator_id: Option<iroh::EndpointId>,
    pub(crate) lease_until_unix_ms: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePreparationStatus {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) admission: Option<skippy_protocol::StageAdmissionDescriptor>,
    pub(crate) activation_codec: skippy_protocol::StageActivationCodec,
    pub(crate) activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
    pub(crate) state: StagePreparationState,
    pub(crate) bytes_done: Option<u64>,
    pub(crate) bytes_total: Option<u64>,
    pub(crate) bind_addr: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) shutdown_generation: u64,
    pub(crate) coordinator_term: u64,
    pub(crate) coordinator_id: Option<iroh::EndpointId>,
    pub(crate) lease_until_unix_ms: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePrepareAcceptedResponse {
    pub(crate) accepted: bool,
    pub(crate) status: StagePreparationStatus,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStatusAck {
    pub(crate) accepted: bool,
    pub(crate) error: Option<String>,
}
