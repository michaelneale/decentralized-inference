//! Stage configuration, topology, and activation contracts.

use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageIdentity {
    pub run_id: String,
    pub request_id: String,
    pub session_id: String,
    pub topology_id: String,
    pub stage_id: String,
    pub stage_index: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoadMode {
    #[default]
    RuntimeSlice,
    LayerPackage,
    ArtifactSlice,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionType {
    #[default]
    Auto,
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    #[default]
    Auto,
    None,
    Layer,
    Row,
    Tensor,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmDsaPolicy {
    #[default]
    Auto,
    V1,
}

/// Versioned encoding used for floating-point activation planes between stages.
///
/// Tokens, positions, masks, routing data, and control fields stay in their
/// exact wire representations. This policy applies only to activation values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum StageActivationCodec {
    /// Exact little-endian IEEE-754 binary32 payloads. Kept as the correctness
    /// oracle and the default until a lossy codec is explicitly qualified.
    #[default]
    #[serde(rename = "raw-f32-v1")]
    RawF32V1,
    /// IEEE-754 binary16 with round-to-nearest, ties-to-even conversion.
    #[serde(rename = "f16-rne-v1")]
    F16RneV1,
    /// bfloat16 with round-to-nearest, ties-to-even conversion.
    #[serde(rename = "bf16-rne-v1")]
    Bf16RneV1,
    /// Per-logical-row symmetric signed int8 with an inline finite F32 scale.
    #[serde(rename = "s8-row-f32-rne-v1")]
    S8RowF32RneV1,
}

impl StageActivationCodec {
    pub const fn identity(self) -> &'static str {
        match self {
            Self::RawF32V1 => "raw-f32-v1",
            Self::F16RneV1 => "f16-rne-v1",
            Self::Bf16RneV1 => "bf16-rne-v1",
            Self::S8RowF32RneV1 => "s8-row-f32-rne-v1",
        }
    }

    pub(crate) const fn binary_wire_id(self) -> i32 {
        match self {
            Self::RawF32V1 => 1,
            Self::F16RneV1 => 2,
            Self::Bf16RneV1 => 3,
            Self::S8RowF32RneV1 => 4,
        }
    }

    pub(crate) const fn from_binary_wire_id(value: i32) -> Option<Self> {
        match value {
            1 => Some(Self::RawF32V1),
            2 => Some(Self::F16RneV1),
            3 => Some(Self::Bf16RneV1),
            4 => Some(Self::S8RowF32RneV1),
            _ => None,
        }
    }
}

/// Admitted activation-codec rules for a topology.
///
/// The per-frame encoding remains [`StageActivationCodec`] (it rides the binary
/// frame header) and `activation_codec` stays the fixed codec for
/// [`StageActivationCodecPolicy::Fixed`] or the mandatory RawF32 fallback
/// codec for [`StageActivationCodecPolicy::AutoLosslessV1`]. The policy only
/// decides which encodings a receiver must admit. Lossless selection under
/// `AutoLosslessV1` decodes bit-identically to raw, so one policy namespace is
/// enough for topology and cache identity; a future lossy policy must
/// additionally namespace per realized frame codec and accumulated error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageActivationCodecPolicy {
    /// Legacy behavior: every frame uses exactly the configured codec.
    #[default]
    #[serde(rename = "fixed")]
    Fixed,
    /// Policy v1: the producer selects RawF32, byte-exact BF16, or byte-exact
    /// F16 per realized frame (BF16 wins ties); every receiver must admit all
    /// three. Requires the configured codec to be RawF32V1 (the mandatory
    /// fallback). Fail closed on any other codec.
    #[serde(rename = "auto-lossless-v1")]
    AutoLosslessV1,
}

impl StageActivationCodecPolicy {
    /// Fail-closed admission of a realized frame codec under this policy.
    /// `AutoLosslessV1` also fails when the configured fallback codec is not
    /// RawF32V1, so no caller can admit an Auto frame under an invalid
    /// fallback.
    pub fn permits(
        self,
        configured_codec: StageActivationCodec,
        frame_codec: StageActivationCodec,
    ) -> bool {
        match self {
            Self::Fixed => configured_codec == frame_codec,
            // S8 stays excluded until its lossy quality is qualified.
            // F16/BF16 are admitted only when the producer's
            // selector proves exact round-trip for the complete frame.
            Self::AutoLosslessV1 => {
                configured_codec == StageActivationCodec::RawF32V1
                    && matches!(
                        frame_codec,
                        StageActivationCodec::RawF32V1
                            | StageActivationCodec::Bf16RneV1
                            | StageActivationCodec::F16RneV1
                    )
            }
        }
    }

    /// Whether this policy is admissible with the configured codec.
    /// `AutoLosslessV1` requires RawF32V1 as its fallback codec.
    pub const fn compatible(self, configured_codec: StageActivationCodec) -> bool {
        match self {
            Self::Fixed => true,
            Self::AutoLosslessV1 => {
                matches!(configured_codec, StageActivationCodec::RawF32V1)
            }
        }
    }

    /// Identity string for topology and cache binding. `Fixed` returns the
    /// configured codec's identity byte-for-byte so existing digests are
    /// unchanged.
    pub const fn identity(self, configured_codec: StageActivationCodec) -> &'static str {
        match self {
            Self::Fixed => configured_codec.identity(),
            Self::AutoLosslessV1 => "auto-lossless-v1",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageConfig {
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    #[serde(default)]
    pub package_ref: Option<String>,
    #[serde(default)]
    pub manifest_sha256: Option<String>,
    #[serde(default)]
    pub source_model_path: Option<String>,
    #[serde(default)]
    pub source_model_sha256: Option<String>,
    #[serde(default)]
    pub source_model_bytes: Option<u64>,
    #[serde(default)]
    pub materialized_path: Option<String>,
    #[serde(default)]
    pub materialized_pinned: bool,
    #[serde(default)]
    pub model_path: Option<String>,
    /// Optional load-time quantization recipe for a SafeTensors checkpoint.
    #[serde(default)]
    pub checkpoint_quantization: Option<String>,
    /// Optional importance matrix for low-bit checkpoint quantization.
    #[serde(default)]
    pub checkpoint_imatrix: Option<String>,
    /// SHA-256 of `checkpoint_imatrix`, used in derived model/cache identity.
    #[serde(default)]
    pub checkpoint_imatrix_sha256: Option<String>,
    #[serde(default)]
    pub projector_path: Option<String>,
    #[serde(default)]
    pub projector_use_gpu: Option<bool>,
    #[serde(default)]
    pub media_marker: Option<String>,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    #[serde(default)]
    pub batch_max_tokens: Option<u32>,
    #[serde(default)]
    pub glm_dsa_policy: GlmDsaPolicy,
    #[serde(default)]
    pub generation_signal_window: Option<u32>,
    /// Floating-point activation encoding for every downstream edge produced
    /// by this stage. Generation 8 peers echo and bind this policy before the
    /// binary data plane starts.
    #[serde(default)]
    pub activation_codec: StageActivationCodec,
    /// Admitted activation-codec rules for this topology. Defaults to
    /// `Fixed(activation_codec)`, which reproduces the legacy single-codec
    /// behavior and identity exactly.
    #[serde(default)]
    pub activation_codec_policy: StageActivationCodecPolicy,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_lane_count")]
    pub lane_count: u32,
    #[serde(default)]
    pub n_batch: Option<u32>,
    #[serde(default)]
    pub n_ubatch: Option<u32>,
    #[serde(default)]
    pub n_gpu_layers: i32,
    #[serde(default)]
    pub mmap: Option<bool>,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default)]
    pub repack: bool,
    #[serde(default)]
    pub op_offload: Option<bool>,
    #[serde(default)]
    pub no_host_buffer: bool,
    #[serde(default)]
    pub check_tensors: bool,
    #[serde(default)]
    pub direct_io: bool,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub split_mode: SplitMode,
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    #[serde(default)]
    pub flash_attn_type: FlashAttentionType,
    #[serde(default)]
    pub kv_offload: Option<bool>,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    #[serde(default)]
    pub swa_full: Option<bool>,
    #[serde(default)]
    pub cache_idle_slots: Option<u32>,
    #[serde(default)]
    pub filter_tensors_on_load: bool,
    /// Exact native tensor names resolved locally from admitted package-v2
    /// tensor IDs. Empty preserves the legacy range-based loader filter.
    #[serde(default)]
    pub resident_tensor_names: Vec<String>,
    #[serde(default)]
    pub selected_device: Option<StageDevice>,
    #[serde(default)]
    pub kv_cache: Option<StageKvCacheConfig>,
    #[serde(default = "default_native_mtp_enabled")]
    pub native_mtp_enabled: bool,
    pub load_mode: LoadMode,
    pub bind_addr: String,
    #[serde(default)]
    pub upstream: Option<PeerConfig>,
    #[serde(default)]
    pub downstream: Option<PeerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageDevice {
    pub backend_device: String,
    #[serde(default)]
    pub stable_id: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageKvCacheMode {
    Disabled,
    Auto,
    Record,
    LookupRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageKvCachePayload {
    Auto,
    ResidentKv,
    KvRecurrent,
    FullState,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageKvCacheConfig {
    #[serde(default = "default_kv_cache_mode")]
    pub mode: StageKvCacheMode,
    #[serde(default = "default_kv_cache_payload")]
    pub payload: StageKvCachePayload,
    #[serde(default = "default_kv_cache_max_entries")]
    pub max_entries: usize,
    #[serde(default)]
    pub max_bytes: u64,
    #[serde(default = "default_kv_cache_min_tokens")]
    pub min_tokens: u64,
    #[serde(default = "default_kv_cache_shared_stride_tokens")]
    pub shared_prefix_stride_tokens: u64,
    #[serde(default = "default_kv_cache_shared_record_limit")]
    pub shared_prefix_record_limit: u64,
}

fn default_kv_cache_mode() -> StageKvCacheMode {
    StageKvCacheMode::Auto
}

fn default_kv_cache_payload() -> StageKvCachePayload {
    StageKvCachePayload::Auto
}

fn default_kv_cache_max_entries() -> usize {
    64
}

fn default_kv_cache_min_tokens() -> u64 {
    64
}

fn default_kv_cache_shared_stride_tokens() -> u64 {
    128
}

fn default_kv_cache_shared_record_limit() -> u64 {
    2
}

fn default_ctx_size() -> u32 {
    512
}

fn default_lane_count() -> u32 {
    4
}

fn default_cache_type() -> String {
    "f16".to_string()
}

fn default_native_mtp_enabled() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PeerConfig {
    pub stage_id: String,
    pub stage_index: u32,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageTopology {
    pub topology_id: String,
    pub model_id: String,
    pub stages: Vec<StageTopologyEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageTopologyEntry {
    pub stage_id: String,
    pub stage_index: u32,
    pub host: Option<String>,
    pub endpoint: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub load_mode: LoadMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationDType {
    Unknown,
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationLayout {
    Opaque,
    TokenMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ActivationDescriptor {
    pub version: u32,
    pub dtype: ActivationDType,
    pub layout: ActivationLayout,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub payload_bytes: u64,
    #[serde(default)]
    pub flags: u64,
    #[serde(default)]
    pub payload_sha256: Option<String>,
}
