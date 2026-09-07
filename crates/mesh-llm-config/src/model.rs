mod built_in_schema;
mod multimodal;
mod profile;
mod runtime;
mod schema_types;

pub use built_in_schema::{
    BuiltInConfigPathResolution, built_in_config_schema_descriptor, built_in_config_settings,
    canonicalize_built_in_config_identifier, canonicalize_built_in_config_path,
    resolve_built_in_config_identifier, resolve_built_in_config_path,
};
pub use multimodal::MultimodalConfig;
pub(crate) use multimodal::merge_multimodal;
pub use runtime::{
    ActivityAdvertisement, ActivityResponse, DEFAULT_DRAIN_TIMEOUT_MAX_SECS,
    DEFAULT_DRAIN_TIMEOUT_SECS, RuntimeActivityConfig, RuntimeMode, StartupFailurePolicy,
};
pub use schema_types::*;

pub use mesh_llm_types::runtime::ModelRuntimeKind;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
pub use skippy_protocol::FlashAttentionType;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub mesh_requirements: MeshRequirementsConfig,
    #[serde(default)]
    pub owner_control: OwnerControlConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OwnerControlConfig {
    #[serde(default)]
    pub bind: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub advertise_addr: Option<std::net::SocketAddr>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct GpuConfig {
    #[serde(default)]
    pub assignment: GpuAssignment,
    #[serde(default)]
    pub parallel: Option<usize>,
}

pub const DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS: u64 = 2;
pub const DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS: u64 = 60 * 60;

fn default_model_target_demand_upgrade_min_requests() -> u64 {
    DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS
}

fn default_model_target_demand_upgrade_max_age_secs() -> u64 {
    DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS
}

fn default_drain_timeout_secs() -> u64 {
    runtime::DEFAULT_DRAIN_TIMEOUT_SECS
}

fn default_drain_timeout_max_secs() -> u64 {
    runtime::DEFAULT_DRAIN_TIMEOUT_MAX_SECS
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub listen_all: bool,
    /// Operating mode. Absent resolves to `Serve` for backward compatibility.
    #[serde(default)]
    pub mode: runtime::RuntimeMode,
    /// How the runtime reacts when a model fails to load during startup.
    #[serde(default)]
    pub startup_failure_policy: runtime::StartupFailurePolicy,
    /// Seconds before forcibly unloading a draining instance (default 30).
    #[serde(default = "default_drain_timeout_secs")]
    pub drain_timeout_secs: u64,
    /// Maximum allowed drain timeout cap in seconds (default 300).
    #[serde(default = "default_drain_timeout_max_secs")]
    pub drain_timeout_max_secs: u64,
    /// Host activity detection and response policy.
    #[serde(default)]
    pub activity: runtime::RuntimeActivityConfig,
    #[serde(default)]
    pub reconcile_model_targets: bool,
    #[serde(default)]
    pub reconcile_model_target_demand_upgrades: bool,
    #[serde(default)]
    pub native_runtime: NativeRuntimeConfig,
    #[serde(default = "default_model_target_demand_upgrade_min_requests")]
    pub model_target_demand_upgrade_min_requests: u64,
    #[serde(default = "default_model_target_demand_upgrade_max_age_secs")]
    pub model_target_demand_upgrade_max_age_secs: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            debug: false,
            listen_all: false,
            mode: runtime::RuntimeMode::default(),
            startup_failure_policy: runtime::StartupFailurePolicy::default(),
            drain_timeout_secs: runtime::DEFAULT_DRAIN_TIMEOUT_SECS,
            drain_timeout_max_secs: runtime::DEFAULT_DRAIN_TIMEOUT_MAX_SECS,
            activity: runtime::RuntimeActivityConfig::default(),
            reconcile_model_targets: false,
            reconcile_model_target_demand_upgrades: false,
            native_runtime: NativeRuntimeConfig::default(),
            model_target_demand_upgrade_min_requests:
                DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS,
            model_target_demand_upgrade_max_age_secs:
                DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NativeRuntimeConfig {
    #[serde(default)]
    pub mesh_version: Option<String>,
    #[serde(default)]
    pub skippy_abi: Option<String>,
    #[serde(default)]
    pub selection: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct MeshRequirementsConfig {
    #[serde(default)]
    pub min_node_version: Option<String>,
    #[serde(default)]
    pub max_node_version: Option<String>,
    #[serde(default)]
    pub min_protocol_version: Option<u32>,
    #[serde(default)]
    pub max_protocol_version: Option<u32>,
    #[serde(default)]
    pub require_release_attestation: bool,
    #[serde(default)]
    pub release_signer_keys: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuAssignment {
    #[default]
    Auto,
    Pinned,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelConfigDefaults {
    #[serde(default)]
    pub model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    pub hardware: Option<HardwareConfig>,
    #[serde(default)]
    pub throughput: Option<ThroughputConfig>,
    #[serde(default)]
    pub topology: Option<ModelTopologyConfig>,
    #[serde(default)]
    pub skippy: Option<SkippyConfig>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    pub multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    pub advanced: Option<AdvancedConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelConfigEntry {
    pub model: String,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
    pub gpu_id: Option<String>,
    pub parallel: Option<usize>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub batch: Option<u32>,
    pub ubatch: Option<u32>,
    pub flash_attention: Option<FlashAttentionType>,
    pub model_fit: Option<ModelFitConfig>,
    pub hardware: Option<HardwareConfig>,
    pub throughput: Option<ThroughputConfig>,
    pub topology: Option<ModelTopologyConfig>,
    pub skippy: Option<SkippyConfig>,
    pub speculative: Option<SpeculativeConfig>,
    pub request_defaults: Option<RequestDefaultsConfig>,
    pub multimodal: Option<MultimodalConfig>,
    pub advanced: Option<AdvancedConfig>,
    pub gpu_id_from_legacy_shim: bool,
}

impl Serialize for ModelConfigEntry {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ModelConfigEntry", 19)?;
        state.serialize_field("model", &self.model)?;
        if let Some(value) = &self.mmproj {
            state.serialize_field("mmproj", value)?;
        }
        if let Some(value) = &self.ctx_size {
            state.serialize_field("ctx_size", value)?;
        }
        if self.gpu_id_from_legacy_shim
            && let Some(value) = &self.gpu_id
        {
            state.serialize_field("gpu_id", value)?;
        }
        if let Some(value) = &self.parallel {
            state.serialize_field("parallel", value)?;
        }
        if let Some(value) = &self.cache_type_k {
            state.serialize_field("cache_type_k", value)?;
        }
        if let Some(value) = &self.cache_type_v {
            state.serialize_field("cache_type_v", value)?;
        }
        if let Some(value) = &self.batch {
            state.serialize_field("batch", value)?;
        }
        if let Some(value) = &self.ubatch {
            state.serialize_field("ubatch", value)?;
        }
        if let Some(value) = &self.flash_attention {
            state.serialize_field("flash_attention", value)?;
        }
        if let Some(value) = &self.model_fit {
            state.serialize_field("model_fit", value)?;
        }
        if let Some(value) = &self.hardware {
            state.serialize_field("hardware", value)?;
        }
        if let Some(value) = &self.throughput {
            state.serialize_field("throughput", value)?;
        }
        if let Some(value) = &self.topology {
            state.serialize_field("topology", value)?;
        }
        if let Some(value) = &self.skippy {
            state.serialize_field("skippy", value)?;
        }
        if let Some(value) = &self.speculative {
            state.serialize_field("speculative", value)?;
        }
        if let Some(value) = &self.request_defaults {
            state.serialize_field("request_defaults", value)?;
        }
        if let Some(value) = &self.multimodal {
            state.serialize_field("multimodal", value)?;
        }
        if let Some(value) = &self.advanced {
            state.serialize_field("advanced", value)?;
        }
        state.end()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelFitConfig {
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub kv_cache_policy: Option<String>,
    #[serde(default)]
    pub kv_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub kv_unified: Option<BoolOrAuto>,
    #[serde(default)]
    pub cache_ram_mib: Option<u64>,
    #[serde(default)]
    pub cache_idle_slots: Option<u32>,
    #[serde(default)]
    pub prompt_cache: Option<BoolOrAuto>,
    #[serde(default)]
    pub prefix_cache: Option<PrefixCacheConfig>,
    #[serde(default)]
    pub keep_tokens: Option<u32>,
    #[serde(default)]
    pub context_shift: Option<BoolOrAuto>,
    #[serde(default)]
    pub swa_full: Option<bool>,
    #[serde(default)]
    pub checkpoint_interval: Option<u32>,
    #[serde(default)]
    pub checkpoint_count: Option<u32>,
    #[serde(default)]
    pub lookup_cache_static: Option<String>,
    #[serde(default)]
    pub lookup_cache_dynamic: Option<String>,
    #[serde(default)]
    pub flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_entries: Option<u32>,
    #[serde(default)]
    pub max_bytes: Option<u64>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub shared_stride_tokens: Option<u32>,
    #[serde(default)]
    pub shared_record_limit: Option<u32>,
    #[serde(default)]
    pub payload_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HardwareConfig {
    #[serde(default)]
    pub model_runtime: Option<ModelRuntimeKind>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub gpu_layers: Option<IntegerOrString>,
    #[serde(default)]
    pub stage_layer_start: Option<u32>,
    #[serde(default)]
    pub stage_layer_end: Option<u32>,
    #[serde(default)]
    pub placement: Option<String>,
    #[serde(default)]
    pub tensor_split: Option<TensorSplitConfig>,
    #[serde(default)]
    pub split_mode: Option<String>,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub cpu_moe: Option<BoolOrAuto>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub rpc_backend: Option<toml::Value>,
    #[serde(default)]
    pub fit_target_mib: Option<u64>,
    #[serde(default)]
    pub safety_margin_gb: Option<f64>,
    #[serde(default)]
    pub fit_context: Option<BoolOrAuto>,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub hf_file: Option<String>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub lora_adapters: Vec<String>,
    #[serde(default)]
    pub control_vectors: Vec<String>,
    #[serde(default)]
    pub check_tensors: Option<bool>,
    /// Tensor type policy applied while opening a SafeTensors checkpoint.
    #[serde(default)]
    pub checkpoint_quantization: Option<String>,
    /// Importance matrix used by low-bit SafeTensors checkpoint quantization.
    #[serde(default)]
    pub checkpoint_imatrix: Option<String>,
    #[serde(default)]
    pub mmap: Option<BoolOrAuto>,
    #[serde(default)]
    pub use_mmap_prefetch: Option<bool>,
    #[serde(default)]
    pub use_mmap_buffer: Option<bool>,
    #[serde(default)]
    pub mlock: Option<bool>,
    #[serde(default)]
    pub direct_io: Option<bool>,
    #[serde(default)]
    pub repack: Option<bool>,
    #[serde(default)]
    pub op_offload: Option<bool>,
    #[serde(default)]
    pub no_host_buffer: Option<bool>,
    #[serde(default)]
    pub warmup: Option<BoolOrAuto>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThroughputConfig {
    #[serde(default)]
    pub parallel: Option<usize>,
    #[serde(default)]
    pub continuous_batching: Option<BoolOrAuto>,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub threads_batch: Option<usize>,
    #[serde(default)]
    pub threads_http: Option<usize>,
    #[serde(default)]
    pub priority: Option<IntegerOrString>,
    #[serde(default)]
    pub poll: Option<BoolOrString>,
    #[serde(default)]
    pub cpu_affinity: Option<StringOrStringList>,
    #[serde(default)]
    pub numa: Option<String>,
    #[serde(default)]
    pub slot_prompt_similarity: Option<f64>,
    #[serde(default)]
    pub sleep_idle_seconds: Option<u64>,
    #[serde(default)]
    pub tuning_profile: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelTopologyMode {
    Locked,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelTopologyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<ModelTopologyMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stages: Option<Vec<ModelTopologyStageConfig>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelTopologyStageConfig {
    pub node: ModelTopologyNodeSelector,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelTopologyNodeSelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

pub fn merge_model_topology(
    defaults: Option<&ModelTopologyConfig>,
    model: Option<&ModelTopologyConfig>,
) -> Option<ModelTopologyConfig> {
    let mut merged = defaults.cloned().unwrap_or_default();
    if let Some(model) = model {
        merged.mode = model.mode.or(merged.mode);
        merged.manifest_sha256 = model.manifest_sha256.clone().or(merged.manifest_sha256);
        merged.stages = model.stages.clone().or(merged.stages);
    }
    if merged.mode.is_none() && merged.manifest_sha256.is_none() && merged.stages.is_none() {
        None
    } else {
        Some(merged)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SkippyConfig {
    #[serde(default)]
    pub stage_model_path: Option<String>,
    #[serde(default)]
    pub source_policy: Option<String>,
    #[serde(default)]
    pub stage_role: Option<String>,
    #[serde(default)]
    pub stage_topology: Option<String>,
    #[serde(default)]
    pub binary_stage_transport: Option<String>,
    #[serde(default)]
    pub openai_frontend_mode: Option<toml::Value>,
    #[serde(default)]
    pub lifecycle_startup_timeout_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_readiness_interval_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_health_interval_ms: Option<u64>,
    #[serde(default)]
    pub prefill_chunking: Option<String>,
    #[serde(default)]
    pub prefill_chunk_size: Option<u32>,
    #[serde(default)]
    pub prefill_chunk_schedule: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeculativeConfig {
    pub strategy: Option<String>,
    pub mode: Option<String>,
    pub draft_model: Option<String>,
    pub draft_hf_repo: Option<String>,
    pub draft_hf_file: Option<String>,
    pub draft_selection_policy: Option<String>,
    pub pairing_fault: Option<String>,
    pub draft_max_tokens: Option<u32>,
    pub draft_min_tokens: Option<u32>,
    pub draft_acceptance_threshold: Option<f64>,
    pub draft_split_probability: Option<f64>,
    pub draft_gpu_layers: Option<i32>,
    pub draft_device: Option<String>,
    pub draft_threads: Option<usize>,
    pub draft_cache_type_k: Option<String>,
    pub draft_cache_type_v: Option<String>,
    pub ngram_min: Option<u32>,
    pub ngram_max: Option<u32>,
    pub ngram_max_proposal_tokens: Option<u32>,
    pub ngram_proposer: Option<String>,
    pub extension_max_tokens: Option<u32>,
    pub native_mtp_reject_cooldown_tokens: Option<u32>,
    pub native_mtp_suppress_cooldown_drafts: Option<bool>,
    pub native_mtp_suppress_cooldown_draft_limit: Option<u32>,
    pub verify_window_min_tokens: Option<u32>,
    pub verify_window_max_tokens: Option<u32>,
    pub verify_window_pipeline_depth: Option<u32>,
    pub verify_window_runahead_tokens: Option<u32>,
    pub spec_default: Option<BoolOrAuto>,
    pub(crate) legacy_draft_model_path_used: bool,
}

impl SpeculativeConfig {
    /// Resolves the three supported policy layers without discarding fields
    /// that are not overridden by a more specific layer.
    pub fn with_precedence(
        overrides: Option<&Self>,
        model: Option<&Self>,
        defaults: Option<&Self>,
    ) -> Self {
        macro_rules! pick {
            ($field:ident) => {
                overrides
                    .and_then(|config| config.$field.clone())
                    .or_else(|| model.and_then(|config| config.$field.clone()))
                    .or_else(|| defaults.and_then(|config| config.$field.clone()))
            };
        }

        Self {
            strategy: pick!(strategy),
            mode: pick!(mode),
            draft_model: pick!(draft_model),
            draft_hf_repo: pick!(draft_hf_repo),
            draft_hf_file: pick!(draft_hf_file),
            draft_selection_policy: pick!(draft_selection_policy),
            pairing_fault: pick!(pairing_fault),
            draft_max_tokens: pick!(draft_max_tokens),
            draft_min_tokens: pick!(draft_min_tokens),
            draft_acceptance_threshold: pick!(draft_acceptance_threshold),
            draft_split_probability: pick!(draft_split_probability),
            draft_gpu_layers: pick!(draft_gpu_layers),
            draft_device: pick!(draft_device),
            draft_threads: pick!(draft_threads),
            draft_cache_type_k: pick!(draft_cache_type_k),
            draft_cache_type_v: pick!(draft_cache_type_v),
            ngram_min: pick!(ngram_min),
            ngram_max: pick!(ngram_max),
            ngram_max_proposal_tokens: pick!(ngram_max_proposal_tokens),
            ngram_proposer: pick!(ngram_proposer),
            extension_max_tokens: pick!(extension_max_tokens),
            native_mtp_reject_cooldown_tokens: pick!(native_mtp_reject_cooldown_tokens),
            native_mtp_suppress_cooldown_drafts: pick!(native_mtp_suppress_cooldown_drafts),
            native_mtp_suppress_cooldown_draft_limit: pick!(
                native_mtp_suppress_cooldown_draft_limit
            ),
            verify_window_min_tokens: pick!(verify_window_min_tokens),
            verify_window_max_tokens: pick!(verify_window_max_tokens),
            verify_window_pipeline_depth: pick!(verify_window_pipeline_depth),
            verify_window_runahead_tokens: pick!(verify_window_runahead_tokens),
            spec_default: pick!(spec_default),
            legacy_draft_model_path_used: overrides
                .filter(|config| config.draft_model.is_some())
                .or_else(|| model.filter(|config| config.draft_model.is_some()))
                .or_else(|| defaults.filter(|config| config.draft_model.is_some()))
                .is_some_and(|config| config.legacy_draft_model_path_used),
        }
    }
}

/// Raw deserialization helper that accepts both `draft_model` and the legacy
/// `draft_model_path` key. The public `SpeculativeConfig` is constructed from
/// this after detecting which key was used.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativeConfigRaw {
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    draft_model: Option<String>,
    #[serde(default)]
    draft_model_path: Option<String>,
    #[serde(default)]
    draft_hf_repo: Option<String>,
    #[serde(default)]
    draft_hf_file: Option<String>,
    #[serde(default)]
    draft_selection_policy: Option<String>,
    #[serde(default)]
    pairing_fault: Option<String>,
    #[serde(default)]
    draft_max_tokens: Option<u32>,
    #[serde(default)]
    draft_min_tokens: Option<u32>,
    #[serde(default)]
    draft_acceptance_threshold: Option<f64>,
    #[serde(default)]
    draft_split_probability: Option<f64>,
    #[serde(default)]
    draft_gpu_layers: Option<i32>,
    #[serde(default)]
    draft_device: Option<String>,
    #[serde(default)]
    draft_threads: Option<usize>,
    #[serde(default)]
    draft_cache_type_k: Option<String>,
    #[serde(default)]
    draft_cache_type_v: Option<String>,
    #[serde(default)]
    ngram_min: Option<u32>,
    #[serde(default)]
    ngram_max: Option<u32>,
    #[serde(default)]
    ngram_max_proposal_tokens: Option<u32>,
    #[serde(default)]
    ngram_proposer: Option<String>,
    #[serde(default)]
    extension_max_tokens: Option<u32>,
    #[serde(default)]
    native_mtp_reject_cooldown_tokens: Option<u32>,
    #[serde(default)]
    native_mtp_suppress_cooldown_drafts: Option<bool>,
    #[serde(default)]
    native_mtp_suppress_cooldown_draft_limit: Option<u32>,
    #[serde(default)]
    verify_window_min_tokens: Option<u32>,
    #[serde(default)]
    verify_window_max_tokens: Option<u32>,
    #[serde(default)]
    verify_window_pipeline_depth: Option<u32>,
    #[serde(default)]
    verify_window_runahead_tokens: Option<u32>,
    #[serde(default)]
    spec_default: Option<BoolOrAuto>,
}

impl<'de> Deserialize<'de> for SpeculativeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = SpeculativeConfigRaw::deserialize(deserializer)?;
        let legacy_used = raw.draft_model_path.is_some();
        if raw.draft_model.is_some() && raw.draft_model_path.is_some() {
            return Err(serde::de::Error::custom(
                "speculative config cannot set both `draft_model` and the legacy `draft_model_path`; \
                 use `draft_model` only",
            ));
        }
        Ok(SpeculativeConfig {
            strategy: raw.strategy,
            mode: raw.mode,
            draft_model: raw.draft_model.or(raw.draft_model_path),
            draft_hf_repo: raw.draft_hf_repo,
            draft_hf_file: raw.draft_hf_file,
            draft_selection_policy: raw.draft_selection_policy,
            pairing_fault: raw.pairing_fault,
            draft_max_tokens: raw.draft_max_tokens,
            draft_min_tokens: raw.draft_min_tokens,
            draft_acceptance_threshold: raw.draft_acceptance_threshold,
            draft_split_probability: raw.draft_split_probability,
            draft_gpu_layers: raw.draft_gpu_layers,
            draft_device: raw.draft_device,
            draft_threads: raw.draft_threads,
            draft_cache_type_k: raw.draft_cache_type_k,
            draft_cache_type_v: raw.draft_cache_type_v,
            ngram_min: raw.ngram_min,
            ngram_max: raw.ngram_max,
            ngram_max_proposal_tokens: raw.ngram_max_proposal_tokens,
            ngram_proposer: raw.ngram_proposer,
            extension_max_tokens: raw.extension_max_tokens,
            native_mtp_reject_cooldown_tokens: raw.native_mtp_reject_cooldown_tokens,
            native_mtp_suppress_cooldown_drafts: raw.native_mtp_suppress_cooldown_drafts,
            native_mtp_suppress_cooldown_draft_limit: raw.native_mtp_suppress_cooldown_draft_limit,
            verify_window_min_tokens: raw.verify_window_min_tokens,
            verify_window_max_tokens: raw.verify_window_max_tokens,
            verify_window_pipeline_depth: raw.verify_window_pipeline_depth,
            verify_window_runahead_tokens: raw.verify_window_runahead_tokens,
            spec_default: raw.spec_default,
            legacy_draft_model_path_used: legacy_used,
        })
    }
}

impl Serialize for SpeculativeConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(32))?;
        map.serialize_entry("strategy", &self.strategy)?;
        map.serialize_entry("mode", &self.mode)?;
        if self.legacy_draft_model_path_used {
            if let Some(ref v) = self.draft_model {
                map.serialize_entry("draft_model_path", v)?;
            }
        } else if let Some(ref v) = self.draft_model {
            map.serialize_entry("draft_model", v)?;
        }
        map.serialize_entry("draft_hf_repo", &self.draft_hf_repo)?;
        map.serialize_entry("draft_hf_file", &self.draft_hf_file)?;
        map.serialize_entry("draft_selection_policy", &self.draft_selection_policy)?;
        map.serialize_entry("pairing_fault", &self.pairing_fault)?;
        map.serialize_entry("draft_max_tokens", &self.draft_max_tokens)?;
        map.serialize_entry("draft_min_tokens", &self.draft_min_tokens)?;
        map.serialize_entry(
            "draft_acceptance_threshold",
            &self.draft_acceptance_threshold,
        )?;
        map.serialize_entry("draft_split_probability", &self.draft_split_probability)?;
        map.serialize_entry("draft_gpu_layers", &self.draft_gpu_layers)?;
        map.serialize_entry("draft_device", &self.draft_device)?;
        map.serialize_entry("draft_threads", &self.draft_threads)?;
        map.serialize_entry("draft_cache_type_k", &self.draft_cache_type_k)?;
        map.serialize_entry("draft_cache_type_v", &self.draft_cache_type_v)?;
        map.serialize_entry("ngram_min", &self.ngram_min)?;
        map.serialize_entry("ngram_max", &self.ngram_max)?;
        map.serialize_entry("ngram_max_proposal_tokens", &self.ngram_max_proposal_tokens)?;
        map.serialize_entry("ngram_proposer", &self.ngram_proposer)?;
        map.serialize_entry("extension_max_tokens", &self.extension_max_tokens)?;
        map.serialize_entry(
            "native_mtp_reject_cooldown_tokens",
            &self.native_mtp_reject_cooldown_tokens,
        )?;
        map.serialize_entry(
            "native_mtp_suppress_cooldown_drafts",
            &self.native_mtp_suppress_cooldown_drafts,
        )?;
        map.serialize_entry(
            "native_mtp_suppress_cooldown_draft_limit",
            &self.native_mtp_suppress_cooldown_draft_limit,
        )?;
        map.serialize_entry("verify_window_min_tokens", &self.verify_window_min_tokens)?;
        map.serialize_entry("verify_window_max_tokens", &self.verify_window_max_tokens)?;
        map.serialize_entry(
            "verify_window_pipeline_depth",
            &self.verify_window_pipeline_depth,
        )?;
        map.serialize_entry(
            "verify_window_runahead_tokens",
            &self.verify_window_runahead_tokens,
        )?;
        map.serialize_entry("spec_default", &self.spec_default)?;
        map.end()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequestDefaultsConfig {
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StringOrStringList>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub typical_p: Option<f64>,
    #[serde(default)]
    pub top_nsigma: Option<f64>,
    #[serde(default)]
    pub dynatemp_range: Option<f64>,
    #[serde(default)]
    pub dynatemp_exponent: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_last_n: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub dry: Option<DrySamplingConfig>,
    #[serde(default)]
    pub xtc: Option<XtcSamplingConfig>,
    #[serde(default)]
    pub adaptive: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub mirostat_mode: Option<IntegerOrString>,
    #[serde(default)]
    pub mirostat_entropy: Option<f64>,
    #[serde(default)]
    pub mirostat_learning_rate: Option<f64>,
    #[serde(default)]
    pub samplers: Option<Vec<String>>,
    #[serde(default)]
    pub sampler_sequence: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub logit_bias: Option<toml::Value>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub backend_sampling: Option<toml::Value>,
    #[serde(default)]
    pub reasoning_format: Option<String>,
    #[serde(default)]
    pub reasoning_enabled: Option<ReasoningEnabled>,
    #[serde(default)]
    pub reasoning_budget: Option<ReasoningBudget>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_file: Option<String>,
    #[serde(default)]
    pub jinja: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<toml::Value>,
    #[serde(default)]
    pub skip_chat_parsing: Option<bool>,
    #[serde(default)]
    pub prefill_assistant: Option<toml::Value>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub grammar: Option<toml::Value>,
    #[serde(default)]
    pub json_schema: Option<toml::Value>,
    #[serde(default)]
    pub logprobs: Option<toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DrySamplingConfig {
    #[serde(default)]
    pub multiplier: Option<f64>,
    #[serde(default)]
    pub base: Option<f64>,
    #[serde(default)]
    pub allowed_length: Option<i32>,
    #[serde(default)]
    pub penalty_last_n: Option<i32>,
    #[serde(default)]
    pub sequence_breakers: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct XtcSamplingConfig {
    #[serde(default)]
    pub probability: Option<f64>,
    #[serde(default)]
    pub threshold: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedConfig {
    #[serde(default)]
    pub server: Option<AdvancedServerConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedServerConfig {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub reuse_port: Option<bool>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub metrics: Option<bool>,
    #[serde(default)]
    pub slots: Option<bool>,
    #[serde(default)]
    pub props: Option<bool>,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub api_prefix: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrAuto {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrString {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum IntegerOrString {
    Integer(i64),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrStringList {
    String(String),
    List(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TensorSplitConfig {
    Ratios(Vec<f64>),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningEnabled {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningBudget {
    Integer(u32),
    String(String),
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReservedObjectConfig {}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawMeshConfig {
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    gpu: GpuConfig,
    #[serde(default)]
    mesh_requirements: MeshRequirementsConfig,
    #[serde(default)]
    owner_control: OwnerControlConfig,
    #[serde(default)]
    telemetry: TelemetryConfig,
    #[serde(default)]
    logging: LoggingConfig,
    #[serde(default)]
    defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    runtime: RuntimeConfig,
    #[serde(default)]
    models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigDefaults {
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    topology: Option<ModelTopologyConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigEntry {
    model: String,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    topology: Option<ModelTopologyConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
}

impl<'de> Deserialize<'de> for MeshConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawMeshConfig::deserialize(deserializer)?;
        Ok(Self {
            version: raw.version,
            gpu: raw.gpu,
            mesh_requirements: raw.mesh_requirements,
            owner_control: raw.owner_control,
            telemetry: raw.telemetry,
            logging: raw.logging,
            defaults: raw.defaults,
            runtime: raw.runtime,
            models: raw.models,
            plugins: raw.plugins,
            extra: raw.extra,
        })
    }
}

impl<'de> Deserialize<'de> for ModelConfigDefaults {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigDefaults::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl<'de> Deserialize<'de> for ModelConfigEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigEntry::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl ModelConfigDefaults {
    fn from_raw(raw: RawModelConfigDefaults) -> Self {
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k,
            raw.cache_type_v,
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let hardware = merge_hardware(raw.hardware, raw.gpu_id, None, None);
        let throughput = merge_throughput(raw.throughput, raw.parallel);
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj);
        Self {
            model_fit,
            hardware,
            throughput,
            topology: raw.topology,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
        }
    }
}

impl ModelConfigEntry {
    fn from_raw(raw: RawModelConfigEntry) -> Self {
        let gpu_id_from_legacy_shim = raw.gpu_id.is_some();
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k.clone(),
            raw.cache_type_v.clone(),
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj.clone());
        let hardware = merge_hardware(
            raw.hardware,
            raw.gpu_id.clone(),
            multimodal.as_ref().and_then(|m| m.mmproj.clone()),
            multimodal.as_ref().and_then(|m| m.mmproj_offload.clone()),
        );
        let throughput = merge_throughput(raw.throughput, raw.parallel);

        Self {
            model: raw.model,
            mmproj: multimodal
                .as_ref()
                .and_then(|config| config.mmproj.clone())
                .or_else(|| hardware.as_ref().and_then(|config| config.mmproj.clone()))
                .or(raw.mmproj),
            ctx_size: model_fit.as_ref().and_then(|config| config.ctx_size),
            gpu_id: hardware
                .as_ref()
                .and_then(|config| config.device.clone())
                .or(raw.gpu_id),
            parallel: throughput.as_ref().and_then(|config| config.parallel),
            cache_type_k: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_k.clone())
                .or(raw.cache_type_k),
            cache_type_v: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_v.clone())
                .or(raw.cache_type_v),
            batch: model_fit.as_ref().and_then(|config| config.batch),
            ubatch: model_fit.as_ref().and_then(|config| config.ubatch),
            flash_attention: model_fit
                .as_ref()
                .and_then(|config| config.flash_attention)
                .or(raw.flash_attention),
            model_fit,
            hardware,
            throughput,
            topology: raw.topology,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
            gpu_id_from_legacy_shim,
        }
    }
}

pub(crate) fn merge_model_fit(
    current: Option<ModelFitConfig>,
    ctx_size: Option<u32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    batch: Option<u32>,
    ubatch: Option<u32>,
    flash_attention: Option<FlashAttentionType>,
) -> Option<ModelFitConfig> {
    let mut config = current.unwrap_or_default();
    config.ctx_size = config.ctx_size.or(ctx_size);
    config.cache_type_k = config.cache_type_k.or(cache_type_k);
    config.cache_type_v = config.cache_type_v.or(cache_type_v);
    config.batch = config.batch.or(batch);
    config.ubatch = config.ubatch.or(ubatch);
    config.flash_attention = config.flash_attention.or(flash_attention);
    if is_model_fit_empty(&config) {
        None
    } else {
        Some(config)
    }
}

pub(crate) fn merge_hardware(
    current: Option<HardwareConfig>,
    gpu_id: Option<String>,
    mmproj: Option<String>,
    mmproj_offload: Option<BoolOrAuto>,
) -> Option<HardwareConfig> {
    let mut config = current.unwrap_or_default();
    config.device = config.device.or(gpu_id);
    config.mmproj = config.mmproj.or(mmproj);
    config.mmproj_offload = config.mmproj_offload.or(mmproj_offload);
    if is_hardware_empty(&config) {
        None
    } else {
        Some(config)
    }
}

pub(crate) fn merge_throughput(
    current: Option<ThroughputConfig>,
    parallel: Option<usize>,
) -> Option<ThroughputConfig> {
    let mut config = current.unwrap_or_default();
    config.parallel = config.parallel.or(parallel);
    if is_throughput_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn is_model_fit_empty(config: &ModelFitConfig) -> bool {
    config == &ModelFitConfig::default()
}

fn is_hardware_empty(config: &HardwareConfig) -> bool {
    config == &HardwareConfig::default()
}

fn is_throughput_empty(config: &ThroughputConfig) -> bool {
    config == &ThroughputConfig::default()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub service_name: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub export_interval_secs: Option<u64>,
    #[serde(default)]
    pub queue_size: Option<usize>,
    #[serde(default)]
    pub prompt_shape_metrics: bool,
    #[serde(default)]
    pub metrics: TelemetryMetricsConfig,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct AuditConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub log_path: Option<std::path::PathBuf>,
    #[serde(default)]
    pub log_format: Option<String>,
    #[serde(default)]
    pub log_level: Option<String>,
    #[serde(default)]
    pub max_file_size_mb: Option<u64>,
    #[serde(default)]
    pub max_files: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryMetricsConfig {
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// Capture mode for logging artifacts. `MetadataOnly` is the default to avoid
/// persisting sensitive content by accident; opt into `RedactedArtifacts` only
/// when explicit redaction guarantees are acceptable.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    #[default]
    MetadataOnly,
    RedactedArtifacts,
}

/// Configuration for the operator logging subsystem. Additive config-v1 section; existing TOML files parse unchanged.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LoggingConfig {
    /// Whether the logging subsystem is enabled (default: true).
    #[serde(default = "default_logging_enabled")]
    pub enabled: bool,

    /// Optional override for the application-state root directory. When absent, uses the runtime default path. Rejects unsafe roots (world-writable, escaping, empty).
    #[serde(default)]
    pub application_state_root: Option<std::path::PathBuf>,

    /// Maximum Unicode characters in each payload-free local presentation
    /// summary line (default: 2048). Must be between 1 and 65_536. Changes
    /// require a process restart.
    #[serde(default = "default_summary_limit")]
    pub summary_line_limit: u64,

    /// Process-restart hard ceiling on entries held by the in-memory replay
    /// buffer (default: 10_000). Must be between 50 and 100_000.
    #[serde(default = "default_event_buffer_size")]
    pub event_buffer_size: u64,

    /// Retention TTL in seconds for application-state data (default: 36 hours / 129_600s). Must be between 3600 and 7_776_000.
    #[serde(default = "default_retention_ttl_secs")]
    pub retention_ttl_secs: u64,

    /// Maximum number of terminal summaries retained locally (default: 100_000).
    /// Active summaries are never eligible for cap pruning. Must be between 1
    /// and 1_000_000. Changes require a process restart.
    #[serde(default = "default_retention_max_rows")]
    pub retention_max_rows: u64,

    /// Initial and live current replay target in entries (default: 128). Must
    /// be between 1 and 10_000 and must not exceed `event_buffer_size`. Changes
    /// apply dynamically.
    #[serde(default = "default_replay_capacity")]
    pub replay_capacity: u32,

    /// Maximum entries in the persistence/dispatch queue (default: 4096). Must
    /// be between 64 and 131_072. It does not control replay retention. Changes
    /// require a process restart.
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: u32,

    /// Artifact capture configuration.
    #[serde(default)]
    pub artifact: LoggingArtifactConfig,

    /// Maximum export batch size in bytes (default: 5 MB). Must be between 64 KiB and 100 MiB.
    #[serde(default = "default_export_limit_bytes")]
    pub export_limit_bytes: u64,

    /// Cleanup cadence in seconds (default: every hour / 3600s). Must be between 300 and 86_400.
    #[serde(default = "default_cleanup_cadence_secs")]
    pub cleanup_cadence_secs: u64,

    /// Webhook dispatch configuration.
    #[serde(default)]
    pub webhook: LoggingWebhookConfig,

    /// Security audit stream configuration. Nested under `[logging]` so the
    /// logging master switch (`logging.enabled`) gates the audit stream too.
    #[serde(default)]
    pub audit: AuditConfig,
}

fn default_logging_enabled() -> bool {
    true
}

fn default_summary_limit() -> u64 {
    2048
}

fn default_event_buffer_size() -> u64 {
    10_000
}

fn default_retention_ttl_secs() -> u64 {
    36 * 3600 // 36 hours
}

fn default_retention_max_rows() -> u64 {
    100_000
}

fn default_replay_capacity() -> u32 {
    128
}

fn default_queue_capacity() -> u32 {
    4096
}

fn default_export_limit_bytes() -> u64 {
    5 * 1024 * 1024 // 5 MB
}

fn default_cleanup_cadence_secs() -> u64 {
    3600 // every hour
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LoggingArtifactConfig {
    /// Capture mode: metadata-only (default) or redacted-artifacts opt-in.
    #[serde(default)]
    pub capture_mode: CaptureMode,

    /// Maximum bytes per individual artifact before truncation (default: 256 KiB). Must be between 1024 and 16 MiB.
    #[serde(default = "default_artifact_byte_limit")]
    pub byte_limit_bytes: u64,

    /// Aggregate byte budget across all artifacts in one batch (default: 8 MiB). Must be between 512 KiB and 500 MiB.
    #[serde(default = "default_artifact_aggregate_bytes")]
    pub aggregate_limit_bytes: u64,
}

fn default_artifact_byte_limit() -> u64 {
    256 * 1024 // 256 KiB
}

fn default_artifact_aggregate_bytes() -> u64 {
    8 * 1024 * 1024 // 8 MiB
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LoggingWebhookConfig {
    /// Whether webhook dispatch is enabled (default: false).
    #[serde(default)]
    pub enabled: bool,

    /// Webhook target URL. Must be a valid HTTP/HTTPS URL when set.
    #[serde(default)]
    pub url: Option<String>,

    /// Maximum delivery attempts per event batch (default: 3). Must be between 1 and 20.
    #[serde(default = "default_webhook_max_attempts")]
    pub max_attempts: u32,

    /// Per-attempt timeout in seconds (default: 15). Must be between 1 and 60.
    #[serde(default = "default_webhook_timeout_secs")]
    pub timeout_secs: u64,

    /// Dead-letter retention window in seconds for failed dispatches (default: 72 hours / 259_200s). Must be between 3600 and 1_555_200.
    #[serde(default = "default_webhook_dead_letter_secs")]
    pub dead_letter_retention_secs: u64,
}

fn default_webhook_max_attempts() -> u32 {
    3
}

fn default_webhook_timeout_secs() -> u64 {
    15
}

fn default_webhook_dead_letter_secs() -> u64 {
    72 * 3600 // 72 hours
}

impl Default for LoggingArtifactConfig {
    fn default() -> Self {
        Self {
            capture_mode: CaptureMode::default(),
            byte_limit_bytes: default_artifact_byte_limit(),
            aggregate_limit_bytes: default_artifact_aggregate_bytes(),
        }
    }
}

impl Default for LoggingWebhookConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: None,
            max_attempts: default_webhook_max_attempts(),
            timeout_secs: default_webhook_timeout_secs(),
            dead_letter_retention_secs: default_webhook_dead_letter_secs(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: default_logging_enabled(),
            application_state_root: None,
            summary_line_limit: default_summary_limit(),
            event_buffer_size: default_event_buffer_size(),
            retention_ttl_secs: default_retention_ttl_secs(),
            retention_max_rows: default_retention_max_rows(),
            replay_capacity: default_replay_capacity(),
            queue_capacity: default_queue_capacity(),
            artifact: LoggingArtifactConfig::default(),
            export_limit_bytes: default_export_limit_bytes(),
            cleanup_cadence_secs: default_cleanup_cadence_secs(),
            webhook: LoggingWebhookConfig::default(),
            audit: AuditConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_ui_enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional URL passed to the plugin as `MESH_LLM_PLUGIN_URL`.
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub settings: BTreeMap<String, toml::Value>,
    #[serde(default, skip_serializing_if = "PluginStartupConfig::is_default")]
    pub startup: PluginStartupConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginWebUiPreference {
    None,
    Enabled,
    Disabled,
}

impl PluginWebUiPreference {
    pub const fn resolve(web_ui_enabled: Option<bool>, declares_web_ui: bool) -> Self {
        match (declares_web_ui, web_ui_enabled) {
            (false, _) => Self::None,
            (true, Some(false)) => Self::Disabled,
            (true, Some(true) | None) => Self::Enabled,
        }
    }
}

impl PluginConfigEntry {
    pub const fn web_ui_preference(&self, declares_web_ui: bool) -> PluginWebUiPreference {
        PluginWebUiPreference::resolve(self.web_ui_enabled, declares_web_ui)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PluginStartupConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_timeout_secs: Option<u64>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub lazy_start: bool,
}

impl PluginStartupConfig {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}
