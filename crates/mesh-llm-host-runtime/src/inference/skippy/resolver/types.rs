use std::path::{Path, PathBuf};

use skippy_protocol::{FlashAttentionType, StageKvCacheMode, StageKvCachePayload};
use skippy_runtime::package::PackageGenerationInfo;
use skippy_server::{EmbeddedOpenAiRequestDefaults, SpeculativeDecodeConfig};

use crate::plugin::{MeshConfig, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig};

pub(super) const BUILTIN_CTX_SIZE: u32 = 4096;
pub(super) const BUILTIN_BATCH: u32 = 512;
/// Matches llama.cpp's own default (`LLAMA_SERVER_DEFAULT_N_UBATCH = 512`) and clears
/// the CUDA SSM SSD kernel gate (`n_tok > SSM_SSD_MIN_TOKENS`, 128, strict), which the
/// previous 128 default missed by exactly one token — forcing every recurrent (mamba)
/// prefill onto the sequential scan fallback. Measured on granite-4.0-h-1b: TTFT p50
/// 0.670 → 0.415 s (C1) and 6.38 → 3.97 s (C8); decode 22.2 → 39.4 tok/s at C8.
/// See WHITE_UBATCH_512_FALSIFICATION_2026_09_08 in the 2026-09-08 competitive bench.
pub(super) const BUILTIN_UBATCH: u32 = 512;
pub(super) const BUILTIN_PARALLEL: usize = 32;
pub(super) const BUILTIN_PREFILL_CHUNK_SIZE: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_START: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_STEP: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_MAX: usize = 512;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_TARGET_MS: f64 = 100.0;
pub(super) const BUILTIN_SAFETY_MARGIN_GB: f64 = 2.0;

#[derive(Clone, Debug)]
pub(crate) struct SkippyConfigResolveRequest<'a> {
    pub(crate) mesh_config: &'a MeshConfig,
    pub(crate) model_id: &'a str,
    pub(crate) model_path: &'a Path,
    pub(crate) model_bytes: u64,
    pub(crate) allocatable_memory_bytes: Option<u64>,
    pub(crate) request_defaults: Option<&'a RequestDefaultsConfig>,
    pub(crate) package_generation: Option<&'a PackageGenerationInfo>,
    /// GGUF metadata for the model being resolved, when available. Used to
    /// guard the size-tiered KV cache default against quantised-KV load
    /// incompatibilities (Flash Attention / block alignment). `None` leaves the
    /// default unguarded — the pre-existing behaviour.
    pub(crate) compact_meta: Option<&'a crate::models::gguf::GgufCompactMeta>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSkippyConfig {
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) model_fit: ResolvedModelFitConfig,
    pub(crate) hardware: ResolvedHardwareConfig,
    pub(crate) throughput: ResolvedThroughputConfig,
    pub(crate) skippy: ResolvedSkippyExecutionConfig,
    pub(crate) speculative: ResolvedSpeculativeConfig,
    pub(crate) request_defaults: ResolvedRequestDefaultsConfig,
    pub(crate) multimodal: ResolvedMultimodalConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedMultimodalConfig {
    pub(crate) projector_url: Option<String>,
    pub(crate) projector_use_gpu: Option<bool>,
    pub(crate) media_marker: Option<String>,
    pub(crate) image_min_tokens: Option<u32>,
    pub(crate) image_max_tokens: Option<u32>,
    pub(crate) batch_max_tokens: Option<u32>,
    pub(crate) glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub(crate) generation_signal_window: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedModelFitConfig {
    pub(crate) ctx_size: u32,
    pub(crate) batch: u32,
    pub(crate) ubatch: u32,
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) kv_cache_policy: String,
    pub(crate) prefix_cache: ResolvedStageKvCache,
    pub(crate) kv_offload: String,
    /// Parsed `kv_offload` for the native tri-state control. `None` covers
    /// both "auto" and any value that did not parse to a bool.
    pub(crate) kv_offload_resolved: Option<bool>,
    pub(crate) kv_unified: Option<bool>,
    pub(crate) swa_full: Option<bool>,
    pub(crate) cache_idle_slots: Option<u32>,
    pub(crate) flash_attention: FlashAttentionType,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedHardwareConfig {
    pub(crate) device: Option<String>,
    pub(crate) gpu_layers: i32,
    pub(crate) mmap: Option<bool>,
    pub(crate) mlock: bool,
    pub(crate) repack: bool,
    pub(crate) op_offload: Option<bool>,
    pub(crate) no_host_buffer: bool,
    pub(crate) check_tensors: bool,
    pub(crate) checkpoint_quantization: Option<String>,
    pub(crate) checkpoint_imatrix: Option<String>,
    pub(crate) direct_io: bool,
    pub(crate) main_gpu: Option<u32>,
    pub(crate) split_mode: skippy_protocol::SplitMode,
    pub(crate) fit_target_mib: Option<u64>,
    pub(crate) resolved_model_path: PathBuf,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) stage_layer_start: Option<u32>,
    pub(crate) stage_layer_end: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedThroughputConfig {
    pub(crate) parallel: usize,
    pub(crate) continuous_batching: String,
    pub(crate) threads: Option<usize>,
    pub(crate) threads_batch: Option<usize>,
    pub(crate) tuning_profile: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSkippyExecutionConfig {
    pub(crate) binary_stage_transport: String,
    pub(crate) prefill_chunking: String,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) prefill_chunk_schedule: Option<String>,
    pub(crate) prefill_controls_explicit: bool,
    pub(crate) lifecycle_startup_timeout_ms: Option<u64>,
    pub(crate) lifecycle_readiness_interval_ms: Option<u64>,
    pub(crate) lifecycle_health_interval_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSpeculativeConfig {
    pub(crate) strategy: String,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) mode: String,
    pub(crate) draft_model_path: Option<PathBuf>,
    pub(crate) pairing_fault: String,
    pub(crate) draft_max_tokens: u32,
    pub(crate) draft_min_tokens: u32,
    pub(crate) explicit: bool,
    pub(crate) draft_n_gpu_layers: Option<i32>,
    pub(crate) decode: SpeculativeDecodeConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedStageKvCache {
    FamilyDefault,
    Disabled,
    Explicit(ResolvedStageKvCacheTemplate),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedStageKvCacheTemplate {
    pub(crate) mode: StageKvCacheMode,
    pub(crate) payload: StageKvCachePayload,
    pub(crate) max_entries: Option<usize>,
    pub(crate) max_bytes: Option<u64>,
    pub(crate) min_tokens: Option<u64>,
    pub(crate) shared_prefix_stride_tokens: Option<u64>,
    pub(crate) shared_prefix_record_limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedRequestDefaultsConfig {
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) presence_penalty: Option<f64>,
    pub(crate) frequency_penalty: Option<f64>,
    pub(crate) seed: Option<i64>,
    pub(crate) logit_bias: Option<toml::Value>,
    pub(crate) top_k: Option<i64>,
    pub(crate) min_p: Option<f64>,
    pub(crate) typical_p: Option<f64>,
    pub(crate) top_nsigma: Option<f64>,
    pub(crate) dynatemp_range: Option<f64>,
    pub(crate) dynatemp_exponent: Option<f64>,
    pub(crate) dry: Option<mesh_llm_config::DrySamplingConfig>,
    pub(crate) xtc: Option<mesh_llm_config::XtcSamplingConfig>,
    pub(crate) mirostat_mode: Option<mesh_llm_config::IntegerOrString>,
    pub(crate) mirostat_entropy: Option<f64>,
    pub(crate) mirostat_learning_rate: Option<f64>,
    pub(crate) samplers: Option<Vec<String>>,
    pub(crate) sampler_sequence: Option<String>,
    pub(crate) ignore_eos: Option<bool>,
    pub(crate) repeat_penalty: Option<f64>,
    pub(crate) repeat_last_n: Option<i64>,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) reasoning_format: Option<String>,
    pub(crate) reasoning_enabled: Option<ReasoningEnabled>,
    pub(crate) reasoning_budget: Option<ReasoningBudget>,
    pub(crate) chat_template: Option<String>,
    pub(crate) chat_template_file: Option<String>,
    pub(crate) jinja: Option<bool>,
    pub(crate) chat_template_kwargs: Option<toml::Value>,
    pub(crate) skip_chat_parsing: Option<bool>,
    pub(crate) prefill_assistant: Option<toml::Value>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) grammar: Option<toml::Value>,
    pub(crate) json_schema: Option<toml::Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedEmbeddedOpenAiArgs {
    pub(crate) model_id: Option<String>,
    pub(crate) default_max_tokens: u32,
    pub(crate) request_defaults: EmbeddedOpenAiRequestDefaults,
    pub(crate) generation_concurrency: usize,
    pub(crate) continuous_batching: bool,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) prefill_chunk_policy: String,
    pub(crate) prefill_chunk_schedule: Option<String>,
    pub(crate) prefill_adaptive_start: usize,
    pub(crate) prefill_adaptive_step: usize,
    pub(crate) prefill_adaptive_max: usize,
    pub(crate) prefill_adaptive_target_ms: f64,
    pub(crate) draft_model_path: Option<PathBuf>,
    pub(crate) speculative_window: usize,
    pub(crate) adaptive_speculative_window: bool,
    pub(crate) draft_n_gpu_layers: Option<i32>,
    pub(crate) speculative: SpeculativeDecodeConfig,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) native_mtp_draft_model_path: Option<PathBuf>,
    pub(crate) native_mtp_max_tokens: usize,
    pub(crate) native_mtp_min_tokens: usize,
    pub(crate) activation_width: i32,
    pub(crate) reply_credit_limit: Option<usize>,
    pub(crate) downstream_connect_timeout_secs: u64,
}
