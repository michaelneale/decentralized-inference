use std::{
    io::Read,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use openai_frontend::OpenAiHookPolicy;
use skippy_protocol::{
    LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::MtpSource;
use skippy_server::{
    DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS, EmbeddedOpenAiArgs, EmbeddedOpenAiRequestDefaults,
    EmbeddedRuntimeOptions, NativeMtpProposalConfig, SpeculativeDecodeConfig, telemetry::Telemetry,
};

use super::super::{
    SkippyDeviceDescriptor, SkippyModelLoadOptions, SkippyPackageIdentity, SkippyTelemetryOptions,
    family_policy_for_model_path, single_stage_config, synthetic_direct_gguf_package,
};
use super::request_defaults::{
    resolve_reasoning_budget, resolve_reasoning_enabled, resolve_reasoning_format,
    resolve_request_logit_bias, resolve_request_repeat_last_n, resolve_request_seed,
    resolve_request_top_k,
};
use super::support::resolve_prefill_chunk_policy;
use super::types::{
    BUILTIN_PREFILL_ADAPTIVE_MAX, BUILTIN_PREFILL_ADAPTIVE_START, BUILTIN_PREFILL_ADAPTIVE_STEP,
    BUILTIN_PREFILL_ADAPTIVE_TARGET_MS, BUILTIN_PREFILL_CHUNK_SIZE, ResolvedEmbeddedOpenAiArgs,
    ResolvedSkippyConfig, ResolvedStageKvCache,
};

/// Default maximum number of draft tokens for native MTP sidecar probes when
/// no explicit `draft_max_tokens` is configured. Three tokens is a reasonable
/// default: long enough to confirm or reject the draft trajectory without
/// over-committing speculative decode resources.
const DEFAULT_NATIVE_MTP_MAX_TOKENS: usize = 3;
const MAX_CHAT_TEMPLATE_BYTES: u64 = 1024 * 1024;

fn read_chat_template(path: &str) -> Result<String> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open request_defaults.chat_template_file {path}"))?;
    let mut bytes = Vec::new();
    file.take(MAX_CHAT_TEMPLATE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read request_defaults.chat_template_file {path}"))?;
    if bytes.len() as u64 > MAX_CHAT_TEMPLATE_BYTES {
        bail!(
            "request_defaults.chat_template_file {path} exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit"
        );
    }
    String::from_utf8(bytes)
        .with_context(|| format!("request_defaults.chat_template_file {path} is not UTF-8"))
}

impl ResolvedSkippyConfig {
    pub(crate) async fn materialize_projector_url(&mut self) -> Result<()> {
        if self.hardware.projector_path.is_some() {
            return Ok(());
        }
        let Some(projector_url) = self.multimodal.projector_url.as_deref() else {
            return Ok(());
        };
        self.hardware.projector_path = Some(
            crate::models::resolve::download_direct_ref_with_progress(projector_url, true)
                .await
                .with_context(|| format!("download multimodal.mmproj_url {projector_url}"))?,
        );
        Ok(())
    }

    pub(crate) fn to_model_load_options(
        &self,
        telemetry: SkippyTelemetryOptions,
    ) -> Result<SkippyModelLoadOptions> {
        self.build_model_load_options(telemetry, false)
    }

    fn build_model_load_options(
        &self,
        telemetry: SkippyTelemetryOptions,
        allow_staged_range: bool,
    ) -> Result<SkippyModelLoadOptions> {
        if !allow_staged_range {
            self.ensure_single_stage_safe()?;
        } else {
            self.ensure_embedded_openai_safe(true)?;
        }
        let mut options = self.base_model_load_options(telemetry);
        options.native_mtp_enabled = self.speculative.native_mtp_enabled;
        // Pre-compute the package identity so single_stage_config skips the
        // SHA-256 hash. Without this the same hash runs again in
        // SkippyModelHandle::load_with_hooks, doubling I/O (issue #717).
        if options.package_identity.is_none() {
            options.package_identity = Some(synthetic_direct_gguf_package(
                &options.model_id,
                &options.model_path,
            )?);
        }
        let stage_config = single_stage_config(&options)?;
        let family_policy = family_policy_for_model_path(&self.hardware.resolved_model_path);
        let kv_cache = self
            .resolve_stage_kv_cache(family_policy.stage_kv_cache_config_for_stage(&stage_config))?;
        Ok(options.with_kv_cache(kv_cache))
    }

    fn base_model_load_options(&self, telemetry: SkippyTelemetryOptions) -> SkippyModelLoadOptions {
        let mut options = SkippyModelLoadOptions::for_direct_gguf(
            self.model_id.clone(),
            self.hardware.resolved_model_path.clone(),
        )
        .with_ctx_size(self.model_fit.ctx_size)
        .with_generation_concurrency(self.throughput.parallel)
        .with_cache_types(&self.model_fit.cache_type_k, &self.model_fit.cache_type_v)
        .with_batch_sizes(Some(self.model_fit.batch), Some(self.model_fit.ubatch))
        .with_thread_counts(self.throughput.threads, self.throughput.threads_batch)
        .with_flash_attn_type(self.model_fit.flash_attention)
        .with_kv_session_controls(
            self.model_fit.kv_offload_resolved,
            self.model_fit.kv_unified,
            self.model_fit.swa_full,
        )
        .with_cache_idle_slots(self.model_fit.cache_idle_slots)
        .with_telemetry(telemetry);

        options.default_max_tokens = self.request_defaults.max_tokens;
        options.n_gpu_layers = self.hardware.gpu_layers;
        options.mmap = self.hardware.mmap;
        options.mlock = self.hardware.mlock;
        options.repack = self.hardware.repack;
        options.op_offload = self.hardware.op_offload;
        options.no_host_buffer = self.hardware.no_host_buffer;
        options.check_tensors = self.hardware.check_tensors;
        options
            .checkpoint_quantization
            .clone_from(&self.hardware.checkpoint_quantization);
        options
            .checkpoint_imatrix
            .clone_from(&self.hardware.checkpoint_imatrix);
        options.direct_io = self.hardware.direct_io;
        options.main_gpu = self.hardware.main_gpu;
        options.split_mode = self.hardware.split_mode;
        options.projector_use_gpu = self.multimodal.projector_use_gpu;
        options
            .media_marker
            .clone_from(&self.multimodal.media_marker);
        options.image_min_tokens = self.multimodal.image_min_tokens;
        options.image_max_tokens = self.multimodal.image_max_tokens;
        options.batch_max_tokens = self.multimodal.batch_max_tokens;
        options.glm_dsa_policy = self.multimodal.glm_dsa_policy;
        options.generation_signal_window = self.multimodal.generation_signal_window;
        if let Some(projector_path) = self.hardware.projector_path.clone() {
            options = options.with_projector_path(projector_path);
        }
        if let (Some(layer_start), Some(layer_end)) = (
            self.hardware.stage_layer_start,
            self.hardware.stage_layer_end,
        ) {
            options = options.with_layer_range(layer_start, layer_end);
        }
        if let Some(device) = self.hardware.device.clone() {
            options = options.with_selected_device(SkippyDeviceDescriptor {
                backend_device: device,
                stable_id: None,
                index: None,
                vram_bytes: None,
            });
        }
        options
    }

    pub(crate) fn to_stage_config(
        &self,
        package_identity: Option<SkippyPackageIdentity>,
        load_mode: LoadMode,
    ) -> Result<StageConfig> {
        self.ensure_embedded_openai_safe(true)?;
        let mut load_options = self.base_model_load_options(SkippyTelemetryOptions::off());
        load_options.native_mtp_enabled = self.speculative.native_mtp_enabled;
        if let Some(package_identity) = package_identity {
            load_options.package_identity = Some(package_identity.clone());
            if self.hardware.stage_layer_end.is_none() {
                load_options.layer_end = Some(package_identity.layer_count);
            }
            load_options.model_path = match &load_mode {
                LoadMode::LayerPackage => PathBuf::from(package_identity.package_ref),
                LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
                    self.hardware.resolved_model_path.clone()
                }
            };
        }
        let mut stage_config = single_stage_config(&load_options)?;
        stage_config.load_mode = load_mode;
        stage_config.filter_tensors_on_load =
            !matches!(stage_config.load_mode, LoadMode::RuntimeSlice)
                || stage_config.layer_start > 0;
        if matches!(stage_config.load_mode, LoadMode::LayerPackage)
            && load_options.package_identity.is_none()
        {
            let synthetic = synthetic_direct_gguf_package(&self.model_id, &self.model_path)?;
            stage_config.package_ref = Some(synthetic.package_ref.clone());
            stage_config.manifest_sha256 = Some(synthetic.manifest_sha256.clone());
        }
        let family_policy = family_policy_for_model_path(&self.hardware.resolved_model_path);
        stage_config.kv_cache = self
            .resolve_stage_kv_cache(family_policy.stage_kv_cache_config_for_stage(&stage_config))?;
        Ok(stage_config)
    }

    pub(crate) fn to_embedded_runtime_options(
        &self,
        telemetry: &SkippyTelemetryOptions,
        package_identity: Option<SkippyPackageIdentity>,
        load_mode: LoadMode,
    ) -> Result<EmbeddedRuntimeOptions> {
        Ok(EmbeddedRuntimeOptions {
            config: self.to_stage_config(package_identity, load_mode.clone())?,
            topology: None,
            n_threads: self.throughput.threads,
            n_threads_batch: self.throughput.threads_batch,
            mtp_source: if self.speculative.native_mtp_enabled {
                if self.speculative.draft_model_path.is_some() {
                    MtpSource::External
                } else {
                    MtpSource::Integrated
                }
            } else {
                MtpSource::Disabled
            },
            metrics_otlp_grpc: telemetry.metrics_otlp_grpc.clone(),
            telemetry_queue_capacity: telemetry.queue_capacity,
            telemetry_level: telemetry.level,
            operation_id: None,
            session_lifecycle_observer: None,
        })
    }

    pub(crate) fn to_embedded_openai_args(
        &self,
        activation_width: i32,
        staged: bool,
    ) -> Result<ResolvedEmbeddedOpenAiArgs> {
        self.ensure_embedded_openai_safe(staged)?;
        let mode = self.speculative_mode_for_embedded(staged);
        let chat_template = match (
            self.request_defaults.chat_template.as_ref(),
            self.request_defaults.chat_template_file.as_ref(),
        ) {
            (Some(template), _) => Some(template.clone()),
            (None, Some(path)) => Some(read_chat_template(path)?),
            (None, None) => None,
        };
        let chat_template_kwargs = self
            .request_defaults
            .chat_template_kwargs
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .context("serialize request_defaults.chat_template_kwargs")?;
        let prefill_assistant = self
            .request_defaults
            .prefill_assistant
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .context("serialize request_defaults.prefill_assistant")?;
        let grammar = self
            .request_defaults
            .grammar
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .context("serialize request_defaults.grammar")?;
        let json_schema = self
            .request_defaults
            .json_schema
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .context("serialize request_defaults.json_schema")?;
        Ok(ResolvedEmbeddedOpenAiArgs {
            model_id: Some(self.model_id.clone()),
            default_max_tokens: self.request_defaults.max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults {
                stop: self.request_defaults.stop.clone(),
                temperature: self.request_defaults.temperature.map(|value| value as f32),
                top_p: self.request_defaults.top_p.map(|value| value as f32),
                presence_penalty: self
                    .request_defaults
                    .presence_penalty
                    .map(|value| value as f32),
                frequency_penalty: self
                    .request_defaults
                    .frequency_penalty
                    .map(|value| value as f32),
                seed: self
                    .request_defaults
                    .seed
                    .map(resolve_request_seed)
                    .transpose()?,
                logit_bias: self
                    .request_defaults
                    .logit_bias
                    .as_ref()
                    .map(resolve_request_logit_bias)
                    .transpose()?,
                top_k: self
                    .request_defaults
                    .top_k
                    .map(resolve_request_top_k)
                    .transpose()?,
                min_p: self.request_defaults.min_p.map(|value| value as f32),
                typical_p: self.request_defaults.typical_p.map(|value| value as f32),
                top_nsigma: self.request_defaults.top_nsigma.map(|value| value as f32),
                dynatemp_range: self
                    .request_defaults
                    .dynatemp_range
                    .map(|value| value as f32),
                dynatemp_exponent: self
                    .request_defaults
                    .dynatemp_exponent
                    .map(|value| value as f32),
                dry: self.request_defaults.dry.as_ref().map(|dry| {
                    skippy_runtime::DrySamplingConfig {
                        multiplier: dry.multiplier.unwrap_or(0.0) as f32,
                        base: dry.base.unwrap_or(1.75) as f32,
                        allowed_length: dry.allowed_length.unwrap_or(2),
                        penalty_last_n: dry.penalty_last_n.unwrap_or(64),
                        sequence_breakers: dry.sequence_breakers.clone().unwrap_or_else(|| {
                            vec!["\n".into(), ":".into(), "\"".into(), "*".into()]
                        }),
                    }
                }),
                xtc: self.request_defaults.xtc.as_ref().map(|xtc| {
                    skippy_runtime::XtcSamplingConfig {
                        probability: xtc.probability.unwrap_or(0.0) as f32,
                        threshold: xtc.threshold.unwrap_or(0.1) as f32,
                    }
                }),
                mirostat_mode: self
                    .request_defaults
                    .mirostat_mode
                    .as_ref()
                    .and_then(|mode| match mode {
                        mesh_llm_config::IntegerOrString::Integer(value) => {
                            i32::try_from(*value).ok()
                        }
                        mesh_llm_config::IntegerOrString::String(value) => value.parse().ok(),
                    }),
                mirostat_entropy: self
                    .request_defaults
                    .mirostat_entropy
                    .map(|value| value as f32),
                mirostat_learning_rate: self
                    .request_defaults
                    .mirostat_learning_rate
                    .map(|value| value as f32),
                samplers: self.request_defaults.samplers.clone(),
                sampler_sequence: self.request_defaults.sampler_sequence.clone(),
                ignore_eos: self.request_defaults.ignore_eos,
                repeat_penalty: self
                    .request_defaults
                    .repeat_penalty
                    .map(|value| value as f32),
                repeat_last_n: self
                    .request_defaults
                    .repeat_last_n
                    .map(resolve_request_repeat_last_n)
                    .transpose()?,
                reasoning_format: self
                    .request_defaults
                    .reasoning_format
                    .as_deref()
                    .and_then(resolve_reasoning_format),
                reasoning_enabled: self
                    .request_defaults
                    .reasoning_enabled
                    .as_ref()
                    .and_then(resolve_reasoning_enabled),
                reasoning_budget: self
                    .request_defaults
                    .reasoning_budget
                    .as_ref()
                    .and_then(resolve_reasoning_budget),
                chat_template,
                jinja: self.request_defaults.jinja,
                chat_template_kwargs,
                skip_chat_parsing: self.request_defaults.skip_chat_parsing,
                prefill_assistant,
                system_prompt: self.request_defaults.system_prompt.clone(),
                grammar,
                json_schema,
            },
            generation_concurrency: self.throughput.parallel,
            continuous_batching: self.throughput.continuous_batching != "false",
            prefill_chunk_size: self.skippy.prefill_chunk_size,
            prefill_chunk_policy: resolve_prefill_chunk_policy(&self.skippy.prefill_chunking),
            prefill_chunk_schedule: self.skippy.prefill_chunk_schedule.clone(),
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            prefill_adaptive_target_ms: BUILTIN_PREFILL_ADAPTIVE_TARGET_MS,
            draft_model_path: if mode == "draft" {
                self.speculative.draft_model_path.clone()
            } else {
                None
            },
            speculative_window: self.speculative_window_for_embedded(mode),
            adaptive_speculative_window: mode == "draft",
            draft_n_gpu_layers: if mode == "draft" || self.speculative.native_mtp_enabled {
                self.speculative.draft_n_gpu_layers
            } else {
                None
            },
            speculative: self.speculative_decode_config(),
            native_mtp_enabled: self.speculative.native_mtp_enabled,
            native_mtp_draft_model_path: if self.speculative.native_mtp_enabled {
                self.speculative.draft_model_path.clone()
            } else {
                None
            },
            native_mtp_max_tokens: self.speculative.decode.native_mtp.max_draft_tokens,
            native_mtp_min_tokens: self.speculative.decode.native_mtp.min_draft_tokens,
            activation_width,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        })
    }

    fn ensure_single_stage_safe(&self) -> Result<()> {
        if self.hardware.stage_layer_start.is_some() || self.hardware.stage_layer_end.is_some() {
            bail!("skippy hardware.stage_layer_start/stage_layer_end are staged-only controls");
        }
        self.ensure_embedded_openai_safe(false)
    }

    fn ensure_embedded_openai_safe(&self, staged: bool) -> Result<()> {
        if !staged && self.skippy.prefill_controls_explicit {
            bail!("skippy prefill chunk controls require staged serving");
        }
        Ok(())
    }

    fn speculative_mode_for_embedded(&self, _staged: bool) -> &'static str {
        if self.speculative.mode == "draft" && self.speculative.draft_model_path.is_some() {
            "draft"
        } else if self.speculative.decode.ngram.is_some()
            && !self.speculative.decode.native_mtp.enabled
        {
            "ngram"
        } else {
            "disabled"
        }
    }

    fn speculative_window_for_embedded(&self, mode: &str) -> usize {
        match mode {
            "draft" => self.speculative.draft_max_tokens as usize,
            "ngram" => self
                .speculative
                .decode
                .ngram
                .as_ref()
                .map_or(0, |ngram| ngram.max_proposal_tokens),
            _ => 0,
        }
    }

    fn resolve_stage_kv_cache(
        &self,
        family_default: Option<StageKvCacheConfig>,
    ) -> Result<Option<StageKvCacheConfig>> {
        match &self.model_fit.prefix_cache {
            ResolvedStageKvCache::FamilyDefault => Ok(family_default),
            ResolvedStageKvCache::Disabled => Ok(Some(StageKvCacheConfig {
                mode: StageKvCacheMode::Disabled,
                payload: StageKvCachePayload::Auto,
                max_entries: 0,
                max_bytes: 0,
                min_tokens: 0,
                shared_prefix_stride_tokens: 0,
                shared_prefix_record_limit: 0,
            })),
            ResolvedStageKvCache::Explicit(template) => {
                let mut cache = family_default.unwrap_or(StageKvCacheConfig {
                    mode: template.mode.clone(),
                    payload: StageKvCachePayload::Auto,
                    max_entries: 128,
                    max_bytes: 0,
                    min_tokens: 256,
                    shared_prefix_stride_tokens: 128,
                    shared_prefix_record_limit: 2,
                });
                cache.mode = template.mode.clone();
                cache.payload = template.payload;
                if let Some(value) = template.max_entries {
                    cache.max_entries = value;
                }
                if let Some(value) = template.max_bytes {
                    cache.max_bytes = value;
                }
                if let Some(value) = template.min_tokens {
                    cache.min_tokens = value;
                }
                if let Some(value) = template.shared_prefix_stride_tokens {
                    cache.shared_prefix_stride_tokens = value;
                }
                if let Some(value) = template.shared_prefix_record_limit {
                    cache.shared_prefix_record_limit = value as u64;
                }
                Ok(Some(cache))
            }
        }
    }
}

impl ResolvedSkippyConfig {
    fn speculative_decode_config(&self) -> SpeculativeDecodeConfig {
        self.speculative.decode.clone()
    }
}

impl ResolvedEmbeddedOpenAiArgs {
    pub(crate) fn direct_single_stage_defaults(
        model_id: String,
        default_max_tokens: u32,
        generation_concurrency: usize,
        native_mtp_enabled: bool,
    ) -> Self {
        Self {
            model_id: Some(model_id),
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            continuous_batching: true,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            prefill_adaptive_target_ms: BUILTIN_PREFILL_ADAPTIVE_TARGET_MS,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            speculative: SpeculativeDecodeConfig {
                native_mtp: NativeMtpProposalConfig {
                    enabled: native_mtp_enabled,
                    max_draft_tokens: if native_mtp_enabled {
                        DEFAULT_NATIVE_MTP_MAX_TOKENS
                    } else {
                        1
                    },
                    min_draft_tokens: 0,
                    reject_cooldown_tokens: 0,
                    suppress_cooldown_drafts: false,
                    suppress_cooldown_draft_limit: 0,
                },
                effective_strategy: if native_mtp_enabled {
                    "native-mtp".to_string()
                } else {
                    "disabled".to_string()
                },
                ..SpeculativeDecodeConfig::default()
            },
            native_mtp_enabled,
            native_mtp_draft_model_path: None,
            native_mtp_max_tokens: if native_mtp_enabled {
                DEFAULT_NATIVE_MTP_MAX_TOKENS
            } else {
                0
            },
            native_mtp_min_tokens: 0,
            activation_width: 0,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
    }

    pub(crate) fn embedded_stage_defaults(
        model_id: Option<String>,
        default_max_tokens: u32,
        generation_concurrency: usize,
        activation_width: i32,
        native_mtp_enabled: bool,
    ) -> Self {
        Self {
            model_id,
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            continuous_batching: true,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            prefill_adaptive_target_ms: BUILTIN_PREFILL_ADAPTIVE_TARGET_MS,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            speculative: SpeculativeDecodeConfig {
                native_mtp: NativeMtpProposalConfig {
                    enabled: native_mtp_enabled,
                    max_draft_tokens: if native_mtp_enabled {
                        DEFAULT_NATIVE_MTP_MAX_TOKENS
                    } else {
                        1
                    },
                    min_draft_tokens: 0,
                    reject_cooldown_tokens: 0,
                    suppress_cooldown_drafts: false,
                    suppress_cooldown_draft_limit: 0,
                },
                effective_strategy: if native_mtp_enabled {
                    "native-mtp".to_string()
                } else {
                    "disabled".to_string()
                },
                ..SpeculativeDecodeConfig::default()
            },
            native_mtp_enabled,
            native_mtp_draft_model_path: None,
            native_mtp_max_tokens: if native_mtp_enabled {
                DEFAULT_NATIVE_MTP_MAX_TOKENS
            } else {
                0
            },
            native_mtp_min_tokens: 0,
            activation_width,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
    }

    pub(crate) fn build(
        self,
        bind_addr: SocketAddr,
        config: StageConfig,
        runtime: Arc<Mutex<skippy_server::runtime_state::RuntimeState>>,
        telemetry: Telemetry,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    ) -> EmbeddedOpenAiArgs {
        EmbeddedOpenAiArgs {
            bind_addr,
            config,
            runtime,
            model_id: self.model_id,
            default_max_tokens: self.default_max_tokens,
            request_defaults: self.request_defaults,
            generation_concurrency: self.generation_concurrency,
            continuous_batching: self.continuous_batching,
            adaptive_generation_min_concurrency: None,
            generation_queue_capacity: self.generation_concurrency.saturating_mul(8).clamp(16, 256),
            generation_admission_timeout_secs: DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
            prefill_chunk_size: self.prefill_chunk_size,
            prefill_chunk_policy: self.prefill_chunk_policy,
            prefill_chunk_schedule: self.prefill_chunk_schedule,
            prefill_adaptive_start: self.prefill_adaptive_start,
            prefill_adaptive_step: self.prefill_adaptive_step,
            prefill_adaptive_max: self.prefill_adaptive_max,
            prefill_adaptive_target_ms: self.prefill_adaptive_target_ms,
            draft_model_path: self.draft_model_path,
            speculative_window: self.speculative_window,
            adaptive_speculative_window: self.adaptive_speculative_window,
            draft_n_gpu_layers: self.draft_n_gpu_layers,
            speculative: self.speculative,
            native_mtp_enabled: self.native_mtp_enabled,
            native_mtp_draft_model_path: self.native_mtp_draft_model_path,
            native_mtp_max_tokens: self.native_mtp_max_tokens,
            native_mtp_min_tokens: self.native_mtp_min_tokens,
            activation_width: self.activation_width,
            reply_credit_limit: self.reply_credit_limit,
            downstream_connect_timeout_secs: self.downstream_connect_timeout_secs,
            downstream_wire_condition: skippy_server::binary_transport::WireCondition::new(
                0.0, None,
            )
            .expect("static downstream wire condition should construct"),
            prediction_returns: None,
            telemetry,
            hook_policy,
            generation_receipt: None,
            generation_lifecycle: None,
            linear_proposal_ingress: None,
            kv_lifecycle_observer: None,
            openai_guardrails: None,
        }
    }
}
