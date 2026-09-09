#![allow(dead_code)]

mod certification;
mod checkpoint;
mod deployment;
mod family_policy;
mod hash_cache;
mod hooks;
mod kv_cache;
mod local_source;
mod materialization;
pub(crate) mod metal_pipeline_cache;
mod package;
mod resolver;
mod stage;
mod topology;

use crate::runtime::{
    NativeSkippyOperationalEvent, record_native_skippy_operational_event, survey,
};
use std::{
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use openai_frontend::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, CompactingOpenAiBackend,
    CompactionConfig, CompletionRequest, CompletionResponse, CompletionStream,
    GuardedOpenAiBackend, GuardrailMode, GuardrailPolicy, GuardrailPolicyHandle,
    GuardrailTelemetrySink, ModelObject, OpenAiBackend, OpenAiHookPolicy, OpenAiRequestContext,
    OpenAiResult,
};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig, StageDevice, StageKvCacheConfig};
use skippy_runtime::{ModelInfo, MtpSource};
use skippy_server::serving_hooks::{ModelServingHooks, SharedModelServingHooksFactory};
use skippy_server::{
    DEFAULT_EMBEDDED_MAX_TOKENS, DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS, EmbeddedOpenAiArgs,
    EmbeddedRuntimeOptions, EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedState,
    OpenAiGuardrailsConfig, OpenAiGuardrailsStatus, OpenAiGuardrailsTarget, SkippyRuntimeHandle,
    binary_transport::PredictionReturnHub, binary_transport::PredictionReturnListener,
    binary_transport::WireCondition, embedded_openai_backend, runtime_state::RuntimeState,
    telemetry::Telemetry, telemetry::TelemetryLevel,
};

pub use certification::{
    CertificationGateStatus, SkippyCertificationRequest, certify_layer_package,
};
pub(crate) use family_policy::{
    family_policy_for_compact_meta, family_policy_for_model_path, family_policy_for_stage_config,
};
pub(crate) use hooks::MeshAutoHookPolicy;
pub(crate) use kv_cache::KvCachePolicy;
#[cfg(test)]
pub(crate) use local_source::local_source_required_for_model;
pub(crate) use local_source::{
    apply_verified_local_source, effective_local_source_required, into_content_addressed_identity,
    is_content_addressed_gguf_ref, register_local_source_policy, unregister_local_source_policy,
    verify_registered_content_source,
};
#[cfg(test)]
pub(crate) use materialization::resolve_package_v2_stage_to_local;
pub use materialization::{
    configure_materialized_stage_cache, is_layer_package_ref, materialized_stage_cache_dir,
    materialized_stages_for_sources, prune_unpinned_materialized_stages,
    remove_materialized_stages_for_sources, resolve_hf_package_to_local,
    resolve_package_v2_full_model_to_local, resolve_stage_load_package,
};
#[cfg(test)]
pub(crate) use package::write_test_package_v2_fixture;
pub use package::{
    SkippyPackageIdentity, identity_from_layer_package, identity_from_package_v2,
    synthetic_direct_gguf_package,
};
pub(crate) use package::{
    direct_gguf_planning_manifest_from_identity, direct_gguf_source_paths, is_package_v2_ref,
    synthetic_content_addressed_gguf_package,
};
pub(crate) use resolver::{
    ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, SkippyConfigResolveRequest,
    resolve_skippy_config_for_selector,
};
pub(crate) use skippy_server::OpenAiGuardrailsStatus as SkippyOpenAiGuardrailsStatus;
pub(crate) use stage::admitted_resident_tensor_names;
#[cfg(test)]
pub(crate) use stage::test_stage_admission;
pub(crate) use stage::{
    LayerRange, SourceModelKind, StageCancelPrepareRequest, StageControlCommand,
    StageControlHandle, StageControlRequest, StageControlResponse, StageCoordinatorClaim,
    StageCoordinatorClaimAck, StageInventoryRequest, StageLayerInventory, StageLoadRequest,
    StageLoadRuntimeSettings, StagePackagePrefetcher, StagePeerDescriptor, StagePreparationState,
    StagePreparationStatus, StagePrepareAcceptedResponse, StagePrepareRequest, StageReadyResponse,
    StageRuntimeState, StageStatusAck, StageStatusFilter, StageStatusSnapshot, StageStopRequest,
    StageTopologyStageDescriptor, spawn_stage_control_loop, stage_load_timeout,
};
#[cfg(test)]
pub(crate) use topology::{StageTopologyParticipant, plan_package_identity_topology};

const BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV: &str = "MESH_LLM_BENCH_DOWNSTREAM_WIRE_DELAY_MS";

fn benchmark_downstream_wire_condition() -> Result<WireCondition> {
    let delay_ms = match env::var(BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV) {
        Ok(value) => parse_benchmark_downstream_wire_delay_ms(&value)?,
        Err(env::VarError::NotPresent) => 0.0,
        Err(env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV} must be valid UTF-8")
        }
    };
    WireCondition::new(delay_ms, None)
}

fn parse_benchmark_downstream_wire_delay_ms(value: &str) -> Result<f64> {
    let delay_ms = value.parse::<f64>().with_context(|| {
        format!("{BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV} must be a finite non-negative number")
    })?;
    if !delay_ms.is_finite() || delay_ms < 0.0 {
        anyhow::bail!("{BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV} must be a finite non-negative number");
    }
    Ok(delay_ms)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SkippyModelState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyModelStatus {
    pub(crate) state: SkippyModelState,
    pub(crate) model_id: String,
    pub(crate) backend: &'static str,
    pub(crate) runtime_loaded: bool,
    pub(crate) package_ref: Option<String>,
    pub(crate) manifest_sha256: Option<String>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) materialized_path: Option<String>,
    pub(crate) materialized_pinned: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    /// Per-lane session state, possibly a frozen snapshot. Display only.
    pub(crate) lanes: Vec<SkippySessionLaneStatus>,
    pub(crate) max_session_tokens: u64,
    /// When [`Self::lanes`] and [`Self::max_session_tokens`] were read, which
    /// may be arbitrarily earlier than this status.
    pub(crate) sessions_captured_at_unix_nanos: i64,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) selected_device: Option<SkippyDeviceDescriptor>,
    pub(crate) openai_guardrails: Option<OpenAiGuardrailsStatus>,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) stage_id: String,
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) started_at_unix_nanos: i64,
    pub(crate) stopped_at_unix_nanos: Option<i64>,
    pub(crate) last_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SkippySessionLaneStatus {
    pub(crate) index: usize,
    pub(crate) active: bool,
    pub(crate) session_id: Option<String>,
    pub(crate) token_count: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SkippyDeviceDescriptor {
    pub(crate) backend_device: String,
    pub(crate) stable_id: Option<String>,
    pub(crate) index: Option<usize>,
    pub(crate) vram_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyModelLoadOptions {
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) ctx_size: u32,
    pub(crate) n_gpu_layers: i32,
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
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_threads: Option<usize>,
    pub(crate) n_threads_batch: Option<usize>,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) kv_offload: Option<bool>,
    pub(crate) kv_unified: Option<bool>,
    pub(crate) swa_full: Option<bool>,
    pub(crate) cache_idle_slots: Option<u32>,
    pub(crate) generation_concurrency: usize,
    pub(crate) default_max_tokens: u32,
    pub(crate) kv_cache: Option<StageKvCacheConfig>,
    pub(crate) embedded_openai: Option<resolver::ResolvedEmbeddedOpenAiArgs>,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: Option<u32>,
    pub(crate) selected_device: Option<SkippyDeviceDescriptor>,
    pub(crate) package_identity: Option<SkippyPackageIdentity>,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) projector_use_gpu: Option<bool>,
    pub(crate) media_marker: Option<String>,
    pub(crate) image_min_tokens: Option<u32>,
    pub(crate) image_max_tokens: Option<u32>,
    pub(crate) batch_max_tokens: Option<u32>,
    pub(crate) glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub(crate) generation_signal_window: Option<u32>,
    pub(crate) telemetry: SkippyTelemetryOptions,
    pub(crate) openai_guardrails: Option<OpenAiGuardrailsConfig>,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) serving_hooks_factory: Option<SharedModelServingHooksFactory>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyTelemetryOptions {
    pub(crate) metrics_otlp_grpc: Option<String>,
    pub(crate) queue_capacity: usize,
    pub(crate) level: TelemetryLevel,
}

impl Default for SkippyTelemetryOptions {
    fn default() -> Self {
        Self::off()
    }
}

impl SkippyTelemetryOptions {
    pub(crate) fn off() -> Self {
        Self {
            metrics_otlp_grpc: None,
            queue_capacity: 0,
            level: TelemetryLevel::Off,
        }
    }

    pub(crate) fn debug(metrics_otlp_grpc: Option<String>) -> Self {
        Self {
            metrics_otlp_grpc,
            queue_capacity: 1024,
            level: TelemetryLevel::Debug,
        }
    }

    pub(crate) fn summary(metrics_otlp_grpc: String) -> Self {
        Self {
            metrics_otlp_grpc: Some(metrics_otlp_grpc),
            queue_capacity: 1024,
            level: TelemetryLevel::Summary,
        }
    }
}

pub(crate) fn default_skippy_openai_guardrails() -> OpenAiGuardrailsConfig {
    skippy_openai_guardrails_for_mode(GuardrailMode::Disabled)
}

pub(crate) fn skippy_openai_guardrails_for_mode(mode: GuardrailMode) -> OpenAiGuardrailsConfig {
    // v1 only wraps hosted Skippy OpenAI backends constructed at the local/staged
    // seams below. MoA `model:"mesh"` arbitration and Virtual LLM consult paths
    // stay unwrapped until they adopt the backend-free guardrail core directly.
    let policy = GuardrailPolicy {
        mode,
        ..GuardrailPolicy::default()
    };
    skippy_openai_guardrails_for_policy_handle(GuardrailPolicyHandle::new(policy))
}

pub(crate) fn skippy_openai_guardrails_for_policy_handle(
    policy: GuardrailPolicyHandle,
) -> OpenAiGuardrailsConfig {
    OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy,
        compaction: Some(CompactionConfig {
            enabled: true,
            ..CompactionConfig::default()
        }),
    }
}

impl SkippyModelLoadOptions {
    pub(crate) fn for_direct_gguf(
        model_id: impl Into<String>,
        model_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_path: model_path.into(),
            ctx_size: 4096,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            generation_concurrency: 1,
            default_max_tokens: DEFAULT_EMBEDDED_MAX_TOKENS,
            kv_cache: None,
            embedded_openai: None,
            layer_start: 0,
            layer_end: None,
            selected_device: None,
            package_identity: None,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            telemetry: SkippyTelemetryOptions::off(),
            openai_guardrails: Some(OpenAiGuardrailsConfig::disabled_for_skippy()),
            native_mtp_enabled: true,
            serving_hooks_factory: None,
        }
    }

    pub(crate) fn with_ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub(crate) fn with_generation_concurrency(mut self, generation_concurrency: usize) -> Self {
        self.generation_concurrency = generation_concurrency;
        self
    }

    pub(crate) fn with_cache_types(mut self, cache_type_k: &str, cache_type_v: &str) -> Self {
        self.cache_type_k = cache_type_k.to_string();
        self.cache_type_v = cache_type_v.to_string();
        self
    }

    pub(crate) fn with_batch_sizes(mut self, n_batch: Option<u32>, n_ubatch: Option<u32>) -> Self {
        self.n_batch = n_batch;
        self.n_ubatch = n_ubatch;
        self
    }

    pub(crate) fn with_thread_counts(
        mut self,
        n_threads: Option<usize>,
        n_threads_batch: Option<usize>,
    ) -> Self {
        self.n_threads = n_threads;
        self.n_threads_batch = n_threads_batch;
        self
    }

    pub(crate) fn with_flash_attn_type(mut self, flash_attn_type: FlashAttentionType) -> Self {
        self.flash_attn_type = flash_attn_type;
        self
    }

    pub(crate) fn with_kv_session_controls(
        mut self,
        kv_offload: Option<bool>,
        kv_unified: Option<bool>,
        swa_full: Option<bool>,
    ) -> Self {
        self.kv_offload = kv_offload;
        self.kv_unified = kv_unified;
        self.swa_full = swa_full;
        self
    }

    pub(crate) fn with_cache_idle_slots(mut self, cache_idle_slots: Option<u32>) -> Self {
        self.cache_idle_slots = cache_idle_slots;
        self
    }

    pub(crate) fn with_layer_end(mut self, layer_end: u32) -> Self {
        self.layer_end = Some(layer_end);
        self
    }

    pub(crate) fn with_layer_range(mut self, layer_start: u32, layer_end: u32) -> Self {
        self.layer_start = layer_start;
        self.layer_end = Some(layer_end);
        self
    }

    pub(crate) fn with_selected_device(mut self, selected_device: SkippyDeviceDescriptor) -> Self {
        self.selected_device = Some(selected_device);
        self
    }

    pub(crate) fn with_projector_path(mut self, projector_path: impl Into<PathBuf>) -> Self {
        self.projector_path = Some(projector_path.into());
        self
    }

    pub(crate) fn with_telemetry(mut self, telemetry: SkippyTelemetryOptions) -> Self {
        self.telemetry = telemetry;
        self
    }

    pub(crate) fn with_kv_cache(mut self, kv_cache: Option<StageKvCacheConfig>) -> Self {
        self.kv_cache = kv_cache;
        self
    }

    pub(crate) fn with_embedded_openai(
        mut self,
        embedded_openai: resolver::ResolvedEmbeddedOpenAiArgs,
    ) -> Self {
        self.embedded_openai = Some(embedded_openai);
        self
    }

    pub(crate) fn with_openai_guardrails(
        mut self,
        openai_guardrails: OpenAiGuardrailsConfig,
    ) -> Self {
        self.openai_guardrails = Some(openai_guardrails);
        self
    }

    pub(crate) fn with_serving_hooks_factory(
        mut self,
        factory: Option<SharedModelServingHooksFactory>,
    ) -> Self {
        self.serving_hooks_factory = factory;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_package_identity(mut self, package_identity: SkippyPackageIdentity) -> Self {
        self.package_identity = Some(package_identity);
        self
    }
}

#[derive(Debug)]
struct HandleState {
    state: SkippyModelState,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

pub(crate) struct SkippyModelHandle {
    runtime: SkippyRuntimeHandle,
    backend: Arc<dyn OpenAiBackend>,
    openai_guardrails: Option<OpenAiGuardrailsConfig>,
    config: StageConfig,
    started_at_unix_nanos: i64,
    status: Arc<Mutex<HandleState>>,
    _prediction_return_listener: Option<PredictionReturnListener>,
}

pub(crate) struct SkippyHttpHandle {
    port: u16,
    server: EmbeddedServerHandle,
}

pub(crate) struct SkippyOpenAiGuardrailOptions {
    config: Option<OpenAiGuardrailsConfig>,
    telemetry: survey::SurveyTelemetry,
}

pub(crate) type NativeModelOpenEventReporter = Box<dyn FnMut(skippy_runtime::RuntimeEvent) + Send>;

impl SkippyOpenAiGuardrailOptions {
    pub(crate) fn new(
        config: Option<OpenAiGuardrailsConfig>,
        telemetry: survey::SurveyTelemetry,
    ) -> Self {
        Self { config, telemetry }
    }
}

impl SkippyHttpHandle {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn status(&self) -> skippy_server::EmbeddedServerStatus {
        self.server.status()
    }

    pub(crate) async fn shutdown(self) -> Result<()> {
        self.server.shutdown().await
    }
}

/// Builds `EmbeddedOpenAiArgs`, filling most fields from `embedded_args` and
/// taking only the handful that differ per load path as parameters.
fn embedded_openai_args_from(
    embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
    config: StageConfig,
    runtime: Arc<Mutex<RuntimeState>>,
    prediction_returns: Option<Arc<PredictionReturnHub>>,
    telemetry: Telemetry,
    hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    serving_hooks: &ModelServingHooks,
) -> Result<EmbeddedOpenAiArgs> {
    let activation_width = if config.downstream.is_some() {
        let descriptor = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("runtime lock poisoned"))?
            .output_activation_boundary()
            .context("stage 0 graph did not expose its output activation boundary")?;
        descriptor.raw_f32_width("output")?
    } else {
        embedded_args.activation_width
    };
    Ok(EmbeddedOpenAiArgs {
        bind_addr: "127.0.0.1:0"
            .parse()
            .expect("static bind address should parse"),
        config,
        runtime,
        model_id: embedded_args.model_id,
        default_max_tokens: embedded_args.default_max_tokens,
        request_defaults: embedded_args.request_defaults,
        generation_concurrency: embedded_args.generation_concurrency,
        continuous_batching: embedded_args.continuous_batching,
        adaptive_generation_min_concurrency: None,
        generation_queue_capacity: embedded_args
            .generation_concurrency
            .saturating_mul(8)
            .clamp(16, 256),
        generation_admission_timeout_secs: DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
        prefill_chunk_size: embedded_args.prefill_chunk_size,
        prefill_chunk_policy: embedded_args.prefill_chunk_policy,
        prefill_chunk_schedule: embedded_args.prefill_chunk_schedule,
        prefill_adaptive_start: embedded_args.prefill_adaptive_start,
        prefill_adaptive_step: embedded_args.prefill_adaptive_step,
        prefill_adaptive_max: embedded_args.prefill_adaptive_max,
        prefill_adaptive_target_ms: embedded_args.prefill_adaptive_target_ms,
        draft_model_path: embedded_args.draft_model_path,
        speculative_window: embedded_args.speculative_window,
        adaptive_speculative_window: embedded_args.adaptive_speculative_window,
        draft_n_gpu_layers: embedded_args.draft_n_gpu_layers,
        speculative: embedded_args.speculative,
        native_mtp_enabled: embedded_args.native_mtp_enabled,
        native_mtp_draft_model_path: embedded_args.native_mtp_draft_model_path,
        native_mtp_max_tokens: embedded_args.native_mtp_max_tokens,
        native_mtp_min_tokens: embedded_args.native_mtp_min_tokens,
        activation_width,
        reply_credit_limit: embedded_args.reply_credit_limit,
        downstream_connect_timeout_secs: embedded_args.downstream_connect_timeout_secs,
        downstream_wire_condition: benchmark_downstream_wire_condition()?,
        prediction_returns,
        telemetry,
        hook_policy,
        generation_receipt: serving_hooks.generation_receipt(),
        linear_proposal_ingress: serving_hooks.linear_proposal_ingress(),
        openai_guardrails: None,
    })
}

fn resolve_serving_hooks(
    factory: Option<&SharedModelServingHooksFactory>,
    runtime: &SkippyRuntimeHandle,
) -> Result<ModelServingHooks> {
    let Some(factory) = factory else {
        return Ok(ModelServingHooks::default());
    };
    let tokenizer = runtime
        .tokenizer_capability()
        .context("loaded Skippy runtime cannot provide its tokenizer capability")?;
    factory
        .create(tokenizer)
        .context("product-neutral serving hook factory rejected the loaded model")
}

struct NativeSkippyStartupAudit {
    ready: bool,
}

impl NativeSkippyStartupAudit {
    fn new() -> Self {
        record_native_skippy_operational_event(NativeSkippyOperationalEvent::RuntimeStartupStarted);
        Self { ready: false }
    }

    fn mark_ready(&mut self) {
        self.ready = true;
        record_native_skippy_operational_event(NativeSkippyOperationalEvent::RuntimeReady);
    }
}

impl Drop for NativeSkippyStartupAudit {
    fn drop(&mut self) {
        if !self.ready {
            record_native_skippy_operational_event(
                NativeSkippyOperationalEvent::RuntimeStartupFailed,
            );
        }
    }
}

impl SkippyModelHandle {
    pub(crate) fn output_activation_boundary(
        &self,
    ) -> Option<skippy_runtime::ActivationBoundaryDesc> {
        self.runtime.output_activation_boundary()
    }

    fn resolved_mtp_source(
        native_mtp_enabled: bool,
        native_mtp_draft_model_path: Option<&Path>,
    ) -> MtpSource {
        if !native_mtp_enabled {
            MtpSource::Disabled
        } else if native_mtp_draft_model_path.is_some() {
            MtpSource::External
        } else {
            MtpSource::Integrated
        }
    }

    pub(crate) fn load(options: SkippyModelLoadOptions) -> Result<Self> {
        Self::load_with_hooks(options, None, survey::SurveyTelemetry::disabled())
    }

    pub(crate) fn load_with_hooks(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        guardrail_telemetry: survey::SurveyTelemetry,
    ) -> Result<Self> {
        let mut lifecycle_audit = NativeSkippyStartupAudit::new();
        let stage_config = single_stage_config(&options)?;
        let mtp_source = Self::resolved_mtp_source(
            options.native_mtp_enabled,
            options
                .embedded_openai
                .as_ref()
                .and_then(|args| args.native_mtp_draft_model_path.as_deref()),
        );
        let runtime = SkippyRuntimeHandle::load(EmbeddedRuntimeOptions {
            config: stage_config.clone(),
            topology: None,
            n_threads: options.n_threads,
            n_threads_batch: options.n_threads_batch,
            mtp_source,
            metrics_otlp_grpc: options.telemetry.metrics_otlp_grpc.clone(),
            telemetry_queue_capacity: options.telemetry.queue_capacity,
            telemetry_level: options.telemetry.level,
        })
        .with_context(|| {
            format!(
                "load skippy runtime for model {} from {}",
                options.model_id,
                options.model_path.display()
            )
        })?;
        let telemetry = Telemetry::new(
            options.telemetry.metrics_otlp_grpc.clone(),
            options.telemetry.queue_capacity,
            stage_config.clone(),
            options.telemetry.level,
        );
        let serving_hooks =
            resolve_serving_hooks(options.serving_hooks_factory.as_ref(), &runtime)?;
        let embedded_args = options.embedded_openai.clone().unwrap_or_else(|| {
            resolver::ResolvedEmbeddedOpenAiArgs::direct_single_stage_defaults(
                options.model_id.clone(),
                options.default_max_tokens,
                options.generation_concurrency,
                options.native_mtp_enabled,
            )
        });
        let openai_guardrails = options.openai_guardrails.clone();
        let binding = embedded_openai_backend(embedded_openai_args_from(
            embedded_args,
            stage_config.clone(),
            runtime.runtime(),
            None,
            telemetry,
            hook_policy,
            &serving_hooks,
        )?)
        .context("construct skippy OpenAI backend")?;
        let backend = wrap_host_guardrail_backend(
            binding.backend,
            openai_guardrails.as_ref(),
            Some(usize::try_from(stage_config.ctx_size).unwrap_or(usize::MAX)),
            guardrail_telemetry.guardrail_sink(),
        );
        lifecycle_audit.mark_ready();
        Ok(Self {
            runtime,
            backend,
            openai_guardrails,
            config: stage_config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _prediction_return_listener: None,
        })
    }

    pub(crate) fn load_with_hooks_and_open_events(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        model_open_event_reporter: Option<NativeModelOpenEventReporter>,
        guardrail_telemetry: survey::SurveyTelemetry,
    ) -> Result<Self> {
        let mut lifecycle_audit = NativeSkippyStartupAudit::new();
        let stage_config = single_stage_config(&options)?;
        let mtp_source = Self::resolved_mtp_source(
            options.native_mtp_enabled,
            options
                .embedded_openai
                .as_ref()
                .and_then(|args| args.native_mtp_draft_model_path.as_deref()),
        );
        let runtime = SkippyRuntimeHandle::load_with_open_events(
            EmbeddedRuntimeOptions {
                config: stage_config.clone(),
                topology: None,
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
                mtp_source,
                metrics_otlp_grpc: options.telemetry.metrics_otlp_grpc.clone(),
                telemetry_queue_capacity: options.telemetry.queue_capacity,
                telemetry_level: options.telemetry.level,
            },
            model_open_event_reporter,
        )
        .with_context(|| {
            format!(
                "load skippy runtime for model {} from {}",
                options.model_id,
                options.model_path.display()
            )
        })?;
        let telemetry = Telemetry::new(
            options.telemetry.metrics_otlp_grpc.clone(),
            options.telemetry.queue_capacity,
            stage_config.clone(),
            options.telemetry.level,
        );
        let serving_hooks =
            resolve_serving_hooks(options.serving_hooks_factory.as_ref(), &runtime)?;
        let embedded_args = options.embedded_openai.clone().unwrap_or_else(|| {
            resolver::ResolvedEmbeddedOpenAiArgs::direct_single_stage_defaults(
                options.model_id.clone(),
                options.default_max_tokens,
                options.generation_concurrency,
                options.native_mtp_enabled,
            )
        });
        let openai_guardrails = options.openai_guardrails.clone();
        let binding = embedded_openai_backend(embedded_openai_args_from(
            embedded_args,
            stage_config.clone(),
            runtime.runtime(),
            None,
            telemetry,
            hook_policy,
            &serving_hooks,
        )?)
        .context("construct skippy OpenAI backend")?;
        let backend = wrap_host_guardrail_backend(
            binding.backend,
            openai_guardrails.as_ref(),
            Some(usize::try_from(stage_config.ctx_size).unwrap_or(usize::MAX)),
            guardrail_telemetry.guardrail_sink(),
        );
        lifecycle_audit.mark_ready();
        Ok(Self {
            runtime,
            backend,
            openai_guardrails,
            config: stage_config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _prediction_return_listener: None,
        })
    }

    pub(crate) fn load_stage0_config(
        config: StageConfig,
        activation_width: i32,
        generation_concurrency: usize,
        default_max_tokens: u32,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
    ) -> Result<Self> {
        let model_id = config.model_id.clone();
        let native_mtp_enabled = config.native_mtp_enabled;
        Self::load_stage0_config_with_openai_args(
            config,
            resolver::ResolvedEmbeddedOpenAiArgs::embedded_stage_defaults(
                Some(model_id),
                default_max_tokens,
                generation_concurrency,
                activation_width,
                native_mtp_enabled,
            ),
            hook_policy,
            telemetry,
            guardrails,
        )
    }

    pub(crate) fn load_stage0_config_with_openai_args(
        config: StageConfig,
        embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
    ) -> Result<Self> {
        let mtp_source = Self::resolved_mtp_source(
            config.native_mtp_enabled,
            embedded_args.native_mtp_draft_model_path.as_deref(),
        );
        Self::load_stage0_runtime_options_with_openai_args(
            EmbeddedRuntimeOptions {
                config,
                topology: None,
                n_threads: None,
                n_threads_batch: None,
                mtp_source,
                metrics_otlp_grpc: telemetry.metrics_otlp_grpc.clone(),
                telemetry_queue_capacity: telemetry.queue_capacity,
                telemetry_level: telemetry.level,
            },
            embedded_args,
            hook_policy,
            telemetry,
            guardrails,
            None,
        )
    }

    pub(crate) fn load_stage0_runtime_options_with_openai_args(
        mut runtime_options: EmbeddedRuntimeOptions,
        mut embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    ) -> Result<Self> {
        let mut lifecycle_audit = NativeSkippyStartupAudit::new();
        configure_materialized_stage_cache();
        let config = &mut runtime_options.config;
        anyhow::ensure!(
            config.load_mode != LoadMode::LayerPackage,
            "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
        );
        if config.kv_cache.is_none() {
            let family_policy = family_policy_for_stage_config(config);
            config.kv_cache = family_policy.stage_kv_cache_config_for_stage(config);
        }
        let runtime_config = config.clone();
        let runtime = SkippyRuntimeHandle::load(runtime_options).with_context(|| {
            format!(
                "load skippy stage 0 runtime for model {} from {:?}",
                runtime_config.model_id, runtime_config.model_path
            )
        })?;
        embedded_args.activation_width = if runtime_config.downstream.is_some() {
            runtime
                .output_activation_boundary()
                .context("stage 0 graph did not expose its output activation boundary")?
                .raw_f32_width("output")?
        } else {
            0
        };
        let telemetry = Telemetry::new(
            telemetry.metrics_otlp_grpc.clone(),
            telemetry.queue_capacity,
            runtime_config.clone(),
            telemetry.level,
        );
        let serving_hooks = resolve_serving_hooks(serving_hooks_factory.as_ref(), &runtime)?;
        let prediction_return_listener = if runtime_config.downstream.is_some() {
            Some(PredictionReturnListener::start(
                runtime_config.bind_addr.parse()?,
            )?)
        } else {
            None
        };
        let prediction_returns = prediction_return_listener
            .as_ref()
            .map(PredictionReturnListener::hub);
        let binding = embedded_openai_backend(embedded_openai_args_from(
            embedded_args,
            runtime_config.clone(),
            runtime.runtime(),
            prediction_returns,
            telemetry,
            hook_policy,
            &serving_hooks,
        )?)
        .context("construct skippy stage 0 OpenAI backend")?;
        let backend = wrap_host_guardrail_backend(
            binding.backend,
            guardrails.config.as_ref(),
            Some(usize::try_from(runtime_config.ctx_size).unwrap_or(usize::MAX)),
            guardrails.telemetry.guardrail_sink(),
        );
        lifecycle_audit.mark_ready();
        Ok(Self {
            runtime,
            backend,
            openai_guardrails: guardrails.config,
            config: runtime_config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _prediction_return_listener: prediction_return_listener,
        })
    }

    pub(crate) fn load_stage0_runtime_options_with_openai_args_and_open_events(
        mut runtime_options: EmbeddedRuntimeOptions,
        mut embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        model_open_event_reporter: Option<NativeModelOpenEventReporter>,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    ) -> Result<Self> {
        let mut lifecycle_audit = NativeSkippyStartupAudit::new();
        configure_materialized_stage_cache();
        let config = &mut runtime_options.config;
        anyhow::ensure!(
            config.load_mode != LoadMode::LayerPackage,
            "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
        );
        if config.kv_cache.is_none() {
            let family_policy = family_policy_for_stage_config(config);
            config.kv_cache = family_policy.stage_kv_cache_config_for_stage(config);
        }
        let runtime_config = config.clone();
        let runtime =
            SkippyRuntimeHandle::load_with_open_events(runtime_options, model_open_event_reporter)
                .with_context(|| {
                    format!(
                        "load skippy stage 0 runtime for model {} from {:?}",
                        runtime_config.model_id, runtime_config.model_path
                    )
                })?;
        embedded_args.activation_width = if runtime_config.downstream.is_some() {
            runtime
                .output_activation_boundary()
                .context("stage 0 graph did not expose its output activation boundary")?
                .raw_f32_width("output")?
        } else {
            0
        };
        let telemetry = Telemetry::new(
            telemetry.metrics_otlp_grpc.clone(),
            telemetry.queue_capacity,
            runtime_config.clone(),
            telemetry.level,
        );
        let serving_hooks = resolve_serving_hooks(serving_hooks_factory.as_ref(), &runtime)?;
        let prediction_return_listener = if runtime_config.downstream.is_some() {
            Some(PredictionReturnListener::start(
                runtime_config.bind_addr.parse()?,
            )?)
        } else {
            None
        };
        let prediction_returns = prediction_return_listener
            .as_ref()
            .map(PredictionReturnListener::hub);
        let binding = embedded_openai_backend(embedded_openai_args_from(
            embedded_args,
            runtime_config.clone(),
            runtime.runtime(),
            prediction_returns,
            telemetry,
            hook_policy,
            &serving_hooks,
        )?)
        .context("construct skippy stage 0 OpenAI backend")?;
        let backend = wrap_host_guardrail_backend(
            binding.backend,
            guardrails.config.as_ref(),
            Some(usize::try_from(runtime_config.ctx_size).unwrap_or(usize::MAX)),
            guardrails.telemetry.guardrail_sink(),
        );
        lifecycle_audit.mark_ready();
        Ok(Self {
            runtime,
            backend,
            openai_guardrails: guardrails.config,
            config: runtime_config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _prediction_return_listener: prediction_return_listener,
        })
    }

    pub(crate) fn backend(&self) -> Arc<dyn OpenAiBackend> {
        self.backend.clone()
    }

    pub(crate) fn openai_guardrails(&self) -> Option<OpenAiGuardrailsStatus> {
        self.openai_guardrails
            .as_ref()
            .map(OpenAiGuardrailsConfig::status)
    }

    pub(crate) fn set_openai_guardrail_mode(
        &self,
        mode: GuardrailMode,
    ) -> Option<OpenAiGuardrailsStatus> {
        let guardrails = self.openai_guardrails.as_ref()?;
        guardrails.policy.set_mode(mode);
        Some(guardrails.status())
    }

    pub(crate) fn start_http(&self, port: u16) -> Result<SkippyHttpHandle> {
        self.start_http_on(([127, 0, 0, 1], port).into())
    }

    pub(crate) fn start_http_on(
        &self,
        bind_addr: std::net::SocketAddr,
    ) -> Result<SkippyHttpHandle> {
        let port = bind_addr.port();
        let tokenizer = self
            .runtime
            .tokenizer_capability()
            .context("loaded Skippy runtime cannot provide its stage-0 tokenizer capability")?;
        let lifecycle_observer =
            crate::logging_runtime_state().and_then(|state| state.openai_lifecycle_observer());
        let server = skippy_server::start_openai_backend_with_tokenizer_and_lifecycle_observer(
            bind_addr,
            self.backend(),
            tokenizer,
            lifecycle_observer,
        );
        Ok(SkippyHttpHandle { port, server })
    }

    pub(crate) fn status(&self) -> SkippyModelStatus {
        let embedded = self.runtime.status();
        let local = self.status.lock().expect("skippy status lock poisoned");
        status_from_parts(
            &self.config,
            &embedded,
            &local,
            self.started_at_unix_nanos,
            self.openai_guardrails(),
        )
    }

    pub(crate) fn shutdown(&self) {
        {
            let mut state = self.status.lock().expect("skippy status lock poisoned");
            if matches!(state.state, SkippyModelState::Stopped) {
                return;
            }
            state.state = SkippyModelState::Stopping;
        }
        record_native_skippy_operational_event(
            NativeSkippyOperationalEvent::RuntimeShutdownStarted,
        );
        self.runtime.shutdown();
        let mut state = self.status.lock().expect("skippy status lock poisoned");
        state.state = SkippyModelState::Stopped;
        state.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for SkippyModelHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn wrap_host_guardrail_backend(
    backend: Arc<dyn OpenAiBackend>,
    openai_guardrails: Option<&OpenAiGuardrailsConfig>,
    context_limit_tokens: Option<usize>,
    telemetry: Option<Arc<dyn GuardrailTelemetrySink>>,
) -> Arc<dyn OpenAiBackend> {
    let Some(openai_guardrails) = openai_guardrails else {
        return backend;
    };
    if !matches!(openai_guardrails.target, OpenAiGuardrailsTarget::Skippy) {
        return backend;
    }

    let backend = match openai_guardrails.compaction {
        Some(mut compaction) => {
            if compaction.context_limit_tokens.is_none() {
                compaction.context_limit_tokens = context_limit_tokens;
            }
            Arc::new(CompactingOpenAiBackend::new(backend, compaction))
        }
        None => backend,
    };
    let guarded =
        GuardedOpenAiBackend::with_policy_handle(backend, openai_guardrails.policy.clone());
    match telemetry {
        Some(telemetry) => Arc::new(guarded.with_telemetry(telemetry)),
        None => Arc::new(guarded),
    }
}

#[async_trait]
impl OpenAiBackend for SkippyModelHandle {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.backend.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }
}

pub(crate) fn single_stage_config(options: &SkippyModelLoadOptions) -> Result<StageConfig> {
    anyhow::ensure!(
        options.ctx_size > 0,
        "skippy ctx_size must be greater than zero"
    );
    anyhow::ensure!(
        options.generation_concurrency > 0,
        "skippy generation_concurrency must be greater than zero"
    );
    if let Some(device) = options.selected_device.as_ref() {
        anyhow::ensure!(
            !device.backend_device.is_empty(),
            "skippy selected backend device must not be empty"
        );
    }
    let package_identity = match options.package_identity.as_ref() {
        Some(identity) => identity.clone(),
        None => synthetic_direct_gguf_package(&options.model_id, &options.model_path)?,
    };
    let layer_start = options.layer_start;
    let layer_end = options.layer_end.unwrap_or(package_identity.layer_count);
    anyhow::ensure!(
        layer_end > 0,
        "skippy stage layer_end must be greater than zero"
    );
    anyhow::ensure!(
        layer_start < layer_end,
        "skippy stage layer range must satisfy layer_start < layer_end"
    );
    let run_id = format!("mesh-skippy-{}", now_unix_nanos());
    let family_policy = family_policy_for_model_path(&options.model_path);
    let checkpoint = checkpoint::prepare(options)?;
    let mut config = StageConfig {
        run_id: run_id.clone(),
        topology_id: format!("topology-{run_id}"),
        model_id: options.model_id.clone(),
        package_ref: Some(package_identity.package_ref),
        manifest_sha256: Some(package_identity.manifest_sha256),
        source_model_path: Some(
            package_identity
                .source_model_path
                .to_string_lossy()
                .to_string(),
        ),
        source_model_sha256: Some(package_identity.source_model_sha256),
        source_model_bytes: Some(package_identity.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(options.model_path.to_string_lossy().to_string()),
        model_part_paths: Vec::new(),
        projector_path: options
            .projector_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        projector_use_gpu: options.projector_use_gpu,
        media_marker: options.media_marker.clone(),
        image_min_tokens: options.image_min_tokens,
        image_max_tokens: options.image_max_tokens,
        batch_max_tokens: options.batch_max_tokens,
        glm_dsa_policy: options.glm_dsa_policy,
        generation_signal_window: options.generation_signal_window,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start,
        layer_end,
        ctx_size: options.ctx_size,
        lane_count: options.generation_concurrency as u32,
        n_batch: options.n_batch,
        n_ubatch: options.n_ubatch,
        n_gpu_layers: options.n_gpu_layers,
        mmap: options.mmap,
        mlock: options.mlock,
        repack: options.repack,
        op_offload: options.op_offload,
        no_host_buffer: options.no_host_buffer,
        check_tensors: options.check_tensors,
        direct_io: options.direct_io,
        main_gpu: options.main_gpu,
        split_mode: options.split_mode,
        cache_type_k: options.cache_type_k.clone(),
        cache_type_v: options.cache_type_v.clone(),
        flash_attn_type: options.flash_attn_type,
        kv_offload: options.kv_offload,
        kv_unified: options.kv_unified,
        swa_full: options.swa_full,
        cache_idle_slots: options.cache_idle_slots,
        filter_tensors_on_load: false,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: options
            .checkpoint_quantization
            .as_ref()
            .map(|_| checkpoint.quantization.canonical_name().to_string()),
        checkpoint_imatrix: checkpoint.imatrix,
        checkpoint_imatrix_sha256: checkpoint.imatrix_sha256,
        selected_device: options.selected_device.clone().map(Into::into),
        kv_cache: None,
        native_mtp_enabled: options.native_mtp_enabled,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    };
    config.kv_cache = options
        .kv_cache
        .clone()
        .or_else(|| family_policy.stage_kv_cache_config_for_stage(&config));
    checkpoint::emit_load_notice(
        &options.model_path,
        checkpoint.quantization,
        config.checkpoint_imatrix.is_some(),
    );
    Ok(config)
}

impl From<SkippyDeviceDescriptor> for StageDevice {
    fn from(device: SkippyDeviceDescriptor) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

impl From<StageDevice> for SkippyDeviceDescriptor {
    fn from(device: StageDevice) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

pub(crate) fn infer_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model metadata {}", path.display()))?;
    let layer_count = info
        .tensors()
        .with_context(|| format!("read model tensors {}", path.display()))?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .with_context(|| format!("infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

fn status_from_parts(
    config: &StageConfig,
    embedded: &EmbeddedRuntimeStatus,
    local: &HandleState,
    started_at_unix_nanos: i64,
    openai_guardrails: Option<OpenAiGuardrailsStatus>,
) -> SkippyModelStatus {
    SkippyModelStatus {
        state: match local.state {
            SkippyModelState::Starting => SkippyModelState::Starting,
            SkippyModelState::Ready => map_embedded_state(embedded.state),
            SkippyModelState::Stopping => SkippyModelState::Stopping,
            SkippyModelState::Stopped => SkippyModelState::Stopped,
            SkippyModelState::Failed => SkippyModelState::Failed,
        },
        model_id: config.model_id.clone(),
        backend: "skippy",
        runtime_loaded: embedded.runtime_loaded,
        package_ref: config.package_ref.clone(),
        manifest_sha256: config.manifest_sha256.clone(),
        source_model_path: config.source_model_path.clone(),
        source_model_sha256: config.source_model_sha256.clone(),
        source_model_bytes: config.source_model_bytes,
        materialized_path: config.materialized_path.clone(),
        materialized_pinned: config.materialized_pinned,
        projector_path: config.projector_path.clone(),
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        lanes: embedded
            .sessions
            .lanes
            .iter()
            .map(|lane| SkippySessionLaneStatus {
                index: lane.index,
                active: lane.active,
                session_id: lane.session_id.clone(),
                token_count: lane.token_count,
            })
            .collect(),
        max_session_tokens: embedded.sessions.max_session_tokens,
        sessions_captured_at_unix_nanos: embedded.sessions_captured_at_unix_nanos,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_gpu_layers: config.n_gpu_layers,
        flash_attn_type: config.flash_attn_type,
        selected_device: config.selected_device.clone().map(Into::into),
        openai_guardrails,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        stage_id: config.stage_id.clone(),
        topology_id: config.topology_id.clone(),
        run_id: config.run_id.clone(),
        started_at_unix_nanos,
        stopped_at_unix_nanos: local
            .stopped_at_unix_nanos
            .or(embedded.stopped_at_unix_nanos),
        last_error: local
            .last_error
            .clone()
            .or_else(|| embedded.last_error.clone()),
    }
}

fn map_embedded_state(state: EmbeddedState) -> SkippyModelState {
    match state {
        EmbeddedState::Starting => SkippyModelState::Starting,
        EmbeddedState::Ready => SkippyModelState::Ready,
        EmbeddedState::Stopping => SkippyModelState::Stopping,
        EmbeddedState::Stopped => SkippyModelState::Stopped,
        EmbeddedState::Failed => SkippyModelState::Failed,
    }
}

fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_frontend::{MESH_COMPACT_FIELD, OpenAiError};
    use serde_json::json;
    use skippy_server::runtime_state::RuntimeSessionStats;
    use skippy_server::telemetry::TelemetryStats;

    #[test]
    fn benchmark_wire_delay_accepts_finite_non_negative_values() {
        assert_eq!(parse_benchmark_downstream_wire_delay_ms("0").unwrap(), 0.0);
        assert_eq!(
            parse_benchmark_downstream_wire_delay_ms("25.5").unwrap(),
            25.5
        );
    }

    #[test]
    fn benchmark_wire_delay_rejects_invalid_values() {
        for value in ["-1", "NaN", "inf", "not-a-number"] {
            assert!(parse_benchmark_downstream_wire_delay_ms(value).is_err());
        }
    }

    #[derive(Default)]
    struct RecordingHostBackend {
        seen_chat: Mutex<Option<ChatCompletionRequest>>,
    }

    #[async_trait]
    impl OpenAiBackend for RecordingHostBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("host-skippy")])
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            *self.seen_chat.lock().expect("seen chat lock poisoned") = Some(request.clone());
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                openai_frontend::Usage::new(0, 0),
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            Err(OpenAiError::unsupported(
                "streaming is not needed by this host wrapper test",
            ))
        }
    }

    fn fake_package_identity(layer_count: u32) -> SkippyPackageIdentity {
        SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_string(),
            source_model_bytes: 1234,
            source_files: Vec::new(),
            layer_weight_bytes: Vec::new(),
            layer_count,
            activation_width: 4096,
            tensor_count: 100,
            generation: None,
        }
    }

    fn fake_stage_config() -> StageConfig {
        single_stage_config(
            &SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36)),
        )
        .expect("fake stage config")
    }

    fn fake_embedded_runtime_status(config: &StageConfig) -> EmbeddedRuntimeStatus {
        EmbeddedRuntimeStatus {
            state: EmbeddedState::Ready,
            run_id: config.run_id.clone(),
            topology_id: config.topology_id.clone(),
            model_id: config.model_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            runtime_loaded: true,
            started_at_unix_nanos: 111,
            stopped_at_unix_nanos: None,
            last_error: None,
            sessions: RuntimeSessionStats {
                lane_count: 1,
                active_sessions: 0,
                idle_sessions: 1,
                idle_resident_prefixes: 0,
                tracked_token_counts: 0,
                max_session_tokens: 2048,
                total_session_tokens: 0,
                lanes: vec![],
            },
            sessions_captured_at_unix_nanos: 111,
            telemetry: TelemetryStats {
                queued: 0,
                sent: 0,
                dropped: 0,
                export_errors: 0,
            },
        }
    }

    #[test]
    fn single_stage_config_materializes_direct_gguf_runtime_slice() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36));

        let config = single_stage_config(&options).unwrap();

        assert_eq!(config.model_id, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.model_path.as_deref(), Some("/models/qwen.gguf"));
        assert_eq!(
            config.package_ref.as_deref(),
            Some("gguf:///models/qwen.gguf")
        );
        assert_eq!(
            config.manifest_sha256.as_deref(),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        );
        assert_eq!(
            config.source_model_path.as_deref(),
            Some("/models/qwen.gguf")
        );
        assert_eq!(config.source_model_bytes, Some(1234));
        assert!(config.materialized_path.is_none());
        assert!(!config.materialized_pinned);
        assert_eq!(config.stage_id, "stage-0");
        assert_eq!(config.stage_index, 0);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 36);
        assert_eq!(config.ctx_size, 8192);
        assert_eq!(config.n_gpu_layers, -1);
        assert!(config.selected_device.is_none());
        assert_eq!(config.load_mode, LoadMode::RuntimeSlice);
        assert!(config.upstream.is_none());
        assert!(config.downstream.is_none());
    }

    #[test]
    fn single_stage_config_canonicalizes_checkpoint_quantization_aliases() {
        let mut options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36));
        options.checkpoint_quantization = Some("Q4_K".to_string());

        let config = single_stage_config(&options).unwrap();

        assert_eq!(config.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn single_stage_config_preserves_projector_path() {
        let options = SkippyModelLoadOptions::for_direct_gguf("Qwen2.5-VL", "/models/qwen-vl.gguf")
            .with_layer_end(36)
            .with_package_identity(fake_package_identity(36))
            .with_projector_path("/models/mmproj-qwen-vl.gguf");

        let config = single_stage_config(&options).unwrap();

        assert_eq!(
            config.projector_path.as_deref(),
            Some("/models/mmproj-qwen-vl.gguf")
        );
    }

    #[test]
    fn single_stage_config_preserves_selected_device_descriptor() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36))
                .with_selected_device(SkippyDeviceDescriptor {
                    backend_device: "CUDA3".into(),
                    stable_id: Some("uuid:GPU-123".into()),
                    index: Some(3),
                    vram_bytes: Some(24_000_000_000),
                });

        let config = single_stage_config(&options).unwrap();
        let device = config.selected_device.expect("device descriptor");

        assert_eq!(device.backend_device, "CUDA3");
        assert_eq!(device.stable_id.as_deref(), Some("uuid:GPU-123"));
        assert_eq!(device.index, Some(3));
        assert_eq!(device.vram_bytes, Some(24_000_000_000));
    }

    #[test]
    fn single_stage_config_rejects_empty_selected_backend_device() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(1)
            .with_selected_device(SkippyDeviceDescriptor {
                backend_device: String::new(),
                stable_id: Some("uuid:GPU-123".into()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            });

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("selected backend device"));
    }

    #[test]
    fn single_stage_config_rejects_empty_layer_range() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(0)
            .with_package_identity(fake_package_identity(1));

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("layer_end"));
    }

    #[test]
    fn embedded_state_maps_to_mesh_skippy_state() {
        assert_eq!(
            map_embedded_state(EmbeddedState::Starting),
            SkippyModelState::Starting
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Ready),
            SkippyModelState::Ready
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Failed),
            SkippyModelState::Failed
        );
    }

    #[test]
    fn status_includes_guardrail_policy_without_private_content() {
        let config = fake_stage_config();
        let embedded = fake_embedded_runtime_status(&config);
        let local = HandleState {
            state: SkippyModelState::Ready,
            stopped_at_unix_nanos: None,
            last_error: None,
        };
        let status = status_from_parts(
            &config,
            &embedded,
            &local,
            222,
            Some(OpenAiGuardrailsStatus {
                mode: "disabled",
                target: "skippy",
                streaming: "pass_through",
                retry_exhaustion: "error",
                small_model_policy: "small_models_only",
                small_param_threshold_b: 9.0,
                max_tool_retries: 1,
                max_structured_retries: 2,
            }),
        );

        let guardrails = serde_json::to_value(
            status
                .openai_guardrails
                .expect("skippy status should include guardrails policy"),
        )
        .expect("guardrails serialize");
        let guardrails = guardrails
            .as_object()
            .expect("guardrails should serialize as an object");

        assert_eq!(guardrails.len(), 8);
        assert_eq!(guardrails.get("mode"), Some(&serde_json::json!("disabled")));
        assert_eq!(guardrails.get("target"), Some(&serde_json::json!("skippy")));
        assert_eq!(
            guardrails.get("streaming"),
            Some(&serde_json::json!("pass_through"))
        );
        assert_eq!(
            guardrails.get("retry_exhaustion"),
            Some(&serde_json::json!("error"))
        );
        assert_eq!(
            guardrails.get("small_model_policy"),
            Some(&serde_json::json!("small_models_only"))
        );
        assert_eq!(
            guardrails.get("small_param_threshold_b"),
            Some(&serde_json::json!(9.0))
        );
        assert_eq!(
            guardrails.get("max_tool_retries"),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            guardrails.get("max_structured_retries"),
            Some(&serde_json::json!(2))
        );

        for forbidden in [
            "prompt",
            "schema",
            "tool_args",
            "tool_names",
            "reserved_tool_prefix",
            "sentinels",
            "raw_tool_names",
            "sentinel_definitions",
        ] {
            assert!(
                guardrails.get(forbidden).is_none(),
                "privacy-safe status should omit {forbidden}"
            );
        }
    }

    #[test]
    fn guardrail_config_status_tracks_shared_policy_handle() {
        let policy = GuardrailPolicyHandle::default();
        let config = skippy_openai_guardrails_for_policy_handle(policy.clone());

        assert_eq!(config.status().mode, "disabled");

        policy.set_mode(GuardrailMode::MetricsOnly);
        assert_eq!(config.status().mode, "metrics");

        policy.set_mode(GuardrailMode::Enforce);
        let status = config.status();
        assert_eq!(status.mode, "enforce");
        assert_eq!(status.streaming, "pass_through");
        assert_eq!(status.retry_exhaustion, "error");
        assert_eq!(status.max_tool_retries, 1);
        assert_eq!(status.max_structured_retries, 2);
    }

    #[tokio::test]
    async fn host_guardrail_wrapper_applies_compaction_when_guardrails_are_disabled() {
        let backend = Arc::new(RecordingHostBackend::default());
        let wrapped = wrap_host_guardrail_backend(
            backend.clone(),
            Some(&OpenAiGuardrailsConfig {
                target: OpenAiGuardrailsTarget::Skippy,
                policy: GuardrailPolicyHandle::default(),
                compaction: Some(CompactionConfig::default()),
            }),
            Some(8),
            None,
        );
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [
                {"role": "tool", "content": "large intermediate result", "tool_call_id": "call_1"},
                {"role": "user", "content": "continue"}
            ],
            (MESH_COMPACT_FIELD): true
        }))
        .expect("valid compacting request");

        wrapped
            .chat_completion(request)
            .await
            .expect("wrapped chat completion");

        let seen = backend
            .seen_chat
            .lock()
            .expect("seen chat lock poisoned")
            .clone()
            .expect("inner backend should see compacted request");
        assert_eq!(
            seen.messages.first().map(|message| message.role.as_str()),
            Some("system")
        );
        assert!(
            seen.messages.iter().all(|message| message.role != "tool"),
            "host-runtime wrapper should run compacting before the embedded backend sees the request"
        );
    }

    #[tokio::test]
    async fn host_guardrail_wrapper_uses_live_policy_mode() {
        let backend = Arc::new(RecordingHostBackend::default());
        let policy = GuardrailPolicyHandle::default();
        let wrapped = wrap_host_guardrail_backend(
            backend.clone(),
            Some(&OpenAiGuardrailsConfig {
                target: OpenAiGuardrailsTarget::Skippy,
                policy: policy.clone(),
                compaction: None,
            }),
            Some(8192),
            None,
        );
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "look this up"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }))
        .expect("valid tool request");

        wrapped.chat_completion(request.clone()).await.unwrap();
        assert_eq!(
            backend
                .seen_chat
                .lock()
                .expect("seen chat lock poisoned")
                .clone()
                .unwrap()
                .tools,
            request.tools
        );

        policy.update(GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        });
        let _ = wrapped.chat_completion(request).await;

        let seen = backend
            .seen_chat
            .lock()
            .expect("seen chat lock poisoned")
            .clone()
            .unwrap();
        let tool_names = seen
            .tools
            .as_ref()
            .and_then(|tools| tools.as_array())
            .unwrap()
            .iter()
            .filter_map(|tool| tool.get("function"))
            .filter_map(|function| function.get("name"))
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>();
        assert!(tool_names.contains(&openai_frontend::MESH_RESPOND_TOOL_NAME));
    }
}
