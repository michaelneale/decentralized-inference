use crate::binary_transport::PredictionReturnHub;
use crate::binary_transport::PredictionReturnReceiver;
use crate::binary_transport::WireCondition;
use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::GenerationLifecycleConfig;
use crate::frontend::GenerationReceiptConfig;
use crate::frontend::LinearProposalIngressConfig;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpStats;
use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::decode_scheduler::VerifyWindowPipelineStats;
use crate::frontend::generation::DraftRunner;
use crate::frontend::generation::GenerationConcurrencyController;
use crate::frontend::generation::GenerationServiceEstimator;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PersistentStageLanePool;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::native_mtp::NativeMtpDecodeTelemetry;
use crate::frontend::prefill::PrefillChunkPolicy;
use crate::frontend::speculative::OpenAiSpeculativeStats;
use crate::kv_integration::KvStageIntegration;
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::FinishReason;
use openai_frontend::OpenAiHookPolicy;
use openai_frontend::Usage;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_runtime::SamplingConfig;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

pub(in crate::frontend) struct GenerationSessionLockEntry {
    pub(in crate::frontend) semaphore: Arc<Semaphore>,
    pub(in crate::frontend) users: AtomicUsize,
}

#[derive(Clone)]
pub(in crate::frontend) struct StageOpenAiBackend {
    pub(in crate::frontend) runtime: Arc<Mutex<RuntimeState>>,
    pub(in crate::frontend) config: StageConfig,
    pub(in crate::frontend) telemetry: Telemetry,
    pub(in crate::frontend) model_id: String,
    pub(in crate::frontend) default_max_tokens: u32,
    pub(in crate::frontend) request_defaults: EmbeddedOpenAiRequestDefaults,
    pub(in crate::frontend) ctx_size: usize,
    pub(in crate::frontend) mode: OpenAiBackendMode,
    pub(in crate::frontend) draft: Option<Arc<Mutex<DraftRunner>>>,
    pub(in crate::frontend) speculative_window: usize,
    pub(in crate::frontend) adaptive_speculative_window: bool,
    pub(in crate::frontend) ngram_max: usize,
    pub(in crate::frontend) speculative: SpeculativeDecodeConfig,
    pub(in crate::frontend) generation_limit: Arc<GenerationConcurrencyController>,
    pub(in crate::frontend) generation_queue_depth: Arc<AtomicUsize>,
    pub(in crate::frontend) generation_queue_limit: usize,
    pub(in crate::frontend) generation_admission_timeout: Duration,
    pub(in crate::frontend) generation_service_estimator: Arc<GenerationServiceEstimator>,
    pub(in crate::frontend) generation_session_locks:
        Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    pub(in crate::frontend) generation_token_budget: Arc<GenerationTokenBudget>,
    pub(in crate::frontend) hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    pub(in crate::frontend) generation_receipt: Option<GenerationReceiptConfig>,
    pub(in crate::frontend) generation_lifecycle: Option<GenerationLifecycleConfig>,
    pub(in crate::frontend) linear_proposal_ingress: Option<LinearProposalIngressConfig>,
    pub(in crate::frontend) kv: Option<Arc<KvStageIntegration>>,
    pub(in crate::frontend) iteration_scheduler: IterationScheduler,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(in crate::frontend) enum OpenAiBackendMode {
    LocalRuntime,
    EmbeddedStageZero {
        config: StageConfig,
        prefill_chunk_policy: PrefillChunkPolicy,
        activation_width: i32,
        downstream_wire_condition: WireCondition,
        prefill_reply_credit_limit: usize,
        lane_pool: Option<Arc<PersistentStageLanePool>>,
        prediction_returns: Option<Arc<PredictionReturnHub>>,
    },
}

impl OpenAiBackendMode {
    pub(in crate::frontend) fn label(&self) -> &'static str {
        match self {
            Self::LocalRuntime => "local-runtime",
            Self::EmbeddedStageZero { .. } => "embedded-stage0",
        }
    }
}

pub(in crate::frontend) struct PhaseTimer {
    pub(in crate::frontend) start_unix_nanos: u64,
    pub(in crate::frontend) start_instant: Instant,
}

impl PhaseTimer {
    pub(in crate::frontend) fn start() -> Self {
        Self {
            start_unix_nanos: now_unix_nanos() as u64,
            start_instant: Instant::now(),
        }
    }

    pub(in crate::frontend) fn elapsed_ms(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64() * 1000.0
    }
}

pub(in crate::frontend) fn decode_token_phase(decode_step: u32) -> &'static str {
    match decode_step {
        0 => "cold",
        1..=7 => "warmup",
        _ => "steady",
    }
}

#[derive(Default)]
pub(in crate::frontend) struct GenerationMetrics {
    pub(in crate::frontend) detokenize_ms: f64,
    pub(in crate::frontend) text_emit_ms: f64,
    pub(in crate::frontend) eog_check_ms: f64,
}

pub(in crate::frontend) struct PreparedTextPrompt {
    pub(in crate::frontend) token_ids: Vec<i32>,
    pub(in crate::frontend) max_tokens: u32,
}

pub(in crate::frontend) struct LocalGeneration<'a> {
    pub(in crate::frontend) prompt_token_ids: &'a [i32],
    pub(in crate::frontend) recurrent_cache_prefix_token_ids: Option<&'a [i32]>,
    pub(in crate::frontend) max_tokens: u32,
    pub(in crate::frontend) sampling: &'a SamplingConfig,
    pub(in crate::frontend) chat_sampling_metadata: Option<&'a str>,
    pub(in crate::frontend) speculative: &'a SpeculativeDecodeConfig,
    pub(in crate::frontend) native_mtp_enabled: bool,
    pub(in crate::frontend) hook_request: Option<ChatCompletionRequest>,
    pub(in crate::frontend) hook_runtime: Option<tokio::runtime::Handle>,
    pub(in crate::frontend) cancellation: Option<&'a openai_frontend::CancellationToken>,
    pub(in crate::frontend) ids: &'a OpenAiGenerationIds,
}

pub(in crate::frontend) struct EmbeddedStageZeroGeneration<'a> {
    pub(in crate::frontend) config: &'a StageConfig,
    pub(in crate::frontend) prefill_chunk_policy: &'a PrefillChunkPolicy,
    pub(in crate::frontend) activation_width: i32,
    pub(in crate::frontend) downstream_wire_condition: WireCondition,
    pub(in crate::frontend) prefill_reply_credit_limit: usize,
    pub(in crate::frontend) lane_pool: Option<Arc<PersistentStageLanePool>>,
    pub(in crate::frontend) prediction_return: Option<PredictionReturnReceiver>,
    pub(in crate::frontend) draft: Option<Arc<Mutex<DraftRunner>>>,
    pub(in crate::frontend) speculative_window: usize,
    pub(in crate::frontend) adaptive_speculative_window: bool,
    pub(in crate::frontend) ngram_max: usize,
    pub(in crate::frontend) speculative: &'a SpeculativeDecodeConfig,
    pub(in crate::frontend) native_mtp_enabled: bool,
    pub(in crate::frontend) prompt_token_ids: &'a [i32],
    pub(in crate::frontend) recurrent_cache_prefix_token_ids: Option<&'a [i32]>,
    pub(in crate::frontend) max_tokens: u32,
    pub(in crate::frontend) sampling: &'a SamplingConfig,
    pub(in crate::frontend) chat_sampling_metadata: Option<&'a str>,
    pub(in crate::frontend) hook_request: Option<ChatCompletionRequest>,
    pub(in crate::frontend) hook_runtime: Option<tokio::runtime::Handle>,
    pub(in crate::frontend) cancellation: Option<&'a openai_frontend::CancellationToken>,
    pub(in crate::frontend) ids: &'a OpenAiGenerationIds,
}

pub(in crate::frontend) struct SplitMultimodalGeneration<'a> {
    pub(in crate::frontend) prompt: PreparedGenerationPrompt,
    pub(in crate::frontend) max_tokens: GenerationTokenLimit,
    pub(in crate::frontend) stop: Option<&'a openai_frontend::StopSequence>,
    pub(in crate::frontend) sampling: SamplingConfig,
    pub(in crate::frontend) cancellation: Option<&'a openai_frontend::CancellationToken>,
    pub(in crate::frontend) ids: OpenAiGenerationIds,
    pub(in crate::frontend) config: StageConfig,
    pub(in crate::frontend) activation_width: i32,
    pub(in crate::frontend) downstream_wire_condition: WireCondition,
    pub(in crate::frontend) lane_pool: Arc<PersistentStageLanePool>,
    pub(in crate::frontend) prediction_return: Option<PredictionReturnReceiver>,
    pub(in crate::frontend) emulation_active: bool,
}

pub(in crate::frontend) struct EmbeddedLocalOutput {
    pub(in crate::frontend) output: skippy_runtime::ActivationFrame,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
}

#[derive(Default)]
pub(in crate::frontend) struct EmbeddedExecutionStats {
    pub(in crate::frontend) stage0_compute_ms: f64,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
    pub(in crate::frontend) activation_encode_ms: f64,
    pub(in crate::frontend) output_activation_bytes: usize,
    pub(in crate::frontend) forward_activation_bytes: usize,
    pub(in crate::frontend) forward_write_ms: f64,
    pub(in crate::frontend) downstream_wait_ms: f64,
}

pub(in crate::frontend) struct EmbeddedStageExecution {
    pub(in crate::frontend) reply: StageReply,
    pub(in crate::frontend) stats: EmbeddedExecutionStats,
    pub(in crate::frontend) elapsed_ms: f64,
}

pub(in crate::frontend) struct EmbeddedFusedFirstDecode {
    pub(in crate::frontend) predicted: i32,
    pub(in crate::frontend) predicted_tokens: Vec<i32>,
    pub(in crate::frontend) native_mtp_draft: Option<NativeMtpDraft>,
    pub(in crate::frontend) reply_stats: StageReplyStats,
    pub(in crate::frontend) execution: EmbeddedExecutionStats,
    pub(in crate::frontend) elapsed_ms: f64,
    pub(in crate::frontend) token_phase: &'static str,
    pub(in crate::frontend) message_kind: &'static str,
}

pub(in crate::frontend) struct GeneratedText {
    pub(in crate::frontend) prompt_tokens: u32,
    pub(in crate::frontend) completion_tokens: u32,
    pub(in crate::frontend) cache_status: &'static str,
    pub(in crate::frontend) cached_prompt_tokens: u32,
    pub(in crate::frontend) matched_prefix_tokens: u32,
    pub(in crate::frontend) suffix_prefill_tokens: u32,
    pub(in crate::frontend) cache_hit_kind: Option<&'static str>,
    pub(in crate::frontend) native_mtp_stats: NativeMtpStats,
    pub(in crate::frontend) native_mtp_decode_telemetry: Option<NativeMtpDecodeTelemetry>,
    pub(in crate::frontend) verify_window_pipeline_stats: Option<VerifyWindowPipelineStats>,
    pub(in crate::frontend) speculative_stats: Option<OpenAiSpeculativeStats>,
    pub(in crate::frontend) prompt_ms: f64,
    pub(in crate::frontend) predicted_ms: f64,
    pub(in crate::frontend) queue_wait_ms: f64,
    pub(in crate::frontend) restore_ms: f64,
    pub(in crate::frontend) text: String,
    pub(in crate::frontend) finish_reason: FinishReason,
    pub(in crate::frontend) detokenize_ms: f64,
    pub(in crate::frontend) text_emit_ms: f64,
    pub(in crate::frontend) eog_check_ms: f64,
}

impl GeneratedText {
    pub(in crate::frontend) fn usage(&self) -> Usage {
        Usage::new(self.prompt_tokens, self.completion_tokens)
            .with_cached_tokens(self.cached_prompt_tokens)
    }

    pub(in crate::frontend) fn timings(&self) -> Option<BTreeMap<String, Value>> {
        let stats = self.native_mtp_stats;
        let native_totals = self
            .native_mtp_decode_telemetry
            .and_then(NativeMtpDecodeTelemetry::composite_proposal_totals)
            .unwrap_or((stats.drafted_tokens, stats.accepted_tokens));
        let (drafted_tokens, accepted_tokens) = self
            .speculative_stats
            .as_ref()
            .filter(|stats| stats.windows > 0)
            .map_or(native_totals, |stats| {
                (
                    u64::try_from(stats.draft_tokens).unwrap_or(u64::MAX),
                    u64::try_from(stats.accepted_tokens).unwrap_or(u64::MAX),
                )
            });
        let mut timings = BTreeMap::from([
            ("prompt_n".to_string(), json!(self.prompt_tokens)),
            ("prompt_ms".to_string(), json!(self.prompt_ms)),
            ("queue_wait_ms".to_string(), json!(self.queue_wait_ms)),
            ("cache_restore_ms".to_string(), json!(self.restore_ms)),
            (
                "suffix_prefill_n".to_string(),
                json!(self.suffix_prefill_tokens),
            ),
            (
                "prompt_per_second".to_string(),
                json!(tokens_per_second(self.prompt_tokens, self.prompt_ms)),
            ),
            ("predicted_n".to_string(), json!(self.completion_tokens)),
            ("predicted_ms".to_string(), json!(self.predicted_ms)),
            (
                "predicted_per_second".to_string(),
                json!(tokens_per_second(self.completion_tokens, self.predicted_ms)),
            ),
            ("draft_n".to_string(), json!(drafted_tokens)),
            ("draft_n_accepted".to_string(), json!(accepted_tokens)),
            (
                "native_mtp_rejected".to_string(),
                json!(stats.rejected_tokens),
            ),
            (
                "native_mtp_proposal_compute_us".to_string(),
                json!(stats.proposal_compute_us),
            ),
            (
                "native_mtp_verification_compute_us".to_string(),
                json!(stats.verification_compute_us),
            ),
            (
                "native_mtp_verifications".to_string(),
                json!(stats.verification_count),
            ),
        ]);
        if let Some(telemetry) = self.native_mtp_decode_telemetry {
            telemetry.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.verify_window_pipeline_stats.as_ref() {
            stats.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.speculative_stats.as_ref() {
            stats.insert_response_timings(&mut timings);
        }
        Some(timings)
    }
}

fn tokens_per_second(token_count: u32, elapsed_ms: f64) -> f64 {
    if elapsed_ms > 0.0 {
        f64::from(token_count) * 1_000.0 / elapsed_ms
    } else {
        0.0
    }
}
