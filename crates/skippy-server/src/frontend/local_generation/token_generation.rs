use super::cache_deadline::cache_operation_deadline;
use super::native_mtp_decode::NativeMtpSpanProgress;
use crate::frontend::NativeMtpDecodeOptions;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpVerifier;
use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::LocalGeneration;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation_receipt::GenerationLifecycleState;
use crate::frontend::generation_receipt::complete_generation_before_cleanup;
use crate::frontend::iteration_scheduler::CacheRuntimeContext;
use crate::frontend::iteration_scheduler::DirectIterationChannel;
use crate::frontend::iteration_scheduler::ScheduledGenerationRequest;
use crate::frontend::iteration_scheduler::ScheduledResumeRequest;
use crate::frontend::linear_proposal::greedy_linear_proposal_admitted;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::saturating_u32;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use crate::kv_integration::{KvStageIntegration, PrefillKvIdentity, StagePrefixCachePayload};
use crate::runtime_state::{RuntimeSessionStats, RuntimeState};
use axum::http::StatusCode;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiErrorKind;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_metrics::attr as attr_key;
use skippy_runtime::NativeMtpDraft as RuntimeNativeMtpDraft;
use skippy_runtime::SamplingConfig;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::receipt_lifecycle::{begin_generation_receipt, commit_local_generation_token};
use super::{LocalGenerationReceiptFinalization, prompt_fits_single_prefill_sample};

mod exact_state_recording;
mod kv_restore;

pub(in crate::frontend) fn resident_capacity_admission_error(
    capacity: &crate::kv_integration::ResidentCapacityDecision,
) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "resident KV capacity admission rejected request: {} token deficit (capacity={}, active={}, pinned={}, request={}, minimum_free={})",
            capacity.admission_deficit_tokens,
            capacity.capacity_tokens,
            capacity.active_tokens,
            capacity.pinned_tokens,
            capacity.request_tokens,
            capacity.minimum_free_tokens,
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

struct PromptPrefillResult {
    prompt_prefill_sample: Option<i32>,
    chat_sampling_configured: bool,
}

struct KvRecordResult {
    /// Pages synchronously written during the restore phase.
    resident_recorded_pages: usize,
    /// Exact-state checkpoint tasks enqueued for detached async execution.
    /// These may not result in a write if the session has already advanced.
    resident_enqueued_checkpoints: usize,
    proactive_eviction_status: &'static str,
    proactive_eviction_error_kind: Option<&'static str>,
    proactive_eviction_target_tokens: u64,
    proactive_evicted_entries: usize,
    proactive_evicted_tokens: u64,
    proactive_eviction_error: Option<anyhow::Error>,
}

fn ensure_cache_operation_active(
    runtime: &mut RuntimeState,
    session_id: &str,
    cache_operation: &CacheRuntimeContext,
) -> OpenAiResult<()> {
    if let Err(error) = cache_operation.ensure_active() {
        let _ = runtime.drop_session_timed(session_id);
        return Err(error);
    }
    Ok(())
}

fn prefill_cache_chunks(
    runtime: &mut RuntimeState,
    session_id: &str,
    tokens: &[i32],
    chunk_tokens: usize,
    cache_operation: &CacheRuntimeContext,
) -> OpenAiResult<()> {
    for chunk in tokens.chunks(chunk_tokens.max(1)) {
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        runtime
            .prefill(session_id, chunk)
            .map_err(openai_backend_error)?;
    }
    ensure_cache_operation_active(runtime, session_id, cache_operation)
}

impl Default for KvRecordResult {
    fn default() -> Self {
        Self {
            resident_recorded_pages: 0,
            resident_enqueued_checkpoints: 0,
            proactive_eviction_status: "disabled",
            proactive_eviction_error_kind: None,
            proactive_eviction_target_tokens: 0,
            proactive_evicted_entries: 0,
            proactive_evicted_tokens: 0,
            proactive_eviction_error: None,
        }
    }
}

fn insert_resident_capacity_attrs(
    attrs: &mut BTreeMap<String, serde_json::Value>,
    decision: &crate::kv_integration::ResidentCapacityDecision,
) {
    let status = if !decision.enabled {
        "disabled"
    } else if !decision.capacity_known {
        "unknown_capacity"
    } else if !decision.admitted {
        "rejected"
    } else if decision.evicted_entries > 0 {
        "evicted"
    } else {
        "admitted"
    };
    attrs.insert(attr_key::KV_CAPACITY_STATUS.to_string(), json!(status));
    attrs.insert(
        attr_key::KV_CAPACITY_TOKENS.to_string(),
        json!(decision.capacity_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_ACTIVE_TOKENS.to_string(),
        json!(decision.active_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PHYSICAL_USED_TOKENS.to_string(),
        json!(decision.physical_used_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PINNED_TOKENS.to_string(),
        json!(decision.pinned_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_REQUEST_TOKENS.to_string(),
        json!(decision.request_tokens),
    );
    attrs.insert(
        "skippy.kv.capacity_inflight_reservations".to_string(),
        json!(decision.inflight_reservations),
    );
    attrs.insert(
        "skippy.kv.capacity_inflight_outstanding_tokens".to_string(),
        json!(decision.inflight_outstanding_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_MINIMUM_FREE_TOKENS.to_string(),
        json!(decision.minimum_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_TARGET_FREE_TOKENS.to_string(),
        json!(decision.target_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PROJECTED_FREE_TOKENS.to_string(),
        json!(decision.projected_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_ADMISSION_DEFICIT_TOKENS.to_string(),
        json!(decision.admission_deficit_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_REQUIRED_EVICTION_TOKENS.to_string(),
        json!(decision.required_eviction_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_EVICTED_ENTRIES.to_string(),
        json!(decision.evicted_entries),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_EVICTED_TOKENS.to_string(),
        json!(decision.evicted_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PHYSICAL_EVICTED_TOKENS.to_string(),
        json!(decision.physical_evicted_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PREDICTED_RECOMPUTE_COST.to_string(),
        json!(decision.predicted_recompute_cost),
    );
}

struct KvRestoreOutcome {
    runtime_sessions_before: Option<RuntimeSessionStats>,
    runtime_sessions_after: Option<RuntimeSessionStats>,
    restored_prefill: bool,
    restored_prefill_tokens: usize,
    capacity: crate::kv_integration::ResidentCapacityDecision,
    record: KvRecordResult,
    prompt_prefill_sample: Option<i32>,
}

pub(super) struct DecodeState {
    pub(super) decoded_tokens: usize,
    pub(super) current: i32,
    /// Target-authoritative tokens emitted by this request. The runtime has
    /// consumed every token in `prompt_token_ids + generated_token_ids` except
    /// the final element: autoregressive decode consumes `current` and returns
    /// the next (still unconsumed) token. This lets us name the recurrent state
    /// at the exact token boundary that was actually captured.
    pub(super) generated_token_ids: Vec<i32>,
    pub(super) stopped: bool,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_wait_max_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_hold_max_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) runtime_sessions_before: Option<RuntimeSessionStats>,
    pub(super) runtime_sessions_after: Option<RuntimeSessionStats>,
    pub(super) hook_request: Option<ChatCompletionRequest>,
    pub(super) hook_runtime: Option<tokio::runtime::Handle>,
    pub(super) generation_hooks_active: bool,
    pub(super) linear_proposal_max_tokens: usize,
    pub(super) linear_context_tokens: Option<Vec<i32>>,
    pub(super) pending_linear_proposal_tokens: Vec<i32>,
    pub(super) emit_token_debug: bool,
    pub(super) native_mtp_options: NativeMtpDecodeOptions,
    pub(super) native_mtp: NativeMtpVerifier,
    /// Whether batched native-MTP spans may commit for this request. Token
    /// equality is only a valid acceptance test under greedy sampling.
    pub(super) native_mtp_span_admitted: bool,
    pub(super) post_prefill_hook_checked: bool,
    pub(super) last_mid_generation_hook_at: Option<usize>,
    pub(super) direct_iteration_channel: Option<DirectIterationChannel>,
}

pub(super) enum LinearProposalProgress {
    NotUsed,
    Continue,
    Stop,
}

pub(in crate::frontend) fn post_decode_checkpoint_tokens(
    prompt_token_ids: &[i32],
    generated_token_ids: &[i32],
) -> Option<Vec<i32>> {
    if generated_token_ids.is_empty() {
        return None;
    }
    let mut checkpoint = Vec::with_capacity(
        prompt_token_ids
            .len()
            .saturating_add(generated_token_ids.len())
            .saturating_sub(1),
    );
    checkpoint.extend_from_slice(prompt_token_ids);
    checkpoint
        .extend_from_slice(&generated_token_ids[..generated_token_ids.len().saturating_sub(1)]);
    Some(checkpoint)
}

pub(in crate::frontend) fn linear_proposal_allowed(
    recurrent_checkpoint_required: bool,
    checkpoint_attempted: bool,
) -> bool {
    !recurrent_checkpoint_required || checkpoint_attempted
}

fn initial_recurrent_checkpoint_attempted(
    recurrent_checkpoint_required: bool,
    prompt_prefill_sample: Option<i32>,
) -> bool {
    !recurrent_checkpoint_required || prompt_prefill_sample.is_some()
}

fn scheduler_generation_driver_eligible(
    max_tokens: u32,
    kv_enabled: bool,
    linear_proposal_enabled: bool,
    draft_enabled: bool,
    native_mtp_enabled: bool,
    speculative_strategy: &str,
    hooks_enabled: bool,
) -> bool {
    // This predicate selects the scheduler's built-in plain-text generation
    // driver. Speculative and extended requests keep their strategy-specific
    // proposal/checkpoint state machine, but every target iteration,
    // verification, and repair operation is submitted to the same scheduler
    // worker, including strategy-specific runtime operations.
    max_tokens > 0
        && !kv_enabled
        && !linear_proposal_enabled
        && !draft_enabled
        && !native_mtp_enabled
        && speculative_strategy == "disabled"
        && !hooks_enabled
}

fn deferred_suffix_prefill_enabled(
    payload: Option<StagePrefixCachePayload>,
    max_tokens: u32,
) -> bool {
    // Resident KV suffixes must remain in the serialized cache operation. Releasing
    // it between restore and suffix prefill lets another request interleave and
    // destroys prefill phase locality on a shared accelerator.
    max_tokens > 0
        && matches!(
            payload,
            Some(StagePrefixCachePayload::KvRecurrent | StagePrefixCachePayload::FullState)
        )
}

fn deferred_suffix_split(
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
    checkpoint_token_count: Option<usize>,
) -> (usize, Option<usize>) {
    let suffix_start = restored_prefill_tokens.min(prompt_token_count);
    let checkpoint = checkpoint_token_count
        .filter(|&boundary| boundary > suffix_start && boundary < prompt_token_count);
    (suffix_start, checkpoint)
}

fn select_deferred_checkpoint_boundary(
    shared_checkpoint: Option<usize>,
    chat_checkpoint: Option<usize>,
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
) -> Option<usize> {
    let valid =
        |boundary: &usize| *boundary > restored_prefill_tokens && *boundary < prompt_token_count;
    shared_checkpoint
        .filter(valid)
        .or_else(|| chat_checkpoint.filter(valid))
}

impl StageOpenAiBackend {
    pub(in crate::frontend) fn generate_local_tokens(
        &self,
        mut request: LocalGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        let session_id = request.ids.session_label.clone();
        let receipt_request_id = request.ids.request_id;
        let receipt_session_id = request.ids.session_id;
        let receipt_prompt_token_ids = begin_generation_receipt(
            self.generation_receipt.as_ref(),
            request.ids,
            request.prompt_token_ids,
        );
        let mut lifecycle = GenerationLifecycleState::new(
            self.generation_lifecycle.as_ref(),
            request.ids.request_id,
            request.ids.session_id,
            request.ids.agent_session_id.clone(),
            request.ids.frontend_request_id,
            request.prompt_token_ids,
        );
        let receipt_observation = self.generation_receipt.as_ref().map(|config| {
            RefCell::new(Some(
                config.observation(
                    usize::try_from(request.max_tokens)
                        .expect("supported targets represent u32 token budgets as usize"),
                ),
            ))
        });
        let mut receipt_cancelled = false;
        let mut receipt_model_generation_elapsed = None;
        let mut cache_stats = GenerationCacheStats::default();
        let mut lifecycle_committed_token_count = 0usize;
        let mut emit_token = |token_id| {
            if let Some(observation) = receipt_observation.as_ref()
                && let Some(observation) = observation.borrow_mut().as_mut()
            {
                observation.record_token(token_id, request.ids.request_started_at.elapsed());
            }
            commit_local_generation_token(
                self.generation_receipt.as_ref(),
                receipt_request_id,
                receipt_session_id,
                &mut lifecycle_committed_token_count,
                token_id,
            );
            lifecycle.commit(token_id, request.ids.request_started_at.elapsed());
            let control = on_token(token_id)?;
            if control == TokenControl::Stop
                && let Some(observation) = receipt_observation.as_ref()
                && let Some(observation) = observation.borrow_mut().as_mut()
            {
                observation.mark_callback_stop();
            }
            if control == TokenControl::Stop {
                lifecycle.mark_callback_stop();
            }
            Ok(control)
        };
        let result = (|| {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                return Err(OpenAiError::backend("request cancelled"));
            }
            if self.uses_scheduler_builtin_driver(&request) {
                let model_generation_elapsed = self.run_scheduled_generation(
                    &request,
                    &session_id,
                    &mut cache_stats,
                    &mut emit_token,
                )?;
                receipt_model_generation_elapsed = Some(model_generation_elapsed);
                return Ok(());
            }
            let prefill = self.prefill_prompt(&request, &session_id, &mut cache_stats)?;
            self.configure_chat_sampling_if_needed(
                &request,
                &session_id,
                prefill.chat_sampling_configured,
            )?;
            let model_generation_elapsed = self.run_scheduler_feature_loop(
                &mut request,
                &session_id,
                prefill.prompt_prefill_sample,
                &mut cache_stats,
                &mut receipt_cancelled,
                &mut emit_token,
            )?;
            receipt_model_generation_elapsed = Some(model_generation_elapsed);
            Ok(())
        })();
        let receipt_observation = receipt_observation
            .as_ref()
            .and_then(|observation| observation.borrow_mut().take());
        let generation_succeeded = result.is_ok();
        if receipt_cancelled {
            lifecycle.mark_cancelled();
        }
        complete_generation_before_cleanup(
            result,
            || {
                let receipt_result = self.finalize_generation_receipt(
                    LocalGenerationReceiptFinalization {
                        session_label: &session_id,
                        request_id: receipt_request_id,
                        session_id: receipt_session_id,
                        agent_session_id: request.ids.agent_session_id.as_deref(),
                        prompt_token_ids: receipt_prompt_token_ids.unwrap_or_default(),
                        observation: receipt_observation,
                        cancelled: receipt_cancelled,
                        model_generation_elapsed: receipt_model_generation_elapsed,
                        frontend_request_id: request.ids.frontend_request_id,
                    },
                    generation_succeeded,
                );
                lifecycle.finish(generation_succeeded);
                receipt_result
            },
            || self.cleanup_local_generation_session(&session_id, request.ids),
        )?;
        Ok(cache_stats)
    }

    fn uses_scheduler_builtin_driver(&self, request: &LocalGeneration<'_>) -> bool {
        let kv_enabled = self.kv.is_some();
        let linear_proposal_enabled = self.linear_proposal_ingress.is_some();
        let draft_enabled = self.draft.is_some();
        let hooks_enabled = self.hook_policy.is_some();
        let eligible = scheduler_generation_driver_eligible(
            request.max_tokens,
            kv_enabled,
            linear_proposal_enabled,
            draft_enabled,
            request.native_mtp_enabled,
            &request.speculative.effective_strategy,
            hooks_enabled,
        );
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("skippy.scheduler.eligible".to_string(), json!(eligible));
        attrs.insert(
            "skippy.scheduler.runtime_owner".to_string(),
            json!("iteration-worker"),
        );
        attrs.insert(
            "skippy.scheduler.generation_driver".to_string(),
            json!(if eligible { "built-in" } else { "feature" }),
        );
        attrs.insert("skippy.scheduler.kv_enabled".to_string(), json!(kv_enabled));
        attrs.insert(
            "skippy.scheduler.linear_proposal_enabled".to_string(),
            json!(linear_proposal_enabled),
        );
        attrs.insert(
            "skippy.scheduler.draft_enabled".to_string(),
            json!(draft_enabled),
        );
        attrs.insert(
            "skippy.scheduler.native_mtp_enabled".to_string(),
            json!(request.native_mtp_enabled),
        );
        attrs.insert(
            "skippy.scheduler.speculative_strategy".to_string(),
            json!(request.speculative.effective_strategy),
        );
        attrs.insert(
            "skippy.scheduler.hooks_enabled".to_string(),
            json!(hooks_enabled),
        );
        self.telemetry.emit_debug("stage.scheduler_route", attrs);
        eligible
    }

    fn run_scheduled_generation(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Duration> {
        let timer = PhaseTimer::start();
        let stats = self.iteration_scheduler.generate(
            ScheduledGenerationRequest {
                id: session_id,
                prompt_tokens: request.prompt_token_ids,
                max_tokens: request.max_tokens,
                sampling: request.sampling.enabled.then_some(request.sampling),
                chat_sampling_metadata: request.chat_sampling_metadata,
                cancellation: request.cancellation,
            },
            emit_token,
        )?;
        cache_stats.suffix_prefill_tokens = saturating_u32(request.prompt_token_ids.len());
        cache_stats.prompt_ms = stats.prompt_ms;
        cache_stats.predicted_ms = stats.predicted_ms;
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "skippy.scheduler.serving_path".to_string(),
            json!("iteration"),
        );
        attrs.insert(
            "skippy.scheduler.prompt_tokens".to_string(),
            json!(request.prompt_token_ids.len()),
        );
        attrs.insert(
            "skippy.scheduler.max_tokens".to_string(),
            json!(request.max_tokens),
        );
        attrs.insert(
            "skippy.scheduler.prompt_ms".to_string(),
            json!(stats.prompt_ms),
        );
        attrs.insert(
            "skippy.scheduler.predicted_ms".to_string(),
            json!(stats.predicted_ms),
        );
        self.emit_openai_phase("stage.openai_scheduler_generation", timer, attrs);
        Ok(Duration::from_secs_f64(
            (stats.prompt_ms + stats.predicted_ms) / 1_000.0,
        ))
    }

    fn can_sample_whole_prompt_in_prefill(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
    ) -> OpenAiResult<bool> {
        if request.max_tokens == 0 || request.prompt_token_ids.len() <= 1 || self.kv.is_some() {
            return Ok(false);
        }
        let scheduler_session_id = session_id.to_string();
        let batch_size = self.iteration_scheduler.execute_runtime(
            "feature-prefill-admission",
            move |runtime| {
                runtime
                    .ensure_session_active(&scheduler_session_id)
                    .map_err(openai_backend_error)?;
                runtime
                    .admit_session_batch_size(&scheduler_session_id)
                    .map_err(openai_backend_error)
            },
        )?;
        Ok(prompt_fits_single_prefill_sample(
            request.prompt_token_ids.len(),
            batch_size,
        ))
    }

    fn prefill_prompt(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<PromptPrefillResult> {
        if self.can_sample_whole_prompt_in_prefill(request, session_id)? {
            let chat_sampling_configured = if let Some(metadata) = request.chat_sampling_metadata {
                self.configure_chat_sampling(
                    session_id,
                    metadata,
                    request.prompt_token_ids.len(),
                    request.sampling.enabled.then_some(request.sampling),
                )?;
                true
            } else {
                false
            };
            return Ok(PromptPrefillResult {
                prompt_prefill_sample: self.prefill_whole_prompt(
                    request,
                    session_id,
                    cache_stats,
                )?,
                chat_sampling_configured,
            });
        }
        if request.prompt_token_ids.len() > 1 {
            let chat_sampling_configured = if self.kv.as_ref().is_some_and(|kv| {
                kv.payload == StagePrefixCachePayload::ResidentKv
                    || deferred_suffix_prefill_enabled(Some(kv.payload), request.max_tokens)
            }) {
                if let Some(metadata) = request.chat_sampling_metadata {
                    self.configure_chat_sampling(
                        session_id,
                        metadata,
                        request.prompt_token_ids.len(),
                        request.sampling.enabled.then_some(request.sampling),
                    )?;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            let prompt_prefill_sample =
                self.restore_or_record_kv(request, session_id, cache_stats)?;
            return Ok(PromptPrefillResult {
                prompt_prefill_sample,
                chat_sampling_configured,
            });
        }
        Ok(PromptPrefillResult {
            prompt_prefill_sample: None,
            chat_sampling_configured: false,
        })
    }

    fn prefill_whole_prompt(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<Option<i32>> {
        let prefill_timer = PhaseTimer::start();
        let runtime_sessions_before = self
            .iteration_scheduler
            .execute_runtime("feature-prefill-stats-before", |runtime| {
                Ok(runtime.session_stats())
            })?;
        let outcome = self.iteration_scheduler.execute_iteration(
            session_id,
            request.prompt_token_ids,
            &[],
            request.sampling.enabled.then_some(request.sampling),
            true,
            skippy_runtime::IterationBatchPhase::Prefill,
        )?;
        let prompt_prefill_sample = Some(outcome.predicted);
        cache_stats.suffix_prefill_tokens = saturating_u32(request.prompt_token_ids.len());
        let runtime_sessions_after = self
            .iteration_scheduler
            .execute_runtime("feature-prefill-stats-after", |runtime| {
                Ok(runtime.session_stats())
            })?;
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.prefill_token_count".to_string(),
            json!(request.prompt_token_ids.len()),
        );
        attrs.insert("llama_stage.prefill_chunk_count".to_string(), json!(1));
        attrs.insert("skippy.kv.restored_prefill".to_string(), json!(false));
        attrs.insert("skippy.kv.restored_prefill_tokens".to_string(), json!(0));
        attrs.insert(
            "skippy.kv.prefill_suffix_tokens".to_string(),
            json!(request.prompt_token_ids.len()),
        );
        attrs.insert("skippy.kv.recorded_pages".to_string(), json!(0));
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(outcome.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(outcome.runtime_lock_hold_ms),
        );
        attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
        Self::insert_runtime_session_stats(
            &mut attrs,
            "llama_stage.runtime_sessions_before",
            &runtime_sessions_before,
        );
        Self::insert_runtime_session_stats(
            &mut attrs,
            "llama_stage.runtime_sessions_after",
            &runtime_sessions_after,
        );
        cache_stats.prompt_ms = prefill_timer.elapsed_ms();
        self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
        Ok(prompt_prefill_sample)
    }

    fn restore_or_record_kv(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<Option<i32>> {
        let prefill_timer = PhaseTimer::start();
        let prefill_tokens =
            Arc::<[i32]>::from(&request.prompt_token_ids[..request.prompt_token_ids.len() - 1]);
        let recurrent_cache_prefix_token_ids = request
            .recurrent_cache_prefix_token_ids
            .map(<[i32]>::to_vec);
        let max_tokens = request.max_tokens;
        let identity_timer = PhaseTimer::start();
        let (lookup_identities, refresh_cache_affinity) = match self.kv.as_ref() {
            Some(kv) => {
                let base = self.local_kv_message_base(session_id, request.ids);
                let identities = Arc::<[PrefillKvIdentity]>::from(kv.lookup_identities(
                    &self.config,
                    &base,
                    0,
                    prefill_tokens.as_ref(),
                ));
                let kv = kv.clone();
                let config = self.config.clone();
                let refresh_identities = Arc::clone(&identities);
                (
                    identities,
                    Box::new(move || kv.peek_cache_affinity(&config, &refresh_identities))
                        as Box<dyn Fn() -> skippy_scheduler::CacheAffinity + Send>,
                )
            }
            None => (
                Arc::<[PrefillKvIdentity]>::from([]),
                Box::new(skippy_scheduler::CacheAffinity::default)
                    as Box<dyn Fn() -> skippy_scheduler::CacheAffinity + Send>,
            ),
        };
        let kv_identity_ms = identity_timer.elapsed_ms();
        let cache_payload = self
            .kv
            .as_ref()
            .map_or(StagePrefixCachePayload::Disabled, |kv| kv.payload);
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = request.ids.clone();
        let scheduler_lookup_identities = Arc::clone(&lookup_identities);
        let scheduler_prefill_tokens = Arc::clone(&prefill_tokens);
        let record_prefill_tokens = Arc::clone(&prefill_tokens);
        let mut scheduler_cache_stats = std::mem::take(cache_stats);
        let cache_operation_deadline = cache_operation_deadline(
            self.generation_admission_timeout,
            request.prompt_token_ids.len(),
        );
        let outcome = self.iteration_scheduler.execute_cache_aware_runtime_timed(
            "feature-kv-restore-prefill-record",
            super::super::iteration_scheduler::CacheAwareRuntimeRequest {
                operation_id: request.ids.request_id_string(),
                deadline: cache_operation_deadline,
                cancellation: request.cancellation,
                prompt_tokens: Arc::clone(&prefill_tokens),
                priority: 0,
                payload: cache_payload,
                refresh_affinity: refresh_cache_affinity,
            },
            move |runtime, cache_operation| {
                let outcome = scheduler_backend.restore_or_record_kv_on_runtime(
                    runtime,
                    cache_operation,
                    &scheduler_ids,
                    &scheduler_session_id,
                    scheduler_prefill_tokens.as_ref(),
                    scheduler_lookup_identities.as_ref(),
                    kv_identity_ms,
                    recurrent_cache_prefix_token_ids.as_deref(),
                    max_tokens,
                    &mut scheduler_cache_stats,
                )?;
                Ok((outcome, scheduler_cache_stats))
            },
        )?;
        let mut runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
        let mut runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
        let mut runtime_lock_acquires = 1usize;
        let (outcome, updated_cache_stats) = outcome.value;
        *cache_stats = updated_cache_stats;
        let KvRestoreOutcome {
            runtime_sessions_before,
            mut runtime_sessions_after,
            restored_prefill,
            restored_prefill_tokens,
            capacity,
            mut record,
            mut prompt_prefill_sample,
        } = outcome;
        let mut prefill_chunk_count = 1usize;
        let mut scheduler_batch_size_max = 1usize;
        let deferred_suffix_prefill =
            deferred_suffix_prefill_enabled(self.kv.as_ref().map(|kv| kv.payload), max_tokens);
        if deferred_suffix_prefill && capacity.admitted {
            let exact_state = self
                .kv
                .as_ref()
                .is_some_and(|kv| kv.payload_is_exact_state());
            let checkpoint = exact_state
                .then(|| {
                    self.deferred_exact_checkpoint_identity(
                        request,
                        session_id,
                        &prefill_tokens,
                        restored_prefill_tokens,
                    )
                })
                .flatten();
            let (mut suffix_start, checkpoint_boundary) = deferred_suffix_split(
                restored_prefill_tokens,
                request.prompt_token_ids.len(),
                checkpoint
                    .as_ref()
                    .and_then(|identity| usize::try_from(identity.identity.token_count).ok()),
            );
            if exact_state && let Some(boundary) = checkpoint_boundary {
                let checkpoint_prefill = self.prefill_suffix(
                    session_id,
                    &request.prompt_token_ids[suffix_start..boundary],
                    None,
                    false,
                    cache_operation_deadline,
                    request.cancellation,
                )?;
                prefill_chunk_count = prefill_chunk_count
                    .saturating_add(checkpoint_prefill.chunk_count)
                    .saturating_sub(1);
                scheduler_batch_size_max =
                    scheduler_batch_size_max.max(checkpoint_prefill.max_batch_size);
                runtime_lock_wait_ms += checkpoint_prefill.runtime_lock_wait_ms;
                runtime_lock_hold_ms += checkpoint_prefill.runtime_lock_hold_ms;
                runtime_lock_acquires =
                    runtime_lock_acquires.saturating_add(checkpoint_prefill.chunk_count);

                if self.enqueue_exact_state_record_at_tokens(
                    session_id,
                    request.ids,
                    request.prompt_token_ids[..boundary].to_vec(),
                    "shared_prefill_checkpoint",
                ) {
                    record.resident_enqueued_checkpoints =
                        record.resident_enqueued_checkpoints.saturating_add(1);
                }
                suffix_start = boundary;
            }

            let suffix_outcome = self.prefill_suffix(
                session_id,
                &request.prompt_token_ids[suffix_start..],
                request.sampling.enabled.then_some(request.sampling),
                true,
                cache_operation_deadline,
                request.cancellation,
            )?;
            prompt_prefill_sample =
                Some(suffix_outcome.predicted.ok_or_else(|| {
                    OpenAiError::backend("deferred suffix prefill was not sampled")
                })?);
            prefill_chunk_count = if prefill_chunk_count == 1 {
                suffix_outcome.chunk_count
            } else {
                prefill_chunk_count.saturating_add(suffix_outcome.chunk_count)
            };
            scheduler_batch_size_max = scheduler_batch_size_max.max(suffix_outcome.max_batch_size);
            runtime_lock_wait_ms += suffix_outcome.runtime_lock_wait_ms;
            runtime_lock_hold_ms += suffix_outcome.runtime_lock_hold_ms;
            runtime_lock_acquires =
                runtime_lock_acquires.saturating_add(suffix_outcome.chunk_count);

            if exact_state {
                // The final chunk samples the prompt's next token but leaves
                // the native session at the full-prompt boundary. Exporting
                // that state is useful for growing prompts, but is strictly
                // best-effort after the preferred shared checkpoint above.
                if self.enqueue_exact_state_record_at_tokens(
                    session_id,
                    request.ids,
                    request.prompt_token_ids.to_vec(),
                    "final_prefill_state",
                ) {
                    record.resident_enqueued_checkpoints =
                        record.resident_enqueued_checkpoints.saturating_add(1);
                }
            } else {
                let scheduler_backend = self.clone();
                let scheduler_session_id = session_id.to_string();
                let scheduler_ids = request.ids.clone();
                let emit_debug = self.telemetry.is_debug_enabled();
                let record_outcome = self.iteration_scheduler.execute_runtime_timed_bounded(
                    "feature-kv-record",
                    request.ids.request_id_string(),
                    cache_operation_deadline,
                    request.cancellation,
                    move |runtime| {
                        let record = scheduler_backend.record_and_evict_kv(
                            runtime,
                            &scheduler_session_id,
                            &scheduler_ids,
                            record_prefill_tokens.as_ref(),
                            restored_prefill,
                            restored_prefill_tokens < record_prefill_tokens.len(),
                        );
                        Ok((record, emit_debug.then(|| runtime.session_stats())))
                    },
                )?;
                runtime_lock_wait_ms += record_outcome.runtime_lock_wait_ms;
                runtime_lock_hold_ms += record_outcome.runtime_lock_hold_ms;
                runtime_lock_acquires = runtime_lock_acquires.saturating_add(1);
                (record, runtime_sessions_after) = record_outcome.value;
            }
        }
        if prompt_prefill_sample.is_some() {
            cache_stats.suffix_prefill_tokens = saturating_u32(
                request
                    .prompt_token_ids
                    .len()
                    .saturating_sub(restored_prefill_tokens),
            );
        }
        let prompt_tail_tokens = usize::from(prompt_prefill_sample.is_some());
        cache_stats.prompt_ms = prefill_timer.elapsed_ms();
        if self.telemetry.is_debug_enabled() {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(
                    request
                        .prompt_token_ids
                        .len()
                        .saturating_sub(1)
                        .saturating_add(prompt_tail_tokens)
                ),
            );
            attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunk_count),
            );
            attrs.insert(
                "skippy.scheduler.prefill_batch_size_max".to_string(),
                json!(scheduler_batch_size_max),
            );
            attrs.insert(
                "skippy.kv.restored_prefill".to_string(),
                json!(restored_prefill),
            );
            attrs.insert(
                "skippy.kv.restored_prefill_tokens".to_string(),
                json!(restored_prefill_tokens),
            );
            attrs.insert(
                "skippy.kv.prefill_suffix_tokens".to_string(),
                json!(
                    request
                        .prompt_token_ids
                        .len()
                        .saturating_sub(1)
                        .saturating_sub(restored_prefill_tokens)
                        .saturating_add(prompt_tail_tokens)
                ),
            );
            attrs.insert(
                "skippy.kv.recorded_pages".to_string(),
                json!(record.resident_recorded_pages),
            );
            attrs.insert(
                "skippy.kv.enqueued_checkpoint_pages".to_string(),
                json!(record.resident_enqueued_checkpoints),
            );
            insert_resident_capacity_attrs(&mut attrs, &capacity);
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_before",
                runtime_sessions_before
                    .as_ref()
                    .expect("debug session snapshot collected"),
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                runtime_sessions_after
                    .as_ref()
                    .expect("debug session snapshot collected"),
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
        }
        self.telemetry.emit(
            "stage.openai_kv_record_decision",
            proactive_eviction_attrs(
                record.proactive_eviction_status,
                record.proactive_eviction_error_kind,
                record.proactive_eviction_target_tokens,
                record.proactive_evicted_entries,
                record.proactive_evicted_tokens,
            ),
        );
        let mut capacity_attrs = self.openai_attrs(request.ids);
        insert_resident_capacity_attrs(&mut capacity_attrs, &capacity);
        self.telemetry
            .emit("stage.openai_kv_capacity_decision", capacity_attrs);
        if !capacity.admitted {
            return Err(resident_capacity_admission_error(&capacity));
        }
        if let Some(error) = record.proactive_eviction_error {
            return Err(openai_backend_error(error));
        }
        Ok(prompt_prefill_sample)
    }

    fn deferred_exact_checkpoint_identity(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        prefill_tokens: &[i32],
        restored_prefill_tokens: usize,
    ) -> Option<PrefillKvIdentity> {
        let kv = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state())?;
        let base = self.local_kv_message_base(session_id, request.ids);
        let shared_checkpoint =
            kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
        let chat_checkpoint = request
            .recurrent_cache_prefix_token_ids
            .filter(|checkpoint_tokens| {
                !checkpoint_tokens.is_empty()
                    && checkpoint_tokens.len() <= prefill_tokens.len()
                    && prefill_tokens.starts_with(checkpoint_tokens)
            })
            .map(|checkpoint_tokens| {
                kv.prefill_identity(&self.config, &base, 0, checkpoint_tokens)
            });
        let shared_boundary = shared_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let chat_boundary = chat_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let selected_boundary = select_deferred_checkpoint_boundary(
            shared_boundary,
            chat_boundary,
            restored_prefill_tokens,
            prefill_tokens.len().saturating_add(1),
        );
        if selected_boundary == shared_boundary {
            shared_checkpoint
        } else {
            chat_checkpoint
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_or_record_kv_on_runtime(
        &self,
        runtime: &mut RuntimeState,
        cache_operation: &CacheRuntimeContext,
        ids: &OpenAiGenerationIds,
        session_id: &str,
        prefill_tokens: &[i32],
        lookup_identities: &[PrefillKvIdentity],
        kv_identity_ms: f64,
        recurrent_cache_prefix_token_ids: Option<&[i32]>,
        max_tokens: u32,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<KvRestoreOutcome> {
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        let emit_debug = self.telemetry.is_debug_enabled();
        let runtime_sessions_before = emit_debug.then(|| runtime.session_stats());
        let (restored_prefill, restored_prefill_tokens, protected_resident_seq_id) =
            if let Some(kv) = self.kv.as_ref() {
                cache_stats.status = "miss";
                self.lookup_and_restore_kv(
                    kv,
                    runtime,
                    session_id,
                    ids,
                    prefill_tokens,
                    lookup_identities,
                    kv_identity_ms,
                    cache_stats,
                )
            } else {
                (false, 0, None)
            };
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        let capacity = if let Some(kv) = self.kv.as_ref() {
            let decode_batch_tokens = u64::from(self.config.n_batch.unwrap_or(2048));
            let target_free_tokens =
                decode_batch_tokens.saturating_add(u64::from(max_tokens).min(decode_batch_tokens));
            let request_tokens = if kv.payload == StagePrefixCachePayload::ResidentKv {
                prefill_tokens
                    .len()
                    .saturating_sub(restored_prefill_tokens)
                    .saturating_add(usize::from(max_tokens > 0)) as u64
            } else {
                prefill_tokens.len() as u64
            };
            kv.admit_resident_capacity(
                runtime,
                session_id,
                request_tokens,
                decode_batch_tokens,
                target_free_tokens,
                protected_resident_seq_id,
            )
            .map_err(openai_backend_error)?
        } else {
            crate::kv_integration::ResidentCapacityDecision {
                admitted: true,
                ..crate::kv_integration::ResidentCapacityDecision::default()
            }
        };
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        if !capacity.admitted {
            return Ok(KvRestoreOutcome {
                runtime_sessions_before,
                runtime_sessions_after: emit_debug.then(|| runtime.session_stats()),
                restored_prefill: false,
                restored_prefill_tokens: 0,
                capacity,
                record: KvRecordResult::default(),
                prompt_prefill_sample: None,
            });
        }
        let deferred_suffix_prefill =
            deferred_suffix_prefill_enabled(self.kv.as_ref().map(|kv| kv.payload), max_tokens);
        if deferred_suffix_prefill {
            runtime
                .ensure_session_active(session_id)
                .map_err(openai_backend_error)?;
            cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
            cache_stats.suffix_prefill_tokens =
                saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
            return Ok(KvRestoreOutcome {
                runtime_sessions_before,
                runtime_sessions_after: emit_debug.then(|| runtime.session_stats()),
                restored_prefill,
                restored_prefill_tokens,
                capacity,
                record: KvRecordResult::default(),
                prompt_prefill_sample: None,
            });
        }
        let prompt_prefill_sample = None;
        let mut decoded_prefill_suffix = false;
        let cache_prefill_chunk_tokens = usize::try_from(self.config.n_ubatch.unwrap_or(256))
            .unwrap_or(2_048)
            .clamp(1, 2_048);
        if restored_prefill_tokens < prefill_tokens.len() {
            decoded_prefill_suffix = true;
            if let Some(checkpoint_tokens) =
                recurrent_cache_prefix_token_ids.filter(|checkpoint_tokens| {
                    !checkpoint_tokens.is_empty()
                        && checkpoint_tokens.len() <= prefill_tokens.len()
                        && prefill_tokens.starts_with(checkpoint_tokens)
                        && restored_prefill_tokens < checkpoint_tokens.len()
                })
            {
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &checkpoint_tokens[restored_prefill_tokens..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
                ensure_cache_operation_active(runtime, session_id, cache_operation)?;
                // This branch is the max_tokens=0 prefill-only fallback. The
                // chat-prefix state must be exported before this same runtime
                // operation advances through the remaining suffix. A detached
                // command cannot run until this operation releases the scheduler,
                // by which point the canonical-position guard would reject the
                // earlier checkpoint. Generation requests use the split, detached
                // FIFO checkpoint path in `restore_or_record_kv` instead.
                let _ = self.record_exact_state_at_tokens(
                    runtime,
                    session_id,
                    ids,
                    checkpoint_tokens,
                    "chat_prefix_checkpoint",
                );
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &prefill_tokens[checkpoint_tokens.len()..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
            } else if let Some(kv) = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state()) {
                // Recurrent and full-state payloads cannot reconstruct a shorter
                // shared prefix from the state at the end of the request. Stop
                // at the near-tail grid boundary while prefilling and snapshot
                // native state there; the final exact state is recorded by
                // `record_and_evict_kv` below. One checkpoint bounds both the
                // extra prefill split and the very large recurrent-state export.
                let base = self.local_kv_message_base(session_id, ids);
                let checkpoint =
                    kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
                if let Some(identity) = checkpoint.filter(|identity| {
                    identity.identity.token_count as usize > restored_prefill_tokens
                }) {
                    let boundary = identity.identity.token_count as usize;
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[restored_prefill_tokens..boundary],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                    ensure_cache_operation_active(runtime, session_id, cache_operation)?;
                    kv.record_exact_state(runtime, session_id, &identity)
                        .map_err(openai_backend_error)?;
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[boundary..],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                } else {
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[restored_prefill_tokens..],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                }
            } else {
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &prefill_tokens[restored_prefill_tokens..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
            }
        }
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
        cache_stats.suffix_prefill_tokens =
            saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
        let record = self.record_and_evict_kv(
            runtime,
            session_id,
            ids,
            prefill_tokens,
            restored_prefill,
            decoded_prefill_suffix,
        );
        let runtime_sessions_after = emit_debug.then(|| runtime.session_stats());
        Ok(KvRestoreOutcome {
            runtime_sessions_before,
            runtime_sessions_after,
            restored_prefill,
            restored_prefill_tokens,
            capacity,
            record,
            prompt_prefill_sample,
        })
    }

    fn configure_chat_sampling(
        &self,
        session_id: &str,
        metadata: &str,
        prompt_token_count: usize,
        sampling: Option<&SamplingConfig>,
    ) -> OpenAiResult<()> {
        let scheduler_session_id = session_id.to_string();
        let scheduler_metadata = metadata.to_string();
        let scheduler_sampling = sampling.cloned();
        self.iteration_scheduler
            .execute_runtime("feature-chat-sampling", move |runtime| {
                runtime
                    .configure_chat_sampling(
                        &scheduler_session_id,
                        &scheduler_metadata,
                        prompt_token_count as u64,
                        scheduler_sampling.as_ref(),
                    )
                    .map_err(openai_backend_error)
            })
    }

    fn configure_chat_sampling_if_needed(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        chat_sampling_configured: bool,
    ) -> OpenAiResult<()> {
        if let Some(metadata) = (!chat_sampling_configured)
            .then_some(request.chat_sampling_metadata)
            .flatten()
        {
            self.configure_chat_sampling(
                session_id,
                metadata,
                request.prompt_token_ids.len(),
                request.sampling.enabled.then_some(request.sampling),
            )?;
        }
        Ok(())
    }

    fn prepare_decode_state(
        &self,
        request: &mut LocalGeneration<'_>,
        session_id: &str,
        prompt_prefill_sample: Option<i32>,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<DecodeState> {
        let mut decoded_tokens = 0usize;
        let mut current = *request
            .prompt_token_ids
            .last()
            .expect("checked non-empty prompt");
        let mut stopped = false;
        let mut generated_token_ids = Vec::new();
        let mut pending_linear_proposal_tokens = Vec::new();
        if let Some(predicted) = prompt_prefill_sample {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                return Err(OpenAiError::backend("request cancelled"));
            }
            current = predicted;
            decoded_tokens += 1;
            generated_token_ids.push(current);
            pending_linear_proposal_tokens.push(current);
            stopped = emit_token(current)? == TokenControl::Stop;
        }
        let hook_request = request.hook_request.take();
        let hook_runtime = request.hook_runtime.take();
        let generation_hooks_active =
            self.generation_hooks_active(&hook_request, hook_runtime.as_ref());
        let linear_proposal_enabled = self.linear_proposal_ingress.is_some()
            && !request.native_mtp_enabled
            && !generation_hooks_active
            && greedy_linear_proposal_admitted(request.sampling, request.chat_sampling_metadata);
        let linear_proposal_max_tokens = if linear_proposal_enabled {
            let scheduler_session_id = session_id.to_string();
            self.iteration_scheduler
                .execute_runtime("linear-proposal-admission", move |runtime| {
                    runtime
                        .admit_session_batch_size(&scheduler_session_id)
                        .map_err(openai_backend_error)
                })?
                .saturating_sub(1)
        } else {
            0
        };
        let linear_context_tokens = (linear_proposal_max_tokens > 0).then(|| {
            let mut tokens = request.prompt_token_ids.to_vec();
            if decoded_tokens > 0 {
                tokens.push(current);
            }
            tokens
        });
        if linear_proposal_max_tokens == 0 {
            pending_linear_proposal_tokens.clear();
        }
        Ok(DecodeState {
            decoded_tokens,
            current,
            generated_token_ids,
            stopped,
            runtime_lock_wait_ms: 0.0,
            runtime_lock_wait_max_ms: 0.0,
            runtime_lock_hold_ms: 0.0,
            runtime_lock_hold_max_ms: 0.0,
            runtime_lock_acquires: 0,
            runtime_sessions_before: None,
            runtime_sessions_after: None,
            hook_request,
            hook_runtime,
            generation_hooks_active,
            linear_proposal_max_tokens,
            linear_context_tokens,
            pending_linear_proposal_tokens,
            emit_token_debug: self.telemetry.is_debug_enabled(),
            native_mtp_options: NativeMtpDecodeOptions::from_config(request.speculative),
            native_mtp: NativeMtpVerifier::default(),
            native_mtp_span_admitted: request.native_mtp_enabled
                && !generation_hooks_active
                && greedy_linear_proposal_admitted(
                    request.sampling,
                    request.chat_sampling_metadata,
                ),
            post_prefill_hook_checked: false,
            last_mid_generation_hook_at: None,
            direct_iteration_channel: None,
        })
    }

    /// Drives feature-specific proposal/checkpoint state while the iteration
    /// scheduler worker remains the sole owner of target runtime execution.
    /// This is not a fallback serving scheduler: native MTP, linear proposals,
    /// hooks, and cache operations submit their target work to that worker.
    fn run_scheduler_feature_loop(
        &self,
        request: &mut LocalGeneration<'_>,
        session_id: &str,
        prompt_prefill_sample: Option<i32>,
        cache_stats: &mut GenerationCacheStats,
        receipt_cancelled: &mut bool,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Duration> {
        let decode_timer = PhaseTimer::start();
        let mut state =
            self.prepare_decode_state(request, session_id, prompt_prefill_sample, emit_token)?;
        let recurrent_checkpoint_required = self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent);
        let resume_scheduler_decode = prompt_prefill_sample.is_some()
            && !state.stopped
            && state.decoded_tokens < request.max_tokens as usize
            && !recurrent_checkpoint_required
            && self.linear_proposal_ingress.is_none()
            && self.draft.is_none()
            && !request.native_mtp_enabled
            && !state.generation_hooks_active;
        if resume_scheduler_decode {
            let initial_generated_tokens = state.generated_token_ids.clone();
            let stats = self.iteration_scheduler.resume_generation(
                ScheduledResumeRequest {
                    id: session_id,
                    prompt_tokens: request.prompt_token_ids,
                    generated_tokens: &initial_generated_tokens,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling.enabled.then_some(request.sampling),
                    cancellation: request.cancellation,
                },
                |token| {
                    state.current = token;
                    state.decoded_tokens = state.decoded_tokens.saturating_add(1);
                    state.generated_token_ids.push(token);
                    emit_token(token)
                },
            )?;
            cache_stats.predicted_ms = stats.predicted_ms;
            let model_generation_elapsed =
                self.emit_decode_summary(request, &mut state, cache_stats, decode_timer)?;
            if state.generated_token_ids.len() > 1 {
                let _ = self.record_post_decode_exact_state(request, session_id, &state);
            }
            return Ok(model_generation_elapsed);
        }
        // A recurrent request needs one serial decode before proposals can
        // advance the native session beyond the prompt boundary. The serial
        // attempt is the gate: a short prompt may be below the policy's
        // minimum checkpoint size, in which case recording is correctly
        // skipped but proposals must not remain disabled for the rest of the
        // request.
        let mut first_post_decode_checkpoint_attempted = initial_recurrent_checkpoint_attempted(
            recurrent_checkpoint_required,
            prompt_prefill_sample,
        );
        while !state.stopped && state.decoded_tokens < request.max_tokens as usize {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                *receipt_cancelled = true;
                break;
            }
            let linear_progress = if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                self.try_execute_linear_proposal(request, session_id, &mut state, emit_token)?
            } else {
                // A proposal can consume the final prompt token and commit
                // multiple target tokens in one call. Give recurrent caches
                // one serial decode first so the exact full-prompt boundary
                // is published before that proposal advances the session.
                LinearProposalProgress::NotUsed
            };
            match linear_progress {
                LinearProposalProgress::Continue => continue,
                LinearProposalProgress::Stop => break,
                LinearProposalProgress::NotUsed => {}
            }
            // A batched MTP span commits multiple tokens from one forward. It
            // shares the recurrent-checkpoint gate with linear proposals for
            // the same reason: it can advance the session past the prompt
            // boundary before that boundary has been published.
            if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                match self
                    .try_execute_native_mtp_span(request, session_id, &mut state, emit_token)?
                {
                    NativeMtpSpanProgress::Continue => continue,
                    NativeMtpSpanProgress::Stop => break,
                    NativeMtpSpanProgress::NotUsed => {}
                }
            }
            let control = self.decode_one_token(request, session_id, &mut state, emit_token)?;
            if !first_post_decode_checkpoint_attempted && state.generated_token_ids.len() == 1 {
                first_post_decode_checkpoint_attempted = true;
                self.record_post_decode_exact_state(request, session_id, &state);
            }
            if state.linear_proposal_max_tokens > 0 {
                state.pending_linear_proposal_tokens.push(state.current);
            }
            if control == TokenControl::Stop {
                break;
            }
        }
        let model_generation_elapsed =
            self.emit_decode_summary(request, &mut state, cache_stats, decode_timer)?;
        // The first-token checkpoint names the prompt boundary. Avoid
        // exporting the same recurrent state twice for a one-token response;
        // longer responses still get their final exact boundary recorded.
        if state.generated_token_ids.len() > 1 {
            let _ = self.record_post_decode_exact_state(request, session_id, &state);
        }
        Ok(model_generation_elapsed)
    }
}

pub(super) trait NativeMtpRuntime {
    fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)>;
}

impl NativeMtpRuntime for RuntimeState {
    fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)> {
        RuntimeState::decode_sampled_mtp(self, session_id, token_id, sampling, max_draft_tokens)
    }
}

pub(super) fn decode_native_mtp(
    runtime: &mut impl NativeMtpRuntime,
    session_id: &str,
    token_id: i32,
    sampling: Option<&SamplingConfig>,
    max_draft_tokens: usize,
) -> OpenAiResult<(i32, Option<NativeMtpDraft>)> {
    let (predicted, draft) = runtime
        .decode_sampled_mtp(session_id, token_id, sampling, max_draft_tokens)
        .map_err(openai_backend_error)?;
    Ok((
        predicted,
        draft.map(|draft| NativeMtpDraft {
            tokens: draft.token_ids,
            proposal_compute_us: draft.proposal_compute_us,
        }),
    ))
}

#[cfg(test)]
#[test]
fn resident_capacity_attrs_report_bounded_rejection_evidence() {
    let mut attrs = BTreeMap::new();
    insert_resident_capacity_attrs(
        &mut attrs,
        &crate::kv_integration::ResidentCapacityDecision {
            enabled: true,
            capacity_known: true,
            admitted: false,
            capacity_tokens: 100,
            active_tokens: 60,
            pinned_tokens: 30,
            request_tokens: 8,
            minimum_free_tokens: 5,
            target_free_tokens: 10,
            projected_free_tokens: 2,
            admission_deficit_tokens: 3,
            ..crate::kv_integration::ResidentCapacityDecision::default()
        },
    );

    assert_eq!(
        attrs.get(attr_key::KV_CAPACITY_STATUS),
        Some(&json!("rejected"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_CAPACITY_ADMISSION_DEFICIT_TOKENS),
        Some(&json!(3))
    );
    assert!(!attrs.contains_key(attr_key::REQUEST_ID));
    assert!(!attrs.contains_key(attr_key::SESSION_ID));
}

#[cfg(test)]
#[test]
fn default_local_text_serving_selects_iteration_scheduler() {
    assert!(scheduler_generation_driver_eligible(
        128, false, false, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        0, false, false, false, false, "disabled", false,
    ));
}

#[cfg(test)]
#[test]
fn resident_kv_suffix_prefill_stays_inside_the_cache_operation() {
    assert!(!deferred_suffix_prefill_enabled(
        Some(StagePrefixCachePayload::ResidentKv),
        1,
    ));
}

#[cfg(test)]
#[test]
fn recurrent_suffix_prefill_requires_a_nonzero_generation_budget() {
    assert!(deferred_suffix_prefill_enabled(
        Some(StagePrefixCachePayload::KvRecurrent),
        1,
    ));
    assert!(deferred_suffix_prefill_enabled(
        Some(StagePrefixCachePayload::FullState),
        1,
    ));
    assert!(!deferred_suffix_prefill_enabled(
        Some(StagePrefixCachePayload::KvRecurrent),
        0,
    ));
    assert!(!deferred_suffix_prefill_enabled(None, 1));
}

#[cfg(test)]
#[test]
fn deferred_suffix_split_respects_restore_and_prompt_boundaries() {
    assert_eq!(deferred_suffix_split(0, 10, Some(4)), (0, Some(4)));
    assert_eq!(deferred_suffix_split(4, 10, Some(4)), (4, None));
    assert_eq!(deferred_suffix_split(8, 10, Some(4)), (8, None));
    assert_eq!(deferred_suffix_split(4, 10, Some(10)), (4, None));
    assert_eq!(deferred_suffix_split(99, 10, Some(4)), (10, None));
}

#[cfg(test)]
#[test]
fn deferred_checkpoint_prefers_shared_grid_boundary_over_chat_boundary() {
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), Some(6), 0, 10),
        Some(8)
    );
    assert_eq!(
        select_deferred_checkpoint_boundary(None, Some(6), 0, 10),
        Some(6)
    );
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(4), Some(6), 4, 10),
        Some(6)
    );
}

#[cfg(test)]
#[test]
fn deferred_exact_prefill_satisfies_the_recurrent_checkpoint_gate() {
    assert!(initial_recurrent_checkpoint_attempted(true, Some(42)));
    assert!(!initial_recurrent_checkpoint_attempted(true, None));
    assert!(initial_recurrent_checkpoint_attempted(false, None));
}

#[cfg(test)]
#[test]
fn speculative_and_extended_serving_paths_select_the_scheduler_feature_driver() {
    assert!(!scheduler_generation_driver_eligible(
        128, true, false, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, true, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, false, true, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128,
        false,
        false,
        false,
        true,
        "native-mtp",
        false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128,
        false,
        false,
        false,
        false,
        "ngram-cache",
        false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, false, false, false, "disabled", true,
    ));
}

#[cfg(test)]
pub(in crate::frontend) fn native_mtp_dispatch_counts_for_test() -> (usize, usize) {
    struct FakeNativeMtpRuntime {
        sampled_calls: usize,
    }

    impl NativeMtpRuntime for FakeNativeMtpRuntime {
        fn decode_sampled_mtp(
            &mut self,
            _session_id: &str,
            _token_id: i32,
            _sampling: Option<&SamplingConfig>,
            _max_draft_tokens: usize,
        ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)> {
            self.sampled_calls += 1;
            Ok((7, None))
        }
    }

    let mut runtime = FakeNativeMtpRuntime { sampled_calls: 0 };
    let (predicted, draft) =
        decode_native_mtp(&mut runtime, "test", 0, None, 1).expect("sampled MTP dispatch");
    assert_eq!(predicted, 7);
    assert!(draft.is_none());
    (runtime.sampled_calls, 0)
}
