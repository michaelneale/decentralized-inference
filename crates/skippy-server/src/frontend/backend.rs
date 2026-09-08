use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::admission::GenerationTokenBudgetRequest;
use crate::frontend::admission::GenerationTokenReservation;
use crate::frontend::generation::ChatOutputStreamParser;
use crate::frontend::generation::GeneratedText;
use crate::frontend::generation::GenerationActiveWorkReservation;
use crate::frontend::generation::GenerationAdmissionClaim;
use crate::frontend::generation::GenerationAdmissionScheduling;
use crate::frontend::generation::GenerationAdmissionWork;
use crate::frontend::generation::GenerationConcurrencyController;
use crate::frontend::generation::GenerationConcurrencyDecision;
use crate::frontend::generation::GenerationConcurrencyObservation;
use crate::frontend::generation::GenerationConcurrencyPermit;
use crate::frontend::generation::GenerationServiceEstimator;
use crate::frontend::generation::GenerationSessionLockEntry;
use crate::frontend::generation::GenerationStream;
use crate::frontend::generation::GenerationStreamEvent;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::OpenAiCacheHints;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::PreparedTextPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::apply_reasoning_visibility;
use crate::frontend::generation::chat_output_parser_required;
use crate::frontend::generation::chat_response_from_generated_text;
use crate::frontend::generation::completion_response_from_generated_text;
use crate::frontend::generation::ensure_requested_model;
use crate::frontend::generation::generation_event_to_chat_chunk;
use crate::frontend::generation::generation_event_to_completion_chunk;
use crate::frontend::generation::generation_predicted_wait_error;
use crate::frontend::generation::generation_queue_timeout_error;
use crate::frontend::generation::template_exposes_reasoning;
use crate::frontend::request::{
    apply_chat_request_defaults, apply_completion_request_defaults, chat_sampling_config,
    chat_template_options, completion_sampling_config, ensure_chat_runtime_features_supported,
    ensure_completion_runtime_features_supported,
};
use crate::runtime_state::RuntimeSessionStats;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream;
use openai_frontend::ChatCompletionOutcome;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::ChatCompletionResponse;
use openai_frontend::ChatCompletionStream;
use openai_frontend::ChatExchangeRoute;
use openai_frontend::CompletionRequest;
use openai_frontend::CompletionResponse;
use openai_frontend::CompletionStream;
use openai_frontend::ModelObject;
use openai_frontend::OpenAiBackend;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiRequestContext;
use openai_frontend::OpenAiResult;
use openai_frontend::TerminalGuard;
use openai_frontend::TerminalGuardedChatStream;
use openai_frontend::apply_chat_hook_outcome;
use openai_frontend::capsule_id_is_valid;
use openai_frontend::chat_mesh_hooks_enabled;
use serde_json::Value;
use serde_json::json;
use skippy_metrics::attr as attr_key;
use skippy_runtime::SamplingConfig;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tokio::sync::TryAcquireError;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::task;

fn request_cancelled_error() -> OpenAiError {
    OpenAiError::cancelled("request cancelled")
}

/// How long a full SSE event channel is treated as merely backed up before
/// the generation worker gives up on it and frees its execution lane.
///
/// Deliberately its own value rather than an alias of
/// the configured generation admission timeout: admission queueing and stream-stall
/// tolerance are unrelated policies, and retuning one must not silently
/// retune the other. It bounds a single send, not a whole generation.
const STREAM_SEND_STALL_TIMEOUT: Duration = Duration::from_secs(10);

/// Forwards generation events to the SSE channel without letting a consumer
/// that has stopped draining pin the generation worker, and the execution
/// lane it holds, forever.
///
/// `mpsc::Sender::blocking_send` waits indefinitely for buffer space. A
/// consumer that stops draining without the channel ever being dropped --
/// a client gone before the server's own disconnect detection notices, e.g.
/// behind a proxy that doesn't propagate the close -- can then pin the
/// generation worker, and the execution lane it holds, forever even after
/// the request is cancelled. This races each send against cancellation and
/// against `stall_timeout` via `tokio::select!` instead of polling, so a
/// consumer draining normally is woken the instant space appears rather than
/// after up to a 20 ms poll tick.
struct StreamEventSender {
    tx: mpsc::Sender<OpenAiResult<GenerationStreamEvent>>,
    runtime: tokio::runtime::Handle,
    stall_timeout: Duration,
    /// Request identifier carried for diagnostics so a stalled or dropped
    /// consumer can be attributed to the exact request that held (and then
    /// freed) an execution lane. `send_terminal` has no request context of
    /// its own, so the id is captured once at construction.
    request_id: String,
    /// Set once the receiver is gone or has stayed full past the stall
    /// timeout. Nothing further can reach the client, so later frames are
    /// dropped rather than waited on again.
    receiver_unreachable: AtomicBool,
    /// Structured telemetry sink for the stall/drop diagnostics. `skippy-server`
    /// routes operator-facing signal through `Telemetry`, not a logging facade,
    /// so a freed execution lane is correlated by `request_id` and observable
    /// without scraping stderr.
    telemetry: Telemetry,
}

impl StreamEventSender {
    fn new(
        tx: mpsc::Sender<OpenAiResult<GenerationStreamEvent>>,
        runtime: tokio::runtime::Handle,
        stall_timeout: Duration,
        request_id: String,
        telemetry: Telemetry,
    ) -> Self {
        Self {
            tx,
            runtime,
            stall_timeout,
            request_id,
            receiver_unreachable: AtomicBool::new(false),
            telemetry,
        }
    }

    /// Emit a structured "execution lane freed" event when a consumer is found
    /// gone or stalled. `outcome` names the failure (dropped vs stalled) and
    /// `frame_kind` names which send path hit it (an in-flight event vs a
    /// terminal frame), so an operator can tell a client cancellation apart
    /// from a stalled consumer pinning a lane without log scraping.
    fn emit_lane_freed(&self, outcome: &str, frame_kind: &str) {
        let mut attrs = BTreeMap::new();
        attrs.insert(attr_key::REQUEST_ID.to_string(), json!(self.request_id));
        attrs.insert("skippy.stream.outcome".to_string(), json!(outcome));
        attrs.insert("skippy.stream.frame_kind".to_string(), json!(frame_kind));
        attrs.insert(
            "skippy.stream.stall_timeout_ms".to_string(),
            json!(self.stall_timeout.as_millis() as u64),
        );
        self.telemetry.emit("stage.openai_stream_lane_freed", attrs);
    }

    /// Mark the receiver unreachable and free the request's execution lane.
    /// Called once nothing further sent on this channel could possibly reach
    /// the client: the receiver dropped, or it stayed full past
    /// `stall_timeout`.
    fn mark_receiver_unreachable(&self, context: &OpenAiRequestContext) {
        self.receiver_unreachable.store(true, Ordering::Release);
        context.cancel();
    }

    /// Send one in-flight event (from the `on_text_chunk` callback). Checks
    /// cancellation first so an already-cancelled request returns
    /// immediately and deterministically, matching the pre-existing
    /// early-return semantics.
    fn send(
        &self,
        event: OpenAiResult<GenerationStreamEvent>,
        context: &OpenAiRequestContext,
    ) -> Result<(), OpenAiError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(OpenAiError::backend("stream receiver unreachable"));
        }
        if context.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let event = match self.tx.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(event)) => event,
            Err(TrySendError::Closed(_)) => {
                self.emit_lane_freed("receiver_dropped", "in_flight");
                self.mark_receiver_unreachable(context);
                return Err(OpenAiError::backend("stream receiver dropped"));
            }
        };
        let cancellation = context.cancellation_token();
        // `tokio::time::sleep` needs an entered runtime the instant it is
        // called, not just when polled, so it must be constructed inside the
        // `block_on`-driven future rather than before it.
        self.runtime.block_on(async {
            let send = self.tx.send(event);
            let sleep = tokio::time::sleep(self.stall_timeout);
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(request_cancelled_error()),
                result = send => match result {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        self.emit_lane_freed("receiver_dropped", "in_flight");
                        self.mark_receiver_unreachable(context);
                        Err(OpenAiError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "in_flight");
                    self.mark_receiver_unreachable(context);
                    Err(OpenAiError::backend(
                        "stream receiver stalled without draining",
                    ))
                }
            }
        })
    }

    /// Send one terminal frame -- everything emitted after generation
    /// finishes: the cancellation error, a parser-finish error, the backend
    /// error, usage, and `Done`.
    ///
    /// Deliberately does **not** consult cancellation. These are exactly the
    /// frames the frontend lifecycle needs when the request *was* cancelled:
    /// on `main`, `blocking_send` delivered them unconditionally. Skipping
    /// them for a cancelled-but-still-live receiver would flip
    /// `stream_lifecycle`'s terminal classification -- an `Err` frame drives
    /// `lifecycle.failed(error)`, which marks `backend_error` and yields
    /// `StreamDropOutcome::BackendError`/`StreamTerminal`; without it,
    /// `drop_outcome()` falls through to `StreamDropOutcome::Cancelled`
    /// instead.
    ///
    /// It does still refuse to wait on a receiver already proven
    /// unreachable: the in-flight send that got us here may already have
    /// waited out `stall_timeout` once, and waiting again would double the
    /// execution lane's hold to `2 * stall_timeout`.
    fn send_terminal(&self, event: OpenAiResult<GenerationStreamEvent>) -> Result<(), OpenAiError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(OpenAiError::backend("stream receiver unreachable"));
        }
        // See the matching comment in `send`: the sleep future must be
        // constructed inside the entered runtime `block_on` provides.
        self.runtime.block_on(async {
            let send = self.tx.send(event);
            let sleep = tokio::time::sleep(self.stall_timeout);
            tokio::select! {
                result = send => match result {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        self.emit_lane_freed("receiver_dropped", "terminal");
                        self.receiver_unreachable.store(true, Ordering::Release);
                        Err(OpenAiError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "terminal");
                    self.receiver_unreachable.store(true, Ordering::Release);
                    Err(OpenAiError::backend(
                        "stream receiver stalled without draining",
                    ))
                }
            }
        })
    }
}

fn should_emit_stream_usage(request_include_usage: bool, context: &OpenAiRequestContext) -> bool {
    request_include_usage || context.observes_stream_usage()
}

struct GenerationSessionPermit {
    registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    key: String,
    entry: Arc<GenerationSessionLockEntry>,
    permit: Option<OwnedSemaphorePermit>,
}

impl GenerationSessionPermit {
    fn new(
        registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
        key: String,
    ) -> OpenAiResult<Self> {
        let entry = {
            let mut locks = registry
                .lock()
                .map_err(|_| OpenAiError::backend("generation session lock map poisoned"))?;
            let entry = locks
                .entry(key.clone())
                .or_insert_with(|| {
                    Arc::new(GenerationSessionLockEntry {
                        semaphore: Arc::new(Semaphore::new(1)),
                        users: AtomicUsize::new(0),
                    })
                })
                .clone();
            // Lookup and lease registration share the registry mutex with
            // cleanup, so a dropping lease cannot remove and replace this
            // entry between those two operations.
            entry.users.fetch_add(1, Ordering::AcqRel);
            entry
        };
        Ok(Self {
            registry,
            key,
            entry,
            permit: None,
        })
    }

    fn try_acquire(&mut self) -> OpenAiResult<bool> {
        match self.entry.semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                self.permit = Some(permit);
                Ok(true)
            }
            Err(TryAcquireError::NoPermits) => Ok(false),
            Err(TryAcquireError::Closed) => {
                Err(OpenAiError::backend("generation session lock closed"))
            }
        }
    }

    async fn acquire_until(
        mut self,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<Self> {
        let permit = if let Some(deadline) = deadline {
            let acquire = tokio::time::timeout_at(
                tokio::time::Instant::from_std(deadline),
                self.entry.semaphore.clone().acquire_owned(),
            );
            tokio::select! {
                result = acquire => result
                    .map_err(|_| generation_queue_timeout_error(admission_timeout))?
                    .map_err(|_| OpenAiError::backend("generation session lock closed"))?,
                () = cancellation.cancelled() => return Err(request_cancelled_error()),
            }
        } else {
            tokio::select! {
                result = self.entry.semaphore.clone().acquire_owned() => result
                    .map_err(|_| OpenAiError::backend("generation session lock closed"))?,
                () = cancellation.cancelled() => return Err(request_cancelled_error()),
            }
        };
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        self.permit = Some(permit);
        Ok(self)
    }
}

impl Drop for GenerationSessionPermit {
    fn drop(&mut self) {
        self.permit.take();
        let Ok(mut locks) = self.registry.lock() else {
            return;
        };
        if self.entry.users.fetch_sub(1, Ordering::AcqRel) == 1
            && locks
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(entry, &self.entry))
        {
            locks.remove(&self.key);
        }
    }
}

fn trusted_generation_session_key(ids: &OpenAiGenerationIds) -> Option<String> {
    ids.agent_session_trusted.then(|| ids.session_id_string())
}

#[derive(Clone)]
struct GenerationAdmissionController {
    generation_limit: Arc<GenerationConcurrencyController>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    generation_service_estimator: Arc<GenerationServiceEstimator>,
    generation_session_locks: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    generation_token_budget: Arc<GenerationTokenBudget>,
}

impl GenerationAdmissionController {
    fn for_backend(backend: &StageOpenAiBackend) -> Self {
        Self {
            generation_limit: backend.generation_limit.clone(),
            generation_queue_depth: backend.generation_queue_depth.clone(),
            generation_queue_limit: backend.generation_queue_limit,
            generation_service_estimator: backend.generation_service_estimator.clone(),
            generation_session_locks: backend.generation_session_locks.clone(),
            generation_token_budget: backend.generation_token_budget.clone(),
        }
    }

    #[cfg(test)]
    async fn acquire(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
        admission_timeout: Duration,
    ) -> OpenAiResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        self.acquire_work(
            ids,
            cancellation,
            admission_timeout,
            GenerationAdmissionWork::default(),
        )
        .await
    }

    #[cfg(test)]
    async fn acquire_work(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
        admission_timeout: Duration,
        work: GenerationAdmissionWork,
    ) -> OpenAiResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        self.acquire_scheduled_work(
            ids,
            cancellation,
            admission_timeout,
            work,
            GenerationAdmissionScheduling::default(),
        )
        .await
    }

    async fn acquire_scheduled_work(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
        admission_timeout: Duration,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> OpenAiResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        let deadline = if admission_timeout.is_zero() {
            None
        } else {
            Some(
                Instant::now()
                    .checked_add(admission_timeout)
                    .ok_or_else(|| {
                        OpenAiError::backend("generation admission deadline overflow")
                    })?,
            )
        };
        let session_permit = self
            .acquire_session_until(ids, deadline, admission_timeout, cancellation)
            .await?;

        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(generation_queue_timeout_error(admission_timeout));
        }
        let generation_permit = self
            .acquire_generation_permit_until(
                deadline,
                admission_timeout,
                cancellation,
                work,
                scheduling,
            )
            .await?;
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        Ok((generation_permit, session_permit))
    }

    async fn acquire_session_until(
        &self,
        ids: &OpenAiGenerationIds,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<Option<GenerationSessionPermit>> {
        let Some(session_key) = trusted_generation_session_key(ids) else {
            return Ok(None);
        };
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let mut session =
            GenerationSessionPermit::new(self.generation_session_locks.clone(), session_key)?;
        if session.try_acquire()? {
            return Ok(Some(session));
        }
        session
            .acquire_until(deadline, admission_timeout, cancellation)
            .await
            .map(Some)
    }

    async fn acquire_generation_permit_until(
        &self,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> OpenAiResult<GenerationAdmissionPermit> {
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let claim = self.generation_limit.admission_queue().claim_or_enqueue(
            self.generation_limit.semaphore(),
            self.generation_token_budget.clone(),
            GenerationTokenBudgetRequest::new(
                usize::try_from(work.prompt_tokens).unwrap_or(usize::MAX),
                u32::try_from(work.decode_tokens).unwrap_or(u32::MAX),
            ),
            scheduling,
            self.generation_queue_depth.clone(),
            self.generation_queue_limit,
        )?;
        match claim {
            GenerationAdmissionClaim::Acquired(resources) => Ok(GenerationAdmissionPermit {
                _lane: self.generation_limit.wrap_permit(resources.lane),
                token_budget: resources.token_budget,
                _active_work: self.generation_service_estimator.start_active(work),
                predicted_wait_ms: Some(0.0),
                demand_epoch: self.generation_limit.demand_epoch(),
                queued_at_start: self.generation_queue_depth.load(Ordering::Acquire) > 0,
                waited_for_lane: false,
                at_capacity_at_start: self.generation_limit.is_at_capacity(),
                started_at: Instant::now(),
            }),
            GenerationAdmissionClaim::Queued(lease) => {
                self.generation_limit.note_queued_demand();
                self.generation_service_estimator
                    .set_concurrency(self.generation_limit.current_limit());
                let predicted_wait_ms = self.generation_service_estimator.predicted_wait_ms();
                let queued_work = self
                    .generation_service_estimator
                    .reserve_queued(work, admission_timeout)
                    .map_err(|predicted_wait_ms| {
                        generation_predicted_wait_error(predicted_wait_ms, admission_timeout)
                    })?;
                let resources = lease
                    .acquire(
                        self.generation_limit.semaphore(),
                        admission_timeout,
                        deadline,
                        cancellation,
                    )
                    .await?;
                Ok(GenerationAdmissionPermit {
                    _lane: self.generation_limit.wrap_permit(resources.lane),
                    token_budget: resources.token_budget,
                    _active_work: queued_work.promote(),
                    predicted_wait_ms,
                    demand_epoch: self.generation_limit.demand_epoch(),
                    queued_at_start: self.generation_queue_depth.load(Ordering::Acquire) > 0,
                    waited_for_lane: true,
                    at_capacity_at_start: self.generation_limit.is_at_capacity(),
                    started_at: Instant::now(),
                })
            }
        }
    }
}

pub(in crate::frontend) struct GenerationAdmissionPermit {
    token_budget: GenerationTokenReservation,
    _lane: GenerationConcurrencyPermit,
    _active_work: GenerationActiveWorkReservation,
    predicted_wait_ms: Option<f64>,
    demand_epoch: u64,
    queued_at_start: bool,
    waited_for_lane: bool,
    at_capacity_at_start: bool,
    started_at: Instant,
}

impl GenerationAdmissionPermit {
    fn token_budget_stats(&self) -> (usize, usize) {
        (
            self.token_budget.tokens(),
            self.token_budget.active_tokens_after_reservation(),
        )
    }

    fn demand_observation(&self) -> GenerationDemandObservation {
        GenerationDemandObservation {
            demand_epoch: self.demand_epoch,
            queued_at_start: self.queued_at_start,
            waited_for_lane: self.waited_for_lane,
            at_capacity_at_start: self.at_capacity_at_start,
            started_at: self.started_at,
        }
    }
}

#[derive(Clone, Copy)]
struct GenerationDemandObservation {
    demand_epoch: u64,
    queued_at_start: bool,
    waited_for_lane: bool,
    at_capacity_at_start: bool,
    started_at: Instant,
}

impl StageOpenAiBackend {
    fn observe_generation_completed(
        &self,
        output: &GeneratedText,
        demand: GenerationDemandObservation,
    ) {
        let work = GenerationAdmissionWork::new(
            usize::try_from(output.prompt_tokens).unwrap_or(usize::MAX),
            output.completion_tokens,
        );
        self.generation_service_estimator.observe_completed(
            work,
            output.prompt_ms,
            output.predicted_ms,
        );
        let completed_tokens = work.prompt_tokens.saturating_add(work.decode_tokens);
        let executed_tokens =
            u64::from(output.suffix_prefill_tokens).saturating_add(work.decode_tokens);
        let saturated = demand.waited_for_lane
            || self.generation_limit.was_saturated_since(
                demand.demand_epoch,
                demand.queued_at_start,
                self.generation_queue_depth.load(Ordering::Acquire) > 0,
            );
        let Some(decision) =
            self.generation_limit
                .observe_completed(GenerationConcurrencyObservation {
                    completed_tokens,
                    executed_tokens,
                    latency_ms: output.prompt_ms + output.predicted_ms,
                    saturated,
                    at_capacity: demand.at_capacity_at_start,
                    started_at: demand.started_at,
                })
        else {
            return;
        };
        self.emit_generation_concurrency_decision(decision);
    }

    fn observe_generation_failed(&self) {
        if let Some(decision) = self.generation_limit.observe_failed() {
            self.emit_generation_concurrency_decision(decision);
        }
    }

    fn emit_generation_concurrency_decision(&self, decision: GenerationConcurrencyDecision) {
        self.generation_service_estimator
            .set_concurrency(decision.current_limit);
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.generation_concurrency_action".to_string(),
            json!(decision.action),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_reason".to_string(),
            json!(decision.reason),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_previous".to_string(),
            json!(decision.previous_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_current".to_string(),
            json!(decision.current_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_ceiling".to_string(),
            json!(self.generation_limit.hard_limit()),
        );
        if let Some(throughput) = decision.throughput_tokens_per_second {
            attrs.insert(
                "llama_stage.generation_throughput_tokens_per_second".to_string(),
                json!(throughput),
            );
        }
        if let Some(p95_latency_ms) = decision.p95_latency_ms {
            attrs.insert(
                "llama_stage.generation_p95_latency_ms".to_string(),
                json!(p95_latency_ms),
            );
        }
        if let Some(p95_hardware_ms_per_token) = decision.p95_hardware_ms_per_token {
            attrs.insert(
                "llama_stage.generation_p95_hardware_ms_per_token".to_string(),
                json!(p95_hardware_ms_per_token),
            );
        }
        if let Some(improvement) = decision.throughput_improvement {
            attrs.insert(
                "llama_stage.generation_throughput_improvement".to_string(),
                json!(improvement),
            );
        }
        if let Some(ratio) = decision.p95_latency_ratio {
            attrs.insert(
                "llama_stage.generation_p95_latency_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(ratio) = decision.hardware_pressure_ratio {
            attrs.insert(
                "llama_stage.generation_hardware_pressure_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(observed_requests) = decision.observed_requests {
            attrs.insert(
                "llama_stage.generation_observed_requests".to_string(),
                json!(observed_requests),
            );
        }
        if let Some(saturated_requests) = decision.saturated_requests {
            attrs.insert(
                "llama_stage.generation_saturated_requests".to_string(),
                json!(saturated_requests),
            );
        }
        self.telemetry
            .emit("stage.openai_generation_concurrency_adapt", attrs);
    }
}

fn insert_generation_admission_attrs(
    attrs: &mut BTreeMap<String, Value>,
    permit: &GenerationAdmissionPermit,
    queue_depth: usize,
    queue_capacity: usize,
) {
    attrs.insert(
        "llama_stage.generation_queue_depth".to_string(),
        json!(queue_depth),
    );
    attrs.insert(
        "skippy.scheduler.admission_waiting".to_string(),
        json!(queue_depth),
    );
    attrs.insert(
        "llama_stage.generation_queue_capacity".to_string(),
        json!(queue_capacity),
    );
    if let Some(predicted_wait_ms) = permit.predicted_wait_ms {
        attrs.insert(
            "llama_stage.generation_predicted_wait_ms".to_string(),
            json!(predicted_wait_ms),
        );
    }
}

fn generation_ids(
    cache: OpenAiCacheHints,
    agent_session_id: Option<&str>,
    context: &OpenAiRequestContext,
) -> OpenAiGenerationIds {
    let frontend_request_id = context
        .request_id()
        .map(|request_id| request_id.as_uuid().into_bytes());
    OpenAiGenerationIds::new_with_trust(
        cache,
        agent_session_id,
        context.has_trusted_agent_session(),
        frontend_request_id,
    )
}

pub(in crate::frontend) async fn run_blocking_generation_worker<T, F, P>(
    permit: P,
    context: OpenAiRequestContext,
    work: F,
) -> Result<T, task::JoinError>
where
    T: Send + 'static,
    F: FnOnce(openai_frontend::CancellationToken) -> T + Send + 'static,
    P: Send + 'static,
{
    task::spawn_blocking(move || {
        let _permit = permit;
        work(context.cancellation_token())
    })
    .await
}

#[async_trait]
impl OpenAiBackend for StageOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let ids = generation_ids(
            OpenAiCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.chat_completion_with_hooks(request, move |mut request| async move {
            self.ensure_model(&request.model)?;
            apply_chat_request_defaults(&mut request, &self.request_defaults)?;
            ensure_chat_runtime_features_supported(&request)?;
            let sampling = chat_sampling_config(&request)?;
            let template_options = chat_template_options(&request, &self.request_defaults)?;
            let parse_chat_output = chat_output_parser_required(&request, &template_options);
            let template_timer = PhaseTimer::start();
            let prompt = self
                .prepare_chat_prompt_offloaded(&request, template_options.clone())
                .await?;
            let mut template_attrs = self.openai_attrs(&ids);
            template_attrs.insert(
                "llama_stage.openai_operation".to_string(),
                json!("chat_completion"),
            );
            template_attrs.insert(
                "llama_stage.chat_message_count".to_string(),
                json!(request.messages.len()),
            );
            template_attrs.insert(
                "llama_stage.prompt_chars".to_string(),
                json!(prompt.text.len()),
            );
            template_attrs.insert(
                "llama_stage.media_item_count".to_string(),
                json!(prompt.media.len()),
            );
            self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
            let max_tokens = GenerationTokenLimit::from_request(
                request.effective_max_tokens(),
                self.default_max_tokens,
            );
            let chat_parse_metadata = prompt.chat_parse_metadata.clone();
            let output = self
                .run_generation(
                    prompt,
                    max_tokens,
                    request.stop.clone(),
                    sampling,
                    Some(request.clone()),
                    context,
                    ids.clone(),
                )
                .await?;
            let response_timer = PhaseTimer::start();
            let parsed_message = if parse_chat_output {
                self.parse_chat_output(
                    &output.text,
                    &request,
                    chat_parse_metadata.as_deref(),
                    false,
                )?
            } else {
                None
            };
            let parsed_message = apply_reasoning_visibility(parsed_message, &template_options);
            let response =
                chat_response_from_generated_text(request.model.clone(), &output, parsed_message);
            let mut response_attrs = self.openai_attrs(&ids);
            response_attrs.insert(
                "llama_stage.openai_operation".to_string(),
                json!("chat_completion"),
            );
            response_attrs.insert(
                "llama_stage.prompt_token_count".to_string(),
                json!(output.prompt_tokens),
            );
            response_attrs.insert(
                "llama_stage.completion_token_count".to_string(),
                json!(output.completion_tokens),
            );
            self.emit_openai_phase(
                "stage.openai_response_build",
                response_timer,
                response_attrs,
            );
            let mut summary_attrs = self.openai_attrs(&ids);
            summary_attrs.insert(
                "llama_stage.openai_operation".to_string(),
                json!("chat_completion"),
            );
            summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
            summary_attrs.insert(
                "llama_stage.prompt_token_count".to_string(),
                json!(output.prompt_tokens),
            );
            summary_attrs.insert(
                "llama_stage.completion_token_count".to_string(),
                json!(output.completion_tokens),
            );
            self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
            Ok(response)
        })
        .await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let ids = generation_ids(
            OpenAiCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        self.chat_completion_stream_with_hooks(request, move |mut request| async move {
            self.ensure_model(&request.model)?;
            apply_chat_request_defaults(&mut request, &self.request_defaults)?;
            ensure_chat_runtime_features_supported(&request)?;
            let sampling = chat_sampling_config(&request)?;
            let include_usage = request.include_usage();
            let template_options = chat_template_options(&request, &self.request_defaults)?;
            let parse_chat_output = chat_output_parser_required(&request, &template_options);
            let emit_reasoning = template_exposes_reasoning(&template_options);
            let template_timer = PhaseTimer::start();
            let prompt = self
                .prepare_chat_prompt_offloaded(&request, template_options)
                .await?;
            let mut template_attrs = self.openai_attrs(&ids);
            template_attrs.insert(
                "llama_stage.openai_operation".to_string(),
                json!("chat_completion_stream"),
            );
            template_attrs.insert(
                "llama_stage.chat_message_count".to_string(),
                json!(request.messages.len()),
            );
            template_attrs.insert(
                "llama_stage.prompt_chars".to_string(),
                json!(prompt.text.len()),
            );
            template_attrs.insert(
                "llama_stage.media_item_count".to_string(),
                json!(prompt.media.len()),
            );
            self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
            let max_tokens = GenerationTokenLimit::from_request(
                request.effective_max_tokens(),
                self.default_max_tokens,
            );
            let model = request.model.clone();
            let stream = self
                .run_generation_stream(
                    prompt,
                    max_tokens,
                    request.stop.clone(),
                    sampling,
                    include_usage,
                    Some(request),
                    parse_chat_output,
                    emit_reasoning,
                    context,
                    ids,
                )
                .await?;
            let stream: ChatCompletionStream =
                Box::pin(stream.map(move |event| generation_event_to_chat_chunk(event, &model)));
            Ok(stream)
        })
        .await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        mut request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        let ids = generation_ids(
            OpenAiCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.ensure_model(&request.model)?;
        apply_completion_request_defaults(&mut request, &self.request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                None,
                context,
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let response = completion_response_from_generated_text(request.model, &output);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn completion_stream(
        &self,
        mut request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        let ids = generation_ids(
            OpenAiCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        self.ensure_model(&request.model)?;
        apply_completion_request_defaults(&mut request, &self.request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let model = request.model.clone();
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion_stream"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                None,
                false,
                false,
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_completion_chunk(event, &model)
        })))
    }
}

impl StageOpenAiBackend {
    async fn acquire_generation_admission(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> OpenAiResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        let result = GenerationAdmissionController::for_backend(self)
            .acquire_scheduled_work(
                ids,
                cancellation,
                self.generation_admission_timeout,
                work,
                scheduling,
            )
            .await;
        if let Err(error) = &result {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "llama_stage.generation_queue_depth".to_string(),
                json!(self.generation_queue_depth.load(Ordering::Acquire)),
            );
            attrs.insert(
                "llama_stage.generation_queue_capacity".to_string(),
                json!(self.generation_queue_limit),
            );
            attrs.insert(
                "llama_stage.generation_admission_status".to_string(),
                json!("rejected"),
            );
            attrs.insert(
                "llama_stage.generation_admission_error".to_string(),
                json!(error.body().error.message),
            );
            if let Some(predicted_wait_ms) = self.generation_service_estimator.predicted_wait_ms() {
                attrs.insert(
                    "llama_stage.generation_predicted_wait_ms".to_string(),
                    json!(predicted_wait_ms),
                );
            }
            self.telemetry
                .emit("stage.openai_generation_admission_rejected", attrs);
        }
        result
    }

    fn generation_admission_scheduling(
        &self,
        prepared_text: Option<&PreparedTextPrompt>,
        ids: &OpenAiGenerationIds,
    ) -> GenerationAdmissionScheduling {
        let Some(prepared) = prepared_text else {
            return GenerationAdmissionScheduling::default();
        };
        let prompt_tokens = Arc::<[i32]>::from(prepared.token_ids.clone());
        let Some(kv) = self.kv.as_ref() else {
            return GenerationAdmissionScheduling::new(
                prompt_tokens,
                Arc::new(skippy_scheduler::CacheAffinity::default),
            );
        };
        let prefill_tokens = prepared
            .token_ids
            .get(..prepared.token_ids.len().saturating_sub(1))
            .unwrap_or_default();
        if prefill_tokens.is_empty() {
            return GenerationAdmissionScheduling::new(
                prompt_tokens,
                Arc::new(skippy_scheduler::CacheAffinity::default),
            );
        }
        let base = self.local_kv_message_base(&ids.session_label, ids);
        let identities = Arc::from(kv.lookup_identities(&self.config, &base, 0, prefill_tokens));
        let kv = Arc::clone(kv);
        let config = self.config.clone();
        GenerationAdmissionScheduling::new(
            prompt_tokens,
            Arc::new(move || kv.peek_cache_affinity(&config, &identities)),
        )
    }

    fn generation_admission_work(
        &self,
        prompt: &PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        prepared_text: Option<&PreparedTextPrompt>,
    ) -> OpenAiResult<GenerationAdmissionWork> {
        if let Some(prepared) = prepared_text {
            return Ok(GenerationAdmissionWork::new(
                prepared.token_ids.len(),
                prepared.max_tokens,
            ));
        }
        let estimated_prompt_tokens = prompt.text.len().div_ceil(4).max(1);
        let decode_tokens = max_tokens.resolve(estimated_prompt_tokens, self.ctx_size)?;
        Ok(GenerationAdmissionWork::new(
            estimated_prompt_tokens,
            decode_tokens,
        ))
    }

    pub(super) fn openai_attrs(&self, ids: &OpenAiGenerationIds) -> BTreeMap<String, Value> {
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            attr_key::SESSION_ID.to_string(),
            json!(ids.session_id_string()),
        );
        attrs.insert(
            attr_key::REQUEST_ID.to_string(),
            json!(ids.request_id_string()),
        );
        attrs.insert(
            "llama_stage.openai_backend".to_string(),
            json!(self.mode.label()),
        );
        if let Some(cache_key) = ids.cache.prompt_cache_key.as_deref() {
            attrs.insert("openai.prompt_cache_key".to_string(), json!(cache_key));
        }
        if let Some(retention) = ids.cache.prompt_cache_retention.as_deref() {
            attrs.insert(
                "openai.prompt_cache_retention".to_string(),
                json!(retention),
            );
        }
        attrs
    }

    pub(super) fn insert_runtime_session_stats(
        attrs: &mut BTreeMap<String, Value>,
        prefix: &str,
        stats: &RuntimeSessionStats,
    ) {
        attrs.insert(
            format!("{prefix}.active_sessions"),
            json!(stats.active_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_sessions"),
            json!(stats.idle_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_resident_prefixes"),
            json!(stats.idle_resident_prefixes),
        );
        attrs.insert(
            format!("{prefix}.tracked_token_counts"),
            json!(stats.tracked_token_counts),
        );
    }

    pub(super) fn emit_openai_phase(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_debug_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn emit_openai_summary(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn ensure_model(&self, requested: &str) -> OpenAiResult<()> {
        ensure_requested_model(&self.model_id, requested)
    }

    async fn chat_completion_with_hooks<F, Fut>(
        &self,
        mut request: ChatCompletionRequest,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionResponse>
    where
        F: FnOnce(ChatCompletionRequest) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionResponse>>,
    {
        let hooks = self
            .hook_policy
            .clone()
            .filter(|_| chat_mesh_hooks_enabled(&request));
        let exchange_id = uuid::Uuid::new_v4().to_string();
        let mut guard = hooks
            .as_ref()
            .map(|hooks| TerminalGuard::new(hooks.clone(), request.clone(), exchange_id.clone()));
        let mut dispatched_request = None;

        if let Some(hooks) = hooks.clone() {
            match hooks.before_chat_completion(&mut request).await {
                Ok(outcome) => {
                    apply_chat_hook_outcome(&mut request, &outcome);
                    let route = ChatExchangeRoute::for_request(&request, exchange_id.clone());
                    hooks.on_effective_chat_completion(&request, &route).await;
                }
                Err(error) => {
                    let reason = error.to_string();
                    if let Some(mut guard) = guard.take() {
                        guard.set_request(request.clone());
                        guard
                            .fire(&ChatCompletionOutcome::Denied {
                                status: error.status().as_u16(),
                                reason: &reason,
                            })
                            .await;
                    }
                    return Err(error);
                }
            }

            let effective = if hooks.observes_dispatched_request() {
                request.clone()
            } else {
                ChatCompletionRequest::default()
            };
            if let Some(guard) = guard.as_mut() {
                guard.set_request(effective.clone());
            }
            dispatched_request = Some(effective);
        }

        let mut result = dispatch(request).await;

        if let (Some(hooks), Some(dispatched_request), Ok(response)) =
            (&hooks, &dispatched_request, &mut result)
            && let Some(marker) = hooks
                .capsule_marker_for_response(dispatched_request, &*response)
                .await
        {
            if capsule_id_is_valid(&marker.capsule_id) {
                response.capsule_marker = Some(marker);
            } else {
                tracing::warn!(
                    capsule_id = %marker.capsule_id,
                    "dropping capsule marker: invalid capsule id"
                );
            }
        }

        if let Some(guard) = guard {
            let error_message;
            let terminal = match &result {
                Ok(response) => ChatCompletionOutcome::Success { response },
                Err(error) => {
                    error_message = error.to_string();
                    ChatCompletionOutcome::Error {
                        status: error.status().as_u16(),
                        message: &error_message,
                    }
                }
            };
            guard.fire(&terminal).await;
        }
        result
    }

    async fn chat_completion_stream_with_hooks<F, Fut>(
        &self,
        mut request: ChatCompletionRequest,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionStream>
    where
        F: FnOnce(ChatCompletionRequest) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionStream>>,
    {
        let hooks = self
            .hook_policy
            .clone()
            .filter(|_| chat_mesh_hooks_enabled(&request));
        let exchange_id = uuid::Uuid::new_v4().to_string();
        let mut guard = hooks
            .as_ref()
            .map(|hooks| TerminalGuard::new(hooks.clone(), request.clone(), exchange_id.clone()));

        if let Some(hooks) = hooks.clone() {
            match hooks.before_chat_completion(&mut request).await {
                Ok(outcome) => {
                    apply_chat_hook_outcome(&mut request, &outcome);
                    let route = ChatExchangeRoute::for_request(&request, exchange_id.clone());
                    hooks.on_effective_chat_completion(&request, &route).await;
                }
                Err(error) => {
                    let reason = error.to_string();
                    if let Some(mut guard) = guard.take() {
                        guard.set_request(request.clone());
                        guard
                            .fire(&ChatCompletionOutcome::Denied {
                                status: error.status().as_u16(),
                                reason: &reason,
                            })
                            .await;
                    }
                    return Err(error);
                }
            }
            if let Some(guard) = guard.as_mut() {
                guard.set_request(request.clone());
            }
        }

        match dispatch(request).await {
            Ok(stream) => Ok(match guard {
                Some(guard) => TerminalGuardedChatStream::pinned(stream, guard),
                None => stream,
            }),
            Err(error) => {
                if let Some(guard) = guard {
                    let message = error.to_string();
                    guard
                        .fire(&ChatCompletionOutcome::Error {
                            status: error.status().as_u16(),
                            message: &message,
                        })
                        .await;
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GeneratedText> {
        let (prompt, prepared_text) = if prompt.has_media() {
            (prompt, None)
        } else {
            let backend = self.clone();
            let ids_for_tokenize = ids.clone();
            task::spawn_blocking(move || {
                let prepared =
                    backend.prepare_text_prompt(&prompt, max_tokens, &ids_for_tokenize)?;
                Ok::<_, OpenAiError>((prompt, Some(prepared)))
            })
            .await
            .map_err(|error| {
                OpenAiError::backend(format!("prompt tokenization task failed: {error}"))
            })??
        };
        let admission_work =
            self.generation_admission_work(&prompt, max_tokens, prepared_text.as_ref())?;
        let admission_scheduling =
            self.generation_admission_scheduling(prepared_text.as_ref(), &ids);
        let admit_timer = PhaseTimer::start();
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation, admission_work, admission_scheduling)
            .await?;
        let admission_wait_ms = admit_timer.elapsed_ms();
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        insert_generation_admission_attrs(
            &mut admit_attrs,
            &permit,
            self.generation_queue_depth.load(Ordering::Acquire),
            self.generation_queue_limit,
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let demand = permit.demand_observation();
        let token_budget_stats = permit.token_budget_stats();
        let backend = self.clone();
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let worker_context = context.clone();
        let mut result =
            run_blocking_generation_worker(permit, worker_context.clone(), move |token| {
                let _session_permit = session_permit;
                let output = backend.generate_text(
                    prompt,
                    max_tokens,
                    prepared_text,
                    token_budget_stats,
                    stop.as_ref(),
                    sampling,
                    hook_request,
                    hook_runtime,
                    Some(&token),
                    ids,
                    |_| Ok(()),
                );
                if worker_context.is_cancelled() {
                    Err(request_cancelled_error())
                } else {
                    output
                }
            })
            .await
            .map_err(|error| OpenAiError::backend(format!("generation task failed: {error}")))?;
        if let Ok(output) = &mut result {
            output.queue_wait_ms = admission_wait_ms;
        }
        if context.is_cancelled() {
            Err(request_cancelled_error())
        } else {
            if let Ok(output) = &result {
                self.observe_generation_completed(output, demand);
            } else {
                self.observe_generation_failed();
            }
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation_stream(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        include_usage: bool,
        hook_request: Option<ChatCompletionRequest>,
        parse_chat_output: bool,
        emit_reasoning: bool,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GenerationStream> {
        let (prompt, prepared_text) = if prompt.has_media() {
            (prompt, None)
        } else {
            let backend = self.clone();
            let ids_for_tokenize = ids.clone();
            task::spawn_blocking(move || {
                let prepared =
                    backend.prepare_text_prompt(&prompt, max_tokens, &ids_for_tokenize)?;
                Ok::<_, OpenAiError>((prompt, Some(prepared)))
            })
            .await
            .map_err(|error| {
                OpenAiError::backend(format!("prompt tokenization task failed: {error}"))
            })??
        };
        let admit_timer = PhaseTimer::start();
        let admission_work =
            self.generation_admission_work(&prompt, max_tokens, prepared_text.as_ref())?;
        let admission_scheduling =
            self.generation_admission_scheduling(prepared_text.as_ref(), &ids);
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation, admission_work, admission_scheduling)
            .await?;
        let admission_wait_ms = admit_timer.elapsed_ms();
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        insert_generation_admission_attrs(
            &mut admit_attrs,
            &permit,
            self.generation_queue_depth.load(Ordering::Acquire),
            self.generation_queue_limit,
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let demand = permit.demand_observation();
        let token_budget_stats = permit.token_budget_stats();
        let backend = self.clone();
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let (tx, rx) = mpsc::channel(16);
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let sender = StreamEventSender::new(
            tx,
            tokio::runtime::Handle::current(),
            STREAM_SEND_STALL_TIMEOUT,
            ids.request_id_string(),
            self.telemetry.clone(),
        );
        let generation_request_needed = self.hook_policy.is_some()
            || hook_request
                .as_ref()
                .is_some_and(crate::frontend::generation::tool_calls_requested);
        let mut hook_request = hook_request;
        let parser_request = if parse_chat_output {
            if generation_request_needed {
                hook_request.clone()
            } else {
                hook_request.take()
            }
        } else {
            None
        };
        let mut chat_stream_parser =
            if let (Some(request), Some(metadata)) = (parser_request, chat_parse_metadata) {
                Some(ChatOutputStreamParser::new(
                    backend.clone(),
                    request,
                    metadata,
                    emit_reasoning,
                )?)
            } else {
                None
            };
        task::spawn_blocking(move || {
            let _session_permit = session_permit;
            let _permit = permit;
            let result = backend.generate_text(
                prompt,
                max_tokens,
                prepared_text,
                token_budget_stats,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                Some(&context.cancellation_token()),
                ids,
                |chunk| {
                    if context.is_cancelled() {
                        return Err(OpenAiError::backend("stream receiver cancelled"));
                    }
                    let events = if let Some(parser) = chat_stream_parser.as_mut() {
                        parser.push_delta(chunk)?
                    } else {
                        vec![GenerationStreamEvent::Delta(chunk.to_string())]
                    };
                    for event in events {
                        sender.send(Ok(event), &context)?;
                    }
                    Ok(())
                },
            );
            if context.is_cancelled() {
                let _ = sender.send_terminal(Err(request_cancelled_error()));
                return;
            }
            match result {
                Ok(mut output) => {
                    output.queue_wait_ms = admission_wait_ms;
                    backend.observe_generation_completed(&output, demand);
                    let finish_reason = if let Some(parser) = chat_stream_parser.as_mut() {
                        match parser.finish(&output.text) {
                            Ok(events) => {
                                for event in events {
                                    if sender.send_terminal(Ok(event)).is_err() {
                                        return;
                                    }
                                }
                                parser.finish_reason(output.finish_reason)
                            }
                            Err(error) => {
                                let _ = sender.send_terminal(Err(error));
                                return;
                            }
                        }
                    } else {
                        output.finish_reason
                    };
                    if should_emit_stream_usage(include_usage, &context)
                        && sender
                            .send_terminal(Ok(GenerationStreamEvent::Usage(
                                output.usage(),
                                output.timings(),
                            )))
                            .is_err()
                    {
                        return;
                    }
                    let _ = sender.send_terminal(Ok(GenerationStreamEvent::Done(finish_reason)));
                }
                Err(error) => {
                    backend.observe_generation_failed();
                    let _ = sender.send_terminal(Err(error));
                }
            }
        });
        Ok(Box::pin(stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })))
    }
}

#[cfg(test)]
mod tests;
