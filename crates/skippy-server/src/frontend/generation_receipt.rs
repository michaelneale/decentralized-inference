use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use anyhow::Result;
use openai_frontend::{OpenAiError, OpenAiResult};

use crate::frontend::{StageOpenAiBackend, openai_backend_error};
use crate::runtime_state::RuntimeState;

const TOKEN_ID_DIGEST_DOMAIN: &[u8] = b"skippy-generation-token-ids-v1\0";
const GENERATION_RECEIPT_QUEUE_CAPACITY: usize = 1_024;

/// Why a successful local generation stopped.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTermination {
    /// The token callback requested a stop, including for an end-of-generation token.
    CallbackStop,
    /// The request consumed its resolved completion-token budget.
    MaxTokens,
    /// The local generation loop observed request cancellation.
    Cancelled,
}

/// Optional digest of the target runtime's full exported state.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationStateDigest {
    /// Number of bytes in the exported runtime state.
    pub byte_length: u64,
    /// BLAKE3 digest of the exported runtime-state bytes.
    pub blake3_digest: [u8; 32],
}

/// Target-authoritative result captured immediately before local session teardown.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationReceipt {
    /// OpenAI request identity.
    pub request_id: u64,
    /// OpenAI session identity.
    pub session_id: u64,
    /// Stable caller-supplied agent session, when admitted at the OpenAI
    /// boundary. Authentication remains an endpoint concern. This is distinct
    /// from Mesh's request-scoped runtime session.
    pub agent_session_id: Option<Box<str>>,
    /// Number of target-tokenized prompt-text IDs supplied to local generation.
    ///
    /// For multimodal requests, media embeddings have no token IDs and are not included.
    pub prompt_token_count: usize,
    /// Stable digest of the target-tokenized prompt-text IDs.
    pub prompt_token_digest: [u8; 32],
    /// Exact target-tokenized prompt-text IDs supplied to local generation.
    pub prompt_token_ids: Arc<[i32]>,
    /// Target-authoritative generated token IDs in callback order.
    pub generated_token_ids: Box<[i32]>,
    /// Canonical runtime position captured before session teardown.
    pub final_session_position: u64,
    /// Why generation stopped successfully.
    pub termination: GenerationTermination,
    /// Time spent in model generation, excluding receipt delivery.
    pub model_generation_elapsed_us: u64,
    /// Backend request-start to first generated-token availability.
    pub request_to_first_token_us: Option<u64>,
    /// Backend request-start to each generated-token availability, in token order.
    pub request_to_token_emission_us: Box<[u64]>,
    /// Optional digest of the target runtime's full exported state.
    pub full_state: Option<GenerationStateDigest>,
    /// See [`GenerationStart::frontend_request_id`].
    pub frontend_request_id: Option<[u8; 16]>,
}

impl GenerationReceipt {
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_fixture(
        request_id: u64,
        session_id: u64,
        termination: GenerationTermination,
    ) -> Self {
        Self {
            request_id,
            session_id,
            agent_session_id: None,
            prompt_token_count: 1,
            prompt_token_digest: [0; 32],
            prompt_token_ids: Arc::from([1]),
            generated_token_ids: vec![1].into_boxed_slice(),
            final_session_position: 1,
            termination,
            model_generation_elapsed_us: 1,
            request_to_first_token_us: Some(1),
            request_to_token_emission_us: vec![1].into_boxed_slice(),
            full_state: None,
            frontend_request_id: None,
        }
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_fixture_with_full_state(
        request_id: u64,
        session_id: u64,
        termination: GenerationTermination,
        byte_length: u64,
    ) -> Self {
        let mut receipt = Self::test_fixture(request_id, session_id, termination);
        receipt.full_state = Some(GenerationStateDigest {
            byte_length,
            blake3_digest: [0; 32],
        });
        receipt
    }
}

/// Target-authoritative beginning of one local generation lifecycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationStart {
    pub request_id: u64,
    pub session_id: u64,
    /// Stable caller-supplied agent session admitted at the OpenAI boundary.
    pub agent_session_id: Option<Box<str>>,
    pub prompt_token_ids: Arc<[i32]>,
    /// Byte-equal to the OpenAI request root `OperationId`
    /// (`OpenAiLifecycleContext.request_id`'s raw UUID bytes) when this
    /// generation was admitted through the OpenAI boundary. `None` for
    /// non-frontend callers. Never projected across the native plugin ABI.
    pub frontend_request_id: Option<[u8; 16]>,
}

/// Target-authoritative termination of a generation that produced no final
/// receipt. The proposal/session adapter uses this boundary to close durable
/// request state instead of leaving later requests blocked behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationAbort {
    pub request_id: u64,
    pub session_id: u64,
}

/// A canonical target-token delta committed during an active generation.
///
/// This is a model-neutral producer observation. It deliberately carries no
/// consumer-specific state: integrations receive only the target-owned
/// request identity, canonical position, and token delta. Normal runtime event
/// adapters must reduce it to bounded counts rather than forwarding token IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationCommit {
    pub request_id: u64,
    pub session_id: u64,
    /// Total generated canonical tokens after applying `token_ids`.
    pub generated_token_count: usize,
    pub token_ids: Box<[i32]>,
}

/// Lightweight terminal summary for lifecycle-only integrations.
///
/// This deliberately contains no canonical session position, token digest, or
/// exported runtime state. Those fields belong to [`GenerationReceipt`], which
/// is available only for local single-stage execution. A lifecycle-only
/// consumer can still observe the complete started/progress/finished-or-aborted
/// sequence without being given evidence that a split stage cannot authoritatively
/// produce.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationCompletion {
    pub request_id: u64,
    pub session_id: u64,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub termination: GenerationTermination,
    pub request_to_first_token_us: Option<u64>,
}

impl GenerationCompletion {
    #[must_use]
    pub fn new(
        request_id: u64,
        session_id: u64,
        prompt_token_count: usize,
        generated_token_count: usize,
        termination: GenerationTermination,
        request_to_first_token_us: Option<u64>,
    ) -> Self {
        Self {
            request_id,
            session_id,
            prompt_token_count,
            generated_token_count,
            termination,
            request_to_first_token_us,
        }
    }
}

/// Receives the complete local-generation lifecycle before runtime teardown.
///
/// The producer submits `begin` before exactly one `record` or `abort`. The
/// compatibility adapter is bounded, asynchronous, at-most-once, and not
/// durable: queue saturation, worker shutdown, or sink failure increments the
/// delivery-failure counter and can drop an observation. Integrations that
/// require shared ordering or custom backpressure should implement
/// [`GenerationLifecycleIngress`] directly.
pub trait GenerationReceiptSink: Send + Sync {
    fn begin(&self, start: &GenerationStart) -> Result<()>;

    /// Optionally observes canonical target-token deltas for integrations that
    /// need lifecycle notifications outside the local proposal hot path.
    fn committed(&self, commit: &GenerationCommit) -> Result<()>;

    fn abort(&self, abort: &GenerationAbort) -> Result<()>;

    /// Records one successful local-generation receipt.
    fn record(&self, receipt: &GenerationReceipt) -> Result<()>;
}

/// One producer-side observation from Skippy's authoritative generation path.
///
/// This is deliberately not a runtime domain event or a consumer projection.
/// Some variants contain exact token IDs for serving integrations that require
/// canonical model evidence. An adapter into a normal runtime event pipeline
/// must project only bounded, privacy-safe summaries and must not forward those
/// token IDs or their prompt-derived digest.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerationLifecycleObservation {
    Started(GenerationStart),
    Committed(GenerationCommit),
    Aborted(GenerationAbort),
    Finished(GenerationCompletion),
    Completed(GenerationReceipt),
}

impl GenerationLifecycleObservation {
    fn deliver_to(self, sink: &dyn GenerationReceiptSink) -> Result<()> {
        match self {
            Self::Started(start) => sink.begin(&start),
            Self::Committed(commit) => sink.committed(&commit),
            Self::Aborted(abort) => sink.abort(&abort),
            // The queued compatibility path is exact-receipt delivery. A
            // lifecycle-only completion must use GenerationLifecycleConfig and
            // must never be silently turned into a fabricated receipt.
            Self::Finished(_) => Err(anyhow::anyhow!(
                "lifecycle-only completion cannot be delivered to an exact receipt sink"
            )),
            Self::Completed(receipt) => sink.record(&receipt),
        }
    }
}

/// Nonblocking ingress for authoritative generation observations.
///
/// Implementations must copy or take ownership of the observation and return
/// promptly. They must not perform formatting, I/O, telemetry export, or call
/// arbitrary application subscribers inline. The future runtime event system
/// can implement this boundary as an adapter into its bounded ingress without
/// changing Skippy's generation loops.
pub trait GenerationLifecycleIngress: Send + Sync {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()>;

    /// Asynchronous delivery failures observed after a successful submission.
    fn delivery_failures(&self) -> u64 {
        0
    }

    /// Reports that a generation completed but its receipt could not be
    /// built: a `canonical_session_position` or `export_full_state`
    /// bookkeeping failure inside receipt construction. This is deliberately
    /// distinct from [`Self::try_submit`]'s `Completed`/`Aborted`
    /// observations -- there is no receipt and no caller-initiated abort to
    /// hand a consumer, only the fact that a terminal receipt could not be
    /// produced for this generation.
    ///
    /// Health-only, exactly like every other method on this trait: an
    /// implementation must count it toward [`Self::delivery_failures`], must
    /// never block, and must never fail. The default no-op is correct for an
    /// implementation with no additional health/terminal bookkeeping to
    /// perform.
    fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
        let _ = unavailable;
    }
}

/// Fans one generation-lifecycle observation out to every installed sink for
/// the exact-receipt path. One sink failing to accept an observation never
/// prevents delivery to the other sinks.
pub struct CompositeGenerationLifecycleIngress {
    sinks: Vec<Arc<dyn GenerationLifecycleIngress>>,
}

impl CompositeGenerationLifecycleIngress {
    #[must_use]
    pub fn new(sinks: Vec<Arc<dyn GenerationLifecycleIngress>>) -> Self {
        Self { sinks }
    }
}

impl GenerationLifecycleIngress for CompositeGenerationLifecycleIngress {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
        let mut first_error = None;
        for sink in &self.sinks {
            if let Err(error) = sink.try_submit(observation.clone()) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn delivery_failures(&self) -> u64 {
        self.sinks.iter().map(|sink| sink.delivery_failures()).sum()
    }

    fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
        for sink in &self.sinks {
            sink.receipt_unavailable(unavailable);
        }
    }
}

/// Lifecycle-only generation observation delivery.
///
/// This is separate from [`GenerationReceiptConfig`]: it emits the same
/// started/progress/abort boundaries but finishes with [`GenerationCompletion`]
/// instead of constructing an exact [`GenerationReceipt`]. Split execution can
/// therefore feed runtime-event consumers without enabling local-only receipt
/// validation or inventing canonical session state.
#[derive(Clone)]
pub struct GenerationLifecycleConfig {
    ingress: Arc<dyn GenerationLifecycleIngress>,
    submission_failures: Arc<AtomicU64>,
}

impl GenerationLifecycleConfig {
    /// Uses an existing bounded, nonblocking lifecycle ingress.
    pub fn from_ingress(ingress: Arc<dyn GenerationLifecycleIngress>) -> Self {
        Self {
            ingress,
            submission_failures: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Number of observations rejected synchronously by the ingress, plus
    /// asynchronous downstream delivery failures reported by it.
    pub fn delivery_failures(&self) -> u64 {
        self.submission_failures
            .load(Ordering::Relaxed)
            .saturating_add(self.ingress.delivery_failures())
    }

    pub(crate) fn begin(&self, start: GenerationStart) {
        self.enqueue(GenerationLifecycleObservation::Started(start));
    }

    pub(crate) fn committed(&self, commit: GenerationCommit) {
        self.enqueue(GenerationLifecycleObservation::Committed(commit));
    }

    pub(crate) fn finished(&self, completion: GenerationCompletion) {
        self.enqueue(GenerationLifecycleObservation::Finished(completion));
    }

    pub(crate) fn abort(&self, abort: GenerationAbort) {
        self.enqueue(GenerationLifecycleObservation::Aborted(abort));
    }

    fn enqueue(&self, observation: GenerationLifecycleObservation) {
        if self.ingress.try_submit(observation).is_err() {
            self.submission_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct QueuedGenerationReceiptSink {
    sender: SyncSender<GenerationLifecycleObservation>,
    delivery_failures: Arc<AtomicU64>,
}

impl QueuedGenerationReceiptSink {
    fn new(sink: Arc<dyn GenerationReceiptSink>) -> Self {
        let (sender, receiver) =
            sync_channel::<GenerationLifecycleObservation>(GENERATION_RECEIPT_QUEUE_CAPACITY);
        let delivery_failures = Arc::new(AtomicU64::new(0));
        let worker_delivery_failures = Arc::clone(&delivery_failures);
        std::thread::Builder::new()
            .name("skippy-generation-receipts".into())
            .spawn(move || {
                while let Ok(observation) = receiver.recv() {
                    if observation.deliver_to(sink.as_ref()).is_err() {
                        worker_delivery_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
            .expect("generation lifecycle delivery thread must start");
        Self {
            sender,
            delivery_failures,
        }
    }
}

impl GenerationLifecycleIngress for QueuedGenerationReceiptSink {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
        self.sender
            .try_send(observation)
            .map_err(|error| match error {
                TrySendError::Full(_) => anyhow::anyhow!("generation lifecycle queue is full"),
                TrySendError::Disconnected(_) => {
                    anyhow::anyhow!("generation lifecycle delivery worker is unavailable")
                }
            })
    }

    fn delivery_failures(&self) -> u64 {
        self.delivery_failures.load(Ordering::Relaxed)
    }

    /// Forwards to the wrapped sink's existing `abort` handler: a
    /// [`GenerationReceiptSink`] has no separate concept of "receipt could
    /// not be built" from "generation produced no final receipt", and
    /// `GenerationAbort`'s own contract already covers exactly that case.
    fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
        // The receipt-build failure is itself a delivery failure even when
        // the health-only abort reaches the worker. A worker sink failure is
        // counted separately by the worker loop.
        self.delivery_failures.fetch_add(1, Ordering::Relaxed);
        let _ = self.try_submit(GenerationLifecycleObservation::Aborted(*unavailable));
    }
}

/// Optional local-generation observation.
///
/// The default configuration records exact token IDs and positions without exporting
/// full model state. Full-state export is intended for exactness checks only: it is
/// deliberately opt-in and must remain disabled for timed measurements.
#[derive(Clone)]
pub struct GenerationReceiptConfig {
    ingress: Arc<dyn GenerationLifecycleIngress>,
    submission_failures: Arc<AtomicU64>,
    recording_failures: Arc<AtomicU64>,
    export_full_state: bool,
}

impl GenerationReceiptConfig {
    /// Creates queued receipt delivery with full-state export disabled.
    ///
    /// The sink is always invoked from an isolated worker. Event-pipeline
    /// integrations should use [`Self::from_lifecycle_ingress`] instead.
    pub fn new(sink: Arc<dyn GenerationReceiptSink>) -> Self {
        Self::from_lifecycle_ingress(Arc::new(QueuedGenerationReceiptSink::new(sink)))
    }

    /// Uses an existing bounded, nonblocking lifecycle ingress.
    ///
    /// This avoids adding a second queue when the host already owns delivery,
    /// ordering, backpressure, and health accounting.
    pub fn from_lifecycle_ingress(ingress: Arc<dyn GenerationLifecycleIngress>) -> Self {
        Self {
            ingress,
            submission_failures: Arc::new(AtomicU64::new(0)),
            recording_failures: Arc::new(AtomicU64::new(0)),
            export_full_state: false,
        }
    }

    #[must_use]
    /// Enables or disables the optional full-state digest.
    pub fn with_full_state_digest(mut self, enabled: bool) -> Self {
        self.export_full_state = enabled;
        self
    }

    /// Reports whether receipt delivery exports and hashes full runtime state.
    pub fn exports_full_state(&self) -> bool {
        self.export_full_state
    }

    /// Number of rejected submissions or asynchronous downstream delivery failures.
    pub fn delivery_failures(&self) -> u64 {
        self.submission_failures
            .load(Ordering::Relaxed)
            .saturating_add(self.ingress.delivery_failures())
    }

    /// Number of local receipt-bookkeeping violations. These failures disable
    /// receipt recording for the affected generation but never fail inference.
    pub fn recording_failures(&self) -> u64 {
        self.recording_failures.load(Ordering::Relaxed)
    }

    /// Reports a receipt-build failure to the underlying ingress. Health-only
    /// like every other delivery path on this type: never returns an error,
    /// never affects the request that produced the generation. Counted
    /// through the ingress's own [`GenerationLifecycleIngress::delivery_failures`]
    /// rather than `Self::submission_failures`, which tracks admission
    /// failures for an observation that was actually built -- this failure
    /// never reaches that far.
    pub(crate) fn receipt_unavailable(&self, request_id: u64, session_id: u64) {
        self.ingress.receipt_unavailable(&GenerationAbort {
            request_id,
            session_id,
        });
    }

    pub(crate) fn observation(&self, max_tokens: usize) -> GenerationReceiptObservation {
        GenerationReceiptObservation::new(max_tokens, Arc::clone(&self.recording_failures))
    }

    pub(crate) fn begin(&self, start: GenerationStart) {
        self.enqueue(GenerationLifecycleObservation::Started(start));
    }

    pub(crate) fn committed(&self, commit: GenerationCommit) {
        self.enqueue(GenerationLifecycleObservation::Committed(commit));
    }

    pub(crate) fn abort(&self, abort: GenerationAbort) {
        self.enqueue(GenerationLifecycleObservation::Aborted(abort));
    }

    pub(crate) fn record(&self, receipt: GenerationReceipt) {
        self.enqueue(GenerationLifecycleObservation::Completed(receipt));
    }

    fn enqueue(&self, observation: GenerationLifecycleObservation) {
        if self.ingress.try_submit(observation).is_err() {
            self.submission_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Stable, platform-independent digest of signed token IDs.
///
/// The encoding is a domain tag, a little-endian `u64` token count, and each
/// signed token ID in little-endian `i32` form.
pub fn generation_token_id_digest(token_ids: &[i32]) -> [u8; 32] {
    let token_count =
        u64::try_from(token_ids.len()).expect("supported targets have at most u64::MAX tokens");
    let mut hasher = blake3::Hasher::new();
    hasher.update(TOKEN_ID_DIGEST_DOMAIN);
    hasher.update(&token_count.to_le_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

pub(crate) struct GenerationReceiptObservation {
    generated_token_ids: Vec<i32>,
    token_emission_elapsed: Vec<Duration>,
    max_tokens: usize,
    recording_failures: Arc<AtomicU64>,
    recording_enabled: bool,
    termination: Option<GenerationTermination>,
    model_generation_elapsed: Option<Duration>,
}

pub(crate) struct LocalGenerationReceiptDelivery<'a> {
    pub(crate) config: &'a GenerationReceiptConfig,
    pub(crate) session_label: &'a str,
    pub(crate) request_id: u64,
    pub(crate) session_id: u64,
    pub(crate) agent_session_id: Option<&'a str>,
    pub(crate) prompt_token_ids: Arc<[i32]>,
    pub(crate) observation: GenerationReceiptObservation,
    pub(crate) frontend_request_id: Option<[u8; 16]>,
}

/// Tracks lifecycle-only observations while a generation executes.
///
/// The exact receipt path has its own recorder and finalization because it
/// must read canonical runtime position and optionally export full state. This
/// tracker intentionally records only bounded lifecycle fields, so it is valid
/// for both local and distributed execution.
pub(crate) struct GenerationLifecycleState {
    config: Option<GenerationLifecycleConfig>,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    generated_token_count: usize,
    request_to_first_token_us: Option<u64>,
    termination: Option<GenerationTermination>,
}

impl GenerationLifecycleState {
    pub(crate) fn new(
        config: Option<&GenerationLifecycleConfig>,
        request_id: u64,
        session_id: u64,
        agent_session_id: Option<Box<str>>,
        frontend_request_id: Option<[u8; 16]>,
        prompt_token_ids: &[i32],
    ) -> Self {
        let prompt_token_count = prompt_token_ids.len();
        let prompt_token_ids = config.map(|_| Arc::<[i32]>::from(prompt_token_ids));
        let state = Self {
            config: config.cloned(),
            request_id,
            session_id,
            prompt_token_count,
            generated_token_count: 0,
            request_to_first_token_us: None,
            termination: None,
        };
        if let Some(config) = config {
            config.begin(GenerationStart {
                request_id,
                session_id,
                agent_session_id,
                prompt_token_ids: prompt_token_ids
                    .expect("lifecycle prompt token IDs exist when configured"),
                frontend_request_id,
            });
        }
        state
    }

    pub(crate) fn commit(&mut self, token_id: i32, request_elapsed: Duration) {
        let Some(config) = self.config.as_ref() else {
            return;
        };
        self.generated_token_count = self.generated_token_count.saturating_add(1);
        self.request_to_first_token_us
            .get_or_insert_with(|| u64::try_from(request_elapsed.as_micros()).unwrap_or(u64::MAX));
        config.committed(GenerationCommit {
            request_id: self.request_id,
            session_id: self.session_id,
            generated_token_count: self.generated_token_count,
            token_ids: vec![token_id].into_boxed_slice(),
        });
    }

    pub(crate) fn mark_callback_stop(&mut self) {
        if self.config.is_some() {
            self.termination = Some(GenerationTermination::CallbackStop);
        }
    }

    pub(crate) fn mark_cancelled(&mut self) {
        if self.config.is_some() && self.termination.is_none() {
            self.termination = Some(GenerationTermination::Cancelled);
        }
    }

    pub(crate) fn finish(mut self, generation_succeeded: bool) {
        let Some(config) = self.config.take() else {
            return;
        };
        if generation_succeeded {
            config.finished(GenerationCompletion::new(
                self.request_id,
                self.session_id,
                self.prompt_token_count,
                self.generated_token_count,
                self.termination.unwrap_or(GenerationTermination::MaxTokens),
                self.request_to_first_token_us,
            ));
        } else {
            config.abort(GenerationAbort {
                request_id: self.request_id,
                session_id: self.session_id,
            });
        }
    }
}

impl Drop for GenerationLifecycleState {
    fn drop(&mut self) {
        let Some(config) = self.config.take() else {
            return;
        };
        config.abort(GenerationAbort {
            request_id: self.request_id,
            session_id: self.session_id,
        });
    }
}

trait GenerationReceiptRuntime {
    fn canonical_session_position(&self, session_label: &str) -> Result<u64>;
    fn export_full_state(&mut self, session_label: &str) -> Result<Vec<u8>>;
}

impl GenerationReceiptRuntime for RuntimeState {
    fn canonical_session_position(&self, session_label: &str) -> Result<u64> {
        self.canonical_session_position(session_label)
    }

    fn export_full_state(&mut self, session_label: &str) -> Result<Vec<u8>> {
        self.export_full_state(session_label)
    }
}

impl GenerationReceiptObservation {
    fn new(max_tokens: usize, recording_failures: Arc<AtomicU64>) -> Self {
        Self {
            generated_token_ids: Vec::with_capacity(max_tokens.min(4_096)),
            token_emission_elapsed: Vec::with_capacity(max_tokens.min(4_096)),
            max_tokens,
            recording_failures,
            recording_enabled: true,
            termination: None,
            model_generation_elapsed: None,
        }
    }

    pub(crate) fn record_token(&mut self, token_id: i32, request_elapsed: Duration) {
        if !self.recording_enabled {
            return;
        }
        if self.generated_token_ids.len() >= self.max_tokens {
            self.reject_recording();
            return;
        }
        if self
            .token_emission_elapsed
            .last()
            .is_some_and(|prior| request_elapsed < *prior)
        {
            self.reject_recording();
            return;
        }
        self.generated_token_ids.push(token_id);
        self.token_emission_elapsed.push(request_elapsed);
    }

    fn reject_recording(&mut self) {
        self.recording_enabled = false;
        self.recording_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn is_recording_enabled(&self) -> bool {
        self.recording_enabled
    }

    pub(crate) fn mark_callback_stop(&mut self) {
        self.termination = Some(GenerationTermination::CallbackStop);
    }

    pub(crate) fn mark_cancelled(&mut self) {
        if self.termination.is_none() {
            self.termination = Some(GenerationTermination::Cancelled);
        }
    }

    pub(crate) fn set_model_generation_elapsed(&mut self, elapsed: Duration) {
        self.model_generation_elapsed = Some(elapsed);
    }

    fn finish(self) -> OpenAiResult<FinishedGenerationObservation> {
        let model_generation_elapsed = self.model_generation_elapsed.ok_or_else(|| {
            OpenAiError::backend("generation receipt is missing model-generation timing")
        })?;
        let request_to_token_emission_us = self
            .token_emission_elapsed
            .into_iter()
            .map(duration_us)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(FinishedGenerationObservation {
            generated_token_ids: self.generated_token_ids.into_boxed_slice(),
            termination: self.termination.unwrap_or(GenerationTermination::MaxTokens),
            model_generation_elapsed_us: duration_us(model_generation_elapsed),
            request_to_first_token_us: request_to_token_emission_us.first().copied(),
            request_to_token_emission_us,
        })
    }
}

struct FinishedGenerationObservation {
    generated_token_ids: Box<[i32]>,
    termination: GenerationTermination,
    model_generation_elapsed_us: u64,
    request_to_first_token_us: Option<u64>,
    request_to_token_emission_us: Box<[u64]>,
}

impl StageOpenAiBackend {
    pub(crate) fn deliver_local_generation_receipt(
        &self,
        delivery: LocalGenerationReceiptDelivery<'_>,
    ) -> OpenAiResult<()> {
        let config = delivery.config;
        let request_id = delivery.request_id;
        let session_id = delivery.session_id;
        let receipt = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            build_generation_receipt(&mut *runtime, delivery)
        };
        deliver_generation_receipt_outcome(config, request_id, session_id, receipt)
    }
}

/// Resolves a built (or failed-to-build) receipt into the request's return
/// value. A `build_generation_receipt` failure -- `canonical_session_position`
/// or `export_full_state` bookkeeping failing after a real, successful
/// generation -- is health-only observability: it is reported through
/// [`GenerationReceiptConfig::receipt_unavailable`] and the original
/// generation's success is returned unchanged. The receipt is never on the
/// request's error path (review defect D1).
pub(crate) fn deliver_generation_receipt_outcome(
    config: &GenerationReceiptConfig,
    request_id: u64,
    session_id: u64,
    receipt: OpenAiResult<GenerationReceipt>,
) -> OpenAiResult<()> {
    match receipt {
        Ok(receipt) => record_generation_receipt(config, receipt),
        Err(_build_failure) => {
            config.receipt_unavailable(request_id, session_id);
            Ok(())
        }
    }
}

fn build_generation_receipt(
    runtime: &mut dyn GenerationReceiptRuntime,
    delivery: LocalGenerationReceiptDelivery<'_>,
) -> OpenAiResult<GenerationReceipt> {
    let observation = delivery.observation.finish()?;
    let final_session_position = runtime
        .canonical_session_position(delivery.session_label)
        .map_err(openai_backend_error)?;
    let full_state = if delivery.config.exports_full_state() {
        let bytes = runtime
            .export_full_state(delivery.session_label)
            .map_err(openai_backend_error)?;
        Some(state_digest(&bytes)?)
    } else {
        None
    };
    Ok(GenerationReceipt {
        request_id: delivery.request_id,
        session_id: delivery.session_id,
        agent_session_id: delivery.agent_session_id.map(Into::into),
        prompt_token_count: delivery.prompt_token_ids.len(),
        prompt_token_digest: generation_token_id_digest(&delivery.prompt_token_ids),
        prompt_token_ids: delivery.prompt_token_ids,
        generated_token_ids: observation.generated_token_ids,
        final_session_position,
        termination: observation.termination,
        model_generation_elapsed_us: observation.model_generation_elapsed_us,
        request_to_first_token_us: observation.request_to_first_token_us,
        request_to_token_emission_us: observation.request_to_token_emission_us,
        full_state,
        frontend_request_id: delivery.frontend_request_id,
    })
}

fn record_generation_receipt(
    config: &GenerationReceiptConfig,
    receipt: GenerationReceipt,
) -> OpenAiResult<()> {
    config.record(receipt);
    Ok(())
}

pub(crate) fn complete_generation_before_cleanup<T>(
    generation_result: OpenAiResult<T>,
    deliver_receipt: impl FnOnce() -> OpenAiResult<()>,
    cleanup: impl FnOnce(),
) -> OpenAiResult<T> {
    let receipt_result = deliver_receipt();
    cleanup();
    match generation_result {
        Ok(output) => {
            receipt_result?;
            Ok(output)
        }
        Err(primary) => Err(primary),
    }
}

fn state_digest(bytes: &[u8]) -> OpenAiResult<GenerationStateDigest> {
    let byte_length = u64::try_from(bytes.len())
        .map_err(|_| OpenAiError::backend("full-state byte length exceeds u64"))?;
    Ok(GenerationStateDigest {
        byte_length,
        blake3_digest: *blake3::hash(bytes).as_bytes(),
    })
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::Mutex;
    use std::sync::mpsc::sync_channel;
    use std::thread;

    use super::*;

    fn test_observation(max_tokens: usize) -> GenerationReceiptObservation {
        GenerationReceiptObservation::new(max_tokens, Arc::new(AtomicU64::new(0)))
    }

    struct FakeRuntime {
        position: Result<u64, &'static str>,
        full_state: Result<Vec<u8>, &'static str>,
    }

    impl GenerationReceiptRuntime for FakeRuntime {
        fn canonical_session_position(&self, _session_label: &str) -> Result<u64> {
            self.position.map_err(anyhow::Error::msg)
        }

        fn export_full_state(&mut self, _session_label: &str) -> Result<Vec<u8>> {
            self.full_state.clone().map_err(anyhow::Error::msg)
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        receipts: Mutex<Vec<GenerationReceipt>>,
        error: Option<&'static str>,
    }

    #[derive(Default)]
    struct RecordingIngress {
        observations: Mutex<Vec<GenerationLifecycleObservation>>,
        error: Option<&'static str>,
    }

    impl GenerationLifecycleIngress for RecordingIngress {
        fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
            self.observations.lock().unwrap().push(observation);
            self.error
                .map_or(Ok(()), |error| Err(anyhow::anyhow!(error)))
        }
    }

    impl GenerationReceiptSink for RecordingSink {
        fn begin(&self, _start: &GenerationStart) -> Result<()> {
            Ok(())
        }

        fn committed(&self, _commit: &GenerationCommit) -> Result<()> {
            Ok(())
        }

        fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
            Ok(())
        }

        fn record(&self, receipt: &GenerationReceipt) -> Result<()> {
            self.receipts.lock().unwrap().push(receipt.clone());
            self.error
                .map_or(Ok(()), |error| Err(anyhow::anyhow!(error)))
        }
    }

    struct BlockingReceiptSink {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl GenerationReceiptSink for BlockingReceiptSink {
        fn begin(&self, _start: &GenerationStart) -> Result<()> {
            self.entered.wait();
            self.release.wait();
            Ok(())
        }

        fn committed(&self, _commit: &GenerationCommit) -> Result<()> {
            Ok(())
        }

        fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
            Ok(())
        }

        fn record(&self, _receipt: &GenerationReceipt) -> Result<()> {
            Ok(())
        }
    }

    fn wait_for_receipts(sink: &RecordingSink, expected: usize) {
        for _ in 0..100 {
            if sink.receipts.lock().unwrap().len() >= expected {
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("timed out waiting for {expected} generation receipts");
    }

    #[test]
    fn token_digest_is_stable_and_order_sensitive() {
        let digest = generation_token_id_digest(&[-1, 0, 1, i32::MAX]);
        assert_eq!(
            digest,
            [
                0x1a, 0xe4, 0xc4, 0x37, 0x7c, 0xce, 0x52, 0xaa, 0x76, 0x66, 0x8c, 0x07, 0xd0, 0x16,
                0xaa, 0x7b, 0x19, 0xfe, 0xd5, 0x8c, 0xbd, 0x35, 0x89, 0x06, 0xe6, 0x10, 0x8f, 0x03,
                0xf7, 0xbf, 0x33, 0x3a,
            ]
        );
        assert_ne!(digest, generation_token_id_digest(&[0, -1, 1, i32::MAX]));
    }

    #[test]
    fn typed_ingress_preserves_authoritative_observation_order_without_an_adapter_queue() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = GenerationReceiptConfig::from_lifecycle_ingress(ingress.clone());
        config.begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3, 4]),
            frontend_request_id: None,
        });
        config.committed(GenerationCommit {
            request_id: 1,
            session_id: 2,
            generated_token_count: 1,
            token_ids: vec![5].into_boxed_slice(),
        });
        config.abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        });

        let observations = ingress.observations.lock().unwrap();
        assert!(matches!(
            observations.as_slice(),
            [
                GenerationLifecycleObservation::Started(_),
                GenerationLifecycleObservation::Committed(_),
                GenerationLifecycleObservation::Aborted(_)
            ]
        ));
        assert_eq!(config.delivery_failures(), 0);
    }

    #[test]
    fn typed_ingress_rejection_is_accounted_without_failing_generation() {
        let ingress = Arc::new(RecordingIngress {
            observations: Mutex::new(Vec::new()),
            error: Some("ingress full"),
        });
        let config = GenerationReceiptConfig::from_lifecycle_ingress(ingress);
        config.abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        });
        assert_eq!(config.delivery_failures(), 1);
    }

    #[test]
    fn lifecycle_only_config_finishes_without_constructing_a_receipt() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = GenerationLifecycleConfig::from_ingress(ingress.clone());
        let mut state = GenerationLifecycleState::new(Some(&config), 1, 2, None, None, &[3, 4]);
        state.commit(5, Duration::from_micros(7));
        state.finish(true);

        let observations = ingress.observations.lock().unwrap();
        assert!(matches!(
            observations.as_slice(),
            [
                GenerationLifecycleObservation::Started(_),
                GenerationLifecycleObservation::Committed(_),
                GenerationLifecycleObservation::Finished(_)
            ]
        ));
        let GenerationLifecycleObservation::Finished(completion) = &observations[2] else {
            unreachable!("lifecycle state must finish with a summary");
        };
        assert_eq!(completion.prompt_token_count, 2);
        assert_eq!(completion.generated_token_count, 1);
        assert_eq!(completion.request_to_first_token_us, Some(7));
        assert_eq!(completion.termination, GenerationTermination::MaxTokens);
        assert_eq!(config.delivery_failures(), 0);
    }

    #[test]
    fn dropping_an_unfinished_lifecycle_aborts_exactly_once() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = GenerationLifecycleConfig::from_ingress(ingress.clone());
        {
            let _state = GenerationLifecycleState::new(Some(&config), 7, 8, None, None, &[9]);
        }

        let observations = ingress.observations.lock().unwrap();
        assert!(matches!(
            observations.as_slice(),
            [
                GenerationLifecycleObservation::Started(_),
                GenerationLifecycleObservation::Aborted(GenerationAbort {
                    request_id: 7,
                    session_id: 8,
                })
            ]
        ));
    }

    #[test]
    fn queued_receipt_unavailable_counts_a_full_queue() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let sink = Arc::new(BlockingReceiptSink {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        });
        let ingress = QueuedGenerationReceiptSink::new(sink);
        ingress
            .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
                request_id: 1,
                session_id: 2,
                agent_session_id: None,
                prompt_token_ids: Arc::from([]),
                frontend_request_id: None,
            }))
            .unwrap();
        entered.wait();
        for request_id in 0..GENERATION_RECEIPT_QUEUE_CAPACITY {
            ingress
                .try_submit(GenerationLifecycleObservation::Aborted(GenerationAbort {
                    request_id: request_id as u64,
                    session_id: 2,
                }))
                .unwrap();
        }

        ingress.receipt_unavailable(&GenerationAbort {
            request_id: 99,
            session_id: 2,
        });
        assert_eq!(ingress.delivery_failures(), 1);
        release.wait();
    }

    #[test]
    fn queued_receipt_unavailable_counts_when_abort_reaches_the_worker() {
        let sink = Arc::new(RecordingSink::default());
        let ingress = QueuedGenerationReceiptSink::new(sink);

        ingress.receipt_unavailable(&GenerationAbort {
            request_id: 9,
            session_id: 10,
        });

        assert_eq!(ingress.delivery_failures(), 1);
    }

    #[test]
    fn queued_receipt_unavailable_counts_a_disconnected_queue() {
        let sink = Arc::new(RecordingSink::default());
        let mut ingress = QueuedGenerationReceiptSink::new(sink);
        let (replacement, receiver) = sync_channel(1);
        let sender = std::mem::replace(&mut ingress.sender, replacement);
        drop(sender);
        drop(receiver);

        ingress.receipt_unavailable(&GenerationAbort {
            request_id: 7,
            session_id: 8,
        });
        assert_eq!(ingress.delivery_failures(), 1);
    }

    #[test]
    fn receipt_prompt_evidence_preserves_exact_signed_token_ids() {
        let prompt = [-1, 0, 7, i32::MAX];
        let receipt = GenerationReceipt {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_count: prompt.len(),
            prompt_token_digest: generation_token_id_digest(&prompt),
            prompt_token_ids: Arc::from(prompt),
            generated_token_ids: vec![9].into_boxed_slice(),
            final_session_position: 4,
            termination: GenerationTermination::MaxTokens,
            model_generation_elapsed_us: 3,
            request_to_first_token_us: Some(1),
            request_to_token_emission_us: vec![1].into_boxed_slice(),
            full_state: None,
            frontend_request_id: None,
        };
        assert_eq!(receipt.prompt_token_ids.as_ref(), prompt);
        assert_eq!(receipt.prompt_token_count, receipt.prompt_token_ids.len());
        assert_eq!(
            receipt.prompt_token_digest,
            generation_token_id_digest(&receipt.prompt_token_ids)
        );
        assert_eq!(
            receipt.generated_token_ids.len(),
            receipt.request_to_token_emission_us.len()
        );
        assert_eq!(
            receipt.request_to_first_token_us,
            receipt.request_to_token_emission_us.first().copied()
        );
    }

    #[test]
    fn observation_keeps_the_callback_stopping_token() {
        let mut observation = test_observation(3);
        observation.record_token(7, Duration::from_micros(11));
        observation.record_token(8, Duration::from_micros(17));
        observation.mark_callback_stop();
        observation.set_model_generation_elapsed(Duration::from_micros(42));
        let finished = observation.finish().unwrap();
        assert_eq!(&*finished.generated_token_ids, &[7, 8]);
        assert_eq!(finished.termination, GenerationTermination::CallbackStop);
        assert_eq!(finished.model_generation_elapsed_us, 42);
        assert_eq!(finished.request_to_first_token_us, Some(11));
        assert_eq!(&*finished.request_to_token_emission_us, &[11, 17]);
    }

    #[test]
    fn observation_bookkeeping_failure_is_counted_without_failing_generation() {
        let config =
            GenerationReceiptConfig::from_lifecycle_ingress(Arc::new(RecordingIngress::default()));
        let mut observation = config.observation(1);
        observation.record_token(7, Duration::ZERO);
        observation.record_token(8, Duration::from_micros(1));
        observation.record_token(9, Duration::from_micros(2));
        assert_eq!(config.recording_failures(), 1);
        assert_eq!(observation.generated_token_ids, [7]);
    }

    #[test]
    fn observation_rejects_non_monotonic_token_timing() {
        let config =
            GenerationReceiptConfig::from_lifecycle_ingress(Arc::new(RecordingIngress::default()));
        let mut observation = config.observation(2);
        observation.record_token(7, Duration::from_micros(2));
        observation.record_token(8, Duration::from_micros(1));
        assert_eq!(config.recording_failures(), 1);
        assert_eq!(observation.generated_token_ids, [7]);
    }

    #[test]
    fn cancellation_precedes_default_max_token_termination() {
        let mut observation = test_observation(1);
        observation.mark_cancelled();
        observation.set_model_generation_elapsed(Duration::ZERO);
        assert_eq!(
            observation.finish().unwrap().termination,
            GenerationTermination::Cancelled
        );

        let mut max_tokens = test_observation(0);
        max_tokens.set_model_generation_elapsed(Duration::ZERO);
        let finished = max_tokens.finish().unwrap();
        assert_eq!(finished.termination, GenerationTermination::MaxTokens);
        assert_eq!(finished.request_to_first_token_us, None);
        assert!(finished.request_to_token_emission_us.is_empty());
    }

    #[test]
    fn state_digest_binds_length_and_bytes() {
        let digest = state_digest(b"state").unwrap();
        assert_eq!(digest.byte_length, 5);
        assert_eq!(digest.blake3_digest, *blake3::hash(b"state").as_bytes());
        assert_ne!(
            digest.blake3_digest,
            state_digest(b"state!").unwrap().blake3_digest
        );
    }

    #[test]
    fn model_free_delivery_validates_position_exports_state_without_blocking_on_sink_errors() {
        let sink = Arc::new(RecordingSink::default());
        let config = GenerationReceiptConfig::new(sink.clone()).with_full_state_digest(true);
        let mut observation = test_observation(1);
        observation.record_token(9, Duration::from_micros(5));
        observation.set_model_generation_elapsed(Duration::from_micros(17));
        let mut runtime = FakeRuntime {
            position: Ok(4),
            full_state: Ok(b"state".to_vec()),
        };
        let receipt = build_generation_receipt(
            &mut runtime,
            LocalGenerationReceiptDelivery {
                config: &config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                agent_session_id: Some("agent-session"),
                prompt_token_ids: Arc::from([4, 5, 6]),
                observation,
                frontend_request_id: None,
            },
        )
        .unwrap();
        record_generation_receipt(&config, receipt).unwrap();
        wait_for_receipts(&sink, 1);
        let receipts = sink.receipts.lock().unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].final_session_position, 4);
        assert_eq!(receipts[0].generated_token_ids.as_ref(), &[9]);
        assert_eq!(
            receipts[0].agent_session_id.as_deref(),
            Some("agent-session")
        );
        assert_eq!(
            receipts[0].full_state.as_ref().unwrap().blake3_digest,
            *blake3::hash(b"state").as_bytes()
        );
        drop(receipts);

        let failing_position = build_generation_receipt(
            &mut FakeRuntime {
                position: Err("position mismatch"),
                full_state: Ok(Vec::new()),
            },
            LocalGenerationReceiptDelivery {
                config: &config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                agent_session_id: None,
                prompt_token_ids: Arc::from([]),
                observation: {
                    let mut observation = test_observation(0);
                    observation.set_model_generation_elapsed(Duration::ZERO);
                    observation
                },
                frontend_request_id: None,
            },
        )
        .unwrap_err();
        assert!(failing_position.to_string().contains("position mismatch"));

        let failing_sink = Arc::new(RecordingSink {
            receipts: Mutex::new(Vec::new()),
            error: Some("sink failed"),
        });
        let failing_config = GenerationReceiptConfig::new(failing_sink);
        let mut observation = test_observation(0);
        observation.set_model_generation_elapsed(Duration::ZERO);
        let receipt = build_generation_receipt(
            &mut runtime,
            LocalGenerationReceiptDelivery {
                config: &failing_config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                agent_session_id: None,
                prompt_token_ids: Arc::from([]),
                observation,
                frontend_request_id: None,
            },
        )
        .unwrap();
        record_generation_receipt(&failing_config, receipt).unwrap();
        for _ in 0..100 {
            if failing_config.delivery_failures() == 1 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(failing_config.delivery_failures(), 1);
    }

    #[test]
    fn receipt_build_failure_never_fails_the_request_and_counts_as_a_delivery_failure() {
        #[derive(Default)]
        struct ReceiptUnavailableCountingIngress {
            receipt_unavailable_calls: Mutex<Vec<GenerationAbort>>,
        }

        impl GenerationLifecycleIngress for ReceiptUnavailableCountingIngress {
            fn try_submit(&self, _observation: GenerationLifecycleObservation) -> Result<()> {
                Ok(())
            }

            fn delivery_failures(&self) -> u64 {
                self.receipt_unavailable_calls.lock().unwrap().len() as u64
            }

            fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
                self.receipt_unavailable_calls
                    .lock()
                    .unwrap()
                    .push(*unavailable);
            }
        }

        let ingress = Arc::new(ReceiptUnavailableCountingIngress::default());
        let config = GenerationReceiptConfig::from_lifecycle_ingress(ingress.clone());
        let mut runtime = FakeRuntime {
            position: Err("session session-x has no tracked position"),
            full_state: Ok(Vec::new()),
        };
        let mut observation = test_observation(0);
        observation.set_model_generation_elapsed(Duration::ZERO);
        let receipt = build_generation_receipt(
            &mut runtime,
            LocalGenerationReceiptDelivery {
                config: &config,
                session_label: "session-x",
                request_id: 7,
                session_id: 8,
                agent_session_id: None,
                prompt_token_ids: Arc::from([]),
                observation,
                frontend_request_id: None,
            },
        );
        assert!(receipt.is_err());
        assert_eq!(config.delivery_failures(), 0);

        let outcome = deliver_generation_receipt_outcome(&config, 7, 8, receipt);
        assert!(outcome.is_ok());
        assert_eq!(config.delivery_failures(), 1);
        assert_eq!(
            ingress.receipt_unavailable_calls.lock().unwrap().as_slice(),
            [GenerationAbort {
                request_id: 7,
                session_id: 8,
            }]
        );
    }

    #[test]
    fn receipt_delivery_precedes_cleanup_and_cleanup_survives_sink_failure() {
        let events = Mutex::new(Vec::new());
        let error = complete_generation_before_cleanup(
            Ok(()),
            || {
                events.lock().unwrap().push("receipt");
                Err(OpenAiError::backend("sink failed"))
            },
            || events.lock().unwrap().push("cleanup"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("sink failed"));
        assert_eq!(*events.lock().unwrap(), ["receipt", "cleanup"]);

        let events = Mutex::new(Vec::new());
        let generation_error = complete_generation_before_cleanup::<()>(
            Err(OpenAiError::backend("generation failed")),
            || {
                events.lock().unwrap().push("abort");
                Ok(())
            },
            || events.lock().unwrap().push("cleanup"),
        )
        .unwrap_err();
        assert!(generation_error.to_string().contains("generation failed"));
        assert_eq!(*events.lock().unwrap(), ["abort", "cleanup"]);
    }

    #[test]
    fn lifecycle_observations_close_before_cleanup() {
        #[derive(Clone)]
        struct OrderedIngress(Arc<Mutex<Vec<&'static str>>>);

        impl GenerationLifecycleIngress for OrderedIngress {
            fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
                let label = match observation {
                    GenerationLifecycleObservation::Started(_) => "started",
                    GenerationLifecycleObservation::Committed(_) => "committed",
                    GenerationLifecycleObservation::Aborted(_) => "aborted",
                    GenerationLifecycleObservation::Finished(_) => "finished",
                    GenerationLifecycleObservation::Completed(_) => "completed",
                };
                self.0.lock().unwrap().push(label);
                Ok(())
            }
        }

        let events = Arc::new(Mutex::new(Vec::new()));
        let config = GenerationReceiptConfig::from_lifecycle_ingress(Arc::new(OrderedIngress(
            Arc::clone(&events),
        )));
        config.begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        });
        config.committed(GenerationCommit {
            request_id: 1,
            session_id: 2,
            generated_token_count: 1,
            token_ids: Box::new([4]),
        });
        complete_generation_before_cleanup(
            Ok(()),
            || {
                config.abort(GenerationAbort {
                    request_id: 1,
                    session_id: 2,
                });
                Ok(())
            },
            || events.lock().unwrap().push("cleanup"),
        )
        .unwrap();

        assert_eq!(
            *events.lock().unwrap(),
            ["started", "committed", "aborted", "cleanup"]
        );
    }
}
