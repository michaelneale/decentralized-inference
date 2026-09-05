//! Host-owned bridge from OpenAI frontend lifecycle boundaries to logging.
//!
//! The frontend emits bounded metadata. This adapter owns the matching request
//! guards and hands the production transport a narrow, body-only capture view.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use mesh_llm_events::logging::{
    events::{LifecycleEvent, TokenUsage},
    identifiers::{AttemptId, RequestId},
    replay::ReplayChannel,
};
use openai_frontend::{
    OpenAiBackendOperation, OpenAiFailure, OpenAiLifecycleContext, OpenAiLifecycleEvent,
    OpenAiLifecycleObserver, OpenAiRejection, OpenAiRequestMethod, OpenAiTerminalResult,
    OpenAiUsage,
};

use super::raw_mesh_lifecycle::FrontendAdmissionDecision;
use super::{
    ArtifactUnavailableReason, LifecycleGuard, LoggingService, ProxyAttemptFinish,
    RawMeshLifecycleOwners, RawMeshProxyAttempt, RawMeshRequestLifecycle, RequestSummaryMetadata,
    TerminalOutcome,
};

/// Fixed upper bound for requests owned by one embedded frontend observer.
///
/// `pub(crate)` (review defect D12) so `logging::raw_mesh_lifecycle` and
/// `network::openai::runtime_events` import this single definition instead
/// of each carrying their own copy; pinned against
/// `runtime_events::config::REQUEST_ROOT_BOUND` by the `request_root_bound`
/// contract test in this module's `tests` block.
pub(crate) const MAX_TRACKED_REQUESTS: usize = 1_024;

/// The host-owned lifecycle attachment for one parsed OpenAI ingress request.
///
/// The attachment keeps terminal ownership at the ingress boundary. Routing
/// receives only the [`OpenAiRouteObserver`] view below, which can publish
/// bounded route/attempt metadata but cannot admit or terminalize a request.
pub(crate) struct OpenAiLifecycleAttachment {
    parent: Option<RawMeshRequestLifecycle>,
    capture: Option<Arc<dyn OpenAiArtifactCapture>>,
    terminalized: bool,
}

/// Narrow production artifact boundary for a parsed OpenAI request. The
/// transport can only submit body bytes and fixed artifact kinds; it never
/// receives filesystem or store access.
pub(crate) trait OpenAiArtifactCapture: Send + Sync {
    fn body_limit_bytes(&self) -> usize {
        256 * 1024
    }

    fn capture_body(
        &self,
        request_id: RequestId,
        kind: &'static str,
        content: &[u8],
        media_kind: Option<&str>,
    );
    fn capture_unavailable(
        &self,
        request_id: RequestId,
        kind: &'static str,
        reason: ArtifactUnavailableReason,
    );
}

/// Request-local bounded assembly for a client-visible SSE response body.
/// The configured per-artifact limit is enforced while frames arrive, before
/// the persistence queue is offered any body bytes.
pub(crate) struct OpenAiStreamArtifactCapture {
    capture: Arc<dyn OpenAiArtifactCapture>,
    request_id: RequestId,
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
    finalized: bool,
}

impl OpenAiStreamArtifactCapture {
    fn new(capture: Arc<dyn OpenAiArtifactCapture>, request_id: RequestId, limit: usize) -> Self {
        Self {
            capture,
            request_id,
            bytes: Vec::new(),
            limit,
            overflowed: false,
            finalized: false,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.overflowed {
            return;
        }
        let Some(next_len) = self.bytes.len().checked_add(bytes.len()) else {
            self.overflowed = true;
            self.bytes.clear();
            return;
        };
        if next_len > self.limit {
            self.overflowed = true;
            self.bytes.clear();
            return;
        }
        self.bytes.extend_from_slice(bytes);
    }

    fn complete(mut self) {
        self.finalized = true;
        if self.overflowed {
            self.capture.capture_unavailable(
                self.request_id,
                "response",
                ArtifactUnavailableReason::CaptureContentLimitExceeded,
            );
        } else {
            self.capture.capture_body(
                self.request_id,
                "response",
                &self.bytes,
                Some("text/event-stream"),
            );
        }
    }
}

impl Drop for OpenAiStreamArtifactCapture {
    fn drop(&mut self) {
        if !self.finalized {
            self.capture.capture_unavailable(
                self.request_id,
                "response",
                ArtifactUnavailableReason::StreamingResponseNotAssembled,
            );
        }
    }
}

/// Metadata-only route view handed to downstream dispatch code.
///
/// An empty view is the normal fail-open value when logging is disabled,
/// retired, or bounded admission cannot allocate a parent.
#[derive(Clone, Copy, Default)]
pub(crate) struct OpenAiRouteObserver<'a> {
    parent: Option<&'a RawMeshRequestLifecycle>,
    capture: Option<&'a Arc<dyn OpenAiArtifactCapture>>,
    request_id: Option<RequestId>,
}

/// An existing lifecycle attempt plus its private logging timestamp.
///
/// Downstream transport code can only finish this through its observer; it
/// cannot terminalize the request parent or enqueue arbitrary proxy records.
pub(crate) struct OpenAiRouteAttempt(RawMeshProxyAttempt);

impl OpenAiLifecycleAttachment {
    pub(crate) fn new(parent: Option<RawMeshRequestLifecycle>) -> Self {
        Self {
            parent,
            capture: None,
            terminalized: false,
        }
    }

    pub(crate) fn with_capture(
        parent: Option<RawMeshRequestLifecycle>,
        capture: Arc<dyn OpenAiArtifactCapture>,
    ) -> Self {
        Self {
            parent,
            capture: Some(capture),
            terminalized: false,
        }
    }

    pub(crate) fn unowned() -> Self {
        Self::new(None)
    }

    pub(crate) fn owns_parent(&self) -> bool {
        self.parent.is_some()
    }

    pub(crate) fn route_observer(&self) -> OpenAiRouteObserver<'_> {
        OpenAiRouteObserver {
            parent: self.parent.as_ref(),
            capture: self.capture.as_ref(),
            request_id: self
                .parent
                .as_ref()
                .map(RawMeshRequestLifecycle::request_id),
        }
    }

    /// Capture only the decoded request body. Raw HTTP headers are never made
    /// available here, so credentials cannot enter artifact capture.
    pub(crate) fn capture_request_body(&self, content: &[u8], media_kind: Option<&str>) {
        if let (Some(parent), Some(capture)) = (self.parent.as_ref(), self.capture.as_ref()) {
            // The parsed body is the only request data admitted here. Do not
            // carry raw request headers into artifact persistence. The caller
            // may provide only the closed, ingress-validated semantic media
            // kind for a recognized JSON endpoint; opaque bodies stay opaque.
            capture.capture_body(parent.request_id(), "request", content, media_kind);
        }
    }

    /// Terminalize exactly once from the owning ingress boundary.
    pub(crate) fn terminal(&mut self, outcome: TerminalOutcome) {
        if self.terminalized {
            return;
        }
        self.terminalized = true;
        if let Some(parent) = self.parent.as_ref() {
            parent.terminal(outcome);
        }
    }
}

impl Drop for OpenAiLifecycleAttachment {
    fn drop(&mut self) {
        if !self.terminalized {
            self.terminal(TerminalOutcome::Dropped(Some(
                "openai_ingress_scope_dropped".into(),
            )));
        }
    }
}

impl<'a> OpenAiRouteObserver<'a> {
    #[cfg(test)]
    pub(crate) fn capture_test_observer(
        request_id: RequestId,
        capture: &'a Arc<dyn OpenAiArtifactCapture>,
    ) -> Self {
        Self {
            parent: None,
            capture: Some(capture),
            request_id: Some(request_id),
        }
    }

    /// Capture a client-visible non-streaming response body. The capture
    /// implementation applies the canonical redactor and configured byte
    /// limit; HTTP headers are intentionally never passed here.
    pub(crate) fn capture_response_body(&self, content: &[u8], media_kind: Option<&str>) {
        if let (Some(request_id), Some(capture)) = (self.request_id, self.capture) {
            capture.capture_body(request_id, "response", content, media_kind);
        }
    }

    /// Persist an explicit state marker when a streaming response cannot be
    /// represented as a bounded semantic payload. This is deliberately not a
    /// fabricated status-only response body.
    pub(crate) fn capture_response_unavailable(&self, reason: ArtifactUnavailableReason) {
        if let (Some(request_id), Some(capture)) = (self.request_id, self.capture) {
            capture.capture_unavailable(request_id, "response", reason);
        }
    }

    pub(crate) fn begin_stream_response_capture(&self) -> Option<OpenAiStreamArtifactCapture> {
        let capture = self.capture?;
        let request_id = self.request_id?;
        Some(OpenAiStreamArtifactCapture::new(
            Arc::clone(capture),
            request_id,
            capture.body_limit_bytes(),
        ))
    }

    pub(crate) fn complete_stream_response_capture(
        &self,
        capture: Option<OpenAiStreamArtifactCapture>,
    ) {
        let Some(capture) = capture else {
            return;
        };
        capture.complete();
    }

    pub(crate) fn abandon_stream_response_capture(
        &self,
        capture: Option<OpenAiStreamArtifactCapture>,
    ) {
        drop(capture);
    }

    pub(crate) fn route_selected(&self, model: Option<&str>) {
        if let Some(parent) = self.parent {
            parent.route_selected(model);
        }
    }

    /// Record one bounded route selection with provider/engine metadata.
    ///
    /// The ingress owner remains responsible for the parent lifecycle; this
    /// observer only exposes the metadata-only route boundary to downstream
    /// transports. The raw lifecycle owner bounds and sanitizes both labels.
    pub(crate) fn route_selected_with_metadata(
        &self,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
    ) {
        if let Some(parent) = self.parent {
            parent.route_selected_with_metadata(model, provider, engine);
        }
    }

    pub(crate) fn stream_started(&self, model: Option<&str>) {
        if let Some(parent) = self.parent {
            parent.stream_started(model);
        }
    }

    /// Record the first bounded stream chunk. The canonical event envelope
    /// has no separate first-token variant, so the first `stream_chunk` marks
    /// that boundary without capturing token text or usage.
    pub(crate) fn stream_first_token(&self) {
        if let Some(parent) = self.parent {
            parent.stream_first_token();
        }
    }

    pub(crate) fn stream_chunk(&self) {
        if let Some(parent) = self.parent {
            parent.stream_chunk();
        }
    }

    pub(crate) fn stream_completed(
        &self,
        usage: Option<mesh_llm_events::logging::events::TokenUsage>,
    ) {
        if let Some(parent) = self.parent {
            parent.stream_completed(usage);
        }
    }

    /// Record a bounded static stream error/cancellation label.
    pub(crate) fn stream_error(&self, label: &'static str) {
        if let Some(parent) = self.parent {
            parent.stream_error(label);
        }
    }

    pub(crate) fn stream_cancelled(&self) {
        if let Some(parent) = self.parent {
            parent.stream_cancelled();
        }
    }

    pub(crate) fn start_attempt(&self) -> Option<AttemptId> {
        self.parent.map(RawMeshRequestLifecycle::start_attempt)
    }

    /// Start a lifecycle attempt that may later persist one bounded proxy
    /// record. Empty observers remain fully fail-open.
    pub(crate) fn start_proxy_attempt(&self) -> Option<OpenAiRouteAttempt> {
        self.parent
            .map(RawMeshRequestLifecycle::start_proxy_attempt)
            .map(OpenAiRouteAttempt)
    }

    pub(crate) fn complete_attempt(&self, attempt_id: Option<AttemptId>, status_code: u16) {
        if let (Some(parent), Some(attempt_id)) = (self.parent, attempt_id) {
            parent.complete_attempt(attempt_id, status_code);
        }
    }

    pub(crate) fn fail_attempt(&self, attempt_id: Option<AttemptId>, label: &'static str) {
        if let (Some(parent), Some(attempt_id)) = (self.parent, attempt_id) {
            parent.fail_attempt(attempt_id, label);
        }
    }

    /// Finish one transport attempt and enqueue a metadata-only proxy record.
    /// This retains the ingress attachment as the sole parent terminal owner.
    pub(crate) fn finish_proxy_attempt(
        &self,
        attempt: Option<OpenAiRouteAttempt>,
        finish: ProxyAttemptFinish,
    ) {
        if let (Some(parent), Some(OpenAiRouteAttempt(attempt))) = (self.parent, attempt) {
            parent.finish_proxy_attempt(attempt, finish);
        }
    }
}

/// Metadata-only OpenAI frontend lifecycle observer owned by the host runtime.
pub(crate) struct OpenAiLifecycleLoggingAdapter {
    service: Arc<LoggingService>,
    raw_mesh_owners: Arc<RawMeshLifecycleOwners>,
    tracked: Mutex<TrackedRequests>,
}

#[derive(Default)]
struct TrackedRequests {
    requests: HashMap<RequestId, TrackedRequest>,
    insertion_order: VecDeque<RequestId>,
}

enum TrackedRequest {
    Active(ActiveRequest),
    Terminal,
}

struct ActiveRequest {
    guard: LifecycleGuard,
    backend_attempt: Option<(OpenAiBackendOperation, AttemptId)>,
    backend_stream_first_item: bool,
    usage: Option<OpenAiUsage>,
}

impl ActiveRequest {
    fn new(guard: LifecycleGuard) -> Self {
        Self {
            guard,
            backend_attempt: None,
            backend_stream_first_item: false,
            usage: None,
        }
    }
}

impl OpenAiLifecycleLoggingAdapter {
    pub(crate) fn new(
        service: Arc<LoggingService>,
        raw_mesh_owners: Arc<RawMeshLifecycleOwners>,
    ) -> Self {
        Self {
            service,
            raw_mesh_owners,
            tracked: Mutex::new(TrackedRequests::default()),
        }
    }

    fn admit(&self, context: &OpenAiLifecycleContext) {
        let request_id = context.request_id;
        self.raw_mesh_owners.admit_frontend(request_id, || {
            let mut tracked = lock_recover(&self.tracked);
            if tracked.requests.contains_key(&request_id) {
                return FrontendAdmissionDecision::Rejected;
            }
            let Ok(evicted) = tracked.make_room() else {
                return FrontendAdmissionDecision::Rejected;
            };

            let (guard, _) = self.service.register_request_with_metadata(
                request_id,
                RequestSummaryMetadata::from_openai_frontend_route(context.route)
                    .with_source(Some("direct_http"))
                    .with_method(Some(openai_method_label(context.method))),
            );
            tracked.requests.insert(
                request_id,
                TrackedRequest::Active(ActiveRequest::new(guard)),
            );
            tracked.insertion_order.push_back(request_id);
            FrontendAdmissionDecision::Registered { evicted }
        });
    }

    fn backend_dispatched(&self, request_id: RequestId, operation: OpenAiBackendOperation) {
        let guard = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            active.guard.clone()
        };

        let metadata = RequestSummaryMetadata::from_parts(
            None,
            None,
            Some("openai_frontend"),
            Some(operation_label(operation)),
        );
        self.service
            .merge_request_metadata(request_id, metadata.clone());
        let event = LifecycleEvent::RouteSelected {
            model: None,
            provider: metadata.provider().map(str::to_owned),
            engine: metadata.engine().map(str::to_owned),
        };
        self.enqueue_operation_event(request_id, event);
        let attempt_id = self.service.start_attempt(request_id, &guard);
        let mut tracked = lock_recover(&self.tracked);
        if let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) {
            active.backend_attempt = Some((operation, attempt_id));
        }
    }

    fn backend_terminal(
        &self,
        request_id: RequestId,
        operation: OpenAiBackendOperation,
        result: OpenAiTerminalResult,
    ) {
        let attempt_id = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            match active.backend_attempt {
                Some((attempt_operation, attempt_id)) if attempt_operation == operation => {
                    active.backend_attempt = None;
                    Some(attempt_id)
                }
                _ => None,
            }
        };
        let Some(attempt_id) = attempt_id else {
            return;
        };
        match result {
            OpenAiTerminalResult::Completed { status_code }
            | OpenAiTerminalResult::CompletedWithUsage { status_code, .. } => {
                self.service
                    .complete_attempt(request_id, attempt_id, Some(status_code));
            }
            OpenAiTerminalResult::Failed { failure, .. } => {
                self.service.fail_attempt(
                    request_id,
                    attempt_id,
                    failure_label(failure).to_owned(),
                );
            }
        }
    }

    fn backend_stream_first_item(&self, request_id: RequestId) {
        let should_emit = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.backend_stream_first_item {
                false
            } else {
                active.backend_stream_first_item = true;
                true
            }
        };
        if should_emit {
            self.enqueue_operation_event(request_id, LifecycleEvent::BackendStreamFirstItem);
        }
    }

    fn response_completed(&self, request_id: RequestId, usage: OpenAiUsage) {
        let should_emit = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.usage.is_some() {
                false
            } else {
                active.usage = Some(usage);
                true
            }
        };
        if should_emit {
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::UsageRecorded {
                    prompt_tokens: Some(u64::from(usage.prompt_tokens)),
                    cached_prompt_tokens: usage.cached_tokens.map(u64::from),
                    completion_tokens: Some(u64::from(usage.completion_tokens)),
                    total_tokens: Some(u64::from(usage.total_tokens)),
                },
            );
        }
    }

    fn stream_terminal(&self, request_id: RequestId, result: OpenAiTerminalResult) {
        let usage = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            active.usage
        };
        match result {
            OpenAiTerminalResult::Completed { .. } => {
                let terminal_usage = usage.and_then(|value| {
                    TokenUsage::from_counts(
                        Some(u64::from(value.prompt_tokens)),
                        Some(u64::from(value.completion_tokens)),
                        Some(u64::from(value.total_tokens)),
                    )
                    .map(|usage| {
                        usage.with_cached_prompt_tokens(value.cached_tokens.map(u64::from))
                    })
                });
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamCompleted {
                        tokens: usage.map(|value| u64::from(value.completion_tokens)),
                        usage: terminal_usage,
                    },
                );
            }
            OpenAiTerminalResult::CompletedWithUsage { usage, .. } => {
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamCompleted {
                        tokens: usage.completion_tokens,
                        usage: Some(usage),
                    },
                );
            }
            OpenAiTerminalResult::Failed { failure, .. } => {
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamError {
                        error: Some(failure_label(failure).to_owned()),
                    },
                );
            }
        }
    }

    fn stream_interrupted(&self, request_id: RequestId, label: &'static str) {
        if lock_recover(&self.tracked).is_active(request_id) {
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::StreamError {
                    error: Some(label.to_owned()),
                },
            );
        }
    }

    fn enqueue_operation_event(&self, request_id: RequestId, event: LifecycleEvent) {
        if let Ok(payload) = serde_json::to_string(&event) {
            let _ = self
                .service
                .enqueue_event(request_id, ReplayChannel::Operations, payload);
        }
    }

    fn terminal(&self, request_id: RequestId, outcome: TerminalOutcome) {
        let guard = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(entry) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            let TrackedRequest::Active(active) = entry else {
                return;
            };
            let guard = active.guard.clone();
            *entry = TrackedRequest::Terminal;
            guard
        };

        // A stale or externally-terminalized guard is harmless: request
        // serving and later frontend events must never depend on this write.
        let _ = self
            .service
            .transition_terminal(request_id, &guard, outcome);
        self.raw_mesh_owners.release_frontend(request_id);
    }

    #[cfg(test)]
    fn tracked_len(&self) -> usize {
        lock_recover(&self.tracked).requests.len()
    }
}

const fn openai_method_label(method: OpenAiRequestMethod) -> &'static str {
    match method {
        OpenAiRequestMethod::Get => "GET",
        OpenAiRequestMethod::Post => "POST",
        OpenAiRequestMethod::Other => "OTHER",
    }
}

impl TrackedRequests {
    fn make_room(&mut self) -> Result<Option<RequestId>, ()> {
        if self.requests.len() < MAX_TRACKED_REQUESTS {
            return Ok(None);
        }
        let Some(oldest) = self.insertion_order.pop_front() else {
            return Err(());
        };
        if matches!(self.requests.get(&oldest), Some(TrackedRequest::Terminal)) {
            self.requests.remove(&oldest);
            Ok(Some(oldest))
        } else {
            self.insertion_order.push_front(oldest);
            Err(())
        }
    }

    fn is_active(&self, request_id: RequestId) -> bool {
        matches!(
            self.requests.get(&request_id),
            Some(TrackedRequest::Active(_))
        )
    }
}

impl OpenAiLifecycleObserver for OpenAiLifecycleLoggingAdapter {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        match event {
            OpenAiLifecycleEvent::Admitted { context } => self.admit(context),
            OpenAiLifecycleEvent::BackendDispatched { context, operation } => {
                self.backend_dispatched(context.request_id, *operation)
            }
            OpenAiLifecycleEvent::BackendTerminal {
                context,
                operation,
                result,
            } => self.backend_terminal(context.request_id, *operation, *result),
            OpenAiLifecycleEvent::StreamFirstItem { context, .. } => {
                self.backend_stream_first_item(context.request_id)
            }
            OpenAiLifecycleEvent::ResponseCompleted { context, usage, .. } => {
                self.response_completed(context.request_id, *usage)
            }
            OpenAiLifecycleEvent::Rejected {
                context, rejection, ..
            } => self.terminal(
                context.request_id,
                TerminalOutcome::Rejected(Some(rejection_label(*rejection).into())),
            ),
            OpenAiLifecycleEvent::NonStreamTerminal { context, result } => {
                self.terminal(context.request_id, terminal_outcome(*result))
            }
            OpenAiLifecycleEvent::StreamTerminal { context, result } => {
                self.stream_terminal(context.request_id, *result);
                self.terminal(context.request_id, terminal_outcome(*result));
            }
            OpenAiLifecycleEvent::StreamCancelled { context } => {
                self.stream_interrupted(context.request_id, "stream_cancelled");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Cancelled(Some("stream_cancelled".into())),
                );
            }
            OpenAiLifecycleEvent::StreamDropped { context } => {
                self.stream_interrupted(context.request_id, "stream_dropped");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Dropped(Some("stream_dropped".into())),
                );
            }
            OpenAiLifecycleEvent::RequestCancelled { context } => self.terminal(
                context.request_id,
                TerminalOutcome::Cancelled(Some("request_cancelled".into())),
            ),
        }
    }
}

fn terminal_outcome(result: OpenAiTerminalResult) -> TerminalOutcome {
    match result {
        OpenAiTerminalResult::Completed { status_code } => {
            TerminalOutcome::CompletedWithStatus(status_code)
        }
        OpenAiTerminalResult::CompletedWithUsage { status_code, usage } => {
            TerminalOutcome::CompletedWithUsage { status_code, usage }
        }
        OpenAiTerminalResult::Failed {
            failure: OpenAiFailure::Cancelled,
            ..
        } => TerminalOutcome::Cancelled(Some("request_cancelled".into())),
        OpenAiTerminalResult::Failed {
            status_code,
            failure,
        } => TerminalOutcome::FailedWithStatus {
            error: failure_label(failure).into(),
            status_code,
        },
    }
}

const fn operation_label(operation: OpenAiBackendOperation) -> &'static str {
    match operation {
        OpenAiBackendOperation::Models => "models",
        OpenAiBackendOperation::ChatCompletion => "chat_completion",
        OpenAiBackendOperation::ChatCompletionStream => "chat_completion_stream",
        OpenAiBackendOperation::Completion => "completion",
        OpenAiBackendOperation::CompletionStream => "completion_stream",
        OpenAiBackendOperation::Responses => "responses",
        OpenAiBackendOperation::ResponsesStream => "responses_stream",
    }
}

const fn rejection_label(rejection: OpenAiRejection) -> &'static str {
    match rejection {
        OpenAiRejection::InvalidRequest => "invalid_request",
        OpenAiRejection::PayloadTooLarge => "payload_too_large",
        OpenAiRejection::MethodNotAllowed => "method_not_allowed",
        OpenAiRejection::NotFound => "not_found",
        OpenAiRejection::AdmissionDenied => "admission_denied",
    }
}

const fn failure_label(failure: OpenAiFailure) -> &'static str {
    match failure {
        OpenAiFailure::Backend => "backend",
        OpenAiFailure::Timeout => "timeout",
        OpenAiFailure::Internal => "internal",
        OpenAiFailure::Cancelled => "cancelled",
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use mesh_llm_events::logging::{events::LifecycleEvent, identifiers::RequestId};
    use openai_frontend::{
        OpenAiBackendOperation, OpenAiFrontendRoute, OpenAiLifecycleContext, OpenAiLifecycleEvent,
        OpenAiRequestMethod, OpenAiTerminalResult, OpenAiUsage,
    };

    use super::*;

    #[derive(Default)]
    struct ArtifactCaptureProbe {
        bodies: Mutex<Vec<Vec<u8>>>,
        unavailable: Mutex<Vec<ArtifactUnavailableReason>>,
    }

    impl OpenAiArtifactCapture for ArtifactCaptureProbe {
        fn capture_body(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            content: &[u8],
            _media_kind: Option<&str>,
        ) {
            self.bodies.lock().unwrap().push(content.to_vec());
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            reason: ArtifactUnavailableReason,
        ) {
            self.unavailable.lock().unwrap().push(reason);
        }
    }

    #[test]
    fn stream_artifact_capture_enforces_the_limit_before_persistence() {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let capture: Arc<dyn OpenAiArtifactCapture> = probe.clone();
        let mut stream = OpenAiStreamArtifactCapture::new(capture, RequestId::new(), 4);

        stream.push(b"1234");
        stream.push(b"5");
        stream.complete();

        assert!(probe.bodies.lock().unwrap().is_empty());
        assert_eq!(
            *probe.unavailable.lock().unwrap(),
            vec![ArtifactUnavailableReason::CaptureContentLimitExceeded]
        );
    }

    #[test]
    fn dropped_stream_artifact_capture_records_an_explicit_unavailable_state() {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let capture: Arc<dyn OpenAiArtifactCapture> = probe.clone();
        let mut stream = OpenAiStreamArtifactCapture::new(capture, RequestId::new(), 64);

        stream.push(b"partial");
        drop(stream);

        assert!(probe.bodies.lock().unwrap().is_empty());
        assert_eq!(
            *probe.unavailable.lock().unwrap(),
            vec![ArtifactUnavailableReason::StreamingResponseNotAssembled]
        );
    }

    fn context(request_id: RequestId) -> OpenAiLifecycleContext {
        OpenAiLifecycleContext::new(
            request_id,
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::ChatCompletions,
        )
    }

    fn adapter() -> (Arc<LoggingService>, OpenAiLifecycleLoggingAdapter) {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let adapter = OpenAiLifecycleLoggingAdapter::new(
            Arc::clone(&service),
            Arc::new(RawMeshLifecycleOwners::default()),
        );
        (service, adapter)
    }

    #[test]
    fn admitted_requests_map_route_and_terminal_once() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: OpenAiBackendOperation::ChatCompletion,
        });
        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context: context.clone(),
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        assert!(
            service
                .registry_ref()
                .get_active(&request_id.as_uuid().to_string())
                .is_none()
        );
        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal request summary");
        assert_eq!(summary.metadata.route(), Some("chat_completions"));
        assert_eq!(summary.metadata.source(), Some("direct_http"));
        assert_eq!(summary.metadata.method(), Some("POST"));
        assert_eq!(summary.metadata.provider(), Some("openai_frontend"));
        assert_eq!(summary.metadata.engine(), Some("chat_completion"));
        let records = service.bus_ref().replay_window().records;
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("route_selected"))
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("completed"))
                .count(),
            1
        );
        assert_eq!(adapter.tracked_len(), 1);
    }

    #[test]
    fn terminalization_releases_frontend_owner_without_disturbing_raw_owner() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter = OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&owners));
        let frontend_request_id = RequestId::new();
        let frontend_context = context(frontend_request_id);
        let raw_request_id = RequestId::new();
        let raw = RawMeshRequestLifecycle::register(
            Arc::clone(&service),
            Arc::clone(&owners),
            raw_request_id,
        )
        .expect("raw request should own its lifecycle");
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: frontend_context.clone(),
        });
        assert!(owners.is_claimed(frontend_request_id));
        assert!(owners.is_claimed(raw_request_id));

        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context: frontend_context.clone(),
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context: frontend_context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        assert!(!owners.is_claimed(frontend_request_id));
        assert!(owners.is_claimed(raw_request_id));
        assert_eq!(adapter.tracked_len(), 1);
        raw.terminal(TerminalOutcome::Completed);
    }

    #[test]
    fn backend_stream_and_usage_events_map_to_canonical_children_once() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);
        let operation = OpenAiBackendOperation::ChatCompletionStream;
        let usage = OpenAiUsage {
            prompt_tokens: 21,
            cached_tokens: Some(13),
            completion_tokens: 8,
            total_tokens: 29,
        };

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context: context.clone(),
            operation,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamFirstItem {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamFirstItem {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::ResponseCompleted {
            context: context.clone(),
            operation,
            usage,
        });
        adapter.observe(&OpenAiLifecycleEvent::ResponseCompleted {
            context: context.clone(),
            operation,
            usage,
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        let events = canonical_events(&service);
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::AttemptStarted { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::AttemptCompleted { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::BackendStreamFirstItem
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::UsageRecorded { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::StreamCompleted {
                    tokens: Some(8),
                    usage: Some(TokenUsage {
                        prompt_tokens: Some(21),
                        cached_prompt_tokens: Some(13),
                        completion_tokens: Some(8),
                        total_tokens: Some(29),
                    }),
                }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::Completed { .. }
            )),
            1
        );
    }

    fn canonical_events(service: &LoggingService) -> Vec<LifecycleEvent> {
        service
            .bus_ref()
            .replay_window()
            .records
            .into_iter()
            .filter_map(|record| {
                let envelope =
                    serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                serde_json::from_str(envelope.get("payload")?.as_str()?).ok()
            })
            .collect()
    }

    fn count_events(
        events: &[LifecycleEvent],
        predicate: impl Fn(&LifecycleEvent) -> bool,
    ) -> usize {
        events.iter().filter(|event| predicate(event)).count()
    }

    #[test]
    fn terminal_before_admission_and_unknown_route_do_not_create_requests() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);

        adapter.observe(&OpenAiLifecycleEvent::StreamDropped {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context,
            operation: OpenAiBackendOperation::Responses,
        });

        assert!(
            service
                .registry_ref()
                .get_active(&request_id.as_uuid().to_string())
                .is_none()
        );
        assert!(service.bus_ref().replay_window().records.is_empty());
    }

    #[test]
    fn raw_mesh_owner_prevents_a_competing_embedded_frontend_parent() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();
        let _raw = super::super::RawMeshRequestLifecycle::register(
            Arc::clone(&service),
            raw_mesh_owners,
            request_id,
        )
        .unwrap();

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });

        assert_eq!(adapter.tracked_len(), 0);
        assert_eq!(
            service
                .bus_ref()
                .replay_window()
                .records
                .into_iter()
                .filter_map(|record| {
                    let envelope =
                        serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                    serde_json::from_str(envelope.get("payload")?.as_str()?).ok()
                })
                .filter(|event| matches!(event, LifecycleEvent::Admitted { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn markerless_authenticated_attribution_and_frontend_admission_choose_one_parent_in_both_orders()
     {
        for attribution_first in [true, false] {
            let service = Arc::new(LoggingService::new_disabled(Default::default()));
            let owners = Arc::new(RawMeshLifecycleOwners::default());
            let adapter =
                OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&owners));
            let request_id = RequestId::new();
            let endpoint_id = "83".repeat(32);
            let authenticated = RequestSummaryMetadata::default().with_caller_identity(
                Some(&endpoint_id),
                Some("192.0.2.83:11204"),
                Some(super::super::CallerPathType::RemoteQuicHttp),
            );
            let request_metadata = RequestSummaryMetadata::from_parts(
                Some("chat_completions"),
                Some("model-a"),
                Some("openai_frontend"),
                Some("chat_completion"),
            )
            .with_source(Some("direct_http"))
            .with_method(Some("POST"));

            let attribution = if attribution_first {
                super::super::RawMeshRemoteAttributionLease::acquire(
                    &service,
                    Arc::clone(&owners),
                    request_id,
                    authenticated.clone(),
                )
                .expect("authenticated attribution")
            } else {
                adapter.admit(&context(request_id));
                service.merge_request_metadata(request_id, request_metadata.clone());
                super::super::RawMeshRemoteAttributionLease::acquire(
                    &service,
                    Arc::clone(&owners),
                    request_id,
                    authenticated,
                )
                .expect("authenticated attribution")
            };
            if attribution_first {
                adapter.admit(&context(request_id));
            }

            let raw = RawMeshRequestLifecycle::register_with_metadata(
                Arc::clone(&service),
                owners,
                request_id,
                request_metadata,
            );
            match raw {
                Some(raw) => raw.terminal(TerminalOutcome::Completed),
                None => adapter.terminal(request_id, TerminalOutcome::Completed),
            }
            drop(attribution);

            let events = canonical_events(&service);
            assert_eq!(
                count_events(&events, |event| matches!(
                    event,
                    LifecycleEvent::Admitted { .. }
                )),
                1,
                "admission order: attribution_first={attribution_first}"
            );
            assert_eq!(
                count_events(&events, |event| matches!(
                    event,
                    LifecycleEvent::Completed { .. }
                )),
                1,
                "terminal order: attribution_first={attribution_first}"
            );
            let summary = service
                .registry_ref()
                .get_recent(&request_id.as_uuid().to_string())
                .expect("terminal summary");
            assert_eq!(summary.metadata.route(), Some("chat_completions"));
            assert_eq!(summary.metadata.model(), Some("model-a"));
            assert_eq!(summary.metadata.provider(), Some("openai_frontend"));
            assert_eq!(summary.metadata.engine(), Some("chat_completion"));
            assert_eq!(summary.metadata.source(), Some("direct_http"));
            assert_eq!(summary.metadata.method(), Some("POST"));
            assert_eq!(
                summary.metadata.caller_endpoint_id(),
                Some(endpoint_id.as_str())
            );
            assert_eq!(summary.metadata.caller_addr(), Some("192.0.2.83:11204"));
            assert_eq!(
                summary.metadata.caller_path_type(),
                Some("remote_quic_http")
            );
        }
    }

    #[test]
    fn ingress_attachment_keeps_direct_frontend_on_one_parent() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();
        let parent =
            RawMeshRequestLifecycle::register(Arc::clone(&service), raw_mesh_owners, request_id)
                .expect("direct ingress should claim one parent");
        let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));

        // A direct host ingress can pass through the embedded frontend, but
        // the shared owner registry prevents it from creating a second parent.
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        assert_eq!(adapter.tracked_len(), 0);

        let observer = attachment.route_observer();
        observer.route_selected(Some("safe-model"));
        let attempt_id = observer
            .start_attempt()
            .expect("owned attachment should allocate attempts");
        observer.complete_attempt(Some(attempt_id), 200);
        attachment.terminal(TerminalOutcome::Completed);

        let events = service.bus_ref().replay_window().records;
        assert_eq!(
            events
                .iter()
                .filter(|record| record.entry.payload.contains("\"type\":\"admitted\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record.entry.payload.contains("\"type\":\"completed\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record
                    .entry
                    .payload
                    .contains("\"type\":\"attempt_started\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record
                    .entry
                    .payload
                    .contains("\"type\":\"attempt_completed\""))
                .count(),
            1
        );
    }

    #[test]
    fn remote_tunnel_suppression_skips_only_the_active_frontend_admission() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();

        let lease = super::super::RawMeshRemoteSuppressionLease::acquire(
            Arc::clone(&raw_mesh_owners),
            request_id,
        )
        .unwrap();
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        assert_eq!(adapter.tracked_len(), 0);

        drop(lease);
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        assert_eq!(adapter.tracked_len(), 1);
    }

    #[test]
    fn terminal_labels_are_bounded_metadata() {
        assert_eq!(
            terminal_outcome(OpenAiTerminalResult::Failed {
                status_code: 504,
                failure: OpenAiFailure::Timeout,
            }),
            TerminalOutcome::FailedWithStatus {
                error: "timeout".into(),
                status_code: 504,
            }
        );
        assert_eq!(
            terminal_outcome(OpenAiTerminalResult::Failed {
                status_code: 499,
                failure: OpenAiFailure::Cancelled,
            }),
            TerminalOutcome::Cancelled(Some("request_cancelled".into()))
        );
        assert_eq!(
            serde_json::to_string(&LifecycleEvent::RouteSelected {
                model: None,
                provider: Some("openai_frontend".into()),
                engine: Some(operation_label(OpenAiBackendOperation::Responses).into()),
            })
            .expect("event should serialize"),
            r#"{"type":"route_selected","provider":"openai_frontend","engine":"responses"}"#
        );
    }

    #[test]
    fn bounded_tracking_rejects_new_admission_when_all_slots_are_active() {
        let (service, adapter) = adapter();
        for _ in 0..MAX_TRACKED_REQUESTS {
            adapter.admit(&context(RequestId::new()));
        }
        let overflow = RequestId::new();
        adapter.admit(&context(overflow));

        assert_eq!(adapter.tracked_len(), MAX_TRACKED_REQUESTS);
        assert!(
            service
                .registry_ref()
                .get_active(&overflow.as_uuid().to_string())
                .is_none()
        );
    }

    /// Review defect D12: pins `runtime_events::config::REQUEST_ROOT_BOUND`
    /// to this module's `MAX_TRACKED_REQUESTS` so the runtime-event
    /// reservation sizing cannot silently desynchronize from the logging
    /// request-tracking bound. Imports both constants through their real
    /// crate-visible paths (not `super::*`) so the test also proves
    /// `MAX_TRACKED_REQUESTS` is reachable crate-wide, matching how
    /// `logging::raw_mesh_lifecycle` and `network::openai::runtime_events`
    /// consume it.
    #[test]
    fn request_root_bound() {
        use crate::logging::MAX_TRACKED_REQUESTS;
        use crate::runtime_events::config::REQUEST_ROOT_BOUND;

        assert_eq!(
            REQUEST_ROOT_BOUND, MAX_TRACKED_REQUESTS,
            "runtime_events::config::REQUEST_ROOT_BOUND must stay pinned to \
             logging::openai_lifecycle::MAX_TRACKED_REQUESTS"
        );
    }
}
