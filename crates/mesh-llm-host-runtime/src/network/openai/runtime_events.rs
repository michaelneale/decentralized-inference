//! Runtime-event producer wiring for OpenAI request/admission/stream
//! outcomes (plan task 11, `.omo/plans/event-system.md` line 286).
//!
//! Adapts `openai-frontend`'s dependency-safe [`OpenAiLifecycleObserver`]
//! events into root/child `RuntimeFact::Request` facts through the host
//! runtime-event engine. The request root [`OperationId`] is minted
//! byte-equal to the logging `RequestId` (task 2's byte-equality rule) at
//! admission and never regenerated, so a consumer can correlate a runtime
//! event stream operation directly against the existing structured log.
//!
//! Exactly one terminal decision per request: `openai-frontend`'s own
//! `RequestLifecycle`/`StreamLifecycle` already guarantee exactly one
//! root-scope terminal event (`NonStreamTerminal`/`StreamTerminal`/
//! `Rejected`/`StreamCancelled`/`StreamDropped`/`RequestCancelled`) and
//! exactly one backend-scope terminal event (`BackendTerminal`) per
//! dispatch -- see `request_lifecycle.rs::terminal_or_transferred` and
//! `stream_lifecycle.rs::claim_terminal`. This adapter mirrors that
//! guarantee rather than reimplementing it: every terminal-shaped branch
//! below `Option::take`s its tracked reservation exactly once and submits
//! its terminal fact through it, so a second observation of the same
//! logical terminal (which `openai-frontend` never produces) would be a
//! silent no-op rather than a write-once-slot rejection.
//!
//! Every emission is best-effort. An absent engine, a reservation-table
//! exhaustion, or a rejected submit never blocks or fails request serving --
//! `let _ =` submits and the `Option<OperationReservation>` degrade path are
//! the enforcement mechanism, matching the established pattern in
//! `runtime/model_lifecycle/events.rs` (task 9) and
//! `inference/skippy/stage/runtime_events.rs` (task 10).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use mesh_llm_events::logging::identifiers::RequestId as LoggingRequestId;
use mesh_llm_runtime_event_contracts::{
    ChildOperationId, FactData, OperationId, Outcome, ReasonCode, RequestEventKind, RequestFact,
    RequestId, RuntimeEventIngress, RuntimeFact, ScopeIdentities,
};
use openai_frontend::{
    OpenAiFailure, OpenAiLifecycleContext, OpenAiLifecycleEvent, OpenAiLifecycleObserver,
    OpenAiRejection, OpenAiTerminalResult,
};

use crate::logging::MAX_TRACKED_REQUESTS;
use crate::runtime_events::engine::OperationReservation;
use crate::runtime_events::runtime_event_engine;

/// Mint the request root [`OperationId`] byte-equal to the logging
/// `RequestId`. `RequestId` wraps a `Uuid` (`mesh-llm-events`); reusing its
/// raw bytes -- never re-deriving or hashing -- is what the byte-equality
/// rule requires.
fn operation_id_for_request(request_id: LoggingRequestId) -> OperationId {
    OperationId::from_bytes(request_id.as_uuid().into_bytes())
}

fn empty_data() -> FactData {
    FactData::default()
}

/// Task 7 (`.omo/plans/event-system-fixes.md`, review defect D5): every
/// request-class fact carries `scope.request_id` = the root uuid text, the
/// same text `operation_id_for_request` above mints byte-equal to the
/// logging request id. This is what lets the wire's `operation.root` and
/// `event.scope.request_id` both be independently checked against the same
/// structured-log request id.
fn scope_for_request(request_id: LoggingRequestId) -> ScopeIdentities {
    ScopeIdentities {
        request_id: RequestId::new(&request_id.as_uuid().to_string()).ok(),
        ..ScopeIdentities::default()
    }
}

fn data_with_scope(request_id: LoggingRequestId) -> FactData {
    FactData {
        scope: scope_for_request(request_id),
        ..FactData::default()
    }
}

fn request_fact(kind: RequestEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Request(RequestFact::with_data(kind, data))
}

/// Reservation-drop synthesis. Never constructed for a request whose root
/// was never reserved; only used by the engine itself when a live guard is
/// dropped before a terminal was submitted (exhaustion, panic-unwind, or an
/// event this adapter does not model).
fn synthetic_request_terminal() -> RuntimeFact {
    request_fact(
        RequestEventKind::RequestFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..empty_data()
        },
    )
}

fn rejection_reason(rejection: OpenAiRejection) -> ReasonCode {
    match rejection {
        OpenAiRejection::NotFound => ReasonCode::MissingArtifact,
        OpenAiRejection::InvalidRequest
        | OpenAiRejection::PayloadTooLarge
        | OpenAiRejection::MethodNotAllowed
        | OpenAiRejection::AdmissionDenied => ReasonCode::InvalidConfiguration,
    }
}

fn failure_reason(failure: OpenAiFailure) -> ReasonCode {
    match failure {
        OpenAiFailure::Timeout => ReasonCode::Timeout,
        OpenAiFailure::Cancelled => ReasonCode::Cancellation,
        OpenAiFailure::Backend | OpenAiFailure::Internal => ReasonCode::InternalRuntimeFailure,
    }
}

/// Map one frontend terminal result onto its `RequestEventKind` mapping-table
/// row. `Completed`/`CompletedWithUsage` -> `RequestCompleted`; a `Cancelled`
/// failure -> `RequestCancelled`; a `Timeout` failure -> `RequestTimedOut`;
/// every remaining failure -> `RequestFailed`.
fn terminal_fact_for_result(
    request_id: LoggingRequestId,
    result: OpenAiTerminalResult,
) -> RuntimeFact {
    match result {
        OpenAiTerminalResult::Completed { .. }
        | OpenAiTerminalResult::CompletedWithUsage { .. } => request_fact(
            RequestEventKind::RequestCompleted,
            FactData {
                outcome: Some(Outcome::Success),
                ..data_with_scope(request_id)
            },
        ),
        OpenAiTerminalResult::Failed {
            failure: OpenAiFailure::Cancelled,
            ..
        } => request_fact(
            RequestEventKind::RequestCancelled,
            FactData {
                outcome: Some(Outcome::Cancelled),
                reason: Some(ReasonCode::Cancellation),
                ..data_with_scope(request_id)
            },
        ),
        OpenAiTerminalResult::Failed {
            failure: OpenAiFailure::Timeout,
            ..
        } => request_fact(
            RequestEventKind::RequestTimedOut,
            FactData {
                outcome: Some(Outcome::Failure),
                reason: Some(ReasonCode::Timeout),
                ..data_with_scope(request_id)
            },
        ),
        OpenAiTerminalResult::Failed { failure, .. } => request_fact(
            RequestEventKind::RequestFailed,
            FactData {
                outcome: Some(Outcome::Failure),
                reason: Some(failure_reason(failure)),
                ..data_with_scope(request_id)
            },
        ),
    }
}

#[derive(Default)]
struct RequestTracking {
    root: Option<OperationReservation>,
    backend: Option<OperationReservation>,
}

#[derive(Default)]
struct TrackedRequests {
    requests: HashMap<LoggingRequestId, RequestTracking>,
    insertion_order: VecDeque<LoggingRequestId>,
}

impl TrackedRequests {
    /// Bound tracking to `MAX_TRACKED_REQUESTS` by evicting the oldest entry.
    /// Any reservation still held by the evicted entry is simply dropped --
    /// `OperationReservation::drop` synthesizes `terminal_not_delivered` on
    /// the engine side, so eviction never leaves a reservation silently
    /// unresolved.
    fn make_room(&mut self) {
        if self.requests.len() < MAX_TRACKED_REQUESTS {
            return;
        }
        if let Some(oldest) = self.insertion_order.pop_front() {
            self.requests.remove(&oldest);
        }
    }

    /// Remove a request once both of its independently terminal scopes have
    /// resolved. A root terminal may arrive before its backend terminal (or
    /// vice versa), so the entry must remain until neither reservation is
    /// live. Keep the insertion queue in sync with the map so completed
    /// requests do not consume the tracking bound forever.
    fn remove_if_resolved(&mut self, request_id: LoggingRequestId) {
        let resolved = self
            .requests
            .get(&request_id)
            .is_some_and(|tracking| tracking.root.is_none() && tracking.backend.is_none());
        if resolved {
            self.requests.remove(&request_id);
            self.insertion_order
                .retain(|tracked_id| *tracked_id != request_id);
        }
    }
}

/// Metadata-only OpenAI frontend lifecycle observer that adapts frontend
/// boundaries onto root/child runtime-event facts. Owned independently from
/// [`crate::logging::OpenAiLifecycleLoggingAdapter`] -- both are fanned out
/// to by [`compose_lifecycle_observer`] since `OpenAiFrontendConfig` accepts
/// only one observer slot.
pub(crate) struct OpenAiRuntimeEventObserver {
    tracked: Mutex<TrackedRequests>,
}

impl OpenAiRuntimeEventObserver {
    pub(crate) fn new() -> Self {
        Self {
            tracked: Mutex::new(TrackedRequests::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TrackedRequests> {
        self.tracked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn admit(&self, context: &OpenAiLifecycleContext) {
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        let operation_id = operation_id_for_request(context.request_id);
        let Some(root) = engine.reserve_root(operation_id, synthetic_request_terminal) else {
            return;
        };
        let _ = root.ingress().try_submit(request_fact(
            RequestEventKind::RequestReceived,
            data_with_scope(context.request_id),
        ));
        let _ = root.ingress().try_submit(request_fact(
            RequestEventKind::RequestAdmitted,
            data_with_scope(context.request_id),
        ));

        let mut tracked = self.lock();
        tracked.make_room();
        tracked.requests.insert(
            context.request_id,
            RequestTracking {
                root: Some(root),
                backend: None,
            },
        );
        tracked.insertion_order.push_back(context.request_id);
    }

    /// Resolve the root reservation's terminal exactly once. A request
    /// whose root was never reserved (degraded/exhausted at admission, or
    /// already resolved) is a silent no-op -- there is nothing to resolve.
    fn resolve_root(&self, request_id: LoggingRequestId, fact: RuntimeFact) {
        let root = {
            let mut tracked = self.lock();
            let Some(tracking) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            let root = tracking.root.take();
            tracked.remove_if_resolved(request_id);
            root
        };
        if let Some(root) = root {
            let _ = root.ingress().try_submit(fact);
        }
    }

    fn backend_dispatched(&self, request_id: LoggingRequestId) {
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        let root_id = operation_id_for_request(request_id);
        let Some(child) =
            engine.reserve_child(root_id, ChildOperationId::new(), synthetic_request_terminal)
        else {
            return;
        };
        let _ = child.ingress().try_submit(request_fact(
            RequestEventKind::RequestExecutionStarted,
            data_with_scope(request_id),
        ));

        let mut tracked = self.lock();
        let Some(tracking) = tracked.requests.get_mut(&request_id) else {
            // The root is untracked (never admitted, already terminal, or
            // evicted) -- drop the freshly reserved child immediately. Its
            // own Drop synthesizes a terminal, so the write-once slot is
            // still resolved even without a live tracking entry to hold it.
            return;
        };
        tracking.backend = Some(child);
    }

    fn backend_terminal(&self, request_id: LoggingRequestId, result: OpenAiTerminalResult) {
        let backend = {
            let mut tracked = self.lock();
            let Some(tracking) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            let backend = tracking.backend.take();
            tracked.remove_if_resolved(request_id);
            backend
        };
        if let Some(backend) = backend {
            let _ = backend
                .ingress()
                .try_submit(terminal_fact_for_result(request_id, result));
        }
    }
}

impl Default for OpenAiRuntimeEventObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiLifecycleObserver for OpenAiRuntimeEventObserver {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        match event {
            OpenAiLifecycleEvent::Admitted { context } => self.admit(context),
            OpenAiLifecycleEvent::Rejected {
                context, rejection, ..
            } => self.resolve_root(
                context.request_id,
                request_fact(
                    RequestEventKind::RequestRejected,
                    FactData {
                        outcome: Some(Outcome::Rejected),
                        reason: Some(rejection_reason(*rejection)),
                        ..data_with_scope(context.request_id)
                    },
                ),
            ),
            OpenAiLifecycleEvent::BackendDispatched { context, .. } => {
                self.backend_dispatched(context.request_id);
            }
            OpenAiLifecycleEvent::BackendTerminal {
                context, result, ..
            } => self.backend_terminal(context.request_id, *result),
            OpenAiLifecycleEvent::NonStreamTerminal { context, result }
            | OpenAiLifecycleEvent::StreamTerminal { context, result } => {
                self.resolve_root(
                    context.request_id,
                    terminal_fact_for_result(context.request_id, *result),
                );
            }
            OpenAiLifecycleEvent::StreamCancelled { context }
            | OpenAiLifecycleEvent::RequestCancelled { context } => self.resolve_root(
                context.request_id,
                request_fact(
                    RequestEventKind::RequestCancelled,
                    FactData {
                        outcome: Some(Outcome::Cancelled),
                        reason: Some(ReasonCode::Cancellation),
                        ..data_with_scope(context.request_id)
                    },
                ),
            ),
            OpenAiLifecycleEvent::StreamDropped { context } => self.resolve_root(
                context.request_id,
                request_fact(
                    RequestEventKind::RequestFailed,
                    FactData {
                        outcome: Some(Outcome::Unknown),
                        reason: Some(ReasonCode::TerminalNotDelivered),
                        ..data_with_scope(context.request_id)
                    },
                ),
            ),
            // `StreamFirstItem`/`ResponseCompleted` have no corresponding
            // `RequestEventKind` row in this task's scope -- request-scope
            // progress and prompt/completion usage are out of §8's Request
            // family, and adapting them would require inventing a kind that
            // does not exist in the inventory. Intentionally unmapped.
            OpenAiLifecycleEvent::StreamFirstItem { .. }
            | OpenAiLifecycleEvent::ResponseCompleted { .. } => {}
        }
    }
}

/// Composite observer that fans a frontend lifecycle event out to every
/// installed sink. `OpenAiFrontendConfig` has exactly one observer slot
/// (`serving_hooks`-style single occupancy), so this is the fan-out point
/// combining the existing logging adapter with this task's runtime-event
/// adapter -- mirroring task 12's planned `GenerationLifecycleIngress`
/// composite for the same single-slot reason.
struct CompositeOpenAiLifecycleObserver {
    sinks: Vec<Arc<dyn OpenAiLifecycleObserver>>,
}

impl OpenAiLifecycleObserver for CompositeOpenAiLifecycleObserver {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        for sink in &self.sinks {
            sink.observe(event);
        }
    }
}

/// Compose an optional existing observer (typically the host logging
/// adapter) with this task's runtime-event observer. Always returns
/// `Some` -- the runtime-event observer degrades to a no-op on every method
/// when no engine is installed, so composing it in is unconditional and
/// never removes the base observer's behavior when `base` is `Some`.
pub(crate) fn compose_lifecycle_observer(
    base: Option<Arc<dyn OpenAiLifecycleObserver>>,
) -> Option<Arc<dyn OpenAiLifecycleObserver>> {
    let mut sinks: Vec<Arc<dyn OpenAiLifecycleObserver>> = Vec::with_capacity(2);
    if let Some(base) = base {
        sinks.push(base);
    }
    sinks.push(Arc::new(OpenAiRuntimeEventObserver::new()));
    Some(Arc::new(CompositeOpenAiLifecycleObserver { sinks }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::RuntimeFact;
    use openai_frontend::{OpenAiFrontendRoute, OpenAiRequestMethod, parse_request_id};

    const REQUEST_ID: &str = "c0a801ef-2a39-4f52-99f5-bdc849127cde";

    fn install_test_engine() -> Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    fn context() -> OpenAiLifecycleContext {
        OpenAiLifecycleContext::new(
            parse_request_id(REQUEST_ID).expect("test UUID should parse"),
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::ChatCompletions,
        )
    }

    fn request_kinds(engine: &RuntimeEventEngine) -> Vec<RequestEventKind> {
        engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::Request(fact) => Some(*fact.kind()),
                _ => None,
            })
            .collect()
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn root_operation_id_is_byte_equal_to_the_logging_request_id() {
        let request_id = parse_request_id(REQUEST_ID).expect("test UUID should parse");
        let operation_id = operation_id_for_request(request_id);
        assert_eq!(operation_id.into_bytes(), request_id.as_uuid().into_bytes());
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn admitted_then_completed_emits_exactly_one_root_terminal() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        assert_eq!(engine.occupied_count(), 1);
        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        engine.drain();

        assert_eq!(engine.occupied_count(), 0);
        let kinds = request_kinds(&engine);
        assert_eq!(
            kinds
                .iter()
                .filter(|kind| **kind == RequestEventKind::RequestCompleted)
                .count(),
            1
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn rejected_request_resolves_without_a_backend_child() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::Rejected {
            context,
            status_code: 400,
            rejection: OpenAiRejection::InvalidRequest,
        });
        engine.drain();

        assert_eq!(engine.occupied_count(), 0);
        assert!(
            request_kinds(&engine).contains(&RequestEventKind::RequestRejected),
            "expected a RequestRejected terminal"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn backend_dispatch_and_terminal_resolve_the_child_before_the_root() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
        });
        assert_eq!(engine.occupied_count(), 2, "root + backend child");
        observer.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        engine.drain();
        assert_eq!(
            engine.occupied_count(),
            1,
            "root still owns the request terminal"
        );
        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn root_terminal_keeps_backend_tracking_until_backend_terminal() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();
        let request_id = context.request_id;

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
        });

        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context: context.clone(),
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        {
            let tracked = observer.lock();
            let request = tracked
                .requests
                .get(&request_id)
                .expect("backend reservation must keep tracking alive");
            assert!(request.root.is_none());
            assert!(request.backend.is_some());
            assert_eq!(
                tracked.insertion_order.iter().copied().collect::<Vec<_>>(),
                vec![request_id]
            );
        }

        observer.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context,
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        let tracked = observer.lock();
        assert!(tracked.requests.is_empty());
        assert!(tracked.insertion_order.is_empty());

        engine.drain();
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn backend_terminal_keeps_root_tracking_until_root_terminal() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();
        let request_id = context.request_id;

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
        });

        observer.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        {
            let tracked = observer.lock();
            let request = tracked
                .requests
                .get(&request_id)
                .expect("root reservation must keep tracking alive");
            assert!(request.root.is_some());
            assert!(request.backend.is_none());
            assert_eq!(
                tracked.insertion_order.iter().copied().collect::<Vec<_>>(),
                vec![request_id]
            );
        }

        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        let tracked = observer.lock();
        assert!(tracked.requests.is_empty());
        assert!(tracked.insertion_order.is_empty());

        engine.drain();
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn cancelled_stream_maps_to_request_cancelled() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::StreamCancelled { context });
        engine.drain();

        assert_eq!(engine.occupied_count(), 0);
        assert!(request_kinds(&engine).contains(&RequestEventKind::RequestCancelled));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn timeout_failure_maps_to_request_timed_out_and_cancellation_maps_to_request_cancelled() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();

        let timeout_context = context();
        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: timeout_context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::StreamTerminal {
            context: timeout_context,
            result: OpenAiTerminalResult::Failed {
                status_code: 504,
                failure: OpenAiFailure::Timeout,
            },
        });
        engine.drain();
        assert!(request_kinds(&engine).contains(&RequestEventKind::RequestTimedOut));
        clear_runtime_event_engine();

        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let cancel_context = context();
        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: cancel_context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::StreamTerminal {
            context: cancel_context,
            result: OpenAiTerminalResult::Failed {
                status_code: 499,
                failure: OpenAiFailure::Cancelled,
            },
        });
        engine.drain();
        assert!(request_kinds(&engine).contains(&RequestEventKind::RequestCancelled));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropped_stream_resolves_the_root_without_a_dropped_kind() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::StreamDropped { context });
        engine.drain();

        assert_eq!(engine.occupied_count(), 0);
        assert!(request_kinds(&engine).contains(&RequestEventKind::RequestFailed));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_degrades_to_no_op_and_never_panics() {
        clear_runtime_event_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();
        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
        });
        observer.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        // No assertions beyond "did not panic": there is no engine to
        // inspect, which is exactly the degraded-but-not-failing contract.
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reservation_exhaustion_degrades_without_failing() {
        let engine = RuntimeEventEngine::with_capacity(0);
        clear_runtime_event_engine();
        install_runtime_event_engine(engine.clone());
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: openai_frontend::OpenAiBackendOperation::ChatCompletion,
        });
        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        assert_eq!(engine.occupied_count(), 0);
        assert!(engine.health().snapshot().reservation_exhausted > 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn composite_observer_fans_out_to_every_sink() {
        use std::sync::Mutex as StdMutex;

        struct Probe(StdMutex<usize>);
        impl OpenAiLifecycleObserver for Probe {
            fn observe(&self, _event: &OpenAiLifecycleEvent) {
                *self.0.lock().unwrap() += 1;
            }
        }

        let probe = Arc::new(Probe(StdMutex::new(0)));
        let composite =
            compose_lifecycle_observer(Some(probe.clone() as Arc<dyn OpenAiLifecycleObserver>))
                .expect("composite is always Some");
        composite.observe(&OpenAiLifecycleEvent::Admitted { context: context() });

        assert_eq!(*probe.0.lock().unwrap(), 1, "base observer must be reached");
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn no_prohibited_fields_reach_the_fact_data_shape() {
        // FactData structurally has no field capable of carrying request
        // content, tool arguments, media, or private URLs -- scope, state,
        // progress, outcome, reason, duration, numeric summaries, and a
        // bounded human summary only. This test documents that guarantee
        // via the emitted fact shape rather than re-deriving FactData's
        // definition. `scope.request_id` is task 7's correlation id, never
        // request content -- everything else on `ScopeIdentities` stays
        // unset.
        let request_id = parse_request_id(REQUEST_ID).expect("test UUID should parse");
        let fact = terminal_fact_for_result(
            request_id,
            OpenAiTerminalResult::Completed { status_code: 200 },
        );
        let RuntimeFact::Request(fact) = fact else {
            panic!("expected a Request fact");
        };
        assert_eq!(
            fact.data().scope,
            mesh_llm_runtime_event_contracts::ScopeIdentities {
                request_id: Some(
                    RequestId::new(&request_id.as_uuid().to_string()).expect("valid request id")
                ),
                ..Default::default()
            }
        );
        assert!(fact.data().summary.is_none());
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn request_facts_carry_scope_request_id_equal_to_the_root_operation_id_text() {
        let engine = install_test_engine();
        let observer = OpenAiRuntimeEventObserver::new();
        let context = context();
        let expected = operation_id_for_request(context.request_id).to_string();

        observer.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        engine.drain();

        let request_ids: Vec<String> = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::Request(fact) => fact
                    .data()
                    .scope
                    .request_id
                    .as_ref()
                    .map(|id| id.as_str().to_string()),
                _ => None,
            })
            .collect();
        assert!(
            !request_ids.is_empty(),
            "expected at least one request fact with a scope.request_id"
        );
        assert!(
            request_ids.iter().all(|id| id == &expected),
            "every request fact must carry the root uuid text: {request_ids:?}"
        );
        clear_runtime_event_engine();
    }
}
