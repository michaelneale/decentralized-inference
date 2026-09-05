//! Host-side [`skippy_server::runtime_state::SessionLifecycleObserver`]
//! implementation, translating real session-lifecycle decisions (plan
//! task 12, §8.7) into `RuntimeFact::Session` facts.
//!
//! Both kinds here (`SessionReset`, `SessionReclaimed`) are StateTransition-
//! class (`delivery/lifecycle.rs`), so submission uses `unreserved_ingress`
//! with a fresh root per event -- session reset/reclaim decisions happen
//! independent of any generation's `OperationId`, matching
//! `runtime_events/kv.rs`'s established pattern for facts with no bounded
//! operation to attach to.

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, FactData, KvRuntimeStateEventKind, KvRuntimeStateFact, NumericSummary,
    NumericSummaryKey, NumericValue, OperationId, OperationScope, Outcome, RuntimeEventIngress,
    RuntimeFact, SessionEventKind, SessionFact,
};
use skippy_server::runtime_state::{SessionLifecycleEvent, SessionLifecycleObserver};

use crate::runtime_events::runtime_event_engine;

fn session_fact(
    kind: SessionEventKind,
    outcome: Outcome,
    numeric_summaries: BoundedNumericSummaries,
) -> RuntimeFact {
    RuntimeFact::Session(SessionFact::with_data(
        kind,
        FactData {
            outcome: Some(outcome),
            numeric_summaries,
            ..FactData::default()
        },
    ))
}

fn kv_state_fact(kind: KvRuntimeStateEventKind, outcome: Outcome) -> RuntimeFact {
    RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
        kind,
        FactData {
            outcome: Some(outcome),
            ..FactData::default()
        },
    ))
}

fn millis_summary(key: &str, value: f64) -> BoundedNumericSummaries {
    NumericSummaryKey::new(key)
        .ok()
        .and_then(|key| {
            BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Floating(value),
            )])
            .ok()
        })
        .unwrap_or_default()
}

fn token_count_summary(key: &str, value: u64) -> BoundedNumericSummaries {
    NumericSummaryKey::new(key)
        .ok()
        .and_then(|key| {
            BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Unsigned(value),
            )])
            .ok()
        })
        .unwrap_or_default()
}

fn fact_for(event: SessionLifecycleEvent) -> Option<RuntimeFact> {
    Some(match event {
        SessionLifecycleEvent::SessionReset { reset_ms } => session_fact(
            SessionEventKind::SessionReset,
            Outcome::Success,
            millis_summary("reset_ms", reset_ms),
        ),
        SessionLifecycleEvent::SessionReclaimed => session_fact(
            SessionEventKind::SessionReclaimed,
            Outcome::Success,
            BoundedNumericSummaries::default(),
        ),
        SessionLifecycleEvent::SessionTrimmed { token_count } => session_fact(
            SessionEventKind::SessionTrimmed,
            Outcome::Success,
            token_count_summary("trimmed_token_count", token_count),
        ),
        SessionLifecycleEvent::RuntimeStateExportCompleted => kv_state_fact(
            KvRuntimeStateEventKind::RuntimeStateExportCompleted,
            Outcome::Success,
        ),
        SessionLifecycleEvent::RuntimeStateExportFailed => kv_state_fact(
            KvRuntimeStateEventKind::RuntimeStateExportFailed,
            Outcome::Failure,
        ),
        SessionLifecycleEvent::RuntimeStateImportCompleted => kv_state_fact(
            KvRuntimeStateEventKind::RuntimeStateImportCompleted,
            Outcome::Success,
        ),
        SessionLifecycleEvent::RuntimeStateImportFailed => kv_state_fact(
            KvRuntimeStateEventKind::RuntimeStateImportFailed,
            Outcome::Failure,
        ),
        // `#[non_exhaustive]`: a future variant this adapter does not yet
        // model must never be submitted.
        _ => return None,
    })
}

pub(crate) struct SkippySessionRuntimeEventObserver;

impl SkippySessionRuntimeEventObserver {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl SessionLifecycleObserver for SkippySessionRuntimeEventObserver {
    fn observe(&self, event: SessionLifecycleEvent) {
        let Some(fact) = fact_for(event) else {
            return;
        };
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
        let _ = ingress.try_submit(fact);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn session_reset_reaches_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippySessionRuntimeEventObserver::new();
        observer.observe(SessionLifecycleEvent::SessionReset { reset_ms: 2.0 });
        assert!(engine.state_lane_kinds().contains(&"session_reset"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn session_trimmed_reaches_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippySessionRuntimeEventObserver::new();
        observer.observe(SessionLifecycleEvent::SessionTrimmed { token_count: 42 });
        assert!(engine.state_lane_kinds().contains(&"session_trimmed"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn session_reclaimed_reaches_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippySessionRuntimeEventObserver::new();
        observer.observe(SessionLifecycleEvent::SessionReclaimed);
        assert!(engine.state_lane_kinds().contains(&"session_reclaimed"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_degrades_to_no_op() {
        clear_runtime_event_engine();
        let observer = SkippySessionRuntimeEventObserver::new();
        observer.observe(SessionLifecycleEvent::SessionReclaimed);
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn export_import_reach_the_state_lane_with_correct_outcomes() {
        let engine = install_test_engine();
        let observer = SkippySessionRuntimeEventObserver::new();
        observer.observe(SessionLifecycleEvent::RuntimeStateExportCompleted);
        observer.observe(SessionLifecycleEvent::RuntimeStateExportFailed);
        observer.observe(SessionLifecycleEvent::RuntimeStateImportCompleted);
        observer.observe(SessionLifecycleEvent::RuntimeStateImportFailed);
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"runtime_state_export_completed"));
        assert!(kinds.contains(&"runtime_state_export_failed"));
        assert!(kinds.contains(&"runtime_state_import_completed"));
        assert!(kinds.contains(&"runtime_state_import_failed"));
        clear_runtime_event_engine();
    }

    #[test]
    fn export_failure_carries_a_failure_outcome_never_success() {
        let fact = fact_for(SessionLifecycleEvent::RuntimeStateExportFailed)
            .expect("known variant must produce a fact");
        let RuntimeFact::KvRuntimeState(fact) = fact else {
            panic!("expected a KvRuntimeState fact");
        };
        assert_eq!(fact.data().outcome, Some(Outcome::Failure));
    }

    #[test]
    fn no_prohibited_fields_reach_the_session_fact_shape() {
        let fact = fact_for(SessionLifecycleEvent::SessionReset { reset_ms: 1.0 })
            .expect("known variant must produce a fact");
        let RuntimeFact::Session(fact) = fact else {
            panic!("expected a Session fact");
        };
        assert!(fact.data().summary.is_none());
        assert_eq!(
            fact.data().scope,
            mesh_llm_runtime_event_contracts::ScopeIdentities::default()
        );
    }
}
