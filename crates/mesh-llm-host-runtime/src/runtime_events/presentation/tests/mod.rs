//! Presentation projection tests: coalescing, terminal/health reservation,
//! JSON/TUI parity, and privacy allowlisting.

mod coalescing;
mod parity;
mod privacy;
mod reservation;
mod wiring;

use std::sync::Mutex;

use mesh_llm_events::OutputEvent;
use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, LogicalModelId, ModelPreparationEventKind, NativeRuntimeEventKind,
    NumericSummary, NumericSummaryKey, NumericValue, OperationId, OperationScope, Outcome,
    Progress, ProgressUnit, ReasonCode, RuntimeFact, ScopeIdentities,
};

use super::subscriber::PresentationSink;

/// Test double recording every emitted `OutputEvent` in order, standing in
/// for the real `mesh_llm_events::emit_event` sink so assertions never
/// depend on the global `OutputManager`.
#[derive(Default)]
pub(super) struct RecordingSink {
    events: Mutex<Vec<OutputEvent>>,
}

impl RecordingSink {
    pub(super) fn drain(&self) -> Vec<OutputEvent> {
        std::mem::take(
            &mut *self
                .events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

impl PresentationSink for RecordingSink {
    fn emit(&self, event: OutputEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }
}

/// A fresh root-only operation scope for a test's exclusive use.
pub(super) fn root_scope() -> OperationScope {
    OperationScope::root_only(OperationId::new())
}

/// A `Progress`-class fact carrying a model-download progress value.
pub(super) fn progress_fact(
    scope: OperationScope,
    current: u64,
    total: Option<u64>,
) -> (OperationScope, RuntimeFact) {
    let data = FactData {
        scope: ScopeIdentities {
            model_id: LogicalModelId::new("m1").ok(),
            ..ScopeIdentities::default()
        },
        progress: Some(Progress::new(current, total, ProgressUnit::Bytes)),
        ..FactData::default()
    };
    (
        scope,
        RuntimeFact::ModelPreparation(FamilyFact::with_data(
            ModelPreparationEventKind::ModelDownloadProgress,
            data,
        )),
    )
}

/// A `Terminal`-class fact with no optional `FactData` fields set.
pub(super) fn terminal_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

/// A `Terminal`-class fact carrying every optional `FactData` field, used to
/// prove the projection surfaces exactly (and only) the allowlisted set.
pub(super) fn kitchen_sink_terminal_fact() -> RuntimeFact {
    let data = FactData {
        scope: ScopeIdentities {
            model_id: LogicalModelId::new("m1").ok(),
            session_id: mesh_llm_runtime_event_contracts::SessionId::new("s1").ok(),
            request_id: mesh_llm_runtime_event_contracts::RequestId::new("r1").ok(),
            ..ScopeIdentities::default()
        },
        outcome: Some(Outcome::Failure),
        reason: Some(ReasonCode::Timeout),
        duration: Some(std::time::Duration::from_millis(1_500)),
        numeric_summaries: mesh_llm_runtime_event_contracts::BoundedNumericSummaries::new(vec![
            NumericSummary::new(
                NumericSummaryKey::new("bytes_per_sec").expect("bounded key"),
                NumericValue::Unsigned(42),
            ),
        ])
        .expect("one summary is within the bound"),
        summary: mesh_llm_runtime_event_contracts::HumanSummary::new("bounded summary").ok(),
        ..FactData::default()
    };
    RuntimeFact::NativeRuntime(FamilyFact::with_data(
        NativeRuntimeEventKind::RuntimeCrashed,
        data,
    ))
}
