use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FactData, FamilyFact, GenerationEventKind, NativeRuntimeEventKind,
    Outcome, ReasonCode, RequestEventKind, RuntimeFact,
};

pub(super) fn terminal_success() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

pub(super) fn progress_fact() -> RuntimeFact {
    RuntimeFact::Generation(FamilyFact::new(GenerationEventKind::GenerationProgress))
}

pub(super) fn diagnostic_fact() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(DiagnosticEventKind::WarningRaised))
}

pub(super) fn state_transition_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

/// The frozen synthesis contract for a dropped/undelivered guard: an
/// otherwise family-correct `Terminal`-class fact carrying
/// `reason: TerminalNotDelivered` and `outcome: Unknown`.
pub(super) fn synthetic_unknown() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}
