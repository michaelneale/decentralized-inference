use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn bypass_defaults_to_disabled() {
    let engine = RuntimeEventEngine::with_capacity(4);
    assert!(!engine.progress_diagnostic_class_bypass());
}

#[test]
fn bypass_enabled_drops_progress_before_any_lane_or_coalescing() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.set_progress_diagnostic_class_bypass(true);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    assert_eq!(
        reservation.ingress().try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress,
        "even a RESERVED operation's progress is bypassed, never coalesced"
    );
    assert_eq!(engine.health().snapshot().dropped_progress, 1);
}

#[test]
fn bypass_enabled_drops_diagnostic_before_the_diagnostic_lane() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.set_progress_diagnostic_class_bypass(true);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::DroppedDiagnostic
    );
    assert_eq!(engine.health().snapshot().dropped_diagnostic, 1);
}

#[test]
fn bypass_enabled_never_touches_terminal_or_state_transition() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.set_progress_diagnostic_class_bypass(true);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "StateTransition class must remain fully active under the bypass"
    );
    assert_eq!(
        engine.state_lane_kinds().len(),
        1,
        "the state lane must actually receive the fact, not just report Accepted"
    );

    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    assert_eq!(
        reservation
            .ingress()
            .try_submit(super::fixtures::terminal_success()),
        SubmitOutcome::Accepted,
        "Terminal class must remain fully active -- reservations and the reducer's wake seam are untouched"
    );
}

#[test]
fn disabling_bypass_again_restores_normal_progress_coalescing() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.set_progress_diagnostic_class_bypass(true);
    engine.set_progress_diagnostic_class_bypass(false);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    assert_eq!(
        reservation.ingress().try_submit(progress_fact()),
        SubmitOutcome::Coalesced,
        "toggling the bypass back off restores the ordinary Progress lane"
    );
    assert_eq!(engine.health().snapshot().dropped_progress, 0);
}
