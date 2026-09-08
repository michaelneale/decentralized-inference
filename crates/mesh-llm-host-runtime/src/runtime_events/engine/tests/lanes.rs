use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown};
use crate::runtime_events::config::DIAGNOSTIC_LANE_DEPTH;
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn progress_on_a_reserved_operation_coalesces() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    assert_eq!(
        reservation.ingress().try_submit(progress_fact()),
        SubmitOutcome::Coalesced
    );
    assert_eq!(
        reservation.ingress().try_submit(progress_fact()),
        SubmitOutcome::Coalesced,
        "a second progress update overwrites the single per-operation slot"
    );
}

#[test]
fn unreserved_progress_is_dropped_and_counted() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine.unreserved_ingress(scope).try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    assert_eq!(engine.health().snapshot().dropped_progress, 1);
}

#[test]
fn state_transitions_accept_then_coalesce_the_same_kind() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Coalesced,
        "repeating the same kind coalesces the latest value rather than growing the lane"
    );
}

#[test]
fn diagnostics_beyond_the_lane_depth_are_dropped_and_counted() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..DIAGNOSTIC_LANE_DEPTH {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
    }

    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::DroppedDiagnostic
    );
    assert_eq!(engine.health().snapshot().dropped_diagnostic, 1);
}
