use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::config::RESERVATION_TABLE_CAPACITY;
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn capacity_matches_the_frozen_derivation() {
    let engine = RuntimeEventEngine::new();
    assert_eq!(engine.replay().len(), 0);
    assert_eq!(RESERVATION_TABLE_CAPACITY, 3_136);
}

#[test]
fn reservation_exhaustion_never_refuses_primary_work() {
    let engine = RuntimeEventEngine::with_capacity(1);
    let _held = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("first reservation succeeds");

    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let exhausted = engine.reserve_root(OperationId::new(), synthetic_unknown);
    assert!(
        exhausted.is_none(),
        "second reservation is exhausted at capacity 1"
    );
    assert_eq!(engine.health().snapshot().reservation_exhausted, 1);

    // Primary work proceeds unreserved: the caller still gets an ingress
    // handle and can still submit its terminal, it just cannot be tracked
    // by a slot.
    let unreserved = engine.unreserved_ingress(scope);
    let outcome = unreserved.try_submit(terminal_success());
    assert_eq!(
        outcome,
        SubmitOutcome::TerminalDeliveryFailed,
        "unreserved delivery is recorded as a delivery failure, never as work refusal"
    );
}

#[test]
fn exact_outcome_and_counter_accounting_for_one_operation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    assert_eq!(
        reservation.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    assert_eq!(engine.drain().applied, 1);

    let snapshot = engine.health().snapshot();
    assert_eq!(snapshot.reservation_exhausted, 0);
    assert_eq!(snapshot.terminal_delivery_failed, 0);
    assert_eq!(engine.replay().len(), 1);
}
