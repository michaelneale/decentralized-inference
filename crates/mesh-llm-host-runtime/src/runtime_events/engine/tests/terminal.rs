use mesh_llm_runtime_event_contracts::{
    ChildOperationId, OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn terminal_submit_accepted_and_second_write_rejected_as_duplicate() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();

    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
}

#[test]
fn unreserved_terminal_is_always_delivery_failed() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
}

#[test]
fn late_terminal_after_release_is_delivery_failed() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    reservation.cancel();

    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
}

#[test]
fn id_mismatched_terminal_against_a_reused_slot_is_delivery_failed() {
    let engine = RuntimeEventEngine::with_capacity(1);
    let first = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve first");
    let stale_ingress = first.ingress();
    first.cancel();

    let second = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve second into the recycled slot");

    // The stale ingress still points at slot 0, now owned by a different
    // operation at a newer generation: the ID no longer matches.
    assert_eq!(
        stale_ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
    assert_eq!(
        second.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
}

#[test]
fn child_terminal_consumes_the_child_reservation_not_the_root() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");

    assert_eq!(
        child.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    // The root's own slot is untouched by the child's terminal write.
    assert_eq!(
        root.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
}
