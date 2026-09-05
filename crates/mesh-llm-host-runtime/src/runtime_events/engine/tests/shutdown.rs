use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{
    diagnostic_fact, state_transition_fact, synthetic_unknown, terminal_success,
};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn shutdown_drains_a_small_wake_list_fully_within_budget() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    reservation.ingress().try_submit(terminal_success());

    let report = engine.shutdown(None);

    assert_eq!(report.started_with, 1);
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 0);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
    assert!(engine.is_shutting_down());
}

#[test]
fn shutdown_past_its_drain_budget_degrades_the_remainder_instead_of_hanging() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..3 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        reservation.ingress().try_submit(terminal_success());
    }

    // Simulate the deadline expiring after exactly one drained item,
    // deterministically (no real sleep or clock dependence).
    let report = engine.shutdown(Some(1));

    assert_eq!(report.started_with, 3);
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 2);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 1);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 2);
}

#[test]
fn shutdown_closes_new_admission_and_submission() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.shutdown(Some(0));

    assert!(
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .is_none()
    );
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    assert_eq!(
        engine
            .unreserved_ingress(scope)
            .try_submit(terminal_success()),
        SubmitOutcome::RejectedShuttingDown
    );
}

#[test]
fn shutdown_chunks_preserve_one_global_prefix_across_state_and_terminal_lanes() {
    let engine = RuntimeEventEngine::with_capacity(128);
    for _ in 0..65 {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
    }
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve terminal");
    assert_eq!(
        reservation.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );

    let report = engine.shutdown(None);
    assert_eq!(report.remaining_after_deadline, 0);
    let sequences: Vec<_> = engine
        .replay()
        .snapshot()
        .into_iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences.len(), 66);
    assert!(
        sequences.windows(2).all(|pair| pair[0] < pair[1]),
        "shutdown replay regressed across its work chunks: {sequences:?}"
    );
}

#[test]
fn shutdown_budget_continues_past_rejected_lane_prefix_to_publish_later_work() {
    let engine = RuntimeEventEngine::with_capacity(128);

    // Two rejected lane entries per reservation make a 66-entry prefix,
    // which is larger than SHUTDOWN_WORK_CHUNK. Cancelling each reservation
    // leaves its already-queued state/diagnostic values physically present,
    // but reservation validation rejects them during shutdown. The terminal
    // below remains the only wake entry throughout that prefix, so a
    // continuation check based only on applied or wake-length changes would
    // stop after the first rejected chunk.
    for _ in 0..33 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve rejected-prefix operation");
        let ingress = reservation.ingress();
        assert_eq!(
            ingress.try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
        reservation.cancel();
    }

    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve eligible terminal");
    assert_eq!(
        reservation.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );

    let report = engine.shutdown(Some(1));
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 0);
    assert_eq!(engine.replay().snapshot().len(), 1);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 0);

    drop(reservation);
}

#[test]
fn shutdown_budget_continues_past_a_stale_wake_prefix() {
    let engine = RuntimeEventEngine::with_capacity(4);

    let stale = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve stale terminal");
    assert_eq!(
        stale.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    stale.cancel();

    let live = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve live terminal");
    assert_eq!(
        live.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );

    let report = engine.shutdown(Some(1));
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 0);
    assert_eq!(engine.replay().snapshot().len(), 1);

    drop(live);
}
