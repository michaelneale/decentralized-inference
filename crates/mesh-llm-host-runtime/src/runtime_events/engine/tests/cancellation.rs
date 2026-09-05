use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{OperationId, Outcome, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{state_transition_fact, synthetic_unknown, terminal_success};
use crate::runtime_events::config::STATE_TRANSITION_LANE_DEPTH;
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn explicit_cancellation_releases_without_a_terminal_or_wake_entry() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    reservation.cancel();

    assert_eq!(engine.drain().applied, 0);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 0);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
}

#[test]
fn stale_state_submission_after_cancellation_is_not_capacity_loss() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..STATE_TRANSITION_LANE_DEPTH {
        let scope = OperationId::new();
        assert_eq!(
            engine
                .unreserved_ingress(mesh_llm_runtime_event_contracts::OperationScope::root_only(
                    scope,
                ))
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
    }

    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reservation remains independent of the state lane");
    let stale_ingress = reservation.ingress();
    reservation.cancel();

    let before = engine.health().snapshot();
    assert_eq!(
        stale_ingress.try_submit(state_transition_fact()),
        SubmitOutcome::RejectedCancelled
    );
    let after = engine.health().snapshot();
    assert_eq!(
        after.state_transition_rejected, before.state_transition_rejected,
        "a stale reservation must not be counted as lane-capacity loss"
    );
    assert_eq!(after.cancelled_reservation_rejected, 1);
    assert_eq!(after.state_degraded, before.state_degraded);
    assert_eq!(after.rebuild_required, before.rebuild_required);
}

#[test]
fn cancellation_drops_the_engine_arc_without_synthesizing_a_terminal() {
    let weak = {
        let engine = RuntimeEventEngine::with_capacity(4);
        let weak = Arc::downgrade(&engine);
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        reservation.cancel();
        weak
    };

    assert!(
        weak.upgrade().is_none(),
        "cancel must release its guard-held engine Arc"
    );
}

#[test]
fn deferred_root_cancellation_drops_its_engine_arc_after_children_settle() {
    let weak = {
        let engine = RuntimeEventEngine::with_capacity(4);
        let weak = Arc::downgrade(&engine);
        let root_id = OperationId::new();
        let root = engine
            .reserve_root(root_id, synthetic_unknown)
            .expect("reserve root");
        let child = engine
            .reserve_child(
                root_id,
                mesh_llm_runtime_event_contracts::ChildOperationId::new(),
                synthetic_unknown,
            )
            .expect("reserve child");

        root.cancel();
        child.cancel();
        engine.drain();
        weak
    };

    assert!(
        weak.upgrade().is_none(),
        "a deferred cancelled root must not keep the engine alive"
    );
}

#[test]
fn dropped_guard_with_no_terminal_synthesizes_terminal_not_delivered_unknown() {
    let engine = RuntimeEventEngine::with_capacity(4);
    {
        let _reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        // Guard drops here without ever submitting a terminal: this is the
        // "forgotten terminal" / crash-equivalent path.
    }

    let report = engine.drain();
    assert_eq!(report.applied, 1);

    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].fact.data().outcome,
        Some(Outcome::Unknown),
        "a forgotten terminal on an otherwise-successful path must degrade, never manufacture a failure outcome"
    );
}

#[test]
fn dropped_guard_after_a_real_terminal_does_not_double_synthesize() {
    let engine = RuntimeEventEngine::with_capacity(4);
    {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
        );
        // Guard drops after a real terminal was already written.
    }

    let report = engine.drain();
    assert_eq!(report.applied, 1);
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].fact.data().outcome, Some(Outcome::Success));
}

#[test]
fn queued_reserved_state_cannot_resurrect_a_cancelled_deferred_root() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let root_ingress = root.ingress();
    let child = engine
        .reserve_child(
            root_id,
            mesh_llm_runtime_event_contracts::ChildOperationId::new(),
            synthetic_unknown,
        )
        .expect("reserve child");
    assert_eq!(
        root_ingress.try_submit(state_transition_fact()),
        mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
    );

    root.cancel();
    child.cancel();
    engine.drain();

    assert!(
        engine.replay().snapshot().is_empty(),
        "a fact queued before cancellation must not reinsert the cancelled root"
    );
}
