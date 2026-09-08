use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, OperationId, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::config::CHILD_SETTLE_GRACE;
use crate::runtime_events::engine::RuntimeEventEngine;

/// Review defect D8 (`.omo/plans/event-system-fixes.md` task 5): the root's
/// release must not cascade-force-release a still-occupied child WITHOUT
/// ever giving it a terminal. Under the old `cascade_children`/
/// `force_complete_child` behavior, `engine.drain()` after the root's own
/// terminal settles immediately force-frees the child's slot (bumping its
/// generation) even though the child never got a terminal -- so the
/// child's own, legitimately in-flight terminal arriving moments later is
/// wrongly rejected as stale (`TerminalDeliveryFailed`) instead of
/// accepted. This is the review's exact "17 of 36 `generation_completed`
/// terminals lost" mechanism under 40 concurrent requests.
#[test]
fn root_terminal_defers_its_own_release_while_a_child_is_still_occupied() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");

    assert_eq!(
        root.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    engine.drain();

    assert_eq!(
        child.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted,
        "a child terminal that arrives while the root has already settled \
         must still be accepted, not force-released out from under it"
    );
    engine.drain();
    assert_eq!(engine.occupied_count(), 0);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 0);
}

/// Task 5's deterministic-scheduler acceptance: "root terminal at
/// sequence n, child terminal at n+1 drained on a later tick, both
/// publish, `terminal_delivery_failed == 0`."
#[test]
fn child_terminal_drained_on_a_later_tick_still_publishes_normally() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");

    // Sequence n: the root's own terminal, drained on the first tick.
    assert_eq!(
        root.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    let tick_one = Instant::now();
    engine.drain_up_to_at(None, tick_one);
    assert_eq!(
        engine.occupied_count(),
        2,
        "the root's own slot release defers while the child is still \
         occupied -- only its TERMINAL applied this pass, not its release"
    );

    // Sequence n+1, submitted after the first tick and drained on a
    // strictly LATER one.
    assert_eq!(
        child.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    let tick_two = tick_one + Duration::from_millis(1);
    engine.drain_up_to_at(None, tick_two);

    assert_eq!(engine.occupied_count(), 0);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 0);
}

#[test]
fn settled_child_slots_are_removed_during_repeated_child_churn() {
    let engine = RuntimeEventEngine::with_capacity(2);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");

    for _ in 0..32 {
        let child = engine
            .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
            .expect("reserve child");
        child.cancel();
    }

    assert!(
        engine
            .children_by_root
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&root_id)
            .is_none(),
        "a root must not retain settled child indices across churn"
    );
    root.cancel();
}

#[test]
fn a_reused_child_index_does_not_delay_unrelated_root_settlement() {
    let engine = RuntimeEventEngine::with_capacity(2);
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
    engine.drain();
    drop(child);

    // The released child index is immediately reused by an unrelated root.
    let unrelated = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reuse child slot for unrelated root");

    assert_eq!(
        root.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    engine.drain();

    assert_eq!(
        engine.occupied_count(),
        1,
        "the stale child generation must not make the settled root wait on an unrelated root"
    );
    unrelated.cancel();
    drop(root);
}

/// Task 5's grace-expiry acceptance: a child that never settles gets a
/// synthesized `terminal_not_delivered` terminal published through its
/// OWN slot once `CHILD_SETTLE_GRACE` elapses, and the root releases
/// right after -- "forgotten-child test degrades, never fails."
#[test]
fn a_forgotten_child_is_synthesized_and_published_at_grace_expiry() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");
    std::mem::forget(child); // never settles on its own

    let start = Instant::now();
    root.ingress().try_submit(terminal_success());
    engine.drain_up_to_at(None, start);
    assert_eq!(engine.occupied_count(), 2, "root deferred, child forgotten");

    // Still short of the deadline: nothing resolves yet.
    engine.drain_up_to_at(None, start + CHILD_SETTLE_GRACE - Duration::from_millis(1));
    assert_eq!(engine.occupied_count(), 2, "grace has not elapsed yet");

    // Grace elapses: the forgotten child's slot is synthesized and
    // enqueued this pass, then applied+published on the next.
    let past_deadline = start + CHILD_SETTLE_GRACE + Duration::from_millis(1);
    engine.drain_up_to_at(None, past_deadline);
    engine.drain_up_to_at(None, past_deadline);

    assert_eq!(
        engine.occupied_count(),
        0,
        "the forgotten child, and then the root, must both settle -- \
         degrading, never leaking a slot forever"
    );
    assert_eq!(
        engine.health().snapshot().terminal_delivery_failed,
        0,
        "a grace-expiry synthesis is a controlled degrade, not a delivery \
         failure -- it publishes a real (synthesized) terminal, unlike the \
         old terminal-less force-release"
    );
}

#[test]
fn grace_expiry_release_frees_capacity_consumed_by_a_forgotten_child() {
    let engine = RuntimeEventEngine::with_capacity(2);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");
    std::mem::forget(child);

    assert!(
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .is_none(),
        "table is full: root + its one outstanding child"
    );

    let start = Instant::now();
    root.ingress().try_submit(terminal_success());
    engine.drain_up_to_at(None, start); // establishes the deferral + deadline
    let past_deadline = start + CHILD_SETTLE_GRACE + Duration::from_millis(1);
    engine.drain_up_to_at(None, past_deadline);
    engine.drain_up_to_at(None, past_deadline);

    assert!(
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .is_some(),
        "releasing the root at grace expiry must also reclaim its \
         forgotten child's slot"
    );
}

#[test]
fn a_child_guard_that_later_drops_after_grace_expiry_release_cannot_resurrect_the_freed_slot() {
    let engine = RuntimeEventEngine::with_capacity(2);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");

    let start = Instant::now();
    root.ingress().try_submit(terminal_success());
    engine.drain_up_to_at(None, start); // establishes the deferral + deadline
    let past_deadline = start + CHILD_SETTLE_GRACE + Duration::from_millis(1);
    engine.drain_up_to_at(None, past_deadline);
    engine.drain_up_to_at(None, past_deadline);
    assert_eq!(engine.occupied_count(), 0);

    // A second, unrelated operation may now legitimately claim a freed
    // slot before the original child guard below ever runs its Drop.
    let unrelated = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("slot is free for reuse");

    drop(child); // the late Drop must not touch the slot `unrelated` now owns

    assert_eq!(
        unrelated.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted,
        "a late child-guard drop must not have written a stray terminal into the reused slot"
    );
    engine.drain_up_to_at(None, past_deadline);
    assert_eq!(engine.occupied_count(), 0);
}

/// Task 5's stress acceptance: "stress test with 64 concurrent roots and
/// two children each shows zero lost terminals" -- review defect D8's
/// 40-concurrent-request scenario, driven through the real
/// reservation/drain API with real OS threads and a `Barrier`.
#[test]
fn sixty_four_concurrent_roots_with_two_children_each_lose_no_terminal() {
    const ROOTS: usize = 64;
    let engine = RuntimeEventEngine::with_capacity(ROOTS * 3 + 8);
    let barrier = Arc::new(Barrier::new(ROOTS));
    let settled = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..ROOTS)
        .map(|_| {
            let engine = engine.clone();
            let barrier = Arc::clone(&barrier);
            let settled = Arc::clone(&settled);
            thread::spawn(move || {
                let root_id = OperationId::new();
                let root = engine
                    .reserve_root(root_id, synthetic_unknown)
                    .expect("reserve root");
                let child_a = engine
                    .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
                    .expect("reserve child a");
                let child_b = engine
                    .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
                    .expect("reserve child b");

                barrier.wait();

                // The root settles FIRST across every worker, exactly
                // matching D8's "root terminal arrives while children
                // are still occupied".
                assert_eq!(
                    root.ingress().try_submit(terminal_success()),
                    SubmitOutcome::Accepted
                );
                engine.drain();
                assert_eq!(
                    child_a.ingress().try_submit(terminal_success()),
                    SubmitOutcome::Accepted,
                    "child a must still be live when the root settled first"
                );
                assert_eq!(
                    child_b.ingress().try_submit(terminal_success()),
                    SubmitOutcome::Accepted,
                    "child b must still be live when the root settled first"
                );
                settled.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
    for _ in 0..8 {
        engine.drain();
    }

    assert_eq!(settled.load(Ordering::Relaxed), ROOTS);
    assert_eq!(engine.occupied_count(), 0);
    assert_eq!(
        engine.health().snapshot().terminal_delivery_failed,
        0,
        "zero lost terminals across concurrent roots with occupied children"
    );
}
