//! Task 6 defect A and defect C (verifier follow-up on
//! `.omo/plans/event-system-fixes.md` task 6): the reducer's `operations`
//! map must be evicted RELEASE-triggered, not merely size-triggered, and
//! an in-flight (still-occupied) operation must never be evicted by
//! either the release-triggered path or the size-triggered capacity
//! backstop.
//!
//! Task 6-fix R1 (`.omo/plans/event-system-fixes.md`): a scope that was
//! NEVER associated with a reservation (`unreserved_ingress` with a fresh
//! `OperationId` per event -- exactly what KV cache lookups, session
//! lifecycle, and node/topology/model-lifecycle observers submit on real
//! serving traffic) can be evicted by NEITHER the release-triggered path
//! above (nothing to release) NOR the settled-only backstop (it may never
//! settle), so it needs -- and now has -- its own independent bound
//! (`UNRESERVED_OPERATION_BOUND`, `ReducerSnapshot::unreserved_order`).

use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{state_transition_fact, synthetic_unknown, terminal_success};
use crate::runtime_events::config::{
    RESERVATION_TABLE_CAPACITY, TOTAL_OPERATION_BOUND, UNRESERVED_OPERATION_BOUND,
};
use crate::runtime_events::engine::RuntimeEventEngine;

/// Task 6 defect A: the parent commit's `evict_settled_over_capacity` only
/// trimmed the `operations` map once it exceeded
/// `RESERVATION_TABLE_CAPACITY`, so 100 sequential reserve -> terminal ->
/// drain cycles left `operation_count()` growing to 100 and pinning there
/// forever -- never shrinking even though every one of those operations
/// had long since settled and released its reservation. Eviction must
/// instead be tied to the release itself, so the map tracks in-flight
/// operations only.
#[test]
fn operation_count_does_not_grow_across_sequential_completions() {
    let engine = RuntimeEventEngine::new();
    for iteration in 0..100 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            SubmitOutcome::Accepted
        );
        engine.drain();

        let tracked = engine.reducer_snapshot().operation_count();
        assert_eq!(
            tracked, 0,
            "iteration {iteration}: a settled root's reservation released \
             in this same drain pass, so the reducer must not still be \
             tracking it; got operation_count() = {tracked}"
        );
    }
}

/// Task 6 defect C: an operation whose reservation is STILL occupied must
/// never be evicted by ANY path -- release-triggered (defect A) or the
/// size-triggered capacity backstop -- even while thousands of sibling
/// operations complete, release, and get evicted around it.
#[test]
fn a_held_reservation_survives_release_triggered_eviction_of_many_completed_siblings() {
    let engine = RuntimeEventEngine::new();
    let held = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let held_scope = held.scope();
    assert_eq!(
        held.ingress().try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "a non-terminal fact keeps the reservation open but still tracks state"
    );
    engine.drain();
    assert!(
        engine.reducer_snapshot().operation(held_scope).is_some(),
        "the held scope's state-transition fact must be tracked"
    );

    for _ in 0..(RESERVATION_TABLE_CAPACITY + 200) {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            SubmitOutcome::Accepted
        );
        engine.drain();
    }

    let snapshot = engine.reducer_snapshot();
    let state = snapshot
        .operation(held_scope)
        .expect("a still-occupied reservation must never be evicted");
    assert!(
        !state.settled,
        "the held operation must still report unsettled"
    );
    assert_eq!(
        snapshot.operation_count(),
        1,
        "release-triggered eviction must leave the map tracking only the \
         still-occupied operation, not pinned at RESERVATION_TABLE_CAPACITY"
    );

    drop(held);
    engine.drain();
}

/// RENAMED + CORRECTED (task 6-fix R1, `.omo/plans/event-system-fixes.md`)
/// from `eviction_backstop_stall_is_observable_in_engine_health`, which
/// this exact scenario used to fail under: proved the settled-only
/// backstop's "nothing left to evict" branch stalled PERMANENTLY once
/// unreserved traffic exceeded `RESERVATION_TABLE_CAPACITY`, and its own
/// doc comment mischaracterized `unreserved_ingress` as merely "the only
/// way" to synthesize that condition in a test -- when in fact six real
/// production call sites (`inference/skippy/runtime_events/kv.rs`,
/// `.../session.rs`, `runtime/node_lifecycle_events.rs`,
/// `runtime/local_split/topology_events.rs`,
/// `runtime/model_lifecycle/events.rs`,
/// `inference/skippy/stage/runtime_events.rs`) submit precisely this
/// unreserved-fresh-`OperationId` shape on real serving traffic (KV cache
/// lookups and session lifecycle observers fire on EVERY request).
///
/// This is a deliberate, narrated correction of a test that was itself
/// documenting a bug now fixed -- the SAME kind of update the parent
/// commit (9a4db1c81) made to
/// `reducer::tests::lock_separation::concurrent_submit_and_drain_never_deadlock_and_settle_deterministically`
/// (`operation_count() == PRODUCERS` -> `== 0`) when release-triggered
/// eviction made the old expectation itself the stale one. The SAME
/// exact traffic shape and volume this test used to drive the stall with
/// (`RESERVATION_TABLE_CAPACITY + 64` unreserved, distinct-scope,
/// never-settling state-transition facts) now proves the OPPOSITE: the
/// map stays exactly bounded and the stall counter never fires, because
/// `RESERVATION_TABLE_CAPACITY + 64 < UNRESERVED_OPERATION_BOUND`
/// (3,136 + 64 = 3,200 < 4,096), so `unreserved_order`'s own LRU retains
/// every one of them without ever needing to evict.
#[test]
fn unreserved_flood_no_longer_stalls_the_capacity_backstop() {
    let engine = RuntimeEventEngine::new();
    assert_eq!(engine.health().snapshot().reducer_eviction_stalled, 0);

    let submitted = RESERVATION_TABLE_CAPACITY + 64;
    for _ in 0..submitted {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted,
            "a distinct scope's state-transition fact never coalesces with another scope's"
        );
    }
    engine.drain();

    assert_eq!(
        engine.reducer_snapshot().operation_count(),
        submitted,
        "every one of these {submitted} unreserved scopes is retained: {submitted} is under \
         UNRESERVED_OPERATION_BOUND ({UNRESERVED_OPERATION_BOUND}), so unreserved_order's LRU \
         never needs to evict any of them yet"
    );
    assert_eq!(
        engine.health().snapshot().reducer_eviction_stalled,
        0,
        "this exact scenario used to permanently stall the settled-only backstop; the \
         unreserved_order bound now retains this legitimate volume without ever exceeding \
         TOTAL_OPERATION_BOUND ({TOTAL_OPERATION_BOUND}), so nothing is stalled"
    );
}

/// Task 6-fix R1 red test: an UNBOUNDED stream of unreserved-ingress
/// state-transition facts, each with a fresh `OperationId` (exactly what
/// KV cache lookups and session lifecycle observers submit on ordinary
/// serving traffic), interleaving submit and drain across MANY separate
/// drain passes so the reducer's persistent `operations` map genuinely
/// accumulates across calls -- unlike a single big batch, which the
/// pre-reducer state-transition lane's own 4,096-key admission cap would
/// already truncate before ever reaching the reducer. 10,000 far exceeds
/// both `RESERVATION_TABLE_CAPACITY` (3,136) and
/// `UNRESERVED_OPERATION_BOUND` (4,096) individually, proving the bound
/// holds for genuinely unbounded traffic, not merely for a batch that
/// happens to fit under the lane's own separate cap.
#[test]
fn unbounded_unreserved_traffic_stays_within_the_frozen_bound() {
    let engine = RuntimeEventEngine::new();
    for _ in 0..10_000 {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
        engine.drain();
    }

    let tracked = engine.reducer_snapshot().operation_count();
    assert!(
        tracked <= UNRESERVED_OPERATION_BOUND,
        "10,000 unreserved submissions (zero of them ever reserved) must be capped at \
         UNRESERVED_OPERATION_BOUND ({UNRESERVED_OPERATION_BOUND}); got operation_count() = \
         {tracked}"
    );
    assert!(
        tracked <= TOTAL_OPERATION_BOUND,
        "operation_count() must never exceed the total structural ceiling; got {tracked}"
    );
}

/// Task 6-fix R1 mixed-traffic test: interleaves a genuinely-reserved,
/// still-occupied (never-terminaled) operation with BOTH ordinary reserved
/// lifecycles (reserve -> terminal -> drain, release-triggered eviction)
/// AND heavy unreserved traffic (fresh `OperationId` per event, no
/// reservation ever backing it). Proves the map stays bounded under the
/// mix AND that the live reserved operation survives -- the exact trap a
/// naive combined-map two-tier sweep (rejected design (b) in
/// baseline2.txt) could fail: without a reservation-provenance signal, a
/// size-triggered sweep cannot distinguish "genuinely still-reserved" from
/// "never-reserved and stale" by inspecting the map alone.
#[test]
fn mixed_reserved_and_unreserved_traffic_stays_bounded_without_evicting_the_live_reservation() {
    let engine = RuntimeEventEngine::new();
    let held = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let held_scope = held.scope();
    assert_eq!(
        held.ingress().try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    engine.drain();

    for i in 0..(RESERVATION_TABLE_CAPACITY + UNRESERVED_OPERATION_BOUND + 500) {
        if i % 3 == 0 {
            let reservation = engine
                .reserve_root(OperationId::new(), synthetic_unknown)
                .expect("reserve");
            assert_eq!(
                reservation.ingress().try_submit(terminal_success()),
                SubmitOutcome::Accepted
            );
        } else {
            let scope = OperationScope::root_only(OperationId::new());
            assert_eq!(
                engine
                    .unreserved_ingress(scope)
                    .try_submit(state_transition_fact()),
                SubmitOutcome::Accepted
            );
        }
        engine.drain();
    }

    let snapshot = engine.reducer_snapshot();
    let held_state = snapshot
        .operation(held_scope)
        .expect("a still-occupied reservation must never be evicted by either bound");
    assert!(
        !held_state.settled,
        "the held operation must still report unsettled"
    );
    let tracked = snapshot.operation_count();
    assert!(
        tracked <= TOTAL_OPERATION_BOUND,
        "mixed reserved/unreserved traffic must stay within TOTAL_OPERATION_BOUND \
         ({TOTAL_OPERATION_BOUND}); got operation_count() = {tracked}"
    );

    drop(held);
    engine.drain();
}
