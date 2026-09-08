use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, OperationId, OperationScope, Outcome, Progress, ProgressUnit,
    RequestEventKind, RuntimeEventIngress, RuntimeFact,
};

use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, RejectReason, apply};

fn terminal_success() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

fn synthetic_unknown() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::new(RequestEventKind::RequestFailed))
}

fn progress_fact(current: u64) -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestQueued,
        FactData {
            progress: Some(Progress::new(current, Some(100), ProgressUnit::Tokens)),
            ..FactData::default()
        },
    ))
}

/// Sixteen producer threads submit terminals concurrently with a drain
/// thread reducing them, synchronized only by a `Barrier` (no sleeps). The
/// reducer's own lock (`reducer_state`) is disjoint from every reservation
/// slot's lock, so this must complete deterministically without deadlock,
/// and every accepted terminal must actually be drained (proven by the
/// `applied` accumulator below reaching exactly `PRODUCERS`, one per
/// terminal, before the loop can exit).
///
/// Task 6-fix defect A (`.omo/plans/event-system-fixes.md`): this test
/// previously asserted `operation_count() == PRODUCERS` -- i.e. that a
/// settled operation stays tracked in the reducer's `operations` map
/// forever, which was precisely the unbounded-growth bug defect A fixes.
/// Every one of these 16 is a bare root reservation with no children, so
/// release-triggered eviction now retires each one in the SAME drain pass
/// that applies its terminal; the corrected expectation is
/// `operation_count() == 0`, which this test now also uses to prove
/// release-triggered eviction itself is race-free under concurrent
/// submit+drain, not just deadlock-free.
#[test]
fn concurrent_submit_and_drain_never_deadlock_and_settle_deterministically() {
    let engine = RuntimeEventEngine::with_capacity(32);
    const PRODUCERS: usize = 16;
    let start = Arc::new(Barrier::new(PRODUCERS + 1));

    let handles: Vec<_> = (0..PRODUCERS)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                let reservation = engine
                    .reserve_root(OperationId::new(), synthetic_unknown)
                    .expect("capacity is large enough for every producer");
                let ingress = reservation.ingress();
                start.wait();
                ingress.try_submit(terminal_success())
            })
        })
        .collect();

    start.wait();
    // Drain concurrently while producers are still submitting; the wake
    // list and reservation table's own locks make this safe, and the
    // reducer never takes a reservation-table lock while reducing.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut applied = 0;
    while applied < PRODUCERS && Instant::now() < deadline {
        let report = engine.drain();
        applied += report.applied;
        if report.applied == 0 {
            thread::yield_now();
        }
    }

    let outcomes = handles
        .into_iter()
        .map(|handle| handle.join().expect("producer thread must not panic"))
        .collect::<Vec<_>>();
    assert!(
        outcomes
            .iter()
            .all(|outcome| *outcome == mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted),
        "every concurrent producer must be accepted: {outcomes:?}"
    );
    assert_eq!(
        applied, PRODUCERS,
        "all accepted terminals must drain before the generous deadline"
    );

    let snapshot = engine.reducer_snapshot();
    assert_eq!(
        snapshot.operation_count(),
        0,
        "every one of these 16 root-only operations settled and released within the same \
         drain pass that applied it, so release-triggered eviction must have retired all 16, \
         even under concurrent submission"
    );
}

/// A genuinely rejected reducer input never mutates the accepted snapshot:
/// apply a valid progress fact, then reject a lower progress value and prove
/// the accepted snapshot still carries the last-valid value.
#[test]
fn transactional_rollback_on_rejected_progress_leaves_snapshot_unchanged() {
    let operation = OperationId::new();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        crate::runtime_events::reducer::ReducerInput {
            scope: OperationScope::root_only(operation),
            ingress_sequence: 0,
            native_sequence: None,
            wall_clock_hint: None,
            synthesized: false,
            reserved: true,
            fact: progress_fact(80),
        },
    ) else {
        panic!("the initial progress fact must apply");
    };
    let outcome = apply(
        &snapshot,
        crate::runtime_events::reducer::ReducerInput {
            scope: OperationScope::root_only(operation),
            ingress_sequence: 1,
            native_sequence: None,
            wall_clock_hint: None,
            synthesized: false,
            reserved: true,
            fact: progress_fact(50),
        },
    );
    assert!(
        matches!(
            outcome,
            ReduceOutcome::Rejected(RejectReason::StaleProgress)
        ),
        "a rejected reducer input must return no replacement snapshot"
    );
    assert_eq!(
        snapshot
            .operation(OperationScope::root_only(operation))
            .expect("operation state")
            .last_progress_current,
        Some(80),
        "a rejected regression must leave the previous progress value intact"
    );
}
