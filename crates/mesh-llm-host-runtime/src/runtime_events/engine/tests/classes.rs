//! Task 4 acceptance: every delivery class (terminal, state-transition,
//! progress, diagnostic) is applied through the reducer and published,
//! coalescing is per `(OperationScope, kind)` -- never globally by kind --
//! and the ONE shared engine-level ingress-sequence counter (`wake.rs`) is
//! consumed by every submit outcome, so a coalesced/dropped/failed input
//! leaves a permanent gap in the published sequence space rather than
//! silently vanishing. Fixes review defect D2
//! (`.omo/plans/event-system-fixes.md` task 4): before this module,
//! `engine::drain` only ever processed the terminal wake list, so state,
//! progress, and diagnostic facts were accepted at the lane but never
//! reached the reducer or a subscriber.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FactData, FamilyFact, OperationId, OperationScope, Outcome,
    RequestEventKind, RuntimeEventIngress, RuntimeFact, Severity, SubmitOutcome,
};

use super::fixtures::{
    diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown, terminal_success,
};
use crate::runtime_events::config::{
    DIAGNOSTIC_LANE_DEPTH, PROGRESS_EXPORT_INTERVAL, STATE_TRANSITION_LANE_DEPTH,
};
use crate::runtime_events::engine::RuntimeEventEngine;

/// The core D2 regression: two DIFFERENT operations reporting the SAME
/// state-transition kind must both reach the reducer and publish -- a
/// global-by-kind lane would silently drop one of them.
#[test]
fn state_transitions_from_different_scopes_never_coalesce_across_scopes() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope_a = OperationScope::root_only(OperationId::new());
    let scope_b = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope_b)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "the same kind for a DIFFERENT scope must never coalesce with scope_a's pending entry"
    );

    let report = engine.drain();
    assert_eq!(
        report.applied, 2,
        "both scopes' state-transition facts must be drained"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(
        frames.len(),
        2,
        "both scopes' facts must be applied and published, not just one"
    );
    let published_scopes: HashSet<OperationScope> =
        frames.iter().map(|frame| frame.scope).collect();
    assert!(published_scopes.contains(&scope_a));
    assert!(published_scopes.contains(&scope_b));
}

/// Same scope, same kind: this SHOULD coalesce (unchanged from before task
/// 4), but the value that eventually publishes must be the LATEST
/// submission's fact and ingress sequence, not the first.
#[test]
fn state_transition_repeat_key_coalesces_to_the_latest_sequence() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Coalesced
    );

    let report = engine.drain();
    assert_eq!(
        report.applied, 1,
        "one coalesced key drains as exactly one entry"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].sequence.get(),
        2,
        "the published frame must carry the SECOND (latest) submission's sequence, not the first"
    );
}

/// A key drained by one `drain()` call is gone from the lane: the next
/// submission for the same `(scope, kind)` starts fresh (`Accepted`, not
/// `Coalesced`).
#[test]
fn a_drained_state_transition_key_starts_fresh_on_the_next_submission() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    engine.drain();

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "the lane must be empty again after a full drain, so the same key is fresh, not a repeat"
    );
    assert_eq!(engine.drain().applied, 1);
    assert_eq!(
        engine.replay().snapshot().len(),
        2,
        "both drains published their own frame"
    );
}

/// Diagnostics currently sit in their bounded queue forever (D2): this
/// proves a submitted diagnostic is actually applied through the reducer
/// and published once `drain()` runs.
#[test]
fn diagnostic_facts_drain_through_the_reducer_and_publish() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..3 {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
    }

    let report = engine.drain();
    assert_eq!(report.applied, 3);
    assert_eq!(
        engine.replay().snapshot().len(),
        3,
        "every accepted diagnostic must reach the stream"
    );
}

/// A diagnostic dropped for exceeding the bounded queue depth must never
/// reach the reducer once the queue drains -- only what was actually
/// accepted publishes.
#[test]
fn diagnostics_dropped_past_the_depth_bound_never_reach_the_reducer() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
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

    engine.drain();
    assert_eq!(
        engine.replay().snapshot().len(),
        DIAGNOSTIC_LANE_DEPTH,
        "the dropped entry must never have been applied or published"
    );
}

#[test]
fn rust_ingress_defaults_preserve_failure_unknown_and_fatal_severity() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let failure_scope = OperationScope::root_only(OperationId::new());
    let unknown_scope = OperationScope::root_only(OperationId::new());
    let failure = RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestExecutionStarted,
        FactData {
            outcome: Some(Outcome::Failure),
            ..FactData::default()
        },
    ));
    let unknown = RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestExecutionStarted,
        FactData {
            outcome: Some(Outcome::Unknown),
            ..FactData::default()
        },
    ));
    assert_eq!(
        engine.unreserved_ingress(failure_scope).try_submit(failure),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine.unreserved_ingress(unknown_scope).try_submit(unknown),
        SubmitOutcome::Accepted
    );

    let fatal_reservation = engine
        .reserve_root(OperationId::new(), super::fixtures::synthetic_unknown)
        .expect("reserve fatal diagnostic");
    assert_eq!(
        fatal_reservation
            .ingress()
            .try_submit(RuntimeFact::Diagnostic(FamilyFact::new(
                DiagnosticEventKind::FatalNativeFailure,
            ))),
        SubmitOutcome::Accepted
    );
    engine.drain();

    let frames = engine.replay().snapshot();
    assert_eq!(
        frames[0].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Error)
    );
    assert_eq!(
        frames[1].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Warning)
    );
    assert_eq!(
        frames[2].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Fatal)
    );
}

/// Progress facts flush at most once per the frozen 100 ms export
/// interval, carrying only the latest coalesced value -- proven with pure
/// `Instant` arithmetic (`drain_up_to_at`), never a real sleep.
#[test]
fn progress_flushes_at_most_once_per_hundred_milliseconds_with_the_latest_value() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    let t0 = Instant::now();

    // Consume the "first call always flushes" baseline with nothing
    // pending yet, so the 100 ms gate below measures from a known instant.
    assert_eq!(engine.drain_up_to_at(None, t0).applied, 0);

    assert_eq!(
        ingress.try_submit(progress_fact()),
        SubmitOutcome::Coalesced
    );
    let mid_flush = engine.drain_up_to_at(None, t0 + Duration::from_millis(40));
    assert_eq!(
        mid_flush.applied, 0,
        "under the 100ms interval, progress must not flush yet"
    );
    assert!(engine.replay().is_empty());

    // A second progress update before the interval elapses overwrites the
    // same per-operation slot -- still only one value pending.
    assert_eq!(
        ingress.try_submit(progress_fact()),
        SubmitOutcome::Coalesced
    );

    let due_flush = engine.drain_up_to_at(None, t0 + PROGRESS_EXPORT_INTERVAL);
    assert_eq!(
        due_flush.applied, 1,
        "exactly one progress frame publishes once the interval elapses"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].sequence.get(),
        2,
        "the flushed frame must carry the LATEST progress submission's sequence"
    );
}

/// Terminal, state-transition, and diagnostic facts submitted for the SAME
/// scope must all apply through the SAME transactional reducer, in
/// ingress-sequence order -- the reducer's own per-scope ordering
/// invariant is unaffected by which lane a fact travels through.
#[test]
fn mixed_classes_for_one_scope_apply_in_ingress_sequence_order() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope)
            .try_submit(diagnostic_fact()),
        SubmitOutcome::Accepted
    );

    let report = engine.drain();
    assert_eq!(report.applied, 2);
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 2);
    let sequences: Vec<u64> = frames.iter().map(|frame| frame.sequence.get()).collect();
    let mut sorted = sequences.clone();
    sorted.sort_unstable();
    assert_eq!(
        sequences, sorted,
        "mixed classes for one scope must publish in ingress-sequence order"
    );
    assert_eq!(sequences, vec![1, 2]);
}

#[test]
fn reserved_state_before_its_terminal_survives_same_pass_release() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );

    engine.drain();
    let sequences: Vec<_> = engine
        .replay()
        .snapshot()
        .into_iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences, vec![1, 2]);
}

#[test]
fn drain_report_excludes_state_rejected_by_reservation_validation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    reservation.cancel();

    let report = engine.drain();
    assert_eq!(report.applied, 0);
    assert_eq!(report.left_queued, 0);
    assert!(engine.replay().is_empty());
}

#[test]
fn drain_report_excludes_facts_rejected_by_the_reducer() {
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
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    let report = engine.drain();
    assert_eq!(report.applied, 1);
    assert_eq!(engine.replay().snapshot().len(), 1);
    assert_eq!(engine.health().snapshot().reducer_rejected, 1);
}

#[test]
fn drain_report_excludes_progress_superseded_by_the_published_frontier() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    let t0 = Instant::now();
    assert_eq!(engine.drain_up_to_at(None, t0).applied, 0);

    assert_eq!(
        ingress.try_submit(progress_fact()),
        SubmitOutcome::Coalesced
    );
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .drain_up_to_at(None, t0 + Duration::from_millis(40))
            .applied,
        1
    );

    let report = engine.drain_up_to_at(None, t0 + PROGRESS_EXPORT_INTERVAL);
    assert_eq!(report.applied, 0);
    assert_eq!(engine.replay().snapshot().len(), 1);
    assert_eq!(engine.health().snapshot().dropped_progress, 1);

    reservation.cancel();
}

/// The crux of task 4: ONE shared atomic counter, consumed by every
/// submit outcome, regardless of class or whether the fact was ever
/// queued anywhere. `peek_next_sequence` never itself consumes a
/// sequence, so each assertion below isolates exactly one `try_submit`
/// call's own consumption.
#[test]
fn every_submit_outcome_consumes_the_shared_ingress_sequence_counter() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    let mut expected = engine.wake().peek_next_sequence();

    // Accepted (state-transition, first time for this key).
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    expected += 1;
    assert_eq!(engine.wake().peek_next_sequence(), expected);

    // Coalesced (repeat kind, same scope).
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Coalesced
    );
    expected += 1;
    assert_eq!(engine.wake().peek_next_sequence(), expected);

    // DroppedProgress (unreserved -- no slot to coalesce into).
    assert_eq!(
        ingress.try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    expected += 1;
    assert_eq!(engine.wake().peek_next_sequence(), expected);

    // TerminalDeliveryFailed (unreserved terminal -- no slot to own it).
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
    expected += 1;
    assert_eq!(engine.wake().peek_next_sequence(), expected);

    // Fill the diagnostic queue to depth (each Accepted, one mint apiece).
    for _ in 0..DIAGNOSTIC_LANE_DEPTH {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
        expected += 1;
    }
    assert_eq!(engine.wake().peek_next_sequence(), expected);

    // DroppedDiagnostic: the queue is now at depth.
    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::DroppedDiagnostic
    );
    expected += 1;
    assert_eq!(engine.wake().peek_next_sequence(), expected);
}

/// The client-visible cursor contract: a dropped input between two
/// published facts consumes a sequence number without ever publishing, so
/// the published stream shows a GAP, never a re-numbering.
#[test]
fn published_sequences_are_non_contiguous_when_a_drop_happens_between_them() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope_a = OperationScope::root_only(OperationId::new());
    let scope_b = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    // Consumes a sequence number but never publishes.
    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope_b)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    engine.drain();
    let sequences: Vec<u64> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(
        sequences,
        vec![1, 3],
        "the dropped progress at sequence 1 leaves a gap, not a re-numbering"
    );
}

/// The state lane's FIFO-of-keys is bounded at
/// `STATE_TRANSITION_LANE_DEPTH`: a new distinct key past the ceiling is
/// rejected without evicting an accepted state.
#[test]
fn state_lane_rejects_a_new_key_past_the_depth_bound() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..STATE_TRANSITION_LANE_DEPTH {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted,
            "every key within the bound is retained"
        );
    }
    let rejected_scope = OperationScope::root_only(OperationId::new());
    assert_eq!(
        engine
            .unreserved_ingress(rejected_scope)
            .try_submit(state_transition_fact()),
        SubmitOutcome::RejectedCapacity
    );
    assert_eq!(engine.health().snapshot().state_transition_rejected, 1);
    assert!(engine.health().snapshot().state_degraded);
    assert!(engine.health().snapshot().rebuild_required);

    let report = engine.drain();
    assert_eq!(
        report.applied, STATE_TRANSITION_LANE_DEPTH,
        "exactly the bound's worth of accepted keys survive to drain"
    );
    assert_eq!(
        engine.replay().snapshot().len(),
        STATE_TRANSITION_LANE_DEPTH
    );
}
