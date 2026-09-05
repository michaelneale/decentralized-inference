use mesh_llm_runtime_event_contracts::Outcome;

use super::fixtures::{input, progress_fact, scope, terminal_fact};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, RejectReason, apply};

#[test]
fn terminal_overtaking_wins_over_later_in_flight_progress() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)))
    else {
        panic!("terminal must apply and settle the operation");
    };
    assert!(snapshot.operation(scope).expect("state").settled);

    let later_progress = apply(&snapshot, input(scope, 1, progress_fact(50)));
    assert!(matches!(
        later_progress,
        ReduceOutcome::Rejected(RejectReason::OperationSettled)
    ));
}

#[test]
fn contradictory_terminal_is_rejected() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)))
    else {
        panic!("first terminal must apply");
    };

    let second_terminal = apply(&snapshot, input(scope, 1, terminal_fact(Outcome::Failure)));
    assert!(matches!(
        second_terminal,
        ReduceOutcome::Rejected(RejectReason::ContradictoryTerminal)
    ));
    assert_eq!(
        snapshot.operation(scope).expect("state").last_outcome,
        Some(Outcome::Success),
        "a rejected contradictory terminal must not overwrite the settled outcome"
    );
}

#[test]
fn a_rejected_input_produces_no_new_snapshot_state() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let before = snapshot.operation_count();

    // A stale-progress input against an operation with no prior state is
    // itself the first fact, so use a settled operation instead: the
    // simplest genuinely-invalid input is a non-terminal fact against an
    // already-settled scope.
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)))
    else {
        panic!("seed terminal must apply");
    };
    let settled_operation_count = snapshot.operation_count();

    let rejection = apply(&snapshot, input(scope, 1, progress_fact(10)));
    assert!(matches!(rejection, ReduceOutcome::Rejected(_)));
    assert_eq!(
        snapshot.operation_count(),
        settled_operation_count,
        "the reducer's own snapshot Arc is never mutated by a rejection"
    );
    assert_eq!(before, 0);
}
