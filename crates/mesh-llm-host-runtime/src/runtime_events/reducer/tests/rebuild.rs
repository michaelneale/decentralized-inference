use mesh_llm_runtime_event_contracts::Outcome;

use super::fixtures::{input, progress_fact, scope, synthesized_terminal_fact, terminal_fact};
use crate::runtime_events::reducer::{
    ReduceOutcome, ReducerSnapshot, apply,
    rebuild::{self, RebuildError, RebuildOutcome},
};

#[test]
fn crash_synthesis_reduces_into_degraded_but_settled_preserving_last_valid() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(42)))
    else {
        panic!("progress must apply first");
    };

    let mut synthesized = input(scope, 1, synthesized_terminal_fact());
    synthesized.synthesized = true;
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, synthesized) else {
        panic!("synthesized terminal must apply");
    };

    let state = snapshot.operation(scope).expect("state");
    assert!(state.settled);
    assert!(state.degraded);
    assert_eq!(
        state.last_progress_current,
        Some(42),
        "last-valid progress must survive a synthesized-terminal settle"
    );
}

#[test]
fn rebuild_success_bumps_generation_and_degrades_unsettled_operations() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(10)))
    else {
        panic!("progress must apply");
    };
    assert_eq!(snapshot.rebuild_generation, 0);

    let RebuildOutcome::Rebuilt(rebuilt) = rebuild::rebuild(&snapshot, 1) else {
        panic!("rebuild to a higher generation must succeed");
    };
    assert_eq!(rebuilt.rebuild_generation, 1);
    let state = rebuilt.operation(scope).expect("state survives rebuild");
    assert!(
        state.degraded,
        "an operation that never settled before the crash is marked degraded"
    );
    assert_eq!(
        state.last_progress_current,
        Some(10),
        "last-valid values are preserved across rebuild, not discarded"
    );

    // Rebuild continues the sequence: applying the next ingress sequence
    // after the rebuild still succeeds against the rebuilt snapshot.
    let continued = apply(&rebuilt, input(scope, 1, terminal_fact(Outcome::Success)));
    assert!(matches!(continued, ReduceOutcome::Applied(_)));
}

#[test]
fn rebuild_failure_on_non_monotonic_generation_leaves_state_coherent() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)))
    else {
        panic!("terminal must apply");
    };
    let RebuildOutcome::Rebuilt(snapshot) = rebuild::rebuild(&snapshot, 1) else {
        panic!("first rebuild must succeed");
    };

    let failure = rebuild::rebuild(&snapshot, 1);
    assert!(matches!(
        failure,
        RebuildOutcome::Failed(RebuildError::NonMonotonicGeneration)
    ));
    assert_eq!(
        snapshot.rebuild_generation, 1,
        "a failed rebuild must not partially mutate the generation"
    );
    assert!(
        snapshot.operation(scope).expect("state").settled,
        "a failed rebuild must leave prior operation state fully intact"
    );
}
