//! Task 6-fix R1 (`.omo/plans/event-system-fixes.md`): pure reducer-level
//! unit coverage for `OperationState::ever_reserved`'s sticky OR-latch and
//! `ReducerSnapshot`'s `unreserved_order` bounded LRU, isolated from the
//! engine so each case is a direct, deterministic `apply()` sequence.
//!
//! Every fact used here is deliberately NON-SETTLING (`progress_fact`, no
//! `outcome`) so the pre-existing settled-only capacity backstop
//! (`evict_settled_over_capacity`, unchanged by this task) never finds a
//! candidate and cannot interfere -- these tests isolate the NEW
//! `unreserved_order` mechanism specifically, matching
//! `engine::tests::eviction`'s identical use of `state_transition_fact()`
//! (no outcome) for the same reason.

use super::fixtures::{input, input_unreserved, progress_fact, scope};
use crate::runtime_events::config::UNRESERVED_OPERATION_BOUND;
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

/// A scope whose only-ever fact arrives unreserved is tracked, but the
/// OLDEST never-reserved entry is evicted once distinct never-reserved
/// scopes exceed `UNRESERVED_OPERATION_BOUND`.
#[test]
fn unreserved_only_scopes_are_capped_at_the_frozen_bound() {
    let mut snapshot = ReducerSnapshot::empty();
    let mut sequence = 0u64;
    let first_unreserved = scope();

    let ReduceOutcome::Applied(next) = apply(
        &snapshot,
        input_unreserved(first_unreserved, sequence, progress_fact(0)),
    ) else {
        panic!("first unreserved scope must apply");
    };
    snapshot = next;
    sequence += 1;

    for _ in 0..UNRESERVED_OPERATION_BOUND {
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input_unreserved(scope(), sequence, progress_fact(0)),
        ) else {
            panic!("every distinct never-reserved scope must apply");
        };
        snapshot = next;
        sequence += 1;
    }

    assert!(
        snapshot.operation_count() <= UNRESERVED_OPERATION_BOUND,
        "never-reserved scopes alone must never exceed UNRESERVED_OPERATION_BOUND, got {}",
        snapshot.operation_count()
    );
    assert!(
        snapshot.operation(first_unreserved).is_none(),
        "the OLDEST never-reserved scope must be evicted once the bound is exceeded"
    );
}

/// A scope reserved from its very first apply is NEVER touched by the
/// unreserved-order LRU, however many never-reserved scopes churn around
/// it -- proves `ever_reserved: true` fully exempts a scope from R1's new
/// bound.
#[test]
fn a_reserved_scope_is_never_evicted_by_the_unreserved_bound() {
    let mut snapshot = ReducerSnapshot::empty();
    let mut sequence = 0u64;
    let reserved_scope = scope();

    let ReduceOutcome::Applied(next) =
        apply(&snapshot, input(reserved_scope, sequence, progress_fact(0)))
    else {
        panic!("reserved scope must apply");
    };
    snapshot = next;
    sequence += 1;

    for _ in 0..(UNRESERVED_OPERATION_BOUND * 2) {
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input_unreserved(scope(), sequence, progress_fact(0)),
        ) else {
            panic!("every distinct never-reserved scope must apply");
        };
        snapshot = next;
        sequence += 1;
    }

    assert!(
        snapshot.operation(reserved_scope).is_some(),
        "a scope that was EVER reserved must never be evicted by the unreserved bound"
    );
}

/// A scope whose FIRST fact is unreserved and whose SECOND fact is
/// reserved "graduates" out of the unreserved-order LRU: `ever_reserved`
/// latches sticky-true, and the scope is no longer subject to R1's bound
/// (mirrors `inference/skippy/runtime_events/mod.rs`'s real shape in
/// reverse -- production always reserves first, but the mechanism must be
/// correct regardless of arrival order).
#[test]
fn a_scope_that_later_becomes_reserved_graduates_out_of_the_unreserved_bound() {
    let mut snapshot = ReducerSnapshot::empty();
    let mut sequence = 0u64;
    let graduating = scope();

    let ReduceOutcome::Applied(next) = apply(
        &snapshot,
        input_unreserved(graduating, sequence, progress_fact(0)),
    ) else {
        panic!("first (unreserved) fact must apply");
    };
    snapshot = next;
    sequence += 1;
    assert!(
        !snapshot
            .operation(graduating)
            .expect("tracked")
            .ever_reserved,
        "must not yet be marked ever_reserved after only an unreserved fact"
    );

    // Second fact: reserved -- flips `ever_reserved` sticky-true. A
    // progress value that does not regress avoids a spurious
    // `StaleProgress` rejection.
    let ReduceOutcome::Applied(next) =
        apply(&snapshot, input(graduating, sequence, progress_fact(1)))
    else {
        panic!("second (reserved) fact must apply");
    };
    snapshot = next;
    sequence += 1;
    assert!(
        snapshot
            .operation(graduating)
            .expect("tracked")
            .ever_reserved,
        "ever_reserved must latch true once ANY fact for this scope was reserved"
    );

    // Flood past the unreserved bound with OTHER never-reserved scopes:
    // the graduated scope must survive because it is no longer tracked in
    // unreserved_order.
    for _ in 0..(UNRESERVED_OPERATION_BOUND * 2) {
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input_unreserved(scope(), sequence, progress_fact(0)),
        ) else {
            panic!("every distinct never-reserved scope must apply");
        };
        snapshot = next;
        sequence += 1;
    }

    assert!(
        snapshot.operation(graduating).is_some(),
        "a scope that graduated to ever_reserved=true must survive the unreserved-only flood"
    );
}
