//! Latest-value-per-operation progress coalescing, proven against the
//! frozen 33ms tick and under a flood of submissions ("tiny queue").

use std::time::{Duration, Instant};

use super::super::coalescer::ProgressCoalescer;
use super::{progress_fact, root_scope};

#[test]
fn flush_before_the_tick_interval_elapses_returns_nothing() {
    let coalescer = ProgressCoalescer::with_interval(Duration::from_millis(33));
    let start = Instant::now();
    let (scope, fact) = progress_fact(root_scope(), 10, Some(100));
    coalescer.submit(scope, fact);

    assert!(
        coalescer
            .flush_at(start + Duration::from_millis(10))
            .is_empty(),
        "a flush inside the tick window must not cross the channel boundary yet"
    );
}

#[test]
fn flush_after_the_tick_interval_elapses_returns_the_latest_value_once() {
    let coalescer = ProgressCoalescer::with_interval(Duration::from_millis(33));
    let scope = root_scope();
    let start = Instant::now();
    let (_, first) = progress_fact(scope, 10, Some(100));
    let (_, second) = progress_fact(scope, 90, Some(100));
    coalescer.submit(scope, first);
    coalescer.submit(scope, second.clone());

    let flushed = coalescer.flush_at(start + Duration::from_millis(40));
    assert_eq!(
        flushed.len(),
        1,
        "exactly one flushed value per operation, not one per submission"
    );
    assert_eq!(flushed[0].0, scope);
    assert_eq!(
        flushed[0].1, second,
        "the latest submitted value wins; the stale one must be discarded"
    );
}

#[test]
fn a_flood_of_progress_submissions_for_one_operation_never_exceeds_one_pending_entry() {
    let coalescer = ProgressCoalescer::with_interval(Duration::from_millis(33));
    let scope = root_scope();
    for step in 0..1_000u64 {
        let (_, fact) = progress_fact(scope, step, Some(1_000));
        coalescer.submit(scope, fact);
    }
    assert_eq!(
        coalescer.pending_len(),
        1,
        "a 1,000-update flood on one operation must coalesce to a single pending entry, \
         proving the channel boundary sees at most one progress update per operation per tick"
    );
}

#[test]
fn distinct_operations_each_get_their_own_pending_slot_bounded_by_operation_count() {
    let coalescer = ProgressCoalescer::with_interval(Duration::from_millis(33));
    for _ in 0..5 {
        let scope = root_scope();
        for step in 0..50u64 {
            let (_, fact) = progress_fact(scope, step, Some(50));
            coalescer.submit(scope, fact);
        }
    }
    assert_eq!(
        coalescer.pending_len(),
        5,
        "the tiny queue holds one entry per LIVE operation, never one per submission"
    );
}

#[test]
fn flushing_clears_the_pending_set_so_a_quiet_operation_produces_nothing_on_the_next_tick() {
    let coalescer = ProgressCoalescer::with_interval(Duration::from_millis(33));
    let scope = root_scope();
    let (_, fact) = progress_fact(scope, 5, Some(10));
    coalescer.submit(scope, fact);
    let start = Instant::now();
    let first_flush = coalescer.flush_at(start + Duration::from_millis(40));
    assert_eq!(first_flush.len(), 1);

    let second_flush = coalescer.flush_at(start + Duration::from_millis(80));
    assert!(
        second_flush.is_empty(),
        "an operation that submitted nothing since the last flush must not repeat stale progress"
    );
}
