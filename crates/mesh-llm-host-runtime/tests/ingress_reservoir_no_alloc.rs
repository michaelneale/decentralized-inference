//! Task 13 (`.omo/plans/event-system-fixes.md`, "Must NOT: allocate on the
//! submit path") -- proves `RuntimeEventEngine::submit`'s ingress-latency
//! recording is allocation-free, via a thread-local counting
//! `#[global_allocator]`. Safe to install here ONLY because integration
//! tests under `tests/*.rs` each compile as their OWN separate binary
//! crate: this allocator applies to this one binary alone, never to the
//! crate's `--lib` unit-test binary or any other integration test.
//!
//! ## Task 13 alloc-fix (post-review strengthening)
//!
//! An adversarial review of the original version of this file found that
//! it asserted only `NET_ALLOCS` -- outstanding allocations, incremented
//! by `alloc`/`realloc` and decremented by `dealloc` -- returning to its
//! starting value. That is a strictly WEAKER property than "no heap
//! allocation": a transient allocation that is both allocated AND freed
//! inside the very same measured `try_submit` call (a stray `format!()`,
//! `to_string()`, or scratch `Vec`) nets to zero and sails straight
//! through a net-only assertion. The review proved this empirically by
//! planting exactly such a `format!()` on the submit path and observing
//! every test in this file stay green -- see
//! `.omo/evidence/event-system-fixes/task-13/alloc-baseline.txt`.
//!
//! `TOTAL_ALLOC_CALLS` below fixes this: a monotonic counter incremented
//! on every `alloc`/`realloc` call and NEVER decremented, so a
//! same-window allocate-then-free still moves it even though it nets to
//! zero on `NET_ALLOCS`. It is the counter that actually discharges "no
//! heap allocation happened" for a measured window. `NET_ALLOCS` is kept
//! alongside it -- it is still a useful, independent signal (a real
//! *leak* across many iterations shows up as steady growth there, which
//! is not quite what a same-window total-calls count highlights) -- but
//! every test below now asserts BOTH. `alloc_zeroed` needs no separate
//! counting: `GlobalAlloc`'s default `alloc_zeroed` implementation calls
//! back into this allocator's own (counted) `alloc`, so any zeroed
//! allocation is already covered by the `alloc` override below without
//! duplicating that logic.
//!
//! ## Delivery-class coverage
//!
//! The original file only ever exercised the `Terminal` delivery class,
//! and its many-call measured loops only ever hit the DUPLICATE-REJECTION
//! path (`SubmitOutcome::TerminalDeliveryFailed`) -- the one successful,
//! `Accepted` write on each reservation was the excluded warm-up call.
//! This file now additionally proves a genuinely DELIVERED (never
//! rejected, never dropped) submission is allocation-free for all four
//! `DeliveryClass` values: `Terminal`, `StateTransition`, `Progress`, and
//! `Diagnostic`. Each new test documents exactly how its measured window
//! is scoped; see also
//! `.omo/evidence/event-system-fixes/task-13/alloc-coverage-note.txt`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use mesh_llm_host_runtime::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FamilyFact, ModelLoadingEventKind, NativeRuntimeEventKind, OperationId,
    OperationScope, RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

thread_local! {
    /// NET outstanding allocations on this thread: `alloc`/`realloc`
    /// increment, `dealloc` decrements. Returning to its starting value
    /// proves nothing LEAKED across a measured window -- it says nothing
    /// about a transient allocate-then-free WITHIN that window. Kept
    /// alongside `TOTAL_ALLOC_CALLS` -- see the module doc comment.
    static NET_ALLOCS: Cell<i64> = const { Cell::new(0) };
    /// TOTAL allocation CALLS on this thread: incremented by `alloc` and
    /// `realloc`, NEVER decremented. This is the counter that actually
    /// proves "no heap allocation happened" for a measured window -- a
    /// same-window allocate-then-free still moves it even though it nets
    /// to zero on `NET_ALLOCS`. See the module doc comment.
    static TOTAL_ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NET_ALLOCS.with(|count| count.set(count.get() - 1));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc that moves counts as one alloc; this file cares about
        // BOTH whether the net count of outstanding allocations moves
        // during the measured window (a moved reallocation) AND whether
        // any allocation call happened at all (the total-calls counter),
        // so counting it toward both is correct for either measurement.
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    // `alloc_zeroed` is intentionally NOT overridden: `GlobalAlloc`'s
    // default implementation calls back into `Self::alloc` (above) and
    // then zeroes the result, so it is already counted on both counters
    // without duplicating that logic here.
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn terminal_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

/// A `StateTransition`-class fact from the SAME family as `terminal_fact`
/// (`NativeRuntimeEventKind`), cross-checked against
/// `crates/mesh-llm-runtime-event-contracts/src/delivery/lifecycle.rs`.
fn state_transition_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

/// A `Progress`-class fact (`delivery/lifecycle.rs`).
fn progress_fact() -> RuntimeFact {
    RuntimeFact::ModelLoading(FamilyFact::new(ModelLoadingEventKind::ModelLoadProgress))
}

/// A `Diagnostic`-class fact (`delivery/execution.rs`).
fn diagnostic_fact() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(DiagnosticEventKind::WarningRaised))
}

fn net_allocs() -> i64 {
    NET_ALLOCS.with(Cell::get)
}

fn total_alloc_calls() -> u64 {
    TOTAL_ALLOC_CALLS.with(Cell::get)
}

/// Drives ONE `try_submit(fact)` call and asserts it (a) reports the
/// expected DELIVERED outcome -- never a rejection or a drop, so this
/// helper actually measures a success path, not a fast-reject shortcut --
/// and (b) costs ZERO total allocation calls. The counting window opens
/// at `before` and closes at `after`, covering ONLY this one call: every
/// caller below builds its engine, its reservation, and any lane warm-up
/// strictly BEFORE calling this helper, so nothing outside `try_submit`
/// itself can be mistaken for what it costs.
fn assert_try_submit_is_alloc_free(
    ingress: &dyn RuntimeEventIngress,
    fact: RuntimeFact,
    expected_outcome: SubmitOutcome,
    class_name: &str,
) {
    let before = total_alloc_calls();
    let outcome = ingress.try_submit(fact);
    let after = total_alloc_calls();

    assert_eq!(
        outcome, expected_outcome,
        "{class_name}: expected a delivered ({expected_outcome:?}) outcome, got \
         {outcome:?} -- a wrong outcome here would mean this test isn't \
         measuring the success path it claims to"
    );
    assert_eq!(
        after, before,
        "{class_name}: a delivered ({expected_outcome:?}) submission must perform \
         zero TOTAL heap allocation calls, not merely net-zero outstanding \
         (before={before}, after={after})"
    );
}

/// Warms the state-transition lane's backing `VecDeque`/`HashMap` (owned
/// by `engine/lanes.rs::StateLane` -- out of Task 13's ownership, which
/// added the ingress-latency reservoir, not this lane) so their
/// first-growth allocations happen BEFORE any measured window opens,
/// mirroring this file's original wake-list warm-up precedent for
/// `Terminal`. Every call here submits a genuinely distinct `(scope,
/// kind)` key (a fresh `OperationId` each time, via `unreserved_ingress`
/// so it never touches the reservation table), so every warm-up
/// submission is itself `Accepted` -- the state-transition lane coalesces
/// a REPEAT key instead of growing its backing collections further, so a
/// repeat key would not actually warm anything up. This leaves the lane
/// with capacity headroom well past the ONE additional key the real test
/// below adds inside its measured window.
fn warm_up_state_transition_lane(engine: &Arc<RuntimeEventEngine>, count: usize) {
    for _ in 0..count {
        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(state_transition_fact());
        assert_eq!(
            outcome,
            SubmitOutcome::Accepted,
            "warm-up state-transition submission (fresh scope) must itself be accepted"
        );
    }
}

/// Warms the diagnostic lane's backing `VecDeque` (owned by
/// `engine/lanes.rs::DiagnosticLane`) for the same reason as
/// `warm_up_state_transition_lane`. Unlike the state-transition lane,
/// diagnostics never coalesce by key -- every submission below the
/// 2,048-entry depth is `Accepted` regardless of key reuse -- so this
/// reuses one scope for every warm-up call.
fn warm_up_diagnostic_lane(engine: &Arc<RuntimeEventEngine>, count: usize) {
    let scope = OperationScope::root_only(OperationId::new());
    for _ in 0..count {
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(diagnostic_fact());
        assert_eq!(
            outcome,
            SubmitOutcome::Accepted,
            "warm-up diagnostic submission must itself be accepted (no coalescing in this lane)"
        );
    }
}

#[test]
fn submit_records_ingress_latency_with_no_heap_allocation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();

    // Warm up: the FIRST `try_submit` on this handle is the only one that
    // can succeed (terminal slots are write-once) and is where any
    // one-time lazy setup along the call chain (e.g. the wake list's
    // `VecDeque` growing from empty) would show up. Excluding it from the
    // measured window isolates what THIS task added -- the reservoir
    // write -- from pre-existing allocation behavior elsewhere in
    // `submit`, which is out of this task's ownership and not what this
    // test is about.
    let _ = ingress.try_submit(terminal_fact());

    let before_net = net_allocs();
    let before_total = total_alloc_calls();
    // Every call after the first is a duplicate-terminal rejection
    // (`TerminalDeliveryFailed`): it still runs the full `submit` body,
    // including the reservoir's `record` call, without ever touching the
    // wake list or reservation table's write-once slot again.
    for _ in 0..1_000 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after_net = net_allocs();
    let after_total = total_alloc_calls();

    assert_eq!(
        after_net, before_net,
        "submit's ingress-latency recording must add no net heap allocation \
         across 1,000 calls (before={before_net}, after={after_net})"
    );
    // The TOTAL-calls assertion (task 13 alloc-fix): the strictly
    // stronger property. A net-only assertion cannot distinguish "zero
    // allocation" from "an equal number of same-window allocate/free
    // pairs"; this one can, because it never decrements.
    assert_eq!(
        after_total, before_total,
        "submit's ingress-latency recording must perform zero TOTAL heap \
         allocation calls across 1,000 calls, not merely net-zero outstanding \
         (before={before_total}, after={after_total})"
    );
}

#[test]
fn submit_crosses_the_reservoir_milestone_with_no_heap_allocation() {
    // A second, independent proof at a scale that actually exercises the
    // 100-sample health-version-bump milestone (`IngressLatencyReservoir::record`'s
    // return value), not just the reservoir's plain ring write.
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();
    let _ = ingress.try_submit(terminal_fact());

    let before_net = net_allocs();
    let before_total = total_alloc_calls();
    for _ in 0..250 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after_net = net_allocs();
    let after_total = total_alloc_calls();

    assert_eq!(
        after_net, before_net,
        "crossing the 100-sample milestone (twice, over 250 calls) must still add \
         no net heap allocation (before={before_net}, after={after_net})"
    );
    assert_eq!(
        after_total, before_total,
        "crossing the 100-sample milestone (twice, over 250 calls) must still \
         perform zero TOTAL heap allocation calls, not merely net-zero \
         outstanding (before={before_total}, after={after_total})"
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `Terminal`. Unlike the two
/// tests above (which measure only DUPLICATE-REJECTED submissions after
/// their one excluded warm-up write), this measures a genuinely fresh,
/// `Accepted` terminal write -- a SECOND reservation's first-and-only
/// submission, on an engine whose wake list was already warmed by an
/// unrelated first reservation.
#[test]
fn submit_delivers_an_accepted_terminal_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    // Warm-up (excluded from the window): a THROWAWAY reservation's own
    // terminal write, purely to grow the wake list's `VecDeque` from
    // empty -- see the module doc comment and this file's original
    // precedent.
    let warm_up = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let warm_outcome = warm_up.ingress().try_submit(terminal_fact());
    assert_eq!(warm_outcome, SubmitOutcome::Accepted);

    // The MEASURED reservation: a fresh scope whose terminal has never
    // been written, so `try_submit` below is a genuinely delivered
    // (`Accepted`) submission, not a rejected duplicate.
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        terminal_fact(),
        SubmitOutcome::Accepted,
        "Terminal",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `StateTransition`. The
/// measured call is the lane's 201st distinct `(scope, kind)` key
/// (`warm_up_state_transition_lane` above installs 200 first), so it is
/// `Accepted` -- never `Coalesced` -- inside the measured window.
#[test]
fn submit_delivers_an_accepted_state_transition_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    warm_up_state_transition_lane(&engine, 200);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        state_transition_fact(),
        SubmitOutcome::Accepted,
        "StateTransition",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `Progress`. `Coalesced` is
/// deliberately the expected outcome here, not `Accepted`:
/// `lanes::submit_progress` can only ever return `Coalesced` (a live
/// handle) or `DroppedProgress` (no handle) -- `SubmitOutcome::Accepted`
/// is not a reachable outcome for a `Progress`-class fact at all -- so
/// `Coalesced` here IS progress's own delivered, never-dropped success
/// case, not a weaker stand-in for it.
///
/// `coalesce_progress` itself only ever writes into the reservation
/// table's OWN slot (`Option<(RuntimeFact, u64)>`), fully preallocated by
/// `ReservationTable::new` at construction -- there is no separate
/// growable collection behind THIS lane the way there is for `Terminal`
/// (wake list), `StateTransition` (`VecDeque` + `HashMap`), or
/// `Diagnostic` (`VecDeque`). But `submit_progress` -- like every lane --
/// still calls the SHARED `engine.wake().next_ingress_sequence()`, which
/// locks the wake list's own `Mutex<Inner>`; on this platform a fresh
/// `std::sync::Mutex`'s FIRST-EVER lock call costs exactly one allocation
/// (confirmed directly with a standalone `std::sync::Mutex` probe -- see
/// `.omo/evidence/event-system-fixes/task-13/alloc-coverage-note.txt`),
/// and every OTHER class's test above happens to warm that same mutex
/// incidentally through its own lane warm-up. Progress has no other
/// reason to submit more than once, so it needs its OWN explicit one-call
/// warm-up for exactly this shared, pre-existing, one-time cost -- never
/// Task 13's ingress-latency reservoir, and never a per-call cost on the
/// steady-state submit path.
#[test]
fn submit_delivers_a_coalesced_progress_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let warm_up = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let warm_outcome = warm_up.ingress().try_submit(progress_fact());
    assert_eq!(warm_outcome, SubmitOutcome::Coalesced);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        progress_fact(),
        SubmitOutcome::Coalesced,
        "Progress",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `Diagnostic`. The measured
/// call is the lane's 201st submission (`warm_up_diagnostic_lane` above
/// installs 200 first); diagnostics never coalesce, so it is `Accepted`
/// regardless of key reuse.
#[test]
fn submit_delivers_an_accepted_diagnostic_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    warm_up_diagnostic_lane(&engine, 200);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        diagnostic_fact(),
        SubmitOutcome::Accepted,
        "Diagnostic",
    );
}
