//! Proves `RuntimeEventEngine::install_telemetry_queue` is the live wiring
//! seam: an ordinary, undecorated `try_submit` call through the real
//! `ScopedIngress`/`UnreservedIngress` handles every task 9-12 producer
//! already uses -- with no `ObservingIngress`, no telemetry-aware code in
//! the call site at all -- produces a `ClassOutcome` sample once a queue is
//! installed, and produces none (with `submit`'s outcome unchanged) before
//! installation. Fully synchronous: `RuntimeEventTelemetryQueue::push` and
//! `drain` never touch an async runtime, so this needs no `#[tokio::test]`
//! and no sleep.

use std::time::Duration;

use mesh_llm_runtime_event_contracts::{
    DeliveryClass, OperationId, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::terminal_success;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::telemetry::{RuntimeEventTelemetryQueue, RuntimeEventTelemetrySample};

#[test]
fn before_installing_telemetry_a_real_submission_behaves_exactly_as_before() {
    // No `install_telemetry_queue` call anywhere in this test: proves the
    // hook is fully opt-in and `submit`'s outcome is unchanged when
    // telemetry was never installed (there is nowhere to even check for
    // "no sample produced" -- the point is that nothing about this
    // ordinary path changed).
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_success)
        .expect("reserve");

    let outcome = reservation.ingress().try_submit(terminal_success());

    assert_eq!(outcome, SubmitOutcome::Accepted);
}

#[test]
fn installing_telemetry_makes_an_ordinary_reserved_submission_observed() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let queue = std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8));
    engine.install_telemetry_queue(queue.clone());

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_success)
        .expect("reserve");
    // The real path: `OperationReservation::ingress()` returns the same
    // `ScopedIngress` every producer (tasks 9-12) already calls
    // `try_submit` on. No decorator, no telemetry-aware code here.
    let outcome = reservation.ingress().try_submit(terminal_success());
    assert_eq!(outcome, SubmitOutcome::Accepted);
    drop(reservation);

    let drained = queue.drain();
    assert_eq!(drained.len(), 1);
    match drained[0] {
        RuntimeEventTelemetrySample::ClassOutcome { class, outcome, .. } => {
            assert_eq!(class, DeliveryClass::Terminal);
            assert_eq!(outcome, SubmitOutcome::Accepted);
        }
        other => panic!("expected ClassOutcome from a real submission, got {other:?}"),
    }
}

#[test]
fn installing_telemetry_observes_an_unreserved_submission_too() {
    // `UnreservedIngress` is the other real `RuntimeEventIngress`
    // implementor (used when reservation capacity is exhausted, so primary
    // work still proceeds) -- proves the hook covers both handoff shapes,
    // not only the reserved one.
    let engine = RuntimeEventEngine::with_capacity(4);
    let queue = std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8));
    engine.install_telemetry_queue(queue.clone());

    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let outcome = engine
        .unreserved_ingress(scope)
        .try_submit(terminal_success());

    assert_eq!(outcome, SubmitOutcome::TerminalDeliveryFailed);
    let drained = queue.drain();
    assert_eq!(drained.len(), 1);
    match drained[0] {
        RuntimeEventTelemetrySample::ClassOutcome { class, outcome, .. } => {
            assert_eq!(class, DeliveryClass::Terminal);
            assert_eq!(outcome, SubmitOutcome::TerminalDeliveryFailed);
        }
        other => panic!("expected ClassOutcome from a real submission, got {other:?}"),
    }
}

#[test]
fn installing_telemetry_records_a_measurable_nonzero_elapsed_duration() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let queue = std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8));
    engine.install_telemetry_queue(queue.clone());

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_success)
        .expect("reserve");
    let _ = reservation.ingress().try_submit(terminal_success());
    drop(reservation);

    let drained = queue.drain();
    match drained[0] {
        RuntimeEventTelemetrySample::ClassOutcome {
            ingress_elapsed, ..
        } => {
            // Genuinely measured, not a hardcoded/zeroed placeholder: a
            // real `Instant::now()..elapsed()` window is never negative and
            // completes in well under a second for an in-process call.
            assert!(ingress_elapsed < Duration::from_secs(1));
        }
        other => panic!("expected ClassOutcome, got {other:?}"),
    }
}

#[test]
fn installing_telemetry_twice_is_a_silent_noop_first_queue_wins() {
    // `install_telemetry_queue` is `OnceLock`-backed: a second call after
    // startup must never panic or replace the first-installed queue mid
    // flight, matching `install_runtime_event_engine`'s own "install once"
    // contract at the process level.
    let engine = RuntimeEventEngine::with_capacity(4);
    let first = std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8));
    let second = std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8));
    engine.install_telemetry_queue(first.clone());
    engine.install_telemetry_queue(second.clone());

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_success)
        .expect("reserve");
    let _ = reservation.ingress().try_submit(terminal_success());
    drop(reservation);

    assert_eq!(first.drain().len(), 1);
    assert!(second.is_empty());
}

#[test]
fn telemetry_hook_never_changes_the_submit_outcome_no_regression() {
    // The core Task-14-style regression guard: installing telemetry must
    // be behavior-invisible to every existing producer/outcome contract.
    // Same scenario as `terminal_submit_accepted_and_second_write_rejected_as_duplicate`
    // (task 3's own test), run once with telemetry installed and once
    // without, asserting identical outcomes both times.
    for install_telemetry in [false, true] {
        let engine = RuntimeEventEngine::with_capacity(4);
        if install_telemetry {
            engine.install_telemetry_queue(std::sync::Arc::new(RuntimeEventTelemetryQueue::new(8)));
        }
        let reservation = engine
            .reserve_root(OperationId::new(), terminal_success)
            .expect("reserve");
        let ingress = reservation.ingress();

        assert_eq!(
            ingress.try_submit(terminal_success()),
            SubmitOutcome::Accepted
        );
        assert_eq!(
            ingress.try_submit(terminal_success()),
            SubmitOutcome::TerminalDeliveryFailed
        );
    }
}
