//! Proves the presentation subscriber is actually attached to a live
//! engine, synchronously and with no scheduler race, and that a fact the
//! engine-owned driver (`runtime_events::driver`, spawned right alongside
//! this subscriber in `run_auto.rs`) applies and publishes reaches the
//! sink through the subscriber's own subscription -- with no drain() call
//! anywhere in this subscriber or this test: draining is exclusively the
//! driver's job as of task 3.

use std::sync::Arc;
use std::time::Duration;

use mesh_llm_runtime_event_contracts::{
    FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeEventIngress,
    RuntimeFact, SubmitOutcome,
};

use super::super::subscriber::{
    PresentationSink, attach, drive_presentation_subscriber, spawn_presentation_subscriber,
};
use super::{RecordingSink, terminal_fact};
use crate::runtime_events::engine::RuntimeEventEngine;

#[tokio::test]
async fn spawn_presentation_subscriber_registers_a_live_subscription_before_returning() {
    let engine = RuntimeEventEngine::new();
    assert_eq!(engine.subscribers().active_count(), 0);

    let handle = spawn_presentation_subscriber(&engine).expect("attach succeeds");

    assert_eq!(
        engine.subscribers().active_count(),
        1,
        "spawn_presentation_subscriber must register its subscription on the engine \
         synchronously, before it returns -- with no scheduler race for a caller to observe"
    );

    handle.abort();
}

#[tokio::test(start_paused = true)]
async fn a_fact_the_engine_owned_driver_applies_reaches_the_sink_via_the_subscription() {
    let engine = RuntimeEventEngine::new();
    let sink = Arc::new(RecordingSink::default());
    let subscription = attach(&engine).expect("attach");

    // Submit a terminal fact and spawn the SAME `runtime_events::driver`
    // task `run_auto.rs` spawns right alongside this subscriber -- proving
    // the subscriber needs nothing of its own to observe it, only the
    // driver applying and publishing it upstream of this subscription.
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reservation");
    let outcome = reservation.ingress().try_submit(terminal_fact());
    assert!(matches!(
        outcome,
        mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
    ));

    let driver = crate::runtime_events::driver::spawn_engine_driver(Arc::clone(&engine));
    let drive_engine = Arc::clone(&engine);
    let drive_sink: Arc<dyn super::super::subscriber::PresentationSink> = sink.clone();
    let drive = tokio::spawn(drive_presentation_subscriber(
        subscription,
        drive_engine,
        drive_sink,
    ));

    tokio::time::advance(Duration::from_millis(50)).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    drive.abort();
    driver.abort();
    let emitted = sink.drain();
    assert!(
        !emitted.is_empty(),
        "the engine-owned driver must apply and publish a queued terminal fact so the \
         presentation subscriber's own subscription receives and routes it"
    );
}

fn is_health_event(event: &mesh_llm_events::OutputEvent) -> bool {
    matches!(
        event,
        mesh_llm_events::OutputEvent::Info { context, .. }
            if context.as_deref() == Some("event_system_health")
    )
}

fn distinct_state_transition_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

/// Task 8-fix E5 (`.omo/plans/event-system-fixes.md`): `drive_presentation_subscriber`'s
/// recv arm calls `maybe_emit_health` after every routed frame (`subscriber.rs`'s
/// `Ok(frame) =>` branch) so a busy subscriber's health delivery is never
/// starved behind the 33ms render tick alone -- but nothing exercised that
/// call before this test: the only two named acceptance tests call
/// `flush_tick` directly (the render-tick path of a SINGLE consumer),
/// never entering this async loop's recv arm at all.
///
/// Isolating the recv arm from the tick arm needs two DIFFERENT mechanisms
/// for the two deliveries this test observes, because `tokio::time::interval`
/// documents that "the first tick completes immediately" -- independent of
/// virtual time, so `start_paused` alone cannot prevent it:
///   1. The FIRST health event is `tick.tick()`'s own unavoidable immediate
///      first fire hitting a fresh `HealthDeliveryGate` (which -- per
///      `HealthDeliveryGate::new`'s documented "never delivered yet"
///      contract -- always delivers on its first eligible check regardless
///      of source). This happens once, before this test submits anything,
///      and is asserted here explicitly so it is never confused with what
///      this test actually proves.
///   2. This test never calls `tokio::time::advance` anywhere below, so the
///      interval's SECOND tick can never become due in tokio's paused
///      virtual clock -- the render-tick branch is permanently unready for
///      the rest of this test. But `HealthDeliveryGate::should_deliver`'s
///      own 1s cadence reads `std::time::Instant::now()` (real wall-clock
///      time, never virtualized by `tokio::time::pause`), so a SECOND
///      delivery is only even ELIGIBLE once real time has actually passed
///      it -- `std::thread::sleep` below blocks this test's single OS
///      thread (the default `current_thread` test runtime) to advance real
///      time without touching tokio's still-paused virtual clock, so the
///      render tick's interval stays exactly where step 1 left it. The
///      second health event this test asserts can therefore ONLY come from
///      the recv arm. Deleting that one call (mutation M4,
///      `.omo/evidence/event-system-fixes/task-08/mutation-proof.txt`)
///      makes this test fail (0 health events after priming, not 1).
#[tokio::test(start_paused = true)]
async fn drive_presentation_subscriber_recv_arm_delivers_health_during_a_steady_frame_stream() {
    let engine = RuntimeEventEngine::with_capacity(1);
    let sink = Arc::new(RecordingSink::default());
    let subscription = attach(&engine).expect("attach");
    let drive_sink: Arc<dyn PresentationSink> = sink.clone();
    let drive = tokio::spawn(drive_presentation_subscriber(
        subscription,
        Arc::clone(&engine),
        drive_sink,
    ));

    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    let primed = sink.drain();
    assert_eq!(
        primed.iter().filter(|event| is_health_event(event)).count(),
        1,
        "the render tick's own immediate first fire must have delivered exactly \
         one health snapshot before this test submitted anything; got: {primed:?}"
    );

    std::thread::sleep(Duration::from_millis(1_100));

    // Bump `health_version` without publishing any frame: the one
    // reservation slot is already exhausted, so a second `reserve_root`
    // call fails and bumps `EngineHealth::bump_reservation_exhausted`
    // (`engine/mod.rs::reserve_scope`) -- there is nothing here for the
    // recv arm's `route_fact` to forward, only a real health change for
    // `maybe_emit_health` to notice on the NEXT delivered frame.
    let held = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("first reservation succeeds");
    assert!(
        engine
            .reserve_root(OperationId::new(), terminal_fact)
            .is_none(),
        "capacity 1 is already held"
    );

    // A steady stream of 20 distinct frames through the REAL engine
    // submit+drain path -- each one is a separate `(scope, kind)` state-
    // lane key (fresh `OperationId` per call), so none coalesces with
    // another, and each publishes its own frame for the recv arm to route.
    for _ in 0..20 {
        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(distinct_state_transition_fact());
        assert_eq!(outcome, SubmitOutcome::Accepted);
        engine.drain();
    }
    for _ in 0..40 {
        tokio::task::yield_now().await;
    }

    drive.abort();
    drop(held);
    let emitted = sink.drain();
    let health_events = emitted
        .iter()
        .filter(|event| is_health_event(event))
        .count();
    let fact_events = emitted.len() - health_events;

    assert!(
        fact_events >= 15,
        "the steady stream of frames must actually reach the sink via the recv arm; \
         got {fact_events} fact events out of {} total: {emitted:?}",
        emitted.len()
    );
    assert_eq!(
        health_events, 1,
        "the recv arm's own maybe_emit_health call must deliver exactly one health \
         snapshot after real time passed the 1s cadence and a real counter changed; \
         with the render tick's interval permanently stuck at its already-consumed \
         first tick, this second delivery can only come from the recv arm; \
         emitted: {emitted:?}"
    );
}
