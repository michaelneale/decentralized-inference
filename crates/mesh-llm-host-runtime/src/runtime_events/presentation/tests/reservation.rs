//! Terminal and health events are reserved: forwarded immediately, every
//! single one, never dropped, coalesced, or starved behind a progress
//! flood on unrelated operations.

use std::sync::Arc;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationScope, RuntimeFact};

use super::super::coalescer::ProgressCoalescer;
use super::super::subscriber::{flush_tick, route_fact};
use super::{RecordingSink, progress_fact, root_scope, terminal_fact};
use crate::runtime_events::health::{EngineHealth, HealthDeliveryGate};
use crate::runtime_events::replay::ReplayFrame;

fn frame_for(scope: OperationScope, fact: RuntimeFact) -> ReplayFrame {
    ReplayFrame {
        sequence: EventSequence::new(1),
        rebuild_generation: 0,
        scope,
        fact: Arc::new(fact),
        recorded_at: Instant::now(),
        // Task 9 (`.omo/plans/event-system-fixes.md`) added this field to
        // `ReplayFrame`; `route_fact`/`flush_tick` never read it, so an
        // empty placeholder is correct here. Mechanical fix only -- no
        // presentation test behavior changes (task 9 does not own
        // `presentation/*`).
        wire_bytes: Arc::from(Vec::new()),
    }
}

#[test]
fn a_terminal_frame_is_emitted_immediately_and_never_buffered_in_the_progress_coalescer() {
    let coalescer = ProgressCoalescer::new();
    let sink = RecordingSink::default();
    let frame = frame_for(root_scope(), terminal_fact());

    route_fact(&coalescer, &sink, &frame);

    assert_eq!(
        coalescer.pending_len(),
        0,
        "a terminal fact must never enter the progress coalescer"
    );
    let emitted = sink.drain();
    assert_eq!(
        emitted.len(),
        1,
        "the terminal fact reaches the sink on its own submission, with no tick required"
    );
}

#[test]
fn a_terminal_frame_is_forwarded_even_while_a_progress_flood_is_pending_for_other_operations() {
    let coalescer = ProgressCoalescer::new();
    let sink = RecordingSink::default();
    for _ in 0..500 {
        let (scope, fact) = progress_fact(root_scope(), 1, Some(2));
        coalescer.submit(scope, fact);
    }

    let terminal_frame = frame_for(root_scope(), terminal_fact());
    route_fact(&coalescer, &sink, &terminal_frame);

    let emitted = sink.drain();
    assert_eq!(
        emitted.len(),
        1,
        "a terminal event is never starved behind a pending progress flood on unrelated operations"
    );
}

fn is_health_event(event: &mesh_llm_events::OutputEvent) -> bool {
    matches!(
        event,
        mesh_llm_events::OutputEvent::Info { context, .. }
            if context.as_deref() == Some("event_system_health")
    )
}

#[test]
fn health_snapshot_is_forwarded_on_flush_without_touching_the_progress_coalescer() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let mut health_gate = HealthDeliveryGate::new();
    let sink = RecordingSink::default();
    health.bump_reservation_exhausted();

    let now = Instant::now();
    flush_tick(&coalescer, &health, || None, &mut health_gate, &sink, now);

    let emitted = sink.drain();
    assert!(
        emitted.iter().any(is_health_event),
        "the coalesced health snapshot must reach the sink on the same tick that flushes progress"
    );
    assert_eq!(
        coalescer.pending_len(),
        0,
        "flushing health must not leave anything behind in the progress coalescer"
    );
}

#[test]
fn health_snapshot_cadence_gate_is_never_bypassed_by_repeated_flush_ticks() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let mut health_gate = HealthDeliveryGate::new();
    let sink = RecordingSink::default();
    health.bump_reservation_exhausted();

    let now = Instant::now();
    flush_tick(&coalescer, &health, || None, &mut health_gate, &sink, now);
    flush_tick(&coalescer, &health, || None, &mut health_gate, &sink, now);

    let health_events = sink.drain().into_iter().filter(is_health_event).count();
    assert_eq!(
        health_events, 1,
        "the per-subscriber HealthDeliveryGate must not be bypassed by repeated flush ticks"
    );
}

// ─── Task 8: per-subscriber delivery, never a shared engine-global gate ─

#[test]
fn two_independent_gates_each_receive_their_own_health_delivery_after_one_bump() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let sink_a = RecordingSink::default();
    let sink_b = RecordingSink::default();
    let mut gate_a = HealthDeliveryGate::new();
    let mut gate_b = HealthDeliveryGate::new();
    health.bump_reservation_exhausted();

    let now = Instant::now();
    flush_tick(&coalescer, &health, || None, &mut gate_a, &sink_a, now);
    flush_tick(&coalescer, &health, || None, &mut gate_b, &sink_b, now);

    for sink in [&sink_a, &sink_b] {
        assert!(
            sink.drain().iter().any(is_health_event),
            "each subscriber's own gate must independently deliver the same counter bump, \
             never starved by another subscriber's gate"
        );
    }
}

#[test]
fn an_idle_gate_delivers_nothing_across_sixty_ticks_while_the_version_is_unchanged() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let sink = RecordingSink::default();
    let mut health_gate = HealthDeliveryGate::new();
    let now = Instant::now();
    // Prime the gate once so the idle phase below starts from an
    // already-delivered baseline, matching a real connection's seeded start.
    flush_tick(&coalescer, &health, || None, &mut health_gate, &sink, now);
    sink.drain();

    for tick in 1..=60u64 {
        flush_tick(
            &coalescer,
            &health,
            || None,
            &mut health_gate,
            &sink,
            now + std::time::Duration::from_secs(tick),
        );
    }

    let health_events = sink.drain().into_iter().filter(is_health_event).count();
    assert_eq!(
        health_events, 0,
        "an idle subscriber must receive nothing for 60s while counters are unchanged"
    );
}

#[test]
fn health_log_line_carries_version_bounds_and_a_null_ingress_p99() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let mut health_gate = HealthDeliveryGate::new();
    let sink = RecordingSink::default();
    health.bump_reservation_exhausted();

    flush_tick(
        &coalescer,
        &health,
        || None,
        &mut health_gate,
        &sink,
        Instant::now(),
    );

    let message = sink
        .drain()
        .into_iter()
        .find_map(|event| match event {
            mesh_llm_events::OutputEvent::Info { message, context }
                if context.as_deref() == Some("event_system_health") =>
            {
                Some(message)
            }
            _ => None,
        })
        .expect("a health line was emitted");

    let bounds = crate::runtime_events::config::EngineConfig::FROZEN;
    assert!(message.contains("version=1"), "line: {message}");
    assert!(
        message.contains(&format!(
            "bounds.reservation_table_capacity={}",
            bounds.reservation_table_capacity
        )),
        "line: {message}"
    );
    assert!(
        message.contains(&format!(
            "bounds.max_concurrent_subscribers={}",
            bounds.max_concurrent_subscribers
        )),
        "line: {message}"
    );
    assert!(message.contains("ingress_p99_us=null"), "line: {message}");
}
