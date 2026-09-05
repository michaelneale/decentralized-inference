//! Wires the progress coalescer and privacy-safe projection to the host
//! runtime-event engine's subscriber registry, producing at most one
//! progress `OutputEvent` per operation per render tick plus every terminal
//! and coalesced-health event, all BEFORE they ever reach
//! `mesh_llm_events::emit_event` -- the TUI's own unbounded
//! `OutputCommand::Event` channel (`mesh-llm-tui`) is never touched by this
//! module and never coalesces anything itself.
//!
//! **Spawned from `runtime/run_auto.rs`**, immediately after
//! `install_runtime_event_engine(...)`, via [`spawn_presentation_subscriber`]
//! -- the full `mesh-llm serve --auto` / TUI-visible path. Deliberately NOT
//! spawned from `runtime/local_model_only.rs`, which keeps its documented,
//! tested "zero management subscribers" invariant (nothing there calls
//! `.subscribers()`).
//!
//! [`spawn_presentation_subscriber`] is split into a synchronous [`attach`]
//! step and an async [`drive_presentation_subscriber`] loop specifically so
//! attachment is observable by the caller with no scheduler race: the
//! subscription is registered on the engine BEFORE the function returns,
//! not at some later point inside a freshly spawned task.
//!
//! This is a PURE consumer: it never calls [`RuntimeEventEngine::drain`] or
//! any other apply/release/flush entry point. The engine-owned driver task
//! (`runtime_events::driver`, spawned right alongside this subscriber in
//! `run_auto.rs`) is the process's one production drain loop; this
//! subscriber's own tick only flushes ITS OWN local, presentation-side
//! state -- the progress coalescer and the cadence-gated health snapshot --
//! and forwards whatever the driver already applied and published, which
//! this subscriber receives over its subscription. Losing this subscriber
//! (a lagged disconnect) degrades observability only; it can never stall
//! or affect the driver's own progress.

use std::sync::Arc;
use std::time::Instant;

use mesh_llm_events::OutputEvent;
use mesh_llm_runtime_event_contracts::DeliveryClass;
use tokio::sync::broadcast::error::RecvError;
use tokio::time::MissedTickBehavior;

use crate::runtime_events::config::TUI_RENDER_TICK;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::health::{EngineHealth, HealthDeliveryGate};
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::subscribers::{SubscribeError, SubscriptionHandle};

use super::coalescer::ProgressCoalescer;
use super::projection::{fact_projection_event, health_projection_event};

/// Where a presentation projection lands. Production wiring targets
/// [`EmitEventSink`]; tests inject a bounded recorder so assertions never
/// depend on the global `mesh_llm_events::OutputManager`.
pub trait PresentationSink: Send + Sync {
    fn emit(&self, event: OutputEvent);
}

/// Production sink: forwards to `mesh_llm_events::emit_event`, the same
/// entry point every hand-written `OutputEvent` call site in this crate
/// uses. A write failure is logged, never propagated -- presentation is
/// never domain authority and must not affect primary work.
pub struct EmitEventSink;

impl PresentationSink for EmitEventSink {
    fn emit(&self, event: OutputEvent) {
        if let Err(error) = mesh_llm_events::emit_event(event) {
            tracing::warn!("presentation projection emit failed: {error}");
        }
    }
}

/// Route one accepted `ReplayFrame`: a `Progress`-class fact coalesces into
/// `coalescer` (see its own bounded, latest-value-wins contract); every
/// other delivery class -- most importantly `Terminal` -- is projected and
/// forwarded to `sink` immediately, with no buffering step of any kind.
pub(super) fn route_fact(
    coalescer: &ProgressCoalescer,
    sink: &dyn PresentationSink,
    frame: &ReplayFrame,
) {
    match frame.fact.delivery_class() {
        DeliveryClass::Progress => coalescer.submit(frame.scope, (*frame.fact).clone()),
        DeliveryClass::Terminal | DeliveryClass::StateTransition | DeliveryClass::Diagnostic => {
            sink.emit(fact_projection_event(&frame.fact));
        }
    }
}

/// Flush `coalescer`'s per-operation latest progress values (bounded to at
/// most once per its own configured interval) and, if `health_gate` says
/// this subscriber's last-delivered health version is stale, forward a
/// fresh snapshot -- both immediately, neither is queued again downstream
/// of this call. Task 8 (`.omo/plans/event-system-fixes.md`, defect D9):
/// `health_gate` replaces the removed engine-global `EngineHealth::publish_at`
/// cadence with THIS subscriber's own independent gate.
pub(super) fn flush_tick(
    coalescer: &ProgressCoalescer,
    health: &EngineHealth,
    ingress_p99_us: impl FnOnce() -> Option<u64>,
    health_gate: &mut HealthDeliveryGate,
    sink: &dyn PresentationSink,
    now: Instant,
) {
    for (_scope, fact) in coalescer.flush_at(now) {
        sink.emit(fact_projection_event(&fact));
    }
    maybe_emit_health(health, ingress_p99_us, health_gate, sink, now);
}

/// Emit a coalesced health snapshot through `sink` only when `health_gate`
/// says the last-delivered version is stale. Shared by [`flush_tick`] (the
/// render-tick path) and [`drive_presentation_subscriber`]'s own recv arm
/// (the after-every-frame path), so a health change is never starved behind
/// either cadence alone.
fn maybe_emit_health(
    health: &EngineHealth,
    ingress_p99_us: impl FnOnce() -> Option<u64>,
    health_gate: &mut HealthDeliveryGate,
    sink: &dyn PresentationSink,
    now: Instant,
) {
    let snapshot = health.snapshot();
    if health_gate.should_deliver(&snapshot, now) {
        sink.emit(health_projection_event(snapshot, ingress_p99_us()));
    }
}

/// Subscribe to `engine` synchronously. Split out from the async drive loop
/// so a caller (`spawn_presentation_subscriber`) can prove the subscription
/// is registered before returning, with no race against the spawned task's
/// own scheduling.
pub fn attach(engine: &RuntimeEventEngine) -> Result<SubscriptionHandle, SubscribeError> {
    engine.subscribers().subscribe()
}

/// Drive the presentation projection loop for an already-`attach`ed
/// `subscription` until the engine's subscriber registry closes or this
/// subscription is disconnected for lagging too far behind. A lagged
/// receiver records the disconnect on engine health and returns --
/// presentation loss degrades observability only, it never blocks or fails
/// primary work. This loop never drains the engine itself (see the module
/// doc): its own tick only flushes the progress coalescer and the
/// cadence-gated health snapshot.
pub async fn drive_presentation_subscriber(
    mut subscription: SubscriptionHandle,
    engine: Arc<RuntimeEventEngine>,
    sink: Arc<dyn PresentationSink>,
) {
    let coalescer = ProgressCoalescer::new();
    let mut health_gate = HealthDeliveryGate::new();
    let mut tick = tokio::time::interval(TUI_RENDER_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            received = subscription.recv() => {
                match received {
                    Ok(frame) => {
                        route_fact(&coalescer, sink.as_ref(), &frame);
                        maybe_emit_health(
                            engine.health(),
                            || engine.ingress_p99_us(),
                            &mut health_gate,
                            sink.as_ref(),
                            Instant::now(),
                        );
                    }
                    Err(RecvError::Lagged(_)) => {
                        subscription.record_disconnect(engine.health());
                        return;
                    }
                    Err(RecvError::Closed) => return,
                }
            }
            _ = tick.tick() => {
                if subscription.lag_bound_exceeded(Instant::now()) {
                    subscription.record_disconnect(engine.health());
                    return;
                }
                flush_tick(
                    &coalescer,
                    engine.health(),
                    || engine.ingress_p99_us(),
                    &mut health_gate,
                    sink.as_ref(),
                    Instant::now(),
                );
            }
        }
    }
}

/// Attach to `engine` synchronously and spawn [`drive_presentation_subscriber`]
/// as a background task using the real [`EmitEventSink`]. This is the
/// `run_auto.rs` wiring point; do NOT call it from `local_model_only.rs`
/// (see the module doc). Returns `Err` only when the engine is already at
/// its concurrent-subscriber cap.
pub fn spawn_presentation_subscriber(
    engine: &Arc<RuntimeEventEngine>,
) -> Result<tokio::task::JoinHandle<()>, SubscribeError> {
    let subscription = attach(engine)?;
    let engine = Arc::clone(engine);
    Ok(tokio::spawn(async move {
        drive_presentation_subscriber(subscription, engine, Arc::new(EmitEventSink)).await;
    }))
}
