//! The engine-owned driver: the single production task that pumps
//! `RuntimeEventEngine::drain()` in every serving mode, so a submitted
//! fact is actually applied through the reducer and published to
//! subscribers -- fixing review defect D3
//! (`.omo/plans/event-system-fixes.md` task 3). Before this module, the
//! ONLY thing that ever called `drain()` in production was the
//! presentation subscriber's own render tick (`presentation/subscriber.rs`),
//! so `--local-model-only` -- which never attaches that subscriber, by
//! design -- never drained a single fact, and nothing in the process ever
//! called `RuntimeEventEngine::shutdown`.
//!
//! [`spawn_engine_driver`] is spawned once per installed engine, immediately
//! after `install_runtime_event_engine`, in BOTH `runtime/run_auto.rs` (the
//! mesh-serve/TUI path, alongside the presentation subscriber) and
//! `runtime/local_model_only.rs` (which keeps its own "zero management
//! subscribers" invariant -- the driver needs no subscriber to do its job).
//! The spawned task loops on `select!` over two independent wake sources:
//!
//! - the engine's `Notify`, signaled by every `SubmitOutcome::Accepted`
//!   submission (`RuntimeEventEngine::submit`), so a terminal, a newly
//!   distinct state-transition, or a diagnostic within its lane depth is
//!   applied and published promptly, not just once per fallback tick.
//!   `Coalesced`/`Dropped*`/`Rejected*` outcomes never signal: progress intentionally
//!   flushes on its own cadence (task 4), never on every coalesce, or
//!   coalescing would lose its whole point.
//! - a `TUI_RENDER_TICK` (33 ms) fallback, so a process with an installed
//!   engine but no active producers right now (idle between requests) still
//!   converges, and a signal that is technically missed (the `Notify`
//!   permit already consumed by a still-in-flight `drain()` call) is
//!   bounded by at most one tick's latency rather than lost.
//!
//! Task 4 owns restructuring what the drain entry point actually applies
//! (terminal and state lanes, then the diagnostic queue, then a 100 ms
//! progress flush); this module calls whatever that entry point currently
//! is (`RuntimeEventEngine::drain()` today) so this call site does not need
//! to change when task 4 lands.
//!
//! Shutdown ([`shutdown_engine_driver`] / [`finalize_engine_driver`]) is
//! called from the SAME shutdown sequence that already emits
//! `node_draining`/`node_stopped` (`runtime/node_lifecycle_events.rs`),
//! AFTER both of those facts are submitted -- so this call's own final
//! drain is what applies and publishes them; a driver aborted first would
//! leave a submitted `node_stopped` sitting in the wake list forever with
//! nothing left to drain it. Production finalization leaves the optional
//! work budget unset, so the wake list and every other bounded lane drain in
//! one shared ingress prefix until the common deadline. The
//! `SHUTDOWN_DRAIN_DEADLINE` (2 s) is shared by driver join and final drain;
//! the final drain checks it between bounded reducer batches. Task 5's
//! `CHILD_SETTLE_GRACE` remains enforced by each normal drain pass, while
//! shutdown force-settles any still-pending root after the driver has stopped.

use std::sync::Arc;
use std::time::Instant;

use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

use super::config::{SHUTDOWN_DRAIN_DEADLINE, TUI_RENDER_TICK};
use super::engine::RuntimeEventEngine;
use super::health::HealthDeliveryGate;
use super::presentation::health_projection_event;

/// A handle which owns the engine-driver task.
///
/// Tokio detaches a task when its raw [`JoinHandle`] is dropped. That is the
/// wrong lifetime for the engine driver: startup can fail after the task is
/// spawned, and a detached driver would retain its `Arc<RuntimeEventEngine>`
/// forever. Owning the handle here makes every early-return path cancel the
/// task and release its engine reference once the scheduler polls the
/// cancellation.
pub struct EngineDriverHandle {
    handle: JoinHandle<()>,
}

impl EngineDriverHandle {
    /// Request cancellation without consuming the owner. The owner still
    /// aborts again from `Drop`, which keeps cancellation safe when callers
    /// use this method on an error path and then return immediately.
    pub fn abort(&self) {
        self.handle.abort();
    }

    /// Abort the driver and wait until the task has stopped. Shutdown owns
    /// this handle so the final engine drain cannot race a still-running
    /// driver pass.
    pub async fn stop_and_wait(mut self) {
        self.handle.abort();
        let _ = (&mut self.handle).await;
    }
}

impl From<JoinHandle<()>> for EngineDriverHandle {
    fn from(handle: JoinHandle<()>) -> Self {
        Self { handle }
    }
}

impl Drop for EngineDriverHandle {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Spawn the engine-owned driver task for `engine`. Keep the returned
/// handle alive for the life of the serving process and hand it to
/// [`shutdown_engine_driver`] (or [`finalize_engine_driver`]) during
/// graceful shutdown -- see the module doc for why aborting it any earlier
/// can silently lose a fact.
#[must_use]
pub fn spawn_engine_driver(engine: Arc<RuntimeEventEngine>) -> EngineDriverHandle {
    EngineDriverHandle::from(tokio::spawn(drive_engine(engine)))
}

async fn drive_engine(engine: Arc<RuntimeEventEngine>) {
    let mut tick = tokio::time::interval(TUI_RENDER_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    // Task 13 (`.omo/plans/event-system-fixes.md`): this driver's OWN
    // `event_system_health` log-line consumer, gated exactly like the v1
    // SSE `live_loop` and the presentation subscriber's `maybe_emit_health`
    // (`runtime_events::health`'s module doc: each independent consumer
    // owns its own `HealthDeliveryGate`, so one consumer's cadence can
    // never starve another's, and none of them is an engine-global gate --
    // defect D9). This is a FOURTH such consumer, not a replacement for
    // the presentation subscriber's own emission: the driver runs in every
    // serving mode INCLUDING `--local-model-only`, which never spawns a
    // presentation subscriber at all (see `presentation::subscriber`'s
    // module doc), so it is the only way the log line reaches that mode.
    let mut health_gate = HealthDeliveryGate::new();
    loop {
        tokio::select! {
            () = engine.notified() => {
                engine.drain();
                maybe_emit_health_log_line(&engine, &mut health_gate);
            }
            _ = tick.tick() => {
                engine.drain();
                maybe_emit_health_log_line(&engine, &mut health_gate);
            }
        }
    }
}

/// Emit the `event_system_health` log line through `mesh_llm_events::emit_event`
/// only when `health_gate` says the last-delivered version is stale.
/// `ingress_p99_us` (task 13) is only computed -- via
/// `RuntimeEventEngine::ingress_p99_us`, which sorts the reservoir's
/// samples -- when actually about to deliver, never on every drain pass.
/// `emit_event` no-ops safely when no output sink is installed, so this is
/// harmless to call unconditionally in a context with no sink (e.g. a
/// test).
fn maybe_emit_health_log_line(engine: &RuntimeEventEngine, health_gate: &mut HealthDeliveryGate) {
    let snapshot = engine.health().snapshot();
    if health_gate.should_deliver(&snapshot, Instant::now()) {
        let event = health_projection_event(snapshot, engine.ingress_p99_us());
        if let Err(error) = mesh_llm_events::emit_event(event) {
            tracing::warn!("engine-driver health log emission failed: {error}");
        }
    }
}

/// Close admission, stop and await the driver, then drain `engine` one final
/// time, bounded by `budget` (see `RuntimeEventEngine::shutdown`). Exposed
/// directly -- rather than only through [`finalize_engine_driver`] -- so a
/// caller that already holds its own engine reference, and every test in
/// this module, can exercise the exact production sequencing without the
/// process-global installed-engine indirection.
pub async fn shutdown_engine_driver(
    engine: &RuntimeEventEngine,
    driver_handle: EngineDriverHandle,
    budget: Option<usize>,
) {
    // Start the deadline before awaiting the task. The same deadline is
    // passed to the exclusive final drain, so task-stop time counts against
    // the shutdown budget rather than being an unbounded prelude.
    let deadline = Instant::now() + SHUTDOWN_DRAIN_DEADLINE;
    engine.close_admission();
    driver_handle.stop_and_wait().await;
    engine.shutdown_until(budget, deadline);
    // The final drain may create the only shutdown/capacity degradation seen
    // during process teardown. Deliver one uncadenced health line after the
    // exclusive drain so that state is observable even when the driver's
    // normal health gate already emitted recently.
    emit_final_health_log_line(engine);
}

/// Production shutdown convenience for both serving modes: shut down the
/// process-local installed engine (`runtime_events::runtime_event_engine`),
/// if one is installed, through [`shutdown_engine_driver`] without an
/// artificial lane-specific cap; the shared deadline bounds the final drain.
/// Stop and await `driver_handle` either way. Call this
/// from the SAME shutdown sequence that already emits
/// `node_draining`/`node_stopped`, after both are submitted (see the
/// module doc).
pub async fn finalize_engine_driver(driver_handle: EngineDriverHandle) {
    match crate::runtime_events::runtime_event_engine() {
        Some(engine) => shutdown_engine_driver(&engine, driver_handle, None).await,
        None => driver_handle.stop_and_wait().await,
    }
}

fn emit_final_health_log_line(engine: &RuntimeEventEngine) {
    let snapshot = engine.health().snapshot();
    let event = health_projection_event(snapshot, engine.ingress_p99_us());
    if let Err(error) = mesh_llm_events::emit_event(event) {
        tracing::warn!("final engine-driver health log emission failed: {error}");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use mesh_llm_events::{OutputEvent, OutputSink, clear_output_sink, set_output_sink};
    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeEventIngress,
        RuntimeFact, SubmitOutcome,
    };

    use super::EngineDriverHandle;
    use super::{finalize_engine_driver, shutdown_engine_driver, spawn_engine_driver};
    use crate::runtime_events::config::{TUI_RENDER_TICK, WAKE_LIST_DEPTH};
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::reservation::TerminalRecord;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    fn terminal_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
    }

    /// Let an already-spawned driver consume the fallback interval's own
    /// immediate first tick (a `tokio::time::interval` quirk: the first
    /// `.tick()` completes right away, before any real period has
    /// elapsed) with nothing queued to drain, so it settles into waiting
    /// for its genuinely-periodic second tick. Tests that need to prove
    /// WHICH branch of the driver's `select!` caused a drain call this
    /// first, before submitting anything, so a later small time-advance
    /// cannot be attributed to that free first tick.
    async fn settle_past_the_free_first_tick() {
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
    }

    /// A startup error can drop the driver before the normal shutdown path
    /// gets a chance to run. The owning wrapper must cancel the task instead
    /// of detaching it, so its engine reference disappears after the runtime
    /// scheduler observes the cancellation.
    #[tokio::test]
    async fn dropping_driver_on_startup_failure_releases_engine_after_scheduler_yields() {
        let engine = RuntimeEventEngine::new();
        let weak = std::sync::Arc::downgrade(&engine);
        let driver = spawn_engine_driver(engine.clone());

        drop(engine);
        assert!(
            weak.upgrade().is_some(),
            "the running driver owns the engine"
        );

        drop(driver);
        tokio::task::yield_now().await;

        assert!(
            weak.upgrade().is_none(),
            "dropping an early-startup driver must not leave a detached task holding the engine"
        );
    }

    /// Task 3 acceptance: "with no subscriber attached, a submitted
    /// terminal is applied and its reservation released within one tick."
    /// Proven by advancing virtual time by LESS than one `TUI_RENDER_TICK`
    /// after the driver has already settled past its free first tick, so
    /// only the `Notify` branch could possibly have fired.
    #[tokio::test(start_paused = true)]
    async fn notify_signal_drains_a_terminal_before_the_next_fallback_tick() {
        let engine = RuntimeEventEngine::new();
        assert_eq!(engine.subscribers().active_count(), 0);
        let driver = spawn_engine_driver(engine.clone());
        settle_past_the_free_first_tick().await;

        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reservation");
        assert_eq!(
            reservation.ingress().try_submit(terminal_fact()),
            SubmitOutcome::Accepted
        );

        tokio::time::advance(TUI_RENDER_TICK / 4).await;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        assert_eq!(
            engine.occupied_count(),
            0,
            "the notify signal -- not the fallback tick, which cannot have fired yet -- \
             must drain and release the reservation"
        );

        driver.abort();
    }

    /// Task 3 acceptance: the driver's fallback tick is what keeps a
    /// system converged even when nothing signals it. Proven by writing a
    /// wake entry directly through the reservation table and wake list --
    /// exactly what `engine::lanes::submit_terminal` does internally,
    /// minus the `Notify` call `RuntimeEventEngine::submit` performs on
    /// top of it -- so no signal is ever sent for this entry.
    #[tokio::test(start_paused = true)]
    async fn fallback_tick_drains_a_wake_entry_that_never_signaled_notify() {
        let engine = RuntimeEventEngine::new();
        let driver = spawn_engine_driver(engine.clone());
        settle_past_the_free_first_tick().await;

        let scope = OperationScope::root_only(OperationId::new());
        let handle = engine.table().reserve(scope).expect("reserve");
        assert!(engine.table().write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
        engine.wake().push_next(handle);

        tokio::time::advance(TUI_RENDER_TICK * 3).await;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        assert_eq!(
            engine.occupied_count(),
            0,
            "the fallback tick alone -- no notify was ever signaled for this entry -- must \
             still drain and release the reservation"
        );

        driver.abort();
    }

    /// Task 3 acceptance: "shutdown test proves `shutdown_degraded` stays
    /// 0 when the wake list is drained within budget."
    #[tokio::test]
    async fn shutdown_within_budget_does_not_degrade() {
        let engine = RuntimeEventEngine::with_capacity(4);
        for _ in 0..2 {
            let reservation = engine
                .reserve_root(OperationId::new(), terminal_fact)
                .expect("reserve");
            reservation.ingress().try_submit(terminal_fact());
        }
        let driver_handle = EngineDriverHandle::from(tokio::spawn(async {}));

        shutdown_engine_driver(&engine, driver_handle, Some(2)).await;

        assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
        assert!(engine.is_shutting_down());
    }

    /// Task 3 acceptance: "...and increments otherwise."
    #[tokio::test]
    async fn shutdown_past_its_budget_degrades_the_remainder() {
        let engine = RuntimeEventEngine::with_capacity(4);
        for _ in 0..3 {
            let reservation = engine
                .reserve_root(OperationId::new(), terminal_fact)
                .expect("reserve");
            reservation.ingress().try_submit(terminal_fact());
        }
        let driver_handle = EngineDriverHandle::from(tokio::spawn(async {}));

        shutdown_engine_driver(&engine, driver_handle, Some(1)).await;

        assert_eq!(engine.health().snapshot().shutdown_degraded, 1);
        assert_eq!(engine.health().snapshot().terminal_delivery_failed, 2);
    }

    /// End-to-end sanity: a real driver drains a fact through its own
    /// notify-driven tick during steady-state operation, and the explicit
    /// shutdown call afterward -- against an already-empty wake list -- is
    /// still safe, reports no degradation, and cleanly stops the task.
    #[tokio::test(start_paused = true)]
    async fn shutdown_engine_driver_is_safe_after_the_driver_already_drained_everything() {
        let engine = RuntimeEventEngine::new();
        let driver = spawn_engine_driver(engine.clone());
        settle_past_the_free_first_tick().await;

        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reservation");
        reservation.ingress().try_submit(terminal_fact());

        tokio::time::advance(TUI_RENDER_TICK).await;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert_eq!(engine.occupied_count(), 0);

        shutdown_engine_driver(&engine, driver, Some(WAKE_LIST_DEPTH)).await;

        assert!(engine.is_shutting_down());
        assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
    }

    /// [`finalize_engine_driver`] is the production call site: it must
    /// resolve the process-global installed engine itself (not require the
    /// caller to hold one) and still shut it down correctly.
    #[tokio::test]
    #[serial_test::serial(runtime_event_engine_state)]
    async fn finalize_engine_driver_shuts_down_the_process_installed_engine() {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        let driver = spawn_engine_driver(engine.clone());

        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reservation");
        reservation.ingress().try_submit(terminal_fact());

        finalize_engine_driver(driver).await;

        assert!(engine.is_shutting_down());
        assert_eq!(
            engine.occupied_count(),
            0,
            "finalize_engine_driver's own engine.shutdown() call drains synchronously, \
             independent of whether the driver task itself ever got scheduled"
        );
        clear_runtime_event_engine();
    }

    #[derive(Default)]
    struct RecordingOutputSink {
        events: StdMutex<Vec<OutputEvent>>,
    }

    impl RecordingOutputSink {
        fn take_events(&self) -> Vec<OutputEvent> {
            std::mem::take(
                &mut *self
                    .events
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
            )
        }
    }

    impl OutputSink for RecordingOutputSink {
        fn emit_event(&self, event: OutputEvent) -> std::io::Result<()> {
            self.events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(event);
            Ok(())
        }
    }

    fn is_health_event(event: &OutputEvent) -> bool {
        matches!(
            event,
            OutputEvent::Info { context, .. } if context.as_deref() == Some("event_system_health")
        )
    }

    /// `mesh_llm_events::emit_event`'s output sink is one PROCESS-GLOBAL
    /// static: any OTHER test in this file (or anywhere else in the
    /// binary) that spawns its own `spawn_engine_driver` -- and there are
    /// several, none of them `#[serial_test::serial]`-marked, since they
    /// never cared about output before this task -- can run concurrently
    /// on a different OS thread and land ITS OWN driver's health line
    /// (indistinguishable "version=0 ..." on a fresh, idle engine) in this
    /// test's recording sink too. Filtering on a distinctive
    /// `reservation_exhausted` count this test alone produces (bumped
    /// this many times before the driver ever spawns) makes the
    /// assertions below robust to that noise without requiring every
    /// driver-spawning test in the crate to coordinate a shared lock.
    const DISTINCTIVE_BUMP_COUNT: usize = 137;

    fn matches_this_test(event: &OutputEvent) -> bool {
        is_health_event(event)
            && matches!(
                event,
                OutputEvent::Info { message, .. }
                    if message.contains(&format!("reservation_exhausted={DISTINCTIVE_BUMP_COUNT}"))
            )
    }

    /// Task 13 (`.omo/plans/event-system-fixes.md`, landmine 3): the
    /// driver's own health-log-line consumer must be gated exactly like
    /// the pre-existing v1 SSE `live_loop` and presentation-subscriber
    /// consumers -- NOT an ungated fourth consumer that would flood
    /// identical `event_system_health` lines on every ~33ms fallback
    /// tick. A fresh `HealthDeliveryGate` delivers unconditionally on its
    /// own first eligible check (see `health.rs`'s
    /// `health_delivery_gate_delivers_on_the_first_check_from_new`), which
    /// this driver hits on its own free first tick (`tokio::time::interval`
    /// completes its first `.tick()` immediately, independent of virtual
    /// time -- the same quirk `drive_presentation_subscriber`'s own
    /// analogous test documents) BEFORE this test submits anything. Five
    /// more fallback ticks with nothing changed must add NOTHING further.
    #[tokio::test(start_paused = true)]
    #[serial_test::serial]
    async fn driver_health_log_line_is_gated_not_flooded_on_every_fallback_tick() {
        clear_output_sink();
        let sink = std::sync::Arc::new(RecordingOutputSink::default());
        set_output_sink(sink.clone());

        let engine = RuntimeEventEngine::new();
        for _ in 0..DISTINCTIVE_BUMP_COUNT {
            engine.health().bump_reservation_exhausted();
        }
        let driver = spawn_engine_driver(engine.clone());
        settle_past_the_free_first_tick().await;

        tokio::time::advance(TUI_RENDER_TICK * 5).await;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        driver.abort();
        clear_output_sink();
        let events = sink.take_events();
        let matched = events
            .iter()
            .filter(|event| matches_this_test(event))
            .count();
        assert_eq!(
            matched,
            1,
            "the driver's free first tick delivers exactly one health line for this test's \
             distinctive state; five more ticks with no counter change must add none -- an \
             ungated consumer would show six; matching events: {:?}",
            events
                .iter()
                .filter(|event| matches_this_test(event))
                .collect::<Vec<_>>()
        );
    }

    /// Complementary half of the gating proof: once a real counter changes
    /// AND real wall-clock time (the gate's cadence is real `Instant::now()`,
    /// never virtualized by `tokio::time::pause`) has passed the 1s
    /// minimum, the driver's gate DOES deliver again -- proving this is a
    /// real change-and-cadence gate, not a permanently-latched one-shot.
    #[tokio::test(start_paused = true)]
    #[serial_test::serial]
    async fn driver_health_log_line_delivers_again_after_a_real_change_past_the_cadence() {
        clear_output_sink();
        let sink = std::sync::Arc::new(RecordingOutputSink::default());
        set_output_sink(sink.clone());

        let engine = RuntimeEventEngine::new();
        for _ in 0..DISTINCTIVE_BUMP_COUNT {
            engine.health().bump_reservation_exhausted();
        }
        let driver = spawn_engine_driver(engine.clone());
        settle_past_the_free_first_tick().await;
        let first_matches = sink
            .take_events()
            .into_iter()
            .filter(matches_this_test)
            .count();
        assert_eq!(first_matches, 1, "the free first tick must deliver once");

        // A real, further counter change past the 1s real-wall-clock
        // cadence (`HealthDeliveryGate`'s cadence reads real
        // `std::time::Instant::now()`, never virtualized by
        // `tokio::time::pause` -- mirrors `wiring.rs`'s identical
        // technique for the presentation subscriber's own recv arm).
        std::thread::sleep(std::time::Duration::from_millis(1_100));
        engine.health().bump_reservation_exhausted();
        tokio::time::advance(TUI_RENDER_TICK).await;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        driver.abort();
        clear_output_sink();
        let events = sink.take_events();
        let second_count = format!("reservation_exhausted={}", DISTINCTIVE_BUMP_COUNT + 1);
        let matched = events
            .iter()
            .filter(|event| {
                is_health_event(event)
                    && matches!(event, OutputEvent::Info { message, .. } if message.contains(&second_count))
            })
            .count();
        assert_eq!(
            matched, 1,
            "a real counter change past the 1s cadence must deliver exactly one more \
             health line reflecting it; got: {events:?}"
        );
    }
}
