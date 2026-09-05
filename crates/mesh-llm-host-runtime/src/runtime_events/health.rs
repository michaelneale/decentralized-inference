//! Coalesced engine health counters.
//!
//! Health is out-of-band from the primary terminal/state/progress lanes: it
//! is never recursively submitted through those lanes. Task 8
//! (`.omo/plans/event-system-fixes.md`, defect D9) replaced the old
//! engine-global "publish at most once per second" gate
//! (`EngineHealth::publish_at`) with a monotonically increasing
//! `health_version`, bumped on every counter mutation: the engine itself
//! never gates delivery at all now. Each independent consumer -- the v1 SSE
//! loop (`api/routes/runtime_events/stream.rs`), the presentation
//! subscriber (`presentation/subscriber.rs`), and the OTLP telemetry
//! sampler (`runtime_events::telemetry::sample_engine`) -- owns its own
//! [`HealthDeliveryGate`] instance and decides for itself when to actually
//! deliver a snapshot ("changed since I last delivered, and at least a
//! second has passed"), so one consumer's cadence can never starve
//! another's: under the old shared gate, a v1 subscriber served entirely
//! from `stream.rs`'s own per-frame check could see as little as one health
//! frame per ~50 minutes when the presentation-subscriber tick kept
//! consuming the ONE shared publish window first.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::config::HEALTH_PUBLISH_MIN_INTERVAL;

/// Point-in-time counters, safe to clone and hand to a consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineHealthSnapshot {
    /// Monotonically increasing; bumped on every counter mutation below.
    /// Compared by [`HealthDeliveryGate`] to decide whether a consumer's
    /// last-delivered snapshot is now stale.
    pub version: u64,
    pub rebuild_generation: u64,
    pub reservation_exhausted: u64,
    pub terminal_delivery_failed: u64,
    pub dropped_progress: u64,
    pub dropped_diagnostic: u64,
    pub replay_evicted: u64,
    pub subscriber_disconnected: u64,
    pub shutdown_degraded: u64,
    pub reducer_rejected: u64,
    /// Distinct state-transition keys rejected at the bounded lane ceiling.
    pub state_transition_rejected: u64,
    /// Reservation-bound state submissions rejected because their generation
    /// was stale or explicitly cancelled. This is not capacity loss.
    pub cancelled_reservation_rejected: u64,
    /// Sticky signal that at least one accepted state observation could not
    /// be retained; callers must refresh from a canonical state source.
    pub state_degraded: bool,
    /// Sticky until an explicit canonical rebuild/reconciliation clears it.
    pub rebuild_required: bool,
    pub event_cutover_divergence: u64,
    pub reducer_eviction_stalled: u64,
}

#[derive(Debug, Default)]
struct Counters {
    rebuild_generation: AtomicU64,
    reservation_exhausted: AtomicU64,
    terminal_delivery_failed: AtomicU64,
    dropped_progress: AtomicU64,
    dropped_diagnostic: AtomicU64,
    replay_evicted: AtomicU64,
    subscriber_disconnected: AtomicU64,
    shutdown_degraded: AtomicU64,
    reducer_rejected: AtomicU64,
    state_transition_rejected: AtomicU64,
    cancelled_reservation_rejected: AtomicU64,
    state_degraded: AtomicBool,
    rebuild_required: AtomicBool,
    event_cutover_divergence: AtomicU64,
    reducer_eviction_stalled: AtomicU64,
}

/// Engine health: coalesced counters plus a monotonic version.
#[derive(Debug, Default)]
pub struct EngineHealth {
    counters: Counters,
    health_version: AtomicU64,
}

impl EngineHealth {
    fn bump_version(&self) {
        self.health_version.fetch_add(1, Ordering::Release);
    }

    pub fn bump_reservation_exhausted(&self) {
        self.counters
            .reservation_exhausted
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_terminal_delivery_failed(&self) {
        self.counters
            .terminal_delivery_failed
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_dropped_progress(&self) {
        self.counters
            .dropped_progress
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_dropped_diagnostic(&self) {
        self.counters
            .dropped_diagnostic
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// One increment, matching the contract every existing call site
    /// (`runtime_events::engine::drain`, out of task 8's file ownership)
    /// still relies on. See [`Self::bump_replay_evicted_by`] for the
    /// count-aware primitive this task adds so a caller that evicts several
    /// frames in one operation can report the real count instead of one
    /// bump per call -- BLOCKED from being wired into the one production
    /// call site that needs it (`engine/drain.rs`'s `apply_and_publish_fact`)
    /// because both that file and `replay.rs` (whose `push`/`push_at`
    /// would need to return the evicted count instead of a `bool`) are
    /// outside task 8's ownership; see the task 8 DoneClaim for the full
    /// analysis and the exact diff a follow-up task can apply.
    pub fn bump_replay_evicted(&self) {
        self.counters.replay_evicted.fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// Frozen `replay_evicted` semantics (task 8,
    /// `.omo/plans/event-system-fixes.md`): one increment per evicted
    /// frame, not per push. Adds `count` in a single atomic op so a caller
    /// that evicted several frames in one operation reports the real
    /// count. `count == 0` is a no-op, including no version bump -- there
    /// is nothing to report.
    pub fn bump_replay_evicted_by(&self, count: u64) {
        if count == 0 {
            return;
        }
        self.counters
            .replay_evicted
            .fetch_add(count, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_subscriber_disconnected(&self) {
        self.counters
            .subscriber_disconnected
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_shutdown_degraded(&self) {
        self.counters
            .shutdown_degraded
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    pub fn bump_reducer_rejected(&self) {
        self.counters
            .reducer_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// Record a state-lane capacity rejection without evicting a previously
    /// accepted key. The degraded/rebuild flags are sticky so consumers can
    /// request canonical state recovery even after the counter is observed.
    pub fn bump_state_transition_rejected(&self) {
        self.counters
            .state_transition_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.counters.state_degraded.store(true, Ordering::Relaxed);
        self.counters
            .rebuild_required
            .store(true, Ordering::Relaxed);
        self.bump_version();
    }

    /// Record a reservation-bound state submission rejected by cancellation or
    /// generation mismatch. Unlike lane-capacity rejection, this does not
    /// mark state degraded or require a rebuild.
    pub fn bump_cancelled_reservation_rejected(&self) {
        self.counters
            .cancelled_reservation_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// Clear the sticky capacity-degradation marker after an authoritative
    /// canonical rebuild/reconciliation has completed.
    pub fn clear_state_degraded(&self) {
        self.counters.state_degraded.store(false, Ordering::Relaxed);
        self.counters
            .rebuild_required
            .store(false, Ordering::Relaxed);
        self.bump_version();
    }

    /// Task 6 (`.omo/plans/event-system-fixes.md`, defect D14): a
    /// `runtime_data::event_cutover` shadow comparison found a legacy
    /// value that disagreed with the reducer's own projection. The legacy
    /// value stays authoritative regardless -- this counter is
    /// observability only, never a cutover trigger.
    pub fn bump_event_cutover_divergence(&self) {
        self.counters
            .event_cutover_divergence
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// Task 6-fix (`.omo/plans/event-system-fixes.md`, "also required" review
    /// finding on top of defect A): bumped when
    /// `ReducerSnapshot::operation_count()` exceeds `TOTAL_OPERATION_BOUND`
    /// -- the reducer's TRUE structural ceiling.
    ///
    /// R1 CORRECTION (task 6-fix): this doc comment used to claim the
    /// condition was "structurally unreachable in production now that
    /// release-triggered eviction keeps the map far below capacity in the
    /// steady state" -- false. That reasoning covered only
    /// reservation-backed scopes; six production call sites
    /// (`unreserved_ingress` with a fresh `OperationId` per event -- KV
    /// cache lookups, session lifecycle, node/topology/model-lifecycle
    /// observers) can never settle and never release, so the settled-only
    /// backstop's "nothing left to evict" branch was genuinely reachable
    /// and would stall PERMANENTLY once such traffic exceeded
    /// `RESERVATION_TABLE_CAPACITY`. `ReducerSnapshot`'s new
    /// `unreserved_order` bounded LRU (`reducer/state.rs`) bounds those
    /// scopes independently, so this counter now compares against
    /// `TOTAL_OPERATION_BOUND` (`RESERVATION_TABLE_CAPACITY +
    /// UNRESERVED_OPERATION_BOUND`) and should stay unreachable in
    /// practice again -- for the right reason this time. Not projected
    /// onto the `runtime_health` wire frame -- that is task 7/8's
    /// `frames.rs` territory, explicitly out of this task's scope.
    pub fn bump_reducer_eviction_stalled(&self) {
        self.counters
            .reducer_eviction_stalled
            .fetch_add(1, Ordering::Relaxed);
        self.bump_version();
    }

    /// Task 13 (`.omo/plans/event-system-fixes.md`): bumped once per
    /// `INGRESS_LATENCY_MIN_SAMPLES` newly recorded ingress-latency
    /// samples (see `ingress_latency::IngressLatencyReservoir::record`),
    /// so a genuinely-changed `ingress_p99_us` is periodically observable
    /// through the existing change-gated health-delivery contract without
    /// bumping (and so flooding delivery) on every single submission. No
    /// dedicated counter is added to `EngineHealthSnapshot` for this --
    /// the p99 VALUE itself, sourced separately from
    /// `RuntimeEventEngine::ingress_p99_us`, is the observable change.
    pub fn bump_for_ingress_latency_milestone(&self) {
        self.bump_version();
    }

    pub fn set_rebuild_generation(&self, value: u64) {
        self.counters
            .rebuild_generation
            .store(value, Ordering::Relaxed);
        self.bump_version();
    }

    /// Snapshot every counter plus the current version.
    #[must_use]
    pub fn snapshot(&self) -> EngineHealthSnapshot {
        // Load the publication marker before the counters. Every counter
        // mutation stores its value before the corresponding Release bump,
        // so an Acquire read of the version publishes a coherent-enough
        // snapshot to independent consumers without locking the counters.
        let version = self.health_version.load(Ordering::Acquire);
        EngineHealthSnapshot {
            version,
            rebuild_generation: self.counters.rebuild_generation.load(Ordering::Relaxed),
            reservation_exhausted: self.counters.reservation_exhausted.load(Ordering::Relaxed),
            terminal_delivery_failed: self
                .counters
                .terminal_delivery_failed
                .load(Ordering::Relaxed),
            dropped_progress: self.counters.dropped_progress.load(Ordering::Relaxed),
            dropped_diagnostic: self.counters.dropped_diagnostic.load(Ordering::Relaxed),
            replay_evicted: self.counters.replay_evicted.load(Ordering::Relaxed),
            subscriber_disconnected: self
                .counters
                .subscriber_disconnected
                .load(Ordering::Relaxed),
            shutdown_degraded: self.counters.shutdown_degraded.load(Ordering::Relaxed),
            reducer_rejected: self.counters.reducer_rejected.load(Ordering::Relaxed),
            state_transition_rejected: self
                .counters
                .state_transition_rejected
                .load(Ordering::Relaxed),
            cancelled_reservation_rejected: self
                .counters
                .cancelled_reservation_rejected
                .load(Ordering::Relaxed),
            state_degraded: self.counters.state_degraded.load(Ordering::Relaxed),
            rebuild_required: self.counters.rebuild_required.load(Ordering::Relaxed),
            event_cutover_divergence: self
                .counters
                .event_cutover_divergence
                .load(Ordering::Relaxed),
            reducer_eviction_stalled: self
                .counters
                .reducer_eviction_stalled
                .load(Ordering::Relaxed),
        }
    }
}

/// Per-consumer cadence gate (task 8, replacing the removed engine-global
/// `EngineHealth::publish_at`): each independent health consumer owns its
/// OWN instance rather than sharing the engine's single gate, so one
/// consumer's cadence can never starve another's.
#[derive(Debug, Default)]
pub struct HealthDeliveryGate {
    last_delivered_version: Option<u64>,
    last_delivered_at: Option<Instant>,
}

impl HealthDeliveryGate {
    /// A gate that has never delivered: the next [`Self::should_deliver`]
    /// call always fires, regardless of the snapshot's version.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A gate that already considers `version` delivered as of `at` -- for
    /// a consumer that already sent an initial health snapshot through some
    /// OTHER path (the v1 SSE loop's `write_initial_frames`, outside this
    /// gate's own bookkeeping), so the gate's first eligible check does not
    /// immediately re-deliver the same snapshot moments later.
    #[must_use]
    pub fn seeded(version: u64, at: Instant) -> Self {
        Self {
            last_delivered_version: Some(version),
            last_delivered_at: Some(at),
        }
    }

    /// `true` exactly when `snapshot.version` differs from the last
    /// delivered version AND at least [`HEALTH_PUBLISH_MIN_INTERVAL`] has
    /// passed since the last delivery (or none has happened yet). Records
    /// the delivery on `true`, so an immediate repeated call with the same
    /// `now` returns `false`.
    pub fn should_deliver(&mut self, snapshot: &EngineHealthSnapshot, now: Instant) -> bool {
        let changed = self.last_delivered_version != Some(snapshot.version);
        let elapsed_ok = match self.last_delivered_at {
            None => true,
            Some(previous) => now.duration_since(previous) >= HEALTH_PUBLISH_MIN_INTERVAL,
        };
        if !(changed && elapsed_ok) {
            return false;
        }
        self.last_delivered_version = Some(snapshot.version);
        self.last_delivered_at = Some(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn health_version_bumps_on_every_counter_mutation() {
        let health = EngineHealth::default();
        assert_eq!(health.snapshot().version, 0);
        health.bump_reservation_exhausted();
        assert_eq!(health.snapshot().version, 1);
        health.bump_replay_evicted_by(4);
        assert_eq!(health.snapshot().version, 2);
        health.set_rebuild_generation(7);
        assert_eq!(health.snapshot().version, 3);
    }

    #[test]
    fn bump_for_ingress_latency_milestone_bumps_version_with_no_dedicated_counter() {
        let health = EngineHealth::default();
        assert_eq!(health.snapshot().version, 0);

        health.bump_for_ingress_latency_milestone();

        assert_eq!(
            health.snapshot(),
            EngineHealthSnapshot {
                version: 1,
                ..EngineHealthSnapshot::default()
            },
            "a milestone bump must move only the version, no other counter"
        );
    }

    #[test]
    fn bump_replay_evicted_by_credits_every_evicted_frame_in_one_call() {
        let health = EngineHealth::default();
        health.bump_replay_evicted_by(4);
        assert_eq!(
            health.snapshot().replay_evicted,
            4,
            "eviction of four frames must count four"
        );
    }

    #[test]
    fn bump_replay_evicted_by_zero_is_a_no_op_including_the_version() {
        let health = EngineHealth::default();
        health.bump_replay_evicted_by(0);
        let snapshot = health.snapshot();
        assert_eq!(snapshot.replay_evicted, 0);
        assert_eq!(snapshot.version, 0);
    }

    #[test]
    fn health_delivery_gate_delivers_on_the_first_check_from_new() {
        let mut gate = HealthDeliveryGate::new();
        assert!(gate.should_deliver(&EngineHealthSnapshot::default(), Instant::now()));
    }

    #[test]
    fn health_delivery_gate_never_delivers_the_same_version_twice() {
        let mut gate = HealthDeliveryGate::new();
        let snapshot = EngineHealthSnapshot {
            version: 3,
            ..Default::default()
        };
        let now = Instant::now();
        assert!(gate.should_deliver(&snapshot, now));
        assert!(!gate.should_deliver(&snapshot, now + HEALTH_PUBLISH_MIN_INTERVAL));
        assert!(!gate.should_deliver(&snapshot, now + Duration::from_secs(60)));
    }

    #[test]
    fn health_delivery_gate_waits_out_the_cadence_window_even_when_the_version_changed_again() {
        let mut gate = HealthDeliveryGate::new();
        let now = Instant::now();
        let first = EngineHealthSnapshot {
            version: 1,
            ..Default::default()
        };
        assert!(gate.should_deliver(&first, now));
        let second = EngineHealthSnapshot {
            version: 2,
            ..Default::default()
        };
        assert!(
            !gate.should_deliver(&second, now + Duration::from_millis(1)),
            "a changed version inside the cadence window must still wait"
        );
        assert!(gate.should_deliver(&second, now + HEALTH_PUBLISH_MIN_INTERVAL));
    }

    #[test]
    fn health_delivery_gate_seeded_treats_the_seed_version_as_already_delivered() {
        let now = Instant::now();
        let mut gate = HealthDeliveryGate::seeded(5, now);
        let unchanged = EngineHealthSnapshot {
            version: 5,
            ..Default::default()
        };
        assert!(
            !gate.should_deliver(&unchanged, now + HEALTH_PUBLISH_MIN_INTERVAL),
            "a seeded gate must not immediately re-deliver the version it was seeded with"
        );
        let changed = EngineHealthSnapshot {
            version: 6,
            ..Default::default()
        };
        assert!(gate.should_deliver(&changed, now + HEALTH_PUBLISH_MIN_INTERVAL));
    }

    #[test]
    fn snapshot_reflects_every_counter() {
        let health = EngineHealth::default();
        health.bump_reservation_exhausted();
        health.bump_terminal_delivery_failed();
        health.bump_dropped_progress();
        health.bump_dropped_diagnostic();
        health.bump_replay_evicted();
        health.bump_subscriber_disconnected();
        health.bump_shutdown_degraded();
        health.bump_reducer_rejected();
        health.bump_state_transition_rejected();
        health.bump_cancelled_reservation_rejected();
        health.bump_event_cutover_divergence();
        health.bump_reducer_eviction_stalled();
        health.set_rebuild_generation(2);

        let snapshot = health.snapshot();
        assert_eq!(
            snapshot,
            EngineHealthSnapshot {
                version: 13,
                rebuild_generation: 2,
                reservation_exhausted: 1,
                terminal_delivery_failed: 1,
                dropped_progress: 1,
                dropped_diagnostic: 1,
                replay_evicted: 1,
                subscriber_disconnected: 1,
                shutdown_degraded: 1,
                reducer_rejected: 1,
                state_transition_rejected: 1,
                cancelled_reservation_rejected: 1,
                state_degraded: true,
                rebuild_required: true,
                event_cutover_divergence: 1,
                reducer_eviction_stalled: 1,
            }
        );
    }
}
