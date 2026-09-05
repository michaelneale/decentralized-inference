//! Frozen bounds for the host runtime-event engine.
//!
//! Every numeric limit named in the plan's "Frozen engine bounds" table lives
//! here as the single private configuration owner. `EngineConfig` is exposed
//! to health through `crate::runtime_events::health`. Changing any value
//! requires the plan's inventory amendment procedure.

use std::time::Duration;

/// Bound on host-initiated request-root operations.
///
/// Mirrors `MAX_TRACKED_REQUESTS` in `logging/openai_lifecycle.rs` (currently
/// `1_024`). That constant is private to its owning module and is treated
/// there as a reference pattern, not an editable file, so this value is
/// pinned by literal plus a contract test rather than a live import.
pub const REQUEST_ROOT_BOUND: usize = 1_024;

/// Observed-child operations admitted per request root.
pub const CHILD_MULTIPLIER: usize = 2;

/// Runtime/model lifecycle operations outside the request-root tree.
pub const LIFECYCLE_OPERATION_BOUND: usize = 64;

/// `request_root_bound * (1 + child_multiplier) + lifecycle_operation_bound`.
pub const RESERVATION_TABLE_CAPACITY: usize =
    REQUEST_ROOT_BOUND * (1 + CHILD_MULTIPLIER) + LIFECYCLE_OPERATION_BOUND;

/// State-transition lane depth.
pub const STATE_TRANSITION_LANE_DEPTH: usize = 4_096;

/// Diagnostic lane depth.
pub const DIAGNOSTIC_LANE_DEPTH: usize = 2_048;

/// R1 fix (task 6-fix, `.omo/plans/event-system-fixes.md`): bound on
/// reducer-tracked operations that were NEVER associated with a
/// reservation (`OperationState::ever_reserved == false` for their whole
/// tracked lifetime). Production observers with no bounded operation to
/// attach to -- KV cache lookups, session lifecycle, and
/// node/topology/model-lifecycle observers -- submit exactly this shape of
/// traffic via `unreserved_ingress`, minting a fresh `OperationId` per
/// event; such a scope can never settle (no Terminal is ever accepted with
/// no `SlotHandle`) and can never be release-evicted (nothing to release),
/// so neither of the reducer's two existing eviction paths can ever touch
/// it. `ReducerSnapshot`'s `unreserved_order` bounded LRU (`reducer/state.rs`)
/// enforces this ceiling directly, mirroring `reducer/domain.rs`'s existing
/// `touch`/`remove_bounded` idiom.
///
/// A direct const-expr alias to `STATE_TRANSITION_LANE_DEPTH`, not a
/// duplicated literal: every one of the six known unreserved-mint call
/// sites is StateTransition-class, so nothing can reach the reducer's
/// `operations` map for one of these scopes without first passing through
/// that lane's own admission cap on distinct `(scope, kind)` keys. Reusing
/// that ceiling for this map's post-drain retention keeps the two related
/// bounds for the SAME traffic category numerically aligned rather than
/// introducing a disconnected second magic number.
pub const UNRESERVED_OPERATION_BOUND: usize = STATE_TRANSITION_LANE_DEPTH;

/// R1 fix total structural ceiling on the reducer's `operations` map:
/// `RESERVATION_TABLE_CAPACITY` (the reservation table's own hard admission
/// cap on concurrently-occupied scopes, backstopped by
/// `reducer::state::evict_settled_over_capacity`) plus
/// `UNRESERVED_OPERATION_BOUND` (enforced independently by
/// `ReducerSnapshot`'s `unreserved_order` bounded LRU). No
/// production-reachable sequence of submissions can push
/// `ReducerSnapshot::operation_count()` above this value; the engine bumps
/// `EngineHealth::bump_reducer_eviction_stalled` if it ever somehow does.
pub const TOTAL_OPERATION_BOUND: usize = RESERVATION_TABLE_CAPACITY + UNRESERVED_OPERATION_BOUND;

/// Wake list depth, equal to the reservation table.
pub const WAKE_LIST_DEPTH: usize = RESERVATION_TABLE_CAPACITY;

/// Replay retention: frame count, age, and byte ceilings (first limit wins).
pub const REPLAY_MAX_FRAMES: usize = 4_096;
pub const REPLAY_MAX_AGE: Duration = Duration::from_secs(300);
pub const REPLAY_MAX_BYTES: usize = 8 * 1024 * 1024;

/// Per-subscriber lag: frame count, age, and byte ceilings (first limit wins).
pub const SUBSCRIBER_LAG_MAX_FRAMES: usize = 1_024;
pub const SUBSCRIBER_LAG_MAX_AGE: Duration = Duration::from_secs(30);
pub const SUBSCRIBER_LAG_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Maximum concurrent v1 subscribers.
pub const MAX_CONCURRENT_SUBSCRIBERS: usize = 32;

/// Reconnect limit: connects per client key per window; client key is peer IP.
pub const RECONNECT_LIMIT_PER_WINDOW: usize = 10;
pub const RECONNECT_WINDOW: Duration = Duration::from_secs(60);
/// Upper bound on distinct reconnect keys retained by the rate limiter.
/// Expired keys are pruned before admission; a new key is rejected at the
/// cap so active keys keep their rate-limit history instead of being evicted.
pub const RECONNECT_KEY_CAP: usize = 4_096;

/// SSE keepalive interval.
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// Health publish cadence: coalesced, at most one frame per second.
pub const HEALTH_PUBLISH_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Progress presentation/export interval.
pub const PROGRESS_EXPORT_INTERVAL: Duration = Duration::from_millis(100);

/// Existing `PRETTY_TUI_REDRAW_INTERVAL`.
pub const TUI_RENDER_TICK: Duration = Duration::from_millis(33);

/// Shutdown drain deadline.
pub const SHUTDOWN_DRAIN_DEADLINE: Duration = Duration::from_secs(2);

/// Child-settle grace before root release (task 5,
/// `.omo/plans/event-system-fixes.md`): how long a root whose own terminal
/// has already applied and published may hold its slot open while at
/// least one child is still occupied, before the engine synthesizes a
/// `terminal_not_delivered` for each remaining child and releases the
/// root anyway. Frozen equal to `SHUTDOWN_DRAIN_DEADLINE`.
pub const CHILD_SETTLE_GRACE: Duration = SHUTDOWN_DRAIN_DEADLINE;

/// Callback ingress p99 budget on certification hosts.
pub const CALLBACK_INGRESS_P99_BUDGET: Duration = Duration::from_micros(100);

/// Task 13 (`.omo/plans/event-system-fixes.md`, defect D13's p99 half),
/// amended via `[[health_bound_amendments]]` in
/// `crates/mesh-llm-runtime-event-contracts/inventory/runtime_events.toml`
/// (see `.omo/evidence/event-system-fixes/task-13/amendment-note.txt`):
/// the fixed ring size backing `runtime_events::ingress_latency::IngressLatencyReservoir`,
/// the in-process, OTLP-independent ingress-latency instrument behind
/// `runtime_health`'s `ingress_p99_us` wire field and the
/// `event_system_health` log line. Numerically equal to
/// `STATE_TRANSITION_LANE_DEPTH` but NOT an alias of it (unlike
/// `UNRESERVED_OPERATION_BOUND` above) -- the two bound structurally
/// unrelated things (a `(scope, kind)` coalescing map vs. a duration-sample
/// ring), so a shared name would misdescribe one of them; see D12
/// (`.omo/plans/event-system-fixes.md`) on the hazard of two bounds
/// coincidentally sharing a value without a real relationship.
pub const INGRESS_LATENCY_RESERVOIR_CAPACITY: usize = 4_096;

/// Task 13, same amendment as [`INGRESS_LATENCY_RESERVOIR_CAPACITY`]: the
/// minimum number of recorded ingress-latency samples before
/// `ingress_p99_us` reports `Some` instead of `null` (frozen: "nullable
/// until 100 samples exist"). Reused as the version-bump milestone cadence
/// -- `RuntimeEventEngine::submit` bumps `EngineHealth`'s version once per
/// this many newly recorded samples, so a genuinely-changed p99 is never
/// permanently stranded behind an unrelated counter, without bumping (and
/// so gating health delivery) on every single submission; see
/// `.omo/evidence/event-system-fixes/task-13/p99-cadence-note.txt`.
pub const INGRESS_LATENCY_MIN_SAMPLES: usize = 100;

/// Read-only snapshot of the frozen bounds, exposed through engine health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub reservation_table_capacity: usize,
    pub state_transition_lane_depth: usize,
    pub diagnostic_lane_depth: usize,
    pub wake_list_depth: usize,
    pub replay_max_frames: usize,
    pub subscriber_lag_max_frames: usize,
    pub max_concurrent_subscribers: usize,
}

impl EngineConfig {
    pub const FROZEN: Self = Self {
        reservation_table_capacity: RESERVATION_TABLE_CAPACITY,
        state_transition_lane_depth: STATE_TRANSITION_LANE_DEPTH,
        diagnostic_lane_depth: DIAGNOSTIC_LANE_DEPTH,
        wake_list_depth: WAKE_LIST_DEPTH,
        replay_max_frames: REPLAY_MAX_FRAMES,
        subscriber_lag_max_frames: SUBSCRIBER_LAG_MAX_FRAMES,
        max_concurrent_subscribers: MAX_CONCURRENT_SUBSCRIBERS,
    };
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::FROZEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_derivation_matches_frozen_formula() {
        assert_eq!(REQUEST_ROOT_BOUND, 1_024);
        assert_eq!(CHILD_MULTIPLIER, 2);
        assert_eq!(LIFECYCLE_OPERATION_BOUND, 64);
        assert_eq!(RESERVATION_TABLE_CAPACITY, 3_136);
    }

    #[test]
    fn wake_list_depth_equals_reservation_table() {
        assert_eq!(WAKE_LIST_DEPTH, RESERVATION_TABLE_CAPACITY);
    }

    #[test]
    fn unreserved_operation_bound_matches_state_transition_lane_depth() {
        assert_eq!(UNRESERVED_OPERATION_BOUND, STATE_TRANSITION_LANE_DEPTH);
        assert_eq!(UNRESERVED_OPERATION_BOUND, 4_096);
    }

    #[test]
    fn total_operation_bound_derivation_matches_frozen_formula() {
        assert_eq!(TOTAL_OPERATION_BOUND, 7_232);
        assert_eq!(
            TOTAL_OPERATION_BOUND,
            RESERVATION_TABLE_CAPACITY + UNRESERVED_OPERATION_BOUND
        );
    }

    #[test]
    fn ingress_latency_bounds_are_frozen() {
        assert_eq!(INGRESS_LATENCY_RESERVOIR_CAPACITY, 4_096);
        assert_eq!(INGRESS_LATENCY_MIN_SAMPLES, 100);
        assert_eq!(
            INGRESS_LATENCY_RESERVOIR_CAPACITY, STATE_TRANSITION_LANE_DEPTH,
            "numerically equal by coincidence, not by shared meaning -- see the doc comment"
        );
    }

    #[test]
    fn frozen_config_exposes_all_bounds() {
        let config = EngineConfig::default();
        assert_eq!(config.reservation_table_capacity, 3_136);
        assert_eq!(config.state_transition_lane_depth, 4_096);
        assert_eq!(config.diagnostic_lane_depth, 2_048);
        assert_eq!(config.replay_max_frames, 4_096);
        assert_eq!(config.subscriber_lag_max_frames, 1_024);
        assert_eq!(config.max_concurrent_subscribers, 32);
    }
}
