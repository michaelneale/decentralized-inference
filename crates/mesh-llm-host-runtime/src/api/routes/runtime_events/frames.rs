//! Wire payload types and SSE byte encoding for the v1 stream.
//!
//! Projection types are explicit `Serialize` structs — no `RuntimeFact`
//! ever derives `Serialize` for the wire (see `event_projection`, which
//! hand-projects only fields drawn from `EVENT_PROJECTION_ALLOWLIST`).
//!
//! The host engine's `ReducerSnapshot` retains bounded per-category domain
//! state (task 6, `.omo/plans/event-system-fixes.md`), projected by
//! `crate::runtime_event_api::state_projection` into
//! `models`/`stages`/`sessions`/`requests`/`devices`/`cache`.
//!
//! Task 7 (`.omo/plans/event-system-fixes.md`) puts operation identity and
//! the inventory's base keys on the wire: `ReplayFrame::scope` (already
//! carried by the reducer/wake pipeline since task 4/5) is projected as
//! `operation` on every `runtime_event`, and producer metadata now travels on
//! each `FamilyFact`. The projection preserves the typed producer/severity and
//! the approved timestamp/sequence fields while excluding the native detail
//! envelope and opaque identifiers.

use mesh_llm_runtime_event_contracts::{
    NumericValue, OperationScope, Outcome, ProcessInstanceId, ProducerSource, ProgressUnit,
    ReasonCode, RuntimeFact, Severity,
};
use serde::Serialize;

use crate::runtime_events::config::EngineConfig;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::health::EngineHealthSnapshot;
use crate::runtime_events::reducer::ReducerSnapshot;
use crate::runtime_events::replay::ReplayFrame;

use crate::runtime_event_api::state_projection;

use super::cursor::Cursor;
use super::recovery::{Gap, GapReason};

/// The full per-event-kind projected JSON key allowlist (deny-by-default):
/// every key `event_projection` may ever emit, drawn from the task-1/3
/// inventory's `projected_event_keys`. `EventProjection`'s own `Serialize`
/// output is asserted a subset of this set by
/// `runtime_event_api_tests::event_projection_keys_are_a_subset_of_the_allowlist_for_every_submitted_kind`.
#[cfg(test)]
pub(super) const EVENT_PROJECTION_ALLOWLIST: &[&str] = &[
    "category",
    "kind",
    "producer",
    "severity",
    "wall_clock_unix_ns",
    "process_monotonic_ns",
    "native_monotonic_ns",
    "native_sequence",
    "summary",
    "scope",
    "state",
    "progress",
    "outcome",
    "reason_code",
    "duration_ms",
    "numeric_summaries",
    "operation",
];

#[cfg(test)]
pub(super) const REQUIRED_ENVELOPE_KEYS: &[&str] = &[
    "version",
    "cursor",
    "process_instance_id",
    "sequence",
    "rebuild_generation",
];

#[cfg(test)]
pub(super) const STATE_TOP_LEVEL_KEYS: &[&str] = &[
    "node", "models", "stages", "sessions", "requests", "devices", "cache",
];

#[derive(Debug, Serialize)]
pub(super) struct Envelope<T> {
    pub(super) version: u8,
    pub(super) cursor: String,
    pub(super) process_instance_id: String,
    pub(super) sequence: u64,
    pub(super) rebuild_generation: u64,
    #[serde(flatten)]
    pub(super) body: T,
}

fn envelope<T>(cursor: Cursor, rebuild_generation: u64, body: T) -> Envelope<T> {
    Envelope {
        version: 1,
        cursor: cursor.encode(),
        process_instance_id: cursor.process_instance.as_uuid().to_string(),
        sequence: cursor.sequence,
        rebuild_generation,
        body,
    }
}

fn encode<T: Serialize>(event: &'static str, cursor: Cursor, payload: &Envelope<T>) -> String {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    format!("id: {}\nevent: {event}\ndata: {json}\n\n", cursor.encode())
}

pub(super) const KEEPALIVE_FRAME: &str = ": keepalive\n\n";

// ─── runtime_state ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct StateBody {
    pub(super) state: state_projection::StateProjection,
}

#[cfg(test)]
pub(super) fn state_frame(engine: &RuntimeEventEngine, cursor: Cursor) -> String {
    let snapshot = engine.reducer_snapshot();
    state_frame_from_snapshot(&snapshot, cursor)
}

pub(super) fn state_frame_from_snapshot(snapshot: &ReducerSnapshot, cursor: Cursor) -> String {
    let rebuild_generation = snapshot.rebuild_generation;
    let payload = envelope(
        cursor,
        rebuild_generation,
        StateBody {
            state: state_projection::build_from_snapshot(snapshot),
        },
    );
    encode("runtime_state", cursor, &payload)
}

// ─── runtime_health ─────────────────────────────────────────────────────

/// Every value of the frozen engine-bounds table (`runtime_events::config`),
/// task 8's `bounds` wire addition -- makes the certification bounds
/// visible without cross-referencing `config.rs` out of band.
#[derive(Debug, Serialize)]
pub(super) struct HealthBoundsProjection {
    pub(super) reservation_table_capacity: usize,
    pub(super) state_transition_lane_depth: usize,
    pub(super) diagnostic_lane_depth: usize,
    pub(super) wake_list_depth: usize,
    pub(super) replay_max_frames: usize,
    pub(super) subscriber_lag_max_frames: usize,
    pub(super) max_concurrent_subscribers: usize,
}

impl From<EngineConfig> for HealthBoundsProjection {
    fn from(config: EngineConfig) -> Self {
        Self {
            reservation_table_capacity: config.reservation_table_capacity,
            state_transition_lane_depth: config.state_transition_lane_depth,
            diagnostic_lane_depth: config.diagnostic_lane_depth,
            wake_list_depth: config.wake_list_depth,
            replay_max_frames: config.replay_max_frames,
            subscriber_lag_max_frames: config.subscriber_lag_max_frames,
            max_concurrent_subscribers: config.max_concurrent_subscribers,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct HealthProjection {
    pub(super) version: u64,
    pub(super) rebuild_generation: u64,
    pub(super) reservation_exhausted: u64,
    pub(super) terminal_delivery_failed: u64,
    pub(super) dropped_progress: u64,
    pub(super) dropped_diagnostic: u64,
    pub(super) replay_evicted: u64,
    pub(super) subscriber_disconnected: u64,
    pub(super) shutdown_degraded: u64,
    pub(super) reducer_rejected: u64,
    pub(super) state_transition_rejected: u64,
    pub(super) cancelled_reservation_rejected: u64,
    pub(super) state_degraded: bool,
    pub(super) rebuild_required: bool,
    pub(super) bounds: HealthBoundsProjection,
    /// Task 13 (`.omo/plans/event-system-fixes.md`) populates this from the
    /// engine's ingress-duration reservoir once 100 samples exist; task 8
    /// lands the wire field as always-nullable so the shape is stable
    /// before that reservoir exists.
    pub(super) ingress_p99_us: Option<u64>,
}

impl From<EngineHealthSnapshot> for HealthProjection {
    fn from(snapshot: EngineHealthSnapshot) -> Self {
        Self {
            version: snapshot.version,
            rebuild_generation: snapshot.rebuild_generation,
            reservation_exhausted: snapshot.reservation_exhausted,
            terminal_delivery_failed: snapshot.terminal_delivery_failed,
            dropped_progress: snapshot.dropped_progress,
            dropped_diagnostic: snapshot.dropped_diagnostic,
            replay_evicted: snapshot.replay_evicted,
            subscriber_disconnected: snapshot.subscriber_disconnected,
            shutdown_degraded: snapshot.shutdown_degraded,
            reducer_rejected: snapshot.reducer_rejected,
            state_transition_rejected: snapshot.state_transition_rejected,
            cancelled_reservation_rejected: snapshot.cancelled_reservation_rejected,
            state_degraded: snapshot.state_degraded,
            rebuild_required: snapshot.rebuild_required,
            bounds: EngineConfig::FROZEN.into(),
            ingress_p99_us: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct HealthBody {
    pub(super) health: HealthProjection,
}

#[cfg(test)]
pub(super) fn health_frame(engine: &RuntimeEventEngine, cursor: Cursor) -> String {
    let snapshot = engine.health().snapshot();
    let ingress_p99_us = engine.ingress_p99_us();
    health_frame_from_snapshot(snapshot, ingress_p99_us, cursor)
}

pub(super) fn health_frame_from_snapshot(
    snapshot: EngineHealthSnapshot,
    ingress_p99_us: Option<u64>,
    cursor: Cursor,
) -> String {
    let mut health: HealthProjection = snapshot.into();
    health.ingress_p99_us = ingress_p99_us;
    let payload = envelope(cursor, snapshot.rebuild_generation, HealthBody { health });
    encode("runtime_health", cursor, &payload)
}

// ─── runtime_event ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct ScopeProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) topology_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stage_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stage_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) device_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct StateTransitionProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) previous: Option<String>,
    pub(super) current: String,
}

#[derive(Debug, Serialize)]
pub(super) struct ProgressProjection {
    pub(super) current: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) total: Option<u64>,
    pub(super) unit: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct NumericSummaryProjection {
    pub(super) key: String,
    pub(super) value: serde_json::Value,
}

/// Root/child operation identity, projected from `ReplayFrame::scope`.
/// `root` is the same UUID text as the logging `request_id` for a
/// request-rooted operation (see `network::openai::runtime_events`'s
/// byte-equality rule); `child` is present only for a scope reserved under
/// `OperationScope::Child`, which is what makes a root's terminal
/// distinguishable on the wire from a backend child's terminal of the same
/// event kind.
#[derive(Debug, Serialize)]
pub(super) struct OperationProjection {
    pub(super) root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) child: Option<String>,
}

fn operation_projection(scope: OperationScope) -> OperationProjection {
    OperationProjection {
        root: scope.root().to_string(),
        child: scope.child().map(|child| child.to_string()),
    }
}

#[derive(Debug, Serialize)]
pub(super) struct EventProjection {
    pub(super) category: &'static str,
    pub(super) kind: &'static str,
    pub(super) producer: &'static str,
    pub(super) severity: &'static str,
    /// Safe, unit-named provenance values. Native source detail, pointers,
    /// and opaque identity fields stay inside the producer contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) wall_clock_unix_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) process_monotonic_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) native_monotonic_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) native_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) operation: Option<OperationProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) scope: Option<ScopeProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) state: Option<StateTransitionProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) progress: Option<ProgressProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(super) numeric_summaries: Vec<NumericSummaryProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) summary: Option<String>,
}

pub(super) fn category(fact: &RuntimeFact) -> &'static str {
    match fact {
        RuntimeFact::NativeRuntime(_) => "native_runtime",
        RuntimeFact::ModelPreparation(_) => "model_preparation",
        RuntimeFact::ModelLoading(_) => "model_loading",
        RuntimeFact::ModelAvailability(_) => "model_availability",
        RuntimeFact::ModelUnloading(_) => "model_unloading",
        RuntimeFact::StageTopology(_) => "stage_topology",
        RuntimeFact::Session(_) => "session",
        RuntimeFact::Request(_) => "request",
        RuntimeFact::Prefill(_) => "prefill",
        RuntimeFact::Generation(_) => "generation",
        RuntimeFact::KvRuntimeState(_) => "kv_runtime_state",
        RuntimeFact::ResourceHealth(_) => "resource_health",
        RuntimeFact::Diagnostic(_) => "diagnostic",
        RuntimeFact::NodeAvailability(_) => "node_availability",
        RuntimeFact::EventSystemHealth(_) => "event_system_health",
    }
}

fn outcome_str(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::Failure => "failure",
        Outcome::Rejected => "rejected",
        Outcome::Cancelled => "cancelled",
        Outcome::Unknown => "unknown",
    }
}

fn reason_code_str(reason: &ReasonCode) -> String {
    match reason {
        ReasonCode::InvalidConfiguration => "invalid_configuration".to_string(),
        ReasonCode::UnsupportedCapability => "unsupported_capability".to_string(),
        ReasonCode::MissingArtifact => "missing_artifact".to_string(),
        ReasonCode::ArtifactIoFailure => "artifact_io_failure".to_string(),
        ReasonCode::ModelFormatOrLoadFailure => "model_format_or_load_failure".to_string(),
        ReasonCode::BackendInitializationFailure => "backend_initialization_failure".to_string(),
        ReasonCode::DeviceUnavailable => "device_unavailable".to_string(),
        ReasonCode::ResourceAllocationFailure => "resource_allocation_failure".to_string(),
        ReasonCode::OutOfMemory => "out_of_memory".to_string(),
        ReasonCode::ContextExhausted => "context_exhausted".to_string(),
        ReasonCode::StageUnavailable => "stage_unavailable".to_string(),
        ReasonCode::Timeout => "timeout".to_string(),
        ReasonCode::Cancellation => "cancellation".to_string(),
        ReasonCode::ProcessCrash => "process_crash".to_string(),
        ReasonCode::IncompatibleAbiOrFeatureSet => "incompatible_abi_or_feature_set".to_string(),
        ReasonCode::InternalRuntimeFailure => "internal_runtime_failure".to_string(),
        ReasonCode::UnknownFailure => "unknown_failure".to_string(),
        ReasonCode::TerminalNotDelivered => "terminal_not_delivered".to_string(),
        ReasonCode::ReservationExhausted => "reservation_exhausted".to_string(),
        ReasonCode::Unknown(code) => code.as_str().to_string(),
    }
}

/// Prefer the producer metadata carried with the fact. The family-derived
/// fallback remains for older callers that construct a fact without metadata.
fn producer_str(fact: &RuntimeFact) -> &'static str {
    if let Some(metadata) = fact.metadata() {
        return match metadata.producer {
            ProducerSource::Native => "native",
            ProducerSource::Rust => "rust",
            ProducerSource::Reconciled => "reconciled",
        };
    }
    match fact {
        RuntimeFact::ResourceHealth(_) | RuntimeFact::Diagnostic(_) => "native",
        _ => "rust",
    }
}

/// Prefer the producer metadata carried with the fact. The outcome-derived
/// fallback keeps older metadata-free facts deterministic.
fn severity_str(fact: &RuntimeFact) -> &'static str {
    if let Some(metadata) = fact.metadata() {
        return match metadata.severity {
            Severity::Trace => "trace",
            Severity::Debug => "debug",
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
            Severity::Fatal => "fatal",
        };
    }
    match fact.data().outcome {
        Some(Outcome::Failure) => "error",
        Some(Outcome::Rejected | Outcome::Cancelled | Outcome::Unknown) => "warning",
        Some(Outcome::Success) => "info",
        None if category(fact) == "diagnostic" => "warning",
        None => "info",
    }
}

fn duration_nanos(value: std::time::Duration) -> u64 {
    u64::try_from(value.as_nanos()).unwrap_or(u64::MAX)
}

/// Project only the typed, bounded provenance fields approved for the public
/// event shape. The native source envelope contains additional ABI/detail
/// data; none of that is serialized here.
fn provenance_fields(fact: &RuntimeFact) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let Some(metadata) = fact.metadata() else {
        return (None, None, None, None);
    };
    let native_source = metadata.native_source.as_ref();
    let native_sequence = native_source.map(|source| source.sequence).or_else(|| {
        metadata
            .native_sequence
            .map(|observation| observation.sequence)
    });
    (
        metadata.wall_clock_unix_ns,
        metadata.process_monotonic_time.map(duration_nanos),
        native_source.map(|source| source.timestamp_mono_ns),
        native_sequence,
    )
}

fn progress_unit_str(unit: ProgressUnit) -> &'static str {
    match unit {
        ProgressUnit::None => "none",
        ProgressUnit::Bytes => "bytes",
        ProgressUnit::Items => "items",
        ProgressUnit::Tensors => "tensors",
        ProgressUnit::Steps => "steps",
        ProgressUnit::Tokens => "tokens",
    }
}

fn numeric_value_json(value: NumericValue) -> serde_json::Value {
    match value {
        NumericValue::Unsigned(value) => serde_json::Value::from(value),
        NumericValue::Signed(value) => serde_json::Value::from(value),
        NumericValue::Floating(value) => serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
    }
}

pub(super) fn event_projection(fact: &RuntimeFact, scope: OperationScope) -> EventProjection {
    let data = fact.data();
    let (wall_clock_unix_ns, process_monotonic_ns, native_monotonic_ns, native_sequence) =
        provenance_fields(fact);
    let fact_scope = &data.scope;
    let scope_projection = if fact_scope.model_id.is_some()
        || fact_scope.topology_id.is_some()
        || fact_scope.stage.is_some()
        || fact_scope.session_id.is_some()
        || fact_scope.request_id.is_some()
        || fact_scope.device_id.is_some()
    {
        Some(ScopeProjection {
            model_id: fact_scope
                .model_id
                .as_ref()
                .map(|id| id.as_str().to_string()),
            topology_id: fact_scope
                .topology_id
                .as_ref()
                .map(|id| id.as_str().to_string()),
            stage_id: fact_scope
                .stage
                .as_ref()
                .map(|stage| stage.id.as_str().to_string()),
            stage_index: fact_scope.stage.as_ref().map(|stage| stage.index),
            session_id: fact_scope
                .session_id
                .as_ref()
                .map(|id| id.as_str().to_string()),
            request_id: fact_scope
                .request_id
                .as_ref()
                .map(|id| id.as_str().to_string()),
            device_id: fact_scope
                .device_id
                .as_ref()
                .map(|id| id.as_str().to_string()),
        })
    } else {
        None
    };

    EventProjection {
        category: category(fact),
        kind: fact.kind_id(),
        producer: producer_str(fact),
        severity: severity_str(fact),
        wall_clock_unix_ns,
        process_monotonic_ns,
        native_monotonic_ns,
        native_sequence,
        operation: Some(operation_projection(scope)),
        scope: scope_projection,
        state: data
            .state
            .as_ref()
            .map(|transition| StateTransitionProjection {
                previous: transition
                    .previous
                    .as_ref()
                    .map(|state| state.as_str().to_string()),
                current: transition.current.as_str().to_string(),
            }),
        progress: data.progress.map(|progress| ProgressProjection {
            current: progress.current,
            total: progress.total,
            unit: progress_unit_str(progress.unit),
        }),
        outcome: data.outcome.map(outcome_str),
        reason_code: data.reason.as_ref().map(reason_code_str),
        duration_ms: data
            .duration
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)),
        numeric_summaries: data
            .numeric_summaries
            .as_slice()
            .iter()
            .map(|summary| NumericSummaryProjection {
                key: summary.key.as_str().to_string(),
                value: numeric_value_json(summary.value),
            })
            .collect(),
        summary: data
            .summary
            .as_ref()
            .map(|summary| summary.as_str().to_string()),
    }
}

#[derive(Debug, Serialize)]
pub(super) struct EventBody {
    pub(super) event: EventProjection,
}

/// `pub(crate)` (task 9, `.omo/plans/event-system-fixes.md`, defect D11):
/// `runtime_events::engine::drain::apply_and_publish_fact` calls this to
/// pre-serialize a frame's `runtime_event` wire bytes exactly once, at
/// push, storing them on `ReplayFrame::wire_bytes` so every later delivery
/// (live subscriber or replay-window catch-up) writes the same bytes
/// without re-running `event_projection` + JSON encoding. That is the ONE
/// call site granted outside `api::routes::runtime_events` for this
/// function; the encoder's output is unchanged (byte-exactness is pinned
/// by `runtime_event_api_tests::sample_frames_fixture_is_byte_exact_for_every_frame_type`,
/// which calls `event_frame_at` directly and is unaffected by this
/// visibility change).
pub(crate) fn event_frame(engine: &RuntimeEventEngine, frame: &ReplayFrame) -> String {
    event_frame_at(engine.process_instance(), frame)
}

/// Instance-parameterized core of [`event_frame`], split out so a
/// byte-exact fixture test can supply a fixed `ProcessInstanceId` instead
/// of the engine's randomly-minted one (see
/// `runtime_event_api_tests::sample_frames_fixture_is_byte_exact_for_every_frame_type`).
pub(super) fn event_frame_at(instance: ProcessInstanceId, frame: &ReplayFrame) -> String {
    let cursor = Cursor::new(instance, frame.sequence.get());
    let payload = envelope(
        cursor,
        frame.rebuild_generation,
        EventBody {
            event: event_projection(&frame.fact, frame.scope),
        },
    );
    encode("runtime_event", cursor, &payload)
}

// ─── runtime_replay_gap ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct ReplayGapBody {
    pub(super) requested_cursor: String,
    pub(super) reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) oldest_available_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) latest_cursor: Option<String>,
}

impl GapReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::StaleInstance => "stale_instance",
            Self::Evicted => "evicted",
        }
    }
}

/// Instance/generation-parameterized replay-gap frame encoder, split out for
/// the same byte-exact-fixture reason as [`event_frame_at`].
pub(super) fn replay_gap_frame_at(
    instance: ProcessInstanceId,
    rebuild_generation: u64,
    gap: &Gap,
) -> String {
    let current_cursor = Cursor::new(instance, gap.latest.or(gap.oldest_available).unwrap_or(0));
    let payload = envelope(
        current_cursor,
        rebuild_generation,
        ReplayGapBody {
            requested_cursor: gap.requested.encode(),
            reason: gap.reason.as_str(),
            oldest_available_cursor: gap
                .oldest_available
                .map(|sequence| Cursor::new(instance, sequence).encode()),
            latest_cursor: gap
                .latest
                .map(|sequence| Cursor::new(instance, sequence).encode()),
        },
    );
    encode("runtime_replay_gap", current_cursor, &payload)
}
