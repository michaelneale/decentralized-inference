//! Privacy-safe `RuntimeFact` -> `OutputEvent` projection.
//!
//! Deny-by-default: every fragment this module may ever fold into a
//! presentation message is named in [`PROJECTED_FRAGMENT_KEYS`], mirroring
//! the API layer's own `EVENT_PROJECTION_ALLOWLIST`
//! (`api/routes/runtime_events/frames.rs`) for the same underlying
//! `RuntimeFact`. Nothing outside `FactData`'s closed, typed field set --
//! and nothing outside this key list -- ever reaches presentation output;
//! there is no path from a producer's raw content to this projection.

use mesh_llm_events::OutputEvent;
use mesh_llm_runtime_event_contracts::{
    NumericValue, Outcome, Progress, ProgressUnit, ReasonCode, RuntimeFact, ScopeIdentities,
    StateTransition,
};

use crate::runtime_events::config::EngineConfig;
use crate::runtime_events::health::EngineHealthSnapshot;

/// Every fragment key [`projected_fragments`] may ever emit. Used by the
/// privacy tests to prove deny-by-default: a fragment key outside this list
/// is a defect, not a missed test case.
#[cfg(test)]
pub(super) const PROJECTED_FRAGMENT_KEYS: &[&str] = &[
    "category",
    "kind",
    "model_id",
    "topology_id",
    "stage_id",
    "session_id",
    "request_id",
    "device_id",
    "state",
    "progress",
    "outcome",
    "reason_code",
    "duration_ms",
    "numeric_summary",
    "summary",
];

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

fn numeric_value_text(value: NumericValue) -> String {
    match value {
        NumericValue::Unsigned(value) => value.to_string(),
        NumericValue::Signed(value) => value.to_string(),
        NumericValue::Floating(value) => value.to_string(),
    }
}

fn scope_fragments(scope: &ScopeIdentities, out: &mut Vec<(&'static str, String)>) {
    if let Some(model_id) = &scope.model_id {
        out.push(("model_id", model_id.as_str().to_string()));
    }
    if let Some(topology_id) = &scope.topology_id {
        out.push(("topology_id", topology_id.as_str().to_string()));
    }
    if let Some(stage) = &scope.stage {
        out.push(("stage_id", format!("{}#{}", stage.id.as_str(), stage.index)));
    }
    if let Some(session_id) = &scope.session_id {
        out.push(("session_id", session_id.as_str().to_string()));
    }
    if let Some(request_id) = &scope.request_id {
        out.push(("request_id", request_id.as_str().to_string()));
    }
    if let Some(device_id) = &scope.device_id {
        out.push(("device_id", device_id.as_str().to_string()));
    }
}

fn state_text(state: &StateTransition) -> String {
    match &state.previous {
        Some(previous) => format!("{}->{}", previous.as_str(), state.current.as_str()),
        None => state.current.as_str().to_string(),
    }
}

fn progress_text(progress: Progress) -> String {
    match progress.total {
        Some(total) => format!(
            "{}/{} {}",
            progress.current,
            total,
            progress_unit_str(progress.unit)
        ),
        None => format!("{} {}", progress.current, progress_unit_str(progress.unit)),
    }
}

/// Every `(key, value)` fragment this fact contributes to its presentation
/// message, in a fixed order. `category` and `kind` are always present;
/// every other fragment appears only when the fact actually carries it.
pub(super) fn projected_fragments(fact: &RuntimeFact) -> Vec<(&'static str, String)> {
    let data = fact.data();
    let mut fragments = vec![
        ("category", category(fact).to_string()),
        ("kind", fact.kind_id().to_string()),
    ];
    scope_fragments(&data.scope, &mut fragments);
    if let Some(state) = &data.state {
        fragments.push(("state", state_text(state)));
    }
    if let Some(progress) = data.progress {
        fragments.push(("progress", progress_text(progress)));
    }
    if let Some(outcome) = data.outcome {
        fragments.push(("outcome", outcome_str(outcome).to_string()));
    }
    if let Some(reason) = &data.reason {
        fragments.push(("reason_code", reason_code_str(reason)));
    }
    if let Some(duration) = data.duration {
        fragments.push(("duration_ms", duration.as_millis().to_string()));
    }
    for summary in data.numeric_summaries.as_slice() {
        fragments.push((
            "numeric_summary",
            format!(
                "{}={}",
                summary.key.as_str(),
                numeric_value_text(summary.value)
            ),
        ));
    }
    if let Some(summary) = &data.summary {
        fragments.push(("summary", summary.as_str().to_string()));
    }
    fragments
}

fn build_message(fact: &RuntimeFact) -> String {
    projected_fragments(fact)
        .into_iter()
        .map(|(key, value)| match key {
            "category" | "kind" => value,
            _ => format!("{key}={value}"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Project one accepted `RuntimeFact` into a privacy-safe `OutputEvent` for
/// operational log/TUI presentation. Both the JSON formatter and the pretty
/// TUI formatter render the SAME `OutputEvent::Info` value, so their output
/// is guaranteed to agree field-for-field (see `tests::parity`).
#[must_use]
pub fn fact_projection_event(fact: &RuntimeFact) -> OutputEvent {
    OutputEvent::Info {
        message: build_message(fact),
        context: Some(category(fact).to_string()),
    }
}

/// Project a coalesced engine-health snapshot into the same privacy-safe
/// shape -- the `event_system_health` log line. Health counters are
/// already bounded/aggregate by construction (`EngineHealthSnapshot`), so
/// nothing further needs allowlisting here. Task 8
/// (`.omo/plans/event-system-fixes.md`) adds `version` plus the same
/// `bounds`/`ingress_p99_us` additions the wire `runtime_health` frame's
/// `HealthProjection` carries, so the log line and the wire frame never
/// diverge in shape. `ingress_p99_us` is not part of `EngineHealthSnapshot`
/// itself (recording a sample must never bump `health_version` on every
/// submission -- see `EngineHealth::bump_for_ingress_latency_milestone`),
/// so task 13 (`.omo/plans/event-system-fixes.md`) threads it in as a
/// separate argument, sourced by the caller from
/// `RuntimeEventEngine::ingress_p99_us`.
#[must_use]
pub fn health_projection_event(
    snapshot: EngineHealthSnapshot,
    ingress_p99_us: Option<u64>,
) -> OutputEvent {
    let bounds = EngineConfig::FROZEN;
    OutputEvent::Info {
        message: format!(
            "version={} reservation_exhausted={} terminal_delivery_failed={} dropped_progress={} \
             dropped_diagnostic={} replay_evicted={} subscriber_disconnected={} \
             shutdown_degraded={} reducer_rejected={} state_transition_rejected={} \
             cancelled_reservation_rejected={} \
             state_degraded={} rebuild_required={} rebuild_generation={} \
             bounds.reservation_table_capacity={} bounds.state_transition_lane_depth={} \
             bounds.diagnostic_lane_depth={} bounds.wake_list_depth={} \
             bounds.replay_max_frames={} bounds.subscriber_lag_max_frames={} \
             bounds.max_concurrent_subscribers={} ingress_p99_us={}",
            snapshot.version,
            snapshot.reservation_exhausted,
            snapshot.terminal_delivery_failed,
            snapshot.dropped_progress,
            snapshot.dropped_diagnostic,
            snapshot.replay_evicted,
            snapshot.subscriber_disconnected,
            snapshot.shutdown_degraded,
            snapshot.reducer_rejected,
            snapshot.state_transition_rejected,
            snapshot.cancelled_reservation_rejected,
            snapshot.state_degraded,
            snapshot.rebuild_required,
            snapshot.rebuild_generation,
            bounds.reservation_table_capacity,
            bounds.state_transition_lane_depth,
            bounds.diagnostic_lane_depth,
            bounds.wake_list_depth,
            bounds.replay_max_frames,
            bounds.subscriber_lag_max_frames,
            bounds.max_concurrent_subscribers,
            ingress_p99_us.map_or_else(|| "null".to_string(), |value| value.to_string()),
        ),
        context: Some("event_system_health".to_string()),
    }
}
