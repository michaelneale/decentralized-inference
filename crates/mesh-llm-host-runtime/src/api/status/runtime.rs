//! Runtime status payload types extracted from oversized status.rs.
//!
//! Contains daemon state derivation, capability flags, lifecycle instance payloads,
//! intent summaries, and activity policy status for backward-compatible API extension.

use serde::{Deserialize, Serialize};

// ─── Daemon State ──────────────────────────────────────────────────────────────

/// Derived runtime daemon state with exact precedence:
/// stopping > degraded > ready_serving > ready_proxying > ready_idle > starting.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    /// Shutdown has been requested; runtime is winding down.
    Stopping,
    /// A policy-relevant terminal failure or priority restoration failure exists.
    Degraded,
    /// Local model serving is active and healthy.
    ReadyServing,
    /// No local serving but at least one healthy remote/plugin route available.
    ReadyProxying,
    /// Listeners are ready but no serving or proxy routes available yet.
    ReadyIdle,
    /// Initial state; listeners not yet ready.
    Starting,
}

/// Derive daemon state with exact precedence order.
pub fn derive_daemon_state(
    shutdown_requested: bool,
    has_terminal_failure: bool,
    priority_degraded: bool,
    local_serving: bool,
    proxying: bool,
    listeners_ready: bool,
) -> DaemonState {
    if shutdown_requested {
        return DaemonState::Stopping;
    }
    if has_terminal_failure || priority_degraded {
        return DaemonState::Degraded;
    }
    if local_serving {
        return DaemonState::ReadyServing;
    }
    if proxying {
        return DaemonState::ReadyProxying;
    }
    if listeners_ready {
        return DaemonState::ReadyIdle;
    }
    DaemonState::Starting
}

// ─── Capability Flags ──────────────────────────────────────────────────────────

/// Coexistence capability booleans representing what the daemon can do right now.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RuntimeCapabilityFlags {
    /// Node has GPU or CPU capacity to serve models locally.
    pub worker_capable: bool,
    /// At least one local model instance is actively serving requests.
    pub local_serving: bool,
    /// Proxying requests to remote peers or plugin endpoints.
    pub proxying: bool,
    /// Plugin ingress endpoints are available for routing.
    pub plugin_ingress: bool,
    /// Accepting inference from local clients (loopback).
    pub accepting_local: bool,
    /// Accepting inference from remote mesh peers.
    pub accepting_remote: bool,
    /// Additive API capability (not a routing role, unlike the booleans
    /// above): advertises `GET /api/runtime/events/v1` when present. A UI
    /// or client must treat its absence, a 404/403, or an unsupported
    /// mandatory frame as "fall back to the legacy stream" rather than an
    /// error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_events: Option<RuntimeEventsCapability>,
}

/// The frozen `capabilities.runtime_events` object on both `GET /api/runtime`
/// and `GET /api/status`. `cursor` names the wire cursor grammar prefix
/// (`rt1:<process-instance-uuid>:<sequence>`), not a live cursor value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeEventsCapability {
    pub version: u8,
    pub endpoint: &'static str,
    pub cursor: &'static str,
}

/// The one frozen value ever advertised for this capability in v1.
pub const RUNTIME_EVENTS_CAPABILITY: RuntimeEventsCapability = RuntimeEventsCapability {
    version: 1,
    endpoint: "/api/runtime/events/v1",
    cursor: "rt1",
};

// ─── Lifecycle Instance Payloads ──────────────────────────────────────────────

/// Bounded lifecycle instance payload for API status output.
#[derive(Clone, Debug, Serialize)]
pub struct LifecycleInstancePayload {
    /// Stable instance identifier.
    pub instance_id: String,
    /// Model reference this instance serves.
    pub model_ref: String,
    /// Current lifecycle state.
    #[serde(rename = "state")]
    pub lifecycle_state: String,
}

// ─── Intent Summary ────────────────────────────────────────────────────────────

/// Source that generated an intent (privacy-safe string representation).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentSourceLabel {
    StartupConfig,
    Cli,
    ApiRequest,
    OwnerLifecycle,
    MeshDemand,
}

/// A single intent entry for the filtered intents API.
/// Shows model ref, source, and desired state only — no raw owner payloads.
#[derive(Clone, Debug, Serialize)]
pub struct IntentEntry {
    pub intent_id: String,
    /// Model reference or spec this intent targets.
    pub model_ref: String,
    /// Profile name (empty string if default).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub profile: String,
    /// Source that generated this intent.
    pub source: IntentSourceLabel,
    /// Desired state: "load" or "unload".
    pub desired_state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_target: Option<String>,
    pub persistence: String,
    pub created_at_secs: u64,
    pub updated_at_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Payload for GET /api/runtime/intents — capped, filtered intent list.
#[derive(Clone, Debug, Serialize)]
pub struct IntentListPayload {
    /// Intent entries (capped at 256).
    pub intents: Vec<IntentEntry>,
    /// Total unique model refs observed before capping.
    pub total_count: usize,
    /// Whether the response was truncated due to the cap.
    pub truncated: bool,
}

/// Summary counts and errors for intents exposed in /api/status runtime data.
#[derive(Clone, Debug, Default, Serialize)]
pub struct IntentSummary {
    /// Total number of durable/configured intent entries.
    pub durable_count: usize,
    /// Number of session-scoped (transient) intent entries.
    pub session_count: usize,
    /// Recent error messages from failed intents (capped at 512 bytes total).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recent_errors: Vec<String>,
}

// ─── Activity Policy Status ────────────────────────────────────────────────────

/// Coarse detector category for activity status output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorCategory {
    /// Detector reports active usage.
    Active,
    /// Detector reports idle state.
    Idle,
    /// Detector is unavailable or not configured.
    #[default]
    Unavailable,
}

/// Manual override mode for activity policy.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityOverrideMode {
    /// Automatic mode — detector-driven decisions.
    #[default]
    Auto,
    /// Forced active — apply the configured capacity-yielding response.
    Active,
    /// Forced idle — accept inference normally.
    Idle,
}

/// Effective activity policy state for API output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityPolicyStateLabel {
    #[default]
    Accepting,
    AcceptingDeprioritized,
    RemotePaused,
    AllPaused,
}

/// Activity policy status for /api/runtime/activity output.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ActivityPolicyStatus {
    /// Current effective state of the activity policy.
    pub effective_state: ActivityPolicyStateLabel,
    /// Current override mode (auto = detector-driven).
    pub override_mode: ActivityOverrideMode,
    /// Coarse detector category only — no raw detector details or timestamps.
    pub detector_category: DetectorCategory,
}

// ─── Intent Source Conversion ──────────────────────────────────────────────────

impl From<crate::runtime::IntentSource> for IntentSourceLabel {
    fn from(source: crate::runtime::IntentSource) -> Self {
        match source {
            crate::runtime::IntentSource::StartupConfig => IntentSourceLabel::StartupConfig,
            crate::runtime::IntentSource::LocalCli => IntentSourceLabel::Cli,
            crate::runtime::IntentSource::ApiLoad | crate::runtime::IntentSource::ApiUnload => {
                IntentSourceLabel::ApiRequest
            }
            crate::runtime::IntentSource::OwnerLoad
            | crate::runtime::IntentSource::OwnerUnload
            | crate::runtime::IntentSource::OwnerEnsure
            | crate::runtime::IntentSource::OwnerDrain => IntentSourceLabel::OwnerLifecycle,
            crate::runtime::IntentSource::MeshDemand => IntentSourceLabel::MeshDemand,
        }
    }
}

// ─── Activity Policy State Conversion ──────────────────────────────────────────

impl From<crate::runtime::activity_policy::ActivityPolicyState> for ActivityPolicyStateLabel {
    fn from(state: crate::runtime::activity_policy::ActivityPolicyState) -> Self {
        match state {
            crate::runtime::activity_policy::ActivityPolicyState::Accepting => {
                ActivityPolicyStateLabel::Accepting
            }
            crate::runtime::activity_policy::ActivityPolicyState::AcceptingDeprioritized => {
                ActivityPolicyStateLabel::AcceptingDeprioritized
            }
            crate::runtime::activity_policy::ActivityPolicyState::RemotePaused => {
                ActivityPolicyStateLabel::RemotePaused
            }
            crate::runtime::activity_policy::ActivityPolicyState::AllPaused => {
                ActivityPolicyStateLabel::AllPaused
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Daemon State Derivation Matrix ──────────────────────────────────────

    #[test]
    fn daemon_state_stopping_has_highest_precedence() {
        assert_eq!(
            derive_daemon_state(true, true, true, true, true, true),
            DaemonState::Stopping
        );
        assert_eq!(
            derive_daemon_state(true, false, false, false, false, false),
            DaemonState::Stopping
        );
    }

    #[test]
    fn daemon_state_degraded_when_terminal_failure() {
        assert_eq!(
            derive_daemon_state(false, true, false, true, true, true),
            DaemonState::Degraded
        );
    }

    #[test]
    fn daemon_state_degraded_when_priority_degraded() {
        assert_eq!(
            derive_daemon_state(false, false, true, true, true, true),
            DaemonState::Degraded
        );
    }

    #[test]
    fn daemon_state_ready_serving_when_local_serving() {
        assert_eq!(
            derive_daemon_state(false, false, false, true, true, true),
            DaemonState::ReadyServing
        );
    }

    #[test]
    fn daemon_state_ready_proxying_when_only_proxying() {
        assert_eq!(
            derive_daemon_state(false, false, false, false, true, true),
            DaemonState::ReadyProxying
        );
    }

    #[test]
    fn daemon_state_ready_idle_when_listeners_ready_only() {
        assert_eq!(
            derive_daemon_state(false, false, false, false, false, true),
            DaemonState::ReadyIdle
        );
    }

    #[test]
    fn daemon_state_starting_when_nothing_ready() {
        assert_eq!(
            derive_daemon_state(false, false, false, false, false, false),
            DaemonState::Starting
        );
    }

    #[test]
    fn daemon_state_precedence_order_complete() {
        // stopping > degraded (failure) > degraded (priority) > ready_serving > ready_proxying > ready_idle > starting
        assert_eq!(
            derive_daemon_state(true, false, false, true, true, true),
            DaemonState::Stopping
        );
        assert_eq!(
            derive_daemon_state(false, true, false, true, true, true),
            DaemonState::Degraded
        );
        assert_eq!(
            derive_daemon_state(false, false, true, true, true, true),
            DaemonState::Degraded
        );
        assert_eq!(
            derive_daemon_state(false, false, false, true, true, true),
            DaemonState::ReadyServing
        );
        assert_eq!(
            derive_daemon_state(false, false, false, false, true, true),
            DaemonState::ReadyProxying
        );
        assert_eq!(
            derive_daemon_state(false, false, false, false, false, true),
            DaemonState::ReadyIdle
        );
        assert_eq!(
            derive_daemon_state(false, false, false, false, false, false),
            DaemonState::Starting
        );
    }

    // ─── Capability Flags Defaults ──────────────────────────────────────────

    #[test]
    fn capability_flags_default_to_false() {
        let flags = RuntimeCapabilityFlags::default();
        assert!(!flags.worker_capable);
        assert!(!flags.local_serving);
        assert!(!flags.proxying);
        assert!(!flags.plugin_ingress);
        assert!(!flags.accepting_local);
        assert!(!flags.accepting_remote);
    }

    // ─── Intent Summary Defaults ────────────────────────────────────────────

    #[test]
    fn intent_summary_defaults_to_zero() {
        let summary = IntentSummary::default();
        assert_eq!(summary.durable_count, 0);
        assert_eq!(summary.session_count, 0);
        assert!(summary.recent_errors.is_empty());
    }

    // ─── Activity Policy Status Defaults ────────────────────────────────────

    #[test]
    fn activity_policy_status_defaults() {
        let status = ActivityPolicyStatus::default();
        assert_eq!(status.override_mode, ActivityOverrideMode::Auto);
        assert_eq!(status.detector_category, DetectorCategory::Unavailable);
    }
}
