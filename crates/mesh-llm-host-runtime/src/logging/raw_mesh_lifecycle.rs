//! Metadata-only lifecycle ownership for raw mesh ingress requests.
//!
//! This owner is shared with the embedded OpenAI observer so a request that
//! entered through raw mesh routing cannot gain a competing frontend parent.
//! Direct loopback requests still belong to the frontend observer because they
//! never claim this raw-ingress ownership.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use mesh_llm_events::logging::identifiers::RequestId;

#[cfg(test)]
use mesh_llm_events::logging::events::LifecycleEvent;

use super::openai_lifecycle::MAX_TRACKED_REQUESTS;
use super::{LifecycleGuard, LoggingService, RequestSummaryMetadata};

#[cfg(test)]
use super::TerminalOutcome;

mod event_emission;
mod ownership;
mod proxy_attempts;
mod remote_attribution;

pub(crate) use proxy_attempts::{ProxyAttemptFinish, RawMeshProxyAttempt};
pub(crate) use remote_attribution::{RawMeshRemoteAttributionLease, RawMeshRemoteSuppressionLease};

#[cfg(test)]
const MAX_RAW_MESH_LIFECYCLE_OWNERS: usize = MAX_TRACKED_REQUESTS;

#[derive(Default)]
pub(crate) struct RawMeshLifecycleOwners {
    coordination: Mutex<LifecycleCoordination>,
    next_token: AtomicU64,
}

#[derive(Default)]
struct LifecycleCoordination {
    owners: HashMap<RequestId, LifecycleOwnerEntry>,
    raw_owner_count: usize,
    remote_attributions: HashMap<RequestId, RemoteAttributionEntry>,
    remote_suppressions: HashMap<RequestId, RemoteSuppressionEntry>,
}

enum LifecycleOwnerEntry {
    Raw(RawMeshLifecycleEntry),
    Frontend,
}

struct RawMeshLifecycleEntry {
    guard: LifecycleGuard,
    token: u64,
    route_selected: bool,
    stream_started: bool,
    first_token_recorded: bool,
    stream_completed: bool,
    stream_error: bool,
}

struct RemoteSuppressionEntry {
    token: u64,
    leases: u32,
}

struct RemoteAttributionEntry {
    token: u64,
    leases: u32,
    metadata: RequestSummaryMetadata,
}

enum RemoteAttributionPlacement {
    Applied,
    Pending(u64),
}

pub(crate) enum FrontendAdmissionDecision {
    Rejected,
    Registered { evicted: Option<RequestId> },
}

pub(crate) struct RawMeshRequestLifecycle {
    service: Arc<LoggingService>,
    owners: Arc<RawMeshLifecycleOwners>,
    request_id: RequestId,
    token: u64,
    guard: LifecycleGuard,
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
