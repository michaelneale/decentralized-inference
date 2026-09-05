//! Transactional reducer: ordering, degradation, and rebuild policy.
//!
//! Pure by construction — `apply` and `rebuild` are `(snapshot, input) ->
//! new snapshot | rejection` functions with no I/O and no locking. The
//! owning `Mutex<Arc<ReducerSnapshot>>` (see `engine::drain`) is the ONLY
//! lock this subsystem introduces, and it is disjoint from the reservation
//! table's per-slot locks used by native ingress, satisfying the
//! lock-separation requirement structurally rather than by convention.
//!
//! Publication is post-acceptance: a caller must only append a replay frame
//! and fan out to subscribers when `apply` returns `Applied`. A `Rejected`
//! input is invisible on the stream by construction — there is no code path
//! from a `Rejected` outcome to `ReplayBuffer::push` /
//! `SubscriberRegistry::publish`.

mod apply;
mod domain;
mod rebuild;
mod state;

#[cfg(test)]
mod tests;

pub use apply::{ReduceOutcome, ReducerInput, apply, evict};
pub use domain::{
    CacheDomainState, DeviceDomainState, DomainState, ModelDomainState, RequestDomainState,
    SessionRecentEntry, StageDomainState,
};
pub use rebuild::{RebuildError, RebuildOutcome, rebuild};
pub use state::{OperationState, ReducerSnapshot, RejectReason};
