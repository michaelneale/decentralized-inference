//! Pure rebuild/replay recovery policy.
//!
//! Rebuild continues the ingress sequence (the wake list's counter is
//! untouched; nothing here resets it) and evicts replay (engine-owned, see
//! `engine::drain::rebuild`). The reducer's own job is: bump
//! `rebuild_generation` monotonically, and degrade every operation that
//! never settled before the crash/restart while preserving its last-valid
//! fields, rather than discarding them.

use std::sync::Arc;

use super::state::ReducerSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildError {
    /// `generation` was not strictly greater than the snapshot's current
    /// generation: a non-monotonic rebuild is an invariant violation and is
    /// refused rather than silently corrupting the generation sequence.
    NonMonotonicGeneration,
}

#[derive(Debug, Clone)]
pub enum RebuildOutcome {
    Rebuilt(Arc<ReducerSnapshot>),
    Failed(RebuildError),
}

/// Attempt a rebuild to `generation`. Success degrades every unsettled
/// operation and advances the generation; failure leaves `snapshot`
/// unchanged so the caller's `Arc` still points at a coherent state.
#[must_use]
pub fn rebuild(snapshot: &Arc<ReducerSnapshot>, generation: u64) -> RebuildOutcome {
    if generation <= snapshot.rebuild_generation {
        return RebuildOutcome::Failed(RebuildError::NonMonotonicGeneration);
    }
    let degraded = snapshot.degrade_unsettled();
    let advanced = degraded.with_generation(generation);
    RebuildOutcome::Rebuilt(Arc::new(advanced))
}
