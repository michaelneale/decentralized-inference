//! Bounded, latest-value-per-operation progress coalescer.
//!
//! A `Progress`-class fact never reaches the presentation channel directly:
//! [`ProgressCoalescer::submit`] replaces the pending value for its
//! operation in place (never queues a second copy for the same operation),
//! and [`ProgressCoalescer::flush_at`] drains at most one fact per
//! operation, no more often than once per configured tick interval. The
//! pending map is bounded by construction: it holds at most one entry per
//! LIVE operation that has submitted progress since the last flush, itself
//! bounded by the reservation table's admission capacity
//! (`runtime_events::config::RESERVATION_TABLE_CAPACITY`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{OperationScope, RuntimeFact};

use crate::runtime_events::config::TUI_RENDER_TICK;

pub struct ProgressCoalescer {
    pending: Mutex<HashMap<OperationScope, RuntimeFact>>,
    last_flush: Mutex<Instant>,
    interval: Duration,
}

impl ProgressCoalescer {
    /// A coalescer ticking at the existing `PRETTY_TUI_REDRAW_INTERVAL`
    /// (33ms), matching the plan's frozen TUI render tick.
    #[must_use]
    pub fn new() -> Self {
        Self::with_interval(TUI_RENDER_TICK)
    }

    /// Same as [`Self::new`] with an explicit tick interval; tests use this
    /// to make tick-boundary behavior reachable without waiting 33ms.
    #[must_use]
    pub fn with_interval(interval: Duration) -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            last_flush: Mutex::new(Instant::now()),
            interval,
        }
    }

    /// Coalesce `fact` into the single pending slot for `scope`. A second
    /// submission for the same `scope` before the next flush overwrites the
    /// first rather than queuing alongside it.
    pub fn submit(&self, scope: OperationScope, fact: RuntimeFact) {
        self.lock_pending().insert(scope, fact);
    }

    /// Drain every pending fact if at least the configured interval has
    /// elapsed since the last flush; otherwise leaves the pending set
    /// untouched and returns empty. `now` is caller-supplied so tick timing
    /// is deterministically testable without sleeping on the wall clock.
    pub fn flush_at(&self, now: Instant) -> Vec<(OperationScope, RuntimeFact)> {
        let mut last_flush = self.lock_last_flush();
        if now.duration_since(*last_flush) < self.interval {
            return Vec::new();
        }
        *last_flush = now;
        drop(last_flush);
        self.lock_pending().drain().collect()
    }

    /// Test-only: the number of distinct operations currently holding an
    /// unflushed progress value.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.lock_pending().len()
    }

    fn lock_pending(&self) -> std::sync::MutexGuard<'_, HashMap<OperationScope, RuntimeFact>> {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn lock_last_flush(&self) -> std::sync::MutexGuard<'_, Instant> {
        self.last_flush
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for ProgressCoalescer {
    fn default() -> Self {
        Self::new()
    }
}
