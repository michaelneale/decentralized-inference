//! Bounded wake list: FIFO ingress-sequence order, sized to the reservation
//! table so a terminal write (one per slot) can never overflow it.
//!
//! `next_sequence` is the engine's ONE ingress-sequence counter (task 4,
//! `.omo/plans/event-system-fixes.md`): every `RuntimeEventEngine::submit`
//! call consumes it exactly once, regardless of delivery class or outcome
//! (`Accepted`, `Coalesced`, `Dropped*`, `Rejected*`,
//! `TerminalDeliveryFailed`) -- there
//! is no second counter anywhere in the engine. A terminal write consumes
//! it through [`Self::push_next`] (mint-and-enqueue as ONE atomic step, so
//! concurrent terminal writes to different scopes can never publish out of
//! sequence order); every other lane, and every terminal failure path,
//! consumes it through the bare [`Self::next_ingress_sequence`] mint and
//! records the value itself (state-transition lane, diagnostic queue,
//! progress slot -- see `engine/lanes.rs` and `engine/drain.rs`).
//! Sequence zero is reserved for the empty snapshot cursor; real ingress
//! starts at one.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::reservation::SlotHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeEntry {
    pub handle: SlotHandle,
    pub ingress_sequence: u64,
}

struct Inner {
    next_sequence: u64,
    entries: VecDeque<WakeEntry>,
}

#[derive(Debug)]
pub struct WakeList {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("next_sequence", &self.next_sequence)
            .field("entries_len", &self.entries.len())
            .finish()
    }
}

impl WakeList {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                next_sequence: 1,
                entries: VecDeque::with_capacity(crate::runtime_events::config::WAKE_LIST_DEPTH),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Mint the next process-local ingress sequence without recording it.
    /// Only useful when the caller does not intend to push a wake entry for
    /// it (e.g. `rebuild`'s monotonicity test); a real terminal submission
    /// should use [`Self::push_next`] so sequence assignment and queue
    /// position stay atomic with each other.
    pub fn next_ingress_sequence(&self) -> u64 {
        let mut inner = self.lock();
        let sequence = inner.next_sequence;
        inner.next_sequence += 1;
        sequence
    }

    /// Assign the next ingress sequence and push `handle` for it as one
    /// atomic step, so concurrent callers can never observe push order
    /// diverge from sequence-assignment order.
    pub fn push_next(&self, handle: SlotHandle) -> u64 {
        let mut inner = self.lock();
        let sequence = inner.next_sequence;
        inner.next_sequence += 1;
        inner.entries.push_back(WakeEntry {
            handle,
            ingress_sequence: sequence,
        });
        sequence
    }

    /// Read the next sequence that would be minted, without consuming it.
    /// Pure observation for the API layer's cursor classification (task
    /// 13): unlike [`Self::next_ingress_sequence`], this never advances the
    /// counter, so calling it repeatedly is side-effect-free.
    #[must_use]
    pub fn peek_next_sequence(&self) -> u64 {
        self.lock().next_sequence
    }

    /// Drain every entry currently queued, oldest first.
    pub fn drain_all(&self) -> Vec<WakeEntry> {
        self.lock().entries.drain(..).collect()
    }

    /// Drain at most `max` entries, oldest first, leaving the rest queued.
    pub fn drain_up_to(&self, max: usize) -> Vec<WakeEntry> {
        let mut inner = self.lock();
        let take = max.min(inner.entries.len());
        inner.entries.drain(..take).collect()
    }

    /// Drain every terminal entry before the exclusive sequence boundary.
    /// The shutdown path uses this after selecting a global ingress prefix so
    /// a later terminal cannot overtake an earlier state or diagnostic fact.
    pub fn drain_before_sequence(&self, exclusive_sequence: u64) -> Vec<WakeEntry> {
        let mut inner = self.lock();
        let take = inner
            .entries
            .iter()
            .take_while(|entry| entry.ingress_sequence < exclusive_sequence)
            .count();
        inner.entries.drain(..take).collect()
    }

    /// Read up to `max` queued terminal sequences without removing them.
    #[must_use]
    pub fn sequences_up_to(&self, max: Option<usize>) -> Vec<u64> {
        let inner = self.lock();
        inner
            .entries
            .iter()
            .take(max.unwrap_or(usize::MAX))
            .map(|entry| entry.ingress_sequence)
            .collect()
    }

    /// Sequence of the oldest entry still queued, used by a partial drain to
    /// keep facts from other lanes behind that terminal boundary.
    #[must_use]
    pub fn first_sequence(&self) -> Option<u64> {
        self.lock()
            .entries
            .front()
            .map(|entry| entry.ingress_sequence)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for WakeList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{OperationId, OperationScope};

    use super::*;
    use crate::runtime_events::reservation::ReservationTable;

    #[test]
    fn drain_returns_entries_in_ingress_sequence_order() {
        let wake = WakeList::new();
        let table = ReservationTable::new(3);
        let handles: Vec<_> = (0..3)
            .map(|_| {
                table
                    .reserve(OperationScope::root_only(OperationId::new()))
                    .unwrap()
            })
            .collect();

        for handle in &handles {
            wake.push_next(*handle);
        }

        let drained = wake.drain_all();
        let sequences: Vec<u64> = drained.iter().map(|entry| entry.ingress_sequence).collect();
        assert_eq!(sequences, vec![1, 2, 3]);
        assert!(wake.is_empty());
    }

    #[test]
    fn drain_up_to_leaves_the_remainder_queued() {
        let wake = WakeList::new();
        let table = ReservationTable::new(2);
        for _ in 0..2 {
            let handle = table
                .reserve(OperationScope::root_only(OperationId::new()))
                .unwrap();
            wake.push_next(handle);
        }

        let first = wake.drain_up_to(1);
        assert_eq!(first.len(), 1);
        assert_eq!(wake.len(), 1);
    }
}
