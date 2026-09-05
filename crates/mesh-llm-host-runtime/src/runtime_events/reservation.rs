//! The reservation table: the only terminal channel.
//!
//! Each slot owns one write-once terminal record. Admission hands out an
//! index plus a generation counter; a later access whose generation does not
//! match the slot's current generation is treated as late/unreserved rather
//! than corrupting a reused slot.

use std::sync::Mutex;

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FamilyFact, OperationId, OperationScope, RuntimeFact, ScopeIdentities,
};

fn default_synthetic_terminal() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(
        DiagnosticEventKind::InvariantProtocolViolation,
    ))
}

/// One terminal write for a slot: the fact plus whether it was synthesized
/// by a dropped guard or shutdown rather than submitted by a producer.
#[derive(Debug, Clone)]
pub struct TerminalRecord {
    pub fact: RuntimeFact,
    pub synthesized: bool,
}

#[derive(Debug, Default)]
struct Slot {
    occupant: Option<OperationScope>,
    terminal: Option<TerminalRecord>,
    /// Explicit cancellation invalidates facts already holding this
    /// generation, including a deferred root that still waits for children.
    cancelled: bool,
    /// The single progress-coalescing value, tagged with the ingress
    /// sequence it was minted with so the engine's periodic flush
    /// (`engine/drain.rs`, task 4) applies it under its real,
    /// gap-preserving position rather than a synthetic re-numbering.
    progress: Option<(RuntimeFact, u64)>,
    /// Family-correct terminal constructor retained with the occupancy so
    /// shutdown can settle a live guard even though the guard itself may be
    /// held by another producer thread.
    synthetic_terminal: Option<fn() -> RuntimeFact>,
    /// Last non-empty scope supplied by an accepted submission for this
    /// generation. Shutdown and guard-drop synthesis use this to retain
    /// typed identities while the producer guard is still held.
    scope_identities: Option<ScopeIdentities>,
    generation: u64,
}

/// Outcome of an admission attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveError {
    Exhausted,
}

/// A `(slot index, generation)` handle. Cheap to copy; used by the engine to
/// address a specific occupancy of a specific slot without holding a guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotHandle {
    pub index: usize,
    pub generation: u64,
}

/// An occupied slot without a terminal, captured for shutdown synthesis.
#[derive(Clone)]
pub(crate) struct UnsettledReservation {
    pub(crate) handle: SlotHandle,
    pub(crate) scope: OperationScope,
    pub(crate) synthetic_terminal: fn() -> RuntimeFact,
    pub(crate) scope_identities: Option<ScopeIdentities>,
}

/// Bounded slab of reservation slots. One write-once terminal per occupied
/// slot; no second terminal lane exists anywhere in this table.
#[derive(Debug)]
pub struct ReservationTable {
    slots: Vec<Mutex<Slot>>,
    free: Mutex<Vec<usize>>,
}

impl ReservationTable {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| Mutex::new(Slot::default())).collect(),
            free: Mutex::new((0..capacity).rev().collect()),
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Admit `scope`, returning a fresh `(index, generation)` handle or
    /// [`ReserveError::Exhausted`] when the table is full.
    pub fn reserve(&self, scope: OperationScope) -> Result<SlotHandle, ReserveError> {
        self.reserve_with_synthesizer(scope, default_synthetic_terminal)
    }

    /// Reserve a slot while retaining the family-provided synthesizer for
    /// shutdown settlement. The plain `reserve` helper remains available for
    /// table-only tests and callers that do not need the engine's synthesis
    /// contract.
    pub fn reserve_with_synthesizer(
        &self,
        scope: OperationScope,
        synthetic_terminal: fn() -> RuntimeFact,
    ) -> Result<SlotHandle, ReserveError> {
        let index = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .ok_or(ReserveError::Exhausted)?;
        let mut slot = self.slot_lock(index);
        slot.occupant = Some(scope);
        slot.terminal = None;
        slot.cancelled = false;
        slot.progress = None;
        slot.synthetic_terminal = Some(synthetic_terminal);
        slot.scope_identities = None;
        slot.generation += 1;
        SlotHandle {
            index,
            generation: slot.generation,
        }
        .pipe_ok()
    }

    /// Write the write-once terminal slot for `handle`. Returns `true` when
    /// this call performed the write, `false` when the handle is stale
    /// (late/unreserved/ID-mismatched) or a terminal was already present
    /// (duplicate).
    pub fn write_terminal(&self, handle: SlotHandle, record: TerminalRecord) -> bool {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation || slot.cancelled || slot.terminal.is_some() {
            return false;
        }
        slot.terminal = Some(record);
        true
    }

    /// Overwrite the single progress-coalescing slot bound to `handle` with
    /// `fact`, tagging it with the ingress `sequence` it was minted with.
    /// Returns `false` for a stale handle.
    pub fn coalesce_progress(&self, handle: SlotHandle, fact: RuntimeFact, sequence: u64) -> bool {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation || slot.cancelled {
            return false;
        }
        slot.progress = Some((fact, sequence));
        true
    }

    /// Remember a non-empty scope from an accepted submission for this
    /// generation. A stale, cancelled, or unoccupied handle is rejected.
    pub fn remember_scope(&self, handle: SlotHandle, scope: ScopeIdentities) -> bool {
        if scope == ScopeIdentities::default() {
            return false;
        }
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation || slot.cancelled || slot.occupant.is_none() {
            return false;
        }
        slot.scope_identities = Some(scope);
        true
    }

    /// Take every currently-pending progress fact across the whole table,
    /// clearing each taken slot's progress value as it is read (the
    /// engine's periodic 100 ms progress flush, task 4). A full-table scan
    /// is bounded by this table's fixed capacity and cheap at that
    /// cadence.
    #[must_use]
    pub fn take_all_progress(&self) -> Vec<(OperationScope, RuntimeFact, u64, SlotHandle)> {
        self.take_progress_before(u64::MAX)
    }

    /// Take only progress minted before `sequence`, leaving later coalesced
    /// values in their slots until an earlier wake prefix is drained.
    #[must_use]
    pub fn take_progress_before(
        &self,
        sequence: u64,
    ) -> Vec<(OperationScope, RuntimeFact, u64, SlotHandle)> {
        self.take_progress_before_limit(sequence, usize::MAX)
    }

    /// Take at most `limit` due progress values, leaving the remainder in
    /// place for a later bounded drain chunk. The table scan is deterministic
    /// and remains bounded by the reservation capacity.
    #[must_use]
    pub fn take_progress_before_limit(
        &self,
        sequence: u64,
        limit: usize,
    ) -> Vec<(OperationScope, RuntimeFact, u64, SlotHandle)> {
        let mut taken = Vec::with_capacity(self.slots.len());
        for index in 0..self.slots.len() {
            if taken.len() >= limit {
                break;
            }
            let mut slot = self.slot_lock(index);
            let Some((_, progress_sequence)) = slot.progress.as_ref() else {
                continue;
            };
            if *progress_sequence >= sequence {
                continue;
            }
            let Some((fact, progress_sequence)) = slot.progress.take() else {
                continue;
            };
            if !slot.cancelled
                && let Some(scope) = slot.occupant
            {
                taken.push((
                    scope,
                    fact,
                    progress_sequence,
                    SlotHandle {
                        index,
                        generation: slot.generation,
                    },
                ));
            }
        }
        taken
    }

    /// Return the ingress sequences of currently live, coalesced progress
    /// values. This is a read-only view used to select one global shutdown
    /// prefix before any lane is drained.
    #[must_use]
    pub fn progress_sequences_before(&self, sequence: u64) -> Vec<u64> {
        self.slots
            .iter()
            .filter_map(|slot| {
                let slot = slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (!slot.cancelled && slot.occupant.is_some())
                    .then_some(slot.progress.as_ref()?.1)
                    .filter(|progress_sequence| *progress_sequence < sequence)
            })
            .collect()
    }

    #[must_use]
    pub fn pending_progress_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| {
                let slot = slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                !slot.cancelled && slot.occupant.is_some() && slot.progress.is_some()
            })
            .count()
    }

    #[must_use]
    pub fn occupant(&self, handle: SlotHandle) -> Option<OperationScope> {
        let slot = self.slot_lock(handle.index);
        (slot.generation == handle.generation).then_some(slot.occupant)?
    }

    /// Return the last non-empty scope remembered for this live generation.
    #[must_use]
    pub fn scope_identities(&self, handle: SlotHandle) -> Option<ScopeIdentities> {
        let slot = self.slot_lock(handle.index);
        (slot.generation == handle.generation && slot.occupant.is_some())
            .then(|| slot.scope_identities.clone())
            .flatten()
    }

    #[must_use]
    pub fn has_terminal(&self, handle: SlotHandle) -> bool {
        let slot = self.slot_lock(handle.index);
        slot.generation == handle.generation && slot.terminal.is_some()
    }

    /// Clone out the written terminal record for `handle`, or `None` for a
    /// stale handle or a slot with no terminal written yet.
    #[must_use]
    pub fn terminal_record(&self, handle: SlotHandle) -> Option<TerminalRecord> {
        let slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation {
            return None;
        }
        slot.terminal.clone()
    }

    #[must_use]
    pub fn is_current(&self, handle: SlotHandle) -> bool {
        self.slot_lock(handle.index).generation == handle.generation
    }

    /// Mark a live reservation generation as cancelled. The slot remains
    /// occupied when root release is deferred behind children, so ingress
    /// paths must consult this bit until `release` advances the generation.
    pub fn mark_cancelled(&self, handle: SlotHandle) -> bool {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation || slot.occupant.is_none() {
            return false;
        }
        slot.cancelled = true;
        slot.progress = None;
        true
    }

    #[must_use]
    pub fn is_cancelled(&self, handle: SlotHandle) -> bool {
        let slot = self.slot_lock(handle.index);
        slot.generation == handle.generation && slot.cancelled
    }

    /// Reclaim `handle`'s slot: clear its contents and return the index to
    /// the free list. Advances the slot generation so any outstanding stale
    /// handle (a dropped guard fired after reuse) is provably invalidated.
    pub fn release(&self, handle: SlotHandle) {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation {
            return;
        }

        slot.occupant = None;
        slot.terminal = None;
        slot.cancelled = false;
        slot.progress = None;
        slot.synthetic_terminal = None;
        slot.scope_identities = None;
        // Advance the generation on release itself, not only on reuse,
        // so a guard that drops after a forced release (e.g. a child
        // whose root already released it) sees an immediate mismatch
        // instead of a window where the freed-but-not-yet-reused slot
        // still matches its stale handle.
        slot.generation += 1;

        // Keep the slot lock held through the free-list insertion. This
        // makes generation validation, cleanup, and reclamation one atomic
        // operation with respect to duplicate/stale releases. `reserve`
        // drops its free-list guard before taking any slot lock, so this
        // slot-then-free order cannot form an ABBA cycle.
        self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle.index);
    }

    /// The occupant of `index` at its *current* generation, regardless of
    /// whether the caller's own handle is stale. Used when force-completing
    /// outstanding children on root release.
    #[must_use]
    pub fn is_occupied(&self, index: usize) -> Option<OperationScope> {
        self.slot_lock(index).occupant
    }

    /// Whether any live, non-cancelled generation currently owns `scope`.
    /// Lane entries retain only the scope and reservation provenance; this
    /// bounded scan lets the drain discard a fact queued before cancellation
    /// without allowing it to resurrect reducer state.
    #[must_use]
    pub fn has_active_scope(&self, scope: OperationScope) -> bool {
        self.slots.iter().any(|slot| {
            let slot = slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.occupant == Some(scope) && !slot.cancelled
        })
    }

    #[must_use]
    pub fn occupied_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| {
                slot.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .occupant
                    .is_some()
            })
            .count()
    }

    #[must_use]
    pub fn current_generation(&self, index: usize) -> u64 {
        self.slot_lock(index).generation
    }

    /// Snapshot every occupied slot that has not yet received a terminal,
    /// retaining the generation and original family synthesizer. The engine
    /// calls this only after admission closes, so the collection cannot race
    /// a new reserve or terminal write.
    #[must_use]
    pub(crate) fn unsettled(&self) -> Vec<UnsettledReservation> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let slot = slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match (
                    slot.occupant,
                    !slot.cancelled,
                    slot.terminal.is_none(),
                    slot.synthetic_terminal,
                ) {
                    (Some(scope), true, true, Some(synthetic_terminal)) => {
                        Some(UnsettledReservation {
                            handle: SlotHandle {
                                index,
                                generation: slot.generation,
                            },
                            scope,
                            synthetic_terminal,
                            scope_identities: slot.scope_identities.clone(),
                        })
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn slot_lock(&self, index: usize) -> std::sync::MutexGuard<'_, Slot> {
        self.slots[index]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

trait PipeOk: Sized {
    fn pipe_ok<E>(self) -> Result<Self, E> {
        Ok(self)
    }
}
impl<T> PipeOk for T {}

#[must_use]
pub fn root_of(scope: OperationScope) -> OperationId {
    scope.root()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, RuntimeFact,
    };

    use super::*;

    fn terminal_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
    }

    #[test]
    fn reserve_then_release_recycles_the_slot() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        assert!(matches!(table.reserve(scope), Err(ReserveError::Exhausted)));

        table.release(handle);
        let reused = table.reserve(scope).expect("reserve after release");
        assert_ne!(reused.generation, handle.generation);
    }

    #[test]
    fn second_terminal_write_is_rejected_as_duplicate() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");

        assert!(table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
        assert!(!table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
    }

    #[test]
    fn stale_handle_after_release_cannot_write_a_terminal() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        table.release(handle);
        let _reused = table.reserve(scope).expect("reuse");

        assert!(!table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
    }

    #[test]
    fn concurrent_duplicate_releases_return_one_slot_to_the_free_list() {
        let table = Arc::new(ReservationTable::new(1));
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        let barrier = Arc::new(Barrier::new(2));

        thread::scope(|scope| {
            for _ in 0..2 {
                let table = Arc::clone(&table);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    table.release(handle);
                });
            }
        });

        let reused = table.reserve(OperationScope::root_only(OperationId::new()));
        assert!(reused.is_ok(), "one release must return the slot");
        assert!(
            matches!(
                table.reserve(OperationScope::root_only(OperationId::new())),
                Err(ReserveError::Exhausted)
            ),
            "a duplicate release must not put the same index on the free list twice"
        );
    }

    #[test]
    fn stale_release_after_slot_reuse_cannot_free_the_new_occupant() {
        let table = ReservationTable::new(1);
        let first = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("first reserve");
        table.release(first);
        let second = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("reuse");

        table.release(first);

        assert!(
            matches!(
                table.reserve(OperationScope::root_only(OperationId::new())),
                Err(ReserveError::Exhausted)
            ),
            "a stale handle must not free a newer generation's occupant"
        );
        assert!(table.is_current(second));
    }
}
