//! Pure, immutable reducer state.
//!
//! `ReducerSnapshot` is never mutated in place: every transition produces a
//! fresh `Arc<ReducerSnapshot>` (clone-on-write over a small bounded map),
//! so a rejected input structurally cannot leave a partially applied
//! snapshot visible to any reader.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{FactData, NativeEmitter, OperationScope, Outcome};

use super::domain::DomainState;
use crate::runtime_events::config::{RESERVATION_TABLE_CAPACITY, UNRESERVED_OPERATION_BOUND};

/// Native emitters are intentionally bounded. An unknown/new emitter is
/// still reduced, but its sequence is treated as untracked until a bounded
/// slot is available; it cannot grow reducer memory without limit.
pub(super) const MAX_NATIVE_SEQUENCE_DOMAINS: usize = 16;

/// Per-operation reduced view. Preserved across rebuild for "last-valid"
/// continuity even when the operation never settles cleanly.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationState {
    pub scope: OperationScope,
    pub has_applied: bool,
    pub settled: bool,
    pub degraded: bool,
    /// R1 fix (task 6-fix, `.omo/plans/event-system-fixes.md`): sticky,
    /// OR-latched (mirrors `degraded`'s own latch-on discipline via
    /// `next.degraded || input.synthesized`) flag for whether ANY fact
    /// applied for this scope so far arrived through a reservation-bound
    /// submission (`ReducerInput::reserved`, threaded from
    /// `engine::submit`'s own `handle.is_some()`). `false` for this
    /// scope's whole tracked lifetime means every fact arrived via
    /// `unreserved_ingress` with no live `SlotHandle` ever backing it --
    /// exactly the shape KV cache lookups, session lifecycle, and
    /// node/topology/model-lifecycle observers submit -- which routes it
    /// into `ReducerSnapshot`'s `unreserved_order` bounded LRU instead of
    /// relying on release-triggered eviction (no reservation to release)
    /// or the settled-only backstop (it may never settle).
    pub ever_reserved: bool,
    pub last_outcome: Option<Outcome>,
    pub last_progress_current: Option<u64>,
    pub last_ingress_sequence: u64,
    pub last_native_sequence: Option<u64>,
    pub native_gap_count: u64,
}

impl OperationState {
    fn new(scope: OperationScope) -> Self {
        Self {
            scope,
            has_applied: false,
            settled: false,
            degraded: false,
            ever_reserved: false,
            last_outcome: None,
            last_progress_current: None,
            last_ingress_sequence: 0,
            last_native_sequence: None,
            native_gap_count: 0,
        }
    }
}

/// Immutable, `Arc`-shared reducer state. Cheap to hand to readers; a
/// writer never mutates an existing instance, only produces a new one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReducerSnapshot {
    operations: HashMap<OperationScope, OperationState>,
    /// High-water marks for process-scoped native sequences. Process
    /// reporters may interleave facts assigned to different operation scopes,
    /// so this state cannot live on `OperationState`. The emitter is the
    /// sequence domain implemented by the native dispatcher.
    pub(super) process_native_sequences: Arc<HashMap<NativeEmitter, u64>>,
    /// R1 fix (task 6-fix, `.omo/plans/event-system-fixes.md`): LRU
    /// touch-order, oldest first, of every scope currently tracked in
    /// `operations` whose `OperationState::ever_reserved` is `false` --
    /// exactly mirrors `reducer/domain.rs`'s existing `*_order`
    /// bounded-eviction idiom, applied here to the top-level operations
    /// map instead of a per-category domain view. Maintained 1:1 with the
    /// never-reserved subset of `operations` by every mutator
    /// (`with_operation`, `without_operation`, `evict_settled_over_capacity`):
    /// never touched directly by callers outside this module.
    unreserved_order: VecDeque<OperationScope>,
    pub rebuild_generation: u64,
    domain: DomainState,
}

impl ReducerSnapshot {
    #[must_use]
    pub fn empty() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[must_use]
    pub fn operation(&self, scope: OperationScope) -> Option<&OperationState> {
        self.operations.get(&scope)
    }

    #[must_use]
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    #[cfg(test)]
    pub(super) fn process_native_sequences_ptr(&self) -> *const HashMap<NativeEmitter, u64> {
        Arc::as_ptr(&self.process_native_sequences)
    }

    /// Bounded per-category domain state (task 6, defect D6): models,
    /// stages, sessions, in-flight requests, devices, and cache, reduced
    /// from the same facts `with_operation` folds into `OperationState`.
    #[must_use]
    pub fn domain(&self) -> &DomainState {
        &self.domain
    }

    pub(super) fn get_or_default(&self, scope: OperationScope) -> OperationState {
        self.operations
            .get(&scope)
            .cloned()
            .unwrap_or_else(|| OperationState::new(scope))
    }

    /// Produce a new snapshot with `scope`'s state replaced and `domain`
    /// installed as the new bounded domain view. The receiver is left
    /// untouched; callers swap the shared `Arc` only after this succeeds,
    /// which is what makes application transactional.
    ///
    /// Runs TWO independent eviction mechanisms after inserting `state`:
    /// [`evict_settled_over_capacity`] as a defensive backstop over the
    /// WHOLE map (task 6-fix defect A, `.omo/plans/event-system-fixes.md`)
    /// -- the PRIMARY eviction mechanism for a reservation-backed scope is
    /// release-triggered (`Self::without_operation`, called by the engine
    /// the moment a scope's reservation is actually released), which keeps
    /// this map far below `RESERVATION_TABLE_CAPACITY` in the steady
    /// state, so this sweep is a no-op on essentially every call for
    /// reservation-backed traffic; and [`touch_unreserved`] / eviction into
    /// `unreserved_order` (R1 fix, task 6-fix) for a scope whose
    /// `state.ever_reserved` is `false` -- a scope with NO reservation
    /// ever backing it can be evicted by NEITHER of the two mechanisms
    /// above (nothing to release; may never settle), so it is bounded
    /// independently at `UNRESERVED_OPERATION_BOUND` here. An in-flight
    /// (unsettled) RESERVED operation is never evicted by either
    /// mechanism.
    /// Variant used by the reducer when an accepted fact updates a
    /// process-scoped native emitter high-water mark.
    #[must_use]
    pub(super) fn with_operation_and_native_sequences(
        &self,
        scope: OperationScope,
        state: OperationState,
        domain: DomainState,
        process_native_sequences: Arc<HashMap<NativeEmitter, u64>>,
    ) -> Self {
        let mut operations = self.operations.clone();
        let mut unreserved_order = self.unreserved_order.clone();
        let ever_reserved = state.ever_reserved;
        operations.insert(scope, state);
        if ever_reserved {
            remove_unreserved_order_entry(&mut unreserved_order, scope);
        } else {
            touch_unreserved(&mut unreserved_order, &mut operations, scope);
        }
        evict_settled_over_capacity(&mut operations, &mut unreserved_order);
        debug_assert!(
            unreserved_order.len() <= UNRESERVED_OPERATION_BOUND,
            "unreserved_order must never exceed UNRESERVED_OPERATION_BOUND (task 6-fix R1)"
        );
        Self {
            operations,
            unreserved_order,
            process_native_sequences,
            rebuild_generation: self.rebuild_generation,
            domain,
        }
    }

    /// Produce a new snapshot with `scope`'s tracked `OperationState`
    /// removed, leaving `domain` and `rebuild_generation` untouched -- the
    /// release-triggered eviction primitive behind
    /// [`super::apply::evict`]. Pure clone-on-write exactly like
    /// [`Self::with_operation_and_native_sequences`]: `self` is never mutated. Also removes
    /// `scope` from `unreserved_order` if present (R1 fix, task 6-fix):
    /// defensive-only for the six known production call sites (a
    /// reservation-backed scope is never in `unreserved_order` in the
    /// first place), but keeps the 1:1 invariant exact for any future
    /// producer that submits an unreserved fact before its scope is ever
    /// reserved.
    #[must_use]
    pub(super) fn without_operation(&self, scope: OperationScope) -> Self {
        let mut operations = self.operations.clone();
        operations.remove(&scope);
        let mut unreserved_order = self.unreserved_order.clone();
        remove_unreserved_order_entry(&mut unreserved_order, scope);
        Self {
            operations,
            unreserved_order,
            process_native_sequences: self.process_native_sequences.clone(),
            rebuild_generation: self.rebuild_generation,
            domain: self.domain.clone(),
        }
    }

    #[must_use]
    pub(super) fn with_generation(&self, generation: u64) -> Self {
        Self {
            operations: self.operations.clone(),
            unreserved_order: self.unreserved_order.clone(),
            process_native_sequences: self.process_native_sequences.clone(),
            rebuild_generation: generation,
            domain: self.domain.clone(),
        }
    }

    #[must_use]
    pub(super) fn degrade_unsettled(&self) -> Self {
        let operations = self
            .operations
            .iter()
            .map(|(scope, state)| {
                let mut next = state.clone();
                if !next.settled {
                    next.degraded = true;
                }
                (*scope, next)
            })
            .collect();
        Self {
            operations,
            unreserved_order: self.unreserved_order.clone(),
            process_native_sequences: self.process_native_sequences.clone(),
            rebuild_generation: self.rebuild_generation,
            domain: self.domain.clone(),
        }
    }
}

/// R1 fix (task 6-fix, `.omo/plans/event-system-fixes.md`): move `scope`
/// to the back of `order` (most-recently-touched); if `scope` is new and
/// `order` is already at `UNRESERVED_OPERATION_BOUND`, evict the single
/// oldest entry from BOTH `order` and `operations` first. Mirrors
/// `reducer/domain.rs::touch`'s identical idiom, applied to
/// `ReducerSnapshot::operations` instead of a per-category domain map.
fn touch_unreserved(
    order: &mut VecDeque<OperationScope>,
    operations: &mut HashMap<OperationScope, OperationState>,
    scope: OperationScope,
) {
    if let Some(position) = order.iter().position(|existing| *existing == scope) {
        order.remove(position);
    } else if order.len() >= UNRESERVED_OPERATION_BOUND
        && let Some(oldest) = order.pop_front()
    {
        operations.remove(&oldest);
    }
    order.push_back(scope);
}

/// Remove `scope` from `order` if present; a no-op otherwise. Mirrors
/// `reducer/domain.rs::remove_bounded`'s order-side half (there is no
/// map-side removal here: callers of this function already own removing
/// `scope` from `operations` through their own path).
fn remove_unreserved_order_entry(order: &mut VecDeque<OperationScope>, scope: OperationScope) {
    if let Some(position) = order.iter().position(|existing| *existing == scope) {
        order.remove(position);
    }
}

/// Evict the oldest settled operations (by `last_ingress_sequence`,
/// ascending) until `operations` is at or below
/// `RESERVATION_TABLE_CAPACITY`. A no-op when nothing is over the bound.
/// Also removes any evicted scope from `unreserved_order` so the two
/// structures never drift out of the 1:1 invariant `with_operation`/
/// `without_operation` otherwise maintain.
fn evict_settled_over_capacity(
    operations: &mut HashMap<OperationScope, OperationState>,
    unreserved_order: &mut VecDeque<OperationScope>,
) {
    while operations.len() > RESERVATION_TABLE_CAPACITY {
        let oldest = operations
            .iter()
            .filter(|(_, state)| state.settled)
            .min_by_key(|(_, state)| state.last_ingress_sequence)
            .map(|(scope, _)| *scope);
        let Some(oldest) = oldest else {
            // Nothing settled left to evict. R1 CORRECTION (task 6-fix,
            // `.omo/plans/event-system-fixes.md`): the prior commit's
            // comment here claimed this branch was "structurally
            // unreachable in production ... the reservation table never
            // admits more than RESERVATION_TABLE_CAPACITY concurrently-
            // occupied scopes" -- that reasoning covers only
            // RESERVATION-BACKED scopes. Six real production call sites
            // (KV cache lookups, session lifecycle, and
            // node/topology/model-lifecycle observers) submit
            // StateTransition-class facts through `unreserved_ingress`
            // with a fresh `OperationId` per event and NO reservation ever
            // backing them: they can never settle (no Terminal is ever
            // accepted with no `SlotHandle`) and can never be released
            // (nothing to release), so this branch WAS genuinely reachable
            // and permanently stalled once such traffic pushed
            // `operations.len()` past capacity with nothing settled among
            // the excess. `unreserved_order`'s own bound
            // (`UNRESERVED_OPERATION_BOUND`, enforced by
            // `touch_unreserved` in `with_operation`) now caps that
            // traffic independently of this sweep, so this branch stays
            // unreachable again in practice -- but for the RIGHT reason
            // this time, not the one above. This stays a defensive break
            // rather than an unbounded loop or a panic.
            break;
        };
        operations.remove(&oldest);
        remove_unreserved_order_entry(unreserved_order, oldest);
    }
}

/// Why a reducer input was not applied. The stream never observes a
/// rejected input: no replay frame, no subscriber fan-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Same-or-earlier ingress sequence already applied for this scope.
    Duplicate,
    /// Progress regressed against the last-applied progress value.
    StaleProgress,
    /// A second terminal for an already-settled operation.
    ContradictoryTerminal,
    /// A non-terminal fact for an already-settled operation.
    OperationSettled,
}

pub(super) fn outcome_of(data: &FactData) -> Option<Outcome> {
    data.outcome
}

pub(super) fn progress_of(data: &FactData) -> Option<u64> {
    data.progress.map(|progress| progress.current)
}
