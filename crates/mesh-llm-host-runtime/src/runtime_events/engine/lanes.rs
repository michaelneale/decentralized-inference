//! Per-delivery-class submit handling.
//!
//! Terminal writes are the only path that touches the reservation table's
//! write-once slot and the wake list; state-transition, diagnostic, and
//! progress facts route through their own bounded structures below, each
//! drained fully by `engine::drain` (task 4,
//! `.omo/plans/event-system-fixes.md`). Every submit function here mints
//! from the SAME shared counter (`wake.rs::next_sequence`) exactly once
//! per call, regardless of outcome -- there is no second counter.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use mesh_llm_runtime_event_contracts::{OperationScope, RuntimeFact, SubmitOutcome};

use super::RuntimeEventEngine;
use crate::runtime_events::config::{DIAGNOSTIC_LANE_DEPTH, STATE_TRANSITION_LANE_DEPTH};
use crate::runtime_events::reservation::{SlotHandle, TerminalRecord};

/// A state-transition lane key: coalescing is per operation scope AND
/// kind, never globally by kind alone (review defect D2) -- two different
/// operations reporting the same kind must never overwrite each other.
type StateLaneKey = (OperationScope, &'static str);

type StateLaneValue = (RuntimeFact, u64, bool, Option<SlotHandle>);
type StateLaneEntry = (OperationScope, RuntimeFact, u64, bool, Option<SlotHandle>);
type DiagnosticEntry = (OperationScope, RuntimeFact, u64, bool, Option<SlotHandle>);

/// Per-engine bounded latest-value lane, keyed by `(OperationScope, kind)`:
/// a repeat key coalesces in place; a new key past the depth ceiling is
/// rejected without evicting an accepted state. Each held value carries the
/// ingress sequence it was minted with.
pub(crate) struct StateLane {
    entries: Mutex<VecDeque<StateLaneKey>>,
    /// `(fact, ingress_sequence, reserved)` -- `reserved` (R1 fix, task
    /// 6-fix, `.omo/plans/event-system-fixes.md`) is threaded from
    /// `submit_state_transition`'s own `handle.is_some()` and carried all
    /// the way to the reducer's `ReducerInput::reserved`. The handle is
    /// retained so a queued fact can be validated against cancellation or a
    /// generation reuse before its terminal releases the slot.
    latest: Mutex<HashMap<StateLaneKey, StateLaneValue>>,
}

impl StateLane {
    /// Test-only: the kinds currently held (latest-value-wins) in this
    /// lane, across every scope. Backs `RuntimeEventEngine::state_lane_kinds()`.
    #[cfg(test)]
    pub(super) fn kinds(&self) -> Vec<&'static str> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(_, kind)| *kind)
            .collect()
    }

    /// Drain only entries minted before `sequence`. A partial terminal drain
    /// leaves later state transitions queued until the remaining wake prefix
    /// is eligible, preserving the single global ingress order across passes.
    pub(super) fn drain_before_limit(&self, sequence: u64, limit: usize) -> Vec<StateLaneEntry> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ready = Vec::with_capacity(limit.min(entries.len()));
        let mut retained = VecDeque::with_capacity(entries.capacity());
        for key in entries.drain(..) {
            let Some((_, entry_sequence, _, _)) = latest.get(&key) else {
                continue;
            };
            if *entry_sequence < sequence && ready.len() < limit {
                if let Some((fact, entry_sequence, reserved, handle)) = latest.remove(&key) {
                    ready.push((key.0, fact, entry_sequence, reserved, handle));
                }
            } else {
                retained.push_back(key);
            }
        }
        *entries = retained;
        ready
    }

    /// Read the currently retained ingress sequences without removing them.
    /// The shutdown path uses this with the other lane views to choose one
    /// global sequence prefix before taking any values.
    pub(super) fn sequences_before(&self, sequence: u64) -> Vec<u64> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries
            .iter()
            .filter_map(|key| {
                latest
                    .get(key)
                    .map(|(_, entry_sequence, _, _)| *entry_sequence)
            })
            .filter(|entry_sequence| *entry_sequence < sequence)
            .collect()
    }

    pub(super) fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// Per-engine bounded diagnostic queue: strict FIFO, each entry carrying
/// the scope, ingress sequence, reservation provenance, and reservation
/// handle (R1 fix, task 6-fix -- see [`StateLane`]'s identical addition)
/// it was submitted with.
pub(crate) struct DiagnosticLane {
    queue: Mutex<VecDeque<DiagnosticEntry>>,
}

impl StateLane {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(STATE_TRANSITION_LANE_DEPTH)),
            latest: Mutex::new(HashMap::with_capacity(STATE_TRANSITION_LANE_DEPTH)),
        }
    }
}

impl Default for StateLane {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticLane {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(DIAGNOSTIC_LANE_DEPTH)),
        }
    }
}

impl Default for DiagnosticLane {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticLane {
    pub(super) fn drain_before_limit(&self, sequence: u64, limit: usize) -> Vec<DiagnosticEntry> {
        let mut queue = self
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ready = Vec::with_capacity(limit.min(queue.len()));
        let mut retained = VecDeque::with_capacity(queue.capacity());
        for entry in queue.drain(..) {
            if entry.2 < sequence && ready.len() < limit {
                ready.push(entry);
            } else {
                retained.push_back(entry);
            }
        }
        *queue = retained;
        ready
    }

    pub(super) fn sequences_before(&self, sequence: u64) -> Vec<u64> {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(_, _, entry_sequence, _, _)| *entry_sequence)
            .filter(|entry_sequence| *entry_sequence < sequence)
            .collect()
    }

    pub(super) fn len(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

pub(super) fn submit_terminal(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let Some(handle) = handle else {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    };
    if engine.table().occupant(handle) != Some(scope) {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    }
    let record = TerminalRecord {
        fact,
        synthesized: false,
    };
    if engine.table().write_terminal(handle, record) {
        // Mint-and-enqueue as ONE atomic step (unchanged `push_next`):
        // splitting these across two lock acquisitions would let a
        // later-minted concurrent submission's push overtake an
        // earlier-minted one's, breaking the wake list's FIFO ==
        // ingress-sequence-order invariant under concurrent terminal
        // writes to different scopes.
        engine.wake().push_next(handle);
        SubmitOutcome::Accepted
    } else {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        SubmitOutcome::TerminalDeliveryFailed
    }
}

pub(super) fn submit_progress(
    engine: &RuntimeEventEngine,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    match handle {
        Some(handle) if engine.table().coalesce_progress(handle, fact, sequence) => {
            SubmitOutcome::Coalesced
        }
        _ => {
            engine.health().bump_dropped_progress();
            SubmitOutcome::DroppedProgress
        }
    }
}

pub(super) fn submit_state_transition(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    let reserved = handle.is_some();
    let handle_for_lane = handle;
    if let Some(handle) = handle
        && (engine.table().occupant(handle) != Some(scope) || engine.table().is_cancelled(handle))
    {
        // A ScopedIngress may outlive an explicit reservation cancellation.
        // Consume its ingress sequence, but never let that stale handle
        // resurrect reducer state after cancellation has evicted it.
        engine.health().bump_cancelled_reservation_rejected();
        return SubmitOutcome::RejectedCancelled;
    }
    let lane = engine.state_lane();
    let key: StateLaneKey = (scope, fact.kind_id());
    // Lock `entries` THEN `latest` -- the SAME order `StateLane::drain`
    // above uses. Taking `latest` first (as this function used to) is an
    // AB-BA inversion against `drain`'s `entries`-then-`latest` order: a
    // concurrent drainer holding `entries` while waiting on `latest` and a
    // submitter holding `latest` while waiting on `entries` deadlock each
    // other. `inference::skippy::runtime_events::tests::concurrent_roots`
    // (task 5) was the first test to actually drive concurrent drain +
    // state-transition submits and surfaced this as an intermittent hang.
    let mut entries = lane
        .entries
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut latest = lane
        .latest
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(value) = latest.get_mut(&key) {
        *value = (fact, sequence, reserved, handle_for_lane);
        return SubmitOutcome::Coalesced;
    }
    if entries.len() >= STATE_TRANSITION_LANE_DEPTH {
        drop(latest);
        drop(entries);
        engine.health().bump_state_transition_rejected();
        return SubmitOutcome::RejectedCapacity;
    }
    latest.insert(key, (fact, sequence, reserved, handle_for_lane));
    entries.push_back(key);
    SubmitOutcome::Accepted
}

pub(super) fn submit_diagnostic(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    let reserved = handle.is_some();
    let handle_for_lane = handle;
    if let Some(handle) = handle
        && (engine.table().occupant(handle) != Some(scope) || engine.table().is_cancelled(handle))
    {
        engine.health().bump_dropped_diagnostic();
        return SubmitOutcome::DroppedDiagnostic;
    }
    let mut queue = engine
        .diagnostic_lane()
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if queue.len() >= DIAGNOSTIC_LANE_DEPTH {
        drop(queue);
        engine.health().bump_dropped_diagnostic();
        return SubmitOutcome::DroppedDiagnostic;
    }
    queue.push_back((scope, fact, sequence, reserved, handle_for_lane));
    SubmitOutcome::Accepted
}
