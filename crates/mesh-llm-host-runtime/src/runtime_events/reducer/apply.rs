//! Pure transactional application: `(snapshot, input) -> Applied | Rejected`.
//!
//! No I/O, no locking. Ordering is ingress-sequence only — native sequence
//! and any wall-clock hint are recorded as data, never consulted to decide
//! acceptance or ordering, per spec §10-11.

use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{
    NativeSequenceDomain, NativeSequenceEvidence, NativeSequenceObservation, OperationScope,
    RuntimeFact,
};

use super::state::{OperationState, ReducerSnapshot, RejectReason, outcome_of, progress_of};

/// One fact to reduce, already assigned its process-local ingress sequence
/// by the engine's wake list (or synthesized by a test driving the reducer
/// directly). `native_sequence` and `wall_clock_hint` are optional,
/// producer-supplied and inert for ordering/acceptance decisions.
#[derive(Debug, Clone)]
pub struct ReducerInput {
    pub scope: OperationScope,
    pub ingress_sequence: u64,
    pub native_sequence: Option<u64>,
    pub wall_clock_hint: Option<i64>,
    pub synthesized: bool,
    /// R1 fix (task 6-fix, `.omo/plans/event-system-fixes.md`): whether
    /// THIS submission arrived through a reservation-bound `ScopedIngress`
    /// (`true`) or an unreserved `UnreservedIngress` (`false`) -- threaded
    /// from `engine::submit`'s own `handle.is_some()`. OR-latched, sticky,
    /// into `OperationState::ever_reserved` in `advance` below; never read
    /// for accept/reject decisions and unrelated to `synthesized` (which
    /// is about terminal provenance, not reservation provenance).
    pub reserved: bool,
    pub fact: RuntimeFact,
}

#[derive(Debug, Clone)]
pub enum ReduceOutcome {
    Applied(Arc<ReducerSnapshot>),
    Rejected(RejectReason),
}

/// Evict `scope`'s tracked `OperationState`, if any -- the RELEASE-triggered
/// eviction path (task 6-fix defect A, `.omo/plans/event-system-fixes.md`:
/// "evict a settled operation from the reducer's operations map when its
/// reservation is released"). Pure and transactional exactly like [`apply`]:
/// `snapshot` itself is never mutated, and a scope that is not currently
/// tracked is a no-op that returns the SAME `Arc` (no allocation), which is
/// the common case for a `Child` scope that settled without ever receiving a
/// `StateTransition`/`Progress`/`Diagnostic` fact first.
///
/// The engine (`engine/drain.rs`) calls this once a scope's
/// reservation-table slot has ACTUALLY been released, and only after every
/// fact drained in the same pass -- including that scope's own terminal, if
/// this pass drained one -- has already been applied through [`apply`], so
/// this can never race an application that would just re-insert the entry a
/// moment later.
#[must_use]
pub fn evict(snapshot: &Arc<ReducerSnapshot>, scope: OperationScope) -> Arc<ReducerSnapshot> {
    if snapshot.operation(scope).is_none() {
        return Arc::clone(snapshot);
    }
    Arc::new(snapshot.without_operation(scope))
}

/// Apply `input` against `snapshot`, returning a fresh snapshot on success.
/// `snapshot` itself is never mutated: rejection leaves the caller's `Arc`
/// exactly as it was, which is the whole transactional guarantee.
#[must_use]
pub fn apply(snapshot: &Arc<ReducerSnapshot>, input: ReducerInput) -> ReduceOutcome {
    let current = snapshot.get_or_default(input.scope);

    if current.has_applied && input.ingress_sequence <= current.last_ingress_sequence {
        return ReduceOutcome::Rejected(RejectReason::Duplicate);
    }

    let data = input.fact.data();
    let incoming_outcome = outcome_of(data);
    let incoming_progress = progress_of(data);

    if current.settled {
        return match incoming_outcome {
            Some(_) => ReduceOutcome::Rejected(RejectReason::ContradictoryTerminal),
            None => ReduceOutcome::Rejected(RejectReason::OperationSettled),
        };
    }

    if let Some(incoming) = incoming_progress
        && let Some(previous) = current.last_progress_current
        && incoming < previous
    {
        return ReduceOutcome::Rejected(RejectReason::StaleProgress);
    }

    let mut process_native_sequences = Arc::clone(&snapshot.process_native_sequences);
    let next = advance(
        &current,
        &input,
        incoming_outcome,
        incoming_progress,
        &mut process_native_sequences,
    );
    let next_domain = snapshot.domain().apply_fact(input.scope, &input.fact);
    ReduceOutcome::Applied(Arc::new(snapshot.with_operation_and_native_sequences(
        input.scope,
        next,
        next_domain,
        process_native_sequences,
    )))
}

fn advance(
    current: &OperationState,
    input: &ReducerInput,
    incoming_outcome: Option<mesh_llm_runtime_event_contracts::Outcome>,
    incoming_progress: Option<u64>,
    process_native_sequences: &mut Arc<
        std::collections::HashMap<mesh_llm_runtime_event_contracts::NativeEmitter, u64>,
    >,
) -> OperationState {
    let mut next = current.clone();
    next.has_applied = true;
    next.ever_reserved = next.ever_reserved || input.reserved;
    next.last_ingress_sequence = input.ingress_sequence;
    let native_observation = input
        .fact
        .metadata()
        .and_then(|metadata| metadata.native_sequence);
    next.last_native_sequence = resolve_native_sequence(
        &mut next,
        process_native_sequences,
        input.native_sequence,
        native_observation,
    );

    if let Some(progress) = incoming_progress {
        next.last_progress_current = Some(progress);
    }
    if let Some(outcome) = incoming_outcome {
        next.settled = true;
        next.last_outcome = Some(outcome);
        next.degraded = next.degraded || input.synthesized;
    }
    next
}

/// Track the native-sequence high-water mark and flag (never act on) a gap.
fn resolve_native_sequence(
    state: &mut OperationState,
    process_native_sequences: &mut Arc<
        std::collections::HashMap<mesh_llm_runtime_event_contracts::NativeEmitter, u64>,
    >,
    incoming: Option<u64>,
    observation: Option<NativeSequenceObservation>,
) -> Option<u64> {
    if let Some(observation) = observation {
        return match observation.domain {
            NativeSequenceDomain::Operation => {
                resolve_operation_sequence(state, observation.sequence, observation.evidence)
            }
            NativeSequenceDomain::Process => {
                resolve_process_sequence(state, process_native_sequences, observation)
            }
        };
    }

    // Legacy reducer callers supplied only the scalar sequence. Preserve
    // that additive API, but keep the old inference scoped to that operation;
    // production native facts carry `NativeSequenceObservation` from the
    // callback adapter and never infer a gap from coalesced facts.
    let Some(incoming) = incoming else {
        return state.last_native_sequence;
    };
    let Some(previous) = state.last_native_sequence else {
        return Some(incoming);
    };
    if incoming <= previous {
        return Some(previous);
    }
    if incoming.saturating_sub(previous) > 1 {
        state.native_gap_count = state.native_gap_count.saturating_add(1);
    }
    Some(incoming)
}

fn resolve_operation_sequence(
    state: &mut OperationState,
    incoming: u64,
    evidence: NativeSequenceEvidence,
) -> Option<u64> {
    if matches!(evidence, NativeSequenceEvidence::Gap) {
        state.native_gap_count = state.native_gap_count.saturating_add(1);
    }
    Some(
        state
            .last_native_sequence
            .map_or(incoming, |previous| previous.max(incoming)),
    )
}

fn resolve_process_sequence(
    state: &mut OperationState,
    process_native_sequences: &mut Arc<
        std::collections::HashMap<mesh_llm_runtime_event_contracts::NativeEmitter, u64>,
    >,
    observation: NativeSequenceObservation,
) -> Option<u64> {
    if matches!(observation.evidence, NativeSequenceEvidence::Gap) {
        state.native_gap_count = state.native_gap_count.saturating_add(1);
    }

    let previous = process_native_sequences.get(&observation.emitter).copied();
    let can_mutate = match previous {
        Some(previous) => observation.sequence > previous,
        None => process_native_sequences.len() < super::state::MAX_NATIVE_SEQUENCE_DOMAINS,
    };
    let high_water = if can_mutate {
        let process_native_sequences = Arc::make_mut(process_native_sequences);
        if let Some(previous) = process_native_sequences.get_mut(&observation.emitter) {
            *previous = observation.sequence;
            *previous
        } else {
            process_native_sequences.insert(observation.emitter, observation.sequence);
            observation.sequence
        }
    } else {
        previous.unwrap_or(observation.sequence)
    };
    Some(
        state
            .last_native_sequence
            .map_or(high_water, |previous| previous.max(high_water)),
    )
}
