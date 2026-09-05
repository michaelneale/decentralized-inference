//! Runtime-event producer wiring for Skippy's authoritative generation
//! lifecycle (plan task 12, `.omo/plans/event-system.md` line 294).
//!
//! Adapts `skippy-server`'s dependency-safe [`GenerationLifecycleIngress`]
//! observations onto `RuntimeFact::Generation`/`Prefill`/`Session`/
//! `KvRuntimeState` facts through the host runtime-event engine, without
//! displacing or replacing the trait or its existing implementors (PR
//! #1149's native-plugin lifecycle ingress). `resolve_serving_hooks` in
//! `inference/skippy/mod.rs` always installs this adapter, and composes it
//! with a native plugin's own ingress via
//! [`skippy_server::frontend::CompositeGenerationLifecycleIngress`] when a
//! plugin is loaded (the serving-hooks slot is single-occupancy).
//!
//! `facts.rs` owns pure fact construction per family; this module owns
//! reservation lifecycle and tracking.
//!
//! A generation whose [`GenerationStart::frontend_request_id`] is set
//! reserves under the OpenAI request root `OperationId` (Task 11's
//! byte-equal root), so a generation started from the OpenAI boundary
//! correlates directly with its request. A generation with no frontend
//! request id (a non-frontend caller) mints its own root instead. Either
//! way that root ID also parents a sibling PREFILL child reservation, since
//! the reservation model supports only one level of nesting (root + child,
//! never grandchild) -- see `begin` below.
//!
//! Tracking has NO independent capacity bound: an entry is inserted only
//! after a reservation succeeds (itself bounded by the engine's reservation
//! table -- the single source of capacity truth) and is removed the instant
//! its terminal resolves, so completed generations never accumulate.
//!
//! Every emission is best-effort: an absent engine, a reservation-table
//! exhaustion, or a rejected submit never fails inference -- `try_submit`
//! always returns `Ok(())`, matching `GenerationLifecycleIngress`'s
//! documented at-most-once, non-blocking contract.

mod facts;
mod kv;
mod session;
#[cfg(test)]
mod tests;

pub(crate) use kv::SkippyKvRuntimeEventObserver;
pub(crate) use session::SkippySessionRuntimeEventObserver;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use mesh_llm_runtime_event_contracts::{
    ChildOperationId, GenerationEventKind, OperationId, RuntimeEventIngress, RuntimeFact,
    SubmitOutcome,
};
use skippy_server::frontend::{
    GenerationAbort, GenerationCommit, GenerationCompletion, GenerationLifecycleIngress,
    GenerationLifecycleObservation, GenerationReceipt, GenerationStart,
};

use crate::runtime_events::engine::OperationReservation;
use crate::runtime_events::runtime_event_engine;

use facts::{
    abort_terminal_fact, empty_data, first_token_produced_fact,
    generation_completion_terminal_fact, generation_fact, prefill_cancelled_fact,
    prefill_completion_terminal_fact, prefill_terminal_fact, progress_fact, receipt_terminal_fact,
    session_active_fact, session_idle_fact, stop_condition_reached_fact,
    synthetic_generation_terminal, synthetic_prefill_terminal,
};

#[derive(Default)]
struct GenerationTracking {
    // Declaration order still matters for LATENCY, though it is no longer
    // load-bearing for correctness (task 5, review defect D8): Rust drops
    // struct fields in declaration order, and `prefill` (a CHILD of the
    // group root in the non-frontend case where `generation` itself
    // occupies that root) drops -- and so queues its own synthesized-
    // terminal wake entry -- before `generation` does, letting both
    // settle in the same drain pass with no wait. Before D8's fix,
    // dropping `generation` (the root) first would have let its release
    // cascade-drop the still-occupied `prefill` slot before `prefill`'s
    // own Drop ever ran, discarding its synthesized terminal; now
    // `engine::drain::release_or_defer` defers the root's own release
    // instead, so `prefill` settles correctly (within `CHILD_SETTLE_GRACE`
    // at worst) regardless of drop order.
    prefill: Option<OperationReservation>,
    generation: Option<OperationReservation>,
    /// Whether §8.10's `first_token_produced` has already been emitted for
    /// this generation. `GenerationCommit` observations repeat per token
    /// batch; this flag makes the real "first commit" signal fire exactly
    /// once.
    first_token_emitted: bool,
}

#[derive(Default)]
struct TrackedGenerations {
    generations: HashMap<(u64, u64), GenerationTracking>,
}

/// Host adapter implementing [`GenerationLifecycleIngress`] over the host
/// runtime-event engine. `try_submit` never fails inference: submission
/// outcomes are counted in `submission_failures` and otherwise discarded.
pub(crate) struct SkippyGenerationRuntimeEventAdapter {
    tracked: Mutex<TrackedGenerations>,
    submission_failures: AtomicU64,
}

impl SkippyGenerationRuntimeEventAdapter {
    pub(crate) fn new() -> Self {
        Self {
            tracked: Mutex::new(TrackedGenerations::default()),
            submission_failures: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TrackedGenerations> {
        self.tracked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn submit(&self, reservation: &OperationReservation, fact: RuntimeFact) {
        if reservation.ingress().try_submit(fact) == SubmitOutcome::TerminalDeliveryFailed {
            self.submission_failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn begin(&self, start: &GenerationStart) {
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        // Either the OpenAI request root (frontend_request_id set) or a
        // freshly minted root scoped to this generation alone. Either way
        // `group_root` is the parent both the generation reservation and
        // its sibling prefill reservation hang off, matching Task 9's
        // root/prep/load sibling-child pattern for two families that each
        // need their own write-once terminal slot under one logical
        // operation.
        let group_root = start
            .frontend_request_id
            .map_or_else(OperationId::new, OperationId::from_bytes);
        let generation = if start.frontend_request_id.is_some() {
            engine.reserve_child(
                group_root,
                ChildOperationId::new(),
                synthetic_generation_terminal,
            )
        } else {
            engine.reserve_root(group_root, synthetic_generation_terminal)
        };
        let Some(generation) = generation else {
            return;
        };
        self.submit(
            &generation,
            generation_fact(GenerationEventKind::GenerationStarted, empty_data()),
        );

        let prefill = engine.reserve_child(
            group_root,
            ChildOperationId::new(),
            synthetic_prefill_terminal,
        );

        // Session activity is a fire-and-forget StateTransition co-emission
        // keyed on the same observation, not a reservation-tracked entry:
        // §9's `SessionActive`/`SessionIdle` never need a write-once slot.
        let ingress = engine.unreserved_ingress(generation.scope());
        let _ = ingress.try_submit(session_active_fact());

        // Duplicate `Started` for the same numeric `(request_id, session_id)`
        // key: `HashMap::insert` REPLACES the entry, and the displaced
        // `GenerationTracking` (bound to `_stale` rather than discarded
        // inline) is dropped in the field order above -- prefill then
        // generation -- exactly as the empty-cleanup path already does, so
        // the superseded generation's reservations synthesize their own
        // `terminal_not_delivered` terminals correctly rather than leaking
        // or silently vanishing. This is a deliberate choice (the numeric
        // key is caller-generated and cannot itself detect staleness), not
        // an accidental HashMap overwrite: see
        // `duplicate_started_for_the_same_key_cascades_the_superseded_generation`.
        let key = (start.request_id, start.session_id);
        let _stale = self.lock().generations.insert(
            key,
            GenerationTracking {
                generation: Some(generation),
                prefill,
                first_token_emitted: false,
            },
        );
    }

    /// Progress observation. Carries a bounded count only -- never the
    /// committed token IDs -- matching the family adapter's carrying
    /// requirement for a normal runtime event pipeline. The FIRST commit
    /// for a generation also emits §8.10's `first_token_produced`, backed
    /// by this same observation (see `facts::first_token_produced_fact`).
    fn committed(&self, commit: &GenerationCommit) {
        let key = (commit.request_id, commit.session_id);
        let mut tracked = self.lock();
        let Some(tracking) = tracked.generations.get_mut(&key) else {
            return;
        };
        let Some(reservation) = tracking.generation.as_ref() else {
            return;
        };
        if !tracking.first_token_emitted {
            tracking.first_token_emitted = true;
            self.submit(reservation, first_token_produced_fact());
        }
        self.submit(reservation, progress_fact(commit.generated_token_count));
    }

    /// Resolves and REMOVES the tracked entry for `key` exactly once. A
    /// generation whose root was never reserved, or was already resolved,
    /// is a silent no-op. `session_idle` rides on whichever of
    /// `generation`/`prefill` is still live (it is StateTransition-class,
    /// so any live reservation's scope works equally well as the
    /// unreserved-ingress anchor).
    ///
    /// `RuntimeState::export_full_state`'s own `SessionLifecycleObserver`
    /// wiring (`runtime_events/session.rs`) is the single source for
    /// `RuntimeStateExportCompleted`/`Failed` now -- this adapter
    /// deliberately does not derive a second one from
    /// `GenerationReceipt::full_state` (plan task 12 round 8): the receipt
    /// field is set by the exact same `RuntimeState::export_full_state`
    /// call, so re-deriving a fact from it here would double-emit the same
    /// logical export for the one opt-in `with_full_state_digest(true)`
    /// path, and would still emit nothing on export failure (the field is
    /// simply absent, indistinguishable from "not requested").
    ///
    /// Resolution order still matters for LATENCY even though it is no
    /// longer load-bearing for correctness (task 5, review defect D8):
    /// submitting the prefill CHILD terminal before the generation
    /// reservation's own terminal in the non-frontend case, where the
    /// generation reservation itself occupies the group ROOT scope, lets
    /// both settle in the SAME drain pass with no wait at all. Before D8's
    /// fix, resolving a ROOT force-cascaded (dropped without a terminal)
    /// any still-occupied CHILD under it the instant the root's own
    /// terminal drained, so this ordering was the only thing standing
    /// between a real prefill terminal and silent loss.
    /// `engine::drain::release_or_defer` now defers the root's OWN slot
    /// release instead of force-releasing its children: a child that
    /// settles within `CHILD_SETTLE_GRACE` (2s) -- true regardless of
    /// submit order -- is applied and published normally, and one that
    /// never settles gets its own synthesized `terminal_not_delivered`
    /// terminal published through its own slot at grace expiry, never
    /// silently dropped.
    fn finish(
        &self,
        key: (u64, u64),
        generation_terminal: RuntimeFact,
        prefill_terminal: RuntimeFact,
        stop_condition_reached: bool,
    ) {
        let Some(tracking) = self.lock().generations.remove(&key) else {
            return;
        };
        let anchor_scope = tracking
            .generation
            .as_ref()
            .or(tracking.prefill.as_ref())
            .map(OperationReservation::scope);
        // `session_idle`/`stop_condition_reached` must be submitted BEFORE
        // either terminal below: both ride `anchor_scope` unreserved, and
        // (task 4, `.omo/plans/event-system-fixes.md`) a StateTransition
        // fact submitted for an already-settled scope is correctly
        // rejected by the reducer as `OperationSettled` -- so submitting
        // them after the scope's own terminal would mint them a HIGHER
        // ingress sequence than that terminal and silently drop them, now
        // that state-transition facts actually reach the reducer instead
        // of sitting unapplied in the engine's state lane.
        if let Some(scope) = anchor_scope
            && let Some(engine) = runtime_event_engine()
        {
            let ingress = engine.unreserved_ingress(scope);
            let _ = ingress.try_submit(session_idle_fact());
            if stop_condition_reached {
                let _ = ingress.try_submit(stop_condition_reached_fact());
            }
        }
        if let Some(prefill) = tracking.prefill.as_ref() {
            self.submit(prefill, prefill_terminal);
        }
        if let Some(generation) = tracking.generation.as_ref() {
            self.submit(generation, generation_terminal);
        }
    }

    fn abort(&self, abort: &GenerationAbort) {
        self.finish(
            (abort.request_id, abort.session_id),
            abort_terminal_fact(),
            prefill_cancelled_fact(),
            false,
        );
    }

    fn record(&self, receipt: &GenerationReceipt) {
        let stop_condition_reached = matches!(
            receipt.termination,
            skippy_server::frontend::GenerationTermination::CallbackStop
        );
        self.finish(
            (receipt.request_id, receipt.session_id),
            receipt_terminal_fact(receipt),
            prefill_terminal_fact(receipt),
            stop_condition_reached,
        );
    }

    fn finish_lightweight(&self, completion: &GenerationCompletion) {
        let stop_condition_reached = matches!(
            completion.termination,
            skippy_server::frontend::GenerationTermination::CallbackStop
        );
        self.finish(
            (completion.request_id, completion.session_id),
            generation_completion_terminal_fact(
                completion.termination,
                completion.prompt_token_count,
                completion.generated_token_count,
            ),
            prefill_completion_terminal_fact(
                completion.termination,
                completion.request_to_first_token_us,
            ),
            stop_condition_reached,
        );
    }

    /// Receipt-build failure (review defect D1): the generation itself
    /// completed, but `canonical_session_position`/`export_full_state`
    /// bookkeeping failed, so `skippy-server` has no receipt to hand
    /// `record`. Bumps engine health directly (unlike the normal
    /// `finish`-only paths, whose reservation submissions are expected to
    /// succeed) and synthesizes the same `terminal_not_delivered` terminals
    /// a dropped, unresolved reservation would produce -- this is a
    /// deliberate, immediate synthesis of that outcome rather than a
    /// caller-cancelled generation, so it never reuses `abort`'s
    /// `Cancellation` reason.
    fn handle_receipt_unavailable(&self, unavailable: &GenerationAbort) {
        if let Some(engine) = runtime_event_engine() {
            engine.health().bump_terminal_delivery_failed();
        }
        self.submission_failures.fetch_add(1, Ordering::Relaxed);
        self.finish(
            (unavailable.request_id, unavailable.session_id),
            synthetic_generation_terminal(),
            synthetic_prefill_terminal(),
            false,
        );
    }
}

impl Default for SkippyGenerationRuntimeEventAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationLifecycleIngress for SkippyGenerationRuntimeEventAdapter {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
        match observation {
            GenerationLifecycleObservation::Started(start) => self.begin(&start),
            GenerationLifecycleObservation::Committed(commit) => self.committed(&commit),
            GenerationLifecycleObservation::Aborted(abort) => self.abort(&abort),
            GenerationLifecycleObservation::Completed(receipt) => self.record(&receipt),
            GenerationLifecycleObservation::Finished(completion) => {
                self.finish_lightweight(&completion)
            }
            // `#[non_exhaustive]`: a future variant this adapter does not
            // yet model is dropped rather than rejected, matching the
            // trait's at-most-once, never-fails contract.
            _ => {}
        }
        Ok(())
    }

    fn delivery_failures(&self) -> u64 {
        self.submission_failures.load(Ordering::Relaxed)
    }

    fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
        self.handle_receipt_unavailable(unavailable);
    }
}
