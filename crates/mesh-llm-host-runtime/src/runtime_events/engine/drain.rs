//! Drain wake entries in ingress-sequence order and apply each through the
//! transactional reducer: only an accepted fact appends a replay frame and
//! fans out to subscribers, so a rejected input never appears on the
//! stream. The reservation is released after the reducer has settled the
//! outcome either way, matching the release-after-ack contract.
//!
//! Every `drain`/`drain_up_to` call also drains the state-transition lane
//! and the diagnostic queue fully, and flushes any progress slots due
//! under the 100 ms export interval (task 4,
//! `.omo/plans/event-system-fixes.md`) -- fixing review defect D2, where
//! only terminal-class facts ever reached the reducer. `engine.drain()`
//! stays the SAME single stable entry point the task-3 driver
//! (`runtime_events::driver`) calls on every `Notify`/fallback tick; this
//! module is the only place that decides what "drain" now does.
//!
//! Task 5 (review defect D8) additionally fixes how a root's release
//! interacts with its still-occupied children: [`release_or_defer`]
//! replaces the old `cascade_children`/`force_complete_child` pair, which
//! force-released every outstanding child the instant the root's own
//! terminal drained WITHOUT ever writing a terminal for it -- so a real,
//! still-in-flight child terminal arriving moments later was rejected as
//! stale (`TerminalDeliveryFailed`) instead of accepted. A root's own
//! terminal still applies and publishes immediately either way; only the
//! ROOT's slot release is now deferred while a child remains occupied,
//! bounded by [`settle_pending_root_releases`]'s `CHILD_SETTLE_GRACE`.
//!
//! Task 6-fix defect A (`.omo/plans/event-system-fixes.md`): a scope whose
//! reservation is actually released here is ALSO evicted from the
//! reducer's `operations` map (`RuntimeEventEngine::evict_operation`,
//! `reducer::evict`), so the map tracks in-flight operations only instead
//! of every settled one forever. [`release_or_defer`] now reports whether
//! it released THIS call (`Some(scope)`) or deferred (`None`); the
//! per-entry loop in [`RuntimeEventEngine::drain_up_to_inner`] batches
//! every scope released this pass and evicts them only AFTER this pass's
//! `pending` facts -- including that scope's own just-drained terminal --
//! have already been applied, so eviction can never race an application
//! that would just re-insert the entry a moment later. [`release_pending_root`]
//! evicts immediately: by the time a deferred root's slot is finally
//! released, its own terminal was already applied in whichever earlier
//! pass first drained it, so there is no such race there.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationId, OperationScope, RuntimeFact};

use super::{ChildSlot, PendingRootRelease, RuntimeEventEngine};
use crate::runtime_events::config::{
    CHILD_SETTLE_GRACE, PROGRESS_EXPORT_INTERVAL, SHUTDOWN_DRAIN_DEADLINE, TOTAL_OPERATION_BOUND,
};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerInput, apply};
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::reservation::{SlotHandle, TerminalRecord};
use crate::runtime_events::wake::WakeEntry;

// Shutdown work is split into small reducer batches so the elapsed deadline
// is observed between bounded pieces of state/diagnostic work. This is an
// implementation chunk size, not a public queue capacity; normal drains
// retain their lane-specific bounds and semantics.
const SHUTDOWN_WORK_CHUNK: usize = 64;

type LaneEntry = (OperationScope, RuntimeFact, u64, bool, Option<SlotHandle>);
type ProgressEntry = (OperationScope, RuntimeFact, u64, SlotHandle);

#[derive(Default)]
struct SequencePrefix {
    wake: Vec<WakeEntry>,
    state: Vec<LaneEntry>,
    diagnostic: Vec<LaneEntry>,
    progress: Vec<ProgressEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrainReport {
    /// Facts successfully reduced and published during this pass.
    pub applied: usize,
    pub left_queued: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DrainPass {
    report: DrainReport,
    /// Queue entries physically popped before reservation, frontier, or
    /// reducer filtering. This drives shutdown continuation independently of
    /// the published-fact count in `report.applied`.
    consumed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShutdownReport {
    pub applied: usize,
    pub started_with: usize,
    pub remaining_after_deadline: usize,
}

impl RuntimeEventEngine {
    /// Drain and apply every currently queued wake entry, the full
    /// state-transition lane, and the full diagnostic queue; flush any
    /// progress slots due under the 100 ms export interval.
    pub fn drain(&self) -> DrainReport {
        self.drain_up_to(None)
    }

    /// Drain and apply at most `max` wake entries, leaving the rest queued
    /// (`None` drains everything currently queued). State, diagnostic, and
    /// due-progress values older than the retained terminal prefix drain in
    /// the same pass; later lane values remain queued so they cannot overtake
    /// that terminal prefix on a subsequent pass.
    pub fn drain_up_to(&self, max: Option<usize>) -> DrainReport {
        let _drain = self
            .drain_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.drain_up_to_inner(max, Instant::now())
    }

    /// Test-only seam for the 100 ms progress-flush gate: identical to
    /// [`Self::drain_up_to`] but takes an explicit `now` instead of
    /// reading the wall clock, so a test can prove "at most one frame per
    /// 100 ms" with pure `Instant` arithmetic -- no real sleep, and no
    /// dependency on `tokio::time::pause` (which does not virtualize
    /// `std::time::Instant::now()`). Mirrors `EngineHealth::publish_at`'s
    /// identical caller-supplied-`now` pattern.
    #[cfg(test)]
    pub(crate) fn drain_up_to_at(&self, max: Option<usize>, now: Instant) -> DrainReport {
        let _drain = self
            .drain_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.drain_up_to_inner(max, now)
    }

    fn drain_up_to_inner(&self, max: Option<usize>, now: Instant) -> DrainReport {
        self.drain_up_to_inner_with_work_budget(max, now, None)
            .report
    }

    /// Variant used by shutdown. A shutdown work budget selects one global
    /// ingress prefix across all lanes, so a later terminal cannot overtake
    /// older state or diagnostic facts. The bounded prefix is applied in
    /// chunks so the cooperative deadline is checked between reducer batches.
    fn drain_up_to_inner_with_work_budget(
        &self,
        max: Option<usize>,
        now: Instant,
        work_budget: Option<usize>,
    ) -> DrainPass {
        // Every fact pulled out of the wake list, state lane, or
        // diagnostic queue this pass is collected here BEFORE any of them
        // is applied, and then applied in ONE ingress-sequence-sorted
        // pass below. Applying per-lane batches back to back (all
        // terminals, then all state-transitions) would let a scope's
        // terminal settle the reducer's per-scope state before an
        // earlier-minted state-transition or diagnostic for that SAME
        // scope -- sitting in a different lane, drained a moment later in
        // program order -- ever applies, spuriously rejecting it as
        // `OperationSettled` even though it was never actually stale.
        // 5th element `reserved` (R1 fix, task 6-fix,
        // `.omo/plans/event-system-fixes.md`): whether this fact arrived
        // through a reservation-bound submission -- always `true` for a
        // drained Terminal record (a terminal is only ever written into an
        // OCCUPIED reservation-table slot) and for a flushed progress slot
        // (progress coalescing itself requires a live `SlotHandle`); comes
        // from the lane's own stored value for state-transition/diagnostic
        // entries, which MAY be `false` (`unreserved_ingress`).
        let mut pending: Vec<(u64, OperationScope, RuntimeFact, bool, bool, bool)> = Vec::new();
        let mut released_now: Vec<OperationScope> = Vec::new();
        let mut applied = 0;
        let consumed;
        {
            // Admission and queue insertion are one short critical section.
            // No reducer, wire serialization, subscriber fan-out, or
            // telemetry work is performed while this gate is held.
            let _ingress = self
                .ingress_gate()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let SequencePrefix {
                wake: entries,
                state: state_entries,
                diagnostic: diagnostic_entries,
                progress: progress_entries,
            } = if let Some(work_budget) = work_budget {
                self.collect_sequence_prefix(max, now, work_budget)
            } else {
                let entries = match max {
                    Some(limit) => self.wake().drain_up_to(limit),
                    None => self.wake().drain_all(),
                };
                // A partial wake drain leaves a terminal prefix in the queue.
                // Do not let a later state/diagnostic/progress lane publish
                // past that prefix; those lane values stay in place for the
                // next pass.
                let lane_cutoff = self.wake().first_sequence().unwrap_or(u64::MAX);
                let state_entries = self
                    .state_lane()
                    .drain_before_limit(lane_cutoff, usize::MAX);
                let diagnostic_entries = self
                    .diagnostic_lane()
                    .drain_before_limit(lane_cutoff, usize::MAX);
                let progress_entries = self.take_due_progress(now, lane_cutoff);
                SequencePrefix {
                    wake: entries,
                    state: state_entries,
                    diagnostic: diagnostic_entries,
                    progress: progress_entries,
                }
            };
            consumed = state_entries.len()
                + diagnostic_entries.len()
                + progress_entries.len()
                + entries.len();
            pending.extend(state_entries.into_iter().map(
                |(scope, fact, sequence, reserved, handle)| {
                    let reservation_valid = !reserved
                        || handle.is_some_and(|handle| {
                            self.table().is_current(handle)
                                && self.table().occupant(handle) == Some(scope)
                                && !self.table().is_cancelled(handle)
                        });
                    (sequence, scope, fact, false, reserved, reservation_valid)
                },
            ));

            pending.extend(diagnostic_entries.into_iter().map(
                |(scope, fact, sequence, reserved, handle)| {
                    let reservation_valid = !reserved
                        || handle.is_some_and(|handle| {
                            self.table().is_current(handle)
                                && self.table().occupant(handle) == Some(scope)
                                && !self.table().is_cancelled(handle)
                        });
                    (sequence, scope, fact, false, reserved, reservation_valid)
                },
            ));

            pending.extend(
                progress_entries
                    .into_iter()
                    .map(|(scope, fact, sequence, handle)| {
                        let reservation_valid = self.table().is_current(handle)
                            && self.table().occupant(handle) == Some(scope)
                            && !self.table().is_cancelled(handle);
                        (sequence, scope, fact, false, true, reservation_valid)
                    }),
            );

            for entry in entries {
                let handle = entry.handle;
                if !self.table().is_current(handle) {
                    continue;
                }
                let Some(scope) = self.table().is_occupied(handle.index) else {
                    continue;
                };
                if !self.table().is_cancelled(handle)
                    && let Some(record) = self.table().terminal_record(handle)
                {
                    pending.push((
                        entry.ingress_sequence,
                        scope,
                        record.fact,
                        record.synthesized,
                        true,
                        true,
                    ));
                }
                if let Some(released_scope) = release_or_defer(self, scope, handle, now) {
                    released_now.push(released_scope);
                }
            }
        }

        pending.sort_by_key(|(sequence, ..)| *sequence);
        for (sequence, scope, fact, synthesized, reserved, reservation_valid) in pending {
            if fact.delivery_class() != mesh_llm_runtime_event_contracts::DeliveryClass::Terminal
                && reserved
                && !reservation_valid
            {
                match fact.delivery_class() {
                    mesh_llm_runtime_event_contracts::DeliveryClass::Progress => {
                        self.health.bump_dropped_progress();
                    }
                    mesh_llm_runtime_event_contracts::DeliveryClass::Diagnostic => {
                        self.health.bump_dropped_diagnostic();
                    }
                    _ => {}
                }
                continue;
            }
            // A coalesced progress value can survive until a later cadence
            // after an unrelated higher-sequence fact has already published.
            // It is explicitly superseded at the publication boundary rather
            // than being applied after a terminal/state transition and
            // regressing the replay cursor. State and terminal entries never
            // take this path and remain lossless within their bounded lanes.
            if fact.delivery_class() == mesh_llm_runtime_event_contracts::DeliveryClass::Progress
                && sequence <= self.published_frontier()
            {
                self.health.bump_dropped_progress();
                continue;
            }
            if self.apply_and_publish_fact(scope, sequence, fact, synthesized, reserved) {
                applied += 1;
            }
        }

        // Defect A (task 6-fix): evict every scope released THIS pass only
        // now that every fact drained this pass has already been applied
        // above -- see the module doc comment for why the ordering matters.
        for scope in released_now {
            self.evict_operation(scope);
        }

        settle_pending_root_releases(self, now);
        DrainPass {
            report: DrainReport {
                applied,
                left_queued: self.wake().len(),
            },
            consumed,
        }
    }

    /// Apply one fact through the transactional reducer and, on
    /// acceptance, append the replay frame and fan it out to subscribers.
    /// Shared by every lane's drain step (terminal, state-transition,
    /// diagnostic, progress) so all four delivery classes go through
    /// EXACTLY the same publication path -- there is no second reducer
    /// path anywhere in the engine.
    fn apply_and_publish_fact(
        &self,
        scope: OperationScope,
        ingress_sequence: u64,
        fact: RuntimeFact,
        synthesized: bool,
        reserved: bool,
    ) -> bool {
        let _publication = self
            .publication_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fact_arc = Arc::new(fact.clone());
        let metadata = fact.metadata();
        let input = ReducerInput {
            scope,
            ingress_sequence,
            native_sequence: metadata.and_then(|metadata| {
                metadata
                    .native_sequence
                    .map(|observation| observation.sequence)
                    .or_else(|| {
                        metadata
                            .native_source
                            .as_ref()
                            .map(|source| source.sequence)
                    })
            }),
            wall_clock_hint: metadata
                .and_then(|metadata| metadata.wall_clock_unix_ns)
                .map(|timestamp| i64::try_from(timestamp).unwrap_or(i64::MAX)),
            synthesized,
            reserved,
            fact,
        };
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ReduceOutcome::Applied(next) = apply(&reducer_state, input) else {
            self.health.bump_reducer_rejected();
            return false;
        };
        // Also-required observability fix (task 6-fix, review finding on
        // top of defect A): `with_operation`'s settled-only capacity
        // backstop used to silently `break` out of its sweep when nothing
        // settled was left to evict, restoring unbounded growth with no
        // counter and no log.
        //
        // R1 CORRECTION (task 6-fix, `.omo/plans/event-system-fixes.md`):
        // the comment that used to sit here claimed release-triggered
        // eviction made that stall "structurally unreachable in the
        // steady state" -- false: six production call sites
        // (`unreserved_ingress` with a fresh `OperationId` per event, no
        // reservation ever backing them) could genuinely drive the
        // settled-only sweep's "nothing left to evict" branch forever.
        // `ReducerSnapshot`'s new `unreserved_order` bounded LRU
        // (`reducer/state.rs`) fixes that by bounding those scopes
        // independently, so the check below is now against
        // `TOTAL_OPERATION_BOUND` (`RESERVATION_TABLE_CAPACITY +
        // UNRESERVED_OPERATION_BOUND`) -- the TRUE combined ceiling both
        // mechanisms together guarantee -- rather than the old
        // reservation-only `RESERVATION_TABLE_CAPACITY`, which legitimate
        // unreserved traffic can now exceed without anything being
        // "stalled". This bump should stay unreachable in practice again,
        // for the right reason this time.
        if next.operation_count() > TOTAL_OPERATION_BOUND {
            self.health.bump_reducer_eviction_stalled();
        }
        *reducer_state = next;
        drop(reducer_state);

        let mut frame = ReplayFrame {
            sequence: EventSequence::new(ingress_sequence),
            rebuild_generation: self.rebuild_generation.load(Ordering::Acquire),
            scope,
            fact: fact_arc,
            recorded_at: Instant::now(),
            // Placeholder, overwritten immediately below. `event_frame`
            // only reads `sequence`/`rebuild_generation`/`scope`/`fact` --
            // never `wire_bytes` itself -- so computing the real bytes
            // against this not-yet-filled `frame` is safe.
            wire_bytes: Arc::from(Vec::new()),
        };
        // Task 9 (`.omo/plans/event-system-fixes.md`, defect D11):
        // serialize this frame's `runtime_event` wire bytes ONCE, here, at
        // push -- not once per subscriber delivery. `frames::event_frame`
        // is the exact byte-for-byte SSE encoder the v1 stream has always
        // used; calling it from here (the one call site granted to
        // engine/drain.rs for this seam) instead of duplicating its logic
        // guarantees these bytes are identical to what a fresh
        // `event_frame` call would have produced (pinned by
        // `runtime_event_api_tests::sample_frames_fixture_is_byte_exact_for_every_frame_type`).
        let encoded = crate::api::routes::runtime_events::frames::event_frame(self, &frame);
        frame.wire_bytes = Arc::from(encoded.into_bytes());

        // Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): `push` now
        // reports the real number of frames it evicted -- a single push can
        // evict more than one (see `replay::ReplayBuffer::push`'s doc
        // comment) -- so every evicted frame is credited here, not one
        // bump per push. `bump_replay_evicted_by` is itself a no-op
        // (including no version bump) when `evicted == 0`.
        let evicted = self.replay.push(frame.clone());
        self.health.bump_replay_evicted_by(evicted as u64);
        self.subscribers.publish(frame);
        self.set_published_frontier(ingress_sequence);
        true
    }

    /// Evict `scope`'s tracked reducer state -- the release-triggered
    /// eviction path (task 6-fix defect A). Callers must only invoke this
    /// once `scope`'s reservation-table slot has ACTUALLY been released
    /// AND every fact drained in the same pass has already been applied
    /// (see the module doc comment and [`release_or_defer`] /
    /// [`release_pending_root`] below, the only two call sites).
    pub(super) fn evict_operation(&self, scope: OperationScope) {
        // Eviction mutates the reducer snapshot that `attach` captures next
        // to replay and the published frontier. Keep the mutation inside the
        // same publication boundary so a reconnect cannot observe a settled
        // scope in one snapshot and its removal in the next at one cursor.
        let _publication = self
            .publication_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *reducer_state = crate::runtime_events::reducer::evict(&reducer_state, scope);
    }

    /// Take every pending progress value when the export cadence is due. This
    /// is called while `ingress_gate` is held, so a progress sequence cannot
    /// be minted and left behind after collection. The values are returned to
    /// the normal ingress-sequence sort and published outside the gate.
    fn take_due_progress(&self, now: Instant, sequence: u64) -> Vec<ProgressEntry> {
        let mut last = self
            .progress_last_flush
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let due = match *last {
            None => true,
            Some(previous) => now.duration_since(previous) >= PROGRESS_EXPORT_INTERVAL,
        };
        if !due {
            return Vec::new();
        }
        *last = Some(now);
        drop(last);
        self.table().take_progress_before(sequence)
    }

    /// Select and drain one global ingress-sequence prefix across the wake
    /// list, state lane, diagnostic queue, and due progress slots. This is
    /// used only for deadline-bounded shutdown work; selecting a per-lane
    /// budget first would let a later terminal overtake older state facts that
    /// remained queued for the next chunk.
    fn collect_sequence_prefix(
        &self,
        max_wake_entries: Option<usize>,
        now: Instant,
        work_budget: usize,
    ) -> SequencePrefix {
        if work_budget == 0 {
            return SequencePrefix::default();
        }

        let wake_probe = self
            .wake()
            .sequences_up_to(max_wake_entries.map(|limit| limit.saturating_add(1)));
        let wake_limit = max_wake_entries.unwrap_or(usize::MAX);
        let lane_cutoff = wake_probe.get(wake_limit).copied().unwrap_or(u64::MAX);
        let mut candidates = wake_probe
            .iter()
            .take(wake_limit)
            .copied()
            .collect::<Vec<_>>();
        candidates.extend(self.state_lane().sequences_before(lane_cutoff));
        candidates.extend(self.diagnostic_lane().sequences_before(lane_cutoff));
        let progress_due = self.progress_flush_due(now);
        if progress_due {
            candidates.extend(self.table().progress_sequences_before(lane_cutoff));
        }
        candidates.sort_unstable();
        candidates.truncate(work_budget);
        let Some(last_sequence) = candidates.last().copied() else {
            return SequencePrefix::default();
        };
        let exclusive_sequence = last_sequence.saturating_add(1);

        let entries = self.wake().drain_before_sequence(exclusive_sequence);
        let state_entries = self
            .state_lane()
            .drain_before_limit(exclusive_sequence, usize::MAX);
        let diagnostic_entries = self
            .diagnostic_lane()
            .drain_before_limit(exclusive_sequence, usize::MAX);
        let progress_entries = if progress_due {
            let progress_entries = self.table().take_progress_before(exclusive_sequence);
            if !progress_entries.is_empty() {
                self.mark_progress_flush(now);
            }
            progress_entries
        } else {
            Vec::new()
        };
        SequencePrefix {
            wake: entries,
            state: state_entries,
            diagnostic: diagnostic_entries,
            progress: progress_entries,
        }
    }

    fn progress_flush_due(&self, now: Instant) -> bool {
        let last = self
            .progress_last_flush
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        last.is_none_or(|previous| now.duration_since(previous) >= PROGRESS_EXPORT_INTERVAL)
    }

    fn mark_progress_flush(&self, now: Instant) {
        *self
            .progress_last_flush
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(now);
    }

    /// Increment `rebuild_generation` and evict every retained replay frame,
    /// simulating a reducer crash/restart recovering into a fresh window.
    pub fn rebuild(&self) -> u64 {
        let _drain = self
            .drain_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _publication = self
            .publication_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let generation = self.rebuild_generation.fetch_add(1, Ordering::AcqRel) + 1;
        let previous_frontier = self.published_frontier();
        self.rebuild_invalidated_through
            .store(previous_frontier, Ordering::Release);
        self.has_rebuild_invalidated_through
            .store(true, Ordering::Release);
        self.health.set_rebuild_generation(generation);
        let evicted = self.replay.evict_all();
        for _ in 0..evicted {
            self.health.bump_replay_evicted();
        }
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let crate::runtime_events::reducer::RebuildOutcome::Rebuilt(next) =
            crate::runtime_events::reducer::rebuild(&reducer_state, generation)
        {
            *reducer_state = next;
        }
        generation
    }

    /// Begin shutdown: block new admission, then drain at most `budget`
    /// accepted work items (`None` drains until the deadline). Entries left
    /// queued past the budget or cooperative deadline are recorded as
    /// shutdown-degraded rather than silently dropped. A root release still
    /// deferred at this point is force-settled after the driver has stopped,
    /// then drained once more instead of retaining its slot for process life.
    pub fn shutdown(&self, budget: Option<usize>) -> ShutdownReport {
        self.shutdown_until(budget, Instant::now() + SHUTDOWN_DRAIN_DEADLINE)
    }

    /// Shutdown variant used by the async driver owner so driver cancellation
    /// and the exclusive final drain share one deadline.
    pub(crate) fn shutdown_until(
        &self,
        budget: Option<usize>,
        deadline: Instant,
    ) -> ShutdownReport {
        let _drain = self
            .drain_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Take the same lock order as drain and cancellation: exclusive
        // drain ownership first, then admission closure. A concurrent
        // cancellation can therefore never hold `drain_gate` while waiting
        // for `ingress_gate` as shutdown waits in the opposite order.
        self.close_admission();
        let started_with = self.wake().len();
        let mut report = DrainReport::default();
        let mut remaining_budget = budget;
        while Instant::now() < deadline && remaining_budget.is_none_or(|remaining| remaining > 0) {
            if !self.drain_shutdown_chunk(deadline, budget, &mut remaining_budget, &mut report) {
                break;
            }
        }
        if !self.pending_root_releases_is_empty() && Instant::now() < deadline {
            let forced_now = Instant::now() + CHILD_SETTLE_GRACE;
            settle_pending_root_releases(self, forced_now);
            while Instant::now() < deadline
                && remaining_budget.is_none_or(|remaining| remaining > 0)
            {
                if !self.drain_shutdown_chunk(deadline, budget, &mut remaining_budget, &mut report)
                {
                    break;
                }
            }
        }
        let (remaining, terminal_remainder) = self.pending_work_counts();
        if remaining > 0 {
            self.health.bump_shutdown_degraded();
            for _ in 0..terminal_remainder {
                self.health.bump_terminal_delivery_failed();
            }
        }
        ShutdownReport {
            applied: report.applied,
            started_with,
            remaining_after_deadline: remaining,
        }
    }

    /// Apply one bounded, globally ordered shutdown prefix. Both shutdown
    /// phases use this helper so synthesis, deadline checks, and budget
    /// accounting cannot drift apart when a deferred root is force-settled.
    fn drain_shutdown_chunk(
        &self,
        deadline: Instant,
        budget: Option<usize>,
        remaining_budget: &mut Option<usize>,
        report: &mut DrainReport,
    ) -> bool {
        if Instant::now() >= deadline || remaining_budget.is_some_and(|remaining| remaining == 0) {
            return false;
        }
        self.synthesize_unsettled_reservations_locked();
        let chunk = remaining_budget.map_or(SHUTDOWN_WORK_CHUNK, |remaining| {
            remaining.min(SHUTDOWN_WORK_CHUNK)
        });
        let pass = self.drain_up_to_inner_with_work_budget(
            budget.map(|limit| limit.min(chunk)),
            Instant::now(),
            Some(chunk),
        );
        report.applied += pass.report.applied;
        if let Some(remaining) = remaining_budget.as_mut() {
            *remaining = (*remaining).saturating_sub(pass.report.applied);
        }
        pass.consumed > 0
    }

    /// Synthesize every occupied slot that still lacks a terminal. The
    /// original family synthesizer is retained in the reservation table, so a
    /// live guard is settled with the same terminal kind as a normal guard
    /// drop. Admission is already closed by the caller; this helper takes the
    /// short admission gate while snapshotting and enqueueing so a late guard
    /// drop cannot race the final synthesis pass.
    fn synthesize_unsettled_reservations_locked(&self) -> usize {
        let _ingress = self
            .ingress_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let unsettled = self.table().unsettled();
        let mut synthesized = 0;
        for unsettled in unsettled {
            let synthetic = (unsettled.synthetic_terminal)();
            let synthetic = match unsettled.scope_identities.as_ref() {
                Some(scope) => synthetic.with_scope(scope),
                None => synthetic,
            };
            let record = TerminalRecord {
                fact: self.fill_ingress_metadata(synthetic, Instant::now()),
                synthesized: true,
            };
            if self.table().occupant(unsettled.handle) == Some(unsettled.scope)
                && self.table().write_terminal(unsettled.handle, record)
            {
                self.wake().push_next(unsettled.handle);
                synthesized += 1;
            }
        }
        synthesized
    }

    fn pending_root_releases_is_empty(&self) -> bool {
        self.pending_root_releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
}

/// Release `handle`'s now-settled slot for `scope` -- or, for a `Root`
/// scope with at least one still-occupied child, DEFER the release
/// instead (review defect D8). A `Child` scope, or a `Root` with no
/// occupied children right now, releases immediately exactly as a plain
/// release always did. Shared by the per-entry drain loop above and
/// `OperationReservation::cancel` (`engine/mod.rs`), so a root released
/// via explicit pre-work cancellation gets the identical deferred-release
/// contract as one released by its own terminal draining.
///
/// Returns `Some(scope)` when this call released the slot immediately --
/// the caller then owns evicting `scope` from the reducer (task 6-fix
/// defect A) once it is safe to (see the module doc comment); returns
/// `None` when the release was deferred, in which case
/// [`release_pending_root`] below evicts once the deferred release
/// actually happens.
pub(super) fn release_or_defer(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: SlotHandle,
    now: Instant,
) -> Option<OperationScope> {
    if let OperationScope::Child { root, .. } = scope {
        engine.table().release(handle);
        // The child is no longer a candidate for root settlement. Remove
        // only this exact generation: the index may already have been
        // reused by an unrelated operation by the time this lock is taken.
        remove_child_slot(engine, root, handle);
        return Some(scope);
    }
    let OperationScope::Root(root) = scope else {
        unreachable!("child scope returned above");
    };
    if has_occupied_children(engine, root) {
        engine
            .pending_root_releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                root,
                PendingRootRelease {
                    handle,
                    deadline: now + CHILD_SETTLE_GRACE,
                },
            );
        return None;
    }
    engine.table().release(handle);
    forget_children(engine, root);
    Some(scope)
}

fn has_occupied_children(engine: &RuntimeEventEngine, root: OperationId) -> bool {
    child_slots(engine, root).into_iter().any(|child| {
        engine
            .table()
            .occupant(SlotHandle {
                index: child.index,
                generation: child.generation,
            })
            .is_some()
    })
}

fn forget_children(engine: &RuntimeEventEngine, root: OperationId) {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&root);
}

fn remove_child_slot(engine: &RuntimeEventEngine, root: OperationId, handle: SlotHandle) {
    let mut children_by_root = engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(children) = children_by_root.get_mut(&root) else {
        return;
    };
    children.retain(|child| child.index != handle.index || child.generation != handle.generation);
    let empty = children.is_empty();
    if empty {
        children_by_root.remove(&root);
    }
}

fn child_slots(engine: &RuntimeEventEngine, root: OperationId) -> Vec<ChildSlot> {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&root)
        .cloned()
        .unwrap_or_default()
}

/// Resolve every root whose own terminal has settled but whose slot
/// release was deferred by [`release_or_defer`]. Called on EVERY drain
/// pass -- the task-3 engine-owned driver ticks this at least every
/// `TUI_RENDER_TICK`, plus immediately on `Notify` -- so a root's grace
/// deadline is enforced without a second background task; the driver's
/// own cadence is the only clock this needs. A root whose children have
/// ALL settled since (drained through the ordinary per-entry loop above,
/// exactly like any other terminal) releases immediately, however much
/// grace time is left. A root still short a child past `deadline` gets
/// each remaining child's OWN synthesized `terminal_not_delivered`
/// written and enqueued through the SAME `write_terminal` + `push_next`
/// mechanism `OperationReservation::drop` already uses for a genuinely-
/// dropped guard, so it is picked up and applied+published by the
/// ordinary per-entry loop on this engine's very next drain call -- there
/// is no second reducer path here, and no fact is applied synchronously
/// inside this function.
fn settle_pending_root_releases(engine: &RuntimeEventEngine, now: Instant) {
    let mut released_roots = Vec::new();
    {
        let _ingress = engine
            .ingress_gate()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let candidates: Vec<(OperationId, SlotHandle, Instant)> = engine
            .pending_root_releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(root, entry)| (*root, entry.handle, entry.deadline))
            .collect();

        for (root, handle, deadline) in candidates {
            let outstanding = occupied_children(engine, root);
            if outstanding.is_empty() {
                if release_pending_root(engine, root, handle) {
                    released_roots.push(root);
                }
                continue;
            }
            if now < deadline {
                continue;
            }
            for child in outstanding {
                synthesize_child_not_delivered(engine, child);
            }
            if release_pending_root(engine, root, handle) {
                released_roots.push(root);
            }
        }
    }
    // Reducer eviction is intentionally outside ingress admission. The drain
    // gate still excludes a concurrent drain/cancel, so no later apply can
    // resurrect one of these released scopes before eviction.
    for root in released_roots {
        engine.evict_operation(OperationScope::root_only(root));
    }
}

fn occupied_children(engine: &RuntimeEventEngine, root: OperationId) -> Vec<ChildSlot> {
    child_slots(engine, root)
        .into_iter()
        .filter(|child| {
            engine
                .table()
                .occupant(SlotHandle {
                    index: child.index,
                    generation: child.generation,
                })
                .is_some()
        })
        .collect()
}

fn release_pending_root(
    engine: &RuntimeEventEngine,
    root: OperationId,
    handle: SlotHandle,
) -> bool {
    if engine.table().occupant(handle).is_none() {
        return false;
    }
    engine.table().release(handle);
    engine
        .pending_root_releases
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&root);
    forget_children(engine, root);
    true
}

/// Write and enqueue `child`'s own synthesized `terminal_not_delivered`
/// terminal -- a no-op if it already settled on its own (a real
/// submission, or a caller-side guard drop), OR if its slot was released
/// and reused by a DIFFERENT operation, between the outstanding-children
/// snapshot in [`settle_pending_root_releases`] and this call.
///
/// Uses `child.generation` -- captured at RESERVE time -- rather than
/// re-reading `current_generation(child.index)` here: re-reading would
/// return whatever generation currently occupies that index, which
/// `write_terminal`'s own generation check would then always match
/// (having just been read from the same slot), landing this child's stale
/// synthesized terminal in a slot a completely different, currently
/// in-flight operation now legitimately owns -- and rejecting THAT
/// operation's real terminal afterward as a duplicate. The reserve-time
/// generation makes `write_terminal` correctly detect the mismatch and
/// no-op instead.
fn synthesize_child_not_delivered(engine: &RuntimeEventEngine, child: ChildSlot) {
    let handle = SlotHandle {
        index: child.index,
        generation: child.generation,
    };
    if engine.table().occupant(handle).is_none() {
        return;
    }
    let synthetic = (child.synthetic_terminal)();
    let synthetic = match engine.table().scope_identities(handle) {
        Some(scope) => synthetic.with_scope(&scope),
        None => synthetic,
    };
    let record = TerminalRecord {
        fact: engine.fill_ingress_metadata(synthetic, Instant::now()),
        synthesized: true,
    };
    if engine.table().write_terminal(handle, record) {
        engine.wake().push_next(handle);
    }
}

// Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): the engine-level
// proof that `apply_and_publish_fact` above credits every frame a single
// `ReplayBuffer::push` evicts, not one bump per push. `engine/mod.rs` has
// no production seam to shrink the replay buffer's `max_age` below the
// frozen 300s `REPLAY_MAX_AGE` (and task 8-fix's grant does not extend to
// adding one there), so `engine_with_tiny_replay_age` below builds a
// `RuntimeEventEngine` by struct literal -- every field matches
// `RuntimeEventEngine::with_capacities` exactly except `replay`, which
// uses a millisecond-scale age bound so the test can force a genuine
// same-push multi-eviction without a real 300s wait.
//
// Task 9 CORRECTION (`.omo/plans/event-system-fixes.md`, defect D11): this
// comment used to claim "only the age dimension can [evict more than one
// frame per push]; the count/byte dimensions can never... since each
// push's own eviction loop already restores the invariant before the next
// push can violate it again" -- true of the COUNT dimension always, and
// was true of the BYTE dimension only while `ReplayBuffer` charged every
// frame the same fixed `APPROX_FRAME_BYTE_COST`. Task 9 replaced that
// fixed cost with each frame's REAL, variable pre-serialized wire-byte
// length (`replay::ReplayFrame::wire_bytes`), so the byte dimension can
// now ALSO evict more than one frame per push -- proven at the
// `ReplayBuffer` level (not here; see the ownership note below) by
// `replay::tests::a_single_large_frame_can_evict_multiple_smaller_frames_via_the_byte_bound`.
// The test below still isolates the AGE dimension specifically
// (`max_bytes: usize::MAX` means the byte bound can never fire here), so
// it remains a valid, UNCHANGED proof of age-driven multi-eviction; it is
// simply no longer the only dimension capable of it.
//
// This is legal because `engine::drain::tests` is a descendant module of
// `engine`, where every field of `RuntimeEventEngine` is defined
// (module-private, not `pub`) -- the same visibility rule that already
// lets this file's own `apply_and_publish_fact` read
// `self.replay`/`self.health` directly.
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, ProcessInstanceId, RuntimeEventIngress, SubmitOutcome,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::runtime_events::engine::lanes::{DiagnosticLane, StateLane};
    use crate::runtime_events::health::EngineHealth;
    use crate::runtime_events::reducer::ReducerSnapshot;
    use crate::runtime_events::replay::ReplayBuffer;
    use crate::runtime_events::reservation::ReservationTable;
    use crate::runtime_events::subscribers::SubscriberRegistry;
    use crate::runtime_events::wake::WakeList;

    fn engine_with_tiny_replay_age(max_age: Duration) -> Arc<RuntimeEventEngine> {
        Arc::new(RuntimeEventEngine {
            table: ReservationTable::new(64),
            wake: WakeList::new(),
            ingress_gate: Mutex::new(()),
            drain_gate: Mutex::new(()),
            publication_gate: Mutex::new(()),
            published_frontier: AtomicU64::new(0),
            rebuild_invalidated_through: AtomicU64::new(0),
            has_rebuild_invalidated_through: AtomicBool::new(false),
            replay: ReplayBuffer::with_bounds(1_000, usize::MAX, max_age),
            subscribers: SubscriberRegistry::with_capacity(64),
            health: EngineHealth::default(),
            children_by_root: Mutex::new(HashMap::new()),
            pending_root_releases: Mutex::new(HashMap::new()),
            shutting_down: AtomicBool::new(false),
            rebuild_generation: AtomicU64::new(0),
            state_lane: StateLane::default(),
            diagnostic_lane: DiagnosticLane::default(),
            reducer_state: Mutex::new(ReducerSnapshot::empty()),
            process_instance: ProcessInstanceId::new(),
            process_started: Instant::now(),
            telemetry: OnceLock::new(),
            progress_diagnostic_class_bypass: AtomicBool::new(false),
            notify: Notify::new(),
            progress_last_flush: Mutex::new(None),
            ingress_latency: crate::runtime_events::ingress_latency::IngressLatencyReservoir::new(),
        })
    }

    fn distinct_state_transition_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
    }

    /// A fresh `OperationScope` each call, submitted unreserved (bypassing
    /// the reservation table entirely) so coalescing never merges it with
    /// a sibling call: the state lane keys on `(OperationScope, kind)`, and
    /// every call here mints a brand new `OperationId`.
    fn submit_one(engine: &Arc<RuntimeEventEngine>) {
        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(distinct_state_transition_fact());
        assert_eq!(outcome, SubmitOutcome::Accepted);
    }

    /// Fails at the parent commit (`apply_and_publish_fact` bumping
    /// `EngineHealth::replay_evicted` by exactly one per push regardless of
    /// the real eviction count) and passes once it reports the real count.
    /// Three distinct facts are drained together so each publishes its own
    /// replay frame within microseconds of the others (well under the tiny
    /// age bound); after real wall-clock time passes that bound, a fourth
    /// push must evict all three in ONE `ReplayBuffer::push` call.
    #[test]
    fn a_single_push_that_evicts_several_stale_frames_credits_every_one_at_the_engine_level() {
        let engine = engine_with_tiny_replay_age(Duration::from_millis(5));

        submit_one(&engine);
        submit_one(&engine);
        submit_one(&engine);
        engine.drain();
        assert_eq!(
            engine.health().snapshot().replay_evicted,
            0,
            "all three frames were recorded within microseconds of each \
             other, well under the 5ms age bound -- nothing is stale yet"
        );
        assert_eq!(engine.replay().len(), 3);

        // Real wall-clock time passing the tiny age bound, well past it for
        // safety margin against scheduling jitter on a loaded machine.
        std::thread::sleep(Duration::from_millis(50));

        submit_one(&engine);
        engine.drain();

        assert_eq!(
            engine.health().snapshot().replay_evicted,
            3,
            "one push evicted three stale frames; the engine-level EngineHealth \
             counter must credit every one of them, not bump by one per push"
        );
        assert_eq!(engine.replay().len(), 1, "only the fresh frame remains");
    }
}
