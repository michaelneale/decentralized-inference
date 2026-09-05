//! Generation-tagged field-level merge for the collector's publish path.
//!
//! Legacy producers publish whole snapshots, so the merge decision is made
//! per dirty scope rather than per struct field: a field at legacy
//! generation takes the legacy write; a field at reducer generation
//! ignores the legacy write (counting it as stale) and keeps whatever the
//! reducer projection already holds. Shadow comparison never changes which
//! generation wins — it only records divergence health.

use std::collections::HashMap;
use std::sync::Mutex;

use super::matrix::{FieldId, Generation, generation_of};

/// Per-field health counters for the generation-tagged merge. Not
/// persisted: like the existing dirty-bit subscription state, this resets
/// on process restart, which is the correct behavior — the compile-time
/// `CUTOVER_MATRIX` is the only durable authority, not accumulated health.
///
/// `divergence` was `#[cfg(test)]`-only through task 5: no producer task
/// had landed a reducer projection to compare against, so the comparison
/// only ever ran in unit tests. Task 6
/// (`.omo/plans/event-system-fixes.md`, defect D14) lands the reducer's
/// per-category domain state and promotes this counter to production for
/// the `Status`/`Inventory` fields (see `collector::update_snapshots`'s
/// shadow-compare calls), while every `CUTOVER_MATRIX` row stays `Legacy`.
#[derive(Default)]
pub(crate) struct ShadowHealth {
    divergence: Mutex<HashMap<FieldId, u64>>,
    stale_legacy_writes: Mutex<HashMap<FieldId, u64>>,
}

impl ShadowHealth {
    #[cfg(test)]
    pub(crate) fn divergence_count(&self, field: FieldId) -> u64 {
        Self::count(&self.divergence, field)
    }

    #[cfg(test)]
    pub(crate) fn stale_legacy_write_count(&self, field: FieldId) -> u64 {
        Self::count(&self.stale_legacy_writes, field)
    }

    #[cfg(test)]
    fn count(counters: &Mutex<HashMap<FieldId, u64>>, field: FieldId) -> u64 {
        *counters
            .lock()
            .expect("shadow health counter lock poisoned")
            .get(&field)
            .unwrap_or(&0)
    }

    /// Records the divergence locally AND bumps the runtime-event engine's
    /// `event_cutover_divergence` counter when an engine is installed (a
    /// no-op otherwise, e.g. in a unit test with no engine running) --
    /// task 6's "count divergences in engine health counters" half.
    fn record_divergence(&self, field: FieldId) {
        *self
            .divergence
            .lock()
            .expect("shadow health divergence lock poisoned")
            .entry(field)
            .or_insert(0) += 1;
        if let Some(engine) = crate::runtime_events::runtime_event_engine() {
            engine.health().bump_event_cutover_divergence();
        }
    }

    fn record_stale_legacy_write(&self, field: FieldId) {
        *self
            .stale_legacy_writes
            .lock()
            .expect("shadow health stale-write lock poisoned")
            .entry(field)
            .or_insert(0) += 1;
    }
}

/// Which generation's write actually took effect for one publish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeOutcome {
    LegacyApplied,
    ReducerApplied,
}

/// Gate one legacy whole-snapshot publish for `field` against its
/// compile-time cutover generation. This is the real
/// `collector::update_snapshots` integration point: it runs BEFORE the
/// mutation closure so a reducer-generation field's legacy write never
/// touches the snapshot at all (not merged and reverted — never applied).
pub(crate) fn should_apply_legacy_write(field: FieldId, health: &ShadowHealth) -> bool {
    apply_legacy_write_for_generation(field, generation_of(field), health)
}

/// The actual gate rule, factored out so tests can drive the real
/// `Reducer` arm directly with an explicit generation — no production
/// `CUTOVER_MATRIX` row is `Reducer` today (see
/// `tests::no_field_has_cut_over_before_its_producer_task_landed`), so
/// `should_apply_legacy_write` alone can never reach that arm in a test
/// run. This is the same code `should_apply_legacy_write` runs, not a
/// reimplementation of it.
pub(crate) fn apply_legacy_write_for_generation(
    field: FieldId,
    generation: Generation,
    health: &ShadowHealth,
) -> bool {
    match generation {
        Generation::Legacy => true,
        Generation::Reducer => {
            health.record_stale_legacy_write(field);
            false
        }
    }
}

/// Full field-level merge including shadow comparison. Called from
/// production for `Status`/`Inventory` (`collector::update_snapshots`)
/// now that task 6 lands a real reducer projection to compare against for
/// those two fields, and directly by unit tests with synthetic typed
/// values for every field (see `tests`). Divergence is recorded but never
/// flips which generation's value wins.
pub(crate) fn merge_legacy_publish<T: Clone + PartialEq>(
    field: FieldId,
    legacy_value: T,
    reducer_projection: Option<&T>,
    health: &ShadowHealth,
) -> (MergeOutcome, T) {
    if let Some(reducer_value) = reducer_projection
        && *reducer_value != legacy_value
    {
        health.record_divergence(field);
    }

    match generation_of(field) {
        Generation::Legacy => (MergeOutcome::LegacyApplied, legacy_value),
        Generation::Reducer => {
            health.record_stale_legacy_write(field);
            let winner = reducer_projection.cloned().unwrap_or(legacy_value);
            (MergeOutcome::ReducerApplied, winner)
        }
    }
}
