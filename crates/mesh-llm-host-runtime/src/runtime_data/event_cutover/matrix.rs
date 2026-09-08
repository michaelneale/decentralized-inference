//! Compile-time field → generation → producer-task cutover matrix.
//!
//! `FieldId` is deliberately keyed at `RuntimeDataDirty`'s scope
//! granularity (the collector's actual publish-path funnel point in
//! `collector::update_snapshots`), not at the granularity of the eleven
//! individual `RuntimeDataProducer` write methods enumerated in the task 5
//! plan bullet — several of those methods publish under the same dirty
//! scope (`mark_status_dirty`, `publish_runtime_status`, and
//! `publish_local_processes` all publish under `STATUS`), so the scope is
//! the only place a single generation authority can be enforced without
//! duplicating bookkeeping per method. See `producer_write_methods.rs` for
//! the full eleven-method → scope mapping the plan's matrix column needs.

use super::super::subscriptions::RuntimeDataDirty;

/// Compile-time authority for one migrated collector dirty-scope. Never
/// advances to `Reducer` until the scope's producer task lands a real
/// reducer projection — flipping this at runtime is explicitly forbidden
/// by the task 5 Must-NOT list; it changes only by editing `CUTOVER_MATRIX`
/// in a future commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Generation {
    Legacy,
    // Constructed only when a future commit edits `CUTOVER_MATRIX` to cut
    // a field over after its producer task lands parity evidence; the
    // variant must exist now so `should_apply_legacy_write`'s match stays
    // exhaustive and so today's tests can prove the Reducer-arm rule.
    #[allow(dead_code)]
    Reducer,
}

/// One migrated `RuntimeDataDirty` scope with a reachable production
/// enforcement path through `collector::update_snapshots`.
///
/// `RuntimeDataDirty::MODELS` is deliberately NOT a `FieldId` variant.
/// Verification traced every production write path and found no producer
/// that publishes under `RuntimeDataDirty::MODELS` through
/// `update_snapshots`/`update_runtime_status` (the only funnel this gate
/// can intercept) — the sole `MODELS` reference is
/// `RuntimeDataProducer::mark_models_dirty`, which is `#[cfg(test)]`-only
/// and calls `collector::mark_dirty` (a bare subscription-version bump
/// with NO snapshot mutation, see `collector.rs:86-88`), never
/// `update_snapshots`. The actual models data (`models`,
/// `available_models`, `requested_models`, `serving_models`,
/// `hosted_models` on `RuntimeStatusSnapshot`) is written under the
/// `STATUS` bit by `publish_runtime_status`, which already has a `Status`
/// row. A `Models` matrix row would assert a generation authority the gate
/// can never enforce — exactly the "silently accept post-cutover legacy
/// writes" failure mode the Must-NOT list forbids. See
/// `decisions.md` (2026-09-02, task 5 correction) for the full trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FieldId {
    Status,
    Routing,
    Processes,
    Inventory,
    Plugins,
    Runtime,
}

/// One row of the field cutover matrix.
pub(crate) struct CutoverEntry {
    pub field: FieldId,
    pub generation: Generation,
    /// Plan task id that must land this scope's reducer producer before
    /// `generation` may advance past `Legacy`. `None` means no producer
    /// task is yet planned for this scope. Read by the matrix invariant
    /// tests and the QA evidence dump; not consulted by production logic
    /// (the compile-time `generation` field is the sole runtime authority).
    #[allow(dead_code)]
    pub producer_task: Option<&'static str>,
}

/// The one compile-time authority table for every migrated dirty scope.
/// Every row is `Generation::Legacy` today: none of tasks 9-12 (the
/// reducer producer tasks) has landed yet, and the Must-NOT list forbids
/// advancing a field before its own producer task lands.
pub(crate) const CUTOVER_MATRIX: [CutoverEntry; 6] = [
    CutoverEntry {
        field: FieldId::Status,
        generation: Generation::Legacy,
        producer_task: Some("9"),
    },
    CutoverEntry {
        field: FieldId::Routing,
        generation: Generation::Legacy,
        producer_task: Some("11"),
    },
    CutoverEntry {
        field: FieldId::Processes,
        generation: Generation::Legacy,
        producer_task: Some("9"),
    },
    CutoverEntry {
        field: FieldId::Inventory,
        generation: Generation::Legacy,
        producer_task: Some("9"),
    },
    CutoverEntry {
        field: FieldId::Plugins,
        generation: Generation::Legacy,
        producer_task: None,
    },
    CutoverEntry {
        field: FieldId::Runtime,
        generation: Generation::Legacy,
        producer_task: Some("10"),
    },
];

/// Look up `field`'s compile-time authority. Every `FieldId` variant has
/// exactly one `CUTOVER_MATRIX` row (proved by
/// `tests::every_field_has_exactly_one_matrix_row`), so the fallback to
/// `Generation::Legacy` is unreachable in practice and exists only to keep
/// this a total function.
pub(crate) fn generation_of(field: FieldId) -> Generation {
    CUTOVER_MATRIX
        .iter()
        .find(|entry| entry.field == field)
        .map_or(Generation::Legacy, |entry| entry.generation)
}

/// Map a single-bit `RuntimeDataDirty` value to its migrated `FieldId`.
/// Returns `None` for the empty dirty set or a multi-bit combination,
/// neither of which `collector::update_snapshots` ever passes in practice
/// (every call site publishes exactly one scope per write).
pub(crate) fn field_of_dirty(dirty: RuntimeDataDirty) -> Option<FieldId> {
    match dirty {
        RuntimeDataDirty::STATUS => Some(FieldId::Status),
        RuntimeDataDirty::ROUTING => Some(FieldId::Routing),
        RuntimeDataDirty::PROCESSES => Some(FieldId::Processes),
        RuntimeDataDirty::INVENTORY => Some(FieldId::Inventory),
        RuntimeDataDirty::PLUGINS => Some(FieldId::Plugins),
        RuntimeDataDirty::RUNTIME => Some(FieldId::Runtime),
        // RuntimeDataDirty::MODELS is deliberately unmapped -- see the
        // `FieldId` doc comment: no production write path publishes MODELS
        // through `update_snapshots`, so this gate cannot enforce it.
        _ => None,
    }
}

/// The `RuntimeDataDirty` scope name backing `field`, as it appears in
/// `RuntimeDataDirty::<NAME>` source text. Used only to mechanically prove
/// each matrix row has a reachable `update_snapshots`/`update_runtime_status`
/// call site in `tests::every_matrix_field_has_a_reachable_call_site`.
#[cfg(test)]
pub(crate) fn dirty_scope_name(field: FieldId) -> &'static str {
    match field {
        FieldId::Status => "STATUS",
        FieldId::Routing => "ROUTING",
        FieldId::Processes => "PROCESSES",
        FieldId::Inventory => "INVENTORY",
        FieldId::Plugins => "PLUGINS",
        FieldId::Runtime => "RUNTIME",
    }
}
