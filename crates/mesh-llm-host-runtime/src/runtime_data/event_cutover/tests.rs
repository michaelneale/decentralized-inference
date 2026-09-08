use super::super::subscriptions::RuntimeDataDirty;
use super::*;
use matrix::CutoverEntry;
use std::collections::BTreeSet;

const ALL_FIELDS: [FieldId; 6] = [
    FieldId::Status,
    FieldId::Routing,
    FieldId::Processes,
    FieldId::Inventory,
    FieldId::Plugins,
    FieldId::Runtime,
];

#[test]
fn every_field_has_exactly_one_matrix_row() {
    for field in ALL_FIELDS {
        let rows: Vec<&CutoverEntry> = CUTOVER_MATRIX.iter().filter(|e| e.field == field).collect();
        assert_eq!(rows.len(), 1, "field {field:?} must have exactly one row");
    }
}

#[test]
fn no_field_has_cut_over_before_its_producer_task_landed() {
    for entry in &CUTOVER_MATRIX {
        assert_eq!(
            entry.generation,
            Generation::Legacy,
            "field {:?} must not have cut over before its producer task ({:?}) lands",
            entry.field,
            entry.producer_task
        );
    }
}

#[test]
fn reducer_generation_row_always_names_a_producer_task() {
    for entry in &CUTOVER_MATRIX {
        if entry.generation == Generation::Reducer {
            assert!(
                entry.producer_task.is_some(),
                "field {:?} cut over to Reducer without a recorded producer task",
                entry.field
            );
        }
    }
}

/// Mechanically proves Defect 1's fix: every matrix field has a reachable
/// `update_snapshots`/`update_runtime_status` call site somewhere in
/// `collector.rs` or `producers.rs`, not merely an entry in the hand-owned
/// matrix. This is the source of the earlier `FieldId::Models` defect —
/// that row had no such call site and would have failed this test had it
/// existed when this test was written.
#[test]
fn every_matrix_field_has_a_reachable_call_site() {
    let collector_source = include_str!("../collector.rs");
    let producers_source = include_str!("../producers.rs");
    let combined = format!("{collector_source}\n{producers_source}");

    // Iterates `CUTOVER_MATRIX` itself, not a separately hand-maintained
    // field list, so this test cannot silently go stale the way the
    // now-removed `FieldId::Models` row did: any future row is checked
    // automatically, with no second list to remember to update.
    for entry in &CUTOVER_MATRIX {
        let field = entry.field;
        let scope = dirty_scope_name(field);
        let snapshot_call = format!("update_snapshots(RuntimeDataDirty::{scope}");
        let status_call = format!("update_runtime_status(RuntimeDataDirty::{scope}");
        assert!(
            combined.contains(&snapshot_call) || combined.contains(&status_call),
            "field {field:?} (RuntimeDataDirty::{scope}) has no reachable \
             update_snapshots/update_runtime_status call site in collector.rs or producers.rs"
        );
    }
}

/// Red proof for the test above: a fabricated dirty-scope name that
/// cannot possibly appear as a real call site must fail the same
/// assertion shape. This stands in for "add an unreachable row, watch it
/// fail" without mutating the real (reviewed) matrix.
#[test]
fn reachable_call_site_check_fails_for_a_fabricated_scope() {
    let combined = format!(
        "{}\n{}",
        include_str!("../collector.rs"),
        include_str!("../producers.rs")
    );
    let fabricated_scope = "NOT_A_REAL_DIRTY_SCOPE";
    let snapshot_call = format!("update_snapshots(RuntimeDataDirty::{fabricated_scope}");
    let status_call = format!("update_runtime_status(RuntimeDataDirty::{fabricated_scope}");
    assert!(
        !combined.contains(&snapshot_call) && !combined.contains(&status_call),
        "fabricated scope unexpectedly matched a real call site"
    );
}

#[test]
fn field_of_dirty_maps_every_migrated_scope() {
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::STATUS),
        Some(FieldId::Status)
    );
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::ROUTING),
        Some(FieldId::Routing)
    );
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::PROCESSES),
        Some(FieldId::Processes)
    );
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::INVENTORY),
        Some(FieldId::Inventory)
    );
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::PLUGINS),
        Some(FieldId::Plugins)
    );
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::RUNTIME),
        Some(FieldId::Runtime)
    );
}

#[test]
fn field_of_dirty_rejects_models_empty_and_multi_bit_sets() {
    // MODELS is deliberately unmapped -- no production write path reaches
    // update_snapshots under this bit (see the FieldId doc comment).
    assert_eq!(field_of_dirty(RuntimeDataDirty::MODELS), None);
    assert_eq!(field_of_dirty(RuntimeDataDirty::default()), None);
    assert_eq!(
        field_of_dirty(RuntimeDataDirty::STATUS | RuntimeDataDirty::ROUTING),
        None
    );
}

#[test]
fn producer_write_methods_match_producers_source() {
    // Defect 2 fix: cross-check the hand-owned matrix's method NAMES
    // against a mechanical extraction from the real producers.rs source,
    // instead of trusting a hand-copied allowlist to stay in sync.
    let extracted: BTreeSet<String> =
        extract_non_test_write_method_names(include_str!("../producers.rs"))
            .into_iter()
            .collect();
    let listed: BTreeSet<String> = PRODUCER_WRITE_METHODS
        .iter()
        .map(|m| m.method.to_string())
        .collect();
    assert_eq!(
        extracted, listed,
        "PRODUCER_WRITE_METHODS has drifted from producers.rs's real write methods"
    );
    assert_eq!(PRODUCER_WRITE_METHODS.len(), 11);
}

#[test]
fn every_producer_write_method_maps_to_a_matrix_field() {
    for method in &PRODUCER_WRITE_METHODS {
        assert!(
            CUTOVER_MATRIX.iter().any(|e| e.field == method.field),
            "producer write method {} maps to a field with no matrix row",
            method.method
        );
    }
}

#[test]
fn legacy_generation_write_is_applied_unmerged() {
    let health = ShadowHealth::default();
    let (outcome, winner) =
        merge_legacy_publish(FieldId::Status, "legacy-value".to_string(), None, &health);
    assert_eq!(outcome, MergeOutcome::LegacyApplied);
    assert_eq!(winner, "legacy-value");
    assert_eq!(health.stale_legacy_write_count(FieldId::Status), 0);
}

#[test]
fn divergence_is_recorded_but_authority_never_flips_at_legacy_generation() {
    let health = ShadowHealth::default();
    let reducer_projection = "reducer-value".to_string();
    let (outcome, winner) = merge_legacy_publish(
        FieldId::Status,
        "legacy-value".to_string(),
        Some(&reducer_projection),
        &health,
    );
    assert_eq!(health.divergence_count(FieldId::Status), 1);
    assert_eq!(outcome, MergeOutcome::LegacyApplied);
    assert_eq!(winner, "legacy-value");
}

#[test]
fn matching_values_do_not_count_as_divergence() {
    let health = ShadowHealth::default();
    let same = "same-value".to_string();
    merge_legacy_publish(FieldId::Status, same.clone(), Some(&same), &health);
    assert_eq!(health.divergence_count(FieldId::Status), 0);
}

#[test]
fn legacy_generation_write_is_accepted_by_the_real_gate() {
    let health = ShadowHealth::default();
    assert!(should_apply_legacy_write(FieldId::Status, &health));
    assert_eq!(health.stale_legacy_write_count(FieldId::Status), 0);
}

/// Defect 3 fix: this drives `apply_legacy_write_for_generation`, the
/// exact function `should_apply_legacy_write` calls with the real matrix
/// lookup -- not a reimplementation of its rule -- with an explicit
/// `Generation::Reducer` since no real `CUTOVER_MATRIX` row is `Reducer`
/// yet (see `no_field_has_cut_over_before_its_producer_task_landed`).
#[test]
fn reducer_generation_legacy_write_is_rejected_and_counted_stale() {
    let health = ShadowHealth::default();
    let applied = apply_legacy_write_for_generation(FieldId::Status, Generation::Reducer, &health);
    assert!(
        !applied,
        "a reducer-generation field must reject the legacy write"
    );
    assert_eq!(health.stale_legacy_write_count(FieldId::Status), 1);
}

#[test]
fn shadow_health_counters_are_per_field_independent() {
    let health = ShadowHealth::default();
    let a = "a".to_string();
    let b = "b".to_string();
    merge_legacy_publish(FieldId::Status, a.clone(), Some(&b), &health);
    merge_legacy_publish(FieldId::Routing, a.clone(), Some(&a), &health);
    assert_eq!(health.divergence_count(FieldId::Status), 1);
    assert_eq!(health.divergence_count(FieldId::Routing), 0);
}

#[test]
fn shadow_health_does_not_persist_across_a_fresh_instance() {
    let health = ShadowHealth::default();
    merge_legacy_publish(FieldId::Status, 1_u64, Some(&2_u64), &health);
    assert_eq!(health.divergence_count(FieldId::Status), 1);

    let restarted = ShadowHealth::default();
    assert_eq!(restarted.divergence_count(FieldId::Status), 0);
}

#[test]
fn cutover_matrix_is_stable_across_reads() {
    let first: Vec<Generation> = CUTOVER_MATRIX.iter().map(|e| e.generation).collect();
    let second: Vec<Generation> = CUTOVER_MATRIX.iter().map(|e| e.generation).collect();
    assert_eq!(first, second);
}

/// Task 6 (`.omo/plans/event-system-fixes.md`, defect D14): a `Status`
/// write must still apply and publish exactly as before the shadow
/// compare was wired in -- the comparison it now runs after the write is
/// pure observability and must never change `update_snapshots`'s return
/// value.
#[test]
fn status_write_runs_the_production_shadow_compare_without_changing_its_own_result() {
    let collector = super::super::collector::RuntimeDataCollector::new();
    let changed = collector.update_runtime_status(RuntimeDataDirty::STATUS, |status| {
        status.llama_ready = true;
        true
    });
    assert!(changed, "a Legacy-generation Status write must still apply");
}

/// Same production integration point, with a live engine installed and a
/// genuine Status/reducer mismatch (the reducer has no available model at
/// all, while the legacy write claims `llama_ready == true`): the
/// `event_cutover_divergence` engine health counter must increment, and
/// the legacy value already written above must remain untouched (every
/// `CUTOVER_MATRIX` row stays `Legacy`).
#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn status_shadow_compare_bumps_engine_health_on_a_genuine_mismatch() {
    crate::runtime_events::clear_runtime_event_engine();
    let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
    crate::runtime_events::install_runtime_event_engine(engine.clone());

    let collector = super::super::collector::RuntimeDataCollector::new();
    let before = engine.health().snapshot().event_cutover_divergence;
    let changed = collector.update_runtime_status(RuntimeDataDirty::STATUS, |status| {
        status.llama_ready = true;
        true
    });
    let after = engine.health().snapshot().event_cutover_divergence;

    assert!(changed, "the legacy write must still apply");
    assert_eq!(
        after,
        before + 1,
        "a genuine Status/reducer mismatch must bump engine health"
    );
    assert!(
        collector.runtime_status_snapshot().llama_ready,
        "the legacy value stays authoritative regardless of the divergence"
    );

    crate::runtime_events::clear_runtime_event_engine();
}
