use std::collections::{BTreeMap, BTreeSet};

use mesh_llm_runtime_event_contracts::all_event_ids;

use crate::support::{NativeFamilyMapping, load_inventory};

const KNOWN_FEATURE_BITS: [u32; 5] = [32, 33, 34, 35, 36];

/// Expected `(feature_bit, native_kind)` -> `event_id` rows, transcribed
/// from `third_party/llama.cpp/patches/0058-0062-skippy-add-*-events-
/// family-bit-*.patch`'s `SKIPPY_RUNTIME_EVENT_KIND_*` literals. This is
/// the independent cross-check for the inventory table: it is built from
/// the native patch queue's numeric constants, not copied from
/// `runtime_events.toml`, so a row silently dropped or mistyped in the
/// TOML fails this comparison.
///
/// Bits renumbered 30-34 -> 32-36 during the origin/main rebase
/// (`.omo/evidence/event-system-fixes/rebase/`): main independently
/// claimed bits 29/30 for unrelated capabilities, shifting this family's
/// bits +2 in lockstep with `runtime_events.toml`, `skippy-ffi`'s
/// `abi.rs`/`lib.rs`, and `skippy-runtime`'s `capability_probe.rs`.
fn expected_rows() -> Vec<(u32, u32, &'static str)> {
    vec![
        (32, 100, "model_load_phase_changed"),
        (32, 101, "model_memory_allocation_summary"),
        (32, 102, "model_load_phase_changed"),
        (32, 103, "model_load_phase_changed"),
        (32, 104, "model_load_phase_changed"),
        (33, 200, "kv_cache_initialization_completed"),
        (33, 201, "cache_pressure_crossed"),
        (33, 202, "cache_pressure_cleared"),
        (33, 203, "context_capacity_approaching_limit"),
        (33, 204, "context_exhausted"),
        (34, 300, "backend_initialization_completed"),
        (34, 301, "device_ready"),
        (34, 302, "device_degraded"),
        (34, 303, "device_unavailable"),
        (34, 304, "device_recovered"),
        (34, 305, "device_lost"),
        (34, 306, "resource_allocation_completed"),
        (34, 307, "out_of_memory_condition"),
        (34, 308, "backend_fallback_activated"),
        (35, 400, "warning_raised"),
        (35, 401, "warning_cleared"),
        (35, 402, "recoverable_native_failure"),
        (35, 403, "fatal_native_failure"),
        (35, 404, "invariant_protocol_violation"),
        (36, 500, "unload_started"),
        (36, 501, "unload_completed"),
        (36, 502, "unload_failed"),
        (36, 503, "forced_unload"),
        (36, 504, "session_draining_started"),
    ]
}

fn rows(mappings: &[NativeFamilyMapping]) -> BTreeMap<(u32, u32), &str> {
    mappings
        .iter()
        .map(|row| ((row.feature_bit, row.native_kind), row.event_id.as_str()))
        .collect()
}

#[test]
fn every_row_is_a_unique_bit_and_kind_pair() {
    let inventory = load_inventory();
    let keys = inventory
        .native_family_mappings
        .iter()
        .map(|row| (row.feature_bit, row.native_kind))
        .collect::<Vec<_>>();
    let unique = keys.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        keys.len(),
        unique.len(),
        "native_family_mappings has a duplicate (feature_bit, native_kind) pair"
    );
}

#[test]
fn every_row_targets_a_real_catalog_event_id() {
    let inventory = load_inventory();
    let enum_ids = all_event_ids().into_iter().collect::<BTreeSet<_>>();
    for row in &inventory.native_family_mappings {
        assert!(
            enum_ids.contains(row.event_id.as_str()),
            "native_family_mappings row (bit {}, kind {} / {}) targets unknown event_id {:?}",
            row.feature_bit,
            row.native_kind,
            row.native_kind_name,
            row.event_id
        );
    }
}

#[test]
fn every_row_is_scoped_to_a_known_feature_bit() {
    let inventory = load_inventory();
    for row in &inventory.native_family_mappings {
        assert!(
            KNOWN_FEATURE_BITS.contains(&row.feature_bit),
            "native_family_mappings row for native_kind {} ({}) uses feature_bit {}, expected one of bits 32-36",
            row.native_kind,
            row.native_kind_name,
            row.feature_bit
        );
    }
}

/// The acceptance-criterion assertion: every bit-32-36 native kind
/// (bit 31 itself, `RUNTIME_EVENT_REPORTER`, carries no kind catalog of its
/// own -- it only gates installing the process-global reporter) maps to
/// exactly one inventory id, matching the native patch queue exactly.
#[test]
fn every_bit_32_to_36_native_kind_maps_to_exactly_one_inventory_id() {
    let inventory = load_inventory();
    let actual = rows(&inventory.native_family_mappings);
    let expected = expected_rows();

    assert_eq!(
        actual.len(),
        expected.len(),
        "native_family_mappings row count diverges from the native patch queue's kind catalog"
    );
    for (bit, kind, event_id) in expected {
        assert_eq!(
            actual.get(&(bit, kind)),
            Some(&event_id),
            "native kind {kind} under feature bit {bit} should map to {event_id:?}"
        );
    }
}

/// Every id this task's Change bullet names a producer for
/// (`ResourceHealthEventKind`: device ready/degraded/unavailable/
/// recovered/lost, allocation, OOM, fallback; `DiagnosticEventKind`:
/// warning raised/cleared, recoverable, fatal, invariant) is reachable
/// through at least one native_family_mappings row -- the mapping-level
/// half of "every producer path exists"; `system::native_runtime`'s own
/// unit tests prove the sink's match arms actually construct each one.
#[test]
fn every_resource_health_and_diagnostic_id_named_by_the_task_has_a_mapping_row() {
    let inventory = load_inventory();
    let targeted = inventory
        .native_family_mappings
        .iter()
        .map(|row| row.event_id.as_str())
        .collect::<BTreeSet<_>>();

    let required = [
        "device_ready",
        "device_degraded",
        "device_unavailable",
        "device_recovered",
        "device_lost",
        "resource_allocation_completed",
        "out_of_memory_condition",
        "backend_fallback_activated",
        "warning_raised",
        "warning_cleared",
        "recoverable_native_failure",
        "fatal_native_failure",
        "invariant_protocol_violation",
    ];
    for event_id in required {
        assert!(
            targeted.contains(event_id),
            "expected native_family_mappings to produce {event_id:?}"
        );
    }
}
