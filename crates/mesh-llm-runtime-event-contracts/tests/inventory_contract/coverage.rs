use std::collections::{BTreeMap, BTreeSet};

use mesh_llm_runtime_event_contracts::all_event_ids;

use crate::support::{inventory_event_ids, load_inventory, load_manifest};

#[test]
fn spec_manifest_and_expansions_cover_section_eight_exactly() {
    let inventory = load_inventory();
    let manifest = load_manifest();
    let manifest_keys = manifest
        .bullets
        .iter()
        .map(|bullet| {
            (
                (bullet.section.as_str(), bullet.ordinal),
                bullet.text.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let expansion_keys = inventory
        .catalog
        .spec_bullets
        .iter()
        .map(|bullet| {
            (
                (bullet.section.as_str(), bullet.ordinal),
                bullet.event_ids.as_slice(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.source, inventory.spec_path);
    assert_eq!(manifest.bullet_count, 137);
    assert_eq!(manifest.bullet_count, manifest_keys.len());
    assert_eq!(
        manifest_keys.keys().collect::<Vec<_>>(),
        expansion_keys.keys().collect::<Vec<_>>()
    );
    assert!(
        manifest
            .bullets
            .iter()
            .all(|bullet| !bullet.family.is_empty())
    );
    assert!(expansion_keys.values().all(|ids| !ids.is_empty()));
}

#[test]
fn handwritten_family_enums_and_inventory_are_bidirectionally_total() {
    let inventory = load_inventory();
    let enum_ids = all_event_ids();
    let enum_set = enum_ids.iter().copied().collect::<BTreeSet<_>>();
    let inventory_set = inventory_event_ids(&inventory);

    assert_eq!(
        enum_ids.len(),
        enum_set.len(),
        "handwritten enum IDs must be unique"
    );
    assert_eq!(enum_set, inventory_set);
    assert!(!inventory_set.contains("model_prepared"));
    assert!(inventory_set.contains("model_preparation_completed"));
}

#[test]
fn duplicate_bullet_expansions_have_explicit_same_fact_rulings() {
    let inventory = load_inventory();
    let mut counts = BTreeMap::new();
    for event_id in inventory
        .catalog
        .spec_bullets
        .iter()
        .flat_map(|bullet| &bullet.event_ids)
    {
        *counts.entry(event_id.as_str()).or_insert(0_u32) += 1;
    }
    let duplicates = counts
        .into_iter()
        .filter_map(|(event_id, count)| (count > 1).then_some(event_id))
        .collect::<BTreeSet<_>>();
    let ruled = inventory
        .same_fact_rulings
        .iter()
        .filter(|ruling| !ruling.ruling.is_empty())
        .flat_map(|ruling| ruling.ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();

    assert_eq!(duplicates, BTreeSet::from(["model_preparation_completed"]));
    assert!(duplicates.is_subset(&ruled));
}
