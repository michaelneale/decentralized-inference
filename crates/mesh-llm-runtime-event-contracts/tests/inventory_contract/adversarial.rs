use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use mesh_llm_runtime_event_contracts::all_event_ids;
use sha2::{Digest, Sha256};

use crate::support::{
    Inventory, SpecManifest, crate_package_names, inventory_event_ids, load_inventory,
    load_manifest, repo_root,
};

#[derive(Debug, Eq, PartialEq)]
enum AcceptanceError {
    DuplicateExpansion,
    EmptyExpansion,
    ExactEventSet,
    InvalidPackage,
    InvalidPath,
    InvalidPrivacy,
    ManifestSet,
    MissingCarrier,
    MissingReason,
    SpecHash,
}

fn validate_acceptance(
    inventory: &Inventory,
    manifest: &SpecManifest,
) -> Result<(), AcceptanceError> {
    let manifest_keys = manifest
        .bullets
        .iter()
        .map(|bullet| (bullet.section.as_str(), bullet.ordinal))
        .collect::<BTreeSet<_>>();
    let expansion_keys = inventory
        .catalog
        .spec_bullets
        .iter()
        .map(|bullet| (bullet.section.as_str(), bullet.ordinal))
        .collect::<BTreeSet<_>>();
    if manifest.bullet_count != 137 || manifest_keys != expansion_keys {
        return Err(AcceptanceError::ManifestSet);
    }
    if inventory
        .catalog
        .spec_bullets
        .iter()
        .any(|bullet| bullet.event_ids.is_empty())
    {
        return Err(AcceptanceError::EmptyExpansion);
    }
    let expected = all_event_ids().into_iter().collect::<BTreeSet<_>>();
    if inventory_event_ids(inventory) != expected {
        return Err(AcceptanceError::ExactEventSet);
    }
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
        .filter_map(|(id, count)| (count > 1).then_some(id))
        .collect::<BTreeSet<_>>();
    if duplicates != BTreeSet::from(["model_preparation_completed"]) {
        return Err(AcceptanceError::DuplicateExpansion);
    }
    if inventory.carriers.len() != 19 {
        return Err(AcceptanceError::MissingCarrier);
    }
    if !inventory
        .reason_codes
        .iter()
        .any(|code| code == "terminal_not_delivered")
        || !inventory
            .reason_codes
            .iter()
            .any(|code| code == "reservation_exhausted")
    {
        return Err(AcceptanceError::MissingReason);
    }
    let root = repo_root();
    if inventory
        .families
        .iter()
        .flat_map(|family| &family.source_paths)
        .any(|path| !root.join(path).is_file())
    {
        return Err(AcceptanceError::InvalidPath);
    }
    let package_names = crate_package_names();
    if inventory
        .families
        .iter()
        .flat_map(|family| &family.owning_crates)
        .any(|package| !package_names.contains(package))
    {
        return Err(AcceptanceError::InvalidPackage);
    }
    let privacy = inventory
        .privacy_classes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if inventory
        .families
        .iter()
        .any(|family| !privacy.contains(family.privacy_class.as_str()))
    {
        return Err(AcceptanceError::InvalidPrivacy);
    }
    let spec = fs::read(root.join(&inventory.spec_path)).expect("ratified spec");
    let digest = Sha256::digest(spec)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if inventory.spec_sha256 != digest {
        return Err(AcceptanceError::SpecHash);
    }
    Ok(())
}

#[test]
fn acceptance_checker_rejects_every_required_adversarial_class() {
    let inventory = load_inventory();
    let manifest = load_manifest();
    let cases = [
        (
            AcceptanceError::ManifestSet,
            mutate_manifest(&inventory, &manifest),
        ),
        (
            AcceptanceError::EmptyExpansion,
            mutate_inventory(&inventory, &manifest, |value| {
                value.catalog.spec_bullets[0].event_ids.clear()
            }),
        ),
        (
            AcceptanceError::ExactEventSet,
            mutate_inventory(&inventory, &manifest, |value| {
                value.catalog.spec_bullets[0].event_ids[0] = "unknown_event".to_owned()
            }),
        ),
        (
            AcceptanceError::DuplicateExpansion,
            mutate_inventory(&inventory, &manifest, |value| {
                value.catalog.spec_bullets[1]
                    .event_ids
                    .push("runtime_resolution_started".to_owned())
            }),
        ),
        (
            AcceptanceError::MissingCarrier,
            mutate_inventory(&inventory, &manifest, |value| {
                value.carriers.remove(0);
            }),
        ),
        (
            AcceptanceError::MissingReason,
            mutate_inventory(&inventory, &manifest, |value| {
                value
                    .reason_codes
                    .retain(|code| code != "terminal_not_delivered")
            }),
        ),
        (
            AcceptanceError::InvalidPath,
            mutate_inventory(&inventory, &manifest, |value| {
                value.families[0].source_paths[0] = "missing.rs".to_owned()
            }),
        ),
        (
            AcceptanceError::InvalidPackage,
            mutate_inventory(&inventory, &manifest, |value| {
                value.families[0].owning_crates[0] = "missing-package".to_owned()
            }),
        ),
        (
            AcceptanceError::InvalidPrivacy,
            mutate_inventory(&inventory, &manifest, |value| {
                value.families[0].privacy_class = "public".to_owned()
            }),
        ),
        (
            AcceptanceError::SpecHash,
            mutate_inventory(&inventory, &manifest, |value| {
                value.spec_sha256 = "stale".to_owned()
            }),
        ),
    ];
    for (expected, (mutated_inventory, mutated_manifest)) in cases {
        assert_eq!(
            validate_acceptance(&mutated_inventory, &mutated_manifest),
            Err(expected)
        );
    }
}

fn mutate_manifest(inventory: &Inventory, manifest: &SpecManifest) -> (Inventory, SpecManifest) {
    let mut manifest = manifest.clone();
    manifest.bullets.remove(0);
    (inventory.clone(), manifest)
}

fn mutate_inventory(
    inventory: &Inventory,
    manifest: &SpecManifest,
    mutate: impl FnOnce(&mut Inventory),
) -> (Inventory, SpecManifest) {
    let mut inventory = inventory.clone();
    mutate(&mut inventory);
    (inventory, manifest.clone())
}
