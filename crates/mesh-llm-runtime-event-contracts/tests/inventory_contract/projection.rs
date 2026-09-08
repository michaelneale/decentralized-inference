use std::collections::{BTreeMap, BTreeSet};

use crate::support::{Inventory, inventory_event_ids, load_inventory};

#[derive(Debug, Eq, PartialEq)]
enum ProjectionError {
    DuplicateOverride,
    ForbiddenKey,
    MissingEventEntry,
    OverbroadProfile,
    UnknownEvent,
    UnknownFamily,
    UnknownKey,
    UnknownProfile,
}

fn resolve_projection_keys(
    inventory: &Inventory,
) -> Result<BTreeMap<String, BTreeSet<String>>, ProjectionError> {
    let event_ids = inventory_event_ids(inventory);
    let allowed = inventory
        .projected_event_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let forbidden = inventory
        .forbidden_projection_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let profiles = inventory
        .projection_profiles
        .iter()
        .map(|profile| {
            (
                profile.name.as_str(),
                profile.keys.iter().cloned().collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for keys in profiles.values() {
        if !keys.is_disjoint(&forbidden) {
            return Err(ProjectionError::ForbiddenKey);
        }
        if !keys.is_subset(&allowed) {
            return Err(ProjectionError::UnknownKey);
        }
        if keys == &allowed {
            return Err(ProjectionError::OverbroadProfile);
        }
    }
    let mut overrides = BTreeMap::new();
    for rule in &inventory.event_projection_overrides {
        if !profiles.contains_key(rule.profile.as_str()) {
            return Err(ProjectionError::UnknownProfile);
        }
        for event_id in &rule.event_ids {
            if !event_ids.contains(event_id.as_str()) {
                return Err(ProjectionError::UnknownEvent);
            }
            if overrides
                .insert(event_id.as_str(), rule.profile.as_str())
                .is_some()
            {
                return Err(ProjectionError::DuplicateOverride);
            }
        }
    }
    let families = inventory
        .families
        .iter()
        .map(|family| (family.section.as_str(), family))
        .collect::<BTreeMap<_, _>>();
    let base = inventory
        .projected_envelope_keys
        .iter()
        .chain(&inventory.base_projected_event_keys)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut resolved = BTreeMap::new();
    for bullet in &inventory.catalog.spec_bullets {
        let family = families
            .get(bullet.section.as_str())
            .ok_or(ProjectionError::UnknownFamily)?;
        for event_id in &bullet.event_ids {
            let profile_name = overrides
                .get(event_id.as_str())
                .copied()
                .or(family.default_projection_profile.as_deref())
                .ok_or(ProjectionError::MissingEventEntry)?;
            let profile = profiles
                .get(profile_name)
                .ok_or(ProjectionError::UnknownProfile)?;
            resolved.insert(event_id.clone(), base.union(profile).cloned().collect());
        }
    }
    Ok(resolved)
}

#[test]
fn event_projection_allowlists_are_total_exact_private_and_deny_by_default() {
    let inventory = load_inventory();
    let resolved = resolve_projection_keys(&inventory).expect("valid projection contract");
    let event_ids = inventory_event_ids(&inventory)
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let terminal = inventory
        .catalog
        .terminal_event_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let progress = inventory
        .catalog
        .progress_event_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let allowed = inventory
        .projected_envelope_keys
        .iter()
        .chain(&inventory.projected_event_keys)
        .cloned()
        .collect::<BTreeSet<_>>();
    let forbidden = inventory
        .forbidden_projection_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    assert_eq!(resolved.keys().cloned().collect::<BTreeSet<_>>(), event_ids);
    assert!(terminal.is_subset(&inventory_event_ids(&inventory)));
    assert!(progress.is_subset(&inventory_event_ids(&inventory)));
    assert!(terminal.is_disjoint(&progress));
    assert!(
        resolved
            .values()
            .all(|keys| keys.is_subset(&allowed) && keys.is_disjoint(&forbidden))
    );
    assert!(resolved.values().collect::<BTreeSet<_>>().len() >= 7);
    assert!(inventory.families.iter().all(|family| matches!(
        family.privacy_class.as_str(),
        "internal" | "trusted_local" | "aggregate"
    )));
    assert!(!resolved["request_received"].contains("reason_code"));
    assert!(resolved["request_failed"].contains("reason_code"));
    assert!(resolved["generation_progress"].contains("progress"));
    assert!(!resolved["generation_completed"].contains("progress"));
    assert!(!resolved["node_starting"].contains("duration_ms"));
}

#[test]
fn projection_contract_rejects_adversarial_mutations() {
    let original = load_inventory();
    let cases = [
        (
            ProjectionError::MissingEventEntry,
            without_family_default(&original),
        ),
        (
            ProjectionError::UnknownKey,
            with_profile_key(&original, "mystery"),
        ),
        (
            ProjectionError::ForbiddenKey,
            with_profile_key(&original, "prompt"),
        ),
        (
            ProjectionError::OverbroadProfile,
            with_overbroad_profile(&original),
        ),
        (
            ProjectionError::DuplicateOverride,
            with_duplicate_override(&original),
        ),
        (
            ProjectionError::UnknownEvent,
            with_unknown_override_event(&original),
        ),
        (
            ProjectionError::UnknownFamily,
            without_event_family(&original),
        ),
    ];
    for (expected, inventory) in cases {
        assert_eq!(resolve_projection_keys(&inventory), Err(expected));
    }
}

fn without_family_default(original: &Inventory) -> Inventory {
    let mut inventory = original.clone();
    inventory.families[0].default_projection_profile = None;
    inventory
}

fn with_profile_key(original: &Inventory, key: &str) -> Inventory {
    let mut inventory = original.clone();
    inventory.projection_profiles[0].keys.push(key.to_owned());
    inventory
}

fn with_overbroad_profile(original: &Inventory) -> Inventory {
    let mut inventory = original.clone();
    inventory.projection_profiles[0]
        .keys
        .clone_from(&inventory.projected_event_keys);
    inventory
}

fn with_duplicate_override(original: &Inventory) -> Inventory {
    let mut inventory = original.clone();
    let duplicated = inventory.event_projection_overrides[0].event_ids[0].clone();
    inventory.event_projection_overrides[1]
        .event_ids
        .push(duplicated);
    inventory
}

fn with_unknown_override_event(original: &Inventory) -> Inventory {
    let mut inventory = original.clone();
    inventory.event_projection_overrides[0]
        .event_ids
        .push("unknown_event".to_owned());
    inventory
}

fn without_event_family(original: &Inventory) -> Inventory {
    let mut inventory = original.clone();
    inventory.families.remove(0);
    inventory
}
