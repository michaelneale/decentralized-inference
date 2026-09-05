use std::collections::BTreeSet;
use std::fs;

use sha2::{Digest, Sha256};

use crate::support::{crate_package_names, crate_root, load_inventory, repo_root};

#[test]
fn carrier_reason_callback_and_terminal_contracts_are_complete() {
    let inventory = load_inventory();
    let carriers = inventory
        .carriers
        .iter()
        .map(|item| item.name.as_str())
        .collect::<BTreeSet<_>>();
    let reasons = inventory
        .reason_codes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let callbacks = inventory
        .native_callback_kinds
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let terminals = inventory
        .terminal_outcome_mappings
        .iter()
        .map(|item| item.variant.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(inventory.carriers.len(), 19);
    assert_eq!(carriers.len(), 19);
    assert!(inventory.carriers.iter().all(|carrier| matches!(
        carrier.location.as_str(),
        "RuntimeEventEnvelope" | "FamilyFact" | "NativeSourceEnvelope"
    )));
    assert_eq!(
        reasons,
        BTreeSet::from([
            "artifact_io_failure",
            "backend_initialization_failure",
            "cancellation",
            "context_exhausted",
            "device_unavailable",
            "incompatible_abi_or_feature_set",
            "internal_runtime_failure",
            "invalid_configuration",
            "missing_artifact",
            "model_format_or_load_failure",
            "out_of_memory",
            "process_crash",
            "reservation_exhausted",
            "resource_allocation_failure",
            "stage_unavailable",
            "terminal_not_delivered",
            "timeout",
            "unknown_failure",
            "unsupported_capability",
        ])
    );
    assert_eq!(
        callbacks,
        BTreeSet::from([
            "BackendDeviceSelected",
            "ModelOpenFailedHandled",
            "ModelOpenFinished",
            "ModelOpenProgress",
            "ModelOpenStarted",
        ])
    );
    assert_eq!(
        terminals,
        BTreeSet::from([
            "Cancelled",
            "Completed",
            "CompletedWithStatus",
            "CompletedWithUsage",
            "Dropped",
            "Failed",
            "FailedWithStatus",
            "Rejected",
            "RejectedWithStatus",
        ])
    );
    assert!(
        inventory
            .terminal_outcome_mappings
            .iter()
            .all(|mapping| !mapping.event_id.is_empty() && !mapping.outcome.is_empty())
    );
}

#[test]
fn source_paths_packages_privacy_and_delivery_are_valid() {
    let inventory = load_inventory();
    let root = repo_root();
    let package_names = crate_package_names();
    let privacy = inventory
        .privacy_classes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let delivery = inventory
        .delivery_classes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();

    assert_eq!(inventory.schema_version, 1);
    assert_eq!(inventory.families.len(), 15);
    assert!(
        inventory
            .amendment_procedure
            .contains("schema-version review")
    );
    for family in &inventory.families {
        assert!(
            family
                .source_paths
                .iter()
                .all(|path| root.join(path).is_file()),
            "invalid source path in {}",
            family.section
        );
        assert!(
            family
                .owning_crates
                .iter()
                .all(|package| package_names.contains(package)),
            "invalid Cargo package in {}",
            family.section
        );
        assert!(!family.producer_symbols.is_empty());
        assert!(privacy.contains(family.privacy_class.as_str()));
        assert!(delivery.contains(family.default_delivery_class.as_str()));
        assert!(!family.enum_name.is_empty() && !family.authoritative_outcome.is_empty());
        assert!(!family.reducer_effect.is_empty() && !family.projection.is_empty());
        assert_eq!(family.test, "inventory_contract");
        assert!((3..=12).contains(&family.migration_task));
    }
}

#[test]
fn spec_hash_and_dependency_leaf_contract_are_current() {
    let inventory = load_inventory();
    let root = repo_root();
    let spec = fs::read(root.join(&inventory.spec_path)).expect("ratified spec");
    let digest = Sha256::digest(spec)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest_text =
        fs::read_to_string(crate_root().join("Cargo.toml")).expect("contract crate manifest");
    let manifest: toml::Value =
        toml::from_str(&manifest_text).expect("contract crate manifest TOML");

    assert_eq!(inventory.spec_sha256, digest);
    let dependencies = manifest["dependencies"]
        .as_table()
        .expect("contract crate dependencies")
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(dependencies, BTreeSet::from(["uuid"]));
}
