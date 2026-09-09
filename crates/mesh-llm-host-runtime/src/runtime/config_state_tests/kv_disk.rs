use super::*;
use mesh_llm_config::{KvDiskTierConfig, KvDiskTierMode};
use std::path::PathBuf;

fn disk_config() -> KvDiskTierConfig {
    KvDiskTierConfig {
        mode: Some(KvDiskTierMode::Fixed),
        directory: Some(PathBuf::from("/var/lib/mesh-llm/kv-cache")),
        budget_mib: Some(32 * 1024),
        minimum_free_mib: Some(16 * 1024),
    }
}

#[test]
fn disk_mode_and_directory_changes_require_restart() {
    let old = disk_config();

    let mut mode = old.clone();
    mode.mode = Some(KvDiskTierMode::Auto);
    assert!(kv_disk_changes_require_restart(&old, &mode));

    let mut directory = old.clone();
    directory.directory = Some(PathBuf::from("/var/lib/mesh-llm/other-cache"));
    assert!(kv_disk_changes_require_restart(&old, &directory));
}

#[test]
fn disk_budget_and_reserve_changes_are_dynamic() {
    let old = disk_config();

    let mut limits = old.clone();
    limits.budget_mib = Some(48 * 1024);
    limits.minimum_free_mib = Some(20 * 1024);

    assert!(!kv_disk_changes_require_restart(&old, &limits));
    assert!(kv_disk_dynamic_limits_changed(&old, &limits));
}

#[test]
fn mixed_static_and_dynamic_disk_change_preserves_both_classifications() {
    let old = disk_config();
    let mut new = old.clone();
    new.directory = Some(PathBuf::from("/var/lib/mesh-llm/other-cache"));
    new.budget_mib = Some(48 * 1024);

    assert!(kv_disk_changes_require_restart(&old, &new));
    assert!(kv_disk_dynamic_limits_changed(&old, &new));
}
