use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Inventory {
    pub schema_version: u32,
    pub spec_path: String,
    pub spec_sha256: String,
    pub amendment_procedure: String,
    pub privacy_classes: Vec<String>,
    pub delivery_classes: Vec<String>,
    pub native_callback_kinds: Vec<String>,
    pub reason_codes: Vec<String>,
    pub projected_envelope_keys: Vec<String>,
    pub projected_event_keys: Vec<String>,
    pub base_projected_event_keys: Vec<String>,
    pub forbidden_projection_keys: Vec<String>,
    pub projection_profiles: Vec<ProjectionProfile>,
    pub event_projection_overrides: Vec<EventProjectionOverride>,
    pub carriers: Vec<Carrier>,
    pub terminal_outcome_mappings: Vec<TerminalMapping>,
    pub same_fact_rulings: Vec<SameFactRuling>,
    pub families: Vec<Family>,
    pub native_family_mappings: Vec<NativeFamilyMapping>,
    pub catalog: Catalog,
}

#[derive(Clone, Debug, Deserialize)]
pub struct NativeFamilyMapping {
    pub feature_bit: u32,
    pub native_kind: u32,
    pub native_kind_name: String,
    pub event_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Carrier {
    pub name: String,
    pub location: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TerminalMapping {
    pub variant: String,
    pub event_id: String,
    pub outcome: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SameFactRuling {
    pub ids: Vec<String>,
    pub ruling: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Family {
    pub section: String,
    pub enum_name: String,
    pub owning_crates: Vec<String>,
    pub source_paths: Vec<String>,
    pub producer_symbols: Vec<String>,
    pub authoritative_outcome: String,
    pub reducer_effect: String,
    pub projection: String,
    pub privacy_class: String,
    pub default_delivery_class: String,
    pub default_projection_profile: Option<String>,
    pub test: String,
    pub migration_task: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Catalog {
    pub terminal_event_ids: Vec<String>,
    pub progress_event_ids: Vec<String>,
    pub spec_bullets: Vec<BulletExpansion>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BulletExpansion {
    pub section: String,
    pub ordinal: u32,
    pub event_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpecManifest {
    pub schema_version: u32,
    pub source: String,
    pub bullet_count: usize,
    pub bullets: Vec<SpecBullet>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpecBullet {
    pub section: String,
    pub family: String,
    pub ordinal: u32,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProjectionProfile {
    pub name: String,
    pub keys: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EventProjectionOverride {
    pub profile: String,
    pub event_ids: Vec<String>,
}

pub fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

pub fn repo_root() -> PathBuf {
    crate_root()
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

pub fn crate_package_names() -> BTreeSet<String> {
    fs::read_dir(repo_root().join("crates"))
        .expect("crates directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("Cargo.toml"))
        .filter(|path| path.is_file())
        .map(|path| fs::read_to_string(path).expect("crate manifest"))
        .map(|text| toml::from_str::<toml::Value>(&text).expect("crate manifest TOML"))
        .filter_map(|value| {
            value
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

pub fn load_inventory() -> Inventory {
    let path = crate_root().join("inventory/runtime_events.toml");
    toml::from_str(&fs::read_to_string(&path).expect("inventory file should be readable"))
        .expect("inventory should parse")
}

pub fn load_manifest() -> SpecManifest {
    let path = crate_root().join("inventory/spec_manifest.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("manifest should be readable"))
        .expect("manifest should parse")
}

pub fn inventory_event_ids(inventory: &Inventory) -> BTreeSet<&str> {
    inventory
        .catalog
        .spec_bullets
        .iter()
        .flat_map(|bullet| bullet.event_ids.iter().map(String::as_str))
        .collect()
}
