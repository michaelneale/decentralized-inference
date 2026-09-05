use super::installed::{
    ConfiguredExternalPlugin, append_installed_plugins,
    configured_disabled_installed_plugin_summary, configured_external_plugin_spec,
};
use super::schema_validation::strict_plugin_schema_availability;
use super::{BLOBSTORE_PLUGIN_ID, PluginStartupOptions, PluginSummary};
use crate::{
    MeshRequirementRejectReason, MeshRequirements, NodeVersionBounds, ProtocolGenerationBounds,
    ReleaseAttestationRequirement,
};
use anyhow::{Context, Result, bail};
pub use mesh_llm_config::{
    BoolOrAuto, ConfigDiagnostic, ConfigEditor, ConfigStore, GpuAssignment, GpuConfig,
    HardwareConfig, IntegerOrString, LocalServingNodeConfig, MeshConfig, MeshRequirementsConfig,
    ModelConfigDefaults, ModelConfigEditor, ModelConfigEntry, ModelDefaultsEditor, ModelFitConfig,
    ModelRuntimeKind, OwnerControlConfig, PluginConfigEditor, PluginConfigEntry,
    PluginStartupConfig, PluginWebUiPreference, ReasoningBudget, ReasoningEnabled,
    RequestDefaultsConfig, SkippyConfig, SpeculativeConfig, StringOrStringList, TelemetryConfig,
    TelemetryMetricsConfig, ThroughputConfig, apply_env_overrides, config_path, config_to_toml,
    parse_config_toml as base_parse_config_toml, validate_config_with_plugin_schemas,
};
use mesh_llm_plugin::MeshVisibility;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ConfigFileValidation {
    pub path: PathBuf,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    let mut config = if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        parse_config_toml(&raw).with_context(|| format!("Invalid config {}", path.display()))?
    } else {
        MeshConfig::default()
    };
    apply_env_overrides(&mut config)?;
    Ok(config)
}

pub fn parse_config_toml(raw: &str) -> Result<MeshConfig> {
    let config = base_parse_config_toml(raw)?;
    validate_config_with_installed_plugin_schemas(&config, Some(raw))?;
    Ok(config)
}

pub fn validate_config_file(override_path: Option<&Path>) -> Result<ConfigFileValidation> {
    let path = config_path(override_path)?;
    if !path.exists() {
        bail!(
            "Failed to read config file {}: file does not exist",
            path.display()
        );
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    let mut config = base_parse_config_toml(&raw)
        .with_context(|| format!("Invalid config {}", path.display()))?;
    apply_env_overrides(&mut config)?;
    let diagnostics =
        validate_config_diagnostics_with_installed_plugin_schemas(&config, Some(&raw));
    Ok(ConfigFileValidation { path, diagnostics })
}

#[cfg(test)]
fn validate_config(config: &MeshConfig) -> Result<()> {
    validate_config_with_installed_plugin_schemas(config, None)
}

pub(crate) fn validate_config_with_installed_plugin_schemas(
    config: &MeshConfig,
    raw_toml: Option<&str>,
) -> Result<()> {
    validate_config_with_plugin_schemas(config, raw_toml, strict_plugin_schema_availability)
}

pub(crate) fn validate_config_diagnostics_with_installed_plugin_schemas(
    config: &MeshConfig,
    raw_toml: Option<&str>,
) -> Vec<mesh_llm_config::ConfigDiagnostic> {
    mesh_llm_config::validate_config_diagnostics_with_plugin_schemas(
        config,
        raw_toml,
        strict_plugin_schema_availability,
    )
}

pub(crate) fn mesh_requirements_config_to_runtime(
    config: &MeshRequirementsConfig,
) -> MeshRequirements {
    MeshRequirements {
        node_version: NodeVersionBounds {
            min: config.min_node_version.clone(),
            max: config.max_node_version.clone(),
        },
        protocol_generation: ProtocolGenerationBounds {
            min: config.min_protocol_version,
            max: config.max_protocol_version,
        },
        release_attestation: ReleaseAttestationRequirement {
            required: config.require_release_attestation,
            allowed_signer_keys: config.release_signer_keys.clone(),
        },
    }
}

pub(crate) fn mesh_requirements_config_from_runtime(
    requirements: &MeshRequirements,
) -> MeshRequirementsConfig {
    MeshRequirementsConfig {
        min_node_version: requirements.node_version.min.clone(),
        max_node_version: requirements.node_version.max.clone(),
        min_protocol_version: requirements.protocol_generation.min,
        max_protocol_version: requirements.protocol_generation.max,
        require_release_attestation: requirements.release_attestation.required,
        release_signer_keys: requirements.release_attestation.allowed_signer_keys.clone(),
    }
}

pub(crate) fn mesh_requirements_validation_error(reason: MeshRequirementRejectReason) -> String {
    match reason {
        MeshRequirementRejectReason::NodeVersionMalformed => {
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)".into()
        }
        MeshRequirementRejectReason::NodeVersionBoundsInvalid => {
            "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version".into()
        }
        MeshRequirementRejectReason::ProtocolGenerationBoundsInvalid => {
            "mesh_requirements.min_protocol_version must be less than or equal to mesh_requirements.max_protocol_version".into()
        }
        MeshRequirementRejectReason::ReleaseSignerUntrusted => {
            "mesh_requirements.release_signer_keys entries must not be empty".into()
        }
        MeshRequirementRejectReason::ReleaseSignerListEmpty => {
            "mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty; certified-build admission is not remote runtime attestation, so trust must be anchored in at least one release signer key".into()
        }
        MeshRequirementRejectReason::ReleaseSignerKeyMalformed => {
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'".into()
        }
        other => format!("mesh_requirements are invalid: {other:?}"),
    }
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[mesh_requirements]
min_node_version = "0.65.0"
min_protocol_version = 1
require_release_attestation = true
release_signer_keys = [
    "ed25519:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    "ed25519:3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
]
"#,
    )
    .expect("config should parse");
    validate_config(&config).expect("min-only config should validate");
    assert_eq!(
        config.mesh_requirements.min_node_version.as_deref(),
        Some("0.65.0")
    );
    assert_eq!(config.mesh_requirements.max_node_version, None);
    assert_eq!(config.mesh_requirements.min_protocol_version, Some(1));
    assert_eq!(config.mesh_requirements.max_protocol_version, None);
    assert!(config.mesh_requirements.require_release_attestation);
    assert_eq!(
        config.mesh_requirements.release_signer_keys,
        vec![
            "ed25519:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a".to_string(),
            "ed25519:3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c".to_string(),
        ]
    );

    let max_only: MeshConfig = toml::from_str(
        r#"
[mesh_requirements]
max_node_version = "0.65.9"
max_protocol_version = 3
"#,
    )
    .expect("config should parse");
    validate_config(&max_only).expect("max-only config should validate");
    assert_eq!(max_only.mesh_requirements.min_node_version, None);
    assert_eq!(
        max_only.mesh_requirements.max_node_version.as_deref(),
        Some("0.65.9")
    );
    assert_eq!(max_only.mesh_requirements.min_protocol_version, None);
    assert_eq!(max_only.mesh_requirements.max_protocol_version, Some(3));

    let full_range: MeshConfig = toml::from_str(
        r#"
[mesh_requirements]
min_node_version = "0.65.0"
max_node_version = "0.65.9"
min_protocol_version = 1
max_protocol_version = 3
"#,
    )
    .expect("config should parse");
    validate_config(&full_range).expect("full-range config should validate");
    assert_eq!(
        full_range.mesh_requirements.min_node_version.as_deref(),
        Some("0.65.0")
    );
    assert_eq!(
        full_range.mesh_requirements.max_node_version.as_deref(),
        Some("0.65.9")
    );
    assert_eq!(full_range.mesh_requirements.min_protocol_version, Some(1));
    assert_eq!(full_range.mesh_requirements.max_protocol_version, Some(3));

    let unset = MeshConfig::default();
    validate_config(&unset).expect("omitted mesh_requirements should validate");
    assert_eq!(unset.mesh_requirements, MeshRequirementsConfig::default());
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_config_rejects_required_attestation_without_signer_keys() {
    let config: MeshConfig = toml::from_str(
        r#"
[mesh_requirements]
require_release_attestation = true
"#,
    )
    .expect("config should parse");
    let err = validate_config(&config)
        .expect_err("require_release_attestation=true with no signer keys must be rejected");
    let message = format!("{err:#}");
    assert!(
        message.contains("certified-build admission is not remote runtime attestation"),
        "operator error must reference the certified-build / runtime-attestation distinction; got: {message}"
    );
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_config_rejects_non_ed25519_signer_key() {
    let config: MeshConfig = toml::from_str(
        r#"
[mesh_requirements]
require_release_attestation = true
release_signer_keys = ["not-an-ed25519-key"]
"#,
    )
    .expect("config should parse");
    let err = validate_config(&config)
        .expect_err("non-ed25519 release_signer_keys entry must be rejected at policy creation");
    let message = format!("{err:#}");
    assert!(
        message.contains("ed25519:<64-character-hex-public-key>"),
        "operator error must spell out the required ed25519:<hex> shape; got: {message}"
    );
}

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
    pub inactive: Vec<PluginSummary>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// Optional plugin URL passed through the generic plugin launch contract.
    pub url: Option<String>,
    /// Extra environment passed only to the plugin process.
    pub env: BTreeMap<String, String>,
    pub startup: PluginStartupOptions,
    pub web_ui_enabled: Option<bool>,
    pub installed_metadata: Option<mesh_llm_plugin_manager::InstalledPluginMetadata>,
}

#[derive(Clone, Copy, Debug)]
pub struct PluginHostMode {
    pub mesh_visibility: MeshVisibility,
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let mut inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blobstore_enabled = true;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLOBSTORE_PLUGIN_ID {
            if entry.command.is_some()
                || !entry.args.is_empty()
                || entry.url.is_some()
                || !entry.startup.is_default()
            {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLOBSTORE_PLUGIN_ID
                );
            }
            blobstore_enabled = enabled;
            continue;
        }
        if !enabled {
            if let Some(summary) = configured_disabled_installed_plugin_summary(entry) {
                inactive.push(summary);
            }
            continue;
        }
        match configured_external_plugin_spec(entry)? {
            ConfiguredExternalPlugin::Active(spec) => externals.push(spec),
            ConfiguredExternalPlugin::Inactive(summary) => inactive.push(summary),
        }
    }

    append_installed_plugins(&mut externals, &mut inactive, &mut names);

    if blobstore_enabled {
        externals.push(blobstore_plugin_spec()?);
    }

    Ok(ResolvedPlugins {
        externals,
        inactive,
    })
}

pub fn blobstore_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLOBSTORE_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLOBSTORE_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
        startup: PluginStartupOptions {
            optional: true,
            ..PluginStartupOptions::default()
        },
        web_ui_enabled: None,
        installed_metadata: None,
    })
}

pub fn bundled_cli_plugin_spec(_name: &str) -> Result<Option<ExternalPluginSpec>> {
    Ok(None)
}

#[cfg(test)]
mod tests;
