use anyhow::Context;
use mesh_llm_config::{
    ConfigAliasPolicy, ConfigApplyMode, ConfigConditionOperator, ConfigConditionValue,
    ConfigConditionalDisable, ConfigConflictRule, ConfigConstraint, ConfigControlAvailability,
    ConfigControlAvailabilitySource, ConfigControlBehavior, ConfigControlCondition,
    ConfigControlSurface, ConfigDisabledWritePolicy, ConfigNumericControl,
    ConfigObjectPropertySchema, ConfigOptionsSource, ConfigPath, ConfigPresentationMetadata,
    ConfigRestartScope, ConfigSchema, ConfigSettingOwner, ConfigSettingSchema, ConfigSupportState,
    ConfigTextFormat, ConfigValueSchema, ConfigVisibility, built_in_config_schema,
};
use mesh_llm_plugin_manager::{
    InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginConstraint,
    InstalledPluginMetadata, InstalledPluginPresentationMetadata, InstalledPluginRestartScope,
    InstalledPluginValueKind, InstalledPluginValueSchema, InstalledPluginVisibility, PluginStore,
    default_store_root,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;

mod plugin_conversion;
use self::plugin_conversion::plugin_control_behavior_from_installed;

#[derive(Clone, Debug, PartialEq)]
pub struct AggregatedConfigSchema {
    settings_by_path: BTreeMap<ConfigPath, AggregatedConfigSchemaEntry>,
    plugin_instances: Vec<ConfigSchemaPluginInstance>,
}

impl AggregatedConfigSchema {
    pub fn get(&self, path: &ConfigPath) -> Option<&AggregatedConfigSchemaEntry> {
        self.settings_by_path.get(path)
    }

    pub fn settings_by_path(&self) -> &BTreeMap<ConfigPath, AggregatedConfigSchemaEntry> {
        &self.settings_by_path
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ConfigPath, &AggregatedConfigSchemaEntry)> {
        self.settings_by_path.iter()
    }

    pub fn plugin_instances(&self) -> &[ConfigSchemaPluginInstance] {
        &self.plugin_instances
    }

    pub fn export_reference(&self) -> ConfigSchemaReference {
        ConfigSchemaReference {
            settings: self
                .settings_by_path
                .values()
                .map(ConfigSchemaReferenceEntry::from)
                .collect(),
            plugin_instances: self.plugin_instances.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConfigSchemaReference {
    pub settings: Vec<ConfigSchemaReferenceEntry>,
    #[serde(default)]
    pub plugin_instances: Vec<ConfigSchemaPluginInstance>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConfigSchemaPluginInstance {
    pub name: String,
    pub enabled: bool,
    pub source_repository: String,
    pub installed_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub has_config_schema: bool,
    pub allow_unvalidated_config: bool,
}

impl From<&InstalledPluginMetadata> for ConfigSchemaPluginInstance {
    fn from(value: &InstalledPluginMetadata) -> Self {
        let config_schema = value
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.config_schema.as_ref());
        Self {
            name: value.name.clone(),
            enabled: value.enabled,
            source_repository: value.source_repository.clone(),
            installed_version: value.installed_version.clone(),
            last_status: value.last_status.clone(),
            last_error: value.last_error.clone(),
            has_config_schema: config_schema.is_some(),
            allow_unvalidated_config: config_schema
                .map(|schema| schema.allow_unvalidated_config)
                .unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConfigSchemaReferenceEntry {
    pub canonical_path: String,
    pub owner: ConfigSettingOwner,
    pub source: ConfigSchemaReferenceSource,
    pub value_schema: ConfigValueSchema,
    pub support: ConfigSupportState,
    pub control_surfaces: Vec<ConfigControlSurface>,
    pub apply_mode: ConfigApplyMode,
    pub restart_scope: ConfigRestartScope,
    pub visibility: ConfigVisibility,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<ConfigConstraint>,
    #[serde(skip_serializing_if = "is_default_alias_policy")]
    pub alias_policy: ConfigAliasPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presentation: Option<ConfigPresentationMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_behavior: Option<ConfigControlBehavior>,
}

impl From<&AggregatedConfigSchemaEntry> for ConfigSchemaReferenceEntry {
    fn from(value: &AggregatedConfigSchemaEntry) -> Self {
        Self {
            canonical_path: value.setting.path.render(),
            owner: value.setting.owner,
            source: ConfigSchemaReferenceSource::from(&value.source),
            value_schema: value.setting.value_schema.clone(),
            support: value.setting.support,
            control_surfaces: value.setting.control_surfaces.clone(),
            apply_mode: value.setting.apply_mode,
            restart_scope: value.setting.restart_scope,
            visibility: value.setting.visibility,
            constraints: value.setting.constraints.clone(),
            alias_policy: value.setting.alias_policy.clone(),
            description: value.setting.description.clone(),
            presentation: value.setting.presentation.clone(),
            control_behavior: value.setting.control_behavior.clone(),
        }
    }
}

fn is_default_alias_policy(value: &ConfigAliasPolicy) -> bool {
    value == &ConfigAliasPolicy::default()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigSchemaReferenceSource {
    BuiltIn,
    Engine {
        engine_id: String,
    },
    Plugin {
        plugin_name: String,
        allow_unvalidated_config: bool,
    },
}

impl From<&AggregatedConfigSchemaSource> for ConfigSchemaReferenceSource {
    fn from(value: &AggregatedConfigSchemaSource) -> Self {
        match value {
            AggregatedConfigSchemaSource::BuiltIn => Self::BuiltIn,
            AggregatedConfigSchemaSource::Engine { engine_id } => Self::Engine {
                engine_id: engine_id.clone(),
            },
            AggregatedConfigSchemaSource::Plugin {
                plugin_name,
                allow_unvalidated_config,
            } => Self::Plugin {
                plugin_name: plugin_name.clone(),
                allow_unvalidated_config: *allow_unvalidated_config,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregatedConfigSchemaEntry {
    pub setting: ConfigSettingSchema,
    pub source: AggregatedConfigSchemaSource,
    pub unknown_policy: AggregatedConfigUnknownPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregatedConfigSchemaSource {
    BuiltIn,
    Engine {
        engine_id: String,
    },
    Plugin {
        plugin_name: String,
        allow_unvalidated_config: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregatedConfigUnknownPolicy {
    Reject,
    PreserveWithDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineConfigSchemaDescriptor {
    pub engine_id: String,
    pub schema: ConfigSchema,
}

#[derive(Debug)]
pub enum AggregatedConfigSchemaError {
    DuplicatePath {
        path: ConfigPath,
        existing_source: AggregatedConfigSchemaSource,
        incoming_source: AggregatedConfigSchemaSource,
    },
    PluginStore(anyhow::Error),
}

impl fmt::Display for AggregatedConfigSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePath {
                path,
                existing_source,
                incoming_source,
            } => write!(
                f,
                "duplicate aggregated config schema path '{}' from {:?}; already registered by {:?}",
                path.render(),
                incoming_source,
                existing_source
            ),
            Self::PluginStore(error) => write!(
                f,
                "failed to load installed plugin schema metadata: {error}"
            ),
        }
    }
}

impl std::error::Error for AggregatedConfigSchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DuplicatePath { .. } => None,
            Self::PluginStore(error) => Some(error.as_ref()),
        }
    }
}

pub fn aggregate_runtime_config_schema(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
) -> Result<AggregatedConfigSchema, AggregatedConfigSchemaError> {
    let root = default_store_root().map_err(AggregatedConfigSchemaError::PluginStore)?;
    let installed_plugins = PluginStore::new(root)
        .list()
        .context("list installed plugins")
        .map_err(AggregatedConfigSchemaError::PluginStore)?;
    aggregate_config_schema_sources(engine_schemas, installed_plugins)
}

pub fn export_runtime_config_schema_reference(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
) -> Result<ConfigSchemaReference, AggregatedConfigSchemaError> {
    aggregate_runtime_config_schema(engine_schemas).map(|schema| schema.export_reference())
}

pub fn aggregate_config_schema_sources(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
    installed_plugins: impl IntoIterator<Item = InstalledPluginMetadata>,
) -> Result<AggregatedConfigSchema, AggregatedConfigSchemaError> {
    let installed_plugins = installed_plugins.into_iter().collect::<Vec<_>>();
    let mut plugin_instances = vec![ConfigSchemaPluginInstance {
        name: crate::plugin::BLOBSTORE_PLUGIN_ID.to_string(),
        enabled: true,
        source_repository: "built-in".to_string(),
        installed_version: crate::VERSION.to_string(),
        last_status: Some("built-in".to_string()),
        last_error: None,
        has_config_schema: false,
        allow_unvalidated_config: false,
    }];
    plugin_instances.extend(
        installed_plugins
            .iter()
            .map(ConfigSchemaPluginInstance::from)
            .filter(|instance| instance.name != crate::plugin::BLOBSTORE_PLUGIN_ID),
    );
    let mut settings_by_path = BTreeMap::new();

    for setting in built_in_config_schema().settings {
        register_setting(
            &mut settings_by_path,
            setting,
            AggregatedConfigSchemaSource::BuiltIn,
            AggregatedConfigUnknownPolicy::Reject,
        )?;
    }

    for engine in engine_schemas {
        let source = AggregatedConfigSchemaSource::Engine {
            engine_id: engine.engine_id,
        };
        for setting in engine.schema.settings {
            register_setting(
                &mut settings_by_path,
                setting,
                source.clone(),
                AggregatedConfigUnknownPolicy::PreserveWithDiagnostics,
            )?;
        }
    }

    for plugin in installed_plugins {
        let Some(schema) = plugin
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.config_schema.as_ref())
        else {
            continue;
        };

        let source = AggregatedConfigSchemaSource::Plugin {
            plugin_name: schema.plugin_name.clone(),
            allow_unvalidated_config: schema.allow_unvalidated_config,
        };
        let unknown_policy = if schema.allow_unvalidated_config {
            AggregatedConfigUnknownPolicy::PreserveWithDiagnostics
        } else {
            AggregatedConfigUnknownPolicy::Reject
        };

        for setting in plugin_settings_from_installed_schema(schema) {
            register_setting(
                &mut settings_by_path,
                setting,
                source.clone(),
                unknown_policy,
            )?;
        }
    }

    Ok(AggregatedConfigSchema {
        settings_by_path,
        plugin_instances,
    })
}

fn register_setting(
    settings_by_path: &mut BTreeMap<ConfigPath, AggregatedConfigSchemaEntry>,
    mut setting: ConfigSettingSchema,
    source: AggregatedConfigSchemaSource,
    unknown_policy: AggregatedConfigUnknownPolicy,
) -> Result<(), AggregatedConfigSchemaError> {
    setting.path = setting.path.normalize_builtin_layout();

    if let Some(existing) = settings_by_path.get(&setting.path) {
        return Err(AggregatedConfigSchemaError::DuplicatePath {
            path: setting.path.clone(),
            existing_source: existing.source.clone(),
            incoming_source: source,
        });
    }

    settings_by_path.insert(
        setting.path.clone(),
        AggregatedConfigSchemaEntry {
            setting,
            source,
            unknown_policy,
        },
    );
    Ok(())
}

fn plugin_settings_from_installed_schema(
    schema: &InstalledPluginConfigSchema,
) -> Vec<ConfigSettingSchema> {
    schema
        .settings
        .iter()
        .map(|setting| ConfigSettingSchema {
            path: ConfigPath::from_fields([
                "plugin",
                schema.plugin_name.as_str(),
                "settings",
                setting.key.as_str(),
            ]),
            alias_policy: ConfigAliasPolicy::default(),
            owner: ConfigSettingOwner::Plugin,
            value_schema: plugin_value_schema_from_installed(&setting.value_schema),
            support: ConfigSupportState::Supported,
            control_surfaces: vec![
                ConfigControlSurface::ConfigFile,
                ConfigControlSurface::OwnerControl,
                ConfigControlSurface::PluginManifest,
            ],
            apply_mode: plugin_apply_mode_from_installed(setting.apply_mode),
            restart_scope: plugin_restart_scope_from_installed(setting.restart_scope),
            visibility: plugin_visibility_from_installed(setting.visibility),
            constraints: setting
                .constraints
                .iter()
                .map(plugin_constraint_from_installed)
                .collect(),
            description: setting.description.clone(),
            presentation: plugin_presentation_from_installed(setting.presentation.as_ref()),
            control_behavior: setting
                .control_behavior
                .as_ref()
                .map(plugin_control_behavior_from_installed),
        })
        .collect()
}

fn plugin_presentation_from_installed(
    presentation: Option<&InstalledPluginPresentationMetadata>,
) -> Option<ConfigPresentationMetadata> {
    presentation.map(|presentation| ConfigPresentationMetadata {
        label: presentation.label.clone(),
        help: presentation.help.clone(),
        category_id: presentation.category_id.clone(),
        category_label: presentation.category_label.clone(),
        category_summary: presentation.category_summary.clone(),
        category_order: presentation.category_order,
        setting_order: presentation.setting_order,
        unit: presentation.unit.clone(),
        placeholder: presentation.placeholder.clone(),
        control_hint: presentation.control_hint.clone(),
        renderer_id: presentation.renderer_id.clone(),
        choices: vec![],
        display_units: vec![],
    })
}

fn plugin_value_schema_from_installed(schema: &InstalledPluginValueSchema) -> ConfigValueSchema {
    match schema.kind {
        InstalledPluginValueKind::Boolean => ConfigValueSchema::Boolean,
        InstalledPluginValueKind::Integer => ConfigValueSchema::Integer,
        InstalledPluginValueKind::Float => ConfigValueSchema::Float,
        InstalledPluginValueKind::String => ConfigValueSchema::String,
        InstalledPluginValueKind::Path => ConfigValueSchema::Path,
        InstalledPluginValueKind::Url => ConfigValueSchema::Url,
        InstalledPluginValueKind::Enum => ConfigValueSchema::Enum {
            values: schema.enum_values.clone(),
        },
        InstalledPluginValueKind::Array => ConfigValueSchema::Array {
            items: Box::new(
                schema
                    .items
                    .as_deref()
                    .map(plugin_value_schema_from_installed)
                    .unwrap_or(ConfigValueSchema::String),
            ),
        },
        InstalledPluginValueKind::Object => ConfigValueSchema::Object {
            properties: schema
                .object_properties
                .iter()
                .map(|property| ConfigObjectPropertySchema {
                    name: property.key.clone(),
                    label: property.key.clone(),
                    required: property.required,
                    value_schema: plugin_value_schema_from_installed(&property.value_schema),
                })
                .collect(),
        },
    }
}

fn plugin_constraint_from_installed(constraint: &InstalledPluginConstraint) -> ConfigConstraint {
    match constraint {
        InstalledPluginConstraint::NonEmpty => ConfigConstraint::NonEmpty,
        InstalledPluginConstraint::Positive => ConfigConstraint::Positive,
        InstalledPluginConstraint::Range { min, max } => ConfigConstraint::Range {
            min: min.clone(),
            max: max.clone(),
        },
        InstalledPluginConstraint::AllowedValues { values } => ConfigConstraint::AllowedValues {
            values: values.clone(),
        },
        InstalledPluginConstraint::Requires { key } => ConfigConstraint::Requires {
            path: ConfigPath::field(key.clone()),
        },
    }
}

fn plugin_apply_mode_from_installed(mode: InstalledPluginApplyMode) -> ConfigApplyMode {
    match mode {
        InstalledPluginApplyMode::StaticOnLoad => ConfigApplyMode::StaticOnLoad,
        InstalledPluginApplyMode::DynamicValidationOnly => ConfigApplyMode::DynamicValidationOnly,
        InstalledPluginApplyMode::DynamicApply => ConfigApplyMode::DynamicApply,
    }
}

fn plugin_restart_scope_from_installed(scope: InstalledPluginRestartScope) -> ConfigRestartScope {
    match scope {
        InstalledPluginRestartScope::None => ConfigRestartScope::None,
        InstalledPluginRestartScope::ModelReload => ConfigRestartScope::ModelReload,
        InstalledPluginRestartScope::ProcessRestart
        | InstalledPluginRestartScope::PluginProcess => ConfigRestartScope::ProcessRestart,
        InstalledPluginRestartScope::MeshRestart => ConfigRestartScope::MeshRestart,
    }
}

fn plugin_visibility_from_installed(visibility: InstalledPluginVisibility) -> ConfigVisibility {
    match visibility {
        InstalledPluginVisibility::User => ConfigVisibility::User,
        InstalledPluginVisibility::Advanced => ConfigVisibility::Advanced,
        InstalledPluginVisibility::Hidden => ConfigVisibility::Hidden,
        InstalledPluginVisibility::Internal => ConfigVisibility::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_config::{ConfigSchemaBuilder, ConfigSettingSchemaBuilder};
    use mesh_llm_plugin_manager::{
        InstalledPluginConfigSchema, InstalledPluginControlBehavior,
        InstalledPluginManifestMetadata, InstalledPluginObjectProperty,
        InstalledPluginSettingSchema, InstalledPluginTextFormat,
    };
    use serde::Serialize;
    use std::path::PathBuf;

    const CONFIG_SCHEMA_REFERENCE_FIXTURE: &str =
        include_str!("../tests/fixtures/config_schema_reference.json");
    const CONFIG_SCHEMA_DEFAULTS_UI_FIXTURE: &str =
        include_str!("../tests/fixtures/config_schema_defaults_ui_reference.json");

    #[derive(Serialize)]
    struct DefaultsUiSchemaReference {
        settings: Vec<DefaultsUiSchemaReferenceEntry>,
    }

    #[derive(Serialize)]
    struct DefaultsUiSchemaReferenceEntry {
        canonical_path: String,
        support: ConfigSupportState,
        source: ConfigSchemaReferenceSource,
    }

    #[test]
    fn aggregated_config_schema_sources() {
        let mut engine_schema = ConfigSchemaBuilder::new();
        let mut engine_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "engine", "vllm", "temperature"]),
            ConfigValueSchema::Float,
        );
        engine_setting.owner(ConfigSettingOwner::Engine);
        engine_schema.setting(engine_setting.build());

        let aggregated = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: engine_schema.build(),
            }],
            [installed_plugin_metadata(
                "blackboard",
                vec![InstalledPluginSettingSchema {
                    key: "retention_days".into(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::Integer,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: vec![InstalledPluginObjectProperty {
                            key: "unused".into(),
                            value_schema: InstalledPluginValueSchema {
                                kind: InstalledPluginValueKind::String,
                                enum_values: Vec::new(),
                                items: None,
                                object_properties: Vec::new(),
                                allow_additional_properties: false,
                            },
                            required: false,
                            description: None,
                        }],
                        allow_additional_properties: false,
                    },
                    required: true,
                    default_json: Some("14".into()),
                    constraints: vec![InstalledPluginConstraint::Range {
                        min: Some("1".into()),
                        max: Some("365".into()),
                    }],
                    apply_mode: InstalledPluginApplyMode::DynamicApply,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::Advanced,
                    description: Some("Retention period in days".into()),
                    presentation: None,
                    control_behavior: None,
                }],
            )],
        )
        .expect("schema aggregation should succeed");

        let built_in = aggregated
            .get(&ConfigPath::field("version"))
            .expect("built-in version setting should be present");
        assert_eq!(built_in.source, AggregatedConfigSchemaSource::BuiltIn);
        assert_eq!(
            built_in.unknown_policy,
            AggregatedConfigUnknownPolicy::Reject
        );

        let engine = aggregated
            .get(&ConfigPath::from_fields([
                "defaults",
                "engine",
                "vllm",
                "temperature",
            ]))
            .expect("engine setting should be present");
        assert_eq!(
            engine.source,
            AggregatedConfigSchemaSource::Engine {
                engine_id: "vllm".into(),
            }
        );
        assert_eq!(
            engine.unknown_policy,
            AggregatedConfigUnknownPolicy::PreserveWithDiagnostics
        );

        let plugin = aggregated
            .get(&ConfigPath::from_fields([
                "plugin",
                "blackboard",
                "settings",
                "retention_days",
            ]))
            .expect("plugin setting should be present under canonical plugin path");
        assert_eq!(
            plugin.source,
            AggregatedConfigSchemaSource::Plugin {
                plugin_name: "blackboard".into(),
                allow_unvalidated_config: false,
            }
        );
        assert_eq!(plugin.unknown_policy, AggregatedConfigUnknownPolicy::Reject);
        assert_eq!(plugin.setting.owner, ConfigSettingOwner::Plugin);
        assert_eq!(
            plugin.setting.path.render(),
            "plugin.blackboard.settings.retention_days"
        );
        assert_eq!(aggregated.plugin_instances().len(), 2);
        assert!(
            aggregated
                .plugin_instances()
                .iter()
                .any(|instance| instance.name == crate::plugin::BLOBSTORE_PLUGIN_ID)
        );
        let blackboard = aggregated
            .plugin_instances()
            .iter()
            .find(|instance| instance.name == "blackboard")
            .expect("blackboard plugin instance should be present");
        assert!(blackboard.has_config_schema);
    }

    #[test]
    fn aggregated_schema_duplicate_paths() {
        let mut duplicate_schema = ConfigSchemaBuilder::new();
        let mut duplicate_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::field("version"),
            ConfigValueSchema::Integer,
        );
        duplicate_setting.owner(ConfigSettingOwner::Engine);
        duplicate_schema.setting(duplicate_setting.build());

        let error = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: duplicate_schema.build(),
            }],
            Vec::<InstalledPluginMetadata>::new(),
        )
        .expect_err("duplicate canonical paths should fail deterministically");

        match error {
            AggregatedConfigSchemaError::DuplicatePath {
                path,
                existing_source,
                incoming_source,
            } => {
                assert_eq!(path.render(), "version");
                assert_eq!(existing_source, AggregatedConfigSchemaSource::BuiltIn);
                assert_eq!(
                    incoming_source,
                    AggregatedConfigSchemaSource::Engine {
                        engine_id: "vllm".into(),
                    }
                );
            }
            other => panic!("unexpected aggregation error: {other}"),
        }
    }

    #[test]
    fn schema_export_preserves_built_in_numeric_control_metadata_and_omits_missing_control_behavior()
     {
        let mut engine_schema = ConfigSchemaBuilder::new();
        let mut engine_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "engine", "vllm", "temperature"]),
            ConfigValueSchema::Float,
        );
        engine_setting.owner(ConfigSettingOwner::Engine);
        engine_schema.setting(engine_setting.build());

        let exported = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: engine_schema.build(),
            }],
            Vec::<InstalledPluginMetadata>::new(),
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let batch = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "defaults.model_fit.batch")
            .expect("built-in batch setting should be present");
        let batch_json =
            serde_json::to_value(batch).expect("reference entry should serialize to json");
        assert_eq!(
            batch_json.pointer("/control_behavior/numeric/min"),
            Some(&serde_json::json!(1.0))
        );
        assert_eq!(
            batch_json.pointer("/control_behavior/numeric/step"),
            Some(&serde_json::json!(1.0))
        );
        assert_eq!(
            batch_json.pointer("/control_behavior/numeric/unit"),
            Some(&serde_json::json!("tokens"))
        );

        let engine = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "defaults.engine.vllm.temperature")
            .expect("engine setting should be present");
        let engine_json =
            serde_json::to_value(engine).expect("reference entry should serialize to json");
        assert!(
            engine_json.get("control_behavior").is_none(),
            "missing control behavior should be omitted from schema reference json"
        );
    }

    #[test]
    fn schema_export_preserves_plugin_path_and_url_value_kinds_and_control_behavior() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            [installed_plugin_metadata(
                "blackboard",
                vec![
                    InstalledPluginSettingSchema {
                        key: "projector_path".into(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Path,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: None,
                        constraints: Vec::new(),
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::Advanced,
                        description: Some("Projector path".into()),
                        presentation: None,
                        control_behavior: Some(InstalledPluginControlBehavior {
                            text_format: Some(InstalledPluginTextFormat::Path),
                            ..InstalledPluginControlBehavior::default()
                        }),
                    },
                    InstalledPluginSettingSchema {
                        key: "projector_url".into(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Url,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: None,
                        constraints: Vec::new(),
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::Advanced,
                        description: Some("Projector URL".into()),
                        presentation: None,
                        control_behavior: Some(InstalledPluginControlBehavior {
                            text_format: Some(InstalledPluginTextFormat::Url),
                            ..InstalledPluginControlBehavior::default()
                        }),
                    },
                ],
            )],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let path_entry = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "plugin.blackboard.settings.projector_path")
            .expect("plugin path setting should be present");
        let path_json =
            serde_json::to_value(path_entry).expect("reference entry should serialize to json");
        assert_eq!(
            path_json.pointer("/value_schema/kind"),
            Some(&serde_json::json!("path"))
        );
        assert_eq!(
            path_json.pointer("/control_behavior/text_format"),
            Some(&serde_json::json!("path"))
        );

        let url_entry = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "plugin.blackboard.settings.projector_url")
            .expect("plugin url setting should be present");
        let url_json =
            serde_json::to_value(url_entry).expect("reference entry should serialize to json");
        assert_eq!(
            url_json.pointer("/value_schema/kind"),
            Some(&serde_json::json!("url"))
        );
        assert_eq!(
            url_json.pointer("/control_behavior/text_format"),
            Some(&serde_json::json!("url"))
        );
    }

    #[test]
    fn schema_export_snapshot() {
        let mut engine_schema = ConfigSchemaBuilder::new();
        let mut engine_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "engine", "vllm", "temperature"]),
            ConfigValueSchema::Float,
        );
        engine_setting.owner(ConfigSettingOwner::Engine);
        engine_setting.control_surface(ConfigControlSurface::Api);
        engine_setting.control_surface(ConfigControlSurface::OwnerControl);
        engine_setting.apply_mode(ConfigApplyMode::DynamicApply);
        engine_setting.restart_scope(ConfigRestartScope::None);
        engine_setting.visibility(ConfigVisibility::Advanced);
        engine_setting.description("Engine temperature override.");
        engine_setting.control_numeric_min(0.0);
        engine_setting.control_numeric_max(2.0);
        engine_setting.control_numeric_step(0.1);
        engine_schema.setting(engine_setting.build());

        let exported = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: engine_schema.build(),
            }],
            [installed_plugin_metadata(
                "blackboard",
                vec![
                    InstalledPluginSettingSchema {
                        key: "retention_days".into(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Integer,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: true,
                        default_json: Some("14".into()),
                        constraints: vec![InstalledPluginConstraint::Range {
                            min: Some("1".into()),
                            max: Some("365".into()),
                        }],
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::Advanced,
                        description: Some("Retention period in days".into()),
                        presentation: Some(
                            mesh_llm_plugin_manager::InstalledPluginPresentationMetadata {
                                label: Some("Retention days".into()),
                                help: Some("How long entries stay available.".into()),
                                category_id: Some("blackboard-retention".into()),
                                category_label: Some("Retention".into()),
                                category_summary: Some("Retention policy".into()),
                                category_order: Some(10),
                                setting_order: Some(20),
                                unit: Some("days".into()),
                                placeholder: None,
                                control_hint: Some("number".into()),
                                renderer_id: None,
                            },
                        ),
                        control_behavior: None,
                    },
                    InstalledPluginSettingSchema {
                        key: "projector_path".into(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Path,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: None,
                        constraints: Vec::new(),
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::Advanced,
                        description: Some("Projector path".into()),
                        presentation: None,
                        control_behavior: Some(InstalledPluginControlBehavior {
                            text_format: Some(InstalledPluginTextFormat::Path),
                            ..InstalledPluginControlBehavior::default()
                        }),
                    },
                    InstalledPluginSettingSchema {
                        key: "projector_url".into(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Url,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: None,
                        constraints: Vec::new(),
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::Advanced,
                        description: Some("Projector URL".into()),
                        presentation: None,
                        control_behavior: Some(InstalledPluginControlBehavior {
                            text_format: Some(InstalledPluginTextFormat::Url),
                            ..InstalledPluginControlBehavior::default()
                        }),
                    },
                ],
            )],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let filtered = ConfigSchemaReference {
            settings: exported
                .settings
                .into_iter()
                .filter(|entry| {
                    matches!(
                        entry.canonical_path.as_str(),
                        "version"
                            | "gpu.assignment"
                            | "owner_control.advertise_addr"
                            | "runtime.debug"
                            | "runtime.listen_all"
                            | "runtime.lifecycle_log_parser"
                            | "telemetry.prompt_shape_metrics"
                            | "defaults.hardware.device"
                            | "defaults.hardware.mmproj"
                            | "defaults.model_fit.batch"
                            | "defaults.multimodal.mmproj"
                            | "defaults.multimodal.mmproj_offload"
                            | "defaults.multimodal.mmproj_url"
                            | "defaults.request_defaults.dry"
                            | "defaults.engine.vllm.temperature"
                            | "models.<model-ref>.hardware.device"
                            | "models.<model-ref>.hardware.rpc_backend"
                            | "plugin.<plugin-name>.startup.connect_timeout_secs"
                            | "plugin.<plugin-name>.url"
                            | "plugin.blackboard.settings.retention_days"
                            | "plugin.blackboard.settings.projector_path"
                            | "plugin.blackboard.settings.projector_url"
                    )
                })
                .collect(),
            plugin_instances: exported.plugin_instances,
        };
        let actual = serde_json::to_string_pretty(&filtered)
            .expect("schema reference export should serialize to json");
        let expected = CONFIG_SCHEMA_REFERENCE_FIXTURE.trim();

        assert_eq!(actual, expected, "schema export snapshot drifted\n{actual}");
    }

    #[test]
    fn schema_export_omits_plugins_without_install_time_schema() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            [InstalledPluginMetadata {
                name: "blackboard".into(),
                source_repository: "mesh-llm/blackboard".into(),
                installed_version: "0.1.0".into(),
                target_triple: "aarch64-apple-darwin".into(),
                downloaded_asset_name: "blackboard.tar.gz".into(),
                install_path: PathBuf::from("/tmp/blackboard"),
                enabled: true,
                manifest: Some(InstalledPluginManifestMetadata {
                    config_schema: None,
                    web_ui: None,
                }),
                last_protocol_version: None,
                last_status: None,
                last_error: None,
            }],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        assert!(exported.settings.iter().all(|entry| {
            !entry
                .canonical_path
                .starts_with("plugin.blackboard.settings.")
        }));
        assert_eq!(exported.plugin_instances.len(), 2);
        let blackboard = exported
            .plugin_instances
            .iter()
            .find(|instance| instance.name == "blackboard")
            .expect("blackboard plugin instance should be present");
        assert!(!blackboard.has_config_schema);
    }

    #[test]
    fn schema_export_classifies_only_logging_retention_and_replay_as_dynamic() {
        let exported = export_runtime_config_schema_reference(std::iter::empty::<
            EngineConfigSchemaDescriptor,
        >())
        .expect("built-in schema should export");

        for path in ["logging.retention_ttl_secs", "logging.replay_capacity"] {
            let setting = exported
                .settings
                .iter()
                .find(|setting| setting.canonical_path == path)
                .expect("logging setting should be exported");
            assert_eq!(setting.apply_mode, ConfigApplyMode::DynamicApply, "{path}");
            assert_eq!(setting.restart_scope, ConfigRestartScope::None, "{path}");
        }

        for path in [
            "logging.retention_max_rows",
            "logging.queue_capacity",
            "logging.cleanup_cadence_secs",
            "logging.artifact.capture_mode",
            "logging.webhook.enabled",
        ] {
            let setting = exported
                .settings
                .iter()
                .find(|setting| setting.canonical_path == path)
                .expect("logging setting should be exported");
            assert_eq!(setting.apply_mode, ConfigApplyMode::StaticOnLoad, "{path}");
            assert_eq!(
                setting.restart_scope,
                ConfigRestartScope::ProcessRestart,
                "{path}"
            );
        }
    }

    #[test]
    fn schema_export_exposes_runtime_and_template_control_metadata() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            [installed_plugin_metadata(
                "blackboard",
                vec![InstalledPluginSettingSchema {
                    key: "retention_days".into(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::Integer,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: true,
                    default_json: Some("14".into()),
                    constraints: vec![InstalledPluginConstraint::Range {
                        min: Some("1".into()),
                        max: Some("365".into()),
                    }],
                    apply_mode: InstalledPluginApplyMode::DynamicApply,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::Advanced,
                    description: Some("Retention period in days".into()),
                    presentation: None,
                    control_behavior: None,
                }],
            )],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let defaults_device = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "defaults.hardware.device")
            .expect("defaults hardware device should be exported");
        assert_eq!(
            defaults_device
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.options_source),
            Some(ConfigOptionsSource::RuntimeGpus)
        );

        let legacy_mmproj = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "defaults.hardware.mmproj")
            .expect("legacy multimodal projector should be exported");
        assert_eq!(legacy_mmproj.value_schema, ConfigValueSchema::Path);
        assert_eq!(
            legacy_mmproj
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.write_policy),
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(
            legacy_mmproj
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.availability.as_ref())
                .map(|availability| availability.enabled),
            Some(false)
        );

        let multimodal_mmproj_url = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "defaults.multimodal.mmproj_url")
            .expect("multimodal projector url should be exported");
        assert_eq!(multimodal_mmproj_url.value_schema, ConfigValueSchema::Url);
        assert_eq!(
            multimodal_mmproj_url
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.text_format),
            Some(ConfigTextFormat::Url)
        );

        let owner_control_advertise_addr = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "owner_control.advertise_addr")
            .expect("owner control advertise addr should be exported");
        assert_eq!(
            owner_control_advertise_addr
                .control_behavior
                .as_ref()
                .map(|behavior| behavior.enable_when.len()),
            Some(1)
        );
        assert_eq!(
            owner_control_advertise_addr
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.disable_when.first())
                .map(|disable| disable.write_policy),
            Some(ConfigDisabledWritePolicy::OmitWhenDisabled)
        );

        let model_rpc_backend = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "models.<model-ref>.hardware.rpc_backend")
            .expect("model rpc backend should be exported");
        assert_eq!(model_rpc_backend.support, ConfigSupportState::Rejected);

        let plugin_url = exported
            .settings
            .iter()
            .find(|entry| entry.canonical_path == "plugin.<plugin-name>.url")
            .expect("plugin url template should be exported");
        assert_eq!(plugin_url.value_schema, ConfigValueSchema::Url);

        let plugin_timeout = exported
            .settings
            .iter()
            .find(|entry| {
                entry.canonical_path == "plugin.<plugin-name>.startup.connect_timeout_secs"
            })
            .expect("plugin timeout template should be exported");
        assert_eq!(plugin_timeout.value_schema, ConfigValueSchema::Integer);
        assert_eq!(
            plugin_timeout
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.numeric.as_ref())
                .and_then(|numeric| numeric.unit.as_deref()),
            Some("sec")
        );

        let blackboard = exported
            .plugin_instances
            .iter()
            .find(|instance| instance.name == "blackboard")
            .expect("blackboard plugin metadata should be exported");
        assert!(blackboard.has_config_schema);
        assert!(!blackboard.allow_unvalidated_config);
    }

    #[test]
    fn defaults_ui_schema_export_snapshot() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            Vec::<InstalledPluginMetadata>::new(),
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let filtered = DefaultsUiSchemaReference {
            settings: exported
                .settings
                .into_iter()
                .filter(|entry| {
                    entry.source == ConfigSchemaReferenceSource::BuiltIn
                        && entry.support == ConfigSupportState::Supported
                        && entry.canonical_path.starts_with("defaults.")
                })
                .map(|entry| DefaultsUiSchemaReferenceEntry {
                    canonical_path: entry.canonical_path,
                    support: entry.support,
                    source: entry.source,
                })
                .collect(),
        };
        let actual = serde_json::to_string_pretty(&filtered)
            .expect("defaults ui schema reference should serialize to json");
        let expected = CONFIG_SCHEMA_DEFAULTS_UI_FIXTURE.trim();

        assert_eq!(
            actual, expected,
            "defaults UI schema export snapshot drifted\n{actual}"
        );
    }

    #[test]
    fn plugin_schema_aggregation_preserves_path_control_metadata_and_unknown_policy() {
        let aggregated = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            [InstalledPluginMetadata {
                name: "blackboard".into(),
                source_repository: "mesh-llm/blackboard".into(),
                installed_version: "0.1.0".into(),
                target_triple: "aarch64-apple-darwin".into(),
                downloaded_asset_name: "blackboard.tar.gz".into(),
                install_path: PathBuf::from("/tmp/blackboard"),
                enabled: true,
                manifest: Some(InstalledPluginManifestMetadata {
                    config_schema: Some(InstalledPluginConfigSchema {
                        plugin_name: "blackboard".into(),
                        schema_version: mesh_llm_plugin_manager::SUPPORTED_PLUGIN_SCHEMA_VERSION,
                        allow_unvalidated_config: true,
                        settings: vec![InstalledPluginSettingSchema {
                            key: "projector_path".into(),
                            value_schema: InstalledPluginValueSchema {
                                kind: InstalledPluginValueKind::Path,
                                enum_values: Vec::new(),
                                items: None,
                                object_properties: Vec::new(),
                                allow_additional_properties: false,
                            },
                            required: false,
                            default_json: None,
                            constraints: Vec::new(),
                            apply_mode: InstalledPluginApplyMode::DynamicApply,
                            restart_scope: InstalledPluginRestartScope::PluginProcess,
                            visibility: InstalledPluginVisibility::Advanced,
                            description: Some("Projector path".into()),
                            presentation: None,
                            control_behavior: Some(InstalledPluginControlBehavior {
                                text_format: Some(InstalledPluginTextFormat::Path),
                                ..InstalledPluginControlBehavior::default()
                            }),
                        }],
                    }),
                    web_ui: None,
                }),
                last_protocol_version: None,
                last_status: None,
                last_error: None,
            }],
        )
        .expect("schema aggregation should succeed");

        let entry = aggregated
            .get(&ConfigPath::from_fields([
                "plugin",
                "blackboard",
                "settings",
                "projector_path",
            ]))
            .expect("plugin setting should be present");

        assert_eq!(
            entry.unknown_policy,
            AggregatedConfigUnknownPolicy::PreserveWithDiagnostics
        );
        assert_eq!(entry.setting.value_schema, ConfigValueSchema::Path);
        assert_eq!(
            entry
                .setting
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.text_format),
            Some(ConfigTextFormat::Path)
        );
    }

    fn installed_plugin_metadata(
        plugin_name: &str,
        settings: Vec<InstalledPluginSettingSchema>,
    ) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: plugin_name.into(),
            source_repository: format!("mesh-llm/{plugin_name}"),
            installed_version: "0.1.0".into(),
            target_triple: "aarch64-apple-darwin".into(),
            downloaded_asset_name: format!("{plugin_name}.tar.gz"),
            install_path: PathBuf::from(format!("/tmp/{plugin_name}")),
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(InstalledPluginConfigSchema {
                    plugin_name: plugin_name.into(),
                    schema_version: mesh_llm_plugin_manager::SUPPORTED_PLUGIN_SCHEMA_VERSION,
                    allow_unvalidated_config: false,
                    settings,
                }),
                web_ui: None,
            }),
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }
}
