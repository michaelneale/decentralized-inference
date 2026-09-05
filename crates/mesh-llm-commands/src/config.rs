use anyhow::{Context, Result, bail};
use mesh_llm_cli::Cli;
use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, ConfigPath, MeshConfig, PluginConditionOperator, PluginConditionValue,
    PluginConditionalDisable, PluginConfigSchema, PluginConflictRule, PluginControlAvailability,
    PluginControlAvailabilitySource, PluginControlBehavior, PluginControlCondition,
    PluginDisabledWritePolicy, PluginNumericControl, PluginObjectPropertySchema,
    PluginOptionsSource, PluginSchemaAvailability, PluginSettingConstraint, PluginSettingSchema,
    PluginTextFormat, PluginValueKind, PluginValueSchema, SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
    apply_env_overrides, config_path, parse_config_toml_structural,
    validate_config_diagnostics_with_plugin_schemas,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConditionOperator, InstalledPluginConditionValue,
    InstalledPluginConditionalDisable, InstalledPluginConfigSchema, InstalledPluginConflictRule,
    InstalledPluginConstraint, InstalledPluginControlAvailability,
    InstalledPluginControlAvailabilitySource, InstalledPluginControlBehavior,
    InstalledPluginControlCondition, InstalledPluginDisabledWritePolicy, InstalledPluginMetadata,
    InstalledPluginObjectProperty, InstalledPluginOptionsSource, InstalledPluginTextFormat,
    InstalledPluginValueKind, InstalledPluginValueSchema, PluginStore, default_store_root,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct ConfigFileValidation {
    path: PathBuf,
    diagnostics: Vec<ConfigDiagnostic>,
}

pub fn run_config_validate(
    cli: &Cli,
    config_path_override: Option<&Path>,
    json: bool,
) -> Result<()> {
    let selected_path = config_path_override.or(cli.config.as_deref());
    let resolved_path = config_path(selected_path).ok();

    match validate_config_file(selected_path) {
        Ok(validation) => handle_validation_result(validation.path, validation.diagnostics, json),
        Err(err) => {
            print_validation_load_error(resolved_path.as_deref(), &err, json)?;
            Err(err).context("config validation failed")
        }
    }
}

fn validate_config_file(override_path: Option<&Path>) -> Result<ConfigFileValidation> {
    let path = config_path(override_path)?;
    if !path.exists() {
        bail!(
            "Failed to read config file {}: file does not exist",
            path.display()
        );
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    let mut config: MeshConfig = parse_config_toml_structural(&raw)
        .with_context(|| format!("Invalid config {}", path.display()))?;
    apply_env_overrides(&mut config)?;
    let diagnostics =
        validate_config_diagnostics_with_plugin_schemas(&config, Some(&raw), plugin_schema);
    Ok(ConfigFileValidation { path, diagnostics })
}

fn plugin_schema(plugin_name: &str) -> PluginSchemaAvailability {
    let Ok(root) = default_store_root() else {
        return PluginSchemaAvailability::NotInstalled;
    };
    let store = PluginStore::new(root);
    let Ok(metadata) = store.load_optional(plugin_name) else {
        return PluginSchemaAvailability::NotInstalled;
    };
    let Some(metadata) = metadata else {
        return PluginSchemaAvailability::NotInstalled;
    };
    plugin_schema_from_metadata(&metadata)
}

fn plugin_schema_from_metadata(metadata: &InstalledPluginMetadata) -> PluginSchemaAvailability {
    let Some(schema) = metadata
        .manifest
        .as_ref()
        .and_then(|manifest| manifest.config_schema.as_ref())
    else {
        return PluginSchemaAvailability::MissingSchema;
    };

    if schema.schema_version != SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION {
        return PluginSchemaAvailability::UnsupportedVersion {
            version: schema.schema_version,
        };
    }

    PluginSchemaAvailability::Available(plugin_schema_from_installed(schema))
}

fn plugin_schema_from_installed(schema: &InstalledPluginConfigSchema) -> PluginConfigSchema {
    PluginConfigSchema {
        plugin_name: schema.plugin_name.clone(),
        schema_version: schema.schema_version,
        allow_unvalidated_config: schema.allow_unvalidated_config,
        settings: schema
            .settings
            .iter()
            .map(|setting| PluginSettingSchema {
                key: setting.key.clone(),
                value_schema: plugin_value_schema_from_installed(&setting.value_schema),
                required: setting.required,
                default_json: setting.default_json.clone(),
                constraints: setting
                    .constraints
                    .iter()
                    .map(plugin_constraint_from_installed)
                    .collect(),
                description: setting.description.clone(),
                control_behavior: setting
                    .control_behavior
                    .as_ref()
                    .map(plugin_control_behavior_from_installed),
            })
            .collect(),
    }
}

fn plugin_control_behavior_from_installed(
    behavior: &InstalledPluginControlBehavior,
) -> PluginControlBehavior {
    PluginControlBehavior {
        numeric: behavior
            .numeric
            .as_ref()
            .map(|numeric| PluginNumericControl {
                min: numeric.min,
                max: numeric.max,
                step: numeric.step,
                soft_min: numeric.soft_min,
                soft_max: numeric.soft_max,
                unit: numeric.unit.clone(),
            }),
        text_format: behavior.text_format.map(plugin_text_format_from_installed),
        options_source: behavior
            .options_source
            .map(plugin_options_source_from_installed),
        availability: behavior
            .availability
            .as_ref()
            .map(plugin_availability_from_installed),
        enable_when: behavior
            .enable_when
            .iter()
            .map(plugin_condition_from_installed)
            .collect(),
        disable_when: behavior
            .disable_when
            .iter()
            .map(plugin_disable_from_installed)
            .collect(),
        conflicts: behavior
            .conflicts
            .iter()
            .map(plugin_conflict_from_installed)
            .collect(),
        write_policy: behavior
            .write_policy
            .map(plugin_write_policy_from_installed),
    }
}

fn plugin_text_format_from_installed(format: InstalledPluginTextFormat) -> PluginTextFormat {
    match format {
        InstalledPluginTextFormat::Plain => PluginTextFormat::Plain,
        InstalledPluginTextFormat::Path => PluginTextFormat::Path,
        InstalledPluginTextFormat::Url => PluginTextFormat::Url,
        InstalledPluginTextFormat::SocketAddr => PluginTextFormat::SocketAddr,
        InstalledPluginTextFormat::Semver => PluginTextFormat::Semver,
        InstalledPluginTextFormat::Ed25519Key => PluginTextFormat::Ed25519Key,
        InstalledPluginTextFormat::CsvPositiveInts => PluginTextFormat::CsvPositiveInts,
    }
}

fn plugin_options_source_from_installed(
    source: InstalledPluginOptionsSource,
) -> PluginOptionsSource {
    match source {
        InstalledPluginOptionsSource::Static => PluginOptionsSource::Static,
        InstalledPluginOptionsSource::RuntimeGpus => PluginOptionsSource::RuntimeGpus,
        InstalledPluginOptionsSource::RuntimeNativeBackends => {
            PluginOptionsSource::RuntimeNativeBackends
        }
        InstalledPluginOptionsSource::RuntimeLocalModels => PluginOptionsSource::RuntimeLocalModels,
        InstalledPluginOptionsSource::RuntimeInstalledPlugins => {
            PluginOptionsSource::RuntimeInstalledPlugins
        }
        InstalledPluginOptionsSource::RuntimeMeshPeers => PluginOptionsSource::RuntimeMeshPeers,
    }
}

fn plugin_availability_from_installed(
    availability: &InstalledPluginControlAvailability,
) -> PluginControlAvailability {
    PluginControlAvailability {
        enabled: availability.enabled,
        reason: availability.reason.clone(),
        note: availability.note.clone(),
        source: match availability.source {
            InstalledPluginControlAvailabilitySource::Static => {
                PluginControlAvailabilitySource::Static
            }
            InstalledPluginControlAvailabilitySource::Runtime => {
                PluginControlAvailabilitySource::Runtime
            }
            InstalledPluginControlAvailabilitySource::Dependency => {
                PluginControlAvailabilitySource::Dependency
            }
            InstalledPluginControlAvailabilitySource::Conflict => {
                PluginControlAvailabilitySource::Conflict
            }
        },
    }
}

fn plugin_condition_from_installed(
    condition: &InstalledPluginControlCondition,
) -> PluginControlCondition {
    PluginControlCondition {
        key: condition.key.clone(),
        operator: match condition.operator {
            InstalledPluginConditionOperator::Equals => PluginConditionOperator::Equals,
            InstalledPluginConditionOperator::NotEquals => PluginConditionOperator::NotEquals,
            InstalledPluginConditionOperator::In => PluginConditionOperator::In,
            InstalledPluginConditionOperator::NotIn => PluginConditionOperator::NotIn,
            InstalledPluginConditionOperator::Present => PluginConditionOperator::Present,
            InstalledPluginConditionOperator::Absent => PluginConditionOperator::Absent,
            InstalledPluginConditionOperator::Truthy => PluginConditionOperator::Truthy,
            InstalledPluginConditionOperator::Falsy => PluginConditionOperator::Falsy,
            InstalledPluginConditionOperator::Range => PluginConditionOperator::Range,
        },
        values: condition
            .values
            .iter()
            .map(|value| match value {
                InstalledPluginConditionValue::Bool(value) => PluginConditionValue::Bool(*value),
                InstalledPluginConditionValue::Integer(value) => {
                    PluginConditionValue::Integer(*value)
                }
                InstalledPluginConditionValue::Float(value) => PluginConditionValue::Float(*value),
                InstalledPluginConditionValue::String(value) => {
                    PluginConditionValue::String(value.clone())
                }
            })
            .collect(),
    }
}

fn plugin_disable_from_installed(
    disable: &InstalledPluginConditionalDisable,
) -> PluginConditionalDisable {
    PluginConditionalDisable {
        condition: plugin_condition_from_installed(&disable.condition),
        reason: disable.reason.clone(),
        note: disable.note.clone(),
        write_policy: plugin_write_policy_from_installed(disable.write_policy),
    }
}

fn plugin_conflict_from_installed(conflict: &InstalledPluginConflictRule) -> PluginConflictRule {
    PluginConflictRule {
        group: conflict.group.clone(),
        condition: plugin_condition_from_installed(&conflict.condition),
        reason: conflict.reason.clone(),
        preferred_key: conflict.preferred_key.clone(),
    }
}

fn plugin_write_policy_from_installed(
    policy: InstalledPluginDisabledWritePolicy,
) -> PluginDisabledWritePolicy {
    match policy {
        InstalledPluginDisabledWritePolicy::PreserveExisting => {
            PluginDisabledWritePolicy::PreserveExisting
        }
        InstalledPluginDisabledWritePolicy::OmitWhenDisabled => {
            PluginDisabledWritePolicy::OmitWhenDisabled
        }
        InstalledPluginDisabledWritePolicy::RejectWhenDisabled => {
            PluginDisabledWritePolicy::RejectWhenDisabled
        }
    }
}

fn plugin_value_schema_from_installed(schema: &InstalledPluginValueSchema) -> PluginValueSchema {
    PluginValueSchema {
        kind: match schema.kind {
            InstalledPluginValueKind::Boolean => PluginValueKind::Boolean,
            InstalledPluginValueKind::Integer => PluginValueKind::Integer,
            InstalledPluginValueKind::Float => PluginValueKind::Float,
            InstalledPluginValueKind::String => PluginValueKind::String,
            InstalledPluginValueKind::Path => PluginValueKind::Path,
            InstalledPluginValueKind::Url => PluginValueKind::Url,
            InstalledPluginValueKind::Enum => PluginValueKind::Enum,
            InstalledPluginValueKind::Array => PluginValueKind::Array,
            InstalledPluginValueKind::Object => PluginValueKind::Object,
        },
        enum_values: schema.enum_values.clone(),
        items: schema
            .items
            .as_deref()
            .map(plugin_value_schema_from_installed)
            .map(Box::new),
        object_properties: schema
            .object_properties
            .iter()
            .map(plugin_object_property_from_installed)
            .collect(),
        allow_additional_properties: schema.allow_additional_properties,
    }
}

fn plugin_object_property_from_installed(
    property: &InstalledPluginObjectProperty,
) -> PluginObjectPropertySchema {
    PluginObjectPropertySchema {
        key: property.key.clone(),
        value_schema: plugin_value_schema_from_installed(&property.value_schema),
        required: property.required,
        description: property.description.clone(),
    }
}

fn plugin_constraint_from_installed(
    constraint: &InstalledPluginConstraint,
) -> PluginSettingConstraint {
    match constraint {
        InstalledPluginConstraint::NonEmpty => PluginSettingConstraint::NonEmpty,
        InstalledPluginConstraint::Positive => PluginSettingConstraint::Positive,
        InstalledPluginConstraint::Range { min, max } => PluginSettingConstraint::Range {
            min: min.clone(),
            max: max.clone(),
        },
        InstalledPluginConstraint::AllowedValues { values } => {
            PluginSettingConstraint::AllowedValues {
                values: values.clone(),
            }
        }
        InstalledPluginConstraint::Requires { key } => {
            PluginSettingConstraint::Requires { key: key.clone() }
        }
    }
}

fn handle_validation_result(
    path: PathBuf,
    diagnostics: Vec<ConfigDiagnostic>,
    json: bool,
) -> Result<()> {
    let report = ConfigValidateReport::from_diagnostics(path, diagnostics);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human_report(&report);
    }

    if report.ok {
        Ok(())
    } else {
        bail!("config validation failed")
    }
}

fn print_validation_load_error(path: Option<&Path>, err: &anyhow::Error, json: bool) -> Result<()> {
    let report = ConfigValidateReport::from_error(path.map(Path::to_path_buf), err.to_string());
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let path = report.path.as_deref().unwrap_or("<unresolved>");
    println!("Config invalid: {path}");
    println!("  error: {err}");
    Ok(())
}

fn print_human_report(report: &ConfigValidateReport) {
    let path = report.path.as_deref().unwrap_or("<unresolved>");
    if report.ok {
        println!("Config valid: {path}");
    } else {
        println!("Config invalid: {path}");
    }

    for diagnostic in &report.diagnostics {
        print_human_diagnostic(diagnostic);
    }
}

fn print_human_diagnostic(diagnostic: &ConfigDiagnosticPayload) {
    let path = diagnostic
        .path
        .as_deref()
        .map(|path| format!(" at {path}"))
        .unwrap_or_default();
    println!(
        "  {} {:?}{}: {}",
        severity_label(diagnostic.severity),
        diagnostic.code,
        path,
        diagnostic.message
    );
    if let Some(help) = diagnostic.help.as_deref() {
        println!("    help: {help}");
    }
}

const fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
    match severity {
        ConfigDiagnosticSeverity::Error => "error",
        ConfigDiagnosticSeverity::Warning => "warning",
        ConfigDiagnosticSeverity::Info => "info",
    }
}

#[derive(Clone, Debug, Serialize)]
struct ConfigValidateReport {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    diagnostics: Vec<ConfigDiagnosticPayload>,
}

impl ConfigValidateReport {
    fn from_diagnostics(path: PathBuf, diagnostics: Vec<ConfigDiagnostic>) -> Self {
        let ok = !diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
        Self {
            ok,
            path: Some(path.display().to_string()),
            error: None,
            diagnostics: diagnostics
                .iter()
                .map(ConfigDiagnosticPayload::from)
                .collect(),
        }
    }

    fn from_error(path: Option<PathBuf>, error: String) -> Self {
        Self {
            ok: false,
            path: path.map(|path| path.display().to_string()),
            error: Some(error),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ConfigDiagnosticPayload {
    code: ConfigDiagnosticCode,
    severity: ConfigDiagnosticSeverity,
    source: ConfigDiagnosticSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_source: Option<ConfigDiagnosticSchemaSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_path: Option<String>,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    help: Option<String>,
}

impl From<&ConfigDiagnostic> for ConfigDiagnosticPayload {
    fn from(diagnostic: &ConfigDiagnostic) -> Self {
        Self {
            code: diagnostic.code,
            severity: diagnostic.severity,
            source: diagnostic.source,
            schema_source: diagnostic.schema_source,
            path: diagnostic.path.as_ref().map(ConfigPath::render),
            canonical_path: diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            message: diagnostic.message.clone(),
            help: diagnostic.help.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_config::{ConfigDiagnosticSeverity, validate_config_diagnostics};
    use std::collections::BTreeSet;
    use tempfile::TempDir;

    const VALID_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_valid.toml"
    ));
    const INVALID_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_invalid.toml"
    ));

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct DiagnosticSignature {
        path: String,
        canonical_path: String,
        severity: &'static str,
        code: &'static str,
    }

    impl DiagnosticSignature {
        fn new(
            path: String,
            canonical_path: String,
            severity: &'static str,
            code: &'static str,
        ) -> Self {
            Self {
                path,
                canonical_path,
                severity,
                code,
            }
        }
    }

    fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
        match severity {
            ConfigDiagnosticSeverity::Error => "error",
            ConfigDiagnosticSeverity::Warning => "warning",
            ConfigDiagnosticSeverity::Info => "info",
        }
    }

    fn code_label(code: ConfigDiagnosticCode) -> &'static str {
        match code {
            ConfigDiagnosticCode::InvalidValue => "invalid_value",
            ConfigDiagnosticCode::MissingRequiredValue => "missing_required_value",
            ConfigDiagnosticCode::UnknownField => "unknown_field",
            ConfigDiagnosticCode::UnsupportedField => "unsupported_field",
            ConfigDiagnosticCode::RejectedField => "rejected_field",
            ConfigDiagnosticCode::MisplacedField => "misplaced_field",
            ConfigDiagnosticCode::SchemaUnavailable => "schema_unavailable",
            ConfigDiagnosticCode::LegacyUnvalidatedConfig => "legacy_unvalidated_config",
            ConfigDiagnosticCode::AliasApplied => "alias_applied",
            ConfigDiagnosticCode::UnsupportedSchemaVersion => "unsupported_schema_version",
        }
    }

    fn write_fixture_file(raw: &str) -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("fixture tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, raw).expect("write fixture config");
        (dir, path)
    }

    fn signatures_from_report(report: &ConfigValidateReport) -> BTreeSet<DiagnosticSignature> {
        report
            .diagnostics
            .iter()
            .map(|diagnostic| {
                DiagnosticSignature::new(
                    diagnostic.path.clone().expect("report should include path"),
                    diagnostic
                        .canonical_path
                        .clone()
                        .expect("report should include canonical path"),
                    severity_label(diagnostic.severity),
                    code_label(diagnostic.code),
                )
            })
            .collect()
    }

    fn expected_signatures(raw: &str) -> BTreeSet<DiagnosticSignature> {
        let config: MeshConfig = toml::from_str(raw).expect("fixture should deserialize");
        validate_config_diagnostics(&config)
            .into_iter()
            .map(|diagnostic| {
                DiagnosticSignature::new(
                    diagnostic
                        .path
                        .as_ref()
                        .map(ConfigPath::render)
                        .expect("validator diagnostics should include path"),
                    diagnostic
                        .canonical_path
                        .as_ref()
                        .map(ConfigPath::render)
                        .expect("validator diagnostics should include canonical path"),
                    severity_label(diagnostic.severity),
                    code_label(diagnostic.code),
                )
            })
            .collect()
    }

    #[test]
    fn config_validate_report_keeps_warning_only_diagnostics_successful() {
        let diagnostic = ConfigDiagnostic::warning(
            ConfigDiagnosticCode::LegacyUnvalidatedConfig,
            ConfigDiagnosticSource::Plugin,
            "plugin accepts unvalidated settings",
        )
        .at_path(plugin_settings_path("flash-moe"));

        let report =
            ConfigValidateReport::from_diagnostics(PathBuf::from("config.toml"), vec![diagnostic]);

        assert!(report.ok);
        assert_eq!(
            report.diagnostics[0].path.as_deref(),
            Some("plugin[\"flash-moe\"].settings")
        );
    }

    #[test]
    fn config_validate_report_marks_error_diagnostics_invalid() {
        let diagnostic = ConfigDiagnostic::error(
            ConfigDiagnosticCode::MissingRequiredValue,
            ConfigDiagnosticSource::Schema,
            "required plugin setting is missing",
        )
        .at_path(plugin_settings_path("flash-moe"));

        let report =
            ConfigValidateReport::from_diagnostics(PathBuf::from("config.toml"), vec![diagnostic]);

        assert!(!report.ok);
    }

    #[test]
    fn config_validate_error_report_serializes_stable_json_shape() {
        let report = ConfigValidateReport::from_error(
            Some(PathBuf::from("/tmp/config.toml")),
            "failed to parse config TOML".to_string(),
        );
        let json = serde_json::to_value(report).unwrap();

        assert_eq!(json["ok"], false);
        assert_eq!(json["path"], "/tmp/config.toml");
        assert_eq!(json["error"], "failed to parse config TOML");
        assert_eq!(json["diagnostics"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn config_validate_file_accepts_schema_driven_valid_fixture() {
        let (_dir, path) = write_fixture_file(VALID_FIXTURE);

        let validation =
            validate_config_file(Some(path.as_path())).expect("valid fixture should validate");

        assert!(validation.diagnostics.is_empty());
        assert_eq!(validation.path, path);
    }

    #[test]
    fn config_validate_file_matches_validator_signatures_for_schema_driven_invalid_fixture() {
        let (_dir, path) = write_fixture_file(INVALID_FIXTURE);

        let validation = validate_config_file(Some(path.as_path()))
            .expect("invalid fixture should deserialize and report diagnostics");
        let report =
            ConfigValidateReport::from_diagnostics(validation.path, validation.diagnostics);

        assert!(!report.ok);
        assert_eq!(
            signatures_from_report(&report),
            expected_signatures(INVALID_FIXTURE)
        );
    }

    #[test]
    fn config_validate_file_reports_diagnostics_for_semantically_invalid_file_without_hard_erroring()
     {
        let (_dir, path) = write_fixture_file(INVALID_FIXTURE);

        let validation = validate_config_file(Some(path.as_path())).expect(
            "a semantically invalid but structurally valid config file must deserialize and \
             report diagnostics through validate_config_file, never hard-error \
             (regression guard for e4a759975, which routed this path through the \
             hard-failing mesh_llm_config::load_config)",
        );

        assert!(
            !validation.diagnostics.is_empty(),
            "the invalid fixture must produce a non-empty diagnostics list"
        );
    }

    #[test]
    #[serial_test::serial]
    fn config_validate_file_still_rejects_invalid_lifecycle_log_parser_env_override() {
        let (_dir, path) = write_fixture_file(VALID_FIXTURE);

        let result = mesh_llm_config::with_env_override_for_test(
            "MESH_LLM_LIFECYCLE_LOG_PARSER",
            "not-a-real-mode",
            || validate_config_file(Some(path.as_path())),
        );

        let err = result.expect_err(
            "an invalid MESH_LLM_LIFECYCLE_LOG_PARSER value must still fail \
             `mesh-llm config validate` (Task 17's env-override contract), even though \
             validate_config_file no longer hard-errors on semantic diagnostics",
        );
        assert!(format!("{err:#}").contains("MESH_LLM_LIFECYCLE_LOG_PARSER"));
    }

    fn plugin_settings_path(plugin_name: &str) -> ConfigPath {
        let mut path = ConfigPath::field("plugin");
        path.push_key(plugin_name).push_field("settings");
        path
    }
}
