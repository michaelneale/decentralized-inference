fn top_level_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting.visibility = if path == "version" {
        ConfigVisibility::Internal
    } else {
        ConfigVisibility::Advanced
    };
    setting
}

fn owner_control_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::OwnerControl,
    ];
    setting.apply_mode = ConfigApplyMode::DynamicApply;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn telemetry_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn logging_audit_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.apply_mode = ConfigApplyMode::StaticOnLoad;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    if matches!(
        path,
        "logging.audit.max_file_size_mb" | "logging.audit.max_files"
    ) {
        setting.constraints.push(ConfigConstraint::Range {
            min: Some("1".to_string()),
            max: None,
        });
    }
    setting
}

fn runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.apply_mode = ConfigApplyMode::DynamicValidationOnly;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn native_runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.apply_mode = ConfigApplyMode::DynamicValidationOnly;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting.description = Some(
        "Native runtime selection is read before dynamic runtime libraries are loaded.".into(),
    );
    setting
}

fn startup_runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn activity_runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn kv_disk_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    dynamic: bool,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::Cli,
        ConfigControlSurface::OwnerControl,
        ConfigControlSurface::Api,
        ConfigControlSurface::Ui,
    ];
    setting.visibility = ConfigVisibility::User;
    if dynamic {
        setting.apply_mode = ConfigApplyMode::DynamicApply;
        setting.restart_scope = ConfigRestartScope::None;
    } else {
        setting.apply_mode = ConfigApplyMode::StaticOnLoad;
        setting.restart_scope = ConfigRestartScope::ProcessRestart;
    }
    setting
}

fn plugin_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::PluginManifest,
    ];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn logging_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile];
    // Logging settings requiring process restart (structural / path / capability changes).
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting.visibility = ConfigVisibility::Advanced;
    setting
}

fn logging_dynamic_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile];
    setting.apply_mode = ConfigApplyMode::DynamicApply;
    setting.restart_scope = ConfigRestartScope::None;
    setting.visibility = ConfigVisibility::Advanced;
    setting
}

fn logging_settings() -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        logging_setting("logging.enabled", ConfigValueSchema::Boolean),
        logging_setting("logging.application_state_root", ConfigValueSchema::Path),
        logging_setting("logging.summary_line_limit", ConfigValueSchema::Integer),
        logging_setting("logging.event_buffer_size", ConfigValueSchema::Integer),
        // Only these two limits have a dynamic service-application contract.
        logging_dynamic_setting("logging.retention_ttl_secs", ConfigValueSchema::Integer),
        logging_setting("logging.retention_max_rows", ConfigValueSchema::Integer),
        logging_dynamic_setting("logging.replay_capacity", ConfigValueSchema::Integer),
        logging_setting("logging.queue_capacity", ConfigValueSchema::Integer),
        logging_setting("logging.cleanup_cadence_secs", ConfigValueSchema::Integer),
        logging_setting(
            "logging.artifact.capture_mode",
            string_enum(["metadata_only", "redacted_artifacts"]),
        ),
        logging_setting(
            "logging.artifact.byte_limit_bytes",
            ConfigValueSchema::Integer,
        ),
        logging_setting(
            "logging.artifact.aggregate_limit_bytes",
            ConfigValueSchema::Integer,
        ),
        logging_setting("logging.export_limit_bytes", ConfigValueSchema::Integer),
        logging_setting("logging.webhook.enabled", ConfigValueSchema::Boolean),
        logging_setting("logging.webhook.url", ConfigValueSchema::Url),
        logging_setting("logging.webhook.max_attempts", ConfigValueSchema::Integer),
        logging_setting("logging.webhook.timeout_secs", ConfigValueSchema::Integer),
        logging_setting(
            "logging.webhook.dead_letter_retention_secs",
            ConfigValueSchema::Integer,
        ),
    ];

    for setting in &mut settings {
        let path = setting.path.render();
        setting.description = match path.as_str() {
            "logging.summary_line_limit" => Some(
                "Maximum number of Unicode characters in each generated request summary."
                    .to_string(),
            ),
            "logging.event_buffer_size" => Some(
                "Maximum number of event entries held in memory for replay."
                    .to_string(),
            ),
            "logging.replay_capacity" => Some(
                "Number of recent events available to reconnecting console clients."
                    .to_string(),
            ),
            "logging.queue_capacity" => Some(
                "Maximum number of pending log entries waiting for persistence and webhook dispatch."
                    .to_string(),
            ),
            _ => setting.description.take(),
        };
        let range = match path.as_str() {
            "logging.summary_line_limit" => Some(("1", "65536")),
            "logging.event_buffer_size" => Some(("50", "100000")),
            "logging.retention_ttl_secs" => Some(("3600", "7776000")),
            "logging.replay_capacity" => Some(("1", "10000")),
            "logging.queue_capacity" => Some(("64", "131072")),
            "logging.cleanup_cadence_secs" => Some(("300", "86400")),
            "logging.artifact.byte_limit_bytes" => Some(("1024", "16777216")),
            "logging.artifact.aggregate_limit_bytes" => Some(("524288", "524288000")),
            "logging.export_limit_bytes" => Some(("65536", "104857600")),
            "logging.webhook.max_attempts" => Some(("1", "20")),
            "logging.webhook.timeout_secs" => Some(("1", "60")),
            "logging.webhook.dead_letter_retention_secs" => Some(("3600", "1555200")),
            _ => None,
        };
        if let Some((min, max)) = range {
            setting.constraints.push(ConfigConstraint::Range {
                min: Some(min.to_string()),
                max: Some(max.to_string()),
            });
        }
    }

    settings
}

fn basic_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    ConfigSettingSchema {
        path: schema_path(path),
        alias_policy: ConfigAliasPolicy::default(),
        owner: ConfigSettingOwner::BuiltIn,
        value_schema,
        support: ConfigSupportState::Supported,
        control_surfaces: vec![ConfigControlSurface::ConfigFile],
        apply_mode: ConfigApplyMode::StaticOnLoad,
        restart_scope: ConfigRestartScope::ModelReload,
        visibility: ConfigVisibility::Advanced,
        constraints: Vec::new(),
        description: Some(path.to_string()),
        presentation: None,
        control_behavior: None,
    }
}

fn rejected_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Rejected;
    setting.restart_scope = ConfigRestartScope::None;
    setting.description = Some(description.to_string());
    setting
}

fn unwired_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Unwired;
    setting
}

fn experimental_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Experimental;
    setting
}

fn unsupported_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Unsupported;
    setting.restart_scope = ConfigRestartScope::None;
    setting
}

fn hidden_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.visibility = ConfigVisibility::Hidden;
    setting.description = Some(description.to_string());
    setting
}

fn hidden_unsupported_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = hidden_setting(path, value_schema, description);
    setting.support = ConfigSupportState::Unsupported;
    setting.restart_scope = ConfigRestartScope::None;
    setting
}

fn schema_path(path: &str) -> ConfigPath {
    ConfigPath::parse_rendered(path).expect("static schema path should parse")
}

fn flat_alias(path: &str) -> ConfigPathAlias {
    ConfigPathAlias {
        path: schema_path(path),
        kind: ConfigPathAliasKind::LegacyLayout,
        note: Some("legacy flattened TOML field".into()),
    }
}

fn string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    ConfigValueSchema::Enum {
        values: values.into_iter().map(str::to_string).collect(),
    }
}

fn string_enum_from_slice(values: &[&str]) -> ConfigValueSchema {
    ConfigValueSchema::Enum {
        values: values.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn kv_cache_type_schema() -> ConfigValueSchema {
    string_enum([
        "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
    ])
}

fn one_of<const N: usize>(variants: [ConfigValueSchema; N]) -> ConfigValueSchema {
    ConfigValueSchema::OneOf {
        variants: variants.into_iter().collect(),
    }
}

fn bool_or_auto_schema() -> ConfigValueSchema {
    bool_or_string_enum(["auto", "true", "false"])
}

fn bool_or_string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    one_of([ConfigValueSchema::Boolean, string_enum(values)])
}

fn integer_or_auto_schema() -> ConfigValueSchema {
    integer_or_string_enum(["auto"])
}

fn integer_or_string_schema() -> ConfigValueSchema {
    one_of([ConfigValueSchema::Integer, ConfigValueSchema::String])
}

fn integer_or_string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    one_of([ConfigValueSchema::Integer, string_enum(values)])
}

fn string_or_list_schema() -> ConfigValueSchema {
    one_of([
        ConfigValueSchema::String,
        ConfigValueSchema::Array {
            items: Box::new(ConfigValueSchema::String),
        },
    ])
}

fn tensor_split_schema() -> ConfigValueSchema {
    one_of([
        ConfigValueSchema::Array {
            items: Box::new(ConfigValueSchema::Float),
        },
        ConfigValueSchema::String,
    ])
}

/// Returns the list of known mesh-llm versions from GitHub releases.
/// This list should be updated during the release process.
fn known_mesh_llm_versions() -> &'static [&'static str] {
    &[
        "0.76.0-rc9",
        "0.76.0-rc8",
        "0.76.0-rc7",
        "0.76.0-rc6",
        "0.76.0-rc5",
        "0.76.0-rc4",
        "0.76.0-rc3",
        "0.76.0-rc2",
        "0.76.0-rc1",
        "0.75.1",
        "0.75.0",
        "0.74.0",
        "0.73.1",
        "0.73.0",
        "0.72.2",
        "0.72.1",
        "0.72.0",
        "0.71.0",
        "0.70.0",
        "0.69.0",
        "0.68.0",
        "0.67.0",
        "0.66.0",
        "0.65.0",
        "0.64.0",
        "0.63.0",
        "0.62.0",
        "0.61.0",
        "0.60.0",
    ]
}

fn apply_aliases(
    settings: &mut [ConfigSettingSchema],
    canonical_path: &str,
    aliases: &[ConfigPathAlias],
) {
    if let Some(setting) = settings
        .iter_mut()
        .find(|setting| setting.path.render() == canonical_path)
    {
        setting.alias_policy.mode = ConfigAliasMode::CanonicalWithLegacyAliases;
        setting.alias_policy.aliases.extend_from_slice(aliases);
    }
}
