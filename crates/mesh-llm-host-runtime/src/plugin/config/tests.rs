use super::*;
use crate::plugin::schema_validation::plugin_schema_availability_from_store_root;
use mesh_llm_config::{
    ConfigDiagnosticCode, ConfigDiagnosticSeverity, FlashAttentionType,
    validate_config_diagnostics_with_plugin_schemas, with_env_override_for_test,
};
use mesh_llm_plugin_manager::{
    InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginConstraint,
    InstalledPluginManifestMetadata, InstalledPluginMetadata, InstalledPluginRestartScope,
    InstalledPluginSettingSchema, InstalledPluginValueKind, InstalledPluginValueSchema,
    InstalledPluginVisibility, PluginStore,
};
use std::collections::BTreeSet;
use tempfile::TempDir;

const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../../tests/fixtures/skippy_full_surface_valid.toml");
const FULL_SURFACE_INVALID_FIXTURE: &str =
    include_str!("../../../tests/fixtures/skippy_full_surface_invalid.toml");

// Every name in `mesh_llm_config::CONFIG_OVERRIDE_ENV_NAMES` must reach a
// production caller through THIS crate's plugin-aware `load_config` /
// `validate_config_file` wrappers -- not just through the base
// `mesh_llm_config::load_config` the wrappers wrap. Update this list (and
// add a dedicated wrapper-path test below) whenever env_overrides.rs
// grows, so a new override can never silently regress into the same
// bypass IT-plan Task 17 found here.
#[test]
fn config_override_env_names_owner_totality_is_unchanged() {
    assert_eq!(
        mesh_llm_config::CONFIG_OVERRIDE_ENV_NAMES,
        &[
            "MESH_LLM_CONFIG",
            "MESH_LLM_LIFECYCLE_LOG_PARSER",
            "MESH_LLM_BENCHMARK_TUNE_TRIAL",
            "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
        ]
    );
}

#[test]
#[serial_test::serial]
fn load_config_through_production_wrapper_applies_lifecycle_log_parser_env_override() {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "version = 1\n").unwrap();

    let result = with_env_override_for_test("MESH_LLM_LIFECYCLE_LOG_PARSER", "enabled", || {
        load_config(Some(&config_path))
    });

    let config = result.expect("config should load through the production wrapper");
    assert_eq!(
        config.runtime.lifecycle_log_parser,
        mesh_llm_config::LifecycleLogParserMode::Enabled,
        "the plugin-aware production load_config wrapper must honor \
             MESH_LLM_LIFECYCLE_LOG_PARSER the same way mesh_llm_config::load_config does"
    );
    assert_eq!(
        config.runtime.lifecycle_log_parser_source,
        mesh_llm_config::ConfigValueSource::Env
    );
}

#[test]
#[serial_test::serial]
fn load_config_through_production_wrapper_applies_env_override_when_file_absent() {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("does-not-exist.toml");

    let result = with_env_override_for_test("MESH_LLM_LIFECYCLE_LOG_PARSER", "enabled", || {
        load_config(Some(&config_path))
    });

    let config =
        result.expect("a default config (no file on disk) must still honor the env override");
    assert_eq!(
        config.runtime.lifecycle_log_parser,
        mesh_llm_config::LifecycleLogParserMode::Enabled
    );
}

#[test]
#[serial_test::serial]
fn load_config_through_production_wrapper_rejects_invalid_lifecycle_log_parser_env_override() {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "version = 1\n").unwrap();

    let result =
        with_env_override_for_test("MESH_LLM_LIFECYCLE_LOG_PARSER", "not-a-real-mode", || {
            load_config(Some(&config_path))
        });

    let err = result.expect_err(
        "an invalid MESH_LLM_LIFECYCLE_LOG_PARSER value must fail startup through the \
             production plugin-aware wrapper, never be silently dropped or defaulted",
    );
    assert!(format!("{err:#}").contains("MESH_LLM_LIFECYCLE_LOG_PARSER"));
}

#[test]
#[serial_test::serial]
fn validate_config_file_through_production_wrapper_rejects_invalid_lifecycle_log_parser_env_override()
 {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "version = 1\n").unwrap();

    let result =
        with_env_override_for_test("MESH_LLM_LIFECYCLE_LOG_PARSER", "not-a-real-mode", || {
            validate_config_file(Some(&config_path))
        });

    let err = result.expect_err(
        "an invalid MESH_LLM_LIFECYCLE_LOG_PARSER value must fail `mesh-llm config validate` \
             through the production plugin-aware wrapper, never be silently dropped",
    );
    assert!(format!("{err:#}").contains("MESH_LLM_LIFECYCLE_LOG_PARSER"));
}

fn documented_matrix_key_paths() -> BTreeSet<String> {
    let matrix = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/skippy/CONFIGURATION.md"
    ));
    matrix
        .lines()
        .filter(|line| line.starts_with('|'))
        .filter_map(|line| {
            let columns: Vec<_> = line.split('|').map(str::trim).collect();
            columns.get(3).copied()
        })
        .filter(|cell| cell.contains('`'))
        .flat_map(|cell| {
            cell.split("<br>")
                .filter_map(|part| {
                    let trimmed = part.trim();
                    trimmed
                        .strip_prefix('`')
                        .and_then(|value| value.strip_suffix('`'))
                })
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn test_model(name: &str) -> ModelConfigEntry {
    ModelConfigEntry {
        model: name.into(),
        mmproj: None,
        ctx_size: None,
        gpu_id: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        batch: None,
        ubatch: None,
        flash_attention: None,
        model_fit: None,
        hardware: None,
        throughput: None,
        topology: None,
        skippy: None,
        speculative: None,
        request_defaults: None,
        multimodal: None,
        advanced: None,
        gpu_id_from_legacy_shim: false,
    }
}

fn installed_plugin_metadata(
    name: &str,
    schema: Option<InstalledPluginConfigSchema>,
) -> InstalledPluginMetadata {
    InstalledPluginMetadata {
        name: name.to_string(),
        source_repository: format!("https://github.com/mesh-llm/{name}"),
        installed_version: "v1.0.0".to_string(),
        target_triple: std::env::consts::ARCH.to_string(),
        downloaded_asset_name: format!("{name}.tar.gz"),
        install_path: std::env::temp_dir().join(format!("mesh-llm-plugin-{name}")),
        enabled: true,
        manifest: Some(InstalledPluginManifestMetadata {
            config_schema: schema,
            web_ui: None,
        }),
        last_protocol_version: Some(1),
        last_status: Some("installed".to_string()),
        last_error: None,
    }
}

fn blackboard_schema(
    allow_unvalidated_config: bool,
    schema_version: u32,
) -> InstalledPluginConfigSchema {
    InstalledPluginConfigSchema {
        plugin_name: "blackboard".to_string(),
        schema_version,
        allow_unvalidated_config,
        settings: vec![
            InstalledPluginSettingSchema {
                key: "retention_days".to_string(),
                value_schema: InstalledPluginValueSchema {
                    kind: InstalledPluginValueKind::Integer,
                    enum_values: Vec::new(),
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: true,
                default_json: Some("14".to_string()),
                constraints: vec![InstalledPluginConstraint::Range {
                    min: Some("1".to_string()),
                    max: Some("365".to_string()),
                }],
                apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: InstalledPluginRestartScope::PluginProcess,
                visibility: InstalledPluginVisibility::User,
                description: Some("Retention window".to_string()),
                presentation: None,
                control_behavior: None,
            },
            InstalledPluginSettingSchema {
                key: "mode".to_string(),
                value_schema: InstalledPluginValueSchema {
                    kind: InstalledPluginValueKind::Enum,
                    enum_values: vec!["strict".to_string(), "relaxed".to_string()],
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: false,
                default_json: Some("\"strict\"".to_string()),
                constraints: Vec::new(),
                apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: InstalledPluginRestartScope::PluginProcess,
                visibility: InstalledPluginVisibility::User,
                description: Some("Conflict mode".to_string()),
                presentation: None,
                control_behavior: None,
            },
        ],
    }
}

fn with_plugin_store<F>(metadata: &[InstalledPluginMetadata], test: F)
where
    F: FnOnce(&Path),
{
    let temp = TempDir::new().unwrap();
    let store = PluginStore::new(temp.path());
    for entry in metadata {
        store.save(entry).unwrap();
    }

    test(temp.path());
}

fn parse_config_toml_with_plugin_store(raw: &str, store_root: &Path) -> Result<MeshConfig> {
    let config = base_parse_config_toml(raw)?;
    validate_config_with_plugin_schemas(&config, Some(raw), |plugin_name| {
        plugin_schema_availability_from_store_root(store_root, plugin_name)
    })?;
    Ok(config)
}

fn validate_config_with_plugin_store(config: &MeshConfig, store_root: &Path) -> Result<()> {
    validate_config_with_plugin_schemas(config, None, |plugin_name| {
        plugin_schema_availability_from_store_root(store_root, plugin_name)
    })
}

fn plugin_config_diagnostics_with_plugin_store(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    store_root: &Path,
) -> Vec<ConfigDiagnostic> {
    validate_config_diagnostics_with_plugin_schemas(config, raw_toml, |plugin_name| {
        plugin_schema_availability_from_store_root(store_root, plugin_name)
    })
}

#[test]
fn parse_unified_config_keeps_plugins_and_models() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[owner_control]
bind = "127.0.0.1:7447"
advertise_addr = "203.0.113.10:7447"

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/model.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf"

[[plugin]]
name = "demo"
command = "/tmp/demo"
"#,
    )
    .unwrap();

    assert_eq!(config.version, Some(1));
    assert_eq!(
        config.owner_control.bind,
        Some("127.0.0.1:7447".parse().unwrap())
    );
    assert_eq!(
        config.owner_control.advertise_addr,
        Some("203.0.113.10:7447".parse().unwrap())
    );
    assert_eq!(config.gpu.assignment, GpuAssignment::Auto);
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.models[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(config.models[0].ctx_size, Some(8192));
    assert_eq!(config.models[0].gpu_id, None);
    assert_eq!(config.models[0].cache_type_k, None);
    assert_eq!(config.models[0].cache_type_v, None);
    assert_eq!(config.models[0].batch, None);
    assert_eq!(config.models[0].ubatch, None);
    assert_eq!(config.models[0].flash_attention, None);
    assert_eq!(
        config.models[1].mmproj.as_deref(),
        Some("bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf")
    );
    assert_eq!(config.models[1].gpu_id, None);
    assert_eq!(config.plugins.len(), 1);
    assert_eq!(config.plugins[0].name, "demo");
}

#[test]
#[serial_test::serial]
fn plugin_config_roundtrip() {
    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(blackboard_schema(
                false,
                mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
            )),
        )],
        |store_root| {
            let raw = r#"
version = 1

[[plugin]]
name = "blackboard"
enabled = true
command = "mesh-blackboard-plugin"

[plugin.settings]
retention_days = 14
mode = "strict"
"#;

            let config = parse_config_toml_with_plugin_store(raw, store_root)
                .expect("strict plugin config should parse");
            assert_eq!(
                config.plugins[0].settings["retention_days"].as_integer(),
                Some(14)
            );
            assert_eq!(config.plugins[0].settings["mode"].as_str(), Some("strict"));

            let rendered = config_to_toml(&config).expect("settings should serialize");
            let reparsed = parse_config_toml_with_plugin_store(&rendered, store_root)
                .expect("rendered config should reparse");
            validate_config_with_plugin_store(&reparsed, store_root)
                .expect("strict plugin config should validate");
            assert_eq!(
                reparsed.plugins[0].settings["retention_days"].as_integer(),
                Some(14)
            );
            assert_eq!(
                reparsed.plugins[0].settings["mode"].as_str(),
                Some("strict")
            );
        },
    );

    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(blackboard_schema(
                true,
                mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
            )),
        )],
        |store_root| {
            let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
arbitrary = "kept"
"#;
            let config = base_parse_config_toml(raw).unwrap();
            let diagnostics =
                plugin_config_diagnostics_with_plugin_store(&config, Some(raw), store_root);
            assert!(diagnostics.iter().any(|diagnostic| {
                diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                    && diagnostic.severity == ConfigDiagnosticSeverity::Warning
            }));
        },
    );
}

#[test]
#[serial_test::serial]
fn plugin_config_validation_failures() {
    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(blackboard_schema(
                false,
                mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
            )),
        )],
        |store_root| {
            let raw = r#"
[[plugin]]
name = "blackboard"
retention_days = 14

[plugin.settings]
mode = "mystery"
unknown = true
"#;

            let config = base_parse_config_toml(raw).unwrap();
            let diagnostics =
                plugin_config_diagnostics_with_plugin_store(&config, Some(raw), store_root);

            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::MisplacedField)
            );
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::UnknownField)
            );
            assert!(diagnostics.iter().any(
                    |diagnostic| diagnostic.code == ConfigDiagnosticCode::MissingRequiredValue
                ));
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::InvalidValue)
            );
        },
    );

    with_plugin_store(&[], |store_root| {
        let raw = r#"
[[plugin]]
name = "missing-plugin"

[plugin.settings]
flag = true
"#;
        let config = base_parse_config_toml(raw).unwrap();
        let diagnostics =
            plugin_config_diagnostics_with_plugin_store(&config, Some(raw), store_root);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::SchemaUnavailable)
        );
    });

    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(blackboard_schema(
                false,
                mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION + 1,
            )),
        )],
        |store_root| {
            let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 30
"#;
            let config = base_parse_config_toml(raw).unwrap();
            let diagnostics =
                plugin_config_diagnostics_with_plugin_store(&config, Some(raw), store_root);
            assert!(diagnostics.iter().any(
                |diagnostic| diagnostic.code == ConfigDiagnosticCode::UnsupportedSchemaVersion
            ));
        },
    );
}

#[test]
fn telemetry_config_deserializes_standard_metrics_settings() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048
prompt_shape_metrics = false

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"

[[plugin]]
name = "metrics"
enabled = true
"#,
    )
    .unwrap();

    assert_eq!(config.telemetry.enabled, Some(true));
    assert_eq!(config.telemetry.service_name.as_deref(), Some("mesh-llm"));
    assert_eq!(
        config.telemetry.endpoint.as_deref(),
        Some("https://otel.example.com")
    );
    assert_eq!(
        config.telemetry.metrics.endpoint.as_deref(),
        Some("https://otel.example.com/v1/metrics")
    );
    assert_eq!(
        config
            .telemetry
            .headers
            .get("authorization")
            .map(String::as_str),
        Some("Bearer TOKEN")
    );
    assert_eq!(config.telemetry.export_interval_secs, Some(15));
    assert_eq!(config.telemetry.queue_size, Some(2048));
    assert!(!config.telemetry.prompt_shape_metrics);
}

#[test]
fn telemetry_config_rejects_zero_queue_size() {
    let config: MeshConfig = toml::from_str(
        r#"
[telemetry]
queue_size = 0
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("telemetry.queue_size must be at least 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn owner_control_config_rejects_ephemeral_non_loopback_bind() {
    let config: MeshConfig = toml::from_str(
        r#"
[owner_control]
bind = "0.0.0.0:0"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert!(err.to_string().contains(
        "owner_control.bind must use a concrete port when binding a non-loopback address"
    ));
}

#[test]
fn owner_control_config_rejects_unspecified_advertise_addr() {
    let config: MeshConfig = toml::from_str(
        r#"
[owner_control]
advertise_addr = "0.0.0.0:18443"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("owner_control.advertise_addr must not use an unspecified IP address")
    );
}

#[test]
fn owner_control_config_rejects_ephemeral_advertise_addr() {
    let config: MeshConfig = toml::from_str(
        r#"
[owner_control]
advertise_addr = "127.0.0.1:0"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("owner_control.advertise_addr must use a concrete port")
    );
}

#[test]
fn telemetry_config_accepts_prompt_shape_metrics_after_review() {
    let config: MeshConfig = toml::from_str(
        r#"
[telemetry]
prompt_shape_metrics = true
"#,
    )
    .unwrap();

    validate_config(&config).expect("prompt shape metrics should validate");
}

#[test]
fn pinned_gpu_config_accepted_pinned_config() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
ctx_size = 8192
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    assert_eq!(config.models[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
}

#[test]
fn pinned_gpu_config_missing_gpu_id_rejected() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(err.to_string().contains(
            "models[0].hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
        ));
}

#[test]
fn pinned_gpu_config_accepts_defaults_hardware_device_for_models() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[defaults.hardware]
device = "CUDA0"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    assert!(config.models[0].hardware.is_none());
}

#[test]
fn pinned_gpu_config_allows_defaults_hardware_without_device_when_models_pin_devices() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[defaults.hardware]
gpu_layers = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
device = "CUDA1"
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    assert_eq!(config.models[0].gpu_id.as_deref(), Some("CUDA1"));
}

#[test]
fn pinned_gpu_config_empty_gpu_id_rejected() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![ModelConfigEntry {
            gpu_id: Some("  \t  ".into()),
            gpu_id_from_legacy_shim: true,
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].hardware.device must not be empty when set")
    );
}

#[test]
fn hardware_gpu_layers_rejects_i32_overflow() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
gpu_layers = 2147483648
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].hardware.gpu_layers must be at most 2147483647"
    );
}

#[test]
fn pinned_gpu_config_auto_assignment_rejects_gpu_id() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        models: vec![ModelConfigEntry {
            gpu_id: Some("pci:0000:65:00.0".into()),
            gpu_id_from_legacy_shim: true,
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].hardware.device must not be set when gpu.assignment = \"auto\"")
    );
}

#[test]
fn pinned_gpu_config_preserves_accepted_gpu_id_string_exactly() {
    let raw = r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = " pci:0000:65:00.0 "
"#;

    let config: MeshConfig = toml::from_str(raw).unwrap();
    validate_config(&config).unwrap();

    assert_eq!(
        config.models[0].gpu_id.as_deref(),
        Some(" pci:0000:65:00.0 ")
    );
}

// ── gpu.parallel validation ──

#[test]
fn gpu_parallel_field_deserializes_from_toml() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "auto"
parallel = 8

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
    )
    .unwrap();

    assert_eq!(config.gpu.parallel, Some(8));
}

#[test]
fn gpu_parallel_defaults_to_none_when_omitted() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
    )
    .unwrap();

    assert_eq!(config.gpu.parallel, None);
}

#[test]
fn gpu_parallel_zero_rejected() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: Some(0),
        },
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("gpu.parallel must be at least 1, got 0"),
        "unexpected error message: {err}"
    );
}

#[test]
fn gpu_parallel_one_accepted() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: Some(1),
        },
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn gpu_parallel_none_accepted() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn gpu_parallel_large_value_accepted() {
    let config = MeshConfig {
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: Some(64),
        },
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn gpu_parallel_unwrap_or_default_is_4() {
    fn parsed_parallel(value: Option<usize>) -> usize {
        value.unwrap_or(4)
    }

    assert_eq!(parsed_parallel(None), 4);
    assert_eq!(parsed_parallel(Some(1)), 1);
    assert_eq!(parsed_parallel(Some(8)), 8);
    assert_eq!(parsed_parallel(Some(64)), 64);
}

#[test]
fn per_model_parallel_valid_value_accepted() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            parallel: Some(8),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };
    validate_config(&config).unwrap();
}

#[test]
fn per_model_parallel_zero_rejected() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            parallel: Some(0),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };
    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].throughput.parallel must be at least 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn per_model_parallel_none_accepted() {
    let config = MeshConfig {
        models: vec![test_model("Qwen3-8B-Q4_K_M")],
        ..MeshConfig::default()
    };
    validate_config(&config).unwrap();
}

#[test]
fn model_runtime_overrides_deserialize_from_toml() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 2048
ubatch = 512
flash_attention = "enabled"
"#,
    )
    .unwrap();

    assert_eq!(config.models[0].cache_type_k.as_deref(), Some("q8_0"));
    assert_eq!(config.models[0].cache_type_v.as_deref(), Some("q4_0"));
    assert_eq!(config.models[0].batch, Some(2048));
    assert_eq!(config.models[0].ubatch, Some(512));
    assert_eq!(
        config.models[0].flash_attention,
        Some(FlashAttentionType::Enabled)
    );
}

#[test]
fn model_cache_type_k_empty_rejected() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            cache_type_k: Some("   ".into()),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].model_fit.cache_type_k must not be empty when set")
    );
}

#[test]
fn model_cache_type_v_empty_rejected() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            cache_type_v: Some("   ".into()),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].model_fit.cache_type_v must not be empty when set")
    );
}

#[test]
fn model_batch_zero_rejected() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            batch: Some(0),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].model_fit.batch must be between 1 and 10000000, got 0")
    );
}

#[test]
fn model_ubatch_zero_rejected() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            ubatch: Some(0),
            ..test_model("Qwen3-8B-Q4_K_M")
        }],
        ..MeshConfig::default()
    };

    let err = validate_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("models[0].model_fit.ubatch must be between 1 and 10000000, got 0")
    );
}

#[test]
fn defaults_nested_sections_preserve_existing_behavior_when_omitted() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
parallel = 4
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    assert!(config.defaults.is_none());
    assert_eq!(config.models[0].ctx_size, Some(8192));
    assert_eq!(config.models[0].parallel, Some(4));
    assert_eq!(
        config.models[0].model_fit.as_ref().and_then(|v| v.ctx_size),
        Some(8192)
    );
    assert_eq!(
        config.models[0]
            .throughput
            .as_ref()
            .and_then(|v| v.parallel),
        Some(4)
    );
}

#[test]
fn nested_defaults_parse_representative_sections() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[defaults.model_fit]
ctx_size = 4096
kv_cache_policy = "balanced"

[defaults.hardware]
gpu_layers = 10

[defaults.throughput]
parallel = 2

[defaults.speculative]
mode = "disabled"

[defaults.request_defaults]
temperature = 0.2

[defaults.multimodal]
image_max_tokens = 4096

[defaults.advanced.server]
alias = "qwen-local"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    let defaults = config.defaults.expect("defaults should parse");
    assert_eq!(defaults.model_fit.and_then(|v| v.ctx_size), Some(4096));
    assert_eq!(
        defaults.hardware.and_then(|v| v.gpu_layers),
        Some(IntegerOrString::Integer(10))
    );
    assert_eq!(defaults.throughput.and_then(|v| v.parallel), Some(2));
    assert_eq!(
        defaults.speculative.and_then(|v| v.mode),
        Some("disabled".into())
    );
}

#[test]
fn canonical_plan_example_auto_sentinels_parse_and_validate() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[defaults.model_fit]
ctx_size = 8192
batch = 512
ubatch = 128
kv_cache_policy = "auto"
cache_type_k = "auto"
cache_type_v = "auto"
kv_offload = "auto"
kv_unified = "auto"
cache_ram_mib = 0
cache_idle_slots = 0
prompt_cache = "auto"
context_shift = "auto"

[defaults.hardware]
gpu_layers = "auto"
safety_margin_gb = 2.0
mmap = "auto"
mlock = false

[defaults.throughput]
parallel = 1
continuous_batching = "auto"
threads = 0
threads_batch = 0
tuning_profile = "balanced"

[defaults.skippy]
prefill_chunking = "auto"
prefill_chunk_size = 0

[defaults.speculative]
mode = "auto"
draft_selection_policy = "auto"
pairing_fault = "warn_disable"
draft_max_tokens = 16
draft_min_tokens = 1
draft_acceptance_threshold = 0.0

[defaults.request_defaults]
temperature = 0.8
top_p = 0.95
top_k = 40
min_p = 0.0
repeat_penalty = 1.0
repeat_last_n = 64
reasoning_format = "auto"
reasoning_budget = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[models.model_fit]
ctx_size = 16384
cache_type_k = "q8_0"
cache_type_v = "q8_0"

[models.hardware]
gpu_layers = 99
device = "cuda:0"
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    let defaults = config.defaults.as_ref().expect("defaults should parse");
    assert!(matches!(
        defaults.model_fit.as_ref().and_then(|v| v.kv_unified.as_ref()),
        Some(BoolOrAuto::String(value)) if value == "auto"
    ));
    assert!(matches!(
        defaults.hardware.as_ref().and_then(|v| v.gpu_layers.as_ref()),
        Some(IntegerOrString::String(value)) if value == "auto"
    ));
    assert!(matches!(
        defaults.request_defaults.as_ref().and_then(|v| v.reasoning_budget.as_ref()),
        Some(ReasoningBudget::String(value)) if value == "auto"
    ));
    assert_eq!(config.models[0].ctx_size, Some(16384));
    assert_eq!(config.models[0].gpu_id.as_deref(), Some("cuda:0"));
}

#[test]
fn partial_hardware_controls_remain_rejected_at_validation_boundary() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
split_mode = "row"
main_gpu = 0
direct_io = true
"#,
    )
    .expect("partial hardware controls must parse before validation");

    let diagnostics = mesh_llm_config::validate_config_diagnostics(&config);
    let rejected = diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::UnsupportedField
                && diagnostic.severity == ConfigDiagnosticSeverity::Error
        })
        .filter_map(|diagnostic| diagnostic.path.as_ref().map(|path| path.render()))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        rejected,
        BTreeSet::from([
            "defaults.hardware.direct_io".to_string(),
            "defaults.hardware.main_gpu".to_string(),
            "defaults.hardware.split_mode".to_string(),
        ])
    );
}

#[test]
fn legacy_flat_fields_normalize_into_nested_sections() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
gpu_id = "pci:0000:65:00.0"
parallel = 6
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 1024
ubatch = 256
flash_attention = "enabled"
mmproj = "projector.gguf"
"#,
    )
    .unwrap();

    let model = &config.models[0];
    assert_eq!(
        model.model_fit.as_ref().and_then(|v| v.ctx_size),
        Some(8192)
    );
    assert_eq!(
        model.hardware.as_ref().and_then(|v| v.device.as_deref()),
        Some("pci:0000:65:00.0")
    );
    assert_eq!(model.throughput.as_ref().and_then(|v| v.parallel), Some(6));
    assert_eq!(
        model
            .model_fit
            .as_ref()
            .and_then(|v| v.cache_type_k.as_deref()),
        Some("q8_0")
    );
    assert_eq!(model.model_fit.as_ref().and_then(|v| v.batch), Some(1024));
    assert_eq!(
        model.multimodal.as_ref().and_then(|v| v.mmproj.as_deref()),
        Some("projector.gguf")
    );
}

#[test]
fn nested_values_override_legacy_shims() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 4096
gpu_id = "legacy-gpu"
parallel = 2
batch = 256
mmproj = "legacy.gguf"

[models.model_fit]
ctx_size = 8192
batch = 1024

[models.hardware]
device = "nested-gpu"

[models.throughput]
parallel = 8

[models.multimodal]
mmproj = "nested.gguf"
"#,
    )
    .unwrap();

    validate_config(&config).unwrap();
    let model = &config.models[0];
    assert_eq!(model.ctx_size, Some(8192));
    assert_eq!(model.batch, Some(1024));
    assert_eq!(model.gpu_id.as_deref(), Some("nested-gpu"));
    assert_eq!(model.parallel, Some(8));
    assert_eq!(model.mmproj.as_deref(), Some("nested.gguf"));
}

#[test]
fn invalid_model_fit_batch_path_is_stable() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
batch = 0
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].model_fit.batch must be between 1 and 10000000, got 0"
    );
}

#[test]
fn invalid_split_mode_path_is_stable() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
split_mode = "diagonal"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].hardware.split_mode must be one of: auto, none, layer, row, tensor"
    );
}

#[test]
fn invalid_reasoning_format_path_is_stable() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
    );
}

#[test]
fn deepseek_legacy_reasoning_format_is_accepted() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "deepseek-legacy"
"#,
    )
    .unwrap();

    validate_config(&config).expect("deepseek-legacy should remain accepted");
}

#[test]
fn invalid_speculative_draft_requires_policy_path_is_stable() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.speculative]
mode = "draft"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].speculative.draft_selection_policy must be set when models[0].speculative.mode = \"draft\" and no explicit draft model source is configured"
    );
}

#[test]
fn invalid_mmproj_conflict_is_rejected() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
mmproj = "hardware.gguf"

[models.multimodal]
mmproj = "multimodal.gguf"
"#,
    )
    .unwrap();

    let err = validate_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "models[0].multimodal.mmproj must match models[0].hardware.mmproj when both are set"
    );
}

#[test]
fn integrated_full_surface_fixture_parses_validates_and_tracks_docs() {
    let config: MeshConfig = toml::from_str(FULL_SURFACE_VALID_FIXTURE).unwrap();

    validate_config(&config).unwrap();
    assert_eq!(config.models.len(), 2);
    assert_eq!(
        config.owner_control.bind,
        Some("127.0.0.1:7447".parse().unwrap())
    );
    assert_eq!(
        config.owner_control.advertise_addr,
        Some("203.0.113.10:7447".parse().unwrap())
    );

    let defaults = config.defaults.as_ref().expect("defaults should parse");
    assert_eq!(
        defaults.model_fit.as_ref().and_then(|fit| fit.ctx_size),
        Some(8192)
    );
    assert_eq!(
        defaults
            .request_defaults
            .as_ref()
            .and_then(|request_defaults| request_defaults.temperature),
        Some(0.2)
    );

    let explicit = &config.models[0];
    assert_eq!(explicit.model, "Qwen/Qwen3-0.6B:Q4_K_M");
    assert_eq!(
        explicit.model_fit.as_ref().and_then(|fit| fit.ctx_size),
        Some(16384)
    );
    assert_eq!(
        explicit
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.stage_layer_start),
        Some(12)
    );
    assert_eq!(
        explicit
            .skippy
            .as_ref()
            .and_then(|skippy| skippy.prefill_chunk_schedule.as_deref()),
        Some("128,256,384")
    );

    let omitted = &config.models[1];
    assert_eq!(omitted.model, "ggml-org/gemma-3-270m-it-GGUF:Q8_0");
    assert!(
        omitted.model_fit.is_none(),
        "omitted per-model model_fit should stay absent"
    );
    assert!(
        omitted.request_defaults.is_none(),
        "omitted per-model request defaults should stay absent"
    );

    let matrix = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/skippy/CONFIGURATION.md"
    ));
    let matrix_keys = documented_matrix_key_paths();
    assert!(
        matrix_keys.len() >= 100,
        "expected a substantial canonical key-path set, found {}",
        matrix_keys.len()
    );
    for key in [
        "model_fit.ctx_size",
        "model_fit.prefix_cache.max_entries",
        "hardware.stage_layer_start",
        "hardware.stage_layer_end",
        "skippy.prefill_chunk_schedule",
        "speculative.draft_gpu_layers",
        "request_defaults.reasoning_budget",
        "multimodal.mmproj",
        "advanced.server.alias",
    ] {
        assert!(matrix.contains(key), "missing matrix doc entry {key}");
    }

    let docs_readme = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/README.md"));
    let usage = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/USAGE.md"));
    let cli = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/CLI.md"));
    assert!(docs_readme.contains("[skippy/CONFIGURATION.md](skippy/CONFIGURATION.md)"));
    assert!(usage.contains("request payload values still win"));
    assert!(cli.contains("Request defaults only fill absent or null request fields"));
    assert!(cli.contains("Staged-only controls stay staged-only."));
}

#[test]
fn integrated_invalid_fixture_reports_batch_then_pinned_device_paths() {
    let invalid: MeshConfig = toml::from_str(FULL_SURFACE_INVALID_FIXTURE).unwrap();
    let batch_error = validate_config(&invalid).unwrap_err().to_string();
    assert_eq!(
        batch_error,
        "models[0].model_fit.batch must be between 1 and 10000000, got 0"
    );

    let repaired_batch = FULL_SURFACE_INVALID_FIXTURE.replace("batch = 0", "batch = 64");
    let repaired_batch = repaired_batch.replace("[defaults.hardware]\ndevice = \"CUDA0\"\n\n", "");
    let repaired: MeshConfig = toml::from_str(&repaired_batch).unwrap();
    let pinned_error = validate_config(&repaired).unwrap_err().to_string();
    assert_eq!(
        pinned_error,
        "models[0].hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
    );
}
