mod authoring;
mod diagnostic;
mod hardware_validation;
mod model;
mod model_validation;
mod plugin_validation;
mod store;
mod validate;
mod validation_support;
#[cfg(test)]
mod website_docs_parity;
mod wiring_status;
mod wiring_validation;

#[cfg(test)]
mod validate_schema_contract;

pub use authoring::{
    ConfigEditor, ConfigSchemaBuilder, ConfigSettingSchemaBuilder, LocalServingNodeConfig,
    ModelConfigEditor, ModelDefaultsEditor, PluginConfigEditor, built_in_config_schema,
};
pub use model::*;
pub use plugin_validation::control_behavior::{
    PluginConditionOperator, PluginConditionValue, PluginConditionalDisable, PluginConflictRule,
    PluginControlAvailability, PluginControlAvailabilitySource, PluginControlBehavior,
    PluginControlCondition, PluginDisabledWritePolicy, PluginNumericControl, PluginOptionsSource,
    PluginTextFormat,
};
pub use plugin_validation::{
    PluginConfigSchema, PluginObjectPropertySchema, PluginSchemaAvailability,
    PluginSettingConstraint, PluginSettingSchema, PluginValueKind, PluginValueSchema,
    SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
};
pub use store::{ConfigStore, config_path, config_to_toml, load_config, parse_config_toml};
pub use validate::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, alias_diagnostic, built_in_support_diagnostic,
    canonical_builtin_diagnostic_path, invalid_value_diagnostic, legacy_validation_error_text,
    rejected_field_diagnostic, unsupported_field_diagnostic, validate_config,
    validate_config_diagnostics, validate_config_diagnostics_with_plugin_schemas,
    validate_config_with_plugin_schemas,
};
pub use wiring_status::{
    WIRING_MANIFEST, WiringBehavior, WiringEntry, WiringStatus, wiring_entry_for_path,
};
pub use wiring_validation::wiring_manifest_diagnostics;

#[cfg(test)]
mod tests {
    use super::{
        ConfigStore, GpuAssignment, LocalServingNodeConfig, MeshConfig, SpeculativeConfig,
        built_in_config_schema, canonicalize_built_in_config_identifier, parse_config_toml,
        validate_config,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn config_store_loads_missing_file_as_default() {
        let temp_dir = TempDir::new().unwrap();
        let store = ConfigStore::open(temp_dir.path().join("config.toml"));

        let config = store.load().unwrap();

        assert!(config.models.is_empty());
    }

    #[test]
    fn speculative_config_precedence_keeps_lower_layer_fields() {
        let defaults = SpeculativeConfig {
            strategy: Some("mtp-cache".to_string()),
            verify_window_pipeline_depth: Some(2),
            ..Default::default()
        };
        let model = SpeculativeConfig {
            ngram_max_proposal_tokens: Some(6),
            ..Default::default()
        };
        let overrides = SpeculativeConfig {
            strategy: Some("mtp".to_string()),
            verify_window_pipeline_depth: Some(3),
            ..Default::default()
        };

        let resolved =
            SpeculativeConfig::with_precedence(Some(&overrides), Some(&model), Some(&defaults));

        assert_eq!(resolved.strategy.as_deref(), Some("mtp"));
        assert_eq!(resolved.ngram_max_proposal_tokens, Some(6));
        assert_eq!(resolved.verify_window_pipeline_depth, Some(3));
    }

    #[test]
    fn plugin_startup_config_round_trips_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 75
init_timeout_secs = 90
optional = true
lazy_start = true
"#,
        )
        .expect("plugin startup config should parse");

        let startup = &config.plugins[0].startup;
        assert_eq!(startup.connect_timeout_secs, Some(75));
        assert_eq!(startup.init_timeout_secs, Some(90));
        assert!(startup.optional);
        assert!(startup.lazy_start);
        validate_config(&config).expect("positive startup timeouts should validate");
    }

    #[test]
    fn plugin_startup_config_rejects_zero_timeouts() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 0
"#,
        )
        .expect("plugin startup config should parse before validation");

        let err = validate_config(&config).expect_err("zero connect timeout must be rejected");

        assert!(
            err.to_string()
                .contains("plugin[0].startup.connect_timeout_secs must be at least 1"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn native_runtime_override_accepts_mesh_version_with_optional_abi_and_selection() {
        let config = parse_config_toml(
            r#"
[runtime.native_runtime]
mesh_version = "0.68.0"
skippy_abi = "0.1.25"
selection = "exact:meshllm-native-runtime-linux-x86_64-cuda12"
"#,
        )
        .expect("native runtime selector should parse");

        assert_eq!(
            config.runtime.native_runtime.mesh_version.as_deref(),
            Some("0.68.0")
        );
        assert_eq!(
            config.runtime.native_runtime.skippy_abi.as_deref(),
            Some("0.1.25")
        );
        assert_eq!(
            config.runtime.native_runtime.selection.as_deref(),
            Some("exact:meshllm-native-runtime-linux-x86_64-cuda12")
        );

        parse_config_toml(
            r#"
[runtime.native_runtime]
mesh_version = "0.68.0"
"#,
        )
        .expect("mesh-version-only native runtime selector should parse");

        let err = parse_config_toml(
            r#"
[runtime.native_runtime]
selection = "cuda12"
"#,
        )
        .expect_err("partial native runtime selector should fail validation");

        assert!(
            err.to_string().contains(
                "runtime.native_runtime override must set mesh_version when skippy_abi or selection is set"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn config_store_add_model_preserves_existing_fields() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
version = 1

[defaults.model_fit]
ctx_size = 8192

[[models]]
model = "Qwen3-4B-Q4_K_M"
ctx_size = 4096
"#,
        )
        .unwrap();
        let store = ConfigStore::open(&path);

        let models = store.add_model_ref("  org/model-GGUF:Q5_K_M  ").unwrap();

        assert_eq!(
            models,
            vec![
                "Qwen3-4B-Q4_K_M".to_string(),
                "org/model-GGUF:Q5_K_M".to_string()
            ]
        );
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[defaults.model_fit]"));
        assert!(raw.contains("ctx_size = 4096"));
        assert_eq!(raw.matches("org/model-GGUF:Q5_K_M").count(), 1);
    }

    #[test]
    fn config_store_save_validates_before_writing() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);
        let config = MeshConfig {
            version: Some(2),
            ..MeshConfig::default()
        };

        let err = store.save(&config).unwrap_err().to_string();

        assert!(err.contains("unsupported config version"));
        assert!(!path.exists());
    }

    #[test]
    fn config_store_update_writes_local_serving_node_without_callers_writing_toml() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);

        let config = store
            .update(|config| {
                config.configure_local_serving_node(LocalServingNodeConfig {
                    model: "Qwen/Qwen3-8B-GGUF:Q4_K_M".into(),
                    device: Some("metal:0".into()),
                    context_size: Some(8192),
                    parallel: Some(2),
                    owner_control_bind: Some("127.0.0.1:0".parse().unwrap()),
                    gpu_assignment: Some(GpuAssignment::Pinned),
                    ..LocalServingNodeConfig::default()
                })?;
                let derived_profile = {
                    let entry = config
                        .config()
                        .models
                        .iter()
                        .find(|m| m.model == "Qwen/Qwen3-8B-GGUF:Q4_K_M")
                        .expect("model entry exists after configure_local_serving_node");
                    entry.derived_profile()
                };
                config
                    .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M", derived_profile)?
                    .max_tokens(1024)
                    .temperature(0.2);
                Ok(())
            })
            .unwrap();

        assert_eq!(config.models.len(), 1);
        let raw = fs::read_to_string(path).unwrap();
        assert!(!raw.contains("model_runtime"));
        assert!(raw.contains("ctx_size = 8192"));
        assert!(raw.contains("temperature = 0.2"));
    }

    #[test]
    fn config_store_update_rejects_local_serving_runtime_without_writing_it() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);

        let error = store
            .update(|config| {
                config.configure_local_serving_node(LocalServingNodeConfig {
                    model: "Qwen/Qwen3-8B-GGUF:Q4_K_M".into(),
                    runtime: Some(mesh_llm_types::runtime::ModelRuntimeKind::Metal),
                    ..LocalServingNodeConfig::default()
                })?;
                Ok(())
            })
            .expect_err("unsupported runtime must be rejected at the authoring boundary");

        assert!(error.to_string().contains("runtime is unsupported"));
        assert!(!path.exists(), "rejected updates must not write a config");
    }

    #[test]
    fn config_editor_updates_plugins_without_callers_writing_toml() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);

        let config = store
            .update(|config| {
                config.enable_builtin_plugin("telemetry")?;
                config
                    .upsert_plugin("endpoint-plugin")?
                    .enabled(true)
                    .url("http://localhost:8000/v1");
                config.upsert_external_plugin("custom-tool", "mesh-tool", ["--serve"])?;
                Ok(())
            })
            .unwrap();

        assert_eq!(config.plugins.len(), 3);
        assert_eq!(
            config
                .plugins
                .iter()
                .find(|plugin| plugin.name == "endpoint-plugin")
                .and_then(|plugin| plugin.url.as_deref()),
            Some("http://localhost:8000/v1")
        );
        assert!(fs::read_to_string(path).unwrap().contains("[[plugin]]"));
    }

    #[test]
    fn parse_config_toml_rejects_unknown_runtime_kind() {
        let err = parse_config_toml(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
model_runtime = "bogus"
"#,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("unknown variant"));
    }

    #[test]
    fn parse_config_toml_rejects_mixed_case_unsupported_runtime_kind() {
        let error = parse_config_toml(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
model_runtime = "Metal"
"#,
        )
        .expect_err("model_runtime has no runtime consumer");

        assert!(
            error
                .to_string()
                .contains("hardware.model_runtime is not supported")
        );
    }

    #[test]
    fn runtime_model_target_reconciliation_deserializes_from_toml() {
        let config = parse_config_toml(
            r#"
version = 1

[runtime]
debug = true
listen_all = true
reconcile_model_targets = true
reconcile_model_target_demand_upgrades = true
model_target_demand_upgrade_min_requests = 4
model_target_demand_upgrade_max_age_secs = 900
"#,
        )
        .unwrap();

        assert!(config.runtime.debug);
        assert!(config.runtime.listen_all);
        assert!(config.runtime.reconcile_model_targets);
        assert!(config.runtime.reconcile_model_target_demand_upgrades);
        assert_eq!(config.runtime.model_target_demand_upgrade_min_requests, 4);
        assert_eq!(config.runtime.model_target_demand_upgrade_max_age_secs, 900);
    }

    #[test]
    fn nested_hardware_device_does_not_serialize_as_legacy_gpu_id() {
        let config = parse_config_toml(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
device = "cuda:0"
"#,
        )
        .unwrap();

        let toml = super::config_to_toml(&config).unwrap();

        assert!(toml.contains("device = \"cuda:0\""));
        assert!(!toml.contains("gpu_id"));
        parse_config_toml(&toml).unwrap();
    }

    #[test]
    fn explicit_legacy_gpu_id_still_serializes_for_legacy_round_trip() {
        let config = parse_config_toml(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
"#,
        )
        .unwrap();

        let toml = super::config_to_toml(&config).unwrap();

        assert!(toml.contains("gpu_id = \"pci:0000:65:00.0\""));
        parse_config_toml(&toml).unwrap();
    }

    #[test]
    fn built_in_schema_exhaustiveness() {
        let schema = built_in_config_schema();
        let canonical_paths: BTreeSet<_> = schema
            .settings
            .iter()
            .map(|setting| setting.path.render())
            .collect();
        assert_eq!(
            canonical_paths.len(),
            schema.settings.len(),
            "duplicate canonical paths in built-in schema"
        );

        assert_eq!(
            schema.settings.len(),
            canonical_public_field_count(),
            "built-in schema count drifted from model-owned config leaf inventory"
        );

        for required in [
            "version",
            "gpu.assignment",
            "owner_control.bind",
            "runtime.debug",
            "runtime.listen_all",
            "telemetry.prompt_shape_metrics",
            "defaults.model_fit.ctx_size",
            "defaults.hardware.rpc_backend",
            "models.<model-ref>.hardware.device",
            "models.<model-ref>.throughput.sleep_idle_seconds",
            "models.<model-ref>.request_defaults.json_schema",
            "plugin.<plugin-name>.startup.connect_timeout_secs",
        ] {
            assert!(
                canonical_paths.contains(required),
                "missing built-in schema descriptor for {required}"
            );
        }
    }

    #[test]
    fn canonical_path_aliases() {
        let cases = [
            ("models[0].gpu_id", "models.<model-ref>.hardware.device"),
            (
                "models[0].ctx_size",
                "models.<model-ref>.model_fit.ctx_size",
            ),
            (
                "models[0].parallel",
                "models.<model-ref>.throughput.parallel",
            ),
            ("models[0].mmproj", "models.<model-ref>.multimodal.mmproj"),
            ("defaults.gpu_id", "defaults.hardware.device"),
            ("defaults.ctx_size", "defaults.model_fit.ctx_size"),
            ("defaults.parallel", "defaults.throughput.parallel"),
            ("defaults.mmproj", "defaults.multimodal.mmproj"),
            (
                "plugin[0].startup.connect_timeout_secs",
                "plugin.<plugin-name>.startup.connect_timeout_secs",
            ),
        ];

        for (alias, canonical) in cases {
            assert_eq!(
                canonicalize_built_in_config_identifier(alias).as_deref(),
                Some(canonical),
                "alias `{alias}` should resolve to canonical `{canonical}`"
            );
        }
    }

    #[test]
    fn authoring_mutators_remain_schema_classified() {
        let canonical_paths: BTreeSet<_> = built_in_config_schema()
            .settings
            .into_iter()
            .map(|setting| setting.path.render())
            .collect();
        let tracked = BTreeMap::from([
            ("ConfigEditor::set_version", vec!["version"]),
            ("ConfigEditor::set_gpu_assignment", vec!["gpu.assignment"]),
            ("ConfigEditor::set_gpu_parallel", vec!["gpu.parallel"]),
            (
                "ConfigEditor::set_owner_control_bind",
                vec!["owner_control.bind"],
            ),
            (
                "ConfigEditor::set_owner_control_advertise_addr",
                vec!["owner_control.advertise_addr"],
            ),
            (
                "ConfigEditor::set_default_runtime",
                vec!["defaults.hardware.model_runtime"],
            ),
            (
                "ConfigEditor::clear_default_runtime",
                vec!["defaults.hardware.model_runtime"],
            ),
            (
                "ConfigEditor::set_default_device",
                vec!["defaults.hardware.device"],
            ),
            (
                "ConfigEditor::clear_default_device",
                vec!["defaults.hardware.device"],
            ),
            (
                "ConfigEditor::set_default_context_size",
                vec!["defaults.model_fit.ctx_size"],
            ),
            (
                "ConfigEditor::configure_local_serving_node",
                vec![
                    "version",
                    "gpu.assignment",
                    "owner_control.bind",
                    "owner_control.advertise_addr",
                    "models.<model-ref>.hardware.device",
                    "models.<model-ref>.model_fit.ctx_size",
                    "models.<model-ref>.throughput.parallel",
                    "models.<model-ref>.multimodal.mmproj",
                ],
            ),
            (
                "ConfigEditor::enable_builtin_plugin",
                vec!["plugin.<plugin-name>.enabled"],
            ),
            (
                "ConfigEditor::disable_plugin",
                vec!["plugin.<plugin-name>.enabled"],
            ),
            (
                "ConfigEditor::upsert_external_plugin",
                vec![
                    "plugin.<plugin-name>.enabled",
                    "plugin.<plugin-name>.command",
                    "plugin.<plugin-name>.args",
                ],
            ),
            (
                "ModelDefaultsEditor::runtime",
                vec!["defaults.hardware.model_runtime"],
            ),
            (
                "ModelDefaultsEditor::clear_runtime",
                vec!["defaults.hardware.model_runtime"],
            ),
            (
                "ModelDefaultsEditor::device",
                vec!["defaults.hardware.device"],
            ),
            (
                "ModelDefaultsEditor::clear_device",
                vec!["defaults.hardware.device"],
            ),
            (
                "ModelDefaultsEditor::context_size",
                vec!["defaults.model_fit.ctx_size"],
            ),
            (
                "ModelDefaultsEditor::parallel",
                vec!["defaults.throughput.parallel"],
            ),
            (
                "ModelConfigEditor::runtime",
                vec!["models.<model-ref>.hardware.model_runtime"],
            ),
            (
                "ModelConfigEditor::clear_runtime",
                vec!["models.<model-ref>.hardware.model_runtime"],
            ),
            (
                "ModelConfigEditor::device",
                vec!["models.<model-ref>.hardware.device"],
            ),
            (
                "ModelConfigEditor::clear_device",
                vec!["models.<model-ref>.hardware.device"],
            ),
            (
                "ModelConfigEditor::context_size",
                vec!["models.<model-ref>.model_fit.ctx_size"],
            ),
            (
                "ModelConfigEditor::parallel",
                vec!["models.<model-ref>.throughput.parallel"],
            ),
            (
                "ModelConfigEditor::cache_types",
                vec![
                    "models.<model-ref>.model_fit.cache_type_k",
                    "models.<model-ref>.model_fit.cache_type_v",
                ],
            ),
            (
                "ModelConfigEditor::max_tokens",
                vec!["models.<model-ref>.request_defaults.max_tokens"],
            ),
            (
                "ModelConfigEditor::temperature",
                vec!["models.<model-ref>.request_defaults.temperature"],
            ),
            (
                "ModelConfigEditor::mmproj",
                vec!["models.<model-ref>.multimodal.mmproj"],
            ),
            (
                "PluginConfigEditor::enabled",
                vec!["plugin.<plugin-name>.enabled"],
            ),
            (
                "PluginConfigEditor::web_ui_enabled",
                vec!["plugin.<plugin-name>.web_ui_enabled"],
            ),
            (
                "PluginConfigEditor::command",
                vec!["plugin.<plugin-name>.command"],
            ),
            (
                "PluginConfigEditor::args",
                vec!["plugin.<plugin-name>.args"],
            ),
            ("PluginConfigEditor::url", vec!["plugin.<plugin-name>.url"]),
            (
                "PluginConfigEditor::connect_timeout_secs",
                vec!["plugin.<plugin-name>.startup.connect_timeout_secs"],
            ),
            (
                "PluginConfigEditor::init_timeout_secs",
                vec!["plugin.<plugin-name>.startup.init_timeout_secs"],
            ),
            (
                "PluginConfigEditor::optional",
                vec!["plugin.<plugin-name>.startup.optional"],
            ),
            (
                "PluginConfigEditor::lazy_start",
                vec!["plugin.<plugin-name>.startup.lazy_start"],
            ),
        ]);
        let ignored = BTreeSet::from([
            "ConfigEditor::new",
            "ConfigEditor::into_config",
            "ConfigEditor::config",
            "ConfigEditor::defaults",
            "ConfigEditor::upsert_model",
            "ConfigEditor::remove_model",
            "ConfigEditor::model_refs",
            "ConfigEditor::upsert_plugin",
            "ModelConfigEditor::model_ref",
            "ModelConfigEditor::derived_profile",
            "PluginConfigEditor::name",
        ]);
        let actual = authoring_public_methods();
        let expected = tracked
            .keys()
            .map(|name| (*name).to_string())
            .chain(ignored.iter().map(|name| (*name).to_string()))
            .collect::<BTreeSet<_>>();

        assert_eq!(
            actual, expected,
            "authoring public method inventory drifted; classify new mutators against the schema registry"
        );

        for (method, paths) in tracked {
            for path in paths {
                assert!(
                    canonical_paths.contains(path),
                    "authoring method {method} references unclassified canonical path {path}"
                );
            }
        }
    }

    fn canonical_public_field_count() -> usize {
        let source_model = include_str!("model.rs");
        let source_multimodal = include_str!("model/multimodal.rs");
        let source_runtime = include_str!("model/runtime.rs");
        let sources = [source_model, source_multimodal, source_runtime];
        let occurrences = [
            ("MeshConfig", 1usize),
            ("OwnerControlConfig", 1),
            ("GpuConfig", 1),
            ("RuntimeConfig", 1),
            ("NativeRuntimeConfig", 1),
            ("MeshRequirementsConfig", 1),
            ("ModelConfigEntry", 1),
            ("ModelFitConfig", 2),
            ("PrefixCacheConfig", 2),
            ("HardwareConfig", 2),
            ("ThroughputConfig", 2),
            ("ModelTopologyConfig", 2),
            ("SkippyConfig", 2),
            ("SpeculativeConfig", 2),
            ("RequestDefaultsConfig", 2),
            ("MultimodalConfig", 2),
            ("AdvancedServerConfig", 2),
            ("TelemetryConfig", 1),
            ("TelemetryMetricsConfig", 1),
            ("AuditConfig", 1),
            ("PluginConfigEntry", 1),
            ("PluginStartupConfig", 1),
            ("RuntimeActivityConfig", 1),
            ("LoggingConfig", 1),
            ("LoggingArtifactConfig", 1),
            ("LoggingWebhookConfig", 1),
        ];
        let nested = [
            "GpuConfig",
            "MeshRequirementsConfig",
            "OwnerControlConfig",
            "RuntimeConfig",
            "NativeRuntimeConfig",
            "TelemetryConfig",
            "TelemetryMetricsConfig",
            "AuditConfig",
            "ModelConfigDefaults",
            "ModelConfigEntry",
            "ModelFitConfig",
            "PrefixCacheConfig",
            "HardwareConfig",
            "ThroughputConfig",
            "ModelTopologyConfig",
            "SkippyConfig",
            "SpeculativeConfig",
            "RequestDefaultsConfig",
            "MultimodalConfig",
            "AdvancedConfig",
            "AdvancedServerConfig",
            "PluginConfigEntry",
            "PluginStartupConfig",
            "RuntimeActivityConfig",
            "LoggingConfig",
            "LoggingArtifactConfig",
            "LoggingWebhookConfig",
        ];
        let ignored = [
            "extra",
            "gpu_id_from_legacy_shim",
            "models",
            "plugins",
            "settings",
        ];

        let mut total = 0usize;
        for (name, multiplier) in occurrences.iter() {
            let leafs = extract_struct_fields(&sources, name)
                .into_iter()
                .filter(|(field, ty)| {
                    !ignored.contains(&field.as_str())
                        && !is_legacy_flat_model_field(name, field)
                        && !nested
                            .iter()
                            .any(|nested_ty| contains_nested_type(ty, nested_ty))
                })
                .count();
            let contribution = leafs * multiplier;
            total += contribution;
        }
        total
    }

    fn extract_struct_fields(sources: &[&str], struct_name: &str) -> Vec<(String, String)> {
        let marker = format!("pub struct {struct_name} {{");
        let source = sources
            .iter()
            .find(|s| s.contains(&marker))
            .unwrap_or_else(|| panic!("struct {} not found in config model sources", struct_name));
        let start = source
            .find(&marker)
            .expect("marker was just confirmed present");
        let body = &source[start + marker.len()..];
        let end = body.find("\n}").expect("struct body terminator");

        body[..end]
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("pub ")
                    .and_then(|line| line.split_once(':'))
                    .map(|(field, ty)| {
                        (
                            field.trim().to_string(),
                            ty.trim().trim_end_matches(',').to_string(),
                        )
                    })
            })
            .collect()
    }

    fn contains_nested_type(type_name: &str, nested: &str) -> bool {
        if type_name == nested {
            return true;
        }
        let option = format!("Option<{nested}>");
        let vec = format!("Vec<{nested}>");
        if type_name == option || type_name == vec {
            return true;
        }
        if type_name.ends_with(&format!("::{nested}")) {
            return true;
        }
        if (type_name.starts_with("Option<") || type_name.starts_with("Vec<"))
            && type_name.ends_with(&format!("::{nested}>"))
        {
            return true;
        }
        false
    }

    fn authoring_public_methods() -> BTreeSet<String> {
        let source = include_str!("authoring.rs");
        let mut methods = BTreeSet::new();

        for (impl_name, marker) in [
            ("ConfigEditor", "impl ConfigEditor {"),
            ("ModelDefaultsEditor", "impl ModelDefaultsEditor<'_> {"),
            ("ModelConfigEditor", "impl ModelConfigEditor<'_> {"),
            ("PluginConfigEditor", "impl PluginConfigEditor<'_> {"),
        ] {
            let body = impl_body(source, marker);
            for line in body.lines() {
                let line = line.trim_start();
                if let Some(signature) = line.strip_prefix("pub fn ") {
                    let name = signature
                        .split_once('(')
                        .map(|(name, _)| name)
                        .expect("public function signature should contain '('");
                    methods.insert(format!("{impl_name}::{name}"));
                }
            }
        }

        methods
    }

    fn impl_body<'a>(source: &'a str, marker: &str) -> &'a str {
        let start = source
            .find(marker)
            .unwrap_or_else(|| panic!("impl marker `{marker}` not found in authoring.rs"));
        let body_start = start + marker.len();
        let mut depth = 1usize;

        for (offset, ch) in source[body_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[body_start..body_start + offset];
                    }
                }
                _ => {}
            }
        }

        panic!("impl marker `{marker}` did not terminate");
    }

    fn is_legacy_flat_model_field(struct_name: &str, field: &str) -> bool {
        struct_name == "ModelConfigEntry"
            && matches!(
                field,
                "mmproj"
                    | "ctx_size"
                    | "gpu_id"
                    | "parallel"
                    | "cache_type_k"
                    | "cache_type_v"
                    | "batch"
                    | "ubatch"
                    | "flash_attention"
            )
    }
}
