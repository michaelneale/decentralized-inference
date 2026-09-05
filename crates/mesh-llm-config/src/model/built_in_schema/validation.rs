use super::*;

#[test]
fn built_in_schema_preserves_union_typed_fields() {
    for path in [
        "models.<model-ref>.model_fit.kv_offload",
        "models.<model-ref>.model_fit.kv_unified",
        "models.<model-ref>.model_fit.prompt_cache",
        "models.<model-ref>.model_fit.context_shift",
        "models.<model-ref>.hardware.cpu_moe",
        "models.<model-ref>.hardware.fit_context",
        "models.<model-ref>.hardware.mmproj_offload",
        "models.<model-ref>.hardware.mmap",
        "models.<model-ref>.hardware.warmup",
        "models.<model-ref>.throughput.continuous_batching",
        "models.<model-ref>.speculative.spec_default",
        "models.<model-ref>.multimodal.mmproj_offload",
    ] {
        assert_eq!(schema_value(path), bool_or_auto_schema());
    }

    assert_eq!(
        schema_value("models.<model-ref>.hardware.gpu_layers"),
        integer_or_auto_schema()
    );
    assert_eq!(
        schema_value("models.<model-ref>.hardware.tensor_split"),
        tensor_split_schema()
    );
    assert_eq!(
        schema_value("models.<model-ref>.throughput.priority"),
        integer_or_string_schema()
    );
    assert_eq!(
        schema_value("models.<model-ref>.throughput.poll"),
        bool_or_string_enum(["auto", "busy", "sleep"])
    );
    assert_eq!(
        schema_value("models.<model-ref>.throughput.cpu_affinity"),
        string_or_list_schema()
    );
    assert_eq!(
        schema_value("models.<model-ref>.request_defaults.stop"),
        string_or_list_schema()
    );
    assert_eq!(
        schema_value("models.<model-ref>.request_defaults.mirostat_mode"),
        integer_or_string_enum(["disabled", "1", "2"])
    );
    assert_eq!(
        schema_value("models.<model-ref>.request_defaults.reasoning_enabled"),
        bool_or_string_enum(["auto", "off", "on"])
    );
    assert_eq!(
        schema_value("models.<model-ref>.request_defaults.reasoning_budget"),
        integer_or_string_enum(["auto", "low", "medium", "high"])
    );
}

#[test]
fn built_in_schema_marks_curated_defaults_user_visible() {
    for path in [
        "defaults.throughput.threads",
        "defaults.throughput.parallel",
        "defaults.model_fit.kv_cache_policy",
        "defaults.request_defaults.temperature",
        "defaults.skippy.binary_stage_transport",
        "defaults.multimodal.mmproj_offload",
    ] {
        assert_eq!(
            schema_setting(path).visibility,
            ConfigVisibility::User,
            "{path}"
        );
    }

    assert_eq!(
        schema_setting("defaults.model_fit.prompt_cache").visibility,
        ConfigVisibility::Advanced
    );
    assert_eq!(
        schema_setting("defaults.hardware.model_runtime").visibility,
        ConfigVisibility::Hidden
    );
    assert_eq!(
        schema_setting("defaults.advanced.server.alias").visibility,
        ConfigVisibility::Advanced
    );
}

#[test]
fn built_in_schema_uses_explicit_path_and_url_value_kinds() {
    assert_eq!(schema_value("telemetry.endpoint"), ConfigValueSchema::Url);
    assert_eq!(
        schema_value("telemetry.metrics.endpoint"),
        ConfigValueSchema::Url
    );
    assert_eq!(
        schema_value("plugin.<plugin-name>.url"),
        ConfigValueSchema::Url
    );
    assert_eq!(
        schema_value("defaults.hardware.model_path"),
        ConfigValueSchema::Path
    );
    assert_eq!(
        schema_value("defaults.hardware.mmproj"),
        ConfigValueSchema::Path
    );
    assert_eq!(
        schema_value("defaults.multimodal.mmproj"),
        ConfigValueSchema::Path
    );
    assert_eq!(
        schema_value("defaults.multimodal.mmproj_url"),
        ConfigValueSchema::Url
    );
    assert_eq!(
        schema_value("defaults.speculative.draft_model"),
        ConfigValueSchema::Path
    );
}

#[test]
fn startup_runtime_settings_require_process_restart() {
    for path in [
        "runtime.debug",
        "runtime.listen_all",
        "runtime.lifecycle_log_parser",
    ] {
        let setting = schema_setting(path);

        assert_eq!(
            setting.control_surfaces,
            vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api],
            "{path}"
        );
        assert_eq!(setting.apply_mode, ConfigApplyMode::StaticOnLoad, "{path}");
        assert_eq!(
            setting.restart_scope,
            ConfigRestartScope::ProcessRestart,
            "{path}"
        );
    }
}

#[test]
fn built_in_schema_exports_model_fit_numeric_controls_and_relative_bounds() {
    let defaults_batch = schema_setting("defaults.model_fit.batch");
    let defaults_ubatch = schema_setting("defaults.model_fit.ubatch");
    let model_ubatch = schema_setting("models.<model-ref>.model_fit.ubatch");

    assert_eq!(numeric_control(&defaults_batch).min, Some(1.0));
    assert_eq!(numeric_control(&defaults_batch).step, Some(1.0));
    assert_eq!(
        numeric_control(&defaults_batch).unit.as_deref(),
        Some("tokens")
    );

    assert_eq!(
        numeric_control(&defaults_ubatch),
        numeric_control(&model_ubatch)
    );
    assert_has_range_constraint(&defaults_ubatch, None, Some("defaults.model_fit.batch"));
    assert_has_range_constraint(
        &model_ubatch,
        None,
        Some("models.<model-ref>.model_fit.batch"),
    );
}

#[test]
fn built_in_schema_keeps_defaults_and_model_hardware_device_semantics_in_sync() {
    let defaults_device = schema_setting("defaults.hardware.device");
    let model_device = schema_setting("models.<model-ref>.hardware.device");

    assert_eq!(
        control_behavior(&defaults_device).options_source,
        Some(ConfigOptionsSource::RuntimeGpus)
    );
    assert_eq!(
        defaults_device.control_behavior,
        model_device.control_behavior
    );
    assert_eq!(
        control_behavior(&defaults_device).enable_when,
        vec![equals_condition("gpu.assignment", "pinned")]
    );
    assert_eq!(
        control_behavior(&defaults_device).disable_when,
        vec![dependency_disable(
            equals_condition("gpu.assignment", "auto"),
            "Set gpu.assignment = \"pinned\" to edit a concrete GPU device.",
        )]
    );
}

#[test]
fn built_in_schema_marks_rejected_hardware_escape_hatches_non_editable() {
    let setting = schema_setting("models.<model-ref>.hardware.rpc_backend");
    let behavior = control_behavior(&setting);

    assert_eq!(setting.support, ConfigSupportState::Rejected);
    assert_eq!(
        behavior.availability.as_ref().map(|value| value.enabled),
        Some(false)
    );
    assert_eq!(
        behavior.availability.as_ref().map(|value| value.source),
        Some(ConfigControlAvailabilitySource::Static)
    );
    assert_eq!(
        setting.default_disabled_write_policy(None),
        Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
    );
}

#[test]
fn built_in_schema_exports_throughput_and_skippy_t5_controls() {
    let threads = schema_setting("defaults.throughput.threads");
    let prefill_chunk_size = schema_setting("defaults.skippy.prefill_chunk_size");
    let prefill_chunk_schedule = schema_setting("defaults.skippy.prefill_chunk_schedule");

    assert_static_choices(
        "defaults.throughput.tuning_profile",
        &["throughput", "balanced", "saver"],
    );
    assert_eq!(numeric_control(&threads).min, Some(0.0));
    assert_eq!(numeric_control(&threads).step, Some(1.0));

    assert_static_choices(
        "defaults.skippy.prefill_chunking",
        &["auto", "fixed", "schedule", "adaptive-ramp"],
    );
    assert_eq!(numeric_control(&prefill_chunk_size).min, Some(1.0));
    assert_eq!(
        control_behavior(&prefill_chunk_size).enable_when,
        vec![equals_condition(
            "defaults.skippy.prefill_chunking",
            "fixed"
        )]
    );
    assert_eq!(
        control_behavior(&prefill_chunk_schedule).text_format,
        Some(ConfigTextFormat::CsvPositiveInts)
    );
    assert_eq!(
        control_behavior(&prefill_chunk_schedule).enable_when,
        vec![equals_condition(
            "defaults.skippy.prefill_chunking",
            "schedule"
        )]
    );
}

#[test]
fn built_in_schema_exports_speculative_and_request_default_t5_controls() {
    let draft_min = schema_setting("defaults.speculative.draft_min_tokens");
    let ngram_max = schema_setting("defaults.speculative.ngram_max");
    let mirostat_entropy = schema_setting("defaults.request_defaults.mirostat_entropy");

    assert_static_choices("defaults.speculative.mode", &["auto", "disabled", "draft"]);
    assert_static_choices(
        "defaults.speculative.draft_selection_policy",
        &["manual", "auto"],
    );
    assert_static_choices(
        "defaults.speculative.pairing_fault",
        &[
            "warn_disable",
            "fail-open",
            "fail-closed",
            "fail_open",
            "fail_closed",
        ],
    );
    assert_has_range_constraint(
        &draft_min,
        None,
        Some("defaults.speculative.draft_max_tokens"),
    );
    assert_has_range_constraint(&ngram_max, Some("defaults.speculative.ngram_min"), None);

    assert_static_choices(
        "defaults.request_defaults.reasoning_format",
        &["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
    );
    assert_eq!(
        control_behavior(&mirostat_entropy).enable_when,
        vec![in_condition(
            "defaults.request_defaults.mirostat_mode",
            &[
                ConfigConditionValue::Integer(1),
                ConfigConditionValue::Integer(2),
                ConfigConditionValue::String("1".to_string()),
                ConfigConditionValue::String("2".to_string()),
            ],
        )]
    );
    assert_eq!(
        control_behavior(&mirostat_entropy).disable_when,
        vec![dependency_disable(
            not_in_condition(
                "defaults.request_defaults.mirostat_mode",
                &[
                    ConfigConditionValue::Integer(1),
                    ConfigConditionValue::Integer(2),
                    ConfigConditionValue::String("1".to_string()),
                    ConfigConditionValue::String("2".to_string()),
                ],
            ),
            "defaults.request_defaults.mirostat_entropy requires defaults.request_defaults.mirostat_mode = 1 or 2",
        )]
    );
}

#[test]
fn built_in_schema_disables_duplicate_multimodal_projector_controls_with_preserve_policy() {
    let mmproj = schema_setting("defaults.hardware.mmproj");
    let offload = schema_setting("defaults.hardware.mmproj_offload");

    assert_eq!(
            control_behavior(&mmproj).availability,
            Some(ConfigControlAvailability {
                enabled: false,
                reason: Some(
                    "Edit defaults.multimodal.mmproj instead of the legacy hardware duplicate."
                        .to_string(),
                ),
                note: Some(
                    "Existing values are preserved on save unless you change defaults.multimodal.mmproj."
                        .to_string(),
                ),
                source: ConfigControlAvailabilitySource::Static,
            })
        );
    assert_eq!(
        control_behavior(&mmproj).write_policy,
        Some(ConfigDisabledWritePolicy::PreserveExisting)
    );
    assert_eq!(
        control_behavior(&offload)
            .availability
            .as_ref()
            .map(|value| value.enabled),
        Some(false)
    );
    assert_eq!(
        control_behavior(&offload).write_policy,
        Some(ConfigDisabledWritePolicy::PreserveExisting)
    );
}

#[test]
fn built_in_schema_exports_telemetry_owner_control_attestation_and_plugin_timeout_controls() {
    let telemetry_interval = schema_setting("telemetry.export_interval_secs");
    let advertise_addr = schema_setting("owner_control.advertise_addr");
    let signer_keys = schema_setting("mesh_requirements.release_signer_keys");
    let plugin_timeout = schema_setting("plugin.<plugin-name>.startup.connect_timeout_secs");

    assert_eq!(numeric_control(&telemetry_interval).min, Some(1.0));
    assert_eq!(
        numeric_control(&telemetry_interval).unit.as_deref(),
        Some("sec")
    );

    assert_eq!(
        control_behavior(&advertise_addr).enable_when,
        vec![present_condition("owner_control.bind")]
    );
    assert_eq!(
        control_behavior(&advertise_addr).disable_when,
        vec![dependency_disable(
            absent_condition("owner_control.bind"),
            "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening",
        )]
    );

    assert_eq!(
        control_behavior(&schema_setting("mesh_requirements.min_node_version")).text_format,
        Some(ConfigTextFormat::Semver)
    );
    assert_eq!(
        control_behavior(&signer_keys).text_format,
        Some(ConfigTextFormat::Ed25519Key)
    );
    assert_eq!(
        control_behavior(&signer_keys).enable_when,
        vec![equals_bool_condition(
            "mesh_requirements.require_release_attestation",
            true,
        )]
    );

    assert_eq!(numeric_control(&plugin_timeout).min, Some(1.0));
    assert_eq!(
        numeric_control(&plugin_timeout).unit.as_deref(),
        Some("sec")
    );
}

#[test]
fn built_in_schema_covers_t5_fallback_choices_or_keeps_open_text_intentional() {
    for (path, expected) in [
        (
            "defaults.throughput.tuning_profile",
            vec!["throughput", "balanced", "saver"],
        ),
        (
            "defaults.speculative.mode",
            vec!["auto", "disabled", "draft"],
        ),
        (
            "defaults.speculative.draft_selection_policy",
            vec!["manual", "auto"],
        ),
        (
            "defaults.speculative.pairing_fault",
            vec![
                "warn_disable",
                "fail-open",
                "fail-closed",
                "fail_open",
                "fail_closed",
            ],
        ),
        (
            "defaults.request_defaults.reasoning_format",
            vec!["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
        ),
        (
            "defaults.speculative.draft_cache_type_k",
            vec![
                "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
            ],
        ),
        (
            "defaults.speculative.draft_cache_type_v",
            vec![
                "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
            ],
        ),
    ] {
        assert_eq!(
            schema_enum_values(path),
            expected.into_iter().map(str::to_string).collect::<Vec<_>>(),
            "{path}"
        );
    }

    for path in [
        "defaults.throughput.numa",
        "defaults.skippy.binary_stage_transport",
    ] {
        assert!(schema_enum_values(path).is_empty(), "{path}");
        assert_ne!(
            schema_setting(path)
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.options_source),
            Some(ConfigOptionsSource::Static),
            "{path}"
        );
    }
}

fn schema_value(path: &str) -> ConfigValueSchema {
    schema_setting(path).value_schema
}

fn schema_setting(path: &str) -> ConfigSettingSchema {
    built_in_config_schema_descriptor(&schema_path(path)).expect("schema setting should exist")
}

fn control_behavior(setting: &ConfigSettingSchema) -> &ConfigControlBehavior {
    setting
        .control_behavior
        .as_ref()
        .expect("control behavior should be present")
}

fn numeric_control(setting: &ConfigSettingSchema) -> ConfigNumericControl {
    control_behavior(setting)
        .numeric
        .clone()
        .expect("numeric control should be present")
}

fn assert_has_range_constraint(
    setting: &ConfigSettingSchema,
    expected_min: Option<&str>,
    expected_max: Option<&str>,
) {
    assert!(
        setting.constraints.iter().any(|constraint| {
            matches!(
                constraint,
                ConfigConstraint::Range { min, max }
                    if min.as_deref() == expected_min && max.as_deref() == expected_max
            )
        }),
        "expected range constraint min={expected_min:?} max={expected_max:?} on {}",
        setting.path.render()
    );
}

fn equals_condition(path: &str, expected: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Equals,
        values: vec![ConfigConditionValue::String(expected.to_string())],
    }
}

fn equals_bool_condition(path: &str, expected: bool) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Equals,
        values: vec![ConfigConditionValue::Bool(expected)],
    }
}

fn in_condition(path: &str, values: &[ConfigConditionValue]) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::In,
        values: values.to_vec(),
    }
}

fn not_in_condition(path: &str, values: &[ConfigConditionValue]) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::NotIn,
        values: values.to_vec(),
    }
}

fn present_condition(path: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Present,
        values: Vec::new(),
    }
}

fn absent_condition(path: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Absent,
        values: Vec::new(),
    }
}

fn dependency_disable(condition: ConfigControlCondition, reason: &str) -> ConfigConditionalDisable {
    ConfigConditionalDisable {
        condition,
        reason: reason.to_string(),
        note: None,
        write_policy: ConfigDisabledWritePolicy::OmitWhenDisabled,
    }
}

fn assert_static_choices(path: &str, expected: &[&str]) {
    let setting = schema_setting(path);

    assert_eq!(
        control_behavior(&setting).options_source,
        Some(ConfigOptionsSource::Static),
        "{path}"
    );
    assert_eq!(
        schema_enum_values(path),
        expected
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>(),
        "{path}"
    );
}

fn schema_enum_values(path: &str) -> Vec<String> {
    enum_values(&schema_value(path))
}

fn enum_values(schema: &ConfigValueSchema) -> Vec<String> {
    match schema {
        ConfigValueSchema::Enum { values } => values.clone(),
        ConfigValueSchema::OneOf { variants } => variants.iter().flat_map(enum_values).collect(),
        _ => Vec::new(),
    }
}

#[test]
fn mesh_requirement_version_options_include_published_releases() {
    let versions = schema_enum_values("mesh_requirements.min_node_version");
    for version in ["0.72.2", "0.73.0", "0.73.1", "0.74.0", "0.75.0", "0.75.1"] {
        assert!(
            versions.iter().any(|candidate| candidate == version),
            "{version}"
        );
    }
}
