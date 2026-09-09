use super::test_support::*;
use super::*;
use crate::plugin::MeshConfig;
use serde_json::Value;
use skippy_protocol::LoadMode;

#[test]
fn tensor_mode_reaches_serialized_stage_config() {
    let mesh_config: MeshConfig = toml::from_str("[defaults.hardware]\nsplit_mode = \"tensor\"\n")
        .expect("hardware config should parse");
    let model = temp_model_file();
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("qwen config should resolve");
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let value = serde_json::to_value(stage).expect("stage config serializes");

    assert_eq!(value["split_mode"], Value::String("tensor".into()));
}

#[test]
fn resolver_rejects_unsupported_hardware_controls_that_cannot_reach_launch() {
    let mesh_config = parse_config(
        r#"
[defaults.hardware]
placement = "auto"
"#,
    );
    let model_file = temp_model_file();

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("defaults.hardware.placement"));
}

#[test]
fn effective_safety_margin_bytes_mirrors_the_fit_margin_rounding() {
    use crate::plugin::{HardwareConfig, ModelConfigDefaults};

    // Nothing configured: the built-in 2 GiB default the fit already applies.
    assert_eq!(effective_safety_margin_bytes(None), 2 * 1024 * 1024 * 1024);

    let half_gib = ModelConfigDefaults {
        hardware: Some(HardwareConfig {
            safety_margin_gb: Some(0.5),
            ..HardwareConfig::default()
        }),
        ..ModelConfigDefaults::default()
    };
    assert_eq!(
        effective_safety_margin_bytes(Some(&half_gib)),
        512 * 1024 * 1024
    );

    let negative = ModelConfigDefaults {
        hardware: Some(HardwareConfig {
            safety_margin_gb: Some(-1.0),
            ..HardwareConfig::default()
        }),
        ..ModelConfigDefaults::default()
    };
    assert_eq!(effective_safety_margin_bytes(Some(&negative)), 0);
}

#[test]
fn effective_safety_margin_bytes_saturates_on_absurd_margins() {
    use crate::plugin::{HardwareConfig, ModelConfigDefaults};

    let absurd = ModelConfigDefaults {
        hardware: Some(HardwareConfig {
            safety_margin_gb: Some(f64::MAX),
            ..HardwareConfig::default()
        }),
        ..ModelConfigDefaults::default()
    };
    assert_eq!(effective_safety_margin_bytes(Some(&absurd)), u64::MAX);
}
