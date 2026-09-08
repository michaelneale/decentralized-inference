use super::*;
use crate::inference::skippy;

#[tokio::test]
async fn different_local_paths_advertise_the_same_logical_model_demand() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let bin_dir = tempfile::tempdir().expect("temporary runtime bin directory");
    let mut options = runtime_options_for_test(&["mesh-llm"]);
    options.bin_dir = Some(bin_dir.path().to_path_buf());
    let logical_model = "shared/local-gguf";
    let mut advertised = Vec::new();

    for filename in ["node-a.gguf", "node-b.gguf"] {
        let model_path = temp_dir.path().join(filename);
        std::fs::write(&model_path, b"gguf").expect("write model");
        let config = plugin::MeshConfig {
            models: vec![plugin::ModelConfigEntry {
                model: logical_model.into(),
                hardware: Some(plugin::HardwareConfig {
                    model_path: Some(model_path.to_string_lossy().into_owned()),
                    ..Default::default()
                }),
                skippy: Some(plugin::SkippyConfig {
                    source_policy: Some("local-required".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        let prepared = prepare_runtime_startup(
            &options,
            &config,
            Some(RuntimeSurface::Serve),
            mesh_llm_config::RuntimeMode::Serve,
        )
        .await
        .expect("strict local startup should prepare")
        .expect("serve startup remains active");
        assert_eq!(prepared.startup_specs[0].model_ref, model_path);
        assert_eq!(
            prepared.startup_specs[0].declared_ref.as_deref(),
            Some(logical_model)
        );
        advertised.push(prepared.requested_model_names);
    }

    assert_eq!(
        advertised,
        vec![
            vec![logical_model.to_string()],
            vec![logical_model.to_string()]
        ]
    );
}

#[tokio::test]
async fn local_required_source_policy_bypasses_remote_startup_resolution() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("local-required.gguf");
    super::write_identity_test_gguf(&model_path, 4096);
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "hf://definitely-not-a-real-org/strict-logical-model@missing".into(),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some(model_path.to_string_lossy().into_owned()),
                ..Default::default()
            }),
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("local-required".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    assert!(specs[0].local_source_required);

    let plans = resolve_startup_models(&specs, true)
        .await
        .expect("strict startup should resolve only the pinned local model");
    assert!(plans[0].local_source_required);
    assert_eq!(
        plans[0].resolved_path,
        model_path.canonicalize().expect("canonical model path")
    );
    assert_eq!(
        plans[0].declared_ref,
        "hf://definitely-not-a-real-org/strict-logical-model@missing"
    );
}

#[test]
fn exact_cli_model_applies_matched_local_required_policy_before_resolution() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "strict-configured-model"]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "strict-configured-model".into(),
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("local-required".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error = build_startup_model_specs(&options, &config)
        .expect_err("matched local-required policy must reject a remote-style model ref");

    assert!(
        error
            .to_string()
            .contains("requires an absolute local GGUF path")
    );
}

#[test]
fn model_fallback_source_policy_overrides_local_required_default() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("fallback.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        defaults: Some(plugin::ModelConfigDefaults {
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("local-required".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        models: vec![plugin::ModelConfigEntry {
            model: "fallback-model".into(),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some(model_path.to_string_lossy().into_owned()),
                ..Default::default()
            }),
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("fallback".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    assert!(!specs[0].local_source_required);
}

#[test]
fn model_inherits_local_required_source_policy_from_defaults_when_unset() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("inherited-local-required.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        defaults: Some(plugin::ModelConfigDefaults {
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("local-required".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        models: vec![plugin::ModelConfigEntry {
            model: "inherited-policy-model".into(),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some(model_path.to_string_lossy().into_owned()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    assert!(specs[0].local_source_required);
}

#[test]
fn pre_accept_policy_covers_bare_gguf_runtime_identity() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("bare-gguf.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("UTF-8 model path"),
    ]);
    let config = strict_local_source_default_config();
    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    let canonical = model_path.canonicalize().expect("canonical model path");
    let runtime_model_ref = models::model_ref_for_path(&canonical);

    register_pre_accept_local_source_policies(&config, &specs);

    assert!(skippy::local_source_required_for_model(
        &runtime_model_ref,
        Some("")
    ));
}

#[test]
fn pre_accept_policy_covers_absolute_model_runtime_identity() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("absolute-model.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        model_path.to_str().expect("UTF-8 model path"),
    ]);
    let config = strict_local_source_default_config();
    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    let canonical = model_path.canonicalize().expect("canonical model path");
    let runtime_model_ref = models::model_ref_for_path(&canonical);

    register_pre_accept_local_source_policies(&config, &specs);

    assert!(skippy::local_source_required_for_model(
        &runtime_model_ref,
        Some("")
    ));
}

fn strict_local_source_default_config() -> plugin::MeshConfig {
    plugin::MeshConfig {
        defaults: Some(plugin::ModelConfigDefaults {
            skippy: Some(plugin::SkippyConfig {
                source_policy: Some("local-required".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn pre_accept_policy_includes_config_models_suppressed_by_cli_startup_override() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let cli_path = temp_dir.path().join("cli.gguf");
    let configured_path = temp_dir.path().join("configured.gguf");
    std::fs::write(&cli_path, b"cli").expect("write CLI model");
    std::fs::write(&configured_path, b"configured").expect("write configured model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        cli_path.to_str().expect("UTF-8 CLI path"),
    ]);
    let configured = plugin::ModelConfigEntry {
        model: "configured-strict-model".into(),
        hardware: Some(plugin::HardwareConfig {
            model_path: Some(configured_path.to_string_lossy().into_owned()),
            ..Default::default()
        }),
        skippy: Some(plugin::SkippyConfig {
            source_policy: Some("local-required".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let configured_profile = configured.derived_profile();
    let config = plugin::MeshConfig {
        models: vec![configured],
        ..Default::default()
    };
    let specs = build_startup_model_specs(&options, &config).expect("CLI startup specs");
    assert!(
        specs
            .iter()
            .all(|spec| { spec.declared_ref.as_deref() != Some("configured-strict-model") })
    );

    register_pre_accept_local_source_policies(&config, &specs);

    assert!(skippy::local_source_required_for_model(
        "configured-strict-model",
        Some(&configured_profile)
    ));
    assert!(skippy::local_source_required_for_model(
        "configured-strict-model",
        None
    ));
}

#[test]
fn pre_accept_policy_uses_profile_with_inherited_defaults() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let cli_path = temp_dir.path().join("cli.gguf");
    let configured_path = temp_dir.path().join("configured.gguf");
    std::fs::write(&cli_path, b"cli").expect("write CLI model");
    std::fs::write(&configured_path, b"configured").expect("write configured model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        cli_path.to_str().expect("UTF-8 CLI path"),
    ]);
    let model_id = format!("configured-inherited-strict-model-{}", std::process::id());
    let configured = plugin::ModelConfigEntry {
        model: model_id.clone(),
        hardware: Some(plugin::HardwareConfig {
            model_path: Some(configured_path.to_string_lossy().into_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let config = plugin::MeshConfig {
        models: vec![configured.clone()],
        ..strict_local_source_default_config()
    };
    let effective_profile = configured
        .with_profile_defaults(config.defaults.as_ref())
        .derived_profile();
    let specs = build_startup_model_specs(&options, &config).expect("CLI startup specs");
    assert!(
        specs
            .iter()
            .all(|spec| spec.declared_ref.as_deref() != Some(model_id.as_str()))
    );
    skippy::register_local_source_policy(&model_id, &effective_profile, false);

    register_pre_accept_local_source_policies(&config, &specs);

    assert!(skippy::local_source_required_for_model(
        &model_id,
        Some(&effective_profile)
    ));
}
