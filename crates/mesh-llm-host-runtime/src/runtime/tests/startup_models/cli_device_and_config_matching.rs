use super::*;

#[test]
fn test_build_startup_model_specs_prefers_cli_models_over_config() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--ctx-size",
        "4096",
    ]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "Ignored-Model".into(),
            mmproj: Some("/tmp/ignored-mmproj.gguf".into()),
            ctx_size: Some(8192),
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
    assert_eq!(specs[0].mmproj_ref, None);
    assert_eq!(specs[0].ctx_size, Some(4096));
    assert_eq!(specs[0].gpu_id, None);
    assert!(!specs[0].resolve_pinned_gpu);
}
#[test]
fn cli_model_exact_config_ref_resolves_pinned_backend_and_keeps_cli_overrides() {
    let mut options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--ctx-size",
        "4096",
    ]);
    options.mmproj = Some(PathBuf::from("/tmp/cli-mmproj.gguf"));
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: Some("/tmp/config-mmproj.gguf".into()),
            ctx_size: Some(8192),
            gpu_id: None,
            parallel: Some(8),
            hardware: Some(plugin::HardwareConfig {
                device: Some("pci:0000:65:00.0".into()),
                model_path: Some("/configured/model.gguf".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
    assert_eq!(specs[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert!(specs[0].resolve_pinned_gpu);
    assert_eq!(specs[0].config_model_id, None);
    assert_eq!(specs[0].ctx_size, Some(4096));
    assert_eq!(
        specs[0].mmproj_ref,
        Some(PathBuf::from("/tmp/cli-mmproj.gguf"))
    );
    assert_eq!(specs[0].parallel, None);

    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: specs[0].mmproj_ref.clone(),
        ctx_size: specs[0].ctx_size,
        gpu_id: specs[0].gpu_id.clone(),
        config_model_id: specs[0].config_model_id.clone(),
        pinned_gpu: None,
        parallel: specs[0].parallel,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .expect("exact CLI config ref should resolve its pinned GPU");
    assert_eq!(
        plans[0].pinned_gpu.as_ref().unwrap().backend_device,
        "CUDA0"
    );
}

#[test]
fn cli_device_overrides_a_persisted_model_pin() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--device",
        "CUDA1",
    ]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            gpu_id: Some("pci:0000:65:00.0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs[0].gpu_id.as_deref(), Some("CUDA1"));
    assert!(specs[0].cli_device_override);

    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];
    let gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
        synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("CUDA1")),
    ];
    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(
        plans[0]
            .pinned_gpu
            .as_ref()
            .map(|gpu| gpu.backend_device.as_str()),
        Some("CUDA1")
    );
}

#[test]
fn cli_device_resolves_under_auto_assignment() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--device",
        "CUDA0",
    ]);
    let config = plugin::MeshConfig::default();
    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert!(specs[0].cli_device_override);
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(
        plans[0]
            .pinned_gpu
            .as_ref()
            .map(|gpu| gpu.backend_device.as_str()),
        Some("CUDA0")
    );
}

#[test]
fn cli_device_auto_falls_back_to_the_persisted_pin() {
    let options =
        runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M", "--device", "Auto"]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            gpu_id: Some("pci:0000:65:00.0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();

    assert_eq!(specs[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert!(!specs[0].cli_device_override);
}

#[test]
fn persisted_gpu_id_rejects_backend_device_name() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            gpu_id: Some("CUDA0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let error = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .expect_err("persisted gpu_id must remain a stable GPU ID");

    assert!(format!("{error:#}").contains("not pinnable"));
}

#[test]
fn persisted_default_device_rejects_backend_device_name() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "ad-hoc-model"]);
    let config = plugin::MeshConfig {
        defaults: Some(plugin::ModelConfigDefaults {
            hardware: Some(plugin::HardwareConfig {
                device: Some("CUDA0".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("ad-hoc-model")
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let error = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .expect_err("persisted default device must remain a stable GPU ID");

    assert!(format!("{error:#}").contains("not pinnable"));
}

#[test]
fn unresolved_cli_device_names_the_available_devices() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--device",
        "CUDA9",
    ]);
    let config = plugin::MeshConfig::default();
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let error = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("did not match any detected GPU backend device"));
    assert!(message.contains("Available devices: CUDA0"));
}

#[test]
fn cli_device_reaches_the_outer_preflight_under_auto_assignment() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--device",
        "pci:0000:ff:ff.7",
    ]);
    let config = plugin::MeshConfig::default();
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];

    let error = preflight_pinned_startup_models(&config, &specs, &mut plans, None, None)
        .expect_err("an impossible CLI device must fail before native startup");

    assert!(format!("{error:#}").contains("failed pinned GPU preflight"));
}

#[test]
fn cli_cpu_device_bypasses_gpu_preflight() {
    let options =
        runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M", "--device", "CPU"]);
    let config = plugin::MeshConfig::default();
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        gpu_id: specs[0].gpu_id.clone(),
        ..startup_model_plan("Qwen3-8B-Q4_K_M")
    }];

    preflight_pinned_startup_models(&config, &specs, &mut plans, None, None).unwrap();

    assert_eq!(plans[0].gpu_id.as_deref(), Some("CPU"));
    assert_eq!(plans[0].pinned_gpu, None);
    assert_eq!(
        startup_device_override(plans[0].gpu_id.as_deref()).as_deref(),
        Some("CPU")
    );
}

#[test]
fn auto_assignment_without_a_device_stays_inert() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
    let config = plugin::MeshConfig::default();
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![startup_model_plan("Qwen3-8B-Q4_K_M")];

    preflight_pinned_startup_models(&config, &specs, &mut plans, None, None).unwrap();

    assert_eq!(plans[0].pinned_gpu, None);
}

#[test]
fn cli_model_exact_config_ref_without_gpu_fails_before_launch() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "configured/model"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "configured/model".into(),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![startup_model_plan("configured/model")];

    let error = preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &[synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))],
        None,
    )
    .expect_err("a selected configured model without a GPU must fail preflight");
    let message = format!("{error:#}");
    assert!(message.contains("startup model 'configured/model'"));
    assert!(message.contains("missing configured gpu_id"));
}

#[test]
fn cli_model_matching_duplicate_config_refs_fails_as_ambiguous() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
    let config = plugin::MeshConfig {
        models: vec![
            plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                gpu_id: Some("pci:0000:65:00.0".into()),
                ctx_size: Some(4096),
                ..Default::default()
            },
            plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                gpu_id: Some("pci:0000:b3:00.0".into()),
                ctx_size: Some(8192),
                ..Default::default()
            },
        ],
        ..plugin::MeshConfig::default()
    };

    let error = build_startup_model_specs(&options, &config)
        .expect_err("a CLI ref cannot select between duplicate configured profiles");
    let message = format!("{error:#}");
    assert!(message.contains("matches multiple configured model entries"));
    assert!(message.contains("Qwen3-8B-Q4_K_M"));
}

#[test]
fn cli_gguf_does_not_match_configured_model_path_for_pinned_gpu() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("configured.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
    ]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "configured/model-ref".into(),
            gpu_id: Some("pci:0000:65:00.0".into()),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some(model_path.display().to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].model_ref, model_path);
    assert_eq!(specs[0].gpu_id, None);
    assert!(!specs[0].resolve_pinned_gpu);
    assert_eq!(specs[0].config_model_id, None);
}

#[test]
fn cli_gguf_inherits_global_pinned_default_without_model_ownership() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("selected.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
    ]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        defaults: Some(plugin::ModelConfigDefaults {
            hardware: Some(plugin::HardwareConfig {
                device: Some("pci:0000:65:00.0".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        models: vec![plugin::ModelConfigEntry {
            model: "configured/model-ref".into(),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some(model_path.display().to_string()),
                device: Some("pci:0000:b3:00.0".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].model_ref, model_path);
    assert_eq!(specs[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert!(!specs[0].resolve_pinned_gpu);
    assert_eq!(specs[0].config_model_id, None);

    let mut plans = vec![startup_model_plan(model_path.to_str().expect("model path"))];
    plans[0].gpu_id = specs[0].gpu_id.clone();
    preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &[synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))],
        None,
    )
    .expect("global pinned default should resolve for an explicit gguf model");
    assert_eq!(
        plans[0].pinned_gpu.as_ref().unwrap().backend_device,
        "CUDA0"
    );
}

#[test]
fn cli_model_matching_is_independent_for_multiple_models() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--model",
        "ad-hoc-model",
    ]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            gpu_id: Some("pci:0000:65:00.0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert!(specs[0].resolve_pinned_gpu);
    assert_eq!(specs[1].gpu_id, None);
    assert!(!specs[1].resolve_pinned_gpu);
}

#[test]
fn cli_unmatched_model_uses_global_pinned_default_without_model_ownership() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "ad-hoc-model"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        defaults: Some(plugin::ModelConfigDefaults {
            hardware: Some(plugin::HardwareConfig {
                device: Some("pci:0000:65:00.0".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        models: vec![plugin::ModelConfigEntry {
            model: "configured/model".into(),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert!(!specs[0].resolve_pinned_gpu);
    assert_eq!(specs[0].config_model_id, None);

    let mut plans = vec![startup_model_plan("ad-hoc-model")];
    plans[0].gpu_id = specs[0].gpu_id.clone();
    preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &[synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))],
        None,
    )
    .expect("global pinned default should resolve for an ad-hoc CLI model");
    assert_eq!(
        plans[0].pinned_gpu.as_ref().unwrap().backend_device,
        "CUDA0"
    );
}
