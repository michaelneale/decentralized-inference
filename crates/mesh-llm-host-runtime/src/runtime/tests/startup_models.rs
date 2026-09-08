use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn parser_mode_runtime_events_maps_every_config_mode() {
    use mesh_llm_config::LifecycleLogParserMode;
    use skippy_runtime::NativeLogParserMode;

    assert_eq!(
        native_log_parser_mode(LifecycleLogParserMode::Auto),
        NativeLogParserMode::Auto
    );
    assert_eq!(
        native_log_parser_mode(LifecycleLogParserMode::Enabled),
        NativeLogParserMode::Enabled
    );
    assert_eq!(
        native_log_parser_mode(LifecycleLogParserMode::Disabled),
        NativeLogParserMode::Disabled
    );
}

#[tokio::test]
async fn advisory_startup_task_does_not_block_runtime_startup() {
    let started = std::sync::Arc::new(tokio::sync::Notify::new());
    let completed = std::sync::Arc::new(AtomicBool::new(false));
    let task_started = std::sync::Arc::clone(&started);
    let task_completed = std::sync::Arc::clone(&completed);

    spawn_advisory_startup_task(move || {
        task_started.notify_one();
        std::thread::sleep(std::time::Duration::from_millis(100));
        task_completed.store(true, Ordering::Release);
    });

    tokio::time::timeout(std::time::Duration::from_secs(1), started.notified())
        .await
        .expect("advisory task should be scheduled");
    assert!(
        !completed.load(Ordering::Acquire),
        "startup must not wait for the advisory task"
    );
}

#[test]
fn runtime_config_enables_debug_and_listen_all_options() {
    let mut options = RuntimeOptions::default();
    let mut config = plugin::MeshConfig::default();
    config.runtime.debug = true;
    config.runtime.listen_all = true;

    apply_runtime_config_options(&mut options, &config);

    assert!(options.debug);
    assert!(options.listen_all);
}

#[test]
fn explicit_debug_and_listen_all_options_survive_false_config_defaults() {
    let mut options = RuntimeOptions {
        debug: true,
        listen_all: true,
        ..RuntimeOptions::default()
    };
    let config = plugin::MeshConfig::default();

    apply_runtime_config_options(&mut options, &config);

    assert!(options.debug);
    assert!(options.listen_all);
}

#[test]
fn cli_audit_path_has_precedence_and_enables_sink_configuration() {
    let cli_path = std::path::PathBuf::from("cli-audit.log");
    let mut options = RuntimeOptions {
        audit_log_path: Some(cli_path.clone()),
        audit_log_format: mesh_llm_events::audit::AuditLogFormat::JsonLines,
        audit_log_level: mesh_llm_events::audit::AuditLevel::Error,
        ..RuntimeOptions::default()
    };
    let mut config = plugin::MeshConfig::default();
    config.logging.audit.enabled = Some(false);
    config.logging.audit.log_path = Some("config-audit.log".into());
    config.logging.audit.log_format = Some("json_lines".into());
    config.logging.audit.log_level = Some("info".into());
    config.logging.audit.max_file_size_mb = Some(7);
    config.logging.audit.max_files = Some(3);

    apply_runtime_config_options(&mut options, &config);

    assert_eq!(options.audit_log_path, Some(cli_path));
    assert_eq!(
        options.audit_log_format,
        mesh_llm_events::audit::AuditLogFormat::JsonLines
    );
    assert_eq!(
        options.audit_log_level,
        mesh_llm_events::audit::AuditLevel::Error
    );
    assert_eq!(options.audit_max_file_size, 7 * 1024 * 1024);
    assert_eq!(options.audit_max_files, 3);
}

#[test]
fn configured_audit_path_implicitly_enables_config_sink() {
    let mut options = RuntimeOptions::default();
    let mut config = plugin::MeshConfig::default();
    config.logging.audit.log_path = Some("configured-audit.log".into());
    config.logging.audit.max_file_size_mb = Some(2);
    config.logging.audit.max_files = Some(4);

    apply_runtime_config_options(&mut options, &config);

    assert_eq!(options.audit_log_path, config.logging.audit.log_path);
    assert_eq!(options.audit_max_file_size, 2 * 1024 * 1024);
    assert_eq!(options.audit_max_files, 4);
}

#[test]
fn early_topology_audit_configuration_applies_configured_and_cli_settings() {
    let mut config = plugin::MeshConfig::default();
    config.logging.audit.log_path = Some("configured-audit.log".into());
    config.logging.audit.log_level = Some("warn".into());
    config.logging.audit.max_file_size_mb = Some(2);
    config.logging.audit.max_files = Some(4);

    let mut local_model_only = RuntimeOptions {
        local_model_only: true,
        ..RuntimeOptions::default()
    };
    let local_model_only_initializations = std::cell::Cell::new(0);
    initialize_early_topology_audit_logging_with(
        &mut local_model_only,
        || Ok(config.clone()),
        |options| {
            local_model_only_initializations.set(local_model_only_initializations.get() + 1);
            assert_eq!(
                options.audit_log_path,
                Some(std::path::PathBuf::from("configured-audit.log"))
            );
            assert_eq!(
                options.audit_log_level,
                mesh_llm_events::audit::AuditLevel::Warn
            );
            assert_eq!(options.audit_max_file_size, 2 * 1024 * 1024);
            assert_eq!(options.audit_max_files, 4);
            Ok(())
        },
    )
    .expect("local-model-only audit initialization succeeds");
    assert_eq!(local_model_only_initializations.get(), 1);

    let cli_path = std::path::PathBuf::from("cli-audit.log");
    let mut plugin = RuntimeOptions {
        plugin: Some("example".to_string()),
        audit_log_path: Some(cli_path.clone()),
        audit_log_level: mesh_llm_events::audit::AuditLevel::Error,
        ..RuntimeOptions::default()
    };
    let plugin_initializations = std::cell::Cell::new(0);
    initialize_early_topology_audit_logging_with(
        &mut plugin,
        || Ok(config),
        |options| {
            plugin_initializations.set(plugin_initializations.get() + 1);
            assert_eq!(options.audit_log_path, Some(cli_path.clone()));
            assert_eq!(
                options.audit_log_level,
                mesh_llm_events::audit::AuditLevel::Error
            );
            assert_eq!(options.audit_max_file_size, 2 * 1024 * 1024);
            assert_eq!(options.audit_max_files, 4);
            Ok(())
        },
    )
    .expect("plugin audit initialization succeeds");
    assert_eq!(plugin_initializations.get(), 1);
}

#[test]
fn early_topology_audit_configuration_load_failures_are_nonfatal() {
    let cli_path = std::path::PathBuf::from("cli-audit.log");
    let mut options = RuntimeOptions {
        plugin: Some("example".to_string()),
        audit_log_path: Some(cli_path.clone()),
        audit_log_level: mesh_llm_events::audit::AuditLevel::Error,
        ..RuntimeOptions::default()
    };

    let initializations = std::cell::Cell::new(0);
    initialize_early_topology_audit_logging_with(
        &mut options,
        || Err(anyhow::anyhow!("invalid config")),
        |options| {
            initializations.set(initializations.get() + 1);
            assert_eq!(options.audit_log_path, Some(cli_path.clone()));
            assert_eq!(
                options.audit_log_level,
                mesh_llm_events::audit::AuditLevel::Error
            );
            Ok(())
        },
    )
    .expect("config load failures do not prevent plugin startup");
    assert_eq!(initializations.get(), 1);
}

#[test]
fn cli_speculative_overrides_take_precedence_without_dropping_model_tuning() {
    let mut config: plugin::MeshConfig = toml::from_str(
        r#"
[defaults.speculative]
strategy = "mtp-cache"
verify_window_pipeline_depth = 2

[[models]]
model = "test/model"

[models.speculative]
ngram_max_proposal_tokens = 6
"#,
    )
    .expect("config parses");
    let mut overrides = plugin::SpeculativeConfig::default();
    overrides.strategy = Some("mtp".to_string());
    overrides.verify_window_pipeline_depth = Some(3);

    apply_runtime_cli_speculative_overrides(&mut config, Some(&overrides));

    let model = config.models[0]
        .speculative
        .as_ref()
        .expect("model speculative config is resolved");
    assert_eq!(model.strategy.as_deref(), Some("mtp"));
    assert_eq!(model.ngram_max_proposal_tokens, Some(6));
    assert_eq!(model.verify_window_pipeline_depth, Some(3));
}

#[test]
fn cli_checkpoint_overrides_are_canonical_and_take_precedence() {
    let temp = tempfile::tempdir().expect("tempdir");
    let imatrix = temp.path().join("test.imatrix");
    std::fs::write(&imatrix, b"fixture").expect("write fixture");
    let mut config: plugin::MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
checkpoint_quantization = "F16"
checkpoint_imatrix = "old-default.dat"

[[models]]
model = "test/model"

[models.hardware]
checkpoint_quantization = "Q8_0"
checkpoint_imatrix = "old-model.dat"
"#,
    )
    .expect("config parses");

    apply_runtime_cli_checkpoint_overrides(&mut config, Some("q4-k"), Some(&imatrix))
        .expect("CLI overrides apply");

    let canonical_imatrix = std::fs::canonicalize(&imatrix)
        .expect("canonical imatrix")
        .to_string_lossy()
        .into_owned();
    let defaults = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref())
        .expect("default hardware");
    assert_eq!(defaults.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
    assert_eq!(
        defaults.checkpoint_imatrix.as_deref(),
        Some(canonical_imatrix.as_str())
    );
    let model = config.models[0].hardware.as_ref().expect("model hardware");
    assert_eq!(model.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
    assert_eq!(
        model.checkpoint_imatrix.as_deref(),
        Some(canonical_imatrix.as_str())
    );
}

#[test]
fn cli_checkpoint_override_rejects_unknown_recipe() {
    let mut config = plugin::MeshConfig::default();
    let error = apply_runtime_cli_checkpoint_overrides(&mut config, Some("Q4_FANTASY"), None)
        .expect_err("unknown recipe must fail");

    assert!(error.to_string().contains("invalid --quant"));
}

fn remote_catalog_layer_entry(
    variant_name: &str,
    curated_name: &str,
    source_repo: &str,
    package_repo: &str,
) -> models::remote_catalog::CatalogEntry {
    let mut variants = std::collections::HashMap::new();
    variants.insert(
        variant_name.to_string(),
        models::remote_catalog::CatalogVariant {
            source: models::remote_catalog::CatalogSource {
                repo: source_repo.to_string(),
                revision: Some("main".to_string()),
                file: Some(format!("{variant_name}.gguf")),
            },
            curated: models::remote_catalog::CatalogCurated {
                name: curated_name.to_string(),
                size: None,
                description: None,
                draft: None,
                moe: None,
                extra_files: Vec::new(),
                mmproj: None,
            },
            packages: vec![models::remote_catalog::CatalogPackage {
                package_type: "layer-package".to_string(),
                repo: package_repo.to_string(),
                layer_count: Some(12),
                total_bytes: Some(42),
            }],
        },
    );
    models::remote_catalog::CatalogEntry {
        schema_version: 1,
        source_repo: source_repo.to_string(),
        variants,
    }
}

fn startup_model_plan(model_ref: &str) -> StartupModelPlan {
    StartupModelPlan {
        declared_ref: model_ref.to_string(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/model.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: None,
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }
}

fn write_identity_test_gguf(path: &std::path::Path, context_length: u32) {
    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(bytes, value);
    }
    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&0_i64.to_le_bytes());
    bytes.extend_from_slice(&8_i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
    push_u32_kv(&mut bytes, "llama.context_length", context_length);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    std::fs::write(path, bytes).expect("write identity test GGUF");
}

fn direct_gguf_startup_spec(path: &std::path::Path, alias: Option<&str>) -> StartupModelSpec {
    StartupModelSpec {
        model_ref: path.to_path_buf(),
        declared_ref: alias.map(str::to_string),
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: None,
        cli_device_override: false,
        resolve_pinned_gpu: false,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }
}

#[tokio::test]
async fn bare_direct_gguf_startup_identity_is_full_content_hash_across_paths() {
    let first_dir = tempfile::tempdir().expect("first tempdir");
    let second_dir = tempfile::tempdir().expect("second tempdir");
    let first_path = first_dir.path().join("first-name.gguf");
    let second_path = second_dir.path().join("other-name.gguf");
    write_identity_test_gguf(&first_path, 4096);
    std::fs::copy(&first_path, &second_path).expect("copy identical GGUF");

    let first = resolve_startup_models(&[direct_gguf_startup_spec(&first_path, None)], true)
        .await
        .expect("resolve first direct GGUF")
        .remove(0);
    let second = resolve_startup_models(&[direct_gguf_startup_spec(&second_path, None)], true)
        .await
        .expect("resolve second direct GGUF")
        .remove(0);
    let first_package = first
        .preindexed_split_package
        .as_ref()
        .expect("direct GGUF package must be indexed at resolution");
    let second_package = second
        .preindexed_split_package
        .as_ref()
        .expect("second direct GGUF package must be indexed at resolution");

    assert_eq!(first.declared_ref, second.declared_ref);
    assert_eq!(
        first.declared_ref,
        format!("local-gguf/sha256-{}", first_package.source_model_sha256)
    );
    assert_eq!(first_package.source_model_sha256.len(), 64);
    assert_eq!(first_package.package_ref, second_package.package_ref);
    assert_ne!(first.resolved_path, second.resolved_path);
}

#[tokio::test]
async fn direct_gguf_explicit_alias_wins_without_changing_content_identity() {
    let first_dir = tempfile::tempdir().expect("first tempdir");
    let second_dir = tempfile::tempdir().expect("second tempdir");
    let first_path = first_dir.path().join("first.gguf");
    let second_path = second_dir.path().join("second.gguf");
    write_identity_test_gguf(&first_path, 4096);
    std::fs::copy(&first_path, &second_path).expect("copy identical GGUF");

    let first = resolve_startup_models(
        &[direct_gguf_startup_spec(&first_path, Some("shared/model"))],
        true,
    )
    .await
    .expect("resolve first aliased GGUF")
    .remove(0);
    let second = resolve_startup_models(
        &[direct_gguf_startup_spec(&second_path, Some("shared/model"))],
        true,
    )
    .await
    .expect("resolve second aliased GGUF")
    .remove(0);

    assert_eq!(first.declared_ref, "shared/model");
    assert_eq!(second.declared_ref, "shared/model");
    assert_eq!(
        first.preindexed_split_package.unwrap().package_ref,
        second.preindexed_split_package.unwrap().package_ref
    );
}

#[test]
#[serial_test::serial]
fn split_layer_package_resolution_checks_remote_catalog_for_model_name() {
    let _catalog_guard =
        models::remote_catalog::set_catalog_entries_for_test(vec![remote_catalog_layer_entry(
            "RemoteSplitOnlyModel-Q4_K_M",
            "Remote Split Only Model Q4_K_M",
            "mesh-test/remote-split-only-model",
            "meshllm/remote-split-only-model-layers",
        )]);

    let resolved = resolve_split_layer_package(
        "Remote Split Only Model",
        Path::new("Remote Split Only Model"),
    );

    assert_eq!(
        resolved,
        Some("hf://meshllm/remote-split-only-model-layers".to_string())
    );
}

#[test]
#[serial_test::serial]
fn split_layer_package_resolution_accepts_package_repo_shorthand() {
    let _catalog_guard =
        models::remote_catalog::set_catalog_entries_for_test(vec![remote_catalog_layer_entry(
            "Qwen3-8B-Q4_K_M",
            "Qwen3 8B Q4_K_M",
            "unsloth/Qwen3-8B-GGUF",
            "meshllm/Qwen3-8B-Q4_K_M-layers",
        )]);

    let resolved = resolve_split_layer_package(
        "meshllm/Qwen3-8B-Q4_K_M-layers",
        Path::new("meshllm/Qwen3-8B-Q4_K_M-layers"),
    );

    assert_eq!(
        resolved,
        Some("hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string())
    );
}

#[test]
#[serial_test::serial]
fn split_layer_package_resolution_probes_hf_manifest_without_name_heuristic() {
    let _catalog_guard = models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    let _probe_guard =
        models::remote_catalog::set_hf_model_file_probe_for_test(|repo, revision, file| {
            repo == "meshllm/custom-package" && revision == "main" && file == "model-package.json"
        });

    let resolved = resolve_split_layer_package(
        "meshllm/custom-package",
        Path::new("meshllm/custom-package"),
    );

    assert_eq!(resolved, Some("hf://meshllm/custom-package".to_string()));
    assert_eq!(
        resolve_split_layer_package(
            "meshllm/custom-package:Q4_K_M",
            Path::new("meshllm/custom-package:Q4_K_M"),
        ),
        None
    );
}

#[test]
#[serial_test::serial]
fn layer_package_resolution_keeps_existing_local_gguf() {
    let _catalog_guard =
        models::remote_catalog::set_catalog_entries_for_test(vec![remote_catalog_layer_entry(
            "LocalModel-Q4_K_M",
            "Local Model Q4_K_M",
            "mesh-test/local-model",
            "meshllm/local-model-layers",
        )]);
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let local_model = temp_dir.path().join("LocalModel-Q4_K_M.gguf");
    std::fs::write(&local_model, b"gguf").expect("write local model");

    let resolved = resolve_split_layer_package("LocalModel-Q4_K_M", &local_model);

    assert_eq!(resolved, None);
}

#[test]
fn runtime_model_capacity_counts_split_gguf_parts() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let first_part = temp_dir.path().join("model-00001-of-00002.gguf");
    let second_part = temp_dir.path().join("model-00002-of-00002.gguf");
    std::fs::write(&first_part, vec![0u8; 100]).expect("write first split part");
    std::fs::write(&second_part, vec![0u8; 200]).expect("write second split part");

    let too_small = runtime_model_capacity_for_path(&first_part, 329);
    assert_eq!(too_small.required_bytes, 330);
    assert!(!too_small.fits);

    let enough = runtime_model_capacity_for_path(&first_part, 330);
    assert_eq!(enough.required_bytes, 330);
    assert!(enough.fits);
}

#[test]
#[serial_test::serial]
fn skippy_native_logging_setup_is_nonfatal_when_log_dir_cannot_be_created() {
    struct RestoreNativeLogs;

    impl Drop for RestoreNativeLogs {
        fn drop(&mut self) {
            skippy_runtime::restore_native_logs();
        }
    }

    let _restore = RestoreNativeLogs;
    let path = std::env::temp_dir().join(format!(
        "mesh-native-log-runtime-file-{}-{}",
        std::process::id(),
        current_time_unix_ms()
    ));
    std::fs::write(&path, b"not a directory").expect("create runtime path file");

    let configured_path = configure_skippy_native_logging(Some(&path));

    std::fs::remove_file(&path).expect("remove runtime path file");
    assert_eq!(configured_path, None);
}

#[test]
#[serial_test::serial]
fn skippy_native_logging_setup_suppresses_logs_without_runtime_dir() {
    struct RestoreNativeLogs;

    impl Drop for RestoreNativeLogs {
        fn drop(&mut self) {
            skippy_runtime::restore_native_logs();
        }
    }

    let _restore = RestoreNativeLogs;
    assert_eq!(configure_skippy_native_logging(None), None);
}

fn synthetic_gpu(index: usize, stable_id: Option<&str>, backend_device: Option<&str>) -> GpuFacts {
    GpuFacts {
        index,
        display_name: format!("GPU {index}"),
        backend_device: backend_device.map(str::to_string),
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: stable_id.map(str::to_string),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }
}

#[tokio::test]
#[serial_test::serial]
#[ignore = "downloads ~800MB from HuggingFace and depends on exact snapshot hash"]
async fn resolve_model_accepts_short_catalog_name_from_hf_cache() {
    let cache_root = std::env::temp_dir().join(format!(
        "mesh-llm-short-name-cache-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&cache_root).unwrap();
    let _hub_cache = EnvVarGuard::set_path("HF_HUB_CACHE", &cache_root);
    let _hf_home = EnvVarGuard::remove("HF_HOME");
    let _xdg_cache_home = EnvVarGuard::remove("XDG_CACHE_HOME");

    let repo_id = "bartowski/Llama-3.2-1B-Instruct-GGUF";
    let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoTypeModel));
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
    let model_path = huggingface_snapshot_path(repo_id, RepoTypeModel, "test-commit")
        .join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
    std::fs::write(&model_path, b"gguf").unwrap();

    let resolved = resolve_model(Path::new("Llama-3.2-1B-Instruct-Q4_K_M"))
        .await
        .unwrap();
    assert_eq!(resolved, model_path);

    let _ = std::fs::remove_dir_all(&cache_root);
}

#[tokio::test]
#[serial_test::serial]
async fn resolve_model_accepts_non_catalog_name_from_hf_cache() {
    let cache_root = std::env::temp_dir().join(format!(
        "mesh-llm-non-catalog-cache-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&cache_root).unwrap();
    let _hub_cache = EnvVarGuard::set_path("HF_HUB_CACHE", &cache_root);
    let _hf_home = EnvVarGuard::remove("HF_HOME");
    let _xdg_cache_home = EnvVarGuard::remove("XDG_CACHE_HOME");

    let repo_id = "someone/Custom-GGUF";
    let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoTypeModel));
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
    let model_path = huggingface_snapshot_path(repo_id, RepoTypeModel, "test-commit")
        .join("Custom-Model-Q4_K_M.gguf");
    std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
    std::fs::write(&model_path, b"gguf").unwrap();

    let resolved_by_stem = resolve_model(Path::new("Custom-Model-Q4_K_M"))
        .await
        .unwrap();
    assert_eq!(resolved_by_stem, model_path);

    let resolved_by_filename = resolve_model(Path::new("Custom-Model-Q4_K_M.gguf"))
        .await
        .unwrap();
    assert_eq!(resolved_by_filename, model_path);

    let _ = std::fs::remove_dir_all(&cache_root);
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set_path(key: &'static str, value: &Path) -> Self {
        let guard = Self {
            key,
            previous: std::env::var_os(key),
        };
        // SAFETY: these serial tests mutate the process environment before
        // model resolution and restore it via Drop before the next test.
        unsafe { std::env::set_var(key, value) };
        guard
    }

    fn remove(key: &'static str) -> Self {
        let guard = Self {
            key,
            previous: std::env::var_os(key),
        };
        // SAFETY: these serial tests mutate the process environment before
        // model resolution and restore it via Drop before the next test.
        unsafe { std::env::remove_var(key) };
        guard
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        restore_env(self.key, self.previous.take());
    }
}

#[test]
fn test_build_serving_list_auto_no_resolved() {
    let resolved: Vec<StartupModelPlan> = vec![];
    let result = build_serving_list(&resolved, "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M");
    assert_eq!(result, vec!["unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"]);
}

#[test]
fn test_build_serving_list_explicit_single_model() {
    let resolved = vec![startup_model_plan("unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M")];
    let result = build_serving_list(&resolved, "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M");
    assert_eq!(result, vec!["unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_build_serving_list_explicit_multi_model() {
    let resolved = vec![
        startup_model_plan("unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"),
        startup_model_plan("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M"),
    ];
    let result = build_serving_list(&resolved, "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M");
    assert_eq!(
        result,
        vec![
            "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M",
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M"
        ]
    );
}

#[test]
fn test_build_serving_list_split_gguf() {
    let resolved = vec![startup_model_plan("MiniMaxAI/MiniMax-M2.5-GGUF:Q4_K_M")];
    let result = build_serving_list(&resolved, "MiniMaxAI/MiniMax-M2.5-GGUF:Q4_K_M");
    assert_eq!(result, vec!["MiniMaxAI/MiniMax-M2.5-GGUF:Q4_K_M"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_build_serving_list_keeps_synthetic_local_ref() {
    let resolved = vec![startup_model_plan("local-gguf/sha256-abcdef0123456789")];
    let result = build_serving_list(&resolved, "local-gguf/sha256-abcdef0123456789");
    assert_eq!(result, vec!["local-gguf/sha256-abcdef0123456789"]);
    assert_eq!(result.len(), 1);
}

mod cli_device_and_config_matching;
mod local_required;

#[tokio::test]
async fn prepare_runtime_startup_defers_model_resolution_until_after_surfaces() {
    let bin_dir = tempfile::tempdir().expect("temporary runtime bin directory");
    let mut options =
        runtime_options_for_test(&["mesh-llm", "--model", "qa.invalid/model@main:missing.gguf"]);
    options.bin_dir = Some(bin_dir.path().to_path_buf());
    let config = plugin::MeshConfig::default();

    let prepared = prepare_runtime_startup(
        &options,
        &config,
        Some(RuntimeSurface::Serve),
        mesh_llm_config::RuntimeMode::Serve,
    )
    .await
    .expect("startup preparation must not resolve the model")
    .expect("serve startup remains active");

    assert_eq!(prepared.startup_specs.len(), 1);
    assert_eq!(
        prepared.startup_specs[0].model_ref,
        PathBuf::from("qa.invalid/model@main:missing.gguf")
    );
    assert_eq!(
        prepared.requested_model_names,
        vec!["qa.invalid/model@main:missing.gguf"]
    );
}

#[test]
fn test_build_startup_model_specs_uses_config_models_when_cli_is_empty() {
    let options = runtime_options_for_test(&["mesh-llm", "--ctx-size", "4096"]);
    let config = plugin::MeshConfig {
        models: vec![
            plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            },
            plugin::ModelConfigEntry {
                model: "bartowski/Qwen2.5-VL/model.gguf".into(),
                mmproj: Some("bartowski/Qwen2.5-VL/mmproj.gguf".into()),
                ctx_size: Some(16384),
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            },
        ],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
    assert_eq!(specs[0].ctx_size, Some(4096));
    assert_eq!(specs[0].gpu_id, None);
    assert!(specs[0].resolve_pinned_gpu);
    assert_eq!(
        specs[1].mmproj_ref,
        Some(PathBuf::from("bartowski/Qwen2.5-VL/mmproj.gguf"))
    );
    assert_eq!(specs[1].ctx_size, Some(4096));
    assert_eq!(specs[1].gpu_id, None);
    assert!(specs[1].resolve_pinned_gpu);
}

#[test]
fn configured_server_alias_becomes_the_served_model_identity() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config: plugin::MeshConfig = toml::from_str(
        r#"
[defaults.advanced.server]
alias = "default-model"

[[models]]
model = "canonical/default-model"

[[models]]
model = "canonical/override-model"

[models.advanced.server]
alias = "public-model"
"#,
    )
    .expect("aliased model config parses");

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");

    assert_eq!(specs[0].model_ref, PathBuf::from("canonical/default-model"));
    assert_eq!(specs[0].declared_ref.as_deref(), Some("default-model"));
    assert_eq!(
        specs[0].config_model_id.as_deref(),
        Some("canonical/default-model")
    );
    assert_eq!(
        specs[1].model_ref,
        PathBuf::from("canonical/override-model")
    );
    assert_eq!(specs[1].declared_ref.as_deref(), Some("public-model"));
    assert_eq!(
        specs[1].config_model_id.as_deref(),
        Some("canonical/override-model")
    );
    assert!(specs.iter().all(|spec| spec.resolve_pinned_gpu));
}

#[test]
fn ad_hoc_gguf_alias_preserves_existing_cli_model_overrides() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("model.gguf");
    let projector_path = temp_dir.path().join("mmproj.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    std::fs::write(&projector_path, b"gguf").expect("write projector");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
        "--model",
        "public-model",
        "--mmproj",
        projector_path.to_str().expect("projector path"),
        "--ctx-size",
        "8192",
    ]);

    let specs =
        build_startup_model_specs(&options, &plugin::MeshConfig::default()).expect("startup specs");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].declared_ref.as_deref(), Some("public-model"));
    assert_eq!(
        specs[0].mmproj_ref.as_deref(),
        Some(projector_path.as_path())
    );
    assert_eq!(specs[0].ctx_size, Some(8192));
    assert!(!specs[0].resolve_pinned_gpu);
}

#[test]
fn ad_hoc_model_inherits_applicable_config_defaults() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
    let config: plugin::MeshConfig = toml::from_str(
        r#"
[defaults.model_fit]
ctx_size = 12288
batch = 768
ubatch = 192
cache_type_k = "q8_0"
cache_type_v = "q4_0"
flash_attention = "enabled"

[defaults.throughput]
parallel = 3
"#,
    )
    .expect("defaults config parses");

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].ctx_size, Some(12288));
    assert_eq!(specs[0].parallel, Some(3));
    assert_eq!(specs[0].cache_type_k.as_deref(), Some("q8_0"));
    assert_eq!(specs[0].cache_type_v.as_deref(), Some("q4_0"));
    assert_eq!(specs[0].n_batch, Some(768));
    assert_eq!(specs[0].n_ubatch, Some(192));
    assert_eq!(specs[0].flash_attention, FlashAttentionType::Enabled);
    assert!(!specs[0].profile.is_empty());
    assert!(!specs[0].resolve_pinned_gpu);
}

#[test]
fn startup_profile_uses_full_model_over_defaults_merge() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config: plugin::MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
repack = true

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
repack = false
"#,
    )
    .expect("profile config parses");
    let model = &config.models[0];

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    let expected_profile = model
        .with_profile_defaults(config.defaults.as_ref())
        .derived_profile();
    let defaults_only_profile = plugin::ModelConfigEntry {
        model: model.model.clone(),
        ..plugin::ModelConfigEntry::default()
    }
    .with_profile_defaults(config.defaults.as_ref())
    .derived_profile();

    assert_eq!(specs[0].profile, expected_profile);
    assert_ne!(specs[0].profile, defaults_only_profile);
}

#[test]
fn ad_hoc_model_inherits_default_mmproj_unless_cli_overrides_it() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let inherited = temp_dir.path().join("inherited-mmproj.gguf");
    let explicit = temp_dir.path().join("explicit-mmproj.gguf");
    std::fs::write(&inherited, b"gguf").expect("write inherited projector");
    std::fs::write(&explicit, b"gguf").expect("write explicit projector");
    let config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[defaults.multimodal]
mmproj = {:?}
"#,
        inherited.to_string_lossy()
    ))
    .expect("defaults config parses");

    let inherited_options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-VL"]);
    let inherited_specs =
        build_startup_model_specs(&inherited_options, &config).expect("inherited startup specs");
    assert_eq!(
        inherited_specs[0].mmproj_ref.as_deref(),
        Some(inherited.as_path())
    );

    let explicit_options = runtime_options_for_test(&[
        "mesh-llm",
        "--model",
        "Qwen3-VL",
        "--mmproj",
        explicit.to_str().expect("explicit projector path"),
    ]);
    let explicit_specs =
        build_startup_model_specs(&explicit_options, &config).expect("explicit startup specs");
    assert_eq!(
        explicit_specs[0].mmproj_ref.as_deref(),
        Some(explicit.as_path())
    );
}

#[tokio::test]
async fn configured_alias_keeps_canonical_model_id_through_resolution() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("model.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "canonical/model"
[models.hardware]
model_path = {:?}
[models.advanced.server]
alias = "public-model"
"#,
        model_path.to_string_lossy()
    ))
    .expect("model config parses");

    let specs = build_startup_model_specs(&runtime_options_for_test(&["mesh-llm"]), &config)
        .expect("startup specs");
    let plans = resolve_local_model_only_startup_models(&specs)
        .await
        .expect("startup plans");

    assert_eq!(plans[0].declared_ref, "public-model");
    assert_eq!(plans[0].config_model_id.as_deref(), Some("canonical/model"));
}

#[test]
fn gguf_with_plain_model_name_binds_the_name_to_the_local_file() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("deepseek.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
        "--model",
        "deepseek-v4-flash",
    ]);

    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "deepseek-v4-flash".into(),
            gpu_id: Some("pci:0000:65:00.0".into()),
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].model_ref, model_path);
    assert_eq!(specs[0].declared_ref.as_deref(), Some("deepseek-v4-flash"));
    assert_eq!(specs[0].gpu_id, None);
    assert!(!specs[0].resolve_pinned_gpu);
}

#[test]
fn gguf_with_hugging_face_model_ref_still_serves_two_models() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("deepseek.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
        "--model",
        "unsloth/Qwen3-8B-GGUF:Q4_K_M",
    ]);

    let specs =
        build_startup_model_specs(&options, &plugin::MeshConfig::default()).expect("startup specs");
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].declared_ref, None);
    assert_eq!(
        specs[1].model_ref,
        PathBuf::from("unsloth/Qwen3-8B-GGUF:Q4_K_M")
    );
    assert_eq!(specs[1].declared_ref, None);
}

#[tokio::test]
async fn gguf_alias_resolves_without_catalog_lookup() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("deepseek.gguf");
    write_identity_test_gguf(&model_path, 4096);
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--gguf",
        model_path.to_str().expect("model path"),
        "--model",
        "deepseek-v4-flash",
    ]);
    let specs =
        build_startup_model_specs(&options, &plugin::MeshConfig::default()).expect("startup specs");

    let plans = resolve_startup_models(&specs, false)
        .await
        .expect("startup models resolve");
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].declared_ref, "deepseek-v4-flash");
    assert_eq!(
        std::fs::canonicalize(&plans[0].resolved_path).expect("canonical resolved path"),
        std::fs::canonicalize(&model_path).expect("canonical model path")
    );
}

#[tokio::test]
async fn config_hardware_model_path_loads_local_file_under_logical_model_identity() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("glm.gguf");
    std::fs::write(&model_path, b"gguf").expect("write model");
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "glm-4.7-flash"
[models.hardware]
model_path = {:?}
[models.advanced.server]
alias = "public-glm"
"#,
        model_path.to_string_lossy()
    ))
    .expect("aliased local model config parses");

    let specs = build_startup_model_specs(&options, &config).expect("startup specs");
    assert_eq!(specs[0].model_ref, model_path);
    assert_eq!(specs[0].declared_ref.as_deref(), Some("public-glm"));

    let plans = resolve_local_model_only_startup_models(&specs)
        .await
        .expect("local startup model resolves");
    assert_eq!(
        plans[0].resolved_path,
        std::fs::canonicalize(&specs[0].model_ref).expect("canonical model path")
    );
    assert_eq!(plans[0].declared_ref, "public-glm");
    assert_eq!(plans[0].config_model_id.as_deref(), Some("glm-4.7-flash"));
}

#[tokio::test]
async fn local_model_only_rejects_catalog_and_relative_model_refs() {
    let specs = [StartupModelSpec {
        model_ref: PathBuf::from("glm-4.7-flash"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: None,
        cli_device_override: false,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: "default".into(),
        resolve_pinned_gpu: false,
    }];

    let error = resolve_local_model_only_startup_models(&specs)
        .await
        .expect_err("direct serving must not resolve a catalog or relative model ref");
    assert!(format!("{error:#}").contains("absolute local model path"));
}

#[test]
fn config_hardware_model_path_rejects_relative_path_before_resolution() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "glm-4.7-flash".into(),
            hardware: Some(plugin::HardwareConfig {
                model_path: Some("relative/model.gguf".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error = build_startup_model_specs(&options, &config).unwrap_err();
    assert!(format!("{error:#}").contains("hardware.model_path"));
    assert!(format!("{error:#}").contains("must be absolute"));
}

#[test]
fn test_build_startup_model_specs_ignores_config_models_for_client() {
    let options = runtime_options_for_test(&["mesh-llm", "--client"]);
    let config = plugin::MeshConfig {
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
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
    assert!(specs.is_empty());
}

#[test]
fn test_build_startup_model_specs_carries_profile_from_config() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        models: vec![
            plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(4096),
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            },
            plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            },
            plugin::ModelConfigEntry {
                model: "Llama-3-8B-Q4_K_M".into(),
                mmproj: None,
                ..Default::default()
            },
        ],
        ..plugin::MeshConfig::default()
    };

    let specs = build_startup_model_specs(&options, &config).unwrap();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
    let profile_4096 = config.models[0].derived_profile();
    let profile_8192 = config.models[1].derived_profile();
    let profile_default = config.models[2].derived_profile();
    assert_eq!(specs[0].profile, profile_4096);
    assert_eq!(specs[1].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
    assert_eq!(specs[1].profile, profile_8192);
    assert_ne!(
        profile_4096, profile_8192,
        "different ctx_size must produce different derived profiles"
    );
    assert_eq!(specs[2].model_ref, PathBuf::from("Llama-3-8B-Q4_K_M"));
    assert_eq!(specs[2].profile, profile_default);
}

#[test]
fn pinned_gpu_startup_preflight_uses_config_gpu_id() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
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
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(8192),
        gpu_id: specs[0].gpu_id.clone(),
        config_model_id: specs[0].config_model_id.clone(),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
        synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("CUDA1")),
    ];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(plans[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert_eq!(
        plans[0].pinned_gpu,
        Some(StartupPinnedGpuTarget {
            index: 0,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA0".into(),
            vram_bytes: 24_000_000_000,
            reserved_bytes: None,
        })
    );
}

#[test]
fn pinned_gpu_startup_preflight_synthesizes_backend_from_binary_flavor() {
    let mut gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
        synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1")),
    ];

    apply_backend_devices_for_flavor(&mut gpus, Some(backend::BinaryFlavor::Vulkan));

    assert_eq!(gpus[0].backend_device.as_deref(), Some("Vulkan0"));
    assert_eq!(gpus[1].backend_device.as_deref(), Some("Vulkan1"));
}

#[test]
fn pinned_gpu_startup_preflight_rejects_synthesized_backend_missing_from_probe() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("Vulkan1"))];
    let backend_probe = backend::BinaryBackendDeviceProbe {
        path: PathBuf::from("/tmp/backend-vulkan"),
        flavor: Some(backend::BinaryFlavor::Vulkan),
        available_devices: vec!["Vulkan0".into(), "CPU".into()],
    };

    let err = preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &gpus,
        Some(&backend_probe),
    )
    .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("requested device Vulkan1 is not supported"));
    assert!(message.contains("Available devices: Vulkan0, CPU"));
}

#[test]
fn pinned_gpu_startup_preflight_canonicalizes_rocm_hip_alias_from_probe() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1"))];
    let backend_probe = backend::BinaryBackendDeviceProbe {
        path: PathBuf::from("/tmp/backend-rocm"),
        flavor: Some(backend::BinaryFlavor::Rocm),
        available_devices: vec!["HIP1".into(), "CPU".into()],
    };

    preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &gpus,
        Some(&backend_probe),
    )
    .unwrap();

    assert_eq!(plans[0].pinned_gpu.as_ref().unwrap().backend_device, "HIP1");
}

#[test]
fn pinned_gpu_startup_preflight_keeps_detected_backend_without_resolved_flavor() {
    let mut gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    apply_backend_devices_for_flavor(&mut gpus, None);

    assert_eq!(gpus[0].backend_device.as_deref(), Some("CUDA0"));
}

#[test]
fn pinned_gpu_startup_preflight_requests_per_gpu_vram_metrics() {
    let metrics = pinned_startup_preflight_metrics();

    assert_eq!(metrics.len(), 4);
    assert!(metrics.contains(&hardware::Metric::GpuName));
    assert!(metrics.contains(&hardware::Metric::GpuFacts));
    assert!(metrics.contains(&hardware::Metric::VramBytes));
    assert!(metrics.contains(&hardware::Metric::IsSoc));
}

#[test]
fn skippy_telemetry_endpoint_enables_summary_without_debug() {
    let options = RuntimeOptions {
        skippy_metrics_otlp_grpc: Some("http://127.0.0.1:14317".to_string()),
        ..RuntimeOptions::default()
    };

    let telemetry = skippy_telemetry_options(&options);

    assert_eq!(
        telemetry.metrics_otlp_grpc.as_deref(),
        Some("http://127.0.0.1:14317")
    );
    assert_eq!(
        telemetry.level,
        skippy_server::telemetry::TelemetryLevel::Summary
    );
}

#[test]
fn skippy_telemetry_debug_keeps_debug_level_when_endpoint_is_set() {
    let options = RuntimeOptions {
        debug: true,
        skippy_metrics_otlp_grpc: Some("http://127.0.0.1:14317".to_string()),
        ..RuntimeOptions::default()
    };

    let telemetry = skippy_telemetry_options(&options);

    assert_eq!(
        telemetry.level,
        skippy_server::telemetry::TelemetryLevel::Debug
    );
}

#[test]
fn pinned_gpu_startup_preflight_unmatched_cli_models_bypass_config_gpu_id() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Ignored-Model".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
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
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: specs[0].gpu_id.clone(),
        config_model_id: specs[0].config_model_id.clone(),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(specs[0].gpu_id, None);
    assert!(!specs[0].resolve_pinned_gpu);
    assert_eq!(plans[0].gpu_id, None);
    assert_eq!(plans[0].pinned_gpu, None);
}

#[test]
fn pinned_gpu_startup_preflight_missing_gpu_id_fails_closed() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: None,
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: None,
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("missing configured gpu_id"));
}

#[test]
fn pinned_gpu_startup_preflight_stores_resolved_pinned_target_in_plan() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), Some("CUDA3"))];
    gpus[0].reserved_bytes = Some(500_000_000);

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    let pinned_gpu = plans[0].pinned_gpu.as_ref().unwrap();
    assert_eq!(pinned_gpu.index, 3);
    assert_eq!(pinned_gpu.stable_id, "uuid:GPU-123");
    assert_eq!(pinned_gpu.backend_device, "CUDA3");
    assert_eq!(pinned_gpu.vram_bytes, 24_000_000_000);
    assert_eq!(pinned_gpu.reserved_bytes, Some(500_000_000));
}

#[test]
fn pinned_gpu_startup_preflight_rejects_resolved_gpu_without_backend_device() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), None)];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("without a backend_device"));
}

#[test]
fn pinned_gpu_startup_preflight_unresolvable_gpu_id_fails_closed() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("did not match any available pinnable GPU"));
}

#[test]
fn bare_serve_without_models_starts_durable_daemon() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let startup_specs = Vec::new();

    assert!(!should_show_serve_config_help(
        Some(RuntimeSurface::Serve),
        &options,
        &startup_specs
    ));
}

#[test]
fn test_should_not_show_serve_config_help_when_models_are_present() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let startup_specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: None,
        cli_device_override: false,
        resolve_pinned_gpu: false,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];

    assert!(!should_show_serve_config_help(
        Some(RuntimeSurface::Serve),
        &options,
        &startup_specs
    ));
}

#[test]
fn test_should_not_show_serve_config_help_for_client_surface() {
    let options = runtime_options_for_test(&["mesh-llm", "--client"]);
    let startup_specs = Vec::new();

    assert!(!should_show_serve_config_help(
        Some(RuntimeSurface::Client),
        &options,
        &startup_specs
    ));
}

#[test]
fn test_should_not_show_serve_config_help_for_auto_serve_without_models() {
    let options = runtime_options_for_test(&["mesh-llm", "--auto"]);
    let startup_specs = Vec::new();

    assert!(!should_show_serve_config_help(
        Some(RuntimeSurface::Serve),
        &options,
        &startup_specs
    ));
}

#[test]
fn test_should_not_show_serve_config_help_for_join_serve_without_models() {
    let options = runtime_options_for_test(&["mesh-llm", "--join", "token"]);
    let startup_specs = Vec::new();

    assert!(!should_show_serve_config_help(
        Some(RuntimeSurface::Serve),
        &options,
        &startup_specs
    ));
}

#[test]
fn shared_mesh_modes_use_concurrency_preserving_resource_planning_profile() {
    assert_eq!(
        runtime_resource_planning_profile(&runtime_options_for_test(&["mesh-llm"])),
        RuntimeResourcePlanningProfile::DedicatedLocal
    );
    assert_eq!(
        runtime_resource_planning_profile(&runtime_options_for_test(&["mesh-llm", "--auto"])),
        RuntimeResourcePlanningProfile::SharedMesh
    );
    assert_eq!(
        runtime_resource_planning_profile(&runtime_options_for_test(&["mesh-llm", "--publish"])),
        RuntimeResourcePlanningProfile::SharedMesh
    );
    assert_eq!(
        runtime_resource_planning_profile(&runtime_options_for_test(&[
            "mesh-llm",
            "--discover",
            "lab",
        ])),
        RuntimeResourcePlanningProfile::SharedMesh
    );
    assert_eq!(
        runtime_resource_planning_profile(&runtime_options_for_test(&[
            "mesh-llm",
            "--join",
            "mesh-token",
        ])),
        RuntimeResourcePlanningProfile::SharedMesh
    );
}

// ---------------------------------------------------------------------------
// Per-model parallel (slots) resolution tests
// ---------------------------------------------------------------------------

/// Scenario 1: No global `gpu.parallel` set; a specific model entry has
/// `parallel = 1`. The model's override value must be applied correctly.
#[test]
fn per_model_parallel_override_applied_when_no_global() {
    let config_models = [ModelConfigEntry {
        model: "my-model".to_string(),
        mmproj: None,
        ctx_size: None,
        gpu_id: None,
        parallel: Some(1),
        cache_type_k: None,
        cache_type_v: None,
        batch: None,
        ubatch: None,
        flash_attention: None,
        ..Default::default()
    }];
    let gpu_config = GpuConfig::default(); // no parallel set

    let slots = resolve_model_parallel_slots(
        config_models
            .iter()
            .find(|model| model.model == "my-model")
            .and_then(|model| model.parallel),
        &gpu_config,
        4,
    );

    assert_eq!(
        slots, 1,
        "model-specific parallel=1 should win when no global"
    );
}

/// Scenario 2: Two models in config — only the second one specifies a
/// `parallel` value. The slot assignment must land on the correct model.
#[test]
fn per_model_parallel_applies_to_correct_model() {
    let config_models = [
        ModelConfigEntry {
            model: "model-a".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        },
        ModelConfigEntry {
            model: "model-b".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: Some(3),
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        },
    ];
    let gpu_config = GpuConfig::default();

    let slots_a = resolve_model_parallel_slots(
        config_models
            .iter()
            .find(|model| model.model == "model-a")
            .and_then(|model| model.parallel),
        &gpu_config,
        4,
    );
    assert_eq!(
        slots_a, 4,
        "model-a should get default 4 when it has no parallel entry"
    );

    let slots_b = resolve_model_parallel_slots(
        config_models
            .iter()
            .find(|model| model.model == "model-b")
            .and_then(|model| model.parallel),
        &gpu_config,
        4,
    );
    assert_eq!(slots_b, 3, "model-b should get its own parallel=3 override");
}

/// Scenario 3: Two models. First has NO parallel setting, second has
/// `parallel = 2`, and global `gpu.parallel = 3`. The first model should
/// fall through to the global (3), while the second uses its own (2).
#[test]
fn per_model_parallel_fallback_to_global_for_missing_entry() {
    let config_models = [
        ModelConfigEntry {
            model: "first".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        },
        ModelConfigEntry {
            model: "second".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: Some(2),
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        },
    ];
    let gpu_config = GpuConfig {
        assignment: GpuAssignment::Auto,
        parallel: Some(3), // global default
    };

    let slots_first = resolve_model_parallel_slots(
        config_models
            .iter()
            .find(|model| model.model == "first")
            .and_then(|model| model.parallel),
        &gpu_config,
        4,
    );
    assert_eq!(
        slots_first, 3,
        "missing model parallel should fall back to gpu.parallel=3"
    );

    let slots_second = resolve_model_parallel_slots(
        config_models
            .iter()
            .find(|model| model.model == "second")
            .and_then(|model| model.parallel),
        &gpu_config,
        4,
    );
    assert_eq!(
        slots_second, 2,
        "model-specific parallel=2 should win over global gpu.parallel=3"
    );
}

// ---------------------------------------------------------------------------
// Publication-state matrix (Issue #240)
// ---------------------------------------------------------------------------
