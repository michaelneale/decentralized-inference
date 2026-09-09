use crate::gpus::tune_apply::{PreparedTunePlan, apply_prepared_tune_plans};
use mesh_llm_config::{ConfigStore, parse_config_toml};
use tempfile::tempdir;
use toml_edit::DocumentMut;

use super::*;

#[test]
fn gpu_tune_apply_preserves_comments_and_writes_nested_fields() {
    let temp = tempdir().expect("tempdir should be created");
    let model_path = write_local_gguf_file(temp.path(), "configured.gguf");
    let canonical_model_path = model_path
        .canonicalize()
        .expect("fixture path should canonicalize");
    let raw_config = format!(
        "# keep header\nversion = 1\n\n[gpu]\nassignment = \"pinned\"\n\n[telemetry]\nservice_name = \"keep-me\"\n\n[defaults.model_fit]\nctx_size = 16384\nbatch = 384\n\n[defaults.hardware]\nfit_target_mib = 12288\n\n[[models]]\nmodel = \"{}\"\n# keep row comment\nctx_size = 8192\ngpu_id = \"pci:0000:00:00.0\"\n",
        canonical_model_path.display()
    );
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, raw_config).expect("fixture config should be written");
    let config = parse_config_toml(
        &std::fs::read_to_string(&config_path).expect("fixture config should be readable"),
    )
    .expect("fixture config should parse");
    let target = configured_target(&canonical_model_path, 0);
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    let store = ConfigStore::open(&config_path);
    let written = apply_prepared_tune_plans(&store, &[PreparedTunePlan::new(target, plan)])
        .expect("tune apply should succeed");

    assert_eq!(written, 1);
    let written_toml =
        std::fs::read_to_string(&config_path).expect("written config should be readable");
    assert!(written_toml.contains("# keep header"));
    assert!(written_toml.contains("# keep row comment"));
    assert!(written_toml.contains("service_name = \"keep-me\""));

    let (_, model_fit_and_rest) = written_toml
        .split_once("[models.model_fit]")
        .expect("model_fit section should be written");
    let (model_fit_section, hardware_and_rest) = model_fit_and_rest
        .split_once("[models.hardware]")
        .expect("hardware section should be written");
    let prefix = written_toml
        .split_once("[models.model_fit]")
        .expect("model_fit section should be present")
        .0;
    let hardware_section = hardware_and_rest;

    assert!(prefix.contains("ctx_size = 8192"));
    assert!(prefix.contains("gpu_id = \"pci:0000:00:00.0\""));
    assert!(
        !prefix
            .lines()
            .any(|line| line.trim() == "cache_type_k = \"q8_0\"")
    );
    assert!(
        !prefix
            .lines()
            .any(|line| line.trim() == "cache_type_v = \"q8_0\"")
    );
    assert!(
        !prefix
            .lines()
            .any(|line| line.trim() == "flash_attention = \"enabled\"")
    );
    assert!(!prefix.lines().any(|line| line.trim() == "ubatch = 512"));
    assert!(
        model_fit_section
            .lines()
            .any(|line| line.trim() == "cache_type_k = \"q8_0\"")
    );
    assert!(
        model_fit_section
            .lines()
            .any(|line| line.trim() == "cache_type_v = \"q8_0\"")
    );
    assert!(
        model_fit_section
            .lines()
            .any(|line| line.trim() == "flash_attention = \"enabled\"")
    );
    assert!(
        model_fit_section
            .lines()
            .any(|line| line.trim() == "ubatch = 512")
    );
    assert!(
        !model_fit_section
            .lines()
            .any(|line| line.trim_start().starts_with("ctx_size ="))
    );
    assert!(
        !model_fit_section
            .lines()
            .any(|line| line.trim_start().starts_with("batch ="))
    );
    assert!(
        hardware_section
            .lines()
            .any(|line| line.trim() == "gpu_layers = -1")
    );
    assert!(
        !hardware_section
            .lines()
            .any(|line| line.trim_start().starts_with("fit_target_mib ="))
    );

    let loaded = store
        .load()
        .expect("written config should validate through ConfigStore");
    assert_eq!(loaded.models.len(), 1);
}

#[test]
fn gpu_tune_apply_appends_unconfigured_local_target_with_canonical_path() {
    let temp = tempdir().expect("tempdir should be created");
    let model_path = write_local_gguf_file(temp.path(), "append-only.gguf");
    let canonical_model_path = model_path
        .canonicalize()
        .expect("fixture path should canonicalize");
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "# append test\nversion = 1\n")
        .expect("fixture config should be written");
    let target = appended_target(&canonical_model_path);
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &mesh_llm_config::MeshConfig::default(),
        target: &target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    let store = ConfigStore::open(&config_path);
    let written = apply_prepared_tune_plans(&store, &[PreparedTunePlan::new(target, plan)])
        .expect("append apply should succeed");

    assert_eq!(written, 1);
    let edited = std::fs::read_to_string(&config_path)
        .expect("written config should be readable")
        .parse::<DocumentMut>()
        .expect("written config should remain valid TOML")
        .to_string();
    assert!(edited.contains(&format!("model = \"{}\"", canonical_model_path.display())));
    assert!(edited.contains("[models.model_fit]"));
    assert!(edited.contains("cache_type_k = \"q8_0\""));
}

#[test]
fn gpu_tune_apply_missing_preserves_legacy_manual_model_fit_fields() {
    let temp = tempdir().expect("tempdir should be created");
    let model_path = write_local_gguf_file(temp.path(), "legacy-manual.gguf");
    let canonical_model_path = model_path
        .canonicalize()
        .expect("fixture path should canonicalize");
    let raw_config = format!(
        "version = 1\n\n[[models]]\nmodel = \"{}\"\nctx_size = 8192\nbatch = 256\nubatch = 64\ncache_type_k = \"f16\"\ncache_type_v = \"f16\"\nflash_attention = \"disabled\"\n",
        canonical_model_path.display()
    );
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, &raw_config).expect("fixture config should be written");
    let config = parse_config_toml(
        &std::fs::read_to_string(&config_path).expect("fixture config should be readable"),
    )
    .expect("fixture config should parse");
    let target = configured_target(&canonical_model_path, 0);
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    let store = ConfigStore::open(&config_path);
    let written = apply_prepared_tune_plans(&store, &[PreparedTunePlan::new(target, plan)])
        .expect("tune apply should succeed");

    assert_eq!(written, 1);
    let edited = std::fs::read_to_string(&config_path).expect("written config should be readable");
    assert!(edited.contains("ctx_size = 8192"));
    assert!(edited.contains("batch = 256"));
    assert!(edited.contains("ubatch = 64"));
    assert!(edited.contains("cache_type_k = \"f16\""));
    assert!(edited.contains("cache_type_v = \"f16\""));
    assert!(edited.contains("flash_attention = \"disabled\""));
    assert!(edited.contains("[models.hardware]"));
    assert!(!edited.contains("[models.model_fit]"));
}

#[test]
fn gpu_tune_replace_existing_writes_nested_recommendations_over_legacy_manual_fields() {
    let temp = tempdir().expect("tempdir should be created");
    let model_path = write_local_gguf_file(temp.path(), "legacy-replace.gguf");
    let canonical_model_path = model_path
        .canonicalize()
        .expect("fixture path should canonicalize");
    let raw_config = format!(
        "version = 1\n\n[[models]]\nmodel = \"{}\"\nctx_size = 8192\nbatch = 256\nubatch = 64\ncache_type_k = \"f16\"\ncache_type_v = \"f16\"\nflash_attention = \"disabled\"\n",
        canonical_model_path.display()
    );
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, &raw_config).expect("fixture config should be written");
    let config = parse_config_toml(
        &std::fs::read_to_string(&config_path).expect("fixture config should be readable"),
    )
    .expect("fixture config should parse");
    let target = configured_target(&canonical_model_path, 0);
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ReplaceExisting,
        config: &config,
        target: &target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    let store = ConfigStore::open(&config_path);
    let written = apply_prepared_tune_plans(&store, &[PreparedTunePlan::new(target, plan)])
        .expect("replace-existing apply should succeed");

    assert_eq!(written, 1);
    let edited = std::fs::read_to_string(&config_path).expect("written config should be readable");
    assert!(edited.contains("[models.model_fit]"));
    assert!(edited.contains("cache_type_k = \"q8_0\""));
    assert!(edited.contains("cache_type_v = \"q8_0\""));
    assert!(edited.contains("flash_attention = \"enabled\""));
    let loaded = store
        .load()
        .expect("written config should validate through ConfigStore");
    let model_fit = loaded.models[0]
        .model_fit
        .as_ref()
        .expect("replace-existing should write nested model_fit overrides");
    assert_eq!(model_fit.cache_type_k.as_deref(), Some("q8_0"));
    assert_eq!(model_fit.cache_type_v.as_deref(), Some("q8_0"));
    assert_eq!(model_fit.ctx_size, Some(65_536));
    assert_eq!(model_fit.batch, Some(512));
    assert_eq!(model_fit.ubatch, Some(512));
}
