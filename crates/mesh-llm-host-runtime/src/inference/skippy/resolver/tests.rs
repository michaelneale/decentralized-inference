use super::test_support::*;
use super::*;
use crate::inference::skippy::SkippyTelemetryOptions;
use crate::plugin::{
    MeshConfig, ModelConfigDefaults, ModelConfigEntry, ReasoningBudget, RequestDefaultsConfig,
};
use anyhow::Context as _;
use serde_json::Value;
use skippy_protocol::{LoadMode, StageKvCacheMode, StageKvCachePayload};
use skippy_server::{EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat};
use std::{io::Write, path::Path};
use tempfile::NamedTempFile;

fn resolve_qwen_config_with_request_defaults(
    mesh_config: &MeshConfig,
    model_path: &Path,
    request_defaults: Option<&RequestDefaultsConfig>,
) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path,
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults,
        package_generation: None,
        compact_meta: None,
    })
    .expect("qwen config should resolve")
}

fn assert_request_override_keeps_load_time_config(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    assert_eq!(without_request.model_fit, with_request.model_fit);
    assert_eq!(without_request.hardware, with_request.hardware);
    assert_eq!(without_request.skippy, with_request.skippy);
}

fn assert_stage_configs_match_for_request_override(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    let baseline_stage = without_request
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("baseline stage config should build");
    let override_stage = with_request
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("override stage config should build");

    assert_eq!(baseline_stage.model_id, override_stage.model_id);
    assert_eq!(baseline_stage.model_path, override_stage.model_path);
    assert_eq!(baseline_stage.ctx_size, override_stage.ctx_size);
    assert_eq!(baseline_stage.lane_count, override_stage.lane_count);
    assert_eq!(baseline_stage.n_batch, override_stage.n_batch);
    assert_eq!(baseline_stage.n_ubatch, override_stage.n_ubatch);
    assert_eq!(baseline_stage.n_gpu_layers, override_stage.n_gpu_layers);
    assert_eq!(baseline_stage.cache_type_k, override_stage.cache_type_k);
    assert_eq!(baseline_stage.cache_type_v, override_stage.cache_type_v);
    assert_eq!(
        baseline_stage.flash_attn_type,
        override_stage.flash_attn_type
    );
    assert_eq!(
        baseline_stage.selected_device,
        override_stage.selected_device
    );
    assert_eq!(baseline_stage.load_mode, override_stage.load_mode);
}

fn assert_openai_args_use_request_time_defaults(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    let baseline_openai = without_request
        .to_embedded_openai_args(4096, true)
        .expect("baseline openai args should build");
    let override_openai = with_request
        .to_embedded_openai_args(4096, true)
        .expect("override openai args should build");

    assert_eq!(baseline_openai.default_max_tokens, 128);
    assert_eq!(override_openai.default_max_tokens, 32);
}

#[test]
fn resolver_applies_precedence_and_keeps_request_defaults_out_of_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
ctx_size = 8192
batch = 512
ubatch = 128
cache_type_v = "q8_0"

[defaults.hardware]
device = "CUDA0"
mmap = true
mlock = false

[defaults.throughput]
parallel = 2

[defaults.request_defaults]
temperature = 0.2
max_tokens = 128

[[models]]
model = "ggml-org/gemma-3-270m-it-GGUF:Q8_0"

[models.model_fit]
ctx_size = 16384
batch = 1024
cache_type_k = "f16"

[models.hardware]
device = "CUDA1"
mmap = false
mlock = true

[models.throughput]
parallel = 3

[models.request_defaults]
temperature = 0.4
"#,
    );
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.7),
        max_tokens: Some(256),
        reasoning_budget: Some(ReasoningBudget::Integer(512)),
        ..Default::default()
    };
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "ggml-org/gemma-3-270m-it-GGUF:Q8_0",
        model_path: model_file.path(),
        model_bytes: 8 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(16 * 1024 * 1024 * 1024),
        request_defaults: Some(&request_defaults),
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.ctx_size, 16384);
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 128);
    assert_eq!(resolved.hardware.device.as_deref(), Some("CUDA1"));
    assert_eq!(resolved.hardware.mmap, Some(false));
    assert!(resolved.hardware.mlock);
    assert_eq!(resolved.throughput.parallel, 3);
    assert_eq!(resolved.request_defaults.max_tokens, 256);
    assert_eq!(resolved.request_defaults.temperature, Some(0.7));
    assert_eq!(
        resolved.request_defaults.reasoning_budget,
        Some(ReasoningBudget::Integer(512))
    );

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    assert_eq!(stage_config.mmap, Some(false));
    assert!(stage_config.mlock);
    let serialized: Value = serde_json::to_value(&stage_config).expect("stage config json");
    let object = serialized.as_object().expect("stage config object");
    assert!(!object.contains_key("request_defaults"));
    assert!(!object.contains_key("temperature"));
    assert_eq!(object.get("ctx_size").and_then(Value::as_u64), Some(16384));
}

#[test]
fn mutually_exclusive_request_defaults_stop_lower_precedence_fill_in() {
    let global = ModelConfigDefaults {
        request_defaults: Some(RequestDefaultsConfig {
            chat_template: Some("global-template".to_string()),
            grammar: Some(toml::Value::String("global-grammar".to_string())),
            ..RequestDefaultsConfig::default()
        }),
        ..ModelConfigDefaults::default()
    };
    let model = ModelConfigEntry {
        request_defaults: Some(RequestDefaultsConfig {
            chat_template_file: Some("model-template.jinja".to_string()),
            json_schema: Some(toml::Value::String("model-schema".to_string())),
            ..RequestDefaultsConfig::default()
        }),
        ..ModelConfigEntry::default()
    };
    let request = RequestDefaultsConfig {
        chat_template: Some("request-template".to_string()),
        json_schema: Some(toml::Value::String("request-schema".to_string())),
        ..RequestDefaultsConfig::default()
    };

    let resolved = super::request_defaults::resolve_request_defaults(
        Some(&global),
        Some(&model),
        Some(&request),
    )
    .expect("request defaults should resolve");
    assert_eq!(resolved.chat_template.as_deref(), Some("request-template"));
    assert_eq!(resolved.chat_template_file, None);
    assert_eq!(
        resolved.grammar, None,
        "the request layer owns the structured-output slot"
    );
    assert_eq!(
        resolved.json_schema,
        Some(toml::Value::String("request-schema".to_string()))
    );

    let resolved =
        super::request_defaults::resolve_request_defaults(Some(&global), Some(&model), None)
            .expect("request defaults should resolve");
    assert_eq!(resolved.chat_template, None);
    assert_eq!(
        resolved.chat_template_file.as_deref(),
        Some("model-template.jinja")
    );
    assert_eq!(resolved.grammar, None);
    assert_eq!(
        resolved.json_schema,
        Some(toml::Value::String("model-schema".to_string()))
    );
}

#[test]
fn multimodal_native_extensions_reach_stage_config_with_model_precedence() {
    let model = temp_model_file();
    let config = parse_config(
        r#"
[defaults.multimodal]
mmproj_offload = false
media_marker = "<defaults-media>"
image_min_tokens = 16
image_max_tokens = 1024
batch_max_tokens = 256
glm_dsa_policy = "auto"
generation_signal_window = 8

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.multimodal]
mmproj_offload = true
media_marker = "<model-media>"
image_min_tokens = 64
image_max_tokens = 2048
batch_max_tokens = 512
glm_dsa_policy = "v1"
generation_signal_window = 24
"#,
    );

    let resolved = resolve_qwen_config_with_request_defaults(&config, model.path(), None);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage = serde_json::to_value(stage).expect("stage config should serialize");

    assert_eq!(stage["projector_use_gpu"], true);
    assert_eq!(stage["media_marker"], "<model-media>");
    assert_eq!(stage["image_min_tokens"], 64);
    assert_eq!(stage["image_max_tokens"], 2048);
    assert_eq!(stage["batch_max_tokens"], 512);
    assert_eq!(stage["glm_dsa_policy"], "v1");
    assert_eq!(stage["generation_signal_window"], 24);
}

#[test]
fn multimodal_native_extensions_preserve_auto_and_omission() {
    let model = temp_model_file();
    let config = parse_config(
        r#"
[defaults.multimodal]
mmproj_offload = "auto"
glm_dsa_policy = "auto"

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"
"#,
    );

    let resolved = resolve_qwen_config_with_request_defaults(&config, model.path(), None);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage = serde_json::to_value(stage).expect("stage config should serialize");

    assert!(stage["projector_use_gpu"].is_null());
    assert!(stage["media_marker"].is_null());
    assert!(stage["image_min_tokens"].is_null());
    assert!(stage["image_max_tokens"].is_null());
    assert!(stage["batch_max_tokens"].is_null());
    assert_eq!(stage["glm_dsa_policy"], "auto");
    assert!(stage["generation_signal_window"].is_null());
}

#[test]
fn deprecated_image_marker_is_rejected_during_static_validation() {
    let config = parse_config(
        r#"
[defaults.multimodal]
image_marker = "<image>"
"#,
    );

    let error = mesh_llm_config::validate_config(&config)
        .expect_err("deprecated image_marker must be rejected before model loading");

    assert_eq!(
        error.to_string(),
        "defaults.multimodal.image_marker is not supported because mtmd removed custom image markers; use defaults.multimodal.media_marker"
    );
}

#[test]
fn resolver_carries_memory_load_controls_into_single_stage_options() {
    let mesh_config = parse_config(
        r#"
[defaults.hardware]
mmap = false
mlock = true
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("config should resolve");

    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");

    assert_eq!(resolved.hardware.mmap, Some(false));
    assert!(resolved.hardware.mlock);
    assert_eq!(load_options.mmap, Some(false));
    assert!(load_options.mlock);
}

#[test]
fn resolver_macro_expands_kv_cache_tuning_profile_and_safety_margin() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"

[defaults.hardware]
safety_margin_gb = 1.5

[defaults.throughput]
tuning_profile = "throughput"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(12 * 1024 * 1024 * 1024),
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
    assert_eq!(resolved.model_fit.kv_offload, "true");
    assert_eq!(resolved.throughput.tuning_profile, "throughput");
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 256);
    assert_eq!(resolved.throughput.parallel, 2);
    assert_eq!(resolved.throughput.continuous_batching, "true");
    assert_eq!(resolved.hardware.fit_target_mib, Some(10_752));
}

#[test]
fn resolver_treats_auto_cache_type_as_policy_selected_cache_type() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"
cache_type_k = "auto"
cache_type_v = "auto"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
}

#[test]
fn resolver_treats_auto_cache_type_case_insensitively() {
    // Test uppercase "AUTO"
    let mesh_config_upper = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"
cache_type_k = "AUTO"
cache_type_v = "AUTO"
"#,
    );

    let resolved_upper = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_upper,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved_upper.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved_upper.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved_upper.model_fit.cache_type_v, "q8_0");

    // Test mixed-case "Auto"
    let mesh_config_mixed = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"
cache_type_k = "Auto"
cache_type_v = "Auto"
"#,
    );

    let resolved_mixed = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_mixed,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved_mixed.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved_mixed.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved_mixed.model_fit.cache_type_v, "q8_0");

    // Test mixed-case "AuTo"
    let mesh_config_mixed2 = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"
cache_type_k = "AuTo"
cache_type_v = "AuTo"
"#,
    );

    let resolved_mixed2 = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_mixed2,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved_mixed2.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved_mixed2.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved_mixed2.model_fit.cache_type_v, "q8_0");
}

#[test]
fn per_model_kv_macro_beats_global_explicit_cache_fields_unless_model_explicit_exists() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
cache_type_k = "f16"
cache_type_v = "f16"
kv_offload = false

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
kv_cache_policy = "saver"
cache_type_v = "q4_0"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q4_0");
    assert_eq!(resolved.model_fit.kv_offload, "true");
}

#[test]
fn kv_unified_true_resolves_and_reaches_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
kv_unified = true
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("kv_unified = true should resolve instead of bailing");

    resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build with kv_unified = true");
}

#[test]
fn swa_full_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let mesh_config_true = parse_config(
        r#"
[defaults.model_fit]
swa_full = true
"#,
    );
    let resolved_true = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_true,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let mesh_config_false = parse_config(
        r#"
[defaults.model_fit]
swa_full = false
"#,
    );
    let resolved_false = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_false,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let stage_true = resolved_true
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage_false = resolved_false
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");

    assert_ne!(
        stage_config_stable_json(&stage_true),
        stage_config_stable_json(&stage_false),
        "swa_full=true and swa_full=false must reach the stage config differently \
         instead of being silently dropped"
    );
}

#[test]
fn kv_offload_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let mesh_config_true = parse_config(
        r#"
[defaults.model_fit]
kv_offload = true
"#,
    );
    let resolved_true = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_true,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let mesh_config_false = parse_config(
        r#"
[defaults.model_fit]
kv_offload = false
"#,
    );
    let resolved_false = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_false,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let stage_true = resolved_true
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage_false = resolved_false
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");

    assert_ne!(
        stage_config_stable_json(&stage_true),
        stage_config_stable_json(&stage_false),
        "kv_offload=true and kv_offload=false must reach the stage config differently \
         instead of being silently dropped"
    );
}

#[test]
fn hardware_repack_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let mesh_config_true = parse_config(
        r#"
[defaults.hardware]
repack = true
"#,
    );
    let resolved_true = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_true,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let mesh_config_false = parse_config(
        r#"
[defaults.hardware]
repack = false
"#,
    );
    let resolved_false = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_false,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let stage_true = resolved_true
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage_false = resolved_false
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");

    assert_ne!(
        stage_config_stable_json(&stage_true),
        stage_config_stable_json(&stage_false),
        "hardware.repack=true and hardware.repack=false must reach the stage config \
         differently instead of being silently dropped"
    );
}

#[test]
fn hardware_repack_per_model_override_beats_default() {
    let model_file = temp_model_file();
    const MODEL_ID: &str = "ggml-org/gemma-3-270m-it-GGUF:Q8_0";

    let mesh_config_override = parse_config(&format!(
        r#"
[defaults.hardware]
repack = false

[[models]]
model = "{MODEL_ID}"

[models.hardware]
repack = true
"#
    ));
    let resolved_override = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_override,
        model_id: MODEL_ID,
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let mesh_config_default_only = parse_config(&format!(
        r#"
[defaults.hardware]
repack = false

[[models]]
model = "{MODEL_ID}"
"#
    ));
    let resolved_default_only = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config_default_only,
        model_id: MODEL_ID,
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let stage_override = resolved_override
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let stage_default_only = resolved_default_only
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");

    assert_ne!(
        stage_config_stable_json(&stage_override),
        stage_config_stable_json(&stage_default_only),
        "a per-model hardware.repack override must reach the stage config differently \
         than the defaults.hardware value it overrides"
    );
}

/// Resolves `toml_str` for `model_id` and returns the resulting stage config
/// as stable (time-varying-identifier-free) JSON, so hardware field wiring
/// tests can assert on materially different downstream state in a few lines.
fn hardware_stage_json(toml_str: &str, model_id: &str, model_path: &Path) -> Value {
    let mesh_config = parse_config(toml_str);
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id,
        model_path,
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap_or_else(|error| panic!("config should resolve: {error}"));
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    stage_config_stable_json(&stage)
}

const HARDWARE_TEST_MODEL_ID: &str = "ggml-org/gemma-3-270m-it-GGUF:Q8_0";

#[test]
fn hardware_op_offload_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\nop_offload = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\nop_offload = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.op_offload=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_op_offload_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nop_offload = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.op_offload override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_op_offload_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nop_offload = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nop_offload = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nop_offload = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.op_offload override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_no_host_buffer_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\nno_host_buffer = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\nno_host_buffer = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.no_host_buffer=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_no_host_buffer_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nno_host_buffer = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.no_host_buffer override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_no_host_buffer_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nno_host_buffer = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nno_host_buffer = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nno_host_buffer = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.no_host_buffer override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_check_tensors_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\ncheck_tensors = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\ncheck_tensors = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.check_tensors=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_checkpoint_quantization_reaches_stage_config() {
    let model_file = temp_model_file();
    let stage = hardware_stage_json(
        "[defaults.hardware]\ncheckpoint_quantization = \"Q4_K_M\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );

    assert_eq!(
        stage.get("checkpoint_quantization"),
        Some(&Value::String("Q4_K_M".to_string()))
    );
}

/// Exercises the same Mesh configuration -> resolver -> stage-config ->
/// skippy-server model-open and inference path used by the embedded host. CI
/// supplies a pinned Hugging Face checkpoint directory rather than checking
/// model bytes into the repository.
#[test]
#[ignore = "requires SKIPPY_SAFETENSORS_SMOKE_DIR with a complete checkpoint"]
fn safetensors_checkpoint_reaches_mesh_host_runtime() -> anyhow::Result<()> {
    let checkpoint = std::env::var_os("SKIPPY_SAFETENSORS_SMOKE_DIR")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("SKIPPY_SAFETENSORS_SMOKE_DIR is not set"))?;
    let quantization = std::env::var("SKIPPY_SAFETENSORS_SMOKE_QUANTIZATION")
        .unwrap_or_else(|_| "preserve".to_string());
    let gpu_layers = std::env::var("SKIPPY_SAFETENSORS_SMOKE_GPU_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<i32>()
        .context("parse SKIPPY_SAFETENSORS_SMOKE_GPU_LAYERS")?;
    let mut synthetic_imatrix = None;
    let checkpoint_imatrix = match std::env::var("SKIPPY_SAFETENSORS_SMOKE_IMATRIX").ok() {
        Some(value) if value == "synthetic" => {
            let mut file = NamedTempFile::new().context("create synthetic importance matrix")?;
            let direct =
                skippy_model::gguf_writer::DirectCheckpoint::open(&checkpoint, 1024 * 1024)?;
            let layout = direct.imatrix_layout()?;
            let entry_count =
                i32::try_from(layout.len()).context("imatrix entry count exceeds i32")?;
            file.write_all(&entry_count.to_le_bytes())?;
            for entry in layout {
                let name = entry.name.as_bytes();
                file.write_all(&i32::try_from(name.len())?.to_le_bytes())?;
                file.write_all(name)?;
                file.write_all(&1_i32.to_le_bytes())?;
                file.write_all(&i32::try_from(entry.value_count)?.to_le_bytes())?;
                for _ in 0..entry.value_count {
                    file.write_all(&1_f32.to_le_bytes())?;
                }
            }
            file.flush()?;
            let path = file.path().to_string_lossy().into_owned();
            synthetic_imatrix = Some(file);
            Some(path)
        }
        Some(path) => Some(path),
        None => None,
    };
    let quantization_toml = toml::Value::String(quantization.clone()).to_string();
    let imatrix_toml = checkpoint_imatrix
        .as_ref()
        .map(|path| {
            format!(
                "checkpoint_imatrix = {}\n",
                toml::Value::String(path.clone())
            )
        })
        .unwrap_or_default();
    let mesh_config = parse_config(&format!(
        "[defaults.model_fit]\nctx_size = 128\nbatch = 128\nubatch = 128\n\
         \n[defaults.hardware]\ngpu_layers = {gpu_layers}\ncheckpoint_quantization = {quantization_toml}\n{imatrix_toml}"
    ));
    let identity =
        crate::inference::skippy::synthetic_direct_gguf_package("safetensors-smoke", &checkpoint)?;
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "safetensors-smoke",
        model_path: &checkpoint,
        model_bytes: identity.source_model_bytes,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })?;
    let stage = resolved.to_stage_config(Some(identity), LoadMode::RuntimeSlice)?;

    assert_eq!(stage.model_path.as_deref(), checkpoint.to_str());
    assert_eq!(
        stage.checkpoint_quantization.as_deref(),
        Some(quantization.as_str())
    );
    let canonical_checkpoint_imatrix = checkpoint_imatrix
        .as_deref()
        .map(std::fs::canonicalize)
        .transpose()?
        .map(|path| path.to_string_lossy().into_owned());
    assert_eq!(
        stage.checkpoint_imatrix.as_deref(),
        canonical_checkpoint_imatrix.as_deref()
    );
    if checkpoint_imatrix.is_some() {
        anyhow::ensure!(
            stage.checkpoint_imatrix_sha256.is_some(),
            "importance matrix digest was not included in stage identity"
        );
    }
    let runtime = skippy_server::runtime_state::load_runtime(&stage)?
        .ok_or_else(|| anyhow::anyhow!("Mesh host did not open the checkpoint"))?;
    let mut runtime = runtime
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokens = runtime.model.tokenize("Hello from Mesh", true)?;
    anyhow::ensure!(
        !tokens.is_empty(),
        "checkpoint tokenizer returned no tokens"
    );
    let first = runtime.prefill_chunked_sampled("safetensors-smoke", &tokens, None)?;
    anyhow::ensure!(first >= 0, "sampled prefill returned invalid token {first}");
    let second = runtime.decode("safetensors-smoke", first)?;
    anyhow::ensure!(second >= 0, "decode returned invalid token {second}");
    drop(synthetic_imatrix);
    Ok(())
}

#[test]
fn hardware_check_tensors_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ncheck_tensors = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.check_tensors override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_check_tensors_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ncheck_tensors = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ncheck_tensors = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ncheck_tensors = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.check_tensors override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_direct_io_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\ndirect_io = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\ndirect_io = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.direct_io=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_direct_io_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ndirect_io = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.direct_io override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_direct_io_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ndirect_io = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ndirect_io = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ndirect_io = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.direct_io override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_main_gpu_variants_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_zero = hardware_stage_json(
        "[defaults.hardware]\nmain_gpu = 0\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_one = hardware_stage_json(
        "[defaults.hardware]\nmain_gpu = 1\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json("", HARDWARE_TEST_MODEL_ID, model_file.path());
    assert_ne!(
        stage_zero, stage_one,
        "hardware.main_gpu=0 and =1 must reach the stage config differently"
    );
    assert_ne!(
        stage_zero, stage_unset,
        "an explicit hardware.main_gpu=0 must reach the stage config differently than \
         leaving it unset (auto)"
    );
}

#[test]
fn hardware_main_gpu_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nmain_gpu = 2\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.main_gpu override with no defaults.hardware value set must \
         reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_main_gpu_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nmain_gpu = 0\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nmain_gpu = 3\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nmain_gpu = 0\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.main_gpu override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_split_mode_variants_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_layer = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"layer\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_row = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"row\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_none = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"none\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_auto = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"auto\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_layer, stage_row,
        "hardware.split_mode=\"layer\" and =\"row\" must reach the stage config differently"
    );
    assert_ne!(
        stage_none, stage_auto,
        "hardware.split_mode=\"none\" and =\"auto\" must reach the stage config differently"
    );
}

#[test]
fn hardware_split_mode_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nsplit_mode = \"row\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.split_mode override with no defaults.hardware value set must \
         reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_split_mode_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nsplit_mode = \"layer\"\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nsplit_mode = \"row\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nsplit_mode = \"layer\"\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.split_mode override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_split_mode_rejects_invalid_value() {
    let model_file = temp_model_file();
    let mesh_config = parse_config("[defaults.hardware]\nsplit_mode = \"bogus\"\n");
    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: HARDWARE_TEST_MODEL_ID,
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect_err("an unrecognized hardware.split_mode value should be rejected");
    assert!(error.to_string().contains("split_mode"));
}

/// JSON representation of a stage config with time-varying identifiers
/// removed, so two configs built moments apart can be compared for
/// deterministic content differences.
fn stage_config_stable_json(config: &skippy_protocol::StageConfig) -> Value {
    let mut json = serde_json::to_value(config).expect("stage config json");
    if let Some(object) = json.as_object_mut() {
        object.remove("run_id");
        object.remove("topology_id");
    }
    json
}
#[test]
fn cache_idle_slots_true_resolves_and_reaches_stage_config() {
    let model_file = temp_model_file();
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
cache_idle_slots = 3
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("cache_idle_slots should resolve instead of bailing");

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    assert_eq!(stage_config.cache_idle_slots, Some(3));
}

#[test]
fn cache_idle_slots_defaults_layer_and_per_model_override_wins() {
    let model_file = temp_model_file();
    let global_two = parse_config(
        r#"
[defaults.model_fit]
cache_idle_slots = 2
"#,
    );
    let resolved_global = resolve_with_config_and_model_path(&global_two, model_file.path());
    assert_eq!(resolved_global.model_fit.cache_idle_slots, Some(2));
    let stage_global = resolved_global
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    assert_eq!(stage_global.cache_idle_slots, Some(2));

    let global_two_model_seven = parse_config(
        r#"
[defaults.model_fit]
cache_idle_slots = 2

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
cache_idle_slots = 7
"#,
    );
    let resolved_override = resolve_with_config(&global_two_model_seven);
    assert_eq!(resolved_override.model_fit.cache_idle_slots, Some(7));
    assert_ne!(
        resolved_global.model_fit.cache_idle_slots,
        resolved_override.model_fit.cache_idle_slots
    );

    let unset = parse_config("");
    let resolved_unset = resolve_with_config(&unset);
    assert_eq!(resolved_unset.model_fit.cache_idle_slots, None);
}

fn resolve_with_config(mesh_config: &MeshConfig) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap()
}

fn resolve_with_config_and_model_path(
    mesh_config: &MeshConfig,
    model_path: &Path,
) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path,
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap()
}

#[test]
fn kv_offload_resolved_reaches_model_load_options_via_kv_cache_policy() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"
"#,
    );

    let model_file = temp_model_file();
    let resolved = resolve_with_config_and_model_path(&mesh_config, model_file.path());
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");

    assert_eq!(resolved.model_fit.kv_offload_resolved, Some(true));
    assert_eq!(load_options.kv_offload, Some(true));
}

#[test]
fn kv_unified_defaults_layer_and_per_model_override_wins() {
    let global_true = parse_config(
        r#"
[defaults.model_fit]
kv_unified = true
"#,
    );
    let model_file = temp_model_file();
    let resolved_global = resolve_with_config_and_model_path(&global_true, model_file.path());
    let load_options_global = resolved_global
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert_eq!(resolved_global.model_fit.kv_unified, Some(true));
    assert_eq!(load_options_global.kv_unified, Some(true));

    let global_true_model_false = parse_config(
        r#"
[defaults.model_fit]
kv_unified = true

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
kv_unified = false
"#,
    );
    let resolved_override = resolve_with_config(&global_true_model_false);
    assert_eq!(resolved_override.model_fit.kv_unified, Some(false));

    let unset = parse_config("");
    let resolved_unset = resolve_with_config(&unset);
    assert_eq!(resolved_unset.model_fit.kv_unified, None);
}

#[test]
fn swa_full_defaults_layer_and_per_model_override_wins() {
    let global_true = parse_config(
        r#"
[defaults.model_fit]
swa_full = true
"#,
    );
    let model_file = temp_model_file();
    let resolved_global = resolve_with_config_and_model_path(&global_true, model_file.path());
    let load_options_global = resolved_global
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert_eq!(resolved_global.model_fit.swa_full, Some(true));
    assert_eq!(load_options_global.swa_full, Some(true));

    let global_true_model_false = parse_config(
        r#"
[defaults.model_fit]
swa_full = true

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
swa_full = false
"#,
    );
    let resolved_override = resolve_with_config(&global_true_model_false);
    assert_eq!(resolved_override.model_fit.swa_full, Some(false));

    let unset = parse_config("");
    let resolved_unset = resolve_with_config(&unset);
    assert_eq!(resolved_unset.model_fit.swa_full, None);
}

#[test]
fn per_model_throughput_macro_beats_global_explicit_fields_unless_model_explicit_exists() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
batch = 64
ubatch = 32

[defaults.throughput]
parallel = 7
continuous_batching = false

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
ubatch = 999

[models.throughput]
tuning_profile = "throughput"
parallel = 11
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.throughput.tuning_profile, "throughput");
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 999);
    assert_eq!(resolved.throughput.parallel, 11);
    assert_eq!(resolved.throughput.continuous_batching, "true");
}

#[test]
fn continuous_batching_reaches_embedded_scheduler_boundary() {
    let enabled = parse_config(
        r#"
[defaults.throughput]
continuous_batching = true
"#,
    );
    let disabled = parse_config(
        r#"
[defaults.throughput]
continuous_batching = false
"#,
    );

    let enabled_args = resolve_with_config(&enabled)
        .to_embedded_openai_args(4096, true)
        .expect("enabled OpenAI args");
    let disabled_args = resolve_with_config(&disabled)
        .to_embedded_openai_args(4096, true)
        .expect("disabled OpenAI args");
    let automatic_args = resolve_with_config(&parse_config(""))
        .to_embedded_openai_args(4096, true)
        .expect("automatic OpenAI args");

    let enabled_debug = format!("{enabled_args:?}");
    let disabled_debug = format!("{disabled_args:?}");
    let automatic_debug = format!("{automatic_args:?}");

    assert_ne!(
        enabled_debug, disabled_debug,
        "the embedded scheduler boundary must retain continuous_batching"
    );
    assert!(enabled_debug.contains("continuous_batching: true"));
    assert!(disabled_debug.contains("continuous_batching: false"));
    assert!(automatic_debug.contains("continuous_batching: true"));
}

#[test]
fn request_overrides_change_request_time_defaults_without_mutating_load_time_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
ctx_size = 4096

[defaults.request_defaults]
temperature = 0.2
max_tokens = 128
"#,
    );
    let model_file = temp_model_file();
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.9),
        max_tokens: Some(32),
        ..Default::default()
    };
    let without_request =
        resolve_qwen_config_with_request_defaults(&mesh_config, model_file.path(), None);
    let with_request = resolve_qwen_config_with_request_defaults(
        &mesh_config,
        model_file.path(),
        Some(&request_defaults),
    );

    assert_request_override_keeps_load_time_config(&without_request, &with_request);
    assert_eq!(without_request.request_defaults.temperature, Some(0.2));
    assert_eq!(with_request.request_defaults.temperature, Some(0.9));
    assert_eq!(without_request.request_defaults.max_tokens, 128);
    assert_eq!(with_request.request_defaults.max_tokens, 32);
    assert_stage_configs_match_for_request_override(&without_request, &with_request);
    assert_openai_args_use_request_time_defaults(&without_request, &with_request);
}

#[test]
fn supported_request_defaults_translate_into_embedded_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.request_defaults]
presence_penalty = 1.0
frequency_penalty = 0.5
seed = 7
logit_bias = { "12" = -4.0 }
repeat_last_n = 32
reasoning_format = "deepseek-legacy"
reasoning_enabled = "on"
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("embedded OpenAI args should build");
    assert_eq!(openai.request_defaults.presence_penalty, Some(1.0));
    assert_eq!(openai.request_defaults.frequency_penalty, Some(0.5));
    assert_eq!(openai.request_defaults.seed, Some(7));
    assert_eq!(openai.request_defaults.repeat_last_n, Some(32));
    assert_eq!(
        openai.request_defaults.reasoning_format,
        Some(EmbeddedReasoningFormat::DeepseekLegacy)
    );
    assert_eq!(
        openai.request_defaults.reasoning_enabled,
        Some(EmbeddedReasoningEnabled::Enabled)
    );
    assert_eq!(
        openai
            .request_defaults
            .logit_bias
            .as_ref()
            .and_then(|value| value.get("12"))
            .and_then(serde_json::Value::as_f64),
        Some(-4.0)
    );

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let serialized = serde_json::to_value(&stage_config).expect("stage config json");
    let object = serialized.as_object().expect("stage config object");
    assert!(!object.contains_key("presence_penalty"));
    assert!(!object.contains_key("repeat_last_n"));
    assert!(!object.contains_key("logit_bias"));
}

#[test]
fn sampling_chat_and_reasoning_defaults_reach_embedded_openai_translation() {
    let mesh_config = parse_config(
        r#"
[defaults.request_defaults]
typical_p = 0.73
top_nsigma = 1.7
dynatemp_range = 0.21
dynatemp_exponent = 1.4
mirostat_mode = 2
mirostat_entropy = 4.5
mirostat_learning_rate = 0.08
samplers = ["dry", "top_k", "typical_p", "temperature"]
sampler_sequence = "dky t"
ignore_eos = true
reasoning_format = "hidden"
reasoning_budget = 384
chat_template = "{{ messages }}"
jinja = true
chat_template_kwargs = { custom_mode = 7 }
skip_chat_parsing = true
prefill_assistant = "draft answer"
system_prompt = "configured system"
grammar = "root ::= 'ok'"
json_schema = { type = "object" }

[defaults.request_defaults.dry]
multiplier = 0.8
base = 1.9
allowed_length = 3
penalty_last_n = 48
sequence_breakers = ["\\n", ":"]

[defaults.request_defaults.xtc]
probability = 0.24
threshold = 0.12
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("sampling, chat, and reasoning defaults should resolve");

    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("embedded OpenAI args should carry request defaults");
    let defaults = openai.request_defaults;
    assert_eq!(defaults.typical_p, Some(0.73));
    assert_eq!(defaults.top_nsigma, Some(1.7));
    assert_eq!(defaults.dynatemp_range, Some(0.21));
    assert_eq!(defaults.dynatemp_exponent, Some(1.4));
    assert_eq!(defaults.dry.as_ref().map(|dry| dry.multiplier), Some(0.8));
    assert_eq!(defaults.xtc.as_ref().map(|xtc| xtc.probability), Some(0.24));
    assert_eq!(defaults.mirostat_mode, Some(2));
    assert_eq!(defaults.ignore_eos, Some(true));
    assert_eq!(
        defaults.reasoning_budget,
        Some(EmbeddedReasoningBudget::Tokens(384))
    );
    assert_eq!(defaults.chat_template.as_deref(), Some("{{ messages }}"));
    assert_eq!(defaults.skip_chat_parsing, Some(true));
    assert_eq!(defaults.system_prompt.as_deref(), Some("configured system"));
    assert_eq!(defaults.grammar, Some(serde_json::json!("root ::= 'ok'")));
    assert_eq!(
        defaults.json_schema,
        Some(serde_json::json!({"type": "object"}))
    );
}

#[test]
fn oversized_chat_template_file_is_rejected_before_runtime_startup() {
    let template = NamedTempFile::new().expect("temp chat template");
    std::fs::write(template.path(), vec![b'x'; 1024 * 1024 + 1]).expect("write chat template");
    let mesh_config = parse_config(&format!(
        "[defaults.request_defaults]\nchat_template_file = {:?}\n",
        template.path().display().to_string()
    ));
    let model_file = temp_model_file();
    let resolved = resolve_qwen_config_with_request_defaults(&mesh_config, model_file.path(), None);

    let error = resolved
        .to_embedded_openai_args(4096, true)
        .expect_err("oversized chat template must be rejected");

    assert!(error.to_string().contains("1048576-byte limit"));
}

#[test]
fn inkling_family_defaults_to_q4_kv() {
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &MeshConfig::default(),
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new("/models/inkling.gguf"),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.cache_type_k, "q4_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q4_0");
}

/// The family q4_0 default must be guarded against the model's own metadata:
/// an Inkling variant with per-head widths not divisible by the q4_0 block
/// size (32) cannot load quantised KV, so the resolver must degrade the
/// default to f16 rather than fail the context build.
#[test]
fn inkling_family_kv_default_degrades_to_f16_for_incompatible_meta() {
    let compact_meta = crate::models::gguf::GgufCompactMeta {
        architecture: "inkling".to_string(),
        context_length: 65_536,
        embedding_size: 4096,
        head_count: 32,
        kv_head_count: 8,
        layer_count: 66,
        key_length: 100,
        value_length: 100,
        ..Default::default()
    };
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &MeshConfig::default(),
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new("/models/inkling.gguf"),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: Some(&compact_meta),
    })
    .unwrap();

    assert_eq!(resolved.model_fit.cache_type_k, "f16");
    assert_eq!(resolved.model_fit.cache_type_v, "f16");
}

#[test]
fn model_name_does_not_override_generic_saver_macro_without_metadata() {
    let mesh_config = parse_config(
        r#"
[[models]]
model = "meshllm/inkling-UD-Q2_K_XL-layers"

[models.model_fit]
kv_cache_policy = "saver"
"#,
    );
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new("/models/inkling.gguf"),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
}

#[test]
fn explicit_inkling_kv_override_beats_family_default() {
    let mesh_config = parse_config(
        r#"
[[models]]
model = "meshllm/inkling-UD-Q2_K_XL-layers"
cache_type_k = "q8_0"
cache_type_v = "q8_0"
"#,
    );
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new("/models/inkling.gguf"),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
}

#[test]
fn explicit_global_inkling_kv_override_beats_family_default() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
cache_type_k = "q8_0"
cache_type_v = "q8_0"

[[models]]
model = "meshllm/inkling-UD-Q2_K_XL-layers"
"#,
    );
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new("/models/inkling.gguf"),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
}

#[test]
fn generic_policy_defers_default_prefix_payload_to_loaded_model() {
    let model_file = temp_model_file();
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &MeshConfig::default(),
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("config should resolve");

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let kv_cache = stage_config
        .kv_cache
        .expect("generic policy should enable prefix cache by default");

    assert_eq!(kv_cache.mode, StageKvCacheMode::LookupRecord);
    assert_eq!(kv_cache.payload, StageKvCachePayload::Auto);
    assert!(kv_cache.max_entries > 0);
    assert!(kv_cache.max_bytes > 0);
}

#[test]
fn explicit_prefix_cache_disable_overrides_supported_family_default() {
    let model_file = temp_model_file();
    let config = parse_config(
        r#"
[defaults.model_fit.prefix_cache]
enabled = false
"#,
    );
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("config should resolve");

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let kv_cache = stage_config
        .kv_cache
        .expect("explicit disable must survive family-default materialization");

    assert_eq!(kv_cache.mode, StageKvCacheMode::Disabled);
}

#[test]
fn staged_controls_propagate_into_stage_config_and_embedded_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
prompt_cache = true

[defaults.model_fit.prefix_cache]
enabled = true
max_entries = 9
min_tokens = 96
shared_stride_tokens = 48
shared_record_limit = 3
payload_mode = "resident-kv"

[defaults.skippy]
prefill_chunking = "schedule"
prefill_chunk_size = 128
prefill_chunk_schedule = "128,256,384"

[defaults.speculative]
mode = "draft"
draft_model_path = "/models/qwen3-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail-open"
draft_max_tokens = 8
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("config should resolve");

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let kv_cache = stage_config
        .kv_cache
        .expect("kv cache should be configured");
    assert_eq!(kv_cache.max_entries, 9);
    assert_eq!(kv_cache.min_tokens, 96);
    assert_eq!(kv_cache.shared_prefix_stride_tokens, 48);
    assert_eq!(kv_cache.shared_prefix_record_limit, 3);
    assert_eq!(kv_cache.payload, StageKvCachePayload::ResidentKv);

    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("embedded args should build");
    assert_eq!(openai.prefill_chunk_policy, "schedule");
    assert_eq!(openai.prefill_chunk_size, 128);
    assert_eq!(
        openai.prefill_chunk_schedule.as_deref(),
        Some("128,256,384")
    );
    assert_eq!(openai.speculative_window, 8);
    assert_eq!(
        openai.draft_model_path.as_deref(),
        Some(Path::new("/models/qwen3-draft.gguf"))
    );
}

#[test]
fn layer_package_translation_does_not_treat_hf_ref_as_direct_gguf() {
    let config = MeshConfig::default();
    let package_ref = "hf://meshllm/Qwen3-8B-Q4_K_M-layers";
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "meshllm/Qwen3-8B-Q4_K_M-layers",
        model_path: Path::new(package_ref),
        model_bytes: 5 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();

    let options = resolved
        .to_embedded_runtime_options(
            &SkippyTelemetryOptions::off(),
            Some(fake_hf_package_identity(36)),
            LoadMode::LayerPackage,
        )
        .unwrap();

    assert_eq!(options.config.load_mode, LoadMode::LayerPackage);
    assert_eq!(options.config.model_path.as_deref(), Some(package_ref));
    let kv_cache = options
        .config
        .kv_cache
        .expect("packaged model should retain its automatic cache policy");
    assert_eq!(kv_cache.mode, StageKvCacheMode::LookupRecord);
    assert_eq!(kv_cache.payload, StageKvCachePayload::Auto);
    assert_eq!(kv_cache.max_bytes, 0);
}

#[test]
fn inkling_layer_package_defers_cache_payload_until_model_load() {
    let config = MeshConfig::default();
    let package_ref = "hf://meshllm/inkling-UD-Q2_K_XL-layers";
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
        model_path: Path::new(package_ref),
        model_bytes: 316 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("Inkling package config should resolve");

    let stage = resolved
        .to_stage_config(Some(fake_hf_package_identity(66)), LoadMode::LayerPackage)
        .expect("Inkling package stage config should build");
    let kv_cache = stage
        .kv_cache
        .expect("Inkling package should retain its automatic cache policy");

    assert_eq!(kv_cache.mode, StageKvCacheMode::LookupRecord);
    assert_eq!(kv_cache.payload, StageKvCachePayload::Auto);
    assert!(kv_cache.max_entries > 0);
    assert_eq!(kv_cache.max_bytes, 0);
    assert_eq!(kv_cache.min_tokens, 256);
}

#[test]
fn resolver_rejects_gpu_layers_i32_overflow() {
    let mesh_config = parse_config(
        r#"
[defaults.hardware]
gpu_layers = 2147483648

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"
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

    assert!(error.contains("hardware.gpu_layers must fit in a 32-bit signed integer"));
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
fn model_level_zero_runahead_overrides_a_positive_global_default() {
    use crate::plugin::SpeculativeConfig;
    let global: SpeculativeConfig = toml::from_str(
        r#"
strategy = "ngram-suffix"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
verify_window_pipeline_depth = 2
verify_window_runahead_tokens = 256
"#,
    )
    .expect("parse global speculative config");
    let model: SpeculativeConfig = toml::from_str(
        r#"
verify_window_runahead_tokens = 0
"#,
    )
    .expect("parse model speculative config");
    let inherited = super::speculative::resolve_speculative_config(
        None,
        Some(&global),
        "meshllm/test-model",
        std::path::Path::new("/nonexistent/test-model.gguf"),
        None,
    )
    .expect("global run-ahead must resolve");
    assert_eq!(inherited.decode.verify_window.runahead_max_tokens, 256);
    let overridden = super::speculative::resolve_speculative_config(
        Some(&model),
        Some(&global),
        "meshllm/test-model",
        std::path::Path::new("/nonexistent/test-model.gguf"),
        None,
    )
    .expect("model-level zero must resolve to fixed-depth mode");
    assert_eq!(
        overridden.decode.verify_window.runahead_max_tokens, 0,
        "Some(0) at the model level must win over the inherited positive default"
    );
}
