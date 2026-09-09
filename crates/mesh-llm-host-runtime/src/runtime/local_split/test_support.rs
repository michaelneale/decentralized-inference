use super::coordinator::*;
use super::loading::*;
use super::*;
use crate::runtime::RuntimeResourcePlanningProfile;
use crate::runtime::local::*;
use crate::runtime::local_package::*;
use crate::runtime::split_planning::RuntimeSliceStagePlan;
use crate::runtime::survey;
use crate::{mesh::NodeRole, plugin};
use iroh::SecretKey;
use sha2::{Digest, Sha256};
use skippy_protocol::{FlashAttentionType, LoadMode};
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn make_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes).public()
}

pub(super) fn package(layer_count: u32) -> skippy::SkippyPackageIdentity {
    skippy::SkippyPackageIdentity {
        package_ref: "gguf:///models/qwen.gguf".to_string(),
        manifest_sha256: "manifest".to_string(),
        source_model_path: PathBuf::from("/models/qwen.gguf"),
        source_model_sha256: "source".to_string(),
        source_model_bytes: u64::from(layer_count) * 1_000_000,
        source_files: Vec::new(),
        layer_weight_bytes: Vec::new(),
        layer_count,
        activation_width: 2048,
        tensor_count: 100,
        generation: None,
    }
}

pub(super) fn stage_load_request(load_mode: LoadMode) -> skippy::StageLoadRequest {
    skippy::StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        runtime_profile: Some(String::new()),
        backend: "skippy".to_string(),
        package_ref: match load_mode {
            LoadMode::LayerPackage => "hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
                "gguf:///models/qwen.gguf".to_string()
            }
        },
        manifest_sha256: "a".repeat(64),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 18,
        layer_end: 36,
        admission: skippy::test_stage_admission(18, 36),
        participant_set_hash: "participants".to_string(),
        topology_hash: "topology".to_string(),
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: Default::default(),
        topology_stages: Vec::new(),
        model_path: Some("/models/qwen.gguf".to_string()),
        source_model_bytes: Some(4_900_000_000),
        source_model_sha256: None,
        local_source_required: false,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
        generation_signal_window: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        ctx_size: 8192,
        lane_count: 4,
        continuous_batching: true,
        n_batch: Some(2048),
        n_ubatch: Some(512),
        n_gpu_layers: -1,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: FlashAttentionType::Auto,
        runtime_settings: Default::default(),
        native_mtp_enabled: true,
        shutdown_generation: 1,
        coordinator_term: 1,
        coordinator_id: None,
        lease_until_unix_ms: u64::MAX,
        load_mode,
        upstream: None,
        downstream: None,
    }
}

#[test]
fn split_generation_cli_device_override_survives_pinned_stage_selection() {
    let mut config = skippy_protocol::StageConfig {
        selected_device: Some(skippy_protocol::StageDevice {
            backend_device: "CPU".to_string(),
            stable_id: None,
            index: None,
            vram_bytes: None,
        }),
        ..Default::default()
    };
    let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
        index: 0,
        stable_id: "pci:0000:65:00.0".to_string(),
        backend_device: "CUDA0".to_string(),
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
    };

    apply_split_generation_pinned_device(&mut config, Some(&pinned_gpu), Some("CPU"));

    assert_eq!(
        config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.as_str()),
        Some("CPU")
    );
}

pub(super) fn split_test_peer(
    seed: u8,
    model_name: &str,
    stage_protocol_generation_supported: bool,
) -> mesh::PeerInfo {
    let id = make_id(seed);
    mesh::PeerInfo {
        id,
        addr: iroh::EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: Vec::new(),
        vram_bytes: 24_000_000_000,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: Vec::new(),
        hosted_models: Vec::new(),
        hosted_models_known: false,
        available_models: Vec::new(),
        requested_models: vec![model_name.to_string()],
        explicit_model_interests: Vec::new(),
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        memory: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: Vec::new(),
        experts_summary: None,
        available_model_sizes: std::collections::HashMap::new(),
        served_model_descriptors: Vec::new(),
        served_model_runtime: Vec::new(),
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: stage_protocol_generation_supported,
        advertised_model_throughput: vec![],
        cache_affinity: None,

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        inference_admission_state: None,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(bytes, value);
}

pub(super) fn write_fake_gguf_model(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    fs::write(path, bytes).unwrap();
}

#[test]
fn split_metadata_reads_a_synthetic_direct_gguf_source() {
    let temp = tempfile::tempdir().unwrap();
    let model_path = temp.path().join("model.gguf");
    write_fake_gguf_model(&model_path);
    let package = skippy::synthetic_direct_gguf_package("test/model", &model_path).unwrap();

    let metadata = scan_layer_package_metadata(&package).expect("direct GGUF metadata");

    assert_eq!(metadata.context_length, 8192);
    assert_eq!(metadata.embedding_size, 4096);
}

pub(super) fn write_test_layer_package(dir: &Path, source_model_bytes: u64) {
    fs::create_dir_all(dir.join("layers")).unwrap();
    fs::write(dir.join("metadata.gguf"), b"metadata").unwrap();
    fs::write(dir.join("embeddings.gguf"), b"embeddings").unwrap();
    fs::write(dir.join("output.gguf"), b"output").unwrap();
    fs::write(dir.join("layers/00000.gguf"), b"layer0").unwrap();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "model_id": "meshllm/test-layer-package",
        "source_model": {
            "path": "/models/test-layer-package.gguf",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "files": [{
                "path": "/models/test-layer-package.gguf",
                "size_bytes": source_model_bytes,
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }]
        },
        "format": "layer-package",
        "layer_count": 1,
        "activation_width": 4096,
        "shared": {
            "metadata": {
                "path": "metadata.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 8,
                "sha256": sha256_hex(b"metadata")
            },
            "embeddings": {
                "path": "embeddings.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 10,
                "sha256": sha256_hex(b"embeddings")
            },
            "output": {
                "path": "output.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 6,
                "sha256": sha256_hex(b"output")
            }
        },
        "layers": [{
            "layer_index": 0,
            "path": "layers/00000.gguf",
            "tensor_count": 1,
            "tensor_bytes": 1,
            "artifact_bytes": 6,
            "sha256": sha256_hex(b"layer0")
        }],
        "skippy_abi_version": "0.1.0",
    });
    fs::write(
        dir.join("model-package.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

pub(super) fn participant(seed: u8) -> SplitParticipant {
    SplitParticipant::new(make_id(seed), 24_000_000_000, None)
}

pub(super) fn stage(
    seed: u8,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
) -> RuntimeSliceStagePlan {
    RuntimeSliceStagePlan {
        stage_id: format!("stage-{stage_index}"),
        stage_index,
        node_id: make_id(seed),
        layer_start,
        layer_end,
        parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
    }
}

pub(super) fn runtime_status_for_stage(
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    state: skippy::StageRuntimeState,
) -> mesh::StageRuntimeStatus {
    mesh::StageRuntimeStatus {
        topology_id: generation.topology_id.clone(),
        run_id: generation.run_id.clone(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: Some("gguf:///model.gguf".to_string()),
        manifest_sha256: Some("direct-gguf:1:model.gguf".to_string()),
        source_model_path: Some("/model.gguf".to_string()),
        source_model_sha256: None,
        source_model_bytes: Some(1),
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        node_id: Some(stage.node_id),
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        admission: Some(skippy::test_stage_admission(
            stage.layer_start,
            stage.layer_end,
        )),
        activation_codec: generation.activation_codec,
        activation_codec_policy: Default::default(),
        state,
        bind_addr: "127.0.0.1:31000".to_string(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: None,
        ctx_size: 512,
        lane_count: 4,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: generation.generation,
    }
}

pub(super) fn local_stage(
    node_id: iroh::EndpointId,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
) -> RuntimeSliceStagePlan {
    RuntimeSliceStagePlan {
        stage_id: format!("stage-{stage_index}"),
        stage_index,
        node_id,
        layer_start,
        layer_end,
        parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
    }
}

#[tokio::test]
async fn split_generation_load_settings_preserves_unpinned_budget_and_resolved_config() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("qwen.gguf");
    let draft_model_path = temp_dir.path().join("qwen-draft.gguf");
    let projector_path = temp_dir.path().join("config-mmproj.gguf");
    write_fake_gguf_model(&model_path);
    write_fake_gguf_model(&draft_model_path);
    fs::write(&projector_path, b"mmproj").unwrap();
    let model_path_toml =
        toml::Value::String(model_path.to_string_lossy().into_owned()).to_string();
    let draft_model_path_toml =
        toml::Value::String(draft_model_path.to_string_lossy().into_owned()).to_string();
    let projector_path_toml =
        toml::Value::String(projector_path.to_string_lossy().into_owned()).to_string();
    let mesh_config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "other/model"

[models.hardware]
model_path = {model_path}

[models.throughput]
threads = 17
threads_batch = 13

[[models]]
model = "Qwen"

[models.model_fit]
ctx_size = 2048
batch = 768
ubatch = 192
cache_type_k = "q4_0"
cache_type_v = "q5_0"

[models.hardware]
model_path = {model_path}
device = "CUDA0"
gpu_layers = 77
mmproj = {projector_path}

[models.throughput]
parallel = 2
threads = 6
threads_batch = 3

[models.skippy]
prefill_chunking = "fixed"
prefill_chunk_size = 96

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = {draft_model_path}
draft_max_tokens = 7
draft_gpu_layers = 11

[models.request_defaults]
max_tokens = 321
temperature = 0.35
stop = ["END"]
"#,
        model_path = model_path_toml,
        draft_model_path = draft_model_path_toml,
        projector_path = projector_path_toml
    ))
    .expect("test mesh config should parse");
    let package = package(40);
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("qwen.gguf");
    write_fake_gguf_model(&model_path);
    let compact_meta =
        crate::models::gguf::scan_gguf_compact_meta(&model_path).expect("synthetic GGUF metadata");
    let local_id = node.id();
    let stages = vec![
        local_stage(local_id, 0, 0, 12),
        local_stage(local_id, 1, 12, 40),
    ];
    let admissions = stages
        .iter()
        .map(|stage| skippy::test_stage_admission(stage.layer_start, stage.layer_end))
        .collect();
    let generation = SplitTopologyGeneration::new(
        "resolver-topology".into(),
        "resolver-run".into(),
        1,
        vec![SplitParticipant::new(local_id, 24_000_000_000, None)],
        stages,
    )
    .with_admissions(admissions)
    .unwrap();

    let mut spec = SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &mesh_config,
        model_ref: "served-qwen",
        config_model_id: Some("Qwen"),
        runtime_profile: "",
        model_path: &model_path,
        package: &package,
        generation: &generation,
        projector_path: Some("/models/fallback-mmproj.gguf".to_string()),
        ctx_size: 8192,
        compact_meta: &compact_meta,
        capacity_budget_bytes: Some(6_000_000_000),
        pinned_gpu: None,
        device_override: None,
        slots: 4,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
        serving_hooks_factory: None,
        local_source_required: false,
    };
    let settings = split_generation_load_settings(&spec)
        .await
        .expect("split settings should resolve");

    assert_eq!(split_allocatable_memory_bytes(&spec), Some(6_000_000_000));
    assert_eq!(settings.load_mode, LoadMode::RuntimeSlice);
    assert_eq!(settings.embedded_openai.activation_width, 0);
    assert_eq!(settings.runtime_options.n_threads, Some(6));
    assert_eq!(settings.runtime_options.n_threads_batch, Some(3));
    assert_eq!(settings.runtime_options.config.ctx_size, 8192);
    assert_eq!(settings.runtime_options.config.lane_count, 4);
    assert_eq!(settings.runtime_options.config.n_batch, Some(768));
    assert_eq!(settings.runtime_options.config.n_ubatch, Some(192));
    assert_eq!(settings.runtime_options.config.n_gpu_layers, 77);
    assert_eq!(
        settings
            .runtime_options
            .config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.as_str()),
        Some("CUDA0")
    );
    assert_eq!(settings.runtime_options.config.cache_type_k, "q4_0");
    assert_eq!(settings.runtime_options.config.cache_type_v, "q5_0");
    assert_eq!(
        settings.runtime_options.config.projector_path.as_deref(),
        Some(projector_path.to_string_lossy().as_ref())
    );
    assert!(!settings.runtime_options.config.native_mtp_enabled);
    assert!(!settings.embedded_openai.native_mtp_enabled);
    assert_eq!(settings.embedded_openai.generation_concurrency, 4);
    assert_eq!(settings.embedded_openai.default_max_tokens, 321);
    assert_eq!(
        settings.embedded_openai.request_defaults.temperature,
        Some(0.35)
    );
    assert_eq!(
        settings.embedded_openai.request_defaults.stop.as_deref(),
        Some(["END".to_string()].as_slice())
    );
    assert_eq!(settings.embedded_openai.prefill_chunk_policy, "fixed");
    assert_eq!(settings.embedded_openai.prefill_chunk_size, 96);
    assert_eq!(
        settings.embedded_openai.draft_model_path.as_deref(),
        Some(draft_model_path.as_path())
    );
    assert_eq!(settings.embedded_openai.speculative_window, 7);
    assert_eq!(settings.embedded_openai.draft_n_gpu_layers, Some(11));
    spec.capacity_budget_bytes = Some(0);
    assert_eq!(
        split_allocatable_memory_bytes(&spec),
        None,
        "a zero optional budget is unset when no pinned-device fallback exists"
    );
}

/// Split stage loading must resolve with the compact metadata scanned during
/// planning: the architecture-driven K/V default gets the same compatibility
/// guard as the planner, so a default the actual GGUF cannot load (here:
/// Inkling → q4_0 with per-head widths not divisible by the q4_0 block size)
/// degrades to f16 at stage load instead of failing the context build.
///
/// The package is deliberately small (10 GB) so the size-tiered policy alone
/// would pick q8_0: the observed q4_0-vs-f16 swing can only come from actual
/// Inkling metadata, pinning the plumbing rather than a model-name heuristic.
/// Split load specifications require this metadata, so both the
/// initial-load and coordinator-replan constructors must carry it; dropping
/// the final resolver handoff would regress this test to q4_0.
#[tokio::test]
async fn split_stage_load_guards_metadata_kv_default_with_planned_metadata() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9338 })
        .await
        .unwrap();
    let mesh_config = plugin::MeshConfig::default();
    // Non-existent path on purpose: only the supplied compact metadata may
    // identify the architecture; the model ref must not influence policy.
    let model_path = std::path::PathBuf::from("/models/inkling-ud-q2-k-xl.gguf");
    let mut identity = package(66);
    identity.package_ref = "hf://Mesh-LLM/test-inkling-package".to_string();
    identity.source_model_bytes = 10 * 1024 * 1024 * 1024;
    let local_id = node.id();
    let generation = SplitTopologyGeneration::new(
        "guard-topology".into(),
        "guard-run".into(),
        1,
        vec![SplitParticipant::new(local_id, 24_000_000_000, None)],
        vec![
            local_stage(local_id, 0, 0, 33),
            local_stage(local_id, 1, 33, 66),
        ],
    );

    // Per-head widths of 100 are not a multiple of the q4_0 block size (32),
    // so the Inkling architecture's quantised default cannot load.
    let incompatible_meta = crate::models::gguf::GgufCompactMeta {
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

    // With the planned metadata, the unloadable architecture default degrades to
    // f16 — the same cache the split planner budgets for.
    let guarded_spec = SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &mesh_config,
        model_ref: "meshllm/inkling-UD-Q2_K_XL-layers",
        config_model_id: None,
        runtime_profile: "",
        model_path: &model_path,
        package: &identity,
        generation: &generation,
        projector_path: None,
        ctx_size: 4096,
        compact_meta: &incompatible_meta,
        capacity_budget_bytes: None,
        pinned_gpu: None,
        device_override: None,
        slots: 1,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
        serving_hooks_factory: None,
        local_source_required: false,
    };
    let guarded = split_generation_load_settings(&guarded_spec)
        .await
        .expect("guarded split settings should resolve");
    assert_eq!(
        guarded.runtime_options.config.cache_type_k, "f16",
        "incompatible architecture default must degrade to f16 at stage load"
    );
    assert_eq!(guarded.runtime_options.config.cache_type_v, "f16");

    // Without metadata the model name carries no architecture authority, so
    // the generic q8_0 size tier remains in effect.
    let unguarded = skippy::resolve_skippy_config_for_selector(
        skippy::SkippyConfigResolveRequest {
            mesh_config: &mesh_config,
            model_id: "meshllm/inkling-UD-Q2_K_XL-layers",
            model_path: &model_path,
            model_bytes: identity.source_model_bytes,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        },
        None,
    )
    .expect("unguarded resolver settings should resolve");
    assert_eq!(
        unguarded.model_fit.cache_type_k, "q8_0",
        "no-metadata stage load keeps the generic size-tier default"
    );
    assert_eq!(unguarded.model_fit.cache_type_v, "q8_0");
}

#[tokio::test]
async fn runtime_resolver_uses_config_identity_and_honors_device_override() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("alias-target.gguf");
    write_fake_gguf_model(&model_path);
    let mesh_config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "other/model-ref"

[models.hardware]
model_path = "{model_path}"
device = "CUDA1"

[models.throughput]
threads = 17
threads_batch = 13

[[models]]
model = "configured/model-ref"

[models.hardware]
model_path = "{model_path}"

[models.throughput]
threads = 9
threads_batch = 5

[models.request_defaults]
max_tokens = 222
"#,
        model_path = model_path.display()
    ))
    .expect("test mesh config should parse");
    let model_bytes = fs::metadata(&model_path).unwrap().len();
    let spec = LocalOpenAiModelStartSpec {
        mesh_config: &mesh_config,
        config_model_id: Some("configured/model-ref"),
        model_path: &model_path,
        model_bytes,
        mmproj_override: None,
        ctx_size_override: None,
        pinned_gpu: None,
        device_override: Some("CPU".to_string()),
        capacity_budget_bytes: node.vram_bytes(),
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        parallel_override: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
        hook_policy: None,
        serving_hooks_factory: None,
        http_bind_addr: "127.0.0.1:0".parse().expect("valid loopback address"),
    };

    let resolved = resolve_local_openai_skippy_config(
        &spec,
        "runtime/served-name",
        model_bytes,
        4096,
        3,
        None,
        None,
    )
    .expect("runtime config should resolve through configured model id");

    assert_eq!(resolved.model_id, "runtime/served-name");
    assert_eq!(resolved.throughput.threads, Some(9));
    assert_eq!(resolved.throughput.threads_batch, Some(5));
    assert_eq!(resolved.request_defaults.max_tokens, 222);
    assert_eq!(resolved.model_fit.ctx_size, 4096);
    assert_eq!(resolved.throughput.parallel, 3);
    assert_eq!(resolved.hardware.device.as_deref(), Some("CPU"));
    // An explicit CLI artifact may use the same served name as a configured
    // model, but must not inherit that entry's path or runtime tuning.
    let cli_model_path = temp_dir.path().join("cli-selected.gguf");
    write_fake_gguf_model(&cli_model_path);
    let cli_model_bytes = fs::metadata(&cli_model_path).unwrap().len();
    let cli_spec = LocalOpenAiModelStartSpec {
        mesh_config: &mesh_config,
        config_model_id: None,
        model_path: &cli_model_path,
        model_bytes: cli_model_bytes,
        mmproj_override: None,
        ctx_size_override: None,
        pinned_gpu: None,
        device_override: None,
        capacity_budget_bytes: node.vram_bytes(),
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        parallel_override: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
        hook_policy: None,
        serving_hooks_factory: None,
        http_bind_addr: "127.0.0.1:0".parse().expect("valid loopback address"),
    };
    let cli_resolved = resolve_local_openai_skippy_config(
        &cli_spec,
        "configured/model-ref",
        cli_model_bytes,
        4096,
        3,
        None,
        None,
    )
    .expect("explicit CLI runtime config should not consult model entries");
    assert_eq!(cli_resolved.hardware.resolved_model_path, cli_model_path);
    assert_eq!(cli_resolved.throughput.threads, None);
    assert_eq!(cli_resolved.throughput.threads_batch, None);
    assert_eq!(
        cli_resolved.request_defaults.max_tokens,
        skippy_server::CONTEXT_BUDGET_MAX_TOKENS
    );
}

#[test]
fn runtime_verified_served_model_descriptor_preserves_identity_and_updates_capabilities() {
    let existing = mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: "Qwen3VL-2B-Instruct-Q4_K_M".into(),
            is_primary: false,
            source_kind: mesh::ModelSourceKind::HuggingFace,
            repository: Some("Qwen/Qwen3-VL-2B-Instruct-GGUF".into()),
            artifact: Some("Qwen3VL-2B-Instruct-Q4_K_M.gguf".into()),
            ..Default::default()
        },
        capabilities_known: false,
        capabilities: models::ModelCapabilities::default(),
        topology: None,
        metadata: None,
    };
    let capabilities = models::ModelCapabilities {
        multimodal: true,
        vision: models::CapabilityLevel::Supported,
        ..Default::default()
    };

    let descriptor = runtime_verified_served_model_descriptor(
        Some(existing),
        "Qwen3VL-2B-Instruct-Q4_K_M",
        "Qwen3VL-2B-Instruct-Q4_K_M",
        capabilities,
    );

    assert_eq!(
        descriptor.identity.source_kind,
        mesh::ModelSourceKind::HuggingFace
    );
    assert_eq!(
        descriptor.identity.repository.as_deref(),
        Some("Qwen/Qwen3-VL-2B-Instruct-GGUF")
    );
    assert!(descriptor.identity.is_primary);
    assert!(descriptor.capabilities_known);
    assert_eq!(descriptor.capabilities, capabilities);
}

#[test]
fn runtime_verified_served_model_descriptor_builds_fallback_identity() {
    let descriptor = runtime_verified_served_model_descriptor(
        None,
        "Primary",
        "Runtime",
        models::ModelCapabilities::default(),
    );

    assert_eq!(descriptor.identity.model_name, "Runtime");
    assert!(!descriptor.identity.is_primary);
    assert_eq!(
        descriptor.identity.source_kind,
        mesh::ModelSourceKind::Unknown
    );
    assert_eq!(
        descriptor.identity.local_file_name.as_deref(),
        Some("Runtime.gguf")
    );
    assert_eq!(
        descriptor.capabilities,
        models::ModelCapabilities::default()
    );
    assert!(descriptor.capabilities_known);
}

pub(super) fn test_stage_status_from_load(
    load: &skippy::StageLoadRequest,
    state: skippy::StageRuntimeState,
) -> skippy::StageStatusSnapshot {
    let boundary = Some(skippy_runtime::ActivationBoundaryDesc {
        version: 1,
        ggml_type: 0,
        layout: 1,
        elements_per_token: 4096,
        bytes_per_token: 4096 * std::mem::size_of::<f32>() as u64,
        required_frame_flags: 0,
        required_sidebands: 0,
    });
    skippy::StageStatusSnapshot {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: load.model_path.clone(),
        source_model_sha256: None,
        source_model_bytes: load.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: load.projector_path.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: Default::default(),
        state,
        bind_addr: "127.0.0.1:31000".to_string(),
        input_activation_boundary: boundary.filter(|_| load.layer_start > 0),
        output_activation_boundary: load.downstream.as_ref().and(boundary),
        selected_device: load.selected_device.clone(),
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        flash_attn_type: load.flash_attn_type,
        error: None,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

#[test]
fn staged_status_fixture_exposes_input_boundary_for_middle_consumers() {
    let middle = stage_load_request(LoadMode::RuntimeSlice);
    assert!(middle.layer_start > 0);
    assert!(middle.upstream.is_none());

    let status = test_stage_status_from_load(&middle, skippy::StageRuntimeState::Ready);
    assert!(status.input_activation_boundary.is_some());
}

pub(super) fn test_stage_status_from_stop(
    stop: &skippy::StageStopRequest,
) -> skippy::StageStatusSnapshot {
    skippy::StageStatusSnapshot {
        topology_id: stop.topology_id.clone(),
        run_id: stop.run_id.clone(),
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stop.stage_id.clone(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        admission: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: Default::default(),
        state: skippy::StageRuntimeState::Stopped,
        bind_addr: String::new(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: None,
        ctx_size: 0,
        lane_count: 0,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: stop.shutdown_generation,
        coordinator_term: stop.coordinator_term,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

pub(super) fn test_preparation_status_from_load(
    load: &skippy::StageLoadRequest,
) -> skippy::StagePreparationStatus {
    skippy::StagePreparationStatus {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: Default::default(),
        state: skippy::StagePreparationState::Available,
        bytes_done: load.source_model_bytes,
        bytes_total: load.source_model_bytes,
        bind_addr: None,
        error: None,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn test_inventory_from_request(
    request: &skippy::StageInventoryRequest,
) -> skippy::StageLayerInventory {
    skippy::StageLayerInventory {
        model_id: request.model_id.clone(),
        package_ref: request.package_ref.clone(),
        manifest_sha256: request.manifest_sha256.clone(),
        layer_count: 40,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 40,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: Some("/models/qwen.gguf".to_string()),
        source_model_bytes: Some(40_000_000),
        source_model_sha256: None,
        content_addressed_local_source: None,
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    }
}
