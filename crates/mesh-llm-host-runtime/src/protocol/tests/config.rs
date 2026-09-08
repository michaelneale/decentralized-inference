use super::*;

#[test]
fn mesh_config_proto_roundtrip() {
    let snapshot = make_config_snapshot();
    let config = proto_config_to_mesh(&snapshot);
    assert_mesh_config_from_proto(&config);

    let roundtripped = mesh_config_to_proto(&config);
    assert_proto_config_roundtrip_matches(&roundtripped, &snapshot);
}

#[test]
fn lifecycle_log_parser_proto_roundtrip_preserves_mode() {
    let mut config = crate::plugin::MeshConfig::default();
    config.runtime.lifecycle_log_parser = mesh_llm_config::LifecycleLogParserMode::Disabled;

    let snapshot = mesh_config_to_proto(&config);
    let restored = proto_config_to_mesh(&snapshot);

    assert_eq!(snapshot.lifecycle_log_parser.as_deref(), Some("disabled"));
    assert_eq!(
        restored.runtime.lifecycle_log_parser,
        mesh_llm_config::LifecycleLogParserMode::Disabled
    );
}

fn assert_mesh_config_from_proto(config: &crate::plugin::MeshConfig) {
    assert_eq!(config.version, Some(1));
    assert_eq!(config.gpu.assignment, crate::plugin::GpuAssignment::Pinned);
    assert_eq!(config.models.len(), 1);
    assert_eq!(config.models[0].model, "Qwen3-8B");
    assert_eq!(config.models[0].mmproj.as_deref(), Some("mmproj-cut"));
    assert_eq!(config.models[0].ctx_size, Some(8192));
    assert_eq!(config.models[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert_eq!(config.plugins.len(), 1);
    assert_eq!(config.plugins[0].name, "demo");
}

fn assert_proto_config_roundtrip_matches(
    roundtripped: &NodeConfigSnapshot,
    snapshot: &NodeConfigSnapshot,
) {
    assert_eq!(roundtripped.version, snapshot.version);
    assert_eq!(
        roundtripped.gpu.as_ref().map(|g| g.assignment),
        Some(crate::proto::node::GpuAssignment::Pinned as i32)
    );
    assert_eq!(roundtripped.models.len(), snapshot.models.len());
    assert_eq!(roundtripped.models[0].model, snapshot.models[0].model);
    assert_eq!(roundtripped.models[0].mmproj, snapshot.models[0].mmproj);
    assert_eq!(roundtripped.models[0].ctx_size, snapshot.models[0].ctx_size);
    assert_eq!(roundtripped.models[0].gpu_id, snapshot.models[0].gpu_id);
    assert_eq!(
        roundtripped.models[0].model_ref,
        snapshot.models[0].model_ref
    );
    assert_eq!(
        roundtripped.models[0].mmproj_ref,
        snapshot.models[0].mmproj_ref
    );
    assert_eq!(roundtripped.plugins.len(), snapshot.plugins.len());
    assert_eq!(roundtripped.plugins[0].name, snapshot.plugins[0].name);
    assert!(
        roundtripped
            .config_toml
            .as_deref()
            .is_some_and(|toml| toml.contains("model = \"Qwen3-8B\"")),
        "re-encoded snapshots should include canonical config_toml payload"
    );
}

#[test]
fn mesh_config_proto_roundtrip_preserves_nested_sections() {
    let config = make_nested_mesh_config();

    let snapshot = mesh_config_to_proto(&config);
    let restored = proto_config_to_mesh(&snapshot);

    let json = serde_json::to_value(&restored).expect("restored config should serialize");
    assert_eq!(json["defaults"]["model_fit"]["kv_unified"], "auto");
    assert_eq!(json["defaults"]["hardware"]["gpu_layers"], "auto");
    assert_eq!(json["defaults"]["throughput"]["parallel"], 3);
    assert_eq!(json["defaults"]["speculative"]["mode"], "auto");
    assert_eq!(
        json["defaults"]["request_defaults"]["reasoning_budget"],
        "auto"
    );
    assert_eq!(
        json["defaults"]["multimodal"]["mmproj"],
        "defaults-projector.gguf"
    );
    assert_eq!(
        json["defaults"]["advanced"]["server"]["alias"],
        "defaults-alias"
    );

    assert_eq!(json["models"][0]["model_fit"]["ctx_size"], 16384);
    assert_eq!(json["models"][0]["hardware"]["gpu_layers"], 99);
    assert_eq!(json["models"][0]["throughput"]["parallel"], 4);
    assert!(json["models"][0]["skippy"].is_null());
    assert_eq!(
        json["models"][0]["speculative"]["draft_selection_policy"],
        "auto"
    );
    assert_eq!(json["models"][0]["request_defaults"]["top_p"], 0.95);
    assert_eq!(
        json["models"][0]["multimodal"]["mmproj"],
        "model-projector.gguf"
    );
    assert_eq!(
        json["models"][0]["advanced"]["server"]["alias"],
        "model-alias"
    );
}

#[test]
fn mesh_config_proto_invalid_full_payload_falls_back_to_legacy_fields() {
    let mut snapshot = make_config_snapshot();
    snapshot.config_toml = Some("not valid toml = [".to_string());

    let restored = proto_config_to_mesh(&snapshot);

    assert_eq!(restored.models[0].model, "Qwen3-8B");
    assert_eq!(restored.models[0].ctx_size, Some(8192));
    assert!(restored.defaults.is_none());
}

#[test]
fn mesh_config_proto_strict_invalid_full_payload_is_rejected() {
    let mut snapshot = make_config_snapshot();
    snapshot.config_toml = Some("not valid toml = [".to_string());

    let err = proto_config_to_mesh_strict(&snapshot).unwrap_err();

    assert!(err.to_string().contains("invalid full config_toml payload"));
}

#[test]
fn mesh_config_proto_strict_legacy_payload_still_restores_fields() {
    let mut snapshot = make_config_snapshot();
    snapshot.config_toml = None;

    let restored = proto_config_to_mesh_strict(&snapshot).unwrap();

    assert_eq!(restored.models[0].model, "Qwen3-8B");
    assert_eq!(restored.models[0].ctx_size, Some(8192));
}

#[test]
fn mesh_config_proto_legacy_payload_preserves_disabled_parser_source() {
    let mut snapshot = make_config_snapshot();
    snapshot.config_toml = None;
    snapshot.lifecycle_log_parser = Some("disabled".to_string());

    let restored = proto_config_to_mesh_strict(&snapshot).unwrap();

    assert_eq!(
        restored.runtime.lifecycle_log_parser,
        mesh_llm_config::LifecycleLogParserMode::Disabled
    );
    assert_eq!(
        restored.runtime.lifecycle_log_parser_source,
        mesh_llm_config::ConfigValueSource::Config
    );
}

#[test]
fn config_sync_prefers_structured_model_refs() {
    let snapshot = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![NodeModelEntry {
            model: "legacy.gguf".to_string(),
            mmproj: Some("legacy-mmproj.gguf".to_string()),
            ctx_size: Some(4096),
            gpu_id: None,
            model_ref: Some(ConfiguredModelRef {
                declared_ref: "structured.gguf".to_string(),
                source_kind: Some("huggingface".to_string()),
                revision: Some("main".to_string()),
            }),
            mmproj_ref: Some(ConfiguredModelRef {
                declared_ref: "structured-mmproj.gguf".to_string(),
                source_kind: Some("huggingface".to_string()),
                revision: Some("main".to_string()),
            }),
        }],
        plugins: vec![],
        config_toml: None,
        mesh_requirements: None,
        lifecycle_log_parser: None,
    };

    let restored = proto_config_to_mesh(&snapshot);

    assert_eq!(restored.models[0].model, "structured.gguf");
    assert_eq!(
        restored.models[0].mmproj.as_deref(),
        Some("structured-mmproj.gguf")
    );
}

#[test]
fn config_sync_empty_structured_refs_fall_back_to_legacy_strings() {
    let snapshot = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![NodeModelEntry {
            model: "legacy.gguf".to_string(),
            mmproj: Some("legacy-mmproj.gguf".to_string()),
            ctx_size: None,
            gpu_id: None,
            model_ref: Some(ConfiguredModelRef {
                declared_ref: "   ".to_string(),
                source_kind: Some("huggingface".to_string()),
                revision: Some("main".to_string()),
            }),
            mmproj_ref: Some(ConfiguredModelRef {
                declared_ref: "".to_string(),
                source_kind: Some("huggingface".to_string()),
                revision: Some("main".to_string()),
            }),
        }],
        plugins: vec![],
        config_toml: None,
        mesh_requirements: None,
        lifecycle_log_parser: None,
    };

    let restored = proto_config_to_mesh(&snapshot);

    assert_eq!(restored.models[0].model, "legacy.gguf");
    assert_eq!(
        restored.models[0].mmproj.as_deref(),
        Some("legacy-mmproj.gguf")
    );
}

#[test]
fn canonical_config_hash_is_stable() {
    let snapshot = make_config_snapshot();
    let hash1 = canonical_config_hash(&snapshot);
    let hash2 = canonical_config_hash(&snapshot);
    assert_eq!(hash1, hash2, "same config must produce the same hash");
    assert_eq!(hash1.len(), 32);

    let mut different = snapshot.clone();
    different.version = 2;
    let hash3 = canonical_config_hash(&different);
    assert_ne!(hash1, hash3, "different config must produce different hash");
}

#[test]
fn canonical_config_hash_changes_when_structured_refs_change_encoding() {
    let mut legacy_only = make_config_snapshot();
    legacy_only.models[0].model_ref = None;
    legacy_only.models[0].mmproj_ref = None;

    let dual_encoded = make_config_snapshot();

    assert_ne!(
        canonical_config_hash(&legacy_only),
        canonical_config_hash(&dual_encoded),
        "legacy-only and dual-encoded snapshots currently have distinct hashes"
    );
}
#[test]
pub(crate) fn mesh_requirements_survive_owner_control_config_round_trip() {
    // Regression: NodeConfigSnapshot used to drop [mesh_requirements] on the
    // owner-control get/apply path, silently stripping admission requirements
    // from an immutable mesh. The proto NodeConfigSnapshot now carries an
    // additive `mesh_requirements` field that mesh_config_to_proto and
    // proto_config_to_mesh round-trip end-to-end.
    use crate::plugin::{MeshRequirementsConfig, OwnerControlConfig};
    let original = crate::plugin::MeshConfig {
        version: Some(1),
        gpu: Default::default(),
        mesh_requirements: MeshRequirementsConfig {
            min_node_version: Some("0.65.0".to_string()),
            max_node_version: Some("0.66.0".to_string()),
            min_protocol_version: Some(1),
            max_protocol_version: Some(3),
            require_release_attestation: true,
            release_signer_keys: vec![
                "ed25519:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
                    .to_string(),
                "ed25519:3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
                    .to_string(),
            ],
        },
        owner_control: OwnerControlConfig::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };
    let snapshot = mesh_config_to_proto(&original);
    assert!(
        snapshot.mesh_requirements.is_some(),
        "non-default mesh_requirements must serialize to the proto snapshot"
    );
    let restored = proto_config_to_mesh(&snapshot);
    assert_eq!(
        restored.mesh_requirements, original.mesh_requirements,
        "mesh_requirements must round-trip through owner-control config get/apply"
    );

    // Default mesh_requirements should remain omitted on the wire so older
    // peers continue to round-trip with absent field semantics.
    let default_only = crate::plugin::MeshConfig::default();
    let default_snapshot = mesh_config_to_proto(&default_only);
    assert!(
        default_snapshot.mesh_requirements.is_none(),
        "default mesh_requirements must not be encoded on the wire"
    );
    let default_restored = proto_config_to_mesh(&default_snapshot);
    assert_eq!(
        default_restored.mesh_requirements,
        crate::plugin::MeshRequirementsConfig::default()
    );
}

#[test]
fn config_sync_empty_config_roundtrip() {
    let config = crate::plugin::MeshConfig::default();
    let snapshot = mesh_config_to_proto(&config);
    let restored = proto_config_to_mesh(&snapshot);
    assert!(restored.models.is_empty());
    assert!(restored.plugins.is_empty());
}

#[test]
fn config_sync_config_toml_roundtrips_additive_defaults_sections() {
    use crate::plugin::{
        ModelConfigDefaults, ModelFitConfig, RequestDefaultsConfig, ThroughputConfig,
    };
    let config = crate::plugin::MeshConfig {
        version: Some(1),
        defaults: Some(ModelConfigDefaults {
            throughput: Some(ThroughputConfig {
                parallel: Some(6),
                ..Default::default()
            }),
            model_fit: Some(ModelFitConfig {
                flash_attention: Some(skippy_protocol::FlashAttentionType::Disabled),
                ..Default::default()
            }),
            request_defaults: Some(RequestDefaultsConfig {
                reasoning_format: Some("deepseek".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let snapshot = mesh_config_to_proto(&config);
    let config_toml = snapshot
        .config_toml
        .as_deref()
        .expect("config TOML should serialize");
    assert!(
        config_toml.contains("parallel") && config_toml.contains("reasoning_format"),
        "config TOML should carry additive defaults values: {config_toml}"
    );

    let restored = proto_config_to_mesh(&snapshot);
    assert_eq!(
        restored
            .extra
            .get("defaults")
            .and_then(|defaults| defaults.get("throughput"))
            .and_then(|throughput| throughput.get("parallel"))
            .and_then(toml::Value::as_integer)
            .or_else(|| {
                restored
                    .defaults
                    .as_ref()
                    .and_then(|defaults| defaults.throughput.as_ref())
                    .and_then(|throughput| throughput.parallel)
                    .map(|parallel| parallel as i64)
            }),
        Some(6)
    );
    assert_eq!(
        restored
            .extra
            .get("defaults")
            .and_then(|defaults| defaults.get("request_defaults"))
            .and_then(|request_defaults| request_defaults.get("reasoning_format"))
            .and_then(toml::Value::as_str)
            .or_else(|| {
                restored
                    .defaults
                    .as_ref()
                    .and_then(|defaults| defaults.request_defaults.as_ref())
                    .and_then(|request_defaults| request_defaults.reasoning_format.as_deref())
            }),
        Some("deepseek")
    );
}

#[test]
fn config_sync_config_hash_determinism() {
    use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};
    let config = crate::plugin::MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![ModelConfigEntry {
            model: "test.gguf".to_string(),
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
        }],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };
    let snap1 = mesh_config_to_proto(&config);
    let snap2 = mesh_config_to_proto(&config);
    let h1 = canonical_config_hash(&snap1);
    let h2 = canonical_config_hash(&snap2);
    assert_eq!(h1, h2, "same config must produce same hash");

    let config2 = crate::plugin::MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![ModelConfigEntry {
            model: "other.gguf".to_string(),
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
        }],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };
    let snap3 = mesh_config_to_proto(&config2);
    let h3 = canonical_config_hash(&snap3);
    assert_ne!(h1, h3, "different config must produce different hash");
}

#[test]
fn mesh_config_proto_roundtrip_preserves_integrated_fixture_and_owner_control_toml() {
    let config: crate::plugin::MeshConfig = toml::from_str(FULL_SURFACE_VALID_FIXTURE).unwrap();
    let snapshot = mesh_config_to_proto(&config);

    assert!(
        snapshot
            .config_toml
            .as_deref()
            .is_some_and(|toml| toml.contains("prefill_chunk_schedule = \"128,256,384\""))
    );

    let restored = proto_config_to_mesh(&snapshot);
    let json = serde_json::to_value(&restored).expect("restored config serializes");
    assert_eq!(json["owner_control"]["bind"], "127.0.0.1:7447");
    assert_eq!(
        json["defaults"]["request_defaults"]["reasoning_budget"],
        256
    );
    assert_eq!(json["models"][0]["hardware"]["stage_layer_start"], 12);
    assert_eq!(
        json["models"][0]["skippy"]["prefill_chunk_schedule"],
        "128,256,384"
    );
    assert_eq!(json["models"][0]["speculative"]["draft_gpu_layers"], 12);
    assert_eq!(
        json["models"][1]["hardware"]["model_path"],
        "/models/gemma.gguf"
    );
}

#[test]
fn pinned_gpu_proto_roundtrip() {
    use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};

    let config = crate::plugin::MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Pinned,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".to_string(),
            mmproj: Some("mmproj-f16.gguf".to_string()),
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".to_string()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };

    let snapshot = mesh_config_to_proto(&config);
    assert_eq!(
        snapshot.gpu.as_ref().map(|gpu| gpu.assignment),
        Some(crate::proto::node::GpuAssignment::Pinned as i32),
        "pinned snapshots must not be downgraded to auto"
    );
    assert_eq!(
        snapshot.models[0].gpu_id.as_deref(),
        Some("pci:0000:65:00.0"),
        "proto snapshot must carry per-model gpu_id"
    );

    let restored = proto_config_to_mesh(&snapshot);
    assert_eq!(restored.gpu.assignment, GpuAssignment::Pinned);
    assert_eq!(
        restored.models[0].gpu_id.as_deref(),
        Some("pci:0000:65:00.0")
    );

    let roundtripped = mesh_config_to_proto(&restored);
    assert_eq!(
        roundtripped.gpu.as_ref().map(|gpu| gpu.assignment),
        Some(crate::proto::node::GpuAssignment::Pinned as i32),
        "re-encoded snapshot must keep pinned assignment"
    );
    assert_eq!(
        roundtripped.models[0].gpu_id.as_deref(),
        Some("pci:0000:65:00.0"),
        "re-encoded snapshot must keep gpu_id presence and value"
    );
}

#[test]
fn pinned_gpu_proto_hash_changes_when_gpu_id_changes() {
    let mut snapshot_a = make_config_snapshot();
    snapshot_a.models[0].gpu_id = Some("pci:0000:65:00.0".to_string());

    let mut snapshot_b = snapshot_a.clone();
    snapshot_b.models[0].gpu_id = Some("pci:0000:66:00.0".to_string());

    assert_ne!(
        canonical_config_hash(&snapshot_a),
        canonical_config_hash(&snapshot_b),
        "changing only gpu_id must change the canonical config hash"
    );
}

#[test]
fn pinned_gpu_proto_missing_gpu_id_decodes_as_none() {
    let snapshot = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Pinned as i32,
        }),
        models: vec![NodeModelEntry {
            model: "Qwen3-8B-Q4_K_M".to_string(),
            mmproj: None,
            ctx_size: Some(4096),
            gpu_id: None,
            model_ref: Some(ConfiguredModelRef {
                declared_ref: "Qwen3-8B-Q4_K_M".to_string(),
                source_kind: None,
                revision: None,
            }),
            mmproj_ref: None,
        }],
        plugins: vec![],
        config_toml: None,
        mesh_requirements: None,
        lifecycle_log_parser: None,
    };

    let encoded = snapshot.encode_to_vec();
    let decoded = NodeConfigSnapshot::decode(encoded.as_slice())
        .expect("payload without gpu_id must still decode");
    let restored = proto_config_to_mesh(&decoded);

    assert_eq!(
        restored.gpu.assignment,
        crate::plugin::GpuAssignment::Pinned
    );
    assert_eq!(restored.models.len(), 1);
    assert_eq!(restored.models[0].gpu_id, None);
    assert_eq!(restored.models[0].ctx_size, Some(4096));
}
