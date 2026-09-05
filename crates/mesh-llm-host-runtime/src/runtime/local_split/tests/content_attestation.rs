use super::super::attestation::*;
use super::*;
use std::path::{Path, PathBuf};

#[test]
fn split_peer_preflight_requires_local_content_capability_only_for_strict_sources() {
    let mut peer = split_test_peer(0x67, "Qwen3-Coder", true);
    peer.local_gguf_content_id_supported = false;

    assert_eq!(
        split_peer_preflight_exclusion_reason(
            &peer,
            "Qwen3-Coder",
            "meshllm/Qwen3-Coder-layers",
            true,
        ),
        Some(SplitParticipantExclusionReason::UnverifiedLocalSource)
    );
    assert_eq!(
        split_peer_preflight_exclusion_reason(
            &peer,
            "Qwen3-Coder",
            "meshllm/Qwen3-Coder-layers",
            false,
        ),
        None
    );

    peer.local_gguf_content_id_supported = true;
    assert_eq!(
        split_peer_preflight_exclusion_reason(
            &peer,
            "Qwen3-Coder",
            "meshllm/Qwen3-Coder-layers",
            true,
        ),
        None
    );
}

#[test]
fn strict_local_load_rechecks_live_peer_capability() {
    let local_node_id = split_test_peer(0x65, "local", true).id;
    let mut peer = split_test_peer(0x67, "Qwen3-Coder", true);
    let target_node_id = peer.id;

    assert!(peer_supports_strict_local_load(
        local_node_id,
        target_node_id,
        &[peer.clone()]
    ));

    peer.local_gguf_content_id_supported = false;
    assert!(!peer_supports_strict_local_load(
        local_node_id,
        target_node_id,
        &[peer]
    ));
    assert!(peer_supports_strict_local_load(
        local_node_id,
        local_node_id,
        &[]
    ));
}

struct StrictMultimodalStageLoads {
    projector_path: PathBuf,
    stage0: skippy::StageLoadRequest,
    worker: skippy::StageLoadRequest,
}

fn strict_multimodal_config(model_path: &Path, projector_path: &Path) -> plugin::MeshConfig {
    toml::from_str(&format!(
        r#"
[[models]]
model = "strict-multimodal"

[models.hardware]
model_path = "{model_path}"
mmproj = "{projector_path}"

[models.skippy]
source_policy = "local-required"
"#,
        model_path = model_path.display(),
        projector_path = projector_path.display(),
    ))
    .expect("strict multimodal config")
}

fn strict_multimodal_generation(
    local_id: iroh::EndpointId,
    remote_id: iroh::EndpointId,
) -> SplitTopologyGeneration {
    let stages = vec![
        local_stage(local_id, 0, 0, 20),
        local_stage(remote_id, 1, 20, 40),
    ];
    let admissions = stages
        .iter()
        .map(|stage| skippy::test_stage_admission(stage.layer_start, stage.layer_end))
        .collect();
    SplitTopologyGeneration::new(
        "strict-multimodal-topology".into(),
        "strict-multimodal-run".into(),
        1,
        vec![
            SplitParticipant::new(local_id, 24_000_000_000, None),
            SplitParticipant::new(remote_id, 24_000_000_000, None),
        ],
        stages,
    )
    .with_admissions(admissions)
    .unwrap()
}

async fn strict_multimodal_stage_loads() -> StrictMultimodalStageLoads {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("strict-model.gguf");
    let projector_path = temp_dir.path().join("coordinator-mmproj.gguf");
    write_fake_gguf_model(&model_path);
    std::fs::write(&projector_path, b"mmproj").unwrap();
    let config = strict_multimodal_config(&model_path, &projector_path);
    let digest = "a".repeat(64);
    let mut package = package(40);
    package.package_ref = format!("local-gguf://sha256/{digest}");
    package.source_model_sha256 = digest;
    let compact_meta =
        crate::models::gguf::scan_gguf_compact_meta(&model_path).expect("synthetic GGUF metadata");
    let local_id = node.id();
    let remote_id = make_id(0x63);
    let generation = strict_multimodal_generation(local_id, remote_id);
    let spec = SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &config,
        model_ref: "strict-multimodal",
        config_model_id: Some("strict-multimodal"),
        runtime_profile: "strict-profile",
        model_path: &model_path,
        package: &package,
        generation: &generation,
        projector_path: Some(projector_path.to_string_lossy().into_owned()),
        ctx_size: 4096,
        compact_meta: &compact_meta,
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
        local_source_required: true,
    };
    let settings = split_generation_load_settings(&spec)
        .await
        .expect("strict split settings");
    let downstream_stage = &generation.stages[1];
    let downstream = skippy::StagePeerDescriptor {
        stage_id: downstream_stage.stage_id.clone(),
        stage_index: downstream_stage.stage_index,
        endpoint: "127.0.0.1:41001".to_string(),
        node_id: Some(downstream_stage.node_id),
    };

    let stage0 = split_runtime_stage_load_request(
        &spec,
        &settings,
        &generation.stages[0],
        Some(downstream),
        "127.0.0.1:41000",
    )
    .unwrap();
    let worker = split_runtime_stage_load_request(
        &spec,
        &settings,
        downstream_stage,
        None,
        "127.0.0.1:41000",
    )
    .unwrap();

    StrictMultimodalStageLoads {
        projector_path,
        stage0,
        worker,
    }
}

#[tokio::test]
async fn strict_multimodal_stage_builder_keeps_local_paths_off_downstream_loads() {
    let loads = strict_multimodal_stage_loads().await;

    assert!(loads.stage0.local_source_required);
    assert!(loads.stage0.model_path.is_none());
    assert_eq!(
        loads.stage0.projector_path.as_deref(),
        Some(loads.projector_path.to_string_lossy().as_ref())
    );
    assert!(loads.worker.local_source_required);
    assert!(loads.worker.model_path.is_none());
    assert!(loads.worker.projector_path.is_none());
}

#[test]
fn strict_runtime_slice_readiness_requires_exact_content_attestation() {
    let digest = "a".repeat(64);
    let mut load = stage_load_request(LoadMode::RuntimeSlice);
    load.local_source_required = true;
    load.package_ref = format!("local-gguf://sha256/{digest}");
    load.source_model_sha256 = Some(digest.clone());
    let mut inventory = skippy::StageLayerInventory {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        layer_count: 36,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 36,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: load.source_model_bytes,
        source_model_sha256: Some(digest.clone()),
        content_addressed_local_source: Some(true),
        source_model_kind: skippy::SourceModelKind::PlainGguf,
    };

    assert!(split_stage_source_is_ready(&inventory, &load));
    inventory.content_addressed_local_source = None;
    assert!(!split_stage_source_is_ready(&inventory, &load));
    inventory.content_addressed_local_source = Some(false);
    assert!(!split_stage_source_is_ready(&inventory, &load));
    inventory.content_addressed_local_source = Some(true);
    inventory.source_model_sha256 = Some("b".repeat(64));
    assert!(!split_stage_source_is_ready(&inventory, &load));
    inventory.source_model_sha256 = Some(digest);
    inventory.manifest_sha256 = "wrong-manifest".to_string();
    assert!(!split_stage_source_is_ready(&inventory, &load));
}

#[test]
fn strict_ready_status_attests_digest_without_worker_path() {
    let digest = "c".repeat(64);
    let mut load = stage_load_request(LoadMode::RuntimeSlice);
    load.local_source_required = true;
    load.package_ref = format!("local-gguf://sha256/{digest}");
    load.source_model_sha256 = Some(digest.clone());
    let mut status = test_stage_status_from_load(&load, skippy::StageRuntimeState::Ready);
    status.source_model_path = None;
    status.source_model_sha256 = Some(digest);

    assert!(strict_ready_status_matches(&status, &load));
    status.source_model_path = Some("/worker/private/model.gguf".to_string());
    assert!(!strict_ready_status_matches(&status, &load));
    status.source_model_path = None;
    status.source_model_sha256 = Some("d".repeat(64));
    assert!(!strict_ready_status_matches(&status, &load));

    let mut accepted = test_stage_status_from_load(&load, skippy::StageRuntimeState::Ready);
    accepted.source_model_path = None;
    accepted.source_model_sha256 = load.source_model_sha256.clone();
    type StatusMutation = fn(&mut skippy::StageStatusSnapshot);
    let mutations: [(&str, StatusMutation); 13] = [
        ("state", |status| {
            status.state = skippy::StageRuntimeState::Failed
        }),
        ("topology", |status| {
            status.topology_id = "other-topology".into()
        }),
        ("run", |status| status.run_id = "other-run".into()),
        ("stage id", |status| status.stage_id = "other-stage".into()),
        ("stage index", |status| {
            status.stage_index = status.stage_index.saturating_add(1)
        }),
        ("layer start", |status| {
            status.layer_start = status.layer_start.saturating_add(1)
        }),
        ("layer end", |status| {
            status.layer_end = status.layer_end.saturating_sub(1)
        }),
        ("admission descriptor", |status| {
            status.admission.as_mut().unwrap().plan_id = "other-plan".into()
        }),
        ("source bytes", |status| {
            status.source_model_bytes = status.source_model_bytes.map(|bytes| bytes + 1)
        }),
        ("shutdown generation", |status| {
            status.shutdown_generation = status.shutdown_generation.saturating_add(1)
        }),
        ("coordinator term", |status| {
            status.coordinator_term = status.coordinator_term.saturating_add(1)
        }),
        ("coordinator id", |status| {
            status.coordinator_id = Some(make_id(99))
        }),
        ("bind address", |status| {
            status.bind_addr = "127.0.0.1:0".into()
        }),
    ];
    for (field, mutate) in mutations {
        let mut changed = accepted.clone();
        mutate(&mut changed);
        assert!(
            !strict_ready_status_matches(&changed, &load),
            "strict attestation accepted mismatched {field}"
        );
    }
}

#[test]
fn strict_inventory_requires_explicit_content_proof_from_worker() {
    let digest = "a".repeat(64);
    let mut package = package(10);
    package.package_ref = format!("local-gguf://sha256/{digest}");
    package.source_model_sha256 = digest.clone();
    let mut inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: package.layer_count,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: package.layer_count,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: Some(package.source_model_bytes),
        source_model_sha256: None,
        content_addressed_local_source: None,
        source_model_kind: skippy::SourceModelKind::PlainGguf,
    };

    assert_eq!(
        split_inventory_package_signal_result(&inventory, "model-a", &package, false, true),
        Err(SplitParticipantExclusionReason::UnverifiedLocalSource)
    );

    inventory.content_addressed_local_source = Some(true);
    inventory.source_model_sha256 = Some("b".repeat(64));
    assert_eq!(
        split_inventory_package_signal_result(&inventory, "model-a", &package, false, true),
        Err(SplitParticipantExclusionReason::UnverifiedLocalSource)
    );

    inventory.source_model_sha256 = Some(digest);
    assert!(
        split_inventory_package_signal_result(&inventory, "model-a", &package, false, true).is_ok()
    );
}
