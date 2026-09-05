use super::*;
use std::{
    fs,
    sync::Arc,
    time::{Duration, Instant},
};

use super::inventory::{
    inventory_source_candidates, prepare_stage_source, resolve_inventory_source,
};
use anyhow::{Result, anyhow};
use skippy_protocol::{FlashAttentionType, LoadMode, StageDevice};
use tokio::sync::{Mutex as TokioMutex, oneshot};

#[tokio::test]
async fn stage_control_shutdown_closes_and_joins_the_control_loop() {
    let handle = spawn_stage_control_loop(None, super::super::SkippyTelemetryOptions::default());
    let sender = handle.sender();

    tokio::time::timeout(Duration::from_secs(1), handle.shutdown())
        .await
        .expect("stage control shutdown should be bounded")
        .expect("stage control shutdown should succeed");
    let (resp, _rx) = oneshot::channel();
    assert!(
        sender
            .send(StageControlCommand {
                request: StageControlRequest::Status(StageStatusFilter {
                    topology_id: None,
                    run_id: None,
                    stage_id: None,
                }),
                resp,
            })
            .is_err(),
        "shutdown must close the stage control command channel"
    );
}

#[tokio::test]
async fn stage_control_shutdown_interrupts_an_active_request() {
    let state = StageControlState::default();
    let preparations = Arc::clone(&state.preparations);
    let preparations_guard = preparations.lock().await;
    let handle = spawn_stage_control_loop_with_state(state);
    let (resp, _rx) = oneshot::channel();
    handle
        .sender()
        .send(StageControlCommand {
            request: StageControlRequest::StatusUpdate(preparation_status_from_load(
                &load_request(),
                StagePreparationState::Assigned,
                None,
            )),
            resp,
        })
        .expect("control command accepted");
    tokio::time::sleep(Duration::from_millis(25)).await;

    tokio::time::timeout(Duration::from_secs(1), handle.shutdown())
        .await
        .expect("shutdown must interrupt the active request")
        .expect("stage control shutdown should succeed");
    drop(preparations_guard);
}

fn load_request() -> StageLoadRequest {
    StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        runtime_profile: Some(String::new()),
        backend: "skippy".to_string(),
        package_ref: "pkg-a".to_string(),
        manifest_sha256: "sha256".to_string(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 12,
        admission: crate::inference::skippy::test_stage_admission(0, 12),
        participant_set_hash: "participants".to_string(),
        topology_hash: "topology".to_string(),
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        topology_stages: Vec::new(),
        model_path: Some("/models/model.gguf".to_string()),
        source_model_bytes: Some(64 * 1024 * 1024 * 1024),
        source_model_sha256: None,
        local_source_required: false,
        projector_path: Some("/models/mmproj.gguf".to_string()),
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
        generation_signal_window: None,
        selected_device: Some(StageDevice {
            backend_device: "CUDA0".to_string(),
            stable_id: Some("GPU-123".to_string()),
            index: Some(0),
            vram_bytes: Some(24_000_000_000),
        }),
        bind_addr: "127.0.0.1:0".to_string(),
        ctx_size: 8192,
        lane_count: 3,
        continuous_batching: true,
        n_batch: Some(2048),
        n_ubatch: Some(512),
        n_gpu_layers: -1,
        mmap: Some(false),
        mlock: true,
        cache_type_k: "f16".to_string(),
        cache_type_v: "q8_0".to_string(),
        flash_attn_type: FlashAttentionType::Enabled,
        runtime_settings: StageLoadRuntimeSettings {
            repack: true,
            op_offload: Some(false),
            no_host_buffer: true,
            check_tensors: true,
            direct_io: true,
            main_gpu: Some(2),
            split_mode: skippy_protocol::SplitMode::Row,
            kv_offload: Some(false),
            kv_unified: Some(true),
            swa_full: Some(false),
            cache_idle_slots: Some(3),
        },
        native_mtp_enabled: true,
        shutdown_generation: 7,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
        load_mode: LoadMode::RuntimeSlice,
        upstream: None,
        downstream: Some(StagePeerDescriptor {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "127.0.0.1:9001".to_string(),
            node_id: None,
        }),
    }
}

fn coordinator_id() -> iroh::EndpointId {
    iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[0x5a; 32]).public())
}

fn coordinator_claim_from_load(
    load: &StageLoadRequest,
    coordinator_id: iroh::EndpointId,
) -> StageCoordinatorClaim {
    StageCoordinatorClaim {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        coordinator_id: coordinator_id.to_string(),
        coordinator_term: load.coordinator_term,
        participant_set_hash: "participants".to_string(),
        topology_hash: "topology".to_string(),
        lease_until_unix_ms: u64::MAX,
    }
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_metadata_only_gguf(path: &std::path::Path, layer_count: u32) {
    write_metadata_only_gguf_with_context(path, layer_count, 4096);
}

fn write_metadata_only_gguf_with_context(
    path: &std::path::Path,
    layer_count: u32,
    context_length: u32,
) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&4i64.to_le_bytes());
    push_gguf_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(&mut bytes, "deepseek4");
    push_gguf_string(&mut bytes, "deepseek4.block_count");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&layer_count.to_le_bytes());
    for (key, value) in [
        ("deepseek4.embedding_length", 2048u32),
        ("deepseek4.context_length", context_length),
    ] {
        push_gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

struct BlockingPackagePrefetcher {
    started: TokioMutex<Option<oneshot::Sender<()>>>,
    release: TokioMutex<Option<oneshot::Receiver<Result<()>>>>,
}

impl BlockingPackagePrefetcher {
    fn new() -> (Self, oneshot::Receiver<()>, oneshot::Sender<Result<()>>) {
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        (
            Self {
                started: TokioMutex::new(Some(started_tx)),
                release: TokioMutex::new(Some(release_rx)),
            },
            started_rx,
            release_tx,
        )
    }
}

#[async_trait::async_trait]
impl StagePackagePrefetcher for BlockingPackagePrefetcher {
    async fn prefetch_stage_package(&self, _request: &StagePrepareRequest) -> Result<()> {
        if let Some(started) = self.started.lock().await.take() {
            let _ = started.send(());
        }
        let Some(release) = self.release.lock().await.take() else {
            return Ok(());
        };
        release
            .await
            .unwrap_or_else(|_| Err(anyhow!("prefetch cancelled")))
    }
}

#[tokio::test]
async fn fenced_prepare_requires_accepted_coordinator_claim() {
    let mut load = load_request();
    let coordinator_id = coordinator_id();
    load.coordinator_term = 11;
    load.coordinator_id = Some(coordinator_id);
    load.lease_until_unix_ms = u64::MAX;
    let mut state = StageControlState::default();

    let response = state
        .prepare(StagePrepareRequest {
            load,
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert!(!response.accepted);
    assert_eq!(response.error.as_deref(), Some("missing coordinator claim"));
    assert_eq!(response.status.state, StagePreparationState::Failed);
}

#[tokio::test]
async fn accepted_coordinator_claim_allows_fenced_prepare() {
    let mut load = load_request();
    let coordinator_id = coordinator_id();
    load.coordinator_term = 11;
    load.coordinator_id = Some(coordinator_id);
    load.lease_until_unix_ms = u64::MAX;
    let claim = coordinator_claim_from_load(&load, coordinator_id);
    let mut state = StageControlState::default();

    let ack = state.claim(claim).await.unwrap();
    assert!(ack.accepted);

    let response = state
        .prepare(StagePrepareRequest {
            load,
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert!(response.accepted);
    assert_eq!(response.status.state, StagePreparationState::Assigned);
}

#[test]
fn stage_config_preserves_backend_neutral_load_fields() {
    let request = load_request();
    let config = stage_config(&request, None, None).unwrap();

    assert_stage_config_core_fields(&config);
    assert_eq!(config.repack, request.runtime_settings.repack);
    assert_eq!(config.op_offload, request.runtime_settings.op_offload);
    assert_eq!(
        config.no_host_buffer,
        request.runtime_settings.no_host_buffer
    );
    assert_eq!(config.check_tensors, request.runtime_settings.check_tensors);
    assert_eq!(config.direct_io, request.runtime_settings.direct_io);
    assert_eq!(config.main_gpu, request.runtime_settings.main_gpu);
    assert_eq!(config.split_mode, request.runtime_settings.split_mode);
    assert_eq!(config.kv_offload, request.runtime_settings.kv_offload);
    assert_eq!(config.kv_unified, request.runtime_settings.kv_unified);
    assert_eq!(config.swa_full, request.runtime_settings.swa_full);
    assert_eq!(
        config.cache_idle_slots,
        request.runtime_settings.cache_idle_slots
    );
}

fn assert_stage_config_core_fields(config: &StageConfig) {
    assert_stage_config_identity(config);
    assert_stage_config_package_fields(config);
    assert_stage_config_execution_fields(config);
}

fn assert_stage_config_identity(config: &StageConfig) {
    assert_eq!(config.topology_id, "topology-a");
    assert_eq!(config.run_id, "run-a");
    assert_eq!(config.model_id, "model-a");
    assert_eq!(config.stage_id, "stage-0");
    assert_eq!(config.stage_index, 0);
    assert_eq!(config.layer_start, 0);
    assert_eq!(config.layer_end, 12);
    assert_eq!(config.lane_count, 3);
}

fn assert_stage_config_package_fields(config: &StageConfig) {
    assert_eq!(config.package_ref.as_deref(), Some("pkg-a"));
    assert_eq!(config.manifest_sha256.as_deref(), Some("sha256"));
    assert_eq!(
        config.source_model_path.as_deref(),
        Some("/models/model.gguf")
    );
    assert!(config.materialized_path.is_none());
    assert!(!config.materialized_pinned);
}

fn assert_stage_config_execution_fields(config: &StageConfig) {
    assert_eq!(config.n_batch, Some(2048));
    assert_eq!(config.n_ubatch, Some(512));
    assert_eq!(config.model_path.as_deref(), Some("/models/model.gguf"));
    assert_eq!(
        config.projector_path.as_deref(),
        Some("/models/mmproj.gguf")
    );
    assert_eq!(config.flash_attn_type, FlashAttentionType::Enabled);
    assert_eq!(
        config
            .selected_device
            .as_ref()
            .map(|d| d.backend_device.as_str()),
        Some("CUDA0")
    );
    assert_eq!(
        config.downstream.as_ref().map(|d| d.stage_id.as_str()),
        Some("stage-1")
    );
    assert!(config.filter_tensors_on_load);
}

#[test]
fn stage_config_prefers_package_source_identity_over_local_ref() {
    let mut request = load_request();
    request.load_mode = LoadMode::LayerPackage;
    request.model_path = Some("/tmp/hf-cache/snapshots/abc123".to_string());
    request.source_model_bytes = Some(123);
    let package = super::super::materialization::ResolvedStagePackage {
        local_ref: "/tmp/hf-cache/snapshots/abc123".to_string(),
        source_model_path: "model-a.gguf".to_string(),
        source_model_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        source_model_bytes: Some(456),
    };

    let config = stage_config(&request, None, Some(&package)).unwrap();

    assert_eq!(
        config.model_path.as_deref(),
        Some("/tmp/hf-cache/snapshots/abc123")
    );
    assert_eq!(config.source_model_path.as_deref(), Some("model-a.gguf"));
    assert_eq!(
        config.source_model_sha256.as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(config.source_model_bytes, Some(456));
}

#[test]
fn stage_config_rejects_empty_selected_backend_device() {
    let mut request = load_request();
    request.selected_device = Some(StageDevice {
        backend_device: String::new(),
        stable_id: Some("uuid:GPU-123".into()),
        index: Some(0),
        vram_bytes: Some(24_000_000_000),
    });

    let err = stage_config(&request, None, None).unwrap_err().to_string();

    assert!(err.contains("selected backend device"));
}

#[test]
fn stage_status_filter_matches_optional_identity_fields() {
    let load = load_request();
    assert!(
        StageStatusFilter {
            topology_id: Some("topology-a".to_string()),
            run_id: None,
            stage_id: Some("stage-0".to_string()),
        }
        .matches(&load)
    );
    assert!(
        !StageStatusFilter {
            topology_id: Some("other".to_string()),
            run_id: None,
            stage_id: None,
        }
        .matches(&load)
    );
}

#[test]
fn materialize_stage_bind_addr_replaces_ephemeral_port() {
    let bind_addr = materialize_stage_bind_addr("127.0.0.1:0".parse().unwrap()).unwrap();
    assert_eq!(bind_addr.ip().to_string(), "127.0.0.1");
    assert_ne!(bind_addr.port(), 0);
}

#[test]
fn stage_load_failure_context_identifies_split_stage_shape() {
    let mut request = load_request();
    request.stage_id = "stage-1".to_string();
    request.stage_index = 1;
    request.layer_start = 12;
    request.layer_end = 24;
    request.bind_addr = "127.0.0.1:4242".to_string();

    let context = stage_load_failure_context(
        &request,
        "binary stage ready handshake failed",
        Some("native loader exited while mapping tensors"),
    );

    assert!(context.contains("model=model-a"));
    assert!(context.contains("topology=topology-a"));
    assert!(context.contains("run=run-a"));
    assert!(context.contains("stage=stage-1"));
    assert!(context.contains("index=1"));
    assert!(context.contains("layers=12..24"));
    assert!(context.contains("mode=RuntimeSlice"));
    assert!(context.contains("bind=127.0.0.1:4242"));
    assert!(context.contains("ctx=8192"));
    assert!(context.contains("lanes=3"));
    assert!(context.contains("source_bytes=68719476736"));
    assert!(context.contains("device=CUDA0"));
    assert!(context.contains("error=binary stage ready handshake failed"));
    assert!(context.contains("last_error=native loader exited while mapping tensors"));
}

#[tokio::test]
async fn binary_stage_ready_probe_waits_for_wire_handshake() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind_addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(75));
        let (mut stream, _) = listener.accept().unwrap();
        skippy_protocol::binary::send_ready(&mut stream).unwrap();
    });

    let started = Instant::now();
    let mut probe = start_binary_stage_ready_probe(bind_addr, Duration::from_secs(2));
    (&mut probe.handle)
        .await
        .expect("join readiness probe")
        .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(50));
    server.join().unwrap();
}

#[tokio::test]
async fn stage_control_shutdown_cancels_and_joins_an_active_readiness_probe() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind_addr = listener.local_addr().unwrap();
    let (accepted_tx, accepted_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        accepted_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(Duration::from_secs(5));
    });

    let state = StageControlState {
        readiness_probe: Some(start_binary_stage_ready_probe(
            bind_addr,
            Duration::from_secs(900),
        )),
        ..Default::default()
    };
    let handle = spawn_stage_control_loop_with_state(state);
    accepted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("readiness probe connected to silent stage");

    tokio::time::timeout(Duration::from_secs(3), handle.shutdown())
        .await
        .expect("shutdown must not wait for the readiness deadline")
        .expect("stage control shutdown should succeed");

    release_tx.send(()).unwrap();
    server.join().unwrap();
}

#[tokio::test]
async fn prepare_stage_records_background_source_availability() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let path = file.path().to_string_lossy().to_string();
    let mut load = load_request();
    load.model_path = Some(path.clone());
    load.package_ref = "gguf:///definitely/missing/model.gguf".to_string();
    load.downstream = None;
    let mut state = StageControlState::default();

    let accepted = state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert!(accepted.accepted);
    assert_eq!(accepted.status.state, StagePreparationState::Assigned);

    let mut last_state = StagePreparationState::Assigned;
    for _ in 0..20 {
        let inventory = state
            .inventory(StageInventoryRequest {
                model_id: load.model_id.clone(),
                runtime_profile: load.runtime_profile.clone(),
                package_ref: load.package_ref.clone(),
                manifest_sha256: load.manifest_sha256.clone(),
                expected_source_model_sha256: None,
                local_source_required: false,
            })
            .await;
        if let Some(status) = inventory
            .preparing_ranges
            .iter()
            .find(|status| status.stage_id == load.stage_id)
        {
            last_state = status.state;
            if status.state == StagePreparationState::Available {
                assert_eq!(status.bytes_done, Some(0));
                assert_eq!(status.bytes_total, Some(0));
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    panic!("prepare did not become available, last state: {last_state:?}");
}

#[tokio::test]
async fn prepare_layer_package_stays_downloading_while_peer_prefetch_is_pending() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };

    let accepted = state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert_eq!(accepted.status.state, StagePreparationState::Assigned);
    started_rx.await.expect("prefetch must start");
    tokio::time::sleep(Duration::from_millis(50)).await;

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;
    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("prepare status must stay visible");
    assert_eq!(status.state, StagePreparationState::Downloading);

    let _ = release_tx.send(Err(anyhow!("peer stalled")));
}

#[tokio::test]
async fn prepare_layer_package_fails_only_after_peer_prefetch_and_local_resolution_fail() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };

    state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();
    started_rx.await.expect("prefetch must start");
    release_tx.send(Err(anyhow!("peer unavailable"))).unwrap();

    for _ in 0..40 {
        let inventory = state
            .inventory(StageInventoryRequest {
                model_id: load.model_id.clone(),
                runtime_profile: load.runtime_profile.clone(),
                package_ref: load.package_ref.clone(),
                manifest_sha256: load.manifest_sha256.clone(),
                expected_source_model_sha256: None,
                local_source_required: false,
            })
            .await;
        if let Some(status) = inventory
            .preparing_ranges
            .iter()
            .find(|status| status.stage_id == load.stage_id)
        {
            if status.state == StagePreparationState::Failed {
                let error = status.error.as_deref().unwrap_or_default();
                assert!(error.contains("not a skippy package ref"));
                assert!(error.contains("peer artifact prefetch failed"));
                return;
            }
            assert_ne!(status.state, StagePreparationState::Failed);
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    panic!("prepare did not fail after peer prefetch and local resolution failed");
}

#[tokio::test]
async fn cancel_prepare_persists_cancelled_status_and_blocks_late_prefetch_result() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };

    state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();
    started_rx.await.expect("prefetch must start before cancel");

    let status = state
        .cancel_prepare(StageCancelPrepareRequest {
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            stage_id: load.stage_id.clone(),
            shutdown_generation: load.shutdown_generation + 1,
        })
        .await;

    assert_eq!(status.state, StagePreparationState::Cancelled);
    assert_eq!(status.model_id, load.model_id);
    assert_eq!(status.package_ref, load.package_ref);
    assert_eq!(status.shutdown_generation, load.shutdown_generation + 1);

    let _ = release_tx.send(Ok(()));
    tokio::time::sleep(Duration::from_millis(50)).await;

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;
    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("cancelled prepare status should remain visible");
    assert_eq!(status.state, StagePreparationState::Cancelled);
}

#[tokio::test]
async fn prepare_preserves_equal_or_newer_cancelled_status() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };
    state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();
    started_rx.await.expect("prefetch must start before cancel");

    let status = state
        .cancel_prepare(StageCancelPrepareRequest {
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            stage_id: load.stage_id.clone(),
            shutdown_generation: load.shutdown_generation + 1,
        })
        .await;
    assert_eq!(status.state, StagePreparationState::Cancelled);

    let response = state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert!(!response.accepted);
    assert_eq!(response.error.as_deref(), Some("stale shutdown generation"));
    assert_eq!(response.status.state, StagePreparationState::Cancelled);
    assert_eq!(
        response.status.error.as_deref(),
        Some("stale shutdown generation")
    );
    assert_eq!(
        response.status.shutdown_generation,
        load.shutdown_generation + 1
    );

    let _ = release_tx.send(Ok(()));
    tokio::time::sleep(Duration::from_millis(50)).await;

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;
    let stored = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("cancelled prepare status should remain visible");
    assert_eq!(stored.state, StagePreparationState::Cancelled);
    assert_eq!(stored.shutdown_generation, load.shutdown_generation + 1);
    assert!(stored.error.is_none());
}

#[tokio::test]
async fn stale_cancel_prepare_keeps_newer_prepare_status() {
    let load = load_request();
    let key = stage_key(&load.topology_id, &load.run_id, &load.stage_id);
    let mut state = StageControlState::default();
    let current = preparation_status_from_load(&load, StagePreparationState::Resolving, None);
    state.preparations.lock().await.insert(key, current.clone());

    let status = state
        .cancel_prepare(StageCancelPrepareRequest {
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            stage_id: load.stage_id.clone(),
            shutdown_generation: load.shutdown_generation.saturating_sub(1),
        })
        .await;

    assert_eq!(status.state, StagePreparationState::Resolving);
    assert_eq!(status.shutdown_generation, current.shutdown_generation);
    assert_eq!(status.error.as_deref(), Some("stale shutdown generation"));
}

#[tokio::test]
async fn status_update_upserts_preparation_status_and_rejects_stale_generation() {
    let load = load_request();
    let mut state = StageControlState::default();
    let mut update = preparation_status_from_load(&load, StagePreparationState::Loading, None);
    update.bytes_done = Some(1024);
    update.bytes_total = Some(4096);

    let ack = state.apply_status_update(update.clone()).await;

    assert!(ack.accepted);
    assert!(ack.error.is_none());
    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;
    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("status update should be visible through inventory");
    assert_eq!(status.state, StagePreparationState::Loading);
    assert_eq!(status.bytes_done, Some(1024));

    let mut stale = update;
    stale.shutdown_generation = stale.shutdown_generation.saturating_sub(1);
    stale.state = StagePreparationState::Failed;
    stale.error = Some("late failure".to_string());

    let ack = state.apply_status_update(stale).await;

    assert!(!ack.accepted);
    assert_eq!(ack.error.as_deref(), Some("stale shutdown generation"));
    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;
    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("newer status should remain visible");
    assert_eq!(status.state, StagePreparationState::Loading);
    assert!(status.error.is_none());
}

#[tokio::test]
async fn inventory_retains_failed_prepare_status() {
    let load = load_request();
    let key = stage_key(&load.topology_id, &load.run_id, &load.stage_id);
    let state = StageControlState::default();
    let mut failed = preparation_status_from_load(&load, StagePreparationState::Failed, None);
    failed.error = Some("source unavailable".to_string());
    state.preparations.lock().await.insert(key, failed);

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            runtime_profile: load.runtime_profile.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            expected_source_model_sha256: None,
            local_source_required: false,
        })
        .await;

    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("failed prepare status should remain visible");
    assert_eq!(status.state, StagePreparationState::Failed);
    assert_eq!(status.error.as_deref(), Some("source unavailable"));
}

#[test]
fn inventory_source_candidates_prefer_explicit_gguf_ref() {
    let request = StageInventoryRequest {
        model_id: "catalog-model".to_string(),
        runtime_profile: Some(String::new()),
        package_ref: "gguf:///tmp/source-model.gguf".to_string(),
        manifest_sha256: "sha256".to_string(),
        expected_source_model_sha256: None,
        local_source_required: false,
    };

    let candidates = inventory_source_candidates(&request);

    assert_eq!(
        candidates[0],
        std::path::PathBuf::from("/tmp/source-model.gguf")
    );
}

#[test]
fn inventory_source_resolves_metadata_only_first_shard() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("DeepSeek-V4-Flash-00001-of-00003.gguf");
    write_metadata_only_gguf(&first, 61);
    fs::write(
        dir.path().join("DeepSeek-V4-Flash-00002-of-00003.gguf"),
        vec![0u8; 7],
    )
    .unwrap();
    fs::write(
        dir.path().join("DeepSeek-V4-Flash-00003-of-00003.gguf"),
        vec![0u8; 11],
    )
    .unwrap();
    let expected_bytes = fs::metadata(&first).unwrap().len() + 7 + 11;

    let inventory = resolve_inventory_source(&StageInventoryRequest {
        model_id: "local-gguf/deepseek-v4-flash".to_string(),
        runtime_profile: Some(String::new()),
        package_ref: format!("gguf://{}", first.display()),
        manifest_sha256: "manifest".to_string(),
        expected_source_model_sha256: None,
        local_source_required: false,
    })
    .expect("metadata-only first shard should still advertise inventory");

    assert_eq!(inventory.layer_count, 61);
    assert_eq!(inventory.bytes, Some(expected_bytes));
    assert_eq!(inventory.kind, SourceModelKind::SplitGguf);
    assert!(
        inventory
            .path
            .ends_with("DeepSeek-V4-Flash-00001-of-00003.gguf")
    );
}

#[tokio::test]
async fn content_addressed_inventory_proves_local_bytes_without_leaking_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("worker-local-name.gguf");
    write_metadata_only_gguf(&path, 61);
    let package =
        super::super::synthetic_content_addressed_gguf_package("logical-model", &path).unwrap();
    let request = StageInventoryRequest {
        model_id: "logical-model".to_string(),
        runtime_profile: Some("strict-profile".to_string()),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        expected_source_model_sha256: Some(package.source_model_sha256.clone()),
        local_source_required: true,
    };

    let inventory = StageControlState::default()
        .inventory(request.clone())
        .await;

    assert_eq!(inventory.layer_count, 61);
    assert_eq!(inventory.source_model_path, None);
    assert_eq!(
        inventory.source_model_sha256.as_deref(),
        Some(package.source_model_sha256.as_str())
    );
    assert_eq!(inventory.content_addressed_local_source, Some(true));
    assert_eq!(inventory.available_ranges.len(), 1);

    std::thread::sleep(Duration::from_millis(50));
    write_metadata_only_gguf(&path, 62);
    let tampered = StageControlState::default().inventory(request).await;
    assert_eq!(tampered.content_addressed_local_source, Some(false));
    assert_eq!(tampered.source_model_sha256, None);
    assert!(tampered.available_ranges.is_empty());
}

#[tokio::test]
async fn local_required_prepare_never_falls_back_to_supplied_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("otherwise-resolvable.gguf");
    write_metadata_only_gguf(&path, 61);
    let mut load = load_request();
    load.model_path = Some(path.to_string_lossy().into_owned());
    load.package_ref = format!("local-gguf://sha256/{}", "c".repeat(64));
    load.manifest_sha256 = "d".repeat(64);
    load.source_model_sha256 = Some("c".repeat(64));
    load.local_source_required = true;
    load.load_mode = LoadMode::RuntimeSlice;

    let error = prepare_stage_source(&load).await.unwrap_err().to_string();

    assert!(error.contains("is not registered"), "{error}");
}

#[test]
fn local_required_load_reverifies_content_after_inventory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("load-source.gguf");
    // Keep this content identity distinct from the inventory verification
    // fixture so both tests remain valid under default parallel execution.
    write_metadata_only_gguf_with_context(&path, 61, 8192);
    let package =
        super::super::synthetic_content_addressed_gguf_package("load-verification-model", &path)
            .unwrap();
    let mut load = load_request();
    load.model_id = "load-verification-model".to_string();
    load.runtime_profile = Some("strict".to_string());
    load.package_ref = package.package_ref;
    load.manifest_sha256 = package.manifest_sha256;
    load.source_model_sha256 = Some(package.source_model_sha256);
    load.local_source_required = true;
    load.load_mode = LoadMode::RuntimeSlice;

    assert!(super::super::apply_verified_local_source(&mut load).unwrap());
    let canonical_path = path.canonicalize().unwrap();
    assert_eq!(load.model_path.as_deref(), canonical_path.to_str());

    std::thread::sleep(Duration::from_millis(50));
    write_metadata_only_gguf_with_context(&path, 62, 8192);
    let error = super::super::apply_verified_local_source(&mut load)
        .expect_err("replaced source must fail the load-time verification")
        .to_string();
    assert!(
        error.contains("no registered local GGUF matches"),
        "{error}"
    );
}

#[test]
fn local_required_profile_rejects_legacy_request_without_raising_fallback_profile() {
    let model_id = format!("mixed-profile-model-{}", std::process::id());
    super::super::register_local_source_policy(&model_id, "strict", true);
    super::super::register_local_source_policy(&model_id, "fallback", false);

    let mut legacy = load_request();
    legacy.model_id = model_id.clone();
    legacy.runtime_profile = None;
    legacy.local_source_required = false;
    let error = super::super::apply_verified_local_source(&mut legacy)
        .expect_err("profile-less legacy request must fail closed")
        .to_string();
    assert!(error.contains("content-addressed RuntimeSlice"), "{error}");

    let mut fallback = load_request();
    fallback.model_id = model_id;
    fallback.runtime_profile = Some("fallback".to_string());
    fallback.local_source_required = false;
    assert!(!super::super::apply_verified_local_source(&mut fallback).unwrap());
}

#[test]
fn stage_load_timeout_keeps_existing_floor_without_size_hint() {
    let mut request = load_request();
    request.source_model_bytes = None;
    request.load_mode = LoadMode::RuntimeSlice;

    assert_eq!(stage_load_timeout(&request), Duration::from_secs(900));
}

#[test]
fn stage_load_timeout_scales_with_size_hints_for_all_load_modes() {
    let mut request = load_request();
    request.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    request.load_mode = LoadMode::RuntimeSlice;

    assert_eq!(stage_load_timeout(&request), Duration::from_secs(1360));

    request.load_mode = LoadMode::LayerPackage;
    assert_eq!(stage_load_timeout(&request), Duration::from_secs(1360));

    request.source_model_bytes = Some(u64::MAX);
    assert_eq!(stage_load_timeout(&request), Duration::from_secs(14400));
}
