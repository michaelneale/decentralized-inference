fn make_test_peer_info(peer_id: EndpointId) -> PeerInfo {
    PeerInfo {
        id: peer_id,
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![ModelRuntimeDescriptor {
            model_name: "Qwen3-8B-Q4_K_M".to_string(),
            identity_hash: Some("sha256:abc123".into()),
            context_length: Some(32768),
            ready: true,
        }],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        owner_summary: OwnershipSummary::default(),
        advertised_model_throughput: vec![],
        cache_affinity: None,
        inference_admission_state: None,

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
    }
}

fn make_test_endpoint_id(seed: u8) -> EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    EndpointId::from(SecretKey::from_bytes(&bytes).public())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    hex::encode(Sha256::digest(bytes))
}

fn owner_control_request(request_id: u64) -> crate::proto::node::OwnerControlRequest {
    crate::proto::node::OwnerControlRequest {
        request_id,
        ..Default::default()
    }
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &std::path::Path) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: mesh tests that use this guard are annotated `#[serial]`, so
        // the process environment key is mutated only while that serial test
        // owns the guard.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn set_str(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: mesh tests that use this guard are annotated `#[serial]`, so
        // the process environment key is mutated only while that serial test
        // owns the guard.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: mesh tests that use this guard are annotated `#[serial]`, so
        // the process environment key is mutated only while that serial test
        // owns the guard.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.previous.take() {
            // SAFETY: restoration runs during drop in the same `#[serial]` mesh
            // test that performed the mutation.
            unsafe { std::env::set_var(self.key, value) };
        } else {
            // SAFETY: restoration runs during drop in the same `#[serial]` mesh
            // test that performed the mutation.
            unsafe { std::env::remove_var(self.key) };
        }
    }
}

fn write_artifact_authorization_package(root: &std::path::Path) -> (String, String) {
    std::fs::create_dir_all(root.join("shared")).unwrap();
    std::fs::create_dir_all(root.join("layers")).unwrap();
    std::fs::create_dir_all(root.join("projectors")).unwrap();
    std::fs::write(root.join("shared/metadata.gguf"), b"metadata").unwrap();
    std::fs::write(root.join("shared/embeddings.gguf"), b"embed").unwrap();
    std::fs::write(root.join("shared/output.gguf"), b"output").unwrap();
    std::fs::write(root.join("layers/layer-000.gguf"), b"layer000").unwrap();
    std::fs::write(root.join("layers/layer-001.gguf"), b"layer001").unwrap();
    std::fs::write(root.join("projectors/mmproj.gguf"), b"projector").unwrap();
    let manifest = serde_json::json!({
        "shared": {
            "metadata": { "path": "shared/metadata.gguf", "sha256": sha256_hex(b"metadata"), "artifact_bytes": 8 },
            "embeddings": { "path": "shared/embeddings.gguf", "sha256": sha256_hex(b"embed"), "artifact_bytes": 5 },
            "output": { "path": "shared/output.gguf", "sha256": sha256_hex(b"output"), "artifact_bytes": 6 }
        },
        "layers": [
            { "layer_index": 0, "path": "layers/layer-000.gguf", "sha256": sha256_hex(b"layer000"), "artifact_bytes": 8 },
            { "layer_index": 1, "path": "layers/layer-001.gguf", "sha256": sha256_hex(b"layer001"), "artifact_bytes": 8 }
        ],
        "projectors": [
            { "kind": "mmproj", "path": "projectors/mmproj.gguf", "sha256": sha256_hex(b"projector"), "artifact_bytes": 9 }
        ]
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    let manifest_sha = sha256_hex(&manifest_bytes);
    std::fs::write(root.join("model-package.json"), manifest_bytes).unwrap();
    ("hf://meshllm/auth-package@abc123".to_string(), manifest_sha)
}

fn write_hf_artifact_stream_package(
    root: &std::path::Path,
) -> (std::path::PathBuf, String, String) {
    let package_dir = root
        .join("models--meshllm--stream-package")
        .join("snapshots")
        .join("abc123");
    let (_package_ref, manifest_sha) = write_artifact_authorization_package(&package_dir);
    (
        package_dir,
        "hf://meshllm/stream-package@abc123".to_string(),
        manifest_sha,
    )
}

fn write_hf_package_v2_artifact_stream_package(
    root: &std::path::Path,
) -> (std::path::PathBuf, String, String, String) {
    use skippy_package_format::{
        Artifact, ArtifactCatalog, PackageManifest, SourceFile, SourceModel, Tensor,
        TensorCatalog, TensorIntegrity, TensorStorage,
    };

    let package_dir = root
        .join("models--meshllm--stream-package-v2")
        .join("snapshots")
        .join("abc123");
    let primary = b"primary";
    let resident = b"resident";
    let unowned = b"unowned";
    std::fs::create_dir_all(package_dir.join("artifacts")).unwrap();
    std::fs::write(package_dir.join("artifacts/primary.gguf"), primary).unwrap();
    std::fs::write(package_dir.join("artifacts/resident.gguf"), resident).unwrap();
    std::fs::write(package_dir.join("artifacts/unowned.gguf"), unowned).unwrap();

    let mut manifest = PackageManifest {
        schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: "meshllm/stream-package-v2".to_string(),
        source_model: SourceModel {
            sha256: sha256_hex(primary),
            metadata_artifact_id: "primary".to_string(),
            repo: Some("meshllm/stream-package-v2".to_string()),
            revision: Some("abc123".to_string()),
            primary_file: Some("model.gguf".to_string()),
            canonical_ref: None,
            distribution_id: None,
            files: vec![SourceFile {
                path: "model.gguf".to_string(),
                byte_size: primary.len() as u64,
                sha256: sha256_hex(primary),
            }],
        },
        format: "gguf".to_string(),
        layer_count: 1,
        model_metadata: std::collections::BTreeMap::from([(
            "general.architecture".to_string(),
            serde_json::Value::String("llama".to_string()),
        )]),
        artifact_catalog: ArtifactCatalog {
            entries: vec![
                Artifact {
                    id: "primary".to_string(),
                    path: "artifacts/primary.gguf".to_string(),
                    byte_size: primary.len() as u64,
                    sha256: sha256_hex(primary),
                },
                Artifact {
                    id: "resident".to_string(),
                    path: "artifacts/resident.gguf".to_string(),
                    byte_size: resident.len() as u64,
                    sha256: sha256_hex(resident),
                },
                Artifact {
                    id: "unowned".to_string(),
                    path: "artifacts/unowned.gguf".to_string(),
                    byte_size: unowned.len() as u64,
                    sha256: sha256_hex(unowned),
                },
            ],
        },
        tensor_catalog: TensorCatalog {
            entries: vec![Tensor {
                id: "resident-tensor".to_string(),
                name: "blk.0.weight".to_string(),
                ggml_type: 0,
                dimensions: vec![1],
                layer_ordinal: Some(0),
                storage: TensorStorage::Owned {
                    artifact_id: "resident".to_string(),
                    data_offset: 0,
                    stored_length: resident.len() as u64,
                    alignment: 1,
                    integrity: TensorIntegrity::ArtifactSha256,
                },
            }],
        },
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: "test".to_string(),
        generator_version: "test".to_string(),
        created_at_unix_secs: 1,
    };
    manifest.package_id = manifest.computed_package_id().unwrap();
    manifest.validate().unwrap();
    let package_id = manifest.package_id.clone();
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    let manifest_sha = sha256_hex(&manifest_bytes);
    std::fs::write(package_dir.join("model-package.json"), manifest_bytes).unwrap();

    (
        package_dir,
        "hf://meshllm/stream-package-v2@abc123".to_string(),
        manifest_sha,
        package_id,
    )
}

fn verified_owner_summary(owner_id: &str) -> OwnershipSummary {
    OwnershipSummary {
        owner_id: Some(owner_id.to_string()),
        status: OwnershipStatus::Verified,
        verified: true,
        ..OwnershipSummary::default()
    }
}

#[tokio::test]
async fn external_inference_endpoint_models_are_advertised_in_gossip() -> anyhow::Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;
    let resolved_plugins = plugin::ResolvedPlugins {
        externals: vec![],
        inactive: vec![],
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    let plugin_manager = plugin::PluginManager::start(
        &resolved_plugins,
        plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await?;
    plugin_manager
        .set_test_inference_endpoints(vec![plugin::InferenceEndpointRoute {
            plugin_name: "endpoint-plugin".into(),
            endpoint_id: "endpoint-plugin".into(),
            address: "http://127.0.0.1:8000/v1".into(),
            models: vec!["lemonade-small".into()],
        }])
        .await;
    node.set_plugin_manager(plugin_manager).await;

    let announcements = node.collect_announcements().await;
    let local = announcements.last().expect("local announcement");

    assert!(local.models.iter().any(|model| model == "lemonade-small"));
    assert!(
        local
            .serving_models
            .iter()
            .any(|model| model == "lemonade-small")
    );
    assert!(
        local
            .hosted_models
            .as_ref()
            .is_some_and(|models| models.iter().any(|model| model == "lemonade-small"))
    );
    Ok(())
}

#[tokio::test]
async fn control_plane_get_watch_apply_config() -> Result<()> {
    use crate::proto::node::{
        ConfigApplyMode, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry, OwnerControlRequest,
    };

    let owner_keypair = test_owner_keypair(0x91, 0x92);
    let tmp =
        std::env::temp_dir().join(format!("mesh-llm-control-config-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp).ok();

    let (server, _secret_key, config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let (_get_endpoint, mut get_send, mut get_recv, requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut get_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                get_config: Some(crate::proto::node::OwnerControlGetConfigRequest {
                    requester_node_id: requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                }),
                ..owner_control_request(1)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let get_envelope = read_owner_control_envelope(&mut get_recv).await?;
    let get_response = get_envelope
        .response
        .expect("get request should return a response");
    let initial_snapshot = get_response
        .get_config
        .expect("get response should carry get_config")
        .snapshot
        .expect("get response should carry a snapshot");
    assert_eq!(initial_snapshot.revision, 0);
    assert_eq!(initial_snapshot.node_id, server.id().as_bytes().to_vec());

    let (_watch_endpoint, mut watch_send, mut watch_recv, watch_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut watch_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                watch_config: Some(crate::proto::node::OwnerControlWatchConfigRequest {
                    requester_node_id: watch_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    include_snapshot: true,
                }),
                ..owner_control_request(2)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let watch_initial = read_owner_control_envelope(&mut watch_recv).await?;
    let watch_initial_snapshot = watch_initial
        .response
        .expect("watch should return a response")
        .watch_config
        .expect("watch response should carry watch_config")
        .snapshot
        .expect("watch should send an initial snapshot first");
    assert_eq!(watch_initial_snapshot.revision, 0);

    let (_apply_endpoint, mut apply_send, mut apply_recv, apply_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    let applied_config = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![NodeModelEntry {
            model: "test-model.gguf".to_string(),
            mmproj: None,
            ctx_size: Some(4096),
            gpu_id: None,
            model_ref: None,
            mmproj_ref: None,
        }],
        plugins: vec![],
        config_toml: None,
        mesh_requirements: None,
    };
    write_len_prefixed(
        &mut apply_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                    requester_node_id: apply_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    expected_revision: 0,
                    config: Some(applied_config.clone()),
                }),
                ..owner_control_request(3)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let apply_envelope = read_owner_control_envelope(&mut apply_recv).await?;
    let apply_response = apply_envelope
        .response
        .expect("apply should return a response")
        .apply_config
        .expect("apply response should carry apply_config");
    assert!(apply_response.success);
    assert_eq!(apply_response.current_revision, 1);
    assert_eq!(apply_response.apply_mode, ConfigApplyMode::Staged as i32);

    let watch_update = read_owner_control_envelope(&mut watch_recv).await?;
    let watch_update = watch_update
        .response
        .expect("watch update should return a response")
        .watch_config
        .expect("watch update should carry watch_config")
        .update
        .expect("watch stream should emit an update after apply");
    assert_eq!(watch_update.revision, 1);
    assert_eq!(watch_update.config_hash, apply_response.config_hash);

    let persisted_before_noop =
        std::fs::read_to_string(&config_path).expect("config should exist after staged apply");
    write_len_prefixed(
        &mut apply_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                    requester_node_id: apply_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    expected_revision: 1,
                    config: Some(applied_config),
                }),
                ..owner_control_request(4)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let noop_envelope = read_owner_control_envelope(&mut apply_recv).await?;
    let noop_response = noop_envelope
        .response
        .expect("noop apply should return a response")
        .apply_config
        .expect("noop apply should carry apply_config");
    assert!(noop_response.success);
    assert_eq!(noop_response.current_revision, 1);
    assert_eq!(noop_response.apply_mode, ConfigApplyMode::Noop as i32);
    let persisted_after_noop =
        std::fs::read_to_string(&config_path).expect("config should still be readable after noop");
    assert_eq!(persisted_before_noop, persisted_after_noop);

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_watch_observes_apply_revision() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig, OwnerControlRequest};

    let owner_keypair = test_owner_keypair(0x93, 0x94);
    let tmp =
        std::env::temp_dir().join(format!("mesh-llm-control-watch-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let (_watch_endpoint, mut watch_send, mut watch_recv, watch_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut watch_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                watch_config: Some(crate::proto::node::OwnerControlWatchConfigRequest {
                    requester_node_id: watch_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    include_snapshot: true,
                }),
                ..owner_control_request(10)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let initial = read_owner_control_envelope(&mut watch_recv).await?;
    let initial_revision = initial
        .response
        .expect("watch should return a response")
        .watch_config
        .expect("watch should return watch_config")
        .snapshot
        .expect("watch should start with a snapshot")
        .revision;

    let (_apply_endpoint, mut apply_send, mut apply_recv, apply_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut apply_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                    requester_node_id: apply_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    expected_revision: initial_revision,
                    config: Some(NodeConfigSnapshot {
                        version: 1,
                        gpu: Some(NodeGpuConfig {
                            assignment: crate::proto::node::GpuAssignment::Auto as i32,
                        }),
                        models: vec![],
                        plugins: vec![],
                        config_toml: None,
                        mesh_requirements: None,
                    }),
                }),
                ..owner_control_request(11)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let apply = read_owner_control_envelope(&mut apply_recv).await?;
    let applied = apply
        .response
        .expect("apply should return a response")
        .apply_config
        .expect("apply should return apply_config");
    assert!(applied.success);

    let update = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        read_owner_control_envelope(&mut watch_recv),
    )
    .await
    .expect("watch stream should emit an update within 5 seconds")?;
    let update = update
        .response
        .expect("watch update should return a response")
        .watch_config
        .expect("watch update should return watch_config")
        .update
        .expect("watch update should carry an update payload");
    assert_eq!(update.revision, initial_revision + 1);
    assert_eq!(update.config_hash, applied.config_hash);

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_watch_without_snapshot_starts_with_accepted() -> Result<()> {
    use crate::proto::node::OwnerControlRequest;

    let owner_keypair = test_owner_keypair(0xA1, 0xA2);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-watch-no-snapshot-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let (_watch_endpoint, mut watch_send, mut watch_recv, watch_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut watch_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                watch_config: Some(crate::proto::node::OwnerControlWatchConfigRequest {
                    requester_node_id: watch_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    include_snapshot: false,
                }),
                ..owner_control_request(12)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;

    let initial = read_owner_control_envelope(&mut watch_recv).await?;
    let watch = initial
        .response
        .expect("watch should return a response")
        .watch_config
        .expect("watch should return watch_config");
    assert!(watch.snapshot.is_none());
    assert!(watch.update.is_none());
    let accepted = watch
        .accepted
        .expect("watch without snapshot should start with accepted");
    assert_eq!(accepted.target_node_id, server.id().as_bytes().to_vec());

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_watch_without_snapshot_observes_apply_revision() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig, OwnerControlRequest};

    let owner_keypair = test_owner_keypair(0xA3, 0xA4);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-watch-no-snapshot-update-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let initial_revision = { server.config_state.lock().await.revision() };
    let (_watch_endpoint, mut watch_send, mut watch_recv, watch_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut watch_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                watch_config: Some(crate::proto::node::OwnerControlWatchConfigRequest {
                    requester_node_id: watch_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    include_snapshot: false,
                }),
                ..owner_control_request(13)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let accepted = read_owner_control_envelope(&mut watch_recv).await?;
    assert!(
        accepted
            .response
            .expect("watch should return a response")
            .watch_config
            .expect("watch should return watch_config")
            .accepted
            .is_some()
    );

    let (_apply_endpoint, mut apply_send, mut apply_recv, apply_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut apply_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                    requester_node_id: apply_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    expected_revision: initial_revision,
                    config: Some(NodeConfigSnapshot {
                        version: 1,
                        gpu: Some(NodeGpuConfig {
                            assignment: crate::proto::node::GpuAssignment::Auto as i32,
                        }),
                        models: vec![],
                        plugins: vec![],
                        config_toml: None,
                        mesh_requirements: None,
                    }),
                }),
                ..owner_control_request(14)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let apply = read_owner_control_envelope(&mut apply_recv).await?;
    let applied = apply
        .response
        .expect("apply should return a response")
        .apply_config
        .expect("apply should return apply_config");
    assert!(applied.success);

    let update = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        read_owner_control_envelope(&mut watch_recv),
    )
    .await
    .expect("watch stream should emit an update within 5 seconds")?;
    let update = update
        .response
        .expect("watch update should return a response")
        .watch_config
        .expect("watch update should return watch_config")
        .update
        .expect("watch update should carry an update payload");
    assert_eq!(update.revision, initial_revision + 1);
    assert_eq!(update.config_hash, applied.config_hash);

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_apply_rejects_stale_revision() -> Result<()> {
    use crate::proto::node::{
        NodeConfigSnapshot, NodeGpuConfig, OwnerControlErrorCode, OwnerControlRequest,
    };

    let owner_keypair = test_owner_keypair(0x95, 0x96);
    let tmp =
        std::env::temp_dir().join(format!("mesh-llm-control-stale-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let initial_hash = { *server.config_state.lock().await.config_hash() };

    let (_apply_endpoint, mut apply_send, mut apply_recv, apply_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    let apply_once = |request_id, expected_revision| crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                requester_node_id: apply_requester_id.as_bytes().to_vec(),
                target_node_id: server.id().as_bytes().to_vec(),
                expected_revision,
                config: Some(NodeConfigSnapshot {
                    version: 1,
                    gpu: Some(NodeGpuConfig {
                        assignment: crate::proto::node::GpuAssignment::Auto as i32,
                    }),
                    models: vec![crate::proto::node::NodeModelEntry {
                        model: "stale-test-model.gguf".to_string(),
                        mmproj: None,
                        ctx_size: Some(2048),
                        gpu_id: None,
                        model_ref: None,
                        mmproj_ref: None,
                    }],
                    plugins: vec![],
                    config_toml: None,
                    mesh_requirements: None,
                }),
            }),
            ..owner_control_request(request_id)
        }),
        response: None,
        error: None,
    };

    write_len_prefixed(&mut apply_send, &apply_once(20, 0).encode_to_vec()).await?;
    let first = read_owner_control_envelope(&mut apply_recv).await?;
    assert!(
        first
            .response
            .expect("first apply should return a response")
            .apply_config
            .expect("first apply should return apply_config")
            .success
    );

    let hash_after_first = { *server.config_state.lock().await.config_hash() };
    write_len_prefixed(&mut apply_send, &apply_once(21, 0).encode_to_vec()).await?;
    let stale = read_owner_control_envelope(&mut apply_recv).await?;
    let stale_error = stale
        .error
        .expect("stale apply should return an error envelope");
    assert_eq!(
        stale_error.code,
        OwnerControlErrorCode::RevisionConflict as i32
    );
    assert_eq!(stale_error.request_id, Some(21));
    assert_eq!(stale_error.current_revision, Some(1));
    assert_eq!(
        { *server.config_state.lock().await.config_hash() },
        hash_after_first
    );
    assert_ne!(initial_hash, hash_after_first);

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_apply_rejects_malformed_full_config_toml() -> Result<()> {
    use crate::proto::node::{
        NodeConfigSnapshot, NodeGpuConfig, OwnerControlErrorCode, OwnerControlRequest,
    };

    let owner_keypair = test_owner_keypair(0x97, 0x98);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-invalid-config-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;

    let initial_revision = { server.config_state.lock().await.revision() };
    let initial_hash = { *server.config_state.lock().await.config_hash() };

    let (_apply_endpoint, mut apply_send, mut apply_recv, apply_requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut apply_send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                apply_config: Some(crate::proto::node::OwnerControlApplyConfigRequest {
                    requester_node_id: apply_requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    expected_revision: initial_revision,
                    config: Some(NodeConfigSnapshot {
                        version: 1,
                        gpu: Some(NodeGpuConfig {
                            assignment: crate::proto::node::GpuAssignment::Auto as i32,
                        }),
                        models: vec![],
                        plugins: vec![],
                        config_toml: Some("not valid toml = [".to_string()),
                        mesh_requirements: None,
                    }),
                }),
                ..owner_control_request(22)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;

    let rejected = read_owner_control_envelope(&mut apply_recv).await?;
    let error = rejected
        .error
        .expect("malformed full config should return an error envelope");
    assert_eq!(error.code, OwnerControlErrorCode::BadRequest as i32);
    assert_eq!(error.request_id, Some(22));
    assert!(error.message.contains("invalid full config_toml payload"));
    assert_eq!(
        server.config_state.lock().await.revision(),
        initial_revision
    );
    assert_eq!(
        *server.config_state.lock().await.config_hash(),
        initial_hash
    );

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn owner_control_client_reuses_connection_for_sequential_requests() -> Result<()> {
    use mesh_client::{
        ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneConnection, InviteToken,
    };
    use std::str::FromStr;

    let owner_keypair = test_owner_keypair(0x89, 0x8a);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-client-reuse-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let endpoint_token = server
        .control_endpoint()
        .await
        .expect("control endpoint should be available for owner-control client test");
    let client = ClientBuilder::new(
        owner_keypair.clone(),
        InviteToken::from_str("mesh-test:owner-control-client-reuse")
            .map_err(|error| anyhow::anyhow!(error))?,
    )
    .build()?;
    let connection = client
        .connect_control_plane(
            ControlPlaneBootstrapOptions::new().with_control_endpoint(endpoint_token),
        )
        .await?;
    let ControlPlaneConnection::OwnerControl(control_client) = connection;

    let snapshot = control_client.get_config().await?;
    let config = snapshot
        .config
        .clone()
        .expect("get-config snapshot should include config");
    // `apply_config` already has the client's bounded unary-response timeout.
    // A shorter outer deadline makes this connection-reuse assertion flaky on
    // loaded CI hosts without testing any additional production behavior.
    let apply = control_client
        .apply_config(snapshot.revision, config)
        .await?;

    assert!(apply.success);
    assert_eq!(apply.current_revision, snapshot.revision + 1);

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
#[serial]
async fn control_plane_refresh_inventory() -> Result<()> {
    use crate::proto::node::OwnerControlRequest;

    let owner_keypair = test_owner_keypair(0x97, 0x98);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-refresh-{}",
        rand::random::<u64>()
    ));
    let hf_cache = tmp.join("hf-cache");
    std::fs::create_dir_all(&hf_cache).ok();
    let _hf_cache_guard = EnvVarGuard::set("HF_HUB_CACHE", &hf_cache);
    let gguf_path = hf_cache.join("Refresh-Test-Q4_K_M.gguf");
    let file = std::fs::File::create(&gguf_path)?;
    file.set_len(600_000_000)?;

    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let expected_model_ref = crate::models::model_ref_for_path(&gguf_path);
    assert!(server.available_models().await.is_empty());

    let (_refresh_endpoint, mut refresh_send, mut refresh_recv, requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    let refresh_request = |request_id| crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            refresh_inventory: Some(crate::proto::node::OwnerControlRefreshInventoryRequest {
                requester_node_id: requester_id.as_bytes().to_vec(),
                target_node_id: server.id().as_bytes().to_vec(),
            }),
            ..owner_control_request(request_id)
        }),
        response: None,
        error: None,
    };

    write_len_prefixed(&mut refresh_send, &refresh_request(30).encode_to_vec()).await?;
    let first = read_owner_control_envelope(&mut refresh_recv).await?;
    let first_refresh = first
        .response
        .expect("refresh should return a response")
        .refresh_inventory
        .expect("refresh should return refresh_inventory");
    let first_inventory = first_refresh
        .inventory
        .expect("refresh should include inventory detail");
    let first_snapshot = first_refresh
        .snapshot
        .expect("refresh should include a config snapshot");
    assert_eq!(first_snapshot.node_id, server.id().as_bytes().to_vec());
    assert!(
        server
            .available_models()
            .await
            .contains(&expected_model_ref)
    );
    let inventory_snapshot = server.runtime_data_collector().local_inventory_snapshot();
    assert!(inventory_snapshot.model_names.contains(&expected_model_ref));
    let mut expected_refs = inventory_snapshot
        .model_names
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    expected_refs.sort();
    assert_eq!(
        first_inventory
            .entries
            .iter()
            .map(|entry| entry.canonical_model_ref.clone())
            .collect::<Vec<_>>(),
        expected_refs
    );
    assert_eq!(
        crate::proto::node::OwnerControlRefreshInventoryDisposition::try_from(
            first_inventory.disposition
        ),
        Ok(crate::proto::node::OwnerControlRefreshInventoryDisposition::Executed)
    );

    write_len_prefixed(&mut refresh_send, &refresh_request(31).encode_to_vec()).await?;
    let second = read_owner_control_envelope(&mut refresh_recv).await?;
    let second_snapshot = second
        .response
        .expect("second refresh should return a response")
        .refresh_inventory
        .expect("second refresh should return refresh_inventory")
        .snapshot
        .expect("second refresh should include a config snapshot");
    assert_eq!(first_snapshot.revision, second_snapshot.revision);
    assert_eq!(
        server
            .available_models()
            .await
            .iter()
            .filter(|model| *model == &expected_model_ref)
            .count(),
        1
    );

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
#[serial]
async fn failed_inventory_refresh_preserves_last_good_snapshot_and_advertisement() -> Result<()> {
    let owner_keypair = test_owner_keypair(0x99, 0x9a);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-refresh-failure-{}",
        rand::random::<u64>()
    ));
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let seeded = crate::models::LocalModelInventorySnapshot {
        model_names: std::collections::HashSet::from(["last-good-model".to_string()]),
        ..Default::default()
    };
    server
        .runtime_data_collector()
        .coalesce_local_inventory_scan({
            let seeded = seeded.clone();
            move || seeded
        })
        .await?;
    server
        .set_available_models(vec!["last-good-model".to_string()])
        .await;

    let error = server
        .refresh_local_inventory_snapshot_with(|| {
            Err(crate::runtime_data::InventoryScanError::LoaderFailed(
                "forced failure".to_string(),
            ))
        })
        .await
        .expect_err("forced scan failure should reach the caller");

    assert!(error.to_string().contains("forced failure"));
    assert_eq!(
        server.runtime_data_collector().local_inventory_snapshot(),
        seeded
    );
    assert_eq!(
        server.available_models().await,
        vec!["last-good-model".to_string()]
    );

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn scan_timeout_releases_caller_while_reconciliation_finishes() -> Result<()> {
    let server = Node::new_for_tests(super::NodeRole::Worker).await?;
    server
        .set_available_models(vec!["last-good-model".to_string()])
        .await;
    let next = crate::models::LocalModelInventorySnapshot {
        model_names: std::collections::HashSet::from(["eventual-model".to_string()]),
        ..Default::default()
    };

    let timed_out = tokio::time::timeout(
        std::time::Duration::from_millis(20),
        server.refresh_local_inventory_snapshot_with({
            let next = next.clone();
            move || {
                std::thread::sleep(std::time::Duration::from_millis(100));
                Ok(next)
            }
        }),
    )
    .await;
    assert!(timed_out.is_err(), "slow scan should release its caller");

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if server.available_models().await == vec!["eventual-model".to_string()] {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("detached scan reconciliation should finish");
    assert_eq!(
        server.runtime_data_collector().local_inventory_snapshot(),
        next
    );
    Ok(())
}
