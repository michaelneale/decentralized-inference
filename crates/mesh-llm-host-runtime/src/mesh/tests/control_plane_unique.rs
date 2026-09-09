#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_plane_legacy_compat_new_client_prefers_control_alpn() -> Result<()> {
    use crate::proto::node::OwnerControlRequest;

    let owner_keypair = test_owner_keypair(0xa3, 0xa4);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-plane-prefers-control-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();

    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let control_addr = Node::decode_invite_token(
        &server
            .control_endpoint()
            .await
            .expect("owner-controlled node should expose control endpoint"),
    )?;

    let wrong_alpn_client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec(), ALPN_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    assert!(
        wrong_alpn_client
            .connect(control_addr.clone(), ALPN_V1)
            .await
            .is_err()
    );

    let (_endpoint, mut send, mut recv, requester_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 41,
                get_config: Some(crate::proto::node::OwnerControlGetConfigRequest {
                    requester_node_id: requester_id.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                }),
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;

    let envelope = read_owner_control_envelope(&mut recv).await?;
    let snapshot = envelope
        .response
        .expect("owner-control request should receive response")
        .get_config
        .expect("response should carry get_config result")
        .snapshot
        .expect("get_config should return initial snapshot");
    assert_eq!(snapshot.node_id, server.id().as_bytes().to_vec());

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_plane_legacy_compat_control_alpn_rejects_legacy_frames() -> Result<()> {
    let owner_keypair = test_owner_keypair(0xa5, 0xa6);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-plane-legacy-json-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();

    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let control_addr = Node::decode_invite_token(
        &server
            .control_endpoint()
            .await
            .expect("owner-controlled node should expose control endpoint"),
    )?;

    let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let conn = client.connect(control_addr, ALPN_CONTROL_V1).await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    write_len_prefixed(&mut send, br#"{"request_id":7,"command":"GetConfig"}"#).await?;

    let rejection = read_owner_control_envelope(&mut recv).await?;
    assert_eq!(
        crate::proto::node::OwnerControlErrorCode::try_from(
            rejection.error.expect("legacy json should be rejected").code,
        )
        .unwrap(),
        crate::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
    );

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_plane_validation_error_preserves_request_id() -> Result<()> {
    let owner_keypair = test_owner_keypair(0xb5, 0xb6);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-plane-invalid-command-{}",
        rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).ok();

    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let (_endpoint, mut send, mut recv, _endpoint_id) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(crate::proto::node::OwnerControlRequest {
                request_id: 7,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;

    let rejection = read_owner_control_envelope(&mut recv).await?;
    let error = rejection
        .error
        .expect("invalid command should be rejected with an error envelope");
    assert_eq!(
        crate::proto::node::OwnerControlErrorCode::try_from(error.code).unwrap(),
        crate::proto::node::OwnerControlErrorCode::UnknownCommand
    );
    assert_eq!(error.request_id, Some(7));

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[test]
fn pinned_gpu_runtime_push_rejects_invalid_pushed_pinned_config_before_apply() {
    let config = crate::plugin::MeshConfig {
        gpu: crate::plugin::GpuConfig {
            assignment: crate::plugin::GpuAssignment::Pinned,
            ..Default::default()
        },
        models: vec![crate::plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..crate::plugin::MeshConfig::default()
    };
    let gpus = vec![crate::system::hardware::GpuFacts {
        index: 0,
        display_name: "GPU 0".into(),
        backend_device: Some("CUDA0".into()),
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: Some("pci:0000:65:00.0".into()),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }];

    let err = preflight_pushed_config_for_current_node_with_gpus(&config, &gpus).unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("did not match any available pinnable GPU"));
}

#[test]
fn pinned_gpu_runtime_push_accepts_valid_pushed_pinned_config() {
    let config = crate::plugin::MeshConfig {
        gpu: crate::plugin::GpuConfig {
            assignment: crate::plugin::GpuAssignment::Pinned,
            ..Default::default()
        },
        models: vec![crate::plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("uuid:GPU-123".into()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..crate::plugin::MeshConfig::default()
    };
    let gpus = vec![crate::system::hardware::GpuFacts {
        index: 3,
        display_name: "GPU 3".into(),
        backend_device: Some("CUDA3".into()),
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: Some("uuid:GPU-123".into()),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }];

    preflight_pushed_config_for_current_node_with_gpus(&config, &gpus).unwrap();
}

#[test]
fn pinned_gpu_runtime_push_rejects_resolved_gpu_without_backend_device() {
    let config = crate::plugin::MeshConfig {
        gpu: crate::plugin::GpuConfig {
            assignment: crate::plugin::GpuAssignment::Pinned,
            ..Default::default()
        },
        models: vec![crate::plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("uuid:GPU-123".into()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..crate::plugin::MeshConfig::default()
    };
    let gpus = vec![crate::system::hardware::GpuFacts {
        index: 3,
        display_name: "GPU 3".into(),
        backend_device: None,
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: Some("uuid:GPU-123".into()),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }];

    let err = preflight_pushed_config_for_current_node_with_gpus(&config, &gpus).unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("without a backend_device"));
}
