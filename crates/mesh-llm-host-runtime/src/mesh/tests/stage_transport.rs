#[tokio::test]
#[serial]
async fn artifact_transfer_peer_eligibility_ignores_public_advertisement_by_default() -> Result<()>
{
    let _transfer_guard = EnvVarGuard::unset("MESH_LLM_ARTIFACT_TRANSFER");
    let node = make_test_node(super::NodeRole::Worker).await?;
    let mut peer = make_test_peer_info(make_test_endpoint_id(0x71));
    peer.artifact_transfer_supported = true;

    assert!(
        !node.artifact_transfer_allowed_for_peer(&peer).await,
        "raw public artifact-transfer advertisement must not make a peer eligible"
    );

    Ok(())
}

#[tokio::test]
#[serial]
async fn artifact_transfer_peer_eligibility_allows_same_or_trusted_owner() -> Result<()> {
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "trusted");
    let node = make_test_node(super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");

    let mut same_owner = make_test_peer_info(make_test_endpoint_id(0x72));
    same_owner.artifact_transfer_supported = true;
    same_owner.owner_summary = verified_owner_summary("owner-a");
    assert!(node.artifact_transfer_allowed_for_peer(&same_owner).await);

    let mut trusted_owner = make_test_peer_info(make_test_endpoint_id(0x73));
    trusted_owner.artifact_transfer_supported = true;
    trusted_owner.owner_summary = verified_owner_summary("owner-b");
    {
        let mut store = node.trust_store.lock().await;
        store.add_trusted_owner("owner-b".to_string(), None);
    }
    assert!(
        node.artifact_transfer_allowed_for_peer(&trusted_owner)
            .await
    );

    let mut untrusted_owner = make_test_peer_info(make_test_endpoint_id(0x74));
    untrusted_owner.artifact_transfer_supported = true;
    untrusted_owner.owner_summary = verified_owner_summary("owner-c");
    assert!(
        !node
            .artifact_transfer_allowed_for_peer(&untrusted_owner)
            .await
    );

    Ok(())
}

#[test]
fn artifact_transfer_authorization_is_limited_to_stage_assignment() {
    let package = tempfile::tempdir().unwrap();
    let (package_ref, manifest_sha256) = write_artifact_authorization_package(package.path());
    let stage0 = make_test_endpoint_id(0x91);
    let stage1 = make_test_endpoint_id(0x92);
    let topology = StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: package_ref.clone(),
        manifest_sha256: manifest_sha256.clone(),
        admissions: Default::default(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: stage0,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: stage1,
                layer_start: 1,
                layer_end: 2,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            },
        ],
    };
    let request = |relative_path: &str, expected_size: u64, expected_sha256: String| {
        skippy_stage_proto::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: stage0.as_bytes().to_vec(),
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            relative_path: relative_path.to_string(),
            offset: 0,
            expected_size: Some(expected_size),
            expected_sha256: Some(expected_sha256),
        }
    };

    let layer0 = request("layers/layer-000.gguf", 8, sha256_hex(b"layer000"));
    assert!(
        artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &layer0,
        )
        .unwrap()
    );

    let mut wrong_topology = layer0.clone();
    wrong_topology.topology_id = "other-topology".to_string();
    assert!(
        !artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &wrong_topology,
        )
        .unwrap()
    );

    let layer1 = request("layers/layer-001.gguf", 8, sha256_hex(b"layer001"));
    assert!(
        !artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &layer1,
        )
        .unwrap()
    );

    let projector = request("projectors/mmproj.gguf", 9, sha256_hex(b"projector"));
    assert!(
        artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &projector,
        )
        .unwrap()
    );

    let manifest = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: stage1.as_bytes().to_vec(),
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        stage_id: "stage-1".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "model-package.json".to_string(),
        offset: 0,
        expected_size: None,
        expected_sha256: None,
    };
    assert!(
        artifact_transfer_allowed_by_topology(&[topology], stage1, package.path(), &manifest)
            .unwrap()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_uses_mesh_subprotocol_and_rejects_dedicated_alpn() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "1");
    let (package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-stream-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-stream-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact".to_string(),
            run_id: "run-artifact".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            admissions: Default::default(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact".to_string(),
        run_id: "run-artifact".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(response.accepted, "artifact response: {:?}", response.error);
    assert_eq!(response.total_size, 8);
    let expected_sha = sha256_hex(b"layer000");
    assert_eq!(response.sha256.as_deref(), Some(expected_sha.as_str()));
    let mut bytes = vec![0u8; response.total_size as usize];
    recv.read_exact(&mut bytes).await?;
    assert_eq!(bytes, b"layer000");

    let mut resume_request = request.clone();
    resume_request.offset = 5;
    let (mut resume_send, mut resume_recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut resume_send, &resume_request.encode_to_vec()).await?;
    resume_send.finish()?;
    let resume_response_buf = read_len_prefixed(&mut resume_recv).await?;
    let resume_response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(resume_response_buf.as_slice())?;
    assert!(
        resume_response.accepted,
        "resume artifact response: {:?}",
        resume_response.error
    );
    assert_eq!(resume_response.total_size, 8);
    assert_eq!(
        resume_response.sha256.as_deref(),
        Some(expected_sha.as_str())
    );
    let mut resumed_bytes =
        vec![0u8; (resume_response.total_size - resume_request.offset) as usize];
    resume_recv.read_exact(&mut resumed_bytes).await?;
    assert_eq!(resumed_bytes, b"000");

    let conn = client.stage_connection_to_peer(server_id).await?;
    let (mut dedicated_send, mut dedicated_recv) = conn.open_bi().await?;
    dedicated_send
        .write_all(&[skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER])
        .await?;
    write_len_prefixed(&mut dedicated_send, &request.encode_to_vec()).await?;
    dedicated_send.finish()?;
    assert!(
        read_len_prefixed(&mut dedicated_recv).await.is_err(),
        "dedicated stage ALPN must reject artifact transfer"
    );
    assert!(package_dir.join("model-package.json").is_file());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn package_v2_artifact_transfer_populates_empty_peer_cache_with_only_admitted_artifacts()
-> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "1");
    let (package_dir, package_ref, manifest_sha256, package_id) =
        write_hf_package_v2_artifact_stream_package(cache.path());
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server.set_mesh_id("artifact-transfer-v2-mesh".to_string()).await;
    client.set_mesh_id("artifact-transfer-v2-mesh".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;

    let mut admission = crate::inference::skippy::test_stage_admission(0, 1);
    admission.package_id = package_id;
    admission.resident_tensor_ids = vec!["resident-tensor".to_string()];
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact-v2".to_string(),
            run_id: "run-artifact-v2".to_string(),
            model_id: "model-artifact-v2".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            admissions: std::collections::BTreeMap::from([(
                "stage-0".to_string(),
                admission.clone(),
            )]),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request_for = |relative_path: &str, bytes: &[u8]| {
        skippy_stage_proto::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: client_id.as_bytes().to_vec(),
            topology_id: "topology-artifact-v2".to_string(),
            run_id: "run-artifact-v2".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            relative_path: relative_path.to_string(),
            offset: 0,
            expected_size: Some(bytes.len() as u64),
            expected_sha256: Some(sha256_hex(bytes)),
        }
    };

    let peer_package = cache
        .path()
        .join("models--meshllm--stream-package-v2")
        .join("empty-peer-cache");
    assert!(!peer_package.exists());
    let mut load = stage_load_request();
    load.topology_id = "topology-artifact-v2".to_string();
    load.run_id = "run-artifact-v2".to_string();
    load.stage_id = "stage-0".to_string();
    load.stage_index = 0;
    load.layer_start = 0;
    load.layer_end = 1;
    load.package_ref = package_ref.clone();
    load.manifest_sha256 = manifest_sha256.clone();
    load.admission = admission;

    let manifest = crate::models::artifact_transfer::manifest_artifact_request(
        &package_ref,
        &manifest_sha256,
    )?;
    let manifest_destination =
        crate::models::artifact_transfer::local_artifact_path(&peer_package, &manifest);
    client
        .fetch_artifact_from_peer(server_id, &load, &manifest, &manifest_destination)
        .await?;
    let carrier = crate::models::artifact_transfer::package_v2_metadata_carrier_request(
        &peer_package,
        &package_ref,
        &manifest_sha256,
    )?;
    let carrier_destination =
        crate::models::artifact_transfer::local_artifact_path(&peer_package, &carrier);
    client
        .fetch_artifact_from_peer(server_id, &load, &carrier, &carrier_destination)
        .await?;
    let admitted = crate::models::artifact_transfer::required_admitted_stage_package_artifacts(
        &peer_package,
        &package_ref,
        &manifest_sha256,
        &load.admission,
    )?;
    for artifact in &admitted {
        let destination =
            crate::models::artifact_transfer::local_artifact_path(&peer_package, artifact);
        client
            .fetch_artifact_from_peer(server_id, &load, artifact, &destination)
            .await?;
    }
    assert_eq!(
        std::fs::read(peer_package.join("artifacts/resident.gguf"))?,
        std::fs::read(package_dir.join("artifacts/resident.gguf"))?
    );
    assert!(peer_package.join("shared/metadata.gguf").is_file());
    assert!(!peer_package.join("artifacts/primary.gguf").exists());
    assert!(!peer_package.join("artifacts/unowned.gguf").exists());

    let unowned_bytes = std::fs::read(package_dir.join("artifacts/unowned.gguf"))?;
    let unowned = request_for("artifacts/unowned.gguf", &unowned_bytes);
    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &unowned.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(!response.accepted);
    assert_eq!(response.error.as_deref(), Some("artifact unavailable"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_rejects_corrupt_same_size_cached_artifact() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "1");
    let (package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    std::fs::write(package_dir.join("layers/layer-000.gguf"), b"corrupt!").unwrap();
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-corrupt-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-corrupt-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact-corrupt".to_string(),
            run_id: "run-artifact-corrupt".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            admissions: Default::default(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact-corrupt".to_string(),
        run_id: "run-artifact-corrupt".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(!response.accepted);
    assert_eq!(response.error.as_deref(), Some("artifact unavailable"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_rejects_public_mesh_without_opt_in() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::unset("MESH_LLM_ARTIFACT_TRANSFER");
    let (_package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-disabled-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-disabled-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact-disabled".to_string(),
            run_id: "run-artifact-disabled".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            admissions: Default::default(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact-disabled".to_string(),
        run_id: "run-artifact-disabled".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(!response.accepted);
    assert_eq!(
        response.error.as_deref(),
        Some("artifact transfer disabled")
    );

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn artifact_transfer_body_read_has_idle_timeout() {
    let (_writer, mut reader) = tokio::io::duplex(8);
    let mut buffer = [0u8; 4];

    let error = read_artifact_transfer_chunk(
        &mut reader,
        &mut buffer,
        std::time::Duration::from_millis(10),
    )
    .await
    .expect_err("stalled body read must time out");

    assert!(
        error
            .to_string()
            .contains("artifact transfer body read idle timeout")
    );
}

#[tokio::test(start_paused = true)]
async fn artifact_transfer_body_write_has_idle_timeout() {
    let (mut writer, _reader) = tokio::io::duplex(1);
    let bytes = [1u8; 1024];

    let error =
        write_artifact_transfer_chunk(&mut writer, &bytes, std::time::Duration::from_millis(10))
            .await
            .expect_err("stalled body write must time out");

    assert!(
        error
            .to_string()
            .contains("artifact transfer body write idle timeout")
    );
}

#[tokio::test(start_paused = true)]
async fn stage_stream_type_read_has_timeout() {
    let (_writer, mut reader) = tokio::io::duplex(1);

    let error = read_stage_stream_type(&mut reader)
        .await
        .expect_err("stalled stage stream type read must time out");

    assert!(
        error
            .to_string()
            .contains("timeout reading skippy stage stream type")
    );
}

#[tokio::test(start_paused = true)]
async fn local_stage_control_response_wait_has_timeout() {
    let (_response_tx, response_rx) = tokio::sync::oneshot::channel();

    let error =
        wait_local_stage_control_response(response_rx, std::time::Duration::from_millis(10))
            .await
            .expect_err("stalled local stage control response must time out");

    assert!(
        error
            .to_string()
            .contains("timeout waiting for local stage control response")
    );
}

#[tokio::test(start_paused = true)]
async fn plugin_frame_prefix_and_body_reads_have_timeouts() {
    use tokio::io::AsyncWriteExt as _;

    let (_prefix_writer, mut prefix_reader) = tokio::io::duplex(1);
    let prefix_error = read_plugin_frame_bytes(&mut prefix_reader, 1024)
        .await
        .expect_err("stalled prefix read must time out");
    assert!(
        prefix_error
            .to_string()
            .contains("timeout reading plugin frame prefix")
    );

    let (mut body_writer, mut body_reader) = tokio::io::duplex(8);
    body_writer.write_all(&4u32.to_le_bytes()).await.unwrap();
    let body_error = read_plugin_frame_bytes(&mut body_reader, 1024)
        .await
        .expect_err("stalled body read must time out");
    assert!(
        body_error
            .to_string()
            .contains("timeout reading plugin frame body")
    );
}

#[tokio::test]
async fn stage_transport_bridge_reservation_rejects_duplicate_request() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0x93);

    let first = node
        .ensure_stage_transport_bridge(peer_id, "topology-race", "run-race", "stage-race")
        .await?;
    assert!(first.starts_with("127.0.0.1:"));
    let key = stage_runtime_status_key("topology-race", "run-race", "stage-race");
    assert!(node.stage_transport_bridge_is_running(&key).await);

    let duplicate = node
        .ensure_stage_transport_bridge(peer_id, "topology-race", "run-race", "stage-race")
        .await
        .expect_err("duplicate bridge setup must be rejected");
    assert!(duplicate.to_string().contains("already exists"));

    node.stop_stage_transport_bridge("topology-race", "run-race", "stage-race")
        .await;

    let recreated = node
        .ensure_stage_transport_bridge(peer_id, "topology-race", "run-race", "stage-race")
        .await?;
    assert!(recreated.starts_with("127.0.0.1:"));
    node.stop_stage_transport_bridge("topology-race", "run-race", "stage-race")
        .await;
    Ok(())
}

#[tokio::test]
async fn stage_transport_bridge_reserved_state_rejects_duplicate_request() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0x94);
    let key = stage_runtime_status_key("topology-reserved", "run-reserved", "stage-reserved");
    let owner = node
        .reserve_stage_transport_bridge_for_test(key.clone())
        .await;

    let duplicate = node
        .ensure_stage_transport_bridge(
            peer_id,
            "topology-reserved",
            "run-reserved",
            "stage-reserved",
        )
        .await
        .expect_err("reserved bridge setup must reject duplicates before an accept task exists");

    assert!(duplicate.to_string().contains("already exists"));
    assert!(
        node.stage_transport_bridge_owner_matches(&key, &owner)
            .await
    );
    node.clear_stage_transport_bridge_for_test(&key).await;
    Ok(())
}

#[tokio::test]
async fn stage_transport_bridge_publish_aborts_handle_when_reservation_was_stopped() -> Result<()> {
    struct AbortNotice(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for AbortNotice {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    let node = make_test_node(super::NodeRole::Worker).await?;
    let key = stage_runtime_status_key("topology-stop", "run-stop", "stage-stop");
    let owner = node
        .reserve_stage_transport_bridge_for_test(key.clone())
        .await;
    node.stop_stage_transport_bridge("topology-stop", "run-stop", "stage-stop")
        .await;
    let (abort_tx, abort_rx) = tokio::sync::oneshot::channel();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(async move {
        let _notice = AbortNotice(Some(abort_tx));
        let _ = started_tx.send(());
        std::future::pending::<()>().await;
    });
    tokio::time::timeout(std::time::Duration::from_millis(250), started_rx)
        .await
        .expect("test accept-loop handle must start before publishing")
        .expect("test accept-loop handle must report startup");

    let published = node
        .publish_stage_transport_bridge(key.clone(), owner, handle)
        .await;

    assert!(!published);
    tokio::time::timeout(std::time::Duration::from_millis(250), abort_rx)
        .await
        .expect("cancelled reservation must abort the escaped accept-loop handle")
        .expect("abort notice sender must be dropped by task cancellation");
    assert!(!node.stage_transport_bridge_exists(&key).await);
    Ok(())
}

#[tokio::test]
async fn stage_transport_bridge_cleanup_removes_only_owning_reservation() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let key = stage_runtime_status_key("topology-cleanup", "run-cleanup", "stage-cleanup");
    let stale_owner = node
        .reserve_stage_transport_bridge_for_test(key.clone())
        .await;
    let current_owner = node
        .reserve_stage_transport_bridge_for_test(stage_runtime_status_key(
            "topology-cleanup",
            "run-cleanup",
            "stage-current",
        ))
        .await;
    node.insert_reserved_stage_transport_bridge_for_test(key.clone(), current_owner.clone())
        .await;

    let stale_cleanup = node
        .remove_stage_transport_bridge_if_owner(&key, &stale_owner)
        .await;

    assert!(stale_cleanup.is_none());
    assert!(
        node.stage_transport_bridge_owner_matches(&key, &current_owner)
            .await
    );
    let current_cleanup = node
        .remove_stage_transport_bridge_if_owner(&key, &current_owner)
        .await;
    assert!(current_cleanup.is_some());
    assert!(!node.stage_transport_bridge_exists(&key).await);
    Ok(())
}

#[test]
fn artifact_transfer_invalid_resume_offset_removes_preserved_partial() {
    let temp = tempfile::tempdir().unwrap();
    let partial = temp.path().join(".model-package.json.stale.part");
    std::fs::write(&partial, b"stale manifest bytes").unwrap();
    let mut guard = PartialArtifactGuard::preserve_on_error(partial.clone());
    let response = skippy_stage_proto::StageArtifactTransferResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        accepted: false,
        total_size: 8,
        sha256: Some(sha256_hex(b"manifest")),
        error: Some(ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR.to_string()),
    };

    Node::remove_invalid_resume_partial(&mut guard, 128, &response);

    assert!(!partial.exists());
}

#[test]
fn artifact_transfer_smaller_resume_response_removes_preserved_partial() {
    let temp = tempfile::tempdir().unwrap();
    let partial = temp.path().join(".model-package.json.oversized.part");
    std::fs::write(&partial, b"stale manifest bytes").unwrap();
    let mut guard = PartialArtifactGuard::preserve_on_error(partial.clone());
    let response = skippy_stage_proto::StageArtifactTransferResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        accepted: true,
        total_size: 8,
        sha256: Some(sha256_hex(b"manifest")),
        error: None,
    };

    Node::remove_invalid_resume_partial(&mut guard, 128, &response);

    assert!(!partial.exists());
}

#[test]
fn partial_artifact_guard_removes_armed_partial_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(".artifact.part");
    std::fs::write(&path, b"partial").unwrap();

    {
        let _guard = PartialArtifactGuard::new(path.clone());
    }

    assert!(!path.exists());
}

#[test]
fn partial_artifact_guard_preserves_disarmed_installed_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(".artifact.part");
    std::fs::write(&path, b"partial").unwrap();

    {
        let mut guard = PartialArtifactGuard::new(path.clone());
        guard.disarm();
    }

    assert!(path.exists());
}
