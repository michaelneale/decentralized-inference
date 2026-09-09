use base64::Engine;
use iroh::{Endpoint, EndpointAddr, SecretKey};
use mesh_client::proto::node::{
    CompactModelMetadata, ConfigApplyMode, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry,
    OwnerControlEnvelope, OwnerControlErrorCode, OwnerControlInventoryEntry,
    OwnerControlKvCacheOperation, OwnerControlRefreshInventory,
    OwnerControlRefreshInventoryDisposition,
};
use mesh_client::protocol::{
    ALPN_CONTROL_V1, MAX_CONTROL_FRAME_BYTES, NODE_PROTOCOL_GENERATION,
    decode_owner_control_envelope, read_len_prefixed, write_len_prefixed,
};
use mesh_client::{
    ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneClientError, ControlPlaneConnection,
    InviteToken, OwnerControlClient, OwnerControlRemoteError, OwnerControlWatchEvent, OwnerKeypair,
};
use prost::Message;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{Mutex, oneshot};

#[derive(Clone)]
struct TestServerState {
    expected_owner_id: String,
    expected_owner_signing_key: Vec<u8>,
    received_apply: Arc<Mutex<Vec<mesh_client::proto::node::OwnerControlApplyConfigRequest>>>,
    watch_closed_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

fn test_owner_keypair(signing_seed: u8, encryption_seed: u8) -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[signing_seed; 32], &[encryption_seed; 32])
        .expect("test owner keypair must be valid")
}

fn control_endpoint_token(addr: &EndpointAddr) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(addr).expect("endpoint addr should serialize"))
}

fn test_snapshot(
    node_id: &[u8; 32],
    revision: u64,
    model: &str,
) -> mesh_client::proto::node::OwnerControlConfigSnapshot {
    mesh_client::proto::node::OwnerControlConfigSnapshot {
        node_id: node_id.to_vec(),
        revision,
        config_hash: vec![revision as u8; 32],
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: Some(NodeGpuConfig {
                assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
            }),
            models: vec![NodeModelEntry {
                model: model.to_string(),
                mmproj: None,
                ctx_size: Some(4096),
                gpu_id: None,
                model_ref: None,
                mmproj_ref: None,
            }],
            plugins: vec![],
            config_toml: None,
            mesh_requirements: None,
        }),
        hostname: Some("control.test".to_string()),
    }
}

async fn make_client() -> mesh_client::MeshClient {
    ClientBuilder::new(
        test_owner_keypair(0x11, 0x12),
        InviteToken::from_str("mesh-test:control-plane").unwrap(),
    )
    .build()
    .expect("mesh client should build")
}

async fn spawn_success_server(
    owner_keypair: &OwnerKeypair,
) -> (Endpoint, String, TestServerState, oneshot::Receiver<()>) {
    spawn_success_server_with_refresh_detail(owner_keypair, true).await
}

async fn spawn_snapshot_only_success_server(
    owner_keypair: &OwnerKeypair,
) -> (Endpoint, String, TestServerState, oneshot::Receiver<()>) {
    spawn_success_server_with_refresh_detail(owner_keypair, false).await
}

async fn spawn_success_server_with_refresh_detail(
    owner_keypair: &OwnerKeypair,
    include_refresh_detail: bool,
) -> (Endpoint, String, TestServerState, oneshot::Receiver<()>) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let (watch_closed_tx, watch_closed_rx) = oneshot::channel();
    let state = TestServerState {
        expected_owner_id: owner_keypair.owner_id(),
        expected_owner_signing_key: owner_keypair.verifying_key().as_bytes().to_vec(),
        received_apply: Arc::new(Mutex::new(Vec::new())),
        watch_closed_tx: Arc::new(Mutex::new(Some(watch_closed_tx))),
    };
    let state_clone = state.clone();
    let endpoint_id = endpoint.id();
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        loop {
            let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                break;
            };
            let state = state_clone.clone();
            let stream_connection = connection.clone();
            tokio::spawn(async move {
                let handshake = read_control_envelope(&mut recv).await;
                let ownership = handshake
                    .handshake
                    .expect("first envelope should be handshake")
                    .ownership
                    .expect("handshake should include ownership");
                assert_eq!(ownership.owner_id, state.expected_owner_id);
                assert_eq!(
                    ownership.owner_sign_public_key,
                    state.expected_owner_signing_key
                );
                assert_eq!(
                    ownership.node_endpoint_id,
                    stream_connection.remote_id().as_bytes().to_vec()
                );

                let request = read_control_envelope(&mut recv)
                    .await
                    .request
                    .expect("second envelope should contain a request");
                if request.get_config.is_some() {
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: Some(
                                    mesh_client::proto::node::OwnerControlGetConfigResponse {
                                        snapshot: Some(test_snapshot(
                                            endpoint_id.as_bytes(),
                                            3,
                                            "get-model.gguf",
                                        )),
                                    },
                                ),
                                watch_config: None,
                                apply_config: None,
                                refresh_inventory: None,
                                load_model: None,
                                unload_model: None,
                                ensure_model: None,
                                drain_model: None,
                                kv_cache: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if let Some(apply) = request.apply_config {
                    state.received_apply.lock().await.push(apply);
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: None,
                                apply_config: Some(
                                    mesh_client::proto::node::OwnerControlApplyConfigResponse {
                                        success: true,
                                        current_revision: 4,
                                        config_hash: vec![0x44; 32],
                                        error: None,
                                        apply_mode: ConfigApplyMode::Staged as i32,
                                        diagnostics: Vec::new(),
                                    },
                                ),
                                refresh_inventory: None,
                                load_model: None,
                                unload_model: None,
                                ensure_model: None,
                                drain_model: None,
                                kv_cache: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if request.refresh_inventory.is_some() {
                    let inventory = include_refresh_detail.then(|| OwnerControlRefreshInventory {
                        entries: vec![OwnerControlInventoryEntry {
                            canonical_model_ref: "hf://mesh/refresh-model-GGUF:Q4_K_M".to_string(),
                            display_name: Some("refresh-model".to_string()),
                            total_size_bytes: 4_096,
                            metadata: Some(CompactModelMetadata {
                                model_key: "hf://mesh/refresh-model-GGUF:Q4_K_M".to_string(),
                                architecture: "llama".to_string(),
                                quantization_type: "Q4_K_M".to_string(),
                                ..Default::default()
                            }),
                        }],
                        disposition: OwnerControlRefreshInventoryDisposition::Executed as i32,
                    });
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: None,
                                apply_config: None,
                                refresh_inventory: Some(mesh_client::proto::node::OwnerControlRefreshInventoryResponse {
                                    snapshot: Some(test_snapshot(endpoint_id.as_bytes(), 5, "refresh-model.gguf")),
                                    inventory,
                                }),
                                load_model: None,
                                unload_model: None,
                                ensure_model: None,
                                drain_model: None,
                                kv_cache: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if let Some(kv_cache) = request.kv_cache {
                    assert_eq!(
                        OwnerControlKvCacheOperation::try_from(kv_cache.operation),
                        Ok(OwnerControlKvCacheOperation::Prune)
                    );
                    assert_eq!(kv_cache.target_bytes, Some(1024));
                    assert_eq!(kv_cache.model_identity.as_deref(), Some("blake3:model"));
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                kv_cache: Some(
                                    mesh_client::proto::node::OwnerControlKvCacheResponse {
                                        status_json:
                                            br#"{"version":1,"effective":{"state":"active"}}"#
                                                .to_vec(),
                                        freed_bytes: Some(512),
                                    },
                                ),
                                ..Default::default()
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if let Some(watch_config) = request.watch_config {
                    let watch_response = if watch_config.include_snapshot {
                        mesh_client::proto::node::OwnerControlWatchConfigResponse {
                            accepted: None,
                            snapshot: Some(test_snapshot(
                                endpoint_id.as_bytes(),
                                6,
                                "watch-model.gguf",
                            )),
                            update: None,
                        }
                    } else {
                        mesh_client::proto::node::OwnerControlWatchConfigResponse {
                            accepted: Some(mesh_client::proto::node::OwnerControlWatchAccepted {
                                target_node_id: endpoint_id.as_bytes().to_vec(),
                            }),
                            snapshot: None,
                            update: None,
                        }
                    };
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: Some(watch_response),
                                apply_config: None,
                                refresh_inventory: None,
                                load_model: None,
                                unload_model: None,
                                ensure_model: None,
                                drain_model: None,
                                kv_cache: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = read_len_prefixed(&mut recv).await;
                    if let Some(tx) = state.watch_closed_tx.lock().await.take() {
                        let _ = tx.send(());
                    }
                }
            });
        }
    });
    (endpoint, token, state, watch_closed_rx)
}

async fn spawn_auth_failure_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let _ = read_control_envelope(&mut recv).await;
        write_control_envelope(
            &mut send,
            OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(mesh_client::proto::node::OwnerControlError {
                    code: OwnerControlErrorCode::Unauthorized as i32,
                    message: "owner attestation rejected".to_string(),
                    request_id: None,
                    current_revision: None,
                }),
            },
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = send.finish();
    });
    (endpoint, token)
}

async fn spawn_control_unsupported_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let request = read_control_envelope(&mut recv).await;
        let request_id = request.request.map(|request| request.request_id);
        write_control_envelope(
            &mut send,
            OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(mesh_client::proto::node::OwnerControlError {
                    code: OwnerControlErrorCode::ControlUnsupported as i32,
                    message: "remote endpoint did not negotiate mesh-llm-control/1".to_string(),
                    request_id,
                    current_revision: None,
                }),
            },
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = send.finish();
    });
    (endpoint, token)
}

async fn spawn_legacy_unknown_lifecycle_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let request = read_control_envelope(&mut recv).await;
        let request_id = request.request.map(|request| request.request_id);
        write_control_envelope(
            &mut send,
            OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(mesh_client::proto::node::OwnerControlError {
                    code: OwnerControlErrorCode::UnknownCommand as i32,
                    message: "owner control request requires exactly one command variant"
                        .to_string(),
                    request_id,
                    current_revision: None,
                }),
            },
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = send.finish();
    });
    (endpoint, token)
}

async fn spawn_stalled_response_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (_send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let _ = read_control_envelope(&mut recv).await;
        std::future::pending::<()>().await;
    });
    (endpoint, token)
}

async fn read_control_envelope(recv: &mut iroh::endpoint::RecvStream) -> OwnerControlEnvelope {
    let bytes = read_len_prefixed(recv)
        .await
        .expect("frame should be len-prefixed");
    decode_owner_control_envelope(&bytes).expect("owner-control envelope should decode")
}

async fn write_control_envelope(
    send: &mut iroh::endpoint::SendStream,
    envelope: OwnerControlEnvelope,
) {
    write_len_prefixed(send, &envelope.encode_to_vec())
        .await
        .expect("owner-control envelope should write");
}

fn owner_control_client(connection: ControlPlaneConnection) -> OwnerControlClient {
    match connection {
        ControlPlaneConnection::OwnerControl(client) => *client,
    }
}

#[tokio::test]
async fn control_plane_client_apply_config_get_watch_refresh_and_close() {
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let client = make_client().await;
    let (server, token, state, watch_closed_rx) = spawn_success_server(&owner_keypair).await;

    let connection = client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
        .await
        .expect("control session should connect");
    let control = owner_control_client(connection);

    let get_snapshot = control
        .get_config()
        .await
        .expect("get_config should succeed");
    assert_eq!(get_snapshot.revision, 3);
    assert_eq!(get_snapshot.hostname.as_deref(), Some("control.test"));

    let apply_response = control
        .apply_config(
            3,
            NodeConfigSnapshot {
                version: 1,
                gpu: Some(NodeGpuConfig {
                    assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
                }),
                models: vec![NodeModelEntry {
                    model: "applied-model.gguf".to_string(),
                    mmproj: None,
                    ctx_size: Some(8192),
                    gpu_id: None,
                    model_ref: None,
                    mmproj_ref: None,
                }],
                plugins: vec![],
                config_toml: None,
                mesh_requirements: None,
            },
        )
        .await
        .expect("apply_config should succeed");
    assert!(apply_response.success);
    assert_eq!(apply_response.current_revision, 4);

    let refresh_snapshot = control
        .refresh_inventory()
        .await
        .expect("refresh_inventory should succeed");
    assert_eq!(refresh_snapshot.revision, 5);

    let scan_result = control
        .scan_refresh()
        .await
        .expect("scan_refresh should return typed inventory detail");
    assert_eq!(scan_result.snapshot.revision, 5);
    let inventory = scan_result.inventory.expect("new server inventory detail");
    assert_eq!(
        OwnerControlRefreshInventoryDisposition::try_from(inventory.disposition),
        Ok(OwnerControlRefreshInventoryDisposition::Executed)
    );
    assert_eq!(inventory.entries.len(), 1);
    assert_eq!(
        inventory.entries[0].canonical_model_ref,
        "hf://mesh/refresh-model-GGUF:Q4_K_M"
    );

    let cache = control
        .kv_cache(
            OwnerControlKvCacheOperation::Prune,
            Some(1024),
            Some("blake3:model".to_string()),
        )
        .await
        .expect("kv_cache should succeed");
    assert_eq!(cache.freed_bytes, Some(512));
    let status: serde_json::Value =
        serde_json::from_slice(&cache.status_json).expect("valid status JSON");
    assert_eq!(status["version"], 1);

    let mut watch = control
        .watch_config(true)
        .await
        .expect("watch_config should open");
    match watch.next().await.expect("watch snapshot should arrive") {
        OwnerControlWatchEvent::Snapshot(snapshot) => assert_eq!(snapshot.revision, 6),
        _ => panic!("expected initial watch snapshot"),
    }
    watch.close().await.expect("watch close should succeed");
    tokio::time::timeout(std::time::Duration::from_secs(5), watch_closed_rx)
        .await
        .expect("server should observe watch close")
        .expect("watch close signal should succeed");

    {
        let apply_requests = state.received_apply.lock().await;
        assert_eq!(apply_requests.len(), 1);
        assert_eq!(apply_requests[0].expected_revision, 3);
        assert_eq!(
            apply_requests[0].config.as_ref().unwrap().models[0].model,
            "applied-model.gguf"
        );
    }
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn scan_refresh_accepts_snapshot_only_old_server_response() {
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let client = make_client().await;
    let (server, token, _state, _watch_closed_rx) =
        spawn_snapshot_only_success_server(&owner_keypair).await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("control session should connect"),
    );

    let result = control
        .scan_refresh()
        .await
        .expect("snapshot-only old-server response should remain successful");

    assert_eq!(result.snapshot.revision, 5);
    assert!(result.inventory.is_none());
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn control_plane_client_watch_without_snapshot_returns_accepted() {
    let client = make_client().await;
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let (server, token, _state, watch_closed_rx) = spawn_success_server(&owner_keypair).await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should use owner-control ALPN"),
    );

    let mut watch = control
        .watch_config(false)
        .await
        .expect("watch_config should open");
    match watch.next().await.expect("watch accepted should arrive") {
        OwnerControlWatchEvent::Accepted(accepted) => {
            assert_eq!(accepted.target_node_id.len(), 32);
        }
        _ => panic!("expected initial watch accepted event"),
    }
    watch.close().await.expect("watch close should succeed");
    tokio::time::timeout(std::time::Duration::from_secs(5), watch_closed_rx)
        .await
        .expect("server should observe watch close")
        .expect("watch close signal should succeed");
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn control_plane_client_watch_stream_has_no_unary_deadline_after_acceptance() {
    let client = make_client().await;
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let (server, token, _state, _watch_closed_rx) = spawn_success_server(&owner_keypair).await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should use owner-control ALPN"),
    );

    let mut watch = control
        .watch_config(false)
        .await
        .expect("watch_config should wait for accepted event");
    match watch
        .next()
        .await
        .expect("pending accepted event should be returned")
    {
        OwnerControlWatchEvent::Accepted(accepted) => assert_eq!(accepted.target_node_id.len(), 32),
        _ => panic!("expected accepted event"),
    }
    let later = tokio::time::timeout(std::time::Duration::from_millis(150), watch.next()).await;
    assert!(
        later.is_err(),
        "accepted watch streams must remain open instead of inheriting unary deadlines"
    );

    watch.close().await.expect("watch close should succeed");
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn oversized_outbound_control_frame_emits_no_prefix_or_body() {
    let server = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let server_endpoint = server.clone();
    let accepted = tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("connection should arrive");
        let connection = incoming.await.expect("connection should negotiate");
        connection.accept_bi().await.expect("stream should arrive")
    });
    let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let connection = client
        .connect(server.addr(), ALPN_CONTROL_V1)
        .await
        .expect("client should connect");
    let (mut send, _recv) = connection.open_bi().await.expect("stream should open");
    let oversized = vec![0u8; MAX_CONTROL_FRAME_BYTES + 1];

    let error = write_len_prefixed(&mut send, &oversized)
        .await
        .expect_err("oversized write should fail");
    assert!(error.to_string().contains("control frame too large"));
    send.finish().expect("empty stream should finish");

    let (_server_send, mut server_recv) = accepted.await.expect("server task should join");
    let received = server_recv
        .read_to_end(4)
        .await
        .expect("empty stream should read");
    assert!(
        received.is_empty(),
        "no length prefix or body may be emitted"
    );

    client.close().await;
    server.close().await;
}

#[tokio::test]
async fn stalled_unary_response_returns_deterministic_timeout() {
    let client = make_client().await;
    let (server, token) = spawn_stalled_response_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should use owner-control ALPN"),
    );

    let error = control
        .get_config()
        .await
        .expect_err("stalled response should time out");
    match error {
        ControlPlaneClientError::Transport(message) => {
            assert_eq!(message, "owner-control unary response timed out after 10s");
        }
        other => panic!("expected transport timeout, got {other:?}"),
    }

    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn control_plane_client_auth_failure_surfaces_structured_error() {
    let client = make_client().await;
    let (server, token) = spawn_auth_failure_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should bootstrap before first request"),
    );

    let err = control
        .get_config()
        .await
        .expect_err("auth failure should bubble up");
    match err {
        ControlPlaneClientError::Remote(OwnerControlRemoteError { code, message, .. }) => {
            assert_eq!(code, OwnerControlErrorCode::Unauthorized);
            assert!(message.contains("rejected"));
        }
        other => panic!("expected structured remote auth error, got {other:?}"),
    }
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn control_plane_client_rejects_alpn_mismatch() {
    let client = make_client().await;
    let (server, token) = spawn_control_unsupported_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("control session should bootstrap before request"),
    );
    let err = control
        .get_config()
        .await
        .expect_err("unsupported configured endpoint should surface a structured error");

    match err {
        ControlPlaneClientError::Remote(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
        }
        other => panic!("expected structured unsupported error, got {other:?}"),
    }
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn lifecycle_command_maps_legacy_unknown_command_to_control_unsupported() {
    let client = make_client().await;
    let (server, token) = spawn_legacy_unknown_lifecycle_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("control session should bootstrap before request"),
    );
    let err = control
        .load_model("legacy/model".to_string(), None)
        .await
        .expect_err("legacy endpoint should reject lifecycle command");

    match err {
        ControlPlaneClientError::Remote(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
            assert_eq!(
                err.message,
                "remote owner-control endpoint does not support load_model"
            );
        }
        other => panic!("expected structured unsupported error, got {other:?}"),
    }
    control.close().await;
    server.close().await;
}

#[tokio::test]
async fn control_plane_client_unreachable_listener_returns_structured_negotiation_error() {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    endpoint.close().await;

    let client = make_client().await;
    let err = match client
        .connect_control_plane(
            ControlPlaneBootstrapOptions::new()
                .with_control_endpoint(token)
                .with_connect_timeout(std::time::Duration::from_millis(100)),
        )
        .await
    {
        Ok(_) => panic!("unreachable listener should fail"),
        Err(err) => err,
    };

    match err {
        ControlPlaneClientError::Negotiation(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnavailable);
            assert!(
                err.message
                    .contains("remote owner-control endpoint is unavailable or unreachable"),
                "message: {}",
                err.message
            );
            assert!(!err.legacy_retry_allowed);
        }
        other => panic!("expected negotiation error, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_client_does_not_silently_fallback_when_endpoint_fails() {
    let client = make_client().await;
    let (server, token) = spawn_control_unsupported_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("configured endpoint should stay on the owner-control lane"),
    );
    let err = control
        .get_config()
        .await
        .expect_err("configured endpoint failure must not silently fall back");

    match err {
        ControlPlaneClientError::Remote(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
        }
        other => panic!("expected structured unsupported error, got {other:?}"),
    }
    control.close().await;
    server.close().await;
}
