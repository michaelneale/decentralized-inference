fn qwen_coder_remote_catalog_entry() -> crate::models::remote_catalog::CatalogEntry {
    use crate::models::remote_catalog::{
        CatalogCurated, CatalogEntry, CatalogSource, CatalogVariant,
    };
    CatalogEntry {
        schema_version: 1,
        source_repo: "Qwen/Qwen3-Coder-Next-GGUF".to_string(),
        variants: HashMap::from([(
            "Qwen3-Coder-Next-Q4_K_M".to_string(),
            CatalogVariant {
                source: CatalogSource {
                    repo: "Qwen/Qwen3-Coder-Next-GGUF".to_string(),
                    revision: Some("main".to_string()),
                    file: Some("Qwen3-Coder-Next-Q4_K_M.gguf".to_string()),
                },
                curated: CatalogCurated {
                    name: "Qwen3-Coder-Next-Q4_K_M".to_string(),
                    size: Some("20GB".to_string()),
                    description: Some("Coding model".to_string()),
                    draft: None,
                    moe: None,
                    extra_files: Vec::new(),
                    mmproj: None,
                },
                packages: Vec::new(),
            },
        )]),
    }
}

fn qwen_coder_remote_catalog_ref() -> String {
    "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()
}

async fn build_test_mesh_api_with_api_port(api_port: u16) -> MeshApi {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let resolved_plugins = plugin::ResolvedPlugins {
        externals: vec![],
        inactive: vec![],
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    let plugin_manager = plugin::PluginManager::start(
        &resolved_plugins,
        plugin::PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .unwrap();
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer = runtime_data_collector.producer(runtime_data::RuntimeDataSource {
        scope: "runtime",
        plugin_data_key: None,
        plugin_endpoint_key: None,
    });
    MeshApi::new(MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port,
        model_size_bytes: 0,
        owner_key_path: None,
        plugin_manager,
        affinity_router: affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

async fn build_test_mesh_api() -> MeshApi {
    build_test_mesh_api_with_api_port(3131).await
}

fn mesh_requirements_test_policy_for_owner(
    origin_owner_id: impl Into<String>,
) -> crate::MeshGenesisPolicy {
    crate::MeshGenesisPolicy::new(
        origin_owner_id,
        1_717_171_717_000,
        crate::MeshRequirements {
            release_attestation: crate::ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["trusted-release".into()],
            },
            ..crate::MeshRequirements::unrestricted()
        },
    )
    .expect("test policy should be valid")
}

fn mesh_requirements_test_policy() -> crate::MeshGenesisPolicy {
    mesh_requirements_test_policy_for_owner("owner-123")
}

pub(crate) fn assert_mesh_requirements_status_excludes_rejected_peers_from_admitted_list() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let state = build_test_mesh_api().await;
            let node = state.node().await;
            let remote = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
                .await
                .unwrap();
            let policy = mesh_requirements_test_policy();
            node.set_active_mesh_policy_for_tests(policy.clone()).await;
            remote.set_active_mesh_policy_for_tests(policy).await;

            node.sync_from_peer_for_tests(&remote).await;

            let status = state.status().await;
            assert!(
                status.peers.is_empty(),
                "rejected peers must not appear admitted"
            );
            assert_eq!(status.recent_mesh_rejections.len(), 1);
            assert_eq!(
                status.recent_mesh_rejections[0].reason,
                crate::MeshRequirementRejectReason::CertifiedBinaryRequired
            );
        });
}

pub(crate) fn assert_mesh_requirements_status_reports_policy_hash_read_only() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let state = build_test_mesh_api().await;
            let node = state.node().await;
            let expected = node
                .set_active_mesh_policy_for_tests(mesh_requirements_test_policy())
                .await;

            let payload = serde_json::to_value(state.status().await).unwrap();
            assert_eq!(
                payload["mesh_requirements"]["policy_hash"],
                serde_json::Value::String(expected.policy_hash.clone())
            );
            assert_eq!(
                payload["mesh_requirements"]["requirements"]["release_attestation"]["required"],
                serde_json::Value::Bool(true)
            );
            let payload_text = payload.to_string();
            assert!(!payload_text.contains("signature"));
            assert!(!payload_text.contains("serialized_addrs"));
            assert!(!payload_text.contains("origin_sign_public_key"));
        });
}

pub(crate) fn assert_mesh_requirements_certified_binary_required_event_text() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let state = build_test_mesh_api().await;
            let node = state.node().await;
            let remote = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
                .await
                .unwrap();
            let policy = mesh_requirements_test_policy();
            node.set_active_mesh_policy_for_tests(policy.clone()).await;
            remote.set_active_mesh_policy_for_tests(policy).await;

            node.sync_from_peer_for_tests(&remote).await;

            let status = state.status().await;
            assert_eq!(
                status.recent_mesh_rejections[0].message,
                "this mesh requires a certified mesh-llm binary; use a certified compiled binary to join."
            );
        });
}

pub(crate) fn assert_mesh_requirements_rejection_events_do_not_expose_tokens() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let state = build_test_mesh_api().await;
            let node = state.node().await;
            let owner = OwnerKeypair::generate();
            let signed_policy = crate::SignedMeshGenesisPolicy::sign(
                mesh_requirements_test_policy_for_owner(owner.owner_id()),
                &owner,
            )
            .unwrap();
            let mut token = crate::SignedBootstrapToken::sign(
                vec![
                    serde_json::to_vec(
                        &mesh::Node::decode_invite_token(&node.invite_token().await).unwrap(),
                    )
                    .unwrap(),
                ],
                &signed_policy,
                Some(1),
                &owner,
            )
            .unwrap();
            token.signature[0] ^= 0xFF;
            let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&token).unwrap());

            let err = node
                .join(&invite_token)
                .await
                .expect_err("join should reject tampered token");
            assert!(
                err.to_string().contains("bootstrap_token_invalid")
                    || err.to_string().contains("join rejected")
            );

            let payload = serde_json::to_value(state.status().await).unwrap();
            let payload_text = payload.to_string();
            assert!(!payload_text.contains(&invite_token));
            assert!(
                !payload_text
                    .contains(&base64::engine::general_purpose::STANDARD.encode(&token.signature))
            );
        });
}

async fn build_test_mesh_api_with_plugin_manager(
    api_port: u16,
    plugin_manager: plugin::PluginManager,
) -> MeshApi {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer = runtime_data_collector.producer(runtime_data::RuntimeDataSource {
        scope: "runtime",
        plugin_data_key: None,
        plugin_endpoint_key: None,
    });
    MeshApi::new(MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port,
        model_size_bytes: 0,
        owner_key_path: None,
        plugin_manager,
        affinity_router: affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

async fn build_inference_endpoint_plugin_manager(models: &[&str]) -> plugin::PluginManager {
    let resolved_plugins = plugin::ResolvedPlugins {
        externals: vec![],
        inactive: vec![],
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    let plugin_manager = plugin::PluginManager::start(
        &resolved_plugins,
        plugin::PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .unwrap();
    plugin_manager
        .set_test_inference_endpoints(vec![plugin::InferenceEndpointRoute {
            plugin_name: "endpoint-plugin".into(),
            endpoint_id: "endpoint-plugin".into(),
            address: "http://127.0.0.1:8000/v1".into(),
            models: models.iter().map(|model| (*model).to_string()).collect(),
        }])
        .await;
    plugin_manager
}

struct OwnerControlTestServer {
    endpoint_token: String,
    task: tokio::task::JoinHandle<()>,
}

async fn spawn_owner_control_test_server() -> OwnerControlTestServer {
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let endpoint_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&endpoint.addr()).unwrap());
    let task = tokio::spawn(async move {
        let Some(incoming) = endpoint.accept().await else {
            return;
        };
        let mut accepting = incoming.accept().unwrap();
        let _ = accepting.alpn().await.unwrap();
        let conn = accepting.await.unwrap();
        let (mut send, mut recv) = conn.accept_bi().await.unwrap();
        let handshake = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let _ = decode_owner_control_envelope(&handshake).unwrap();
        let request = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let envelope = decode_owner_control_envelope(&request).unwrap();
        let request_id = envelope.request.as_ref().unwrap().request_id;
        let response = OwnerControlEnvelope {
            r#gen: mesh_llm_protocol::NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id,
                get_config: Some(OwnerControlGetConfigResponse {
                    snapshot: Some(OwnerControlConfigSnapshot {
                        node_id: vec![7; 32],
                        revision: 42,
                        config_hash: vec![9; 32],
                        config: Some(NodeConfigSnapshot {
                            version: 1,
                            gpu: None,
                            models: Vec::new(),
                            plugins: Vec::new(),
                            config_toml: None,
                            mesh_requirements: None,
                        }),
                        hostname: Some("control-target".to_string()),
                    }),
                }),
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
            }),
            error: None,
        };
        write_len_prefixed(&mut send, &response.encode_to_vec())
            .await
            .unwrap();
        let _ = send.finish();
        tokio::time::sleep(Duration::from_millis(100)).await;
    });
    OwnerControlTestServer {
        endpoint_token,
        task,
    }
}

struct OwnerControlApplyTestServer {
    endpoint_token: String,
    task: tokio::task::JoinHandle<()>,
    received_apply: Option<oneshot::Receiver<OwnerControlApplyConfigRequest>>,
}

enum OwnerControlApplyTestResponse {
    Success(OwnerControlApplyConfigResponse),
    Error {
        code: OwnerControlErrorCode,
        message: String,
        current_revision: Option<u64>,
    },
}

async fn spawn_owner_control_apply_test_server(
    response: OwnerControlApplyTestResponse,
) -> OwnerControlApplyTestServer {
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let endpoint_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&endpoint.addr()).unwrap());
    let (apply_tx, apply_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let Some(incoming) = endpoint.accept().await else {
            return;
        };
        let mut accepting = incoming.accept().unwrap();
        let _ = accepting.alpn().await.unwrap();
        let conn = accepting.await.unwrap();
        let (mut send, mut recv) = conn.accept_bi().await.unwrap();
        let handshake = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let _ = decode_owner_control_envelope(&handshake).unwrap();
        let request = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let envelope = decode_owner_control_envelope(&request).unwrap();
        let request = envelope
            .request
            .expect("owner-control request should be present");
        let request_id = request.request_id;
        let apply = request
            .apply_config
            .expect("expected apply-config request for apply response");
        let _ = apply_tx.send(apply);
        let envelope = match response {
            OwnerControlApplyTestResponse::Success(response) => OwnerControlEnvelope {
                r#gen: mesh_llm_protocol::NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: Some(OwnerControlResponse {
                    request_id,
                    get_config: None,
                    watch_config: None,
                    apply_config: Some(response),
                    refresh_inventory: None,
                    load_model: None,
                    unload_model: None,
                    ensure_model: None,
                    drain_model: None,
                }),
                error: None,
            },
            OwnerControlApplyTestResponse::Error {
                code,
                message,
                current_revision,
            } => OwnerControlEnvelope {
                r#gen: mesh_llm_protocol::NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(OwnerControlError {
                    code: code as i32,
                    message,
                    request_id: Some(request_id),
                    current_revision,
                }),
            },
        };
        write_len_prefixed(&mut send, &envelope.encode_to_vec())
            .await
            .unwrap();
        let _ = send.finish();
        tokio::time::sleep(Duration::from_millis(100)).await;
    });
    OwnerControlApplyTestServer {
        endpoint_token,
        task,
        received_apply: Some(apply_rx),
    }
}

fn management_post_request(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
}

fn full_mesh_config_fixture() -> crate::plugin::MeshConfig {
    serde_json::from_value(json!({
        "version": 1,
        "gpu": {
            "assignment": "auto",
            "parallel": 2
        },
        "owner_control": {
            "bind": "127.0.0.1:7447",
            "advertise_addr": "127.0.0.1:7447"
        },
        "telemetry": {
            "enabled": true,
            "service_name": "mesh-llm-control",
            "endpoint": "http://127.0.0.1:4317",
            "headers": {
                "authorization": "Bearer control-test"
            },
            "export_interval_secs": 30,
            "queue_size": 256,
            "prompt_shape_metrics": false,
            "metrics": {
                "endpoint": "http://127.0.0.1:4318"
            }
        },
        "models": [
            {
                "model": "hf://meshllm/base@main:Q4_K_M",
                "mmproj": "hf://meshllm/base@main:mmproj.gguf",
                "ctx_size": 8192,
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "batch": 512,
                "ubatch": 256
            }
        ],
        "plugin": [
            {
                "name": "telemetry",
                "enabled": true,
                "command": "mesh-telemetry"
            }
        ]
    }))
    .unwrap()
}

fn merge_json_object(target: &mut serde_json::Value, source: serde_json::Value) {
    let target = target
        .as_object_mut()
        .expect("target JSON should be an object for config merge");
    let source = source
        .as_object()
        .expect("source JSON should be an object for config merge");
    target.extend(source.clone());
}

async fn unreachable_owner_control_endpoint_token() -> String {
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&endpoint.addr()).unwrap());
    drop(endpoint);
    token
}

async fn spawn_management_test_server(
    state: MeshApi,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    spawn_management_test_server_on(std::net::SocketAddr::from(([127, 0, 0, 1], 0)), state).await
}

async fn spawn_management_test_server_on(
    bind_addr: std::net::SocketAddr,
    state: MeshApi,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    let listener = TcpListener::bind(bind_addr).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        handle_request(stream, &state).await
    });
    (addr, handle)
}

async fn send_management_request(addr: std::net::SocketAddr, raw_request: String) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(raw_request.as_bytes()).await.unwrap();
    let _ = stream.shutdown().await;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    String::from_utf8(response).unwrap()
}

fn json_body(response: &str) -> serde_json::Value {
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    serde_json::from_str(body).unwrap_or(serde_json::Value::Null)
}

async fn replace_test_wakeable_inventory(state: &MeshApi, entries: Vec<WakeableInventoryEntry>) {
    let inventory = { state.inner.lock().await.wakeable_inventory.clone() };
    inventory.replace_for_tests(entries).await;
}

fn make_test_wakeable_entry(logical_id: &str, model: &str, vram_gb: f32) -> WakeableInventoryEntry {
    WakeableInventoryEntry {
        logical_id: logical_id.to_string(),
        models: vec![model.to_string()],
        vram_gb,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Sleeping,
        wake_eta_secs: Some(45),
    }
}

fn make_test_peer(
    seed: u8,
    role: mesh::NodeRole,
    serving_models: Vec<&str>,
    hosted_models: Vec<&str>,
    hosted_models_known: bool,
) -> mesh::PeerInfo {
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public());
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role,
        first_joined_mesh_ts: None,
        models: Vec::new(),
        vram_bytes: 24_000_000_000,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: serving_models.into_iter().map(str::to_string).collect(),
        hosted_models: hosted_models.into_iter().map(str::to_string).collect(),
        hosted_models_known,
        available_models: Vec::new(),
        requested_models: Vec::new(),
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
        available_model_sizes: HashMap::new(),
        served_model_descriptors: Vec::new(),
        served_model_runtime: Vec::new(),
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        advertised_model_throughput: vec![],
        cache_affinity: None,

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        inference_admission_state: None,
    }
}

#[derive(Clone)]
struct BlobstoreApiTestBridge {
    plugin_name: String,
    store: blobstore::BlobStore,
}

impl BlobstoreApiTestBridge {
    fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

impl plugin::PluginRpcBridge for BlobstoreApiTestBridge {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        let expected_plugin_name = self.plugin_name.clone();
        let store = self.store.clone();
        Box::pin(async move {
            if plugin_name != expected_plugin_name {
                return Err(Self::error_response(format!(
                    "Unsupported test plugin '{}'",
                    plugin_name
                )));
            }
            if method != "tools/call" {
                return Err(Self::error_response(format!(
                    "Unsupported method '{}'",
                    method
                )));
            }

            let request: mesh_llm_plugin::OperationRequest = serde_json::from_str(&params_json)
                .map_err(|err| Self::error_response(err.to_string()))?;
            let result_json = match request.name.as_str() {
                blobstore::PUT_REQUEST_OBJECT_TOOL => {
                    let request: blobstore::PutRequestObjectRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .put_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                blobstore::COMPLETE_REQUEST_TOOL | blobstore::ABORT_REQUEST_TOOL => {
                    let request: blobstore::FinishRequestRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                _ => {
                    return Err(Self::error_response(format!(
                        "Unsupported blobstore tool '{}'",
                        request.name
                    )));
                }
            };

            Ok(plugin::RpcResult { result_json })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

fn temp_blobstore_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mesh-llm-api-server-{name}-{}",
        rand::random::<u64>()
    ))
}

async fn build_blobstore_api_plugin_manager() -> (plugin::PluginManager, std::path::PathBuf) {
    let plugin_name = "blobstore";
    let root = temp_blobstore_root("blobstore");
    let bridge = BlobstoreApiTestBridge {
        plugin_name: plugin_name.into(),
        store: blobstore::BlobStore::new(root.clone()),
    };
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
    let mut manifests = HashMap::new();
    manifests.insert(
        plugin_name.to_string(),
        mesh_llm_plugin::plugin_manifest![mesh_llm_plugin::capability(
            blobstore::OBJECT_STORE_CAPABILITY
        ),],
    );
    plugin_manager
        .set_test_manifests(manifests.into_iter().collect())
        .await;
    (plugin_manager, root)
}

async fn spawn_capturing_upstream(
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let response = response_body.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        let _ = request_tx.send(request.raw);

        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        );
        stream.write_all(resp.as_bytes()).await.unwrap();
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

async fn spawn_streaming_upstream(
    content_type: &str,
    chunks: Vec<(Duration, Vec<u8>)>,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let content_type = content_type.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        let _ = request_tx.send(request.raw);

        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(header.as_bytes()).await.is_err() {
            return;
        }

        for (delay, chunk) in chunks {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let chunk_header = format!("{:x}\r\n", chunk.len());
            if stream.write_all(chunk_header.as_bytes()).await.is_err() {
                return;
            }
            if stream.write_all(&chunk).await.is_err() {
                return;
            }
            if stream.write_all(b"\r\n").await.is_err() {
                return;
            }
        }

        let _ = stream.write_all(b"0\r\n\r\n").await;
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

async fn read_until_contains(stream: &mut TcpStream, needle: &[u8], timeout: Duration) -> Vec<u8> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut response = Vec::new();
    while !contains_bytes(&response, needle) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "timed out waiting for {:?} in response: {}",
            String::from_utf8_lossy(needle),
            String::from_utf8_lossy(&response)
        );
        let mut chunk = [0u8; 4096];
        let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
            .await
            .expect("timed out waiting for response bytes")
            .unwrap();
        assert!(n > 0, "unexpected EOF while waiting for response bytes");
        response.extend_from_slice(&chunk[..n]);
    }
    response
}

async fn assert_no_stream_bytes_within(stream: &mut TcpStream, timeout: Duration) {
    let mut chunk = [0u8; 4096];
    match tokio::time::timeout(timeout, stream.read(&mut chunk)).await {
        Err(_) => {}
        Ok(Ok(0)) => {}
        Ok(Ok(n)) => panic!(
            "unexpected stream bytes within {:?}: {}",
            timeout,
            String::from_utf8_lossy(&chunk[..n])
        ),
        Ok(Err(error)) => panic!("unexpected stream read error within {:?}: {error}", timeout),
    }
}

async fn build_collector_backed_plugin_manager() -> plugin::PluginManager {
    struct NoopBridge;

    impl plugin::PluginRpcBridge for NoopBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> plugin::BridgeFuture<Result<plugin::RpcResult, crate::plugin::proto::ErrorResponse>>
        {
            Box::pin(async {
                Err(crate::plugin::proto::ErrorResponse {
                    code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
                    message: "unexpected request".into(),
                    data_json: String::new(),
                })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    let plugin_manager = plugin::PluginManager::for_test_bridge(
        &["collector-plugin"],
        std::sync::Arc::new(NoopBridge),
    );
    plugin_manager
        .set_test_manifests(std::collections::BTreeMap::from([(
            "collector-plugin".into(),
            crate::plugin::proto::PluginManifest {
                capabilities: vec!["chat".into()],
                endpoints: vec![crate::plugin::proto::EndpointManifest {
                    endpoint_id: "chat-http".into(),
                    kind: crate::plugin::proto::EndpointKind::Inference as i32,
                    transport_kind:
                        crate::plugin::proto::EndpointTransportKind::EndpointTransportHttp as i32,
                    protocol: Some("openai_compatible".into()),
                    address: Some("http://127.0.0.1:4010/v1".into()),
                    args: vec![],
                    namespace: Some("chat".into()),
                    supports_streaming: true,
                    managed_by_plugin: false,
                }],
                ..Default::default()
            },
        )]))
        .await;
    plugin_manager
        .publish_test_bridge_snapshot("collector-plugin")
        .await
        .expect("collector-backed plugin manager");
    plugin_manager
}

async fn seed_runtime_data_api_state(state: &MeshApi) {
    {
        let mut inner = state.inner.lock().await;
        inner.primary_backend = Some("legacy-backend".into());
        inner.is_host = false;
        inner.llama_ready = false;
        inner.llama_port = Some(9999);
        inner.local_processes = vec![RuntimeProcessPayload {
            name: "legacy-model".into(),
            instance_id: None,
            backend: "legacy-backend".into(),
            status: "ready".into(),
            port: 9999,
            pid: 111,
            slots: 4,
            context_length: None,
            profile: String::new(),
        }];
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                runtime_status.primary_model = Some("collector-model".into());
                runtime_status.primary_backend = Some("collector-backend".into());
                runtime_status.is_host = true;
                runtime_status.llama_ready = true;
                runtime_status.llama_port = Some(9337);
                true
            });
        inner
            .runtime_data_producer
            .publish_local_processes(|local_processes| {
                local_processes.clear();
                local_processes.push(runtime_data::RuntimeProcessSnapshot {
                    model: "collector-model".into(),
                    instance_id: Some("runtime-1".into()),
                    profile: String::new(),
                    backend: "collector-backend".into(),
                    pid: 777,
                    port: 9337,
                    slots: 4,
                    context_length: Some(0),
                    command: Some("llama-server".into()),
                    state: "ready".into(),
                    start: Some(1_700_000_000),
                    health: Some("ready".into()),
                });
                true
            });
        inner.runtime_data_producer.publish_llama_metrics_snapshot(
            runtime_data::RuntimeLlamaMetricsSnapshot {
                status: runtime_data::RuntimeLlamaEndpointStatus::Ready,
                last_attempt_unix_ms: Some(1_700_000_001_000),
                last_success_unix_ms: Some(1_700_000_001_000),
                error: None,
                raw_text: Some("llama_requests_processing 2\n".into()),
                samples: vec![runtime_data::RuntimeLlamaMetricSample {
                    name: "llama_requests_processing".into(),
                    labels: std::collections::BTreeMap::new(),
                    value: 2.0,
                }],
            },
        );
        inner.runtime_data_producer.publish_llama_slots_snapshot(
            runtime_data::RuntimeLlamaSlotsSnapshot {
                status: runtime_data::RuntimeLlamaEndpointStatus::Ready,
                model: Some("collector-model".into()),
                instance_id: Some("runtime-1".into()),
                last_attempt_unix_ms: Some(1_700_000_001_500),
                last_success_unix_ms: Some(1_700_000_001_500),
                error: None,
                slots: vec![runtime_data::RuntimeLlamaSlotSnapshot {
                    id: Some(0),
                    id_task: Some(42),
                    n_ctx: Some(8192),
                    speculative: Some(false),
                    is_processing: Some(true),
                    next_token: Some(json!({"id": 99})),
                    params: Some(json!({"temperature": 0.2})),
                    extra: json!({"state": "busy"}),
                }],
            },
        );
    }
    let node = state.node().await;
    node.record_stage_status(
        Some(node.id()),
        crate::inference::skippy::StageStatusSnapshot {
            topology_id: "topology-1".into(),
            run_id: "run-1".into(),
            model_id: "collector-model".into(),
            backend: "package".into(),
            package_ref: Some("hf://mesh/test-model".into()),
            manifest_sha256: Some("manifest-sha".into()),
            source_model_path: Some("/models/test.gguf".into()),
            source_model_sha256: Some("source-sha".into()),
            source_model_bytes: Some(1_234),
            materialized_path: Some("/tmp/mesh/stage-0.gguf".into()),
            materialized_pinned: true,
            projector_path: Some("/models/mmproj.gguf".into()),
            stage_id: "stage-0".into(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 12,
            admission: Some(crate::inference::skippy::test_stage_admission(0, 12)),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            activation_codec_policy: Default::default(),
            state: crate::inference::skippy::StageRuntimeState::Ready,
            bind_addr: "127.0.0.1:39100".into(),
            input_activation_boundary: None,
            output_activation_boundary: None,
            selected_device: Some(skippy_protocol::StageDevice {
                backend_device: "Metal0".into(),
                stable_id: Some("metal:0".into()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            }),
            ctx_size: 8192,
            lane_count: 2,
            n_batch: Some(2048),
            n_ubatch: Some(512),
            flash_attn_type: skippy_protocol::FlashAttentionType::Enabled,
            error: None,
            shutdown_generation: 7,
            coordinator_term: 11,
            coordinator_id: Some(node.id()),
            lease_until_unix_ms: 999_999,
        },
    )
    .await;
}

async fn request_management_json(state: MeshApi, path: &str) -> serde_json::Value {
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response for {path}: {response}"
    );
    handle.abort();
    json_body(&response)
}

fn response_header<'a>(response: &'a str, name: &str) -> Option<&'a str> {
    response
        .split("\r\n\r\n")
        .next()
        .unwrap_or_default()
        .lines()
        .find_map(|line| {
            let (header_name, value) = line.split_once(':')?;
            header_name.eq_ignore_ascii_case(name).then(|| value.trim())
        })
}

fn assert_runtime_status_payload(status_body: &serde_json::Value) {
    assert_eq!(status_body["model_name"], json!("collector-model"));
    assert_eq!(status_body["llama_ready"], json!(true));
    assert_eq!(
        status_body["runtime"]["backend"],
        json!("collector-backend")
    );
    assert_eq!(
        status_body["runtime"]["models"][0]["name"],
        json!("collector-model")
    );
    assert_eq!(
        status_body["runtime"]["models"][0]["instance_id"],
        json!("runtime-1")
    );
    assert_eq!(
        status_body["runtime"]["models"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["model_id"],
        json!("collector-model")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["package_ref"],
        json!("hf://mesh/test-model")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["materialized_pinned"],
        json!(true)
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["projector_path"],
        json!("/models/mmproj.gguf")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["multimodal"],
        json!(true)
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["selected_device"]["backend_device"],
        json!("Metal0")
    );
    assert!(status_body.get("mesh_models").is_none());
}

fn assert_runtime_llama_payload(llama_body: &serde_json::Value) {
    assert_eq!(llama_body["metrics"]["status"], json!("ready"));
    assert_eq!(
        llama_body["metrics"]["samples"][0]["name"],
        json!("llama_requests_processing")
    );
    assert_eq!(
        llama_body["items"]["metrics"][0]["name"],
        json!("llama_requests_processing")
    );
    assert_eq!(llama_body["slots"]["status"], json!("ready"));
    assert_eq!(llama_body["slots"]["instance_id"], json!("runtime-1"));
    assert_eq!(llama_body["slots"]["slots"][0]["id_task"], json!(42));
    assert_eq!(
        llama_body["slots"]["slots"][0]["extra"]["state"],
        json!("busy")
    );
    assert_eq!(llama_body["items"]["slots_total"], json!(1));
    assert_eq!(llama_body["items"]["slots_busy"], json!(1));
    assert_eq!(llama_body["items"]["slots"][0]["index"], json!(0));
    assert_eq!(
        llama_body["items"]["slots"][0]["is_processing"],
        json!(true)
    );
    assert_eq!(
        llama_body["instances"][0]["instance_id"],
        json!("runtime-1")
    );
    assert_eq!(
        llama_body["instances"][0]["model"],
        json!("collector-model")
    );
    assert_eq!(
        llama_body["instances"][0]["slots"]["status"],
        json!("ready")
    );
    assert_eq!(llama_body["instances"][0]["items"]["slots_busy"], json!(1));
}
