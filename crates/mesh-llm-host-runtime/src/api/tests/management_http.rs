#[tokio::test]
async fn test_management_request_parser_handles_fragmented_post_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = br#"{"text":"fragmented"}"#;
    let headers = format!(
        "POST /api/plugins/demo/http/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let header_split = headers.find("\r\n\r\n").unwrap() + 2;
    let body_split = 8;
    let (server_ready_tx, server_ready_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        server_ready_tx.send(()).unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            proxy::read_http_request(&mut stream),
        )
        .await
        .unwrap()
        .unwrap()
    });

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.set_nodelay(true).unwrap();
        server_ready_rx.await.unwrap();
        stream
            .write_all(&headers.as_bytes()[..header_split])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream
            .write_all(&headers.as_bytes()[header_split..])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream.write_all(&body[..body_split]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream.write_all(&body[body_split..]).await.unwrap();
        let mut sink = [0u8; 1];
        let _ = stream.read(&mut sink).await;
    });

    client.await.unwrap();
    let request = server.await.unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/plugins/demo/http/post");
    assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
}

#[tokio::test]
async fn management_health_is_a_json_liveness_response() {
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "response: {response}"
    );
    assert!(response.contains("Content-Type: application/json"));
    assert_eq!(json_body(&response)["status"], "ok");
    assert_eq!(json_body(&response)["mesh"]["status"], "standalone");
    assert_eq!(json_body(&response)["serving"]["status"], "idle");
}

#[tokio::test]
#[serial]
async fn trusted_local_management_mutation_persists_the_tcp_caller_address() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    crate::initialize_logging_foundation(&config).await;

    let state = build_test_mesh_api().await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind management listener");
    let address = listener.local_addr().expect("management listener address");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept management caller");
        handle_request(stream, &state).await
    });
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let body = r#""active""#;
    let request = format!(
        "PUT /api/runtime/activity/override HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_id.as_uuid(),
        body.len(),
        body,
    );
    let mut client = TcpStream::connect(address)
        .await
        .expect("connect management caller");
    let caller_addr = client
        .local_addr()
        .expect("management caller address")
        .to_string();
    client
        .write_all(request.as_bytes())
        .await
        .expect("write management mutation");
    client.shutdown().await.expect("finish management request");
    let mut response = Vec::new();
    client
        .read_to_end(&mut response)
        .await
        .expect("read management response");
    server
        .await
        .expect("management task joins")
        .expect("request");
    assert!(
        String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"),
        "trusted-local mutation succeeds"
    );

    let logging = crate::logging_runtime_state().expect("installed logging runtime");
    let request_key = request_id.as_uuid().to_string();
    let active = logging
        .service_for_test()
        .expect("logging service")
        .registry_ref()
        .get_recent(&request_key)
        .expect("active request summary");
    assert_eq!(active.metadata.caller_addr(), Some(caller_addr.as_str()));
    assert_eq!(active.metadata.caller_path_type(), Some("local_http"));
    logging.pump_persistence_for_test().await;
    let durable = logging
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_key)
        .expect("durable request query")
        .expect("durable request summary");
    assert_eq!(durable.caller_addr.as_deref(), Some(caller_addr.as_str()));
    assert_eq!(durable.caller_path_type.as_deref(), Some("local_http"));
    assert!(durable.caller_endpoint_id.is_none());
}

#[tokio::test]
#[serial]
async fn logs_and_read_only_management_routes_never_self_record() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    crate::initialize_logging_foundation(&config).await;
    let state = build_test_mesh_api().await;

    let mut request_ids = Vec::new();
    for path in ["/api/logs/requests", "/api/status"] {
        let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
        let (address, server) = spawn_management_test_server(state.clone()).await;
        let response = send_management_request(
            address,
            format!(
                "GET {path} HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\n\r\n",
                request_id.as_uuid()
            ),
        )
        .await;
        server
            .await
            .expect("management task joins")
            .expect("request");
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "response: {response}"
        );
        request_ids.push(request_id);
    }

    let service = crate::logging_runtime_state()
        .expect("installed logging runtime")
        .service_for_test()
        .expect("logging service");
    for request_id in request_ids {
        assert!(
            service
                .registry_ref()
                .get_recent(&request_id.as_uuid().to_string())
                .is_none(),
            "excluded route must not register a lifecycle"
        );
    }
}

#[tokio::test]
async fn management_health_remains_available_in_headless_mode() {
    let state = build_test_mesh_api().await;
    state.set_headless(true).await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "response: {response}"
    );
    assert_eq!(json_body(&response)["status"], "ok");
}

#[tokio::test]
async fn management_health_reports_client_worker_and_serving_modes() {
    let state = build_test_mesh_api().await;
    let node = state.node().await;

    node.set_role(crate::mesh::NodeRole::Client).await;
    let (address, server) = spawn_management_test_server(state.clone()).await;
    let client_response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();
    assert_eq!(json_body(&client_response)["mode"], "client");
    assert_eq!(
        json_body(&client_response)["serving"]["status"],
        "not_applicable"
    );

    node.set_role(crate::mesh::NodeRole::Worker).await;
    let (address, server) = spawn_management_test_server(state.clone()).await;
    let worker_response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();
    assert_eq!(json_body(&worker_response)["mode"], "worker");
    assert_eq!(json_body(&worker_response)["serving"]["status"], "idle");

    state.update(true, true).await;
    node.set_role(crate::mesh::NodeRole::Host { http_port: 9337 })
        .await;
    node.set_hosted_models(vec!["served-model".to_string()])
        .await;
    let (address, server) = spawn_management_test_server(state).await;
    let serving_response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();
    assert_eq!(json_body(&serving_response)["mode"], "serving");
    assert_eq!(json_body(&serving_response)["serving"]["status"], "healthy");
    assert_eq!(
        json_body(&serving_response)["serving"]["models"],
        serde_json::json!(["served-model"])
    );
}

#[tokio::test]
async fn management_health_reports_ready_local_worker_stage_models_only() {
    let state = build_test_mesh_api().await;
    let node = state.node().await;
    let remote = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let local_id = node.id();
    node.stage_topologies
        .lock()
        .await
        .record_status(health_test_stage_status(
            Some(local_id),
            "local-split-model",
            crate::inference::skippy::StageRuntimeState::Ready,
        ));
    let mut remote_status = health_test_stage_status(
        Some(remote.id()),
        "remote-split-model",
        crate::inference::skippy::StageRuntimeState::Ready,
    );
    remote_status.stage_id = "remote-health-stage".to_string();
    node.stage_topologies
        .lock()
        .await
        .record_status(remote_status);

    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert_eq!(json_body(&response)["mode"], "worker");
    assert_eq!(json_body(&response)["serving"]["status"], "healthy");
    assert_eq!(
        json_body(&response)["serving"]["models"],
        serde_json::json!(["local-split-model"])
    );
}

#[tokio::test]
async fn management_health_reports_failed_worker_stage_as_unhealthy() {
    let state = build_test_mesh_api().await;
    let node = state.node().await;
    let local_id = node.id();
    node.stage_topologies
        .lock()
        .await
        .record_status(health_test_stage_status(
            Some(local_id),
            "failed-split-model",
            crate::inference::skippy::StageRuntimeState::Failed,
        ));

    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert_eq!(json_body(&response)["mode"], "worker");
    assert_eq!(json_body(&response)["serving"]["status"], "unhealthy");
    assert_eq!(
        json_body(&response)["serving"]["models"],
        serde_json::json!([])
    );
}

#[tokio::test]
async fn management_health_reports_failed_host_process_as_unhealthy() {
    let state = build_test_mesh_api().await;
    let node = state.node().await;
    node.set_role(crate::mesh::NodeRole::Host { http_port: 9337 })
        .await;
    {
        let inner = state.inner.lock().await;
        inner
            .runtime_data_producer
            .publish_local_processes(|processes| {
                processes.clear();
                processes.push(crate::runtime_data::RuntimeProcessSnapshot {
                    model: "exited-host-model".to_string(),
                    state: "exited".to_string(),
                    ..Default::default()
                });
                true
            });
    }

    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert_eq!(json_body(&response)["mode"], "serving");
    assert_eq!(json_body(&response)["serving"]["status"], "unhealthy");
    assert_eq!(
        json_body(&response)["serving"]["models"],
        serde_json::json!([])
    );
}

#[tokio::test]
async fn management_health_treats_graceful_host_shutdown_as_idle() {
    for process_state in ["shutting down", "stopped"] {
        let state = build_test_mesh_api().await;
        let node = state.node().await;
        node.set_role(crate::mesh::NodeRole::Host { http_port: 9337 })
            .await;
        {
            let inner = state.inner.lock().await;
            inner
                .runtime_data_producer
                .publish_local_processes(|processes| {
                    processes.clear();
                    processes.push(crate::runtime_data::RuntimeProcessSnapshot {
                        model: "stopping-host-model".to_string(),
                        state: process_state.to_string(),
                        ..Default::default()
                    });
                    true
                });
        }

        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
        )
        .await;
        server.await.unwrap().unwrap();

        assert_eq!(json_body(&response)["mode"], "serving");
        assert_eq!(
            json_body(&response)["serving"]["status"],
            "idle",
            "state={process_state}"
        );
    }
}

#[tokio::test]
async fn management_health_reports_failed_local_stage_on_serving_host() {
    let state = build_test_mesh_api().await;
    let node = state.node().await;
    node.set_role(crate::mesh::NodeRole::Host { http_port: 9337 })
        .await;
    node.set_hosted_models(vec!["healthy-host-model".to_string()])
        .await;
    node.stage_topologies
        .lock()
        .await
        .record_status(health_test_stage_status(
            Some(node.id()),
            "failed-host-stage-model",
            crate::inference::skippy::StageRuntimeState::Failed,
        ));

    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert_eq!(json_body(&response)["mode"], "serving");
    assert_eq!(json_body(&response)["serving"]["status"], "degraded");
    assert_eq!(
        json_body(&response)["serving"]["models"],
        serde_json::json!(["healthy-host-model"])
    );
}

#[tokio::test]
async fn management_health_reports_cached_plugin_models_for_serving_host() {
    let plugin_manager = build_inference_endpoint_plugin_manager(&["plugin-model"]).await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    let node = state.node().await;
    node.set_role(crate::mesh::NodeRole::Host { http_port: 9337 })
        .await;

    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert_eq!(json_body(&response)["mode"], "serving");
    assert_eq!(json_body(&response)["serving"]["status"], "healthy");
    assert_eq!(
        json_body(&response)["serving"]["models"],
        serde_json::json!(["plugin-model"])
    );
}

fn health_test_stage_status(
    node_id: Option<iroh::EndpointId>,
    model_id: &str,
    state: crate::inference::skippy::StageRuntimeState,
) -> crate::mesh::StageRuntimeStatus {
    crate::mesh::StageRuntimeStatus {
        topology_id: "health-topology".to_string(),
        run_id: "health-run".to_string(),
        model_id: model_id.to_string(),
        backend: "skippy".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: "health-stage".to_string(),
        stage_index: 1,
        node_id,
        layer_start: 0,
        layer_end: 1,
        admission: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        state,
        bind_addr: "127.0.0.1:9000".to_string(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: None,
        ctx_size: 128,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        error: None,
        shutdown_generation: 1,
    }
}

#[tokio::test]
#[serial]
async fn management_mutation_propagates_request_id_and_records_one_terminal_lifecycle() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        replay_capacity: 16,
        ..Default::default()
    })
    .await;
    let request_id = "00000000-0000-4000-8000-000000000011";
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;

    let body = r#"{"toml":"version = 1\n","path":"x"}"#;
    let response = send_management_request(
        address,
        format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\nx-request-id: {request_id}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains(&format!("x-request-id: {request_id}")));
    let records = crate::logging_runtime_state()
        .and_then(|state| state.replay_bus())
        .expect("enabled logging replay bus")
        .replay_window()
        .records;
    let lifecycle_records = records
        .iter()
        .filter(|record| record.entry.payload.contains(request_id))
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle_records
            .iter()
            .filter(|record| record.entry.payload.contains("admitted"))
            .count(),
        1
    );
    assert_eq!(
        lifecycle_records
            .iter()
            .filter(|record| record.entry.payload.contains("completed"))
            .count(),
        1
    );
    assert!(lifecycle_records.iter().any(|record| {
        record.entry.payload.contains("management_api")
            && record.entry.payload.contains("management_post")
    }));

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn management_4xx_records_rejected_terminal_lifecycle() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        replay_capacity: 16,
        ..Default::default()
    })
    .await;
    let request_id = "00000000-0000-4000-8000-000000000031";
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;

    let response = send_management_request(
        address,
        format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\nx-request-id: {request_id}\r\nContent-Length: 0\r\n\r\n"
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    let records = crate::logging_runtime_state()
        .and_then(|state| state.replay_bus())
        .expect("enabled logging replay bus")
        .replay_window()
        .records;
    let lifecycle_records = records
        .iter()
        .filter(|record| record.entry.payload.contains(request_id))
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle_records
            .iter()
            .filter(|record| record.entry.payload.contains("rejected"))
            .count(),
        1
    );
    assert!(
        lifecycle_records
            .iter()
            .any(|record| record.entry.payload.contains("management_http_rejected"))
    );

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn management_5xx_records_failed_terminal_lifecycle() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        replay_capacity: 16,
        ..Default::default()
    })
    .await;
    let request_id = "00000000-0000-4000-8000-000000000032";
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;

    let response = send_management_request(
        address,
        format!(
            "POST /api/runtime/mesh-guardrails HTTP/1.1\r\nHost: localhost\r\nx-request-id: {request_id}\r\nContent-Length: 0\r\n\r\n"
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
    let records = crate::logging_runtime_state()
        .and_then(|state| state.replay_bus())
        .expect("enabled logging replay bus")
        .replay_window()
        .records;
    let lifecycle_records = records
        .iter()
        .filter(|record| record.entry.payload.contains(request_id))
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle_records
            .iter()
            .filter(|record| record.entry.payload.contains("failed"))
            .count(),
        1
    );
    assert!(
        lifecycle_records
            .iter()
            .any(|record| record.entry.payload.contains("management_http_failed"))
    );

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
async fn test_api_events_sends_initial_payload_and_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("HTTP/1.1 200 OK"));
    assert!(initial_text.contains("Content-Type: text/event-stream"));
    assert!(initial_text.contains("\"llama_ready\":false"));

    state.update(true, true).await;
    let updated =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"llama_ready\":true"));
    assert!(updated_text.contains("\"is_host\":true"));

    drop(stream);
    handle.abort();
}

#[tokio::test]
async fn test_api_events_push_publication_state_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let _initial = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"private\"",
        Duration::from_secs(2),
    )
    .await;

    state
        .set_publication_state(crate::api::PublicationState::PublishFailed)
        .await;
    let updated = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"publish_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"publication_state\":\"publish_failed\""));

    drop(stream);
    handle.abort();
}
