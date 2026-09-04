use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type PromptShapeObservation = (Option<String>, Option<u64>, Option<u64>);

fn mesh_request_plan(
    model: &str,
    target_hosts: Vec<iroh::EndpointId>,
    affinity_applied: bool,
) -> MeshRequestPlan {
    MeshRequestPlan {
        effective_model: Some(model.to_string()),
        auto_session_key: None,
        target_hosts,
        affinity_applied,
    }
}

#[derive(Default)]
struct PromptShapeSink {
    observations: std::sync::Mutex<Vec<PromptShapeObservation>>,
}

impl crate::network::metrics::RoutingTelemetrySink for PromptShapeSink {
    fn observe_inflight_requests(&self, _current: u64) {}

    fn record_model_request(
        &self,
        _model: Option<&str>,
        _attempts: usize,
        _outcome: crate::network::metrics::RequestOutcome,
    ) {
    }

    fn record_route_attempt(
        &self,
        _model: Option<&str>,
        _target: &crate::network::metrics::AttemptTarget,
        _outcome: crate::network::metrics::AttemptOutcome,
    ) {
    }

    fn record_prompt_shape(
        &self,
        model: Option<&str>,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        _outcome: crate::network::metrics::RequestOutcome,
    ) {
        self.observations.lock().expect("prompt-shape lock").push((
            model.map(str::to_string),
            prompt_tokens,
            completion_tokens,
        ));
    }
}

fn test_peer_serving_model(peer_id: iroh::EndpointId, model: &str) -> mesh::PeerInfo {
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![model.to_string()],
        hosted_models: vec![model.to_string()],
        hosted_models_known: true,
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
        served_model_descriptors: vec![local_gguf_descriptor(model)],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![],
        cache_affinity: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        inference_admission_state: None,
    }
}

async fn test_node_with_remote_models(models: &[(&str, iroh::EndpointId)]) -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node should start");
    for (model, peer_id) in models {
        node.insert_test_peer(test_peer_serving_model(*peer_id, model))
            .await;
    }
    node
}
fn text_auto_request() -> BufferedHttpRequest {
    let body = serde_json::json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let body_bytes = serde_json::to_vec(&body).expect("request body should serialize");
    BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        client_path: "/v1/chat/completions".to_string(),
        request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
        body_json: Some(body),
        body_json_attempted: true,
        body_bytes: Some(body_bytes),
        body_len_bytes: 0,
        completion_tokens: None,
        model_name: Some("auto".to_string()),
        stream: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    }
}
fn unparsed_chat_request(model: &str) -> BufferedHttpRequest {
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a scheduler."},
            {"role": "user", "content": "Explain this trace."}
        ]
    });
    let body_bytes = serde_json::to_vec(&body).expect("request body should serialize");
    BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        client_path: "/v1/chat/completions".to_string(),
        request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_len_bytes: body_bytes.len(),
        body_bytes: Some(body_bytes),
        completion_tokens: None,
        model_name: Some(model.to_string()),
        stream: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    }
}
fn large_tokenize_request(model: &str) -> BufferedHttpRequest {
    BufferedHttpRequest {
        raw: b"exact tokenizer request bytes".to_vec(),
        method: "POST".to_string(),
        path: "/v1/tokenize".to_string(),
        client_path: "/v1/tokenize".to_string(),
        request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 140_000,
        completion_tokens: None,
        model_name: Some(model.to_string()),
        stream: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    }
}
fn local_gguf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.to_string(),
            source_kind: mesh::ModelSourceKind::LocalGguf,
            local_file_name: Some(format!("{model_name}.gguf")),
            ..Default::default()
        },
        ..Default::default()
    }
}
#[test]
fn test_remote_retry_policy_only_retries_uncommitted_failures() {
    assert!(should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::RetryableUnavailable
    ));
    assert!(should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::RetryableTimeout
    ));
    assert!(!should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::RetryableContextOverflow
    ));
    assert!(!should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::RetryableResponseQuality(ResponseQualityFailure::EmptyAssistantOutput)
    ));
    assert!(!should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::ClientDisconnected
    ));
    assert!(!should_retry_uncommitted_remote_attempt(
        RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
            cache_cost: None,
        }
    ));
}

#[test]
fn passive_mesh_plan_spreads_overlap_and_releases_both_reservations() {
    let first = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let second = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let plan = mesh_request_plan("test", vec![first, second], false);
    let affinity = AffinityRouter::new();

    let (first_order, first_reservation) = reserve_mesh_request_target(&plan, &affinity);
    let (second_order, second_reservation) = reserve_mesh_request_target(&plan, &affinity);

    assert_eq!(first_order, vec![first, second]);
    assert_eq!(second_order, vec![second, first]);
    assert_eq!(affinity.stats_snapshot().reservation_active, 2);
    drop(first_reservation);
    drop(second_reservation);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[test]
fn passive_mesh_plan_keeps_affinity_authoritative_under_pressure() {
    let first = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let second = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let unaffined = mesh_request_plan("test", vec![first, second], false);
    let affined = mesh_request_plan("test", vec![first, second], true);
    let affinity = AffinityRouter::new();

    let (_, pressure) = reserve_mesh_request_target(&unaffined, &affinity);
    let (affined_order, affined_reservation) = reserve_mesh_request_target(&affined, &affinity);

    assert_eq!(affined_order, vec![first, second]);
    drop(pressure);
    drop(affined_reservation);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn remote_tokenizer_plan_routes_identity_model_without_context_rejection() -> Result<()> {
    let model = "acme/code-model:Q4_K_M";
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let node = test_node_with_remote_models(&[(model, peer_id)]).await;
    let mut peer = test_peer_serving_model(peer_id, model);
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: model.to_owned(),
        identity_hash: None,
        context_length: Some(32_768),
        ready: true,
    }];
    node.insert_test_peer(peer).await;
    let affinity = AffinityRouter::new();
    let mut request = large_tokenize_request(model);
    let raw_before_plan = request.raw.clone();

    let generation_budget =
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    assert!(generation_budget.is_some_and(|tokens| tokens > 32_768));
    assert!(
        order_remote_hosts_by_context(
            &node,
            model,
            generation_budget,
            std::slice::from_ref(&peer_id),
        )
        .await
        .is_empty(),
        "a generation budget would incorrectly reject the tokenizer target"
    );

    let plan = build_mesh_request_plan(&node, &mut request, false, &affinity)
        .await
        .map_err(|_| anyhow::anyhow!("tokenizer request plan should resolve"))?;

    assert_eq!(request_context_budget(&request), None);
    assert_eq!(plan.effective_model.as_deref(), Some(model));
    assert_eq!(plan.target_hosts, vec![peer_id]);
    assert_eq!(request.raw, raw_before_plan);
    assert!(request.body_json.is_none());
    assert!(!request.body_json_attempted);
    Ok(())
}

#[test]
fn tokenizer_effective_model_cannot_override_authoritative_identity() {
    let model = "acme/code-model:Q4_K_M";
    let mut request = large_tokenize_request(model);
    let raw_before = request.raw.clone();

    rewrite_effective_model(&mut request, Some("different/internal-model"));

    assert_eq!(request.model_name.as_deref(), Some(model));
    assert_eq!(request.raw, raw_before);
}

#[test]
fn single_remote_target_still_derives_a_cache_prefix() {
    let model = "acme/code-model:Q4_K_M";
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let affinity = AffinityRouter::with_config(true, true);
    let mut request = unparsed_chat_request(model);

    let prepared = prepare_mesh_targets(
        &mut request,
        Some(model),
        std::slice::from_ref(&peer_id),
        &affinity,
    );

    assert!(request.body_json_attempted);
    assert!(request.body_json.is_some());
    assert!(prepared.prefix_hash.is_some());
}

#[tokio::test]
async fn prefix_kill_switch_prevents_cache_evidence_reordering() -> Result<()> {
    use mesh_llm_routing::cache_inventory::{
        CACHE_AFFINITY_SALT_BYTES, CacheAffinityAdvertisement, CacheAffinityEntry, CacheTier,
        prefix_digest,
    };

    let model = "acme/code-model:Q4_K_M";
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let node = test_node_with_remote_models(&[(model, peer_id)]).await;
    let affinity = AffinityRouter::with_config(false, true);
    let mut request = unparsed_chat_request(model);
    let mut prepared = prepare_mesh_targets(
        &mut request,
        Some(model),
        std::slice::from_ref(&peer_id),
        &affinity,
    );
    let prefix_hash = prepared.prefix_hash.expect("derived prefix");
    let salt = [3; CACHE_AFFINITY_SALT_BYTES];
    let mut peer = test_peer_serving_model(peer_id, model);
    peer.cache_affinity = Some(CacheAffinityAdvertisement {
        salt,
        epoch: 1,
        generated_at_unix_ms: crate::mesh::current_time_unix_ms(),
        ttl_ms: 120_000,
        entries: vec![CacheAffinityEntry {
            model: model.to_string(),
            prefix_digest: prefix_digest(&salt, model, prefix_hash),
            matched_tokens: 512,
            suffix_prefill_tokens: 32,
            tier: CacheTier::L1,
            restore_micros: 0,
            queue_delay_micros: 0,
            prefill_micros_per_token: 0,
        }],
    });
    node.insert_test_peer(peer).await;

    let ordered = order_mesh_target_hosts(&node, Some(model), None, &mut prepared, &affinity).await;

    assert_eq!(ordered, vec![peer_id]);
    assert!(prepared.cache_target.is_none());
    assert_eq!(affinity.stats_snapshot().prefix_lookups, 0);
    Ok(())
}

#[tokio::test]
async fn cached_auto_model_stays_sticky_when_no_ready_remote_model_exists() -> Result<()> {
    let cached_model = "cached-cooling-model-31B";
    let alternate_model = "alternate-cooling-model-31B";
    let cached_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let alternate_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let node = test_node_with_remote_models(&[
        (cached_model, cached_peer),
        (alternate_model, alternate_peer),
    ])
    .await;
    let affinity = AffinityRouter::new();
    let key = 0xA11CE;
    affinity.remember_auto_model(key, cached_model);
    affinity.record_target_outcome(
        Some(cached_model),
        &election::InferenceTarget::Remote(cached_peer),
        TargetHealthOutcome::Unavailable,
    );
    affinity.record_target_outcome(
        Some(alternate_model),
        &election::InferenceTarget::Remote(alternate_peer),
        TargetHealthOutcome::Unavailable,
    );
    let descriptors = vec![
        local_gguf_descriptor(cached_model),
        local_gguf_descriptor(alternate_model),
    ];
    let media = router::MediaRequirements::default();
    let caps = crate::models::ModelCapabilities::default();
    let available = vec![
        router::RoutingCandidate::unscored(cached_model, caps),
        router::RoutingCandidate::unscored(alternate_model, caps),
    ];
    let ready_models = auto_route::ready_remote_models(&node, None, &available, &affinity).await;
    assert!(ready_models.is_empty());

    let cached = lookup_cached_auto_model(
        &node,
        &descriptors,
        &affinity,
        Some(key),
        &media,
        &ready_models,
    )
    .await;

    assert_eq!(cached.as_deref(), Some(cached_model));
    assert_eq!(
        affinity.lookup_auto_model(key).as_deref(),
        Some(cached_model)
    );
    Ok(())
}

#[tokio::test]
async fn auto_model_cache_switches_when_ready_alternate_exists() -> Result<()> {
    let cached_model = "cached-cooling-model-31B";
    let alternate_model = "ready-alternate-model-31B";
    let cached_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let alternate_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let node = test_node_with_remote_models(&[
        (cached_model, cached_peer),
        (alternate_model, alternate_peer),
    ])
    .await;
    let affinity = AffinityRouter::new();
    let key = 0xB0B;
    affinity.remember_auto_model(key, cached_model);
    affinity.record_target_outcome(
        Some(cached_model),
        &election::InferenceTarget::Remote(cached_peer),
        TargetHealthOutcome::Unavailable,
    );
    let served = vec![cached_model.to_string(), alternate_model.to_string()];
    let descriptors = vec![
        local_gguf_descriptor(cached_model),
        local_gguf_descriptor(alternate_model),
    ];
    let mut request = text_auto_request();

    let resolved = resolve_auto_model_request(AutoModelRequestArgs {
        node: &node,
        request: &mut request,
        served: &served,
        descriptors: &descriptors,
        is_auto_request: true,
        auto_session_key: Some(key),
        required_tokens: None,
        affinity: &affinity,
    })
    .await;

    assert!(matches!(
        resolved,
        AutoModelResolution::Model(Some(model)) if model == alternate_model
    ));
    assert_eq!(
        affinity.lookup_auto_model(key).as_deref(),
        Some(alternate_model)
    );
    Ok(())
}
#[test]
fn test_capture_path_for_request_uses_client_path() {
    let request = BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".to_string(),
        path: "/v1/chat/completions?foo=1".to_string(),
        client_path: "/v1/responses?foo=1".to_string(),
        request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 0,
        completion_tokens: None,
        stream: None,
        model_name: Some("qwen".to_string()),
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::OpenAiResponsesStream,
        correlation_id: None,
    };

    assert_eq!(capture_path_for_request(&request), "/v1/responses?foo=1");
}

#[tokio::test]
async fn routed_completion_http_boundary_preserves_configured_served_alias() -> Result<()> {
    let backend = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let backend_port = backend.local_addr()?.port();
    let backend_task = tokio::spawn(async move {
        let (mut stream, _) = backend.accept().await.expect("backend connection");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).await.expect("backend request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        let request = String::from_utf8_lossy(&request);
        assert!(request.contains(r#""model":"public-model""#));
        let body = r#"{"id":"completion-1","model":"public-model","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("backend response");
        stream.shutdown().await.expect("backend shutdown");
    });

    let downstream = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let downstream_address = downstream.local_addr()?;
    let client = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(downstream_address)
            .await
            .expect("downstream connection");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("downstream response");
        String::from_utf8(response).expect("utf-8 response")
    });
    let (downstream_stream, _) = downstream.accept().await?;
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
    let request_body = r#"{"model":"public-model","messages":[{"role":"user","content":"hello"}]}"#;
    let prefetched = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_body.len(),
        request_body
    );

    let outcome = route_to_target(
        node,
        downstream_stream.into(),
        Some("public-model"),
        election::InferenceTarget::Local(backend_port),
        prefetched.as_bytes(),
        RouteTargetContext {
            request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
            response_adapter: ResponseAdapter::None,
            route_observer: OpenAiRouteObserver::default(),
        },
    )
    .await;

    assert!(matches!(
        outcome,
        RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200,
            ..
        }
    ));
    let response = client.await.expect("downstream client task");
    assert!(response.contains(r#""model":"public-model""#));
    backend_task.await.expect("backend task");
    Ok(())
}

#[tokio::test]
async fn named_model_route_records_prompt_shape_from_usage() -> Result<()> {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
    let sink = std::sync::Arc::new(PromptShapeSink::default());
    node.set_routing_telemetry_sink(Some(sink.clone()));
    let request = text_auto_request();

    let outcome = finalize_route_model_result(
        &node,
        "public-model",
        &request,
        std::time::Instant::now(),
        1,
        RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200,
            usage: TokenUsage {
                prompt_tokens: Some(13),
                completion_tokens: Some(5),
                ..Default::default()
            },
        },
        &election::InferenceTarget::Local(9337),
    );

    assert!(matches!(
        outcome,
        RouteDispatchOutcome::RespondedWithUsage { .. }
    ));
    assert_eq!(
        sink.observations
            .lock()
            .expect("prompt-shape lock")
            .as_slice(),
        &[(Some("public-model".to_string()), Some(13), Some(5))]
    );
    Ok(())
}
