#[test]
fn derive_local_node_state_prefers_client() {
    let node_state = MeshApi::derive_local_node_state(true, true, true, true, "Qwen");
    assert_eq!(node_state, NodeState::Client);
    assert_eq!(MeshApi::derive_node_status(node_state), "Client");
}

#[test]
fn derive_local_node_state_returns_standby_without_ready_runtime() {
    let node_state = MeshApi::derive_local_node_state(false, false, false, false, "Qwen");

    assert_eq!(node_state, NodeState::Standby);
    assert_eq!(MeshApi::derive_node_status(node_state), "Standby");
}

#[test]
fn derive_local_node_state_returns_loading_for_declared_but_unready_work() {
    let host_loading = MeshApi::derive_local_node_state(false, true, false, false, "Qwen");
    let worker_loading = MeshApi::derive_local_node_state(false, false, false, true, "Qwen");

    assert_eq!(host_loading, NodeState::Loading);
    assert_eq!(worker_loading, NodeState::Loading);
    assert_eq!(MeshApi::derive_node_status(host_loading), "Loading");
    assert_eq!(MeshApi::derive_node_status(worker_loading), "Loading");
}

#[test]
fn derive_local_node_state_returns_serving_for_ready_runtime() {
    let host_serving = MeshApi::derive_local_node_state(false, true, true, false, "Qwen");
    let worker_serving = MeshApi::derive_local_node_state(false, false, true, true, "Qwen");

    assert_eq!(host_serving, NodeState::Serving);
    assert_eq!(worker_serving, NodeState::Serving);
    assert_eq!(MeshApi::derive_node_status(host_serving), "Serving");
    assert_eq!(MeshApi::derive_node_status(worker_serving), "Serving");
}

#[test]
fn derive_local_node_state_never_emits_legacy_idle_or_split_labels() {
    let labels = [
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            true, true, true, true, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, false, false, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, true, false, false, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, true, true, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, false, false, "",
        )),
    ];

    for label in labels {
        assert!(matches!(
            label.as_str(),
            "Client" | "Standby" | "Loading" | "Serving"
        ));
        assert_ne!(label, "Idle");
        assert_ne!(label, "Serving (split)");
        assert_ne!(label, "Worker (split)");
    }
}

fn make_test_state_endpoint_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    iroh::EndpointId::from(iroh::SecretKey::from_bytes(&bytes).public())
}

fn make_test_state_peer(seed: u8, role: mesh::NodeRole) -> mesh::PeerInfo {
    let id = make_test_state_endpoint_id(seed);
    mesh::PeerInfo {
        id,
        addr: iroh::EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role,
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
        last_seen: Instant::now(),
        last_mentioned: Instant::now(),
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
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        first_joined_mesh_ts: None,
        advertised_model_throughput: vec![],
        cache_affinity: None,

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        inference_admission_state: None,
    }
}

fn make_legacy_peer_fixture(
    seed: u8,
    role: mesh::NodeRole,
    serving_models: Vec<&str>,
) -> mesh::PeerInfo {
    let mut peer = make_test_state_peer(seed, role);
    peer.version = Some("0.54.0".into());
    peer.serving_models = serving_models.into_iter().map(str::to_string).collect();
    peer.hosted_models = vec![];
    peer.hosted_models_known = false;
    peer.served_model_runtime = vec![];
    peer
}

#[test]
fn derive_peer_state_prefers_client_role() {
    let mut peer = make_test_state_peer(1, mesh::NodeRole::Client);
    peer.serving_models = vec!["Qwen".into()];
    peer.hosted_models = vec!["Qwen".into()];
    peer.hosted_models_known = true;
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: Some(8192),
        ready: true,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Client);
}

#[test]
fn derive_peer_state_returns_serving_for_ready_runtime() {
    let mut peer = make_test_state_peer(2, mesh::NodeRole::Host { http_port: 9337 });
    peer.serving_models = vec!["Qwen".into()];
    peer.hosted_models = vec!["Qwen".into()];
    peer.hosted_models_known = true;
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: Some(8192),
        ready: true,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
}

#[test]
fn derive_peer_state_returns_loading_for_assigned_but_unready_peer() {
    let mut peer = make_test_state_peer(3, mesh::NodeRole::Worker);
    peer.serving_models = vec!["Qwen".into()];
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: None,
        ready: false,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Loading);
}

#[test]
fn derive_peer_state_returns_standby_for_connected_idle_peer() {
    let peer = make_test_state_peer(4, mesh::NodeRole::Worker);

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Standby);
}

#[test]
fn derive_peer_state_falls_back_to_legacy_serving_models() {
    let mut peer = make_test_state_peer(5, mesh::NodeRole::Worker);
    peer.serving_models = vec!["Qwen".into()];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
}

#[test]
fn legacy_peer_fixture_uses_backend_state_fallback() {
    let serving_peer =
        make_legacy_peer_fixture(6, mesh::NodeRole::Host { http_port: 9337 }, vec!["Qwen"]);
    let standby_peer = make_legacy_peer_fixture(7, mesh::NodeRole::Worker, vec![]);

    assert_eq!(
        MeshApi::derive_peer_state(&serving_peer),
        NodeState::Serving
    );
    assert_eq!(
        MeshApi::derive_peer_state(&standby_peer),
        NodeState::Standby
    );
}

#[test]
fn test_decode_runtime_model_path_decodes_percent_not_plus() {
    // %20 is a space; + is a literal plus in URL paths (not a space)
    assert_eq!(
        decode_runtime_model_path("/api/runtime/models/Llama%203.2+1B", "/api/runtime/models/"),
        Some("Llama 3.2+1B".into())
    );
}

#[test]
fn test_decode_runtime_model_path_decodes_utf8_multibyte() {
    // é is U+00E9, encoded in UTF-8 as 0xC3 0xA9
    assert_eq!(
        decode_runtime_model_path("/api/runtime/models/mod%C3%A9le", "/api/runtime/models/"),
        Some("modéle".into())
    );
    // invalid UTF-8 sequence should return None
    assert_eq!(
        decode_runtime_model_path("/api/runtime/models/%80", "/api/runtime/models/"),
        None
    );
}
