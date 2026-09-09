pub(super) fn make_test_peer(id: EndpointId, rtt_ms: Option<u32>, vram_gb: u64) -> PeerInfo {
    PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: vram_gb * 1024 * 1024 * 1024,
        rtt_ms,
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
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        owner_summary: OwnershipSummary::default(),
        advertised_model_throughput: vec![],
        cache_affinity: None,

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        inference_admission_state: None,
    }
}
pub(super) fn test_owner_keypair(
    signing_seed: u8,
    encryption_seed: u8,
) -> crate::crypto::OwnerKeypair {
    crate::crypto::OwnerKeypair::from_bytes(&[signing_seed; 32], &[encryption_seed; 32])
        .expect("test owner keypair must be valid")
}

pub(super) fn requirement_policy_owner() -> crate::crypto::OwnerKeypair {
    test_owner_keypair(0xb1, 0xb2)
}

pub(super) fn proto_signed_node_ownership(
    ownership: &crate::crypto::SignedNodeOwnership,
) -> crate::proto::node::SignedNodeOwnership {
    crate::proto::node::SignedNodeOwnership {
        version: ownership.claim.version,
        cert_id: ownership.claim.cert_id.clone(),
        owner_id: ownership.claim.owner_id.clone(),
        owner_sign_public_key: hex::decode(&ownership.claim.owner_sign_public_key)
            .expect("test owner_sign_public_key must decode"),
        node_endpoint_id: hex::decode(&ownership.claim.node_endpoint_id)
            .expect("test node_endpoint_id must decode"),
        issued_at_unix_ms: ownership.claim.issued_at_unix_ms,
        expires_at_unix_ms: ownership.claim.expires_at_unix_ms,
        node_label: ownership.claim.node_label.clone(),
        hostname_hint: ownership.claim.hostname_hint.clone(),
        signature: hex::decode(&ownership.signature).expect("test signature must decode"),
    }
}

pub(super) async fn open_owner_control_stream(
    target: &Node,
    owner_keypair: &crate::crypto::OwnerKeypair,
) -> Result<(
    Endpoint,
    iroh::endpoint::SendStream,
    iroh::endpoint::RecvStream,
    EndpointId,
)> {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let ownership = sign_node_ownership(
        owner_keypair,
        endpoint.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let control_addr = Node::decode_invite_token(
        &target
            .control_endpoint()
            .await
            .expect("control endpoint should be available for owner-control tests"),
    )?;
    let conn = endpoint.connect(control_addr, ALPN_CONTROL_V1).await?;
    let (mut send, recv) = conn.open_bi().await?;
    write_len_prefixed(
        &mut send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: Some(crate::proto::node::OwnerControlHandshake {
                ownership: Some(proto_signed_node_ownership(&ownership)),
            }),
            request: None,
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let endpoint_id = endpoint.id();
    Ok((endpoint, send, recv, endpoint_id))
}

pub(super) async fn read_owner_control_envelope(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<crate::proto::node::OwnerControlEnvelope> {
    let bytes = crate::protocol::read_len_prefixed(recv).await?;
    let envelope = crate::proto::node::OwnerControlEnvelope::decode(bytes.as_slice())?;
    envelope
        .validate_frame()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(envelope)
}

pub(super) async fn start_owner_control_test_server(
    owner_keypair: &crate::crypto::OwnerKeypair,
    config_dir: &std::path::Path,
) -> Result<(Node, SecretKey, std::path::PathBuf)> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::NodeRole::Host { http_port: 9337 }).await?;
    let config_path = config_dir.join("config.toml");
    *node.config_state.lock().await =
        crate::runtime::config_state::ConfigState::load(&config_path).unwrap_or_default();

    let ownership = sign_node_ownership(
        owner_keypair,
        node.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let trust_store = TrustStore::default();
    let owner_summary = verify_node_ownership(
        Some(&ownership),
        node.id().as_bytes(),
        &trust_store,
        TrustPolicy::Off,
        current_time_unix_ms(),
    );
    *node.owner_attestation.lock().await = Some(ownership);
    *node.owner_summary.lock().await = owner_summary;
    *node.trust_store.lock().await = trust_store;
    node.maybe_start_control_listener(secret_key.clone(), None, None)
        .await?;
    Ok((node, secret_key, config_path))
}

/// Wait until `node` has `target` in its peers list. Times out after 5 s.
/// Poll `node.peers()` until `target` appears in the list.
///
/// Panics (via `expect`) if `target` is not admitted within 5 seconds.
pub(super) async fn wait_for_peer(node: &Node, target: EndpointId) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if node.peers().await.iter().any(|p| p.id == target) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("peer was not admitted within 5 s");
}

pub(super) fn requirement_policy(trusted_signer: &str) -> crate::MeshGenesisPolicy {
    crate::MeshGenesisPolicy::new(
        requirement_policy_owner().owner_id(),
        1_717_171_717_000,
        crate::MeshRequirements {
            node_version: crate::NodeVersionBounds::default(),
            protocol_generation: crate::ProtocolGenerationBounds {
                min: Some(NODE_PROTOCOL_GENERATION),
                max: Some(NODE_PROTOCOL_GENERATION),
            },
            release_attestation: crate::ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec![trusted_signer.to_string()],
            },
        },
    )
    .expect("test mesh policy should validate")
}

pub(super) fn requirement_policy_without_release_attestation() -> crate::MeshGenesisPolicy {
    crate::MeshGenesisPolicy::new(
        requirement_policy_owner().owner_id(),
        1_717_171_717_000,
        crate::MeshRequirements {
            node_version: crate::NodeVersionBounds::default(),
            protocol_generation: crate::ProtocolGenerationBounds {
                min: Some(NODE_PROTOCOL_GENERATION),
                max: Some(NODE_PROTOCOL_GENERATION),
            },
            release_attestation: crate::ReleaseAttestationRequirement {
                required: false,
                allowed_signer_keys: vec![],
            },
        },
    )
    .expect("test mesh policy should validate")
}

pub(super) fn test_release_signing_key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

pub(super) fn test_release_signer_key_id(seed: u8) -> String {
    format!(
        "ed25519:{}",
        hex::encode(test_release_signing_key(seed).verifying_key().as_bytes())
    )
}

pub(super) fn test_release_attestation_with_seed(seed: u8) -> crate::ReleaseBuildAttestation {
    let signing_key = test_release_signing_key(seed);
    let mut attestation = crate::ReleaseBuildAttestation {
        version: 1,
        node_version: crate::VERSION.to_string(),
        build_id: "test-build".into(),
        commit: "deadbeef".into(),
        target_triple: "x86_64-apple-darwin".into(),
        supported_protocol_generation_min: Some(NODE_PROTOCOL_GENERATION),
        supported_protocol_generation_max: Some(NODE_PROTOCOL_GENERATION),
        artifact_digest: Some("sha256:test".into()),
        signer_key_id: test_release_signer_key_id(seed),
        signature_algorithm: "ed25519".into(),
        signature: vec![0; 64],
    };
    attestation.signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &attestation
            .canonical_bytes()
            .expect("canonical release attestation bytes"),
    )
    .to_bytes()
    .to_vec();
    attestation
}

pub(super) fn test_release_attestation(signer_key_id: &str) -> crate::ReleaseBuildAttestation {
    let mut attestation = test_release_attestation_with_seed(9);
    attestation.signer_key_id = signer_key_id.into();
    attestation
}

pub(super) fn direct_proof_signing_key(seed: u8) -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes)
}

pub(super) fn direct_proof_for_announcement(
    sender_seed: u8,
    mesh_id: &str,
    policy_hash: &str,
    release_attestation: Option<&crate::ReleaseBuildAttestation>,
) -> crate::DirectNodeAdmissionProof {
    direct_proof_for_announcement_at(
        sender_seed,
        mesh_id,
        policy_hash,
        release_attestation,
        current_time_unix_ms(),
    )
}

pub(super) fn direct_proof_for_announcement_at(
    sender_seed: u8,
    mesh_id: &str,
    policy_hash: &str,
    release_attestation: Option<&crate::ReleaseBuildAttestation>,
    timestamp_unix_ms: u64,
) -> crate::DirectNodeAdmissionProof {
    let signing_key =
        ed25519_dalek::SigningKey::from_bytes(&direct_proof_signing_key(sender_seed).to_bytes());
    let attestation_hash = release_attestation
        .map(|attestation| {
            attestation
                .canonical_hash_hex()
                .unwrap_or_else(|_| "invalid-release-attestation".to_string())
        })
        .unwrap_or_else(|| "missing-release-attestation".to_string());
    let mut proof = crate::DirectNodeAdmissionProof {
        version: 1,
        sender_id: make_test_endpoint_id(sender_seed).as_bytes().to_vec(),
        mesh_id: mesh_id.to_string(),
        policy_hash: policy_hash.to_string(),
        attestation_hash,
        timestamp_unix_ms,
        signature_algorithm: "ed25519".to_string(),
        signature: vec![],
    };
    proof.signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &proof
            .canonical_bytes()
            .expect("canonical direct proof bytes"),
    )
    .to_bytes()
    .to_vec();
    proof
}

pub(super) async fn install_requirement_policy(
    node: &Node,
    policy: &crate::MeshGenesisPolicy,
) -> Result<()> {
    let mesh_id = policy
        .policy_derived_mesh_id()
        .map_err(|reason| anyhow::anyhow!("invalid test mesh id: {reason:?}"))?;
    let policy_hash = policy
        .canonical_hash_hex()
        .map_err(|reason| anyhow::anyhow!("invalid test policy hash: {reason:?}"))?;
    let owner = requirement_policy_owner();
    let signed_policy = crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner)
        .map_err(|reason| anyhow::anyhow!("invalid test signed policy: {reason:?}"))?;
    let token = crate::SignedBootstrapToken::sign(
        vec![serde_json::to_vec(&node.endpoint_addr_for_advertisement())?],
        &signed_policy,
        Some(current_time_unix_ms() + SIGNED_BOOTSTRAP_TOKEN_LIFETIME_MS),
        &owner,
    )
    .map_err(|reason| anyhow::anyhow!("invalid test bootstrap token: {reason:?}"))?;
    node.install_requirement_aware_mesh_state(
        mesh_id,
        policy_hash,
        policy.clone(),
        Some(signed_policy),
        Some(token),
    )
    .await
}

pub(super) async fn configure_requirement_node(
    node: &Node,
    policy: &crate::MeshGenesisPolicy,
    signer: Option<&str>,
) -> Result<()> {
    install_requirement_policy(node, policy).await?;
    *node.release_attestation.lock().await = signer.map(test_release_attestation);
    Ok(())
}

pub(super) fn requirement_peer_announcement(
    sender_seed: u8,
    policy: &crate::MeshGenesisPolicy,
    release_attestation: Option<crate::ReleaseBuildAttestation>,
    direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
) -> super::PeerAnnouncement {
    super::PeerAnnouncement {
        addr: EndpointAddr {
            id: make_test_endpoint_id(sender_seed),
            addrs: Default::default(),
        },
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: Some(crate::VERSION.to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some(policy.policy_derived_mesh_id().expect("mesh id")),
        mesh_policy_hash: Some(policy.canonical_hash_hex().expect("policy hash")),
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
        served_model_runtime: vec![],
        owner_attestation: None,
        genesis_policy: None,
        release_attestation,
        direct_admission_proof,
        artifact_transfer_supported: true,
        stage_protocol_generation_supported: true,
        stage_status_list_supported: true,
        local_gguf_content_id_supported: true,
        advertised_model_throughput: vec![],
        cache_affinity: None,
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
        inference_admission_state: None,
        attested_log_head: None,
    }
}
