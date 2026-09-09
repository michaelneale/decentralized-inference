use super::*;

pub(crate) fn assert_mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&joiner, &policy, Some(&trusted_signer))
            .await
            .expect("configure joiner policy");

        host.start_accepting();
        joiner.start_accepting();
        joiner.sync_from_peer_for_tests(&host).await;
        host.sync_from_peer_for_tests(&joiner).await;

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
    });
}

pub(crate) fn assert_mesh_requirements_inbound_rejects_before_topology_announcement() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&joiner, &policy, None)
            .await
            .expect("configure joiner policy");

        host.start_accepting();
        joiner.start_accepting();

        let _error = joiner
            .join(&host.invite_token().await)
            .await
            .expect_err("join should fail");
        assert!(
            joiner.peers().await.iter().all(|peer| peer.id != host.id()),
            "inbound rejection must happen before the joiner receives host topology"
        );
        assert!(
            host.peers().await.iter().all(|peer| peer.id != joiner.id()),
            "host must not admit the rejected inbound peer"
        );
    });
}

pub(crate) fn assert_mesh_requirements_outbound_rejects_before_peer_promotion() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let initiator = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("initiator node");
        let remote = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("remote node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&initiator, &policy, Some(&trusted_signer))
            .await
            .expect("configure initiator policy");
        configure_requirement_node(&remote, &policy, None)
            .await
            .expect("configure remote policy");

        initiator.start_accepting();
        remote.start_accepting();

        initiator
            .connect_to_peer(remote.endpoint_addr_for_advertisement())
            .await
            .expect_err("outbound connect should fail before promotion");
        assert!(
            initiator
                .peers()
                .await
                .iter()
                .all(|peer| peer.id != remote.id()),
            "noncompliant outbound peer must never become admitted/routable"
        );
        assert!(
            !initiator
                .state
                .lock()
                .await
                .connections
                .contains_key(&remote.id()),
            "failed outbound admission must not leave a cached connection"
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_missing_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let ann = requirement_peer_announcement(
            0x8f,
            &policy,
            Some(test_release_attestation(&trusted_signer)),
            None,
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(
            !is_peer_admitted(&node.state.lock().await.peers.clone(), &peer_id),
            "missing direct proof must reject before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofMissing
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_invalid_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let mut direct_proof = direct_proof_for_announcement(
            0x8e,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
        );
        direct_proof.signature[0] ^= 0x01;
        let ann = requirement_peer_announcement(
            0x8e,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(
            !is_peer_admitted(&node.state.lock().await.peers.clone(), &peer_id),
            "invalid direct proof must reject before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_stale_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let direct_proof = direct_proof_for_announcement_at(
            0x8d,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
            current_time_unix_ms() - crate::DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS - 1,
        );
        let ann = requirement_peer_announcement(
            0x8d,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofStale
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let direct_proof = direct_proof_for_announcement(
            0x8c,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
        );
        let ann = requirement_peer_announcement(
            0x8b,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofSenderIdMismatch
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_missing_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let ann = requirement_peer_announcement(0x8a, &policy, None, None);
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofMissing
        );
    });
}

/// On the fast auto-join probe, if `apply_gossip_announcements` fails after the
/// dispatcher has already been spawned, the winning candidate must be both
/// dropped from `state.connections` AND have its QUIC connection closed (so the
/// dispatcher unwinds and no orphaned, keep-alive'd connection lingers), and the
/// `Err` must propagate so the caller falls back to the serial join path.
pub(crate) fn assert_fast_join_apply_failure_closes_connection_and_propagates_err() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        // Joiner enforces a release-attestation requirement.
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        let joiner = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("joiner test node");
        configure_requirement_node(&joiner, &policy, Some(&trusted_signer))
            .await
            .expect("configure joiner policy");

        // Bootstrap peer accepts a real QUIC connection from the joiner.
        let bootstrap = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("bootstrap test node");
        bootstrap.start_accepting();
        joiner.start_accepting();

        let bootstrap_id = bootstrap.id();
        let bootstrap_addr = bootstrap.endpoint_addr_for_advertisement();
        let conn = connect_mesh(&joiner.endpoint, bootstrap_addr.clone())
            .await
            .expect("joiner connects to bootstrap");

        // Self-announcement from the bootstrap peer carrying NO release
        // attestation. `apply_announced_peer` hits the `peer_id == remote`
        // branch, `validate_direct_peer_requirements` rejects it, and
        // `apply_gossip_announcements` returns `Err`.
        let mut self_ann = requirement_peer_announcement(0x00, &policy, None, None);
        self_ann.addr = super::super::EndpointAddr {
            id: bootstrap_id,
            addrs: Default::default(),
        };
        let announcements = vec![(self_ann.addr.clone(), self_ann.clone())];

        let success = super::super::gossip::JoinProbeSuccess::new_for_tests(
            joiner.invite_token().await,
            None,
            super::super::EndpointAddr {
                id: bootstrap_id,
                addrs: Default::default(),
            },
            conn.clone(),
            announcements,
            42,
        );

        let result = joiner.commit_join_probe_success(success).await;
        assert!(
            result.is_err(),
            "apply failure must propagate Err so the caller falls back to serial join"
        );

        // The tracked entry must be gone.
        assert!(
            !joiner
                .state
                .lock()
                .await
                .connections
                .contains_key(&bootstrap_id),
            "failed candidate must be removed from tracked connections"
        );

        // The QUIC connection must be closed, not merely untracked. If it were
        // only untracked, `closed()` would hang here because the keep-alive
        // would hold the orphaned connection open.
        let closed = tokio::time::timeout(std::time::Duration::from_secs(2), conn.closed()).await;
        assert!(
            closed.is_ok(),
            "QUIC connection must be closed on apply failure, not left orphaned"
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let mut direct_proof = direct_proof_for_announcement(
            0x89,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        direct_proof.signature[0] ^= 0x01;
        let ann = requirement_peer_announcement(0x89, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_stale_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement_at(
            0x88,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
            current_time_unix_ms() - crate::DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS - 1,
        );
        let ann = requirement_peer_announcement(0x88, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofStale
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof()
 {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement(
            0x87,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        let ann = requirement_peer_announcement(0x86, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofSenderIdMismatch
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_accepts_valid_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement(
            0x85,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        let ann = requirement_peer_announcement(0x85, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(is_peer_admitted(
            &node.state.lock().await.peers.clone(),
            &peer_id
        ));
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_untrusted_release_signer() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x91);
        let ann = super::super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::super::NodeRole::Worker,
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
            release_attestation: Some(test_release_attestation_with_seed(10)),
            direct_admission_proof: Some(direct_proof_for_announcement(
                0x91,
                &policy.policy_derived_mesh_id().expect("mesh id"),
                &policy.canonical_hash_hex().expect("policy hash"),
                Some(&test_release_attestation_with_seed(10)),
            )),
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
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "add_peer must reject untrusted release signers before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::ReleaseSignerUntrusted
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_invalid_release_attestation_signature() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x90);
        let mut invalid_attestation = test_release_attestation_with_seed(9);
        invalid_attestation.signature[0] ^= 0x01;
        let invalid_direct_proof = direct_proof_for_announcement(
            0x90,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&invalid_attestation),
        );
        let ann = super::super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::super::NodeRole::Worker,
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
            release_attestation: Some(invalid_attestation),
            direct_admission_proof: Some(invalid_direct_proof),
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
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "add_peer must reject cryptographically invalid release attestations before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_wrong_mesh_id() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x92);
        let ann = super::super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::super::NodeRole::Worker,
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
            mesh_id: Some("mesh-wrong".to_string()),
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
            release_attestation: Some(test_release_attestation(&test_release_signer_key_id(9))),
            direct_admission_proof: Some(direct_proof_for_announcement(
                0x92,
                "mesh-wrong",
                &policy.canonical_hash_hex().expect("policy hash"),
                Some(&test_release_attestation(&test_release_signer_key_id(9))),
            )),
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
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "direct peers advertising the wrong mesh must be rejected before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::MeshPolicyMismatch
        );
    });
}

pub(crate) fn assert_mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let bridge = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("bridge node");
        let client = make_test_node(super::super::NodeRole::Client)
            .await
            .expect("client node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        host.set_hosted_models(vec!["remote-coding-model".to_string()])
            .await;
        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&bridge, &policy, Some(&trusted_signer))
            .await
            .expect("configure bridge policy");
        configure_requirement_node(&client, &policy, Some(&trusted_signer))
            .await
            .expect("configure client policy");

        host.start_accepting();
        bridge.start_accepting();
        client.start_accepting();

        bridge.sync_from_peer_for_tests(&host).await;
        assert!(bridge.peers().await.iter().any(|peer| peer.id == host.id()));

        client.sync_from_peer_for_tests(&bridge).await;
        assert!(
            client
                .peers()
                .await
                .iter()
                .any(|peer| peer.id == bridge.id())
        );

        let peers = client.state.lock().await.peers.clone();
        assert!(
            peers.contains_key(&host.id()),
            "host should still be tracked as a hint"
        );
        assert!(
            !is_peer_admitted(&peers, &host.id()),
            "transitive gossip must not admit the host without a direct proof path"
        );
        assert!(
            !client
                .hosts_for_model("remote-coding-model")
                .await
                .contains(&host.id()),
            "transitive-only host must not be routable before direct verification"
        );

        let _conn = client
            .connection_to_peer(host.id())
            .await
            .expect("direct connection should promote the host");
        wait_for_peer(&client, host.id()).await;
        assert!(
            client
                .hosts_for_model("remote-coding-model")
                .await
                .contains(&host.id()),
            "host should become routable only after direct verification"
        );
    });
}

pub(crate) fn assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect_async());
}

pub(crate) fn assert_mesh_requirements_gossip_rejects_direct_sender_before_payload_effects() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let original_mesh_id = node.mesh_id().await;
        let transitive = requirement_peer_announcement(
            0x83,
            &policy,
            Some(test_release_attestation(&trusted_signer)),
            Some(direct_proof_for_announcement(
                0x83,
                &policy.policy_derived_mesh_id().expect("mesh id"),
                &policy.canonical_hash_hex().expect("policy hash"),
                Some(&test_release_attestation(&trusted_signer)),
            )),
        );
        let transitive_id = transitive.addr.id;
        let mut direct = requirement_peer_announcement(0x84, &policy, None, None);
        direct.model_demand.insert(
            "untrusted-demand".to_string(),
            ModelDemand {
                last_active: now_secs(),
                request_count: 1,
            },
        );
        let direct_id = direct.addr.id;

        let result = node
            .apply_announced_peers(
                direct_id,
                &[
                    (transitive.addr.clone(), transitive),
                    (direct.addr.clone(), direct),
                ],
                Some(7),
                Some(NODE_PROTOCOL_GENERATION),
                false,
            )
            .await;

        assert!(result.is_err(), "direct sender without proof must reject");
        assert_eq!(node.mesh_id().await, original_mesh_id);
        assert!(node.get_demand().is_empty());
        let state = node.state.lock().await;
        assert!(
            !state.peers.contains_key(&direct_id),
            "rejected direct sender must not be admitted"
        );
        assert!(
            !state.peers.contains_key(&transitive_id),
            "transitive payload entries must not be adopted before direct validation"
        );
        assert!(state.requirement_rejected_peers.contains(&direct_id));
    });
}

async fn assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect_async() {
    let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
        .await
        .expect("host node");
    let trusted_signer = test_release_signer_key_id(9);
    let policy = requirement_policy(&trusted_signer);

    configure_requirement_node(&host, &policy, Some(&trusted_signer))
        .await
        .expect("configure host policy");

    let release_attestation = test_release_attestation(&trusted_signer);
    let bridge_proof = direct_proof_for_announcement(
        0x81,
        &policy.policy_derived_mesh_id().expect("mesh id"),
        &policy.canonical_hash_hex().expect("policy hash"),
        Some(&release_attestation),
    );
    let bridge_ann =
        requirement_peer_announcement(0x81, &policy, Some(release_attestation), Some(bridge_proof));
    let bridge_id = bridge_ann.addr.id;
    host.add_peer(
        bridge_id,
        bridge_ann.addr.clone(),
        &bridge_ann,
        Some(NODE_PROTOCOL_GENERATION),
    )
    .await;

    let rejected_ann = requirement_peer_announcement(0x82, &policy, None, None);
    let rejected_id = rejected_ann.addr.id;
    host.add_peer(
        rejected_id,
        rejected_ann.addr.clone(),
        &rejected_ann,
        Some(NODE_PROTOCOL_GENERATION),
    )
    .await;

    let admitted_ids: Vec<_> = host.peers().await.into_iter().map(|peer| peer.id).collect();
    assert_eq!(admitted_ids, vec![bridge_id]);
    assert!(
        admitted_ids
            .into_iter()
            .all(|peer_id| peer_id != rejected_id),
        "rejected peer messages must not change mesh membership"
    );
}

pub(crate) fn assert_mesh_requirements_join_rejects_invalid_bootstrap_token() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let owner = crate::crypto::OwnerKeypair::generate();
        let policy = crate::MeshGenesisPolicy::new(
            owner.owner_id(),
            1_717_171_717_000,
            requirement_policy(&test_release_signer_key_id(9)).requirements,
        )
        .expect("policy should validate");
        let signed_policy =
            crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner).expect("signed policy");
        let addr_bytes = serde_json::to_vec(&host.endpoint_addr_for_advertisement())
            .expect("serializable endpoint addr");

        host.start_accepting();
        joiner.start_accepting();

        let mut token = crate::SignedBootstrapToken::sign(
            vec![addr_bytes],
            &signed_policy,
            Some(current_time_unix_ms() + 60_000),
            &owner,
        )
        .expect("bootstrap token should sign");
        token.signature[0] ^= 0x01;
        let tampered = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            serde_json::to_vec(&token).expect("serializable token"),
        );

        let err = joiner
            .join(&tampered)
            .await
            .expect_err("tampered bootstrap tokens must be rejected");
        assert!(err.to_string().contains("bootstrap_token_invalid"));
        assert!(joiner.peers().await.is_empty());
        let recent = joiner.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BootstrapTokenInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_join_accepts_matching_bootstrap_before_policy_state_installed()
 {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        let policy_hash = policy.canonical_hash_hex().expect("policy hash");
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node_with_requirements(
            super::super::NodeRole::Worker,
            policy.requirements.clone(),
        )
        .await
        .expect("joiner node");

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        *joiner.release_attestation.lock().await = Some(test_release_attestation(&trusted_signer));

        assert!(
            joiner.active_mesh_policy_state().await.is_none(),
            "fresh constrained joiner must not have active policy state before joining"
        );
        host.start_accepting();
        joiner.start_accepting();

        let token = match parse_invite_token(&host.invite_token().await)
            .expect("matching bootstrap token should parse")
        {
            InviteTokenMaterial::Signed(token) => token,
            InviteTokenMaterial::Legacy(_) => panic!("requirements invite should be signed"),
        };
        joiner
            .validate_bootstrap_token(&token)
            .await
            .expect("matching bootstrap token should validate");
        joiner
            .install_requirement_aware_mesh_state(
                token.mesh_id.clone(),
                token.policy_hash.clone(),
                token.genesis_policy.clone(),
                None,
                Some(*token),
            )
            .await
            .expect("matching bootstrap token should install policy");
        joiner.sync_from_peer_for_tests(&host).await;
        host.sync_from_peer_for_tests(&joiner).await;

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
        let active = joiner
            .active_mesh_policy_state()
            .await
            .expect("join should install active policy state");
        assert_eq!(active.policy_hash, policy_hash);
        assert_eq!(
            *joiner.mesh_policy_hash.lock().await,
            None,
            "requirement-aware joins publish policy hash through the coherent snapshot, not the legacy lock"
        );
    });
}

pub(crate) fn assert_mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("joiner node");

        host.start_accepting();
        joiner.start_accepting();
        assert!(
            matches!(
                parse_invite_token(&host.invite_token().await)
                    .expect("legacy unrestricted invite should parse"),
                InviteTokenMaterial::Legacy(_)
            ),
            "unrestricted legacy meshes should still emit legacy join-compatible invites"
        );
        joiner.sync_from_peer_for_tests(&host).await;
        host.sync_from_peer_for_tests(&joiner).await;

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
    });
}

pub(crate) fn assert_named_mesh_id_uses_documented_sha256_derivation() {
    let mesh_id =
        crate::mesh::identity_persistence::generate_mesh_id(Some("alpha"), Some("nostr-pubkey"))
            .expect("named mesh id should derive");

    assert_eq!(
        mesh_id,
        "b2c089186a5c8a81d3fad5a6e8b3f5d90d77eebcaebbcb6528e8aa15f7238572"
    );
}

struct HomeGuard(Option<std::ffi::OsString>);

impl HomeGuard {
    fn set(path: &std::path::Path) -> Self {
        let previous = std::env::var_os("HOME");
        // SAFETY: requirement tests using this guard run serially, and Drop restores HOME.
        unsafe { std::env::set_var("HOME", path) };
        Self(previous)
    }
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => {
                // SAFETY: this guard restores the process environment key it exclusively changed.
                unsafe { std::env::set_var("HOME", value) }
            }
            None => {
                // SAFETY: this guard restores the process environment key it exclusively changed.
                unsafe { std::env::remove_var("HOME") }
            }
        }
    }
}

pub(crate) fn assert_persisted_random_mesh_id_is_preserved() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());

    let persisted = "legacy-random-mesh-id";
    let mesh_dir = temp.path().join(".mesh-llm");
    std::fs::create_dir_all(&mesh_dir).expect("mesh dir");
    std::fs::write(mesh_dir.join("mesh-id"), persisted).expect("persist mesh id");

    let mesh_id = crate::mesh::identity_persistence::generate_mesh_id(None, None)
        .expect("persisted id should load");

    assert_eq!(mesh_id, persisted);
}

pub(crate) fn assert_requirement_invite_signing_failure_uses_cached_fallback() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        let owner = requirement_policy_owner();
        let valid_policy = requirement_policy_without_release_attestation();
        let signed_policy = crate::SignedMeshGenesisPolicy::sign(valid_policy.clone(), &owner)
            .expect("signed policy");
        let cached_token = crate::SignedBootstrapToken::sign(
            vec![
                serde_json::to_vec(&node.endpoint_addr_for_advertisement())
                    .expect("serializable addr"),
            ],
            &signed_policy,
            Some(current_time_unix_ms() + SIGNED_BOOTSTRAP_TOKEN_LIFETIME_MS),
            &owner,
        )
        .expect("cached token");
        let mut invalid_policy = valid_policy.clone();
        invalid_policy.version = u32::MAX;

        let invite = node
            .requirement_aware_invite_token(
                &node.endpoint_addr_for_advertisement(),
                valid_policy.policy_derived_mesh_id().expect("mesh id"),
                valid_policy.canonical_hash_hex().expect("policy hash"),
                invalid_policy,
                Some(signed_policy),
                Some(cached_token.clone()),
            )
            .await;

        assert_eq!(
            invite,
            super::super::encode_signed_bootstrap_token(&cached_token)
        );
    });
}

pub(crate) fn assert_malformed_genesis_policy_fails_closed() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let mesh_dir = temp.path().join(".mesh-llm");
    std::fs::create_dir_all(&mesh_dir).expect("mesh dir");
    std::fs::write(mesh_dir.join("mesh-genesis-policy.json"), b"not-json")
        .expect("write malformed policy");
    let node = node_with_requirement_owner(requirement_policy_without_release_attestation());

    let error = node
        .load_or_create_signed_genesis_policy()
        .expect_err("malformed policy must fail closed");

    assert!(error.to_string().contains("parse"));
}

pub(crate) fn assert_unverified_genesis_policy_fails_closed() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let policy = requirement_policy_without_release_attestation();
    let mut signed =
        crate::SignedMeshGenesisPolicy::sign(policy.clone(), &requirement_policy_owner())
            .expect("signed policy");
    signed.signature[0] ^= 0x01;
    write_genesis_policy(temp.path(), &signed);
    let node = node_with_requirement_owner(policy);

    let error = node
        .load_or_create_signed_genesis_policy()
        .expect_err("unverified policy must fail closed");

    assert!(error.to_string().contains("verify"));
}

pub(crate) fn assert_mismatched_genesis_policy_fails_closed() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let persisted_policy = requirement_policy_without_release_attestation();
    let signed =
        crate::SignedMeshGenesisPolicy::sign(persisted_policy, &requirement_policy_owner())
            .expect("signed policy");
    write_genesis_policy(temp.path(), &signed);
    let node = node_with_requirement_owner(requirement_policy(&test_release_signer_key_id(9)));

    let error = node
        .load_or_create_signed_genesis_policy()
        .expect_err("mismatched policy must fail closed");

    assert!(error.to_string().contains("does not match"));
}

pub(crate) fn assert_requirement_state_reads_coherent_snapshot() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let mut node = make_test_node(super::super::NodeRole::Worker)
            .await
            .expect("test node");
        node.owner_keypair = Some(requirement_policy_owner());
        let policy = requirement_policy_without_release_attestation();
        install_requirement_policy(&node, &policy)
            .await
            .expect("install policy");
        *node.mesh_id.lock().await = Some("stale-mesh".to_string());
        *node.mesh_policy_hash.lock().await = Some("stale-policy".to_string());
        *node.signed_genesis_policy.lock().await = None;
        *node.bootstrap_token.lock().await = None;

        let active = node
            .active_mesh_policy_state()
            .await
            .expect("active policy state");
        let expected_mesh_id = active.mesh_id.clone();
        let expected_policy_hash = active.policy_hash.clone();
        let invite = match parse_invite_token(&node.invite_token().await)
            .expect("requirement-aware invite should parse")
        {
            InviteTokenMaterial::Signed(token) => token,
            InviteTokenMaterial::Legacy(_) => panic!("requirements invite should be signed"),
        };
        let local = node
            .collect_announcements()
            .await
            .into_iter()
            .find(|announcement| announcement.addr.id == node.id())
            .expect("local announcement");
        let routing_table = node.routing_table().await;

        assert_eq!(
            active.mesh_id,
            policy.policy_derived_mesh_id().expect("mesh id")
        );
        assert_eq!(node.mesh_id().await, Some(expected_mesh_id.clone()));
        assert_eq!(local.mesh_id, Some(expected_mesh_id.clone()));
        assert_eq!(local.mesh_policy_hash, Some(expected_policy_hash.clone()));
        assert!(local.genesis_policy.is_some());
        assert_eq!(invite.policy_hash, expected_policy_hash);
        assert_eq!(invite.mesh_id, expected_mesh_id);
        assert_eq!(routing_table.mesh_id, Some(invite.mesh_id));
    });
}

pub(crate) fn assert_concurrent_requirement_state_installs_do_not_overwrite() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        // Given two different requirement-aware mesh states and one fresh node.
        let node = std::sync::Arc::new(
            make_test_node(super::super::NodeRole::Worker)
                .await
                .expect("test node"),
        );
        let first_policy = requirement_policy_without_release_attestation();
        let second_policy = requirement_policy(&test_release_signer_key_id(9));
        let first_mesh_id = first_policy
            .policy_derived_mesh_id()
            .expect("first mesh id");
        let second_mesh_id = second_policy
            .policy_derived_mesh_id()
            .expect("second mesh id");
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(3));

        // When both installs begin from the same empty state.
        let first = {
            let node = node.clone();
            let barrier = barrier.clone();
            let mesh_id = first_mesh_id.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                node.install_requirement_aware_mesh_state(
                    mesh_id,
                    first_policy.canonical_hash_hex().expect("first hash"),
                    first_policy,
                    None,
                    None,
                )
                .await
            })
        };
        let second = {
            let node = node.clone();
            let barrier = barrier.clone();
            let mesh_id = second_mesh_id.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                node.install_requirement_aware_mesh_state(
                    mesh_id,
                    second_policy.canonical_hash_hex().expect("second hash"),
                    second_policy,
                    None,
                    None,
                )
                .await
            })
        };
        barrier.wait().await;
        let first_result = first.await.expect("first install task");
        let second_result = second.await.expect("second install task");

        // Then exactly one install wins and the losing state cannot overwrite it.
        assert_ne!(first_result.is_ok(), second_result.is_ok());
        let installed = node
            .active_mesh_policy_state()
            .await
            .expect("one requirement state should be installed");
        let expected_mesh_id = if first_result.is_ok() {
            first_mesh_id
        } else {
            second_mesh_id
        };
        assert_eq!(installed.mesh_id, expected_mesh_id);
    });
}

fn node_with_requirement_owner(policy: crate::MeshGenesisPolicy) -> super::super::Node {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut node = runtime
        .block_on(make_test_node(super::super::NodeRole::Worker))
        .expect("test node");
    node.owner_keypair = Some(requirement_policy_owner());
    node.local_mesh_requirements = policy.requirements;
    node
}

fn write_genesis_policy(home: &std::path::Path, signed: &crate::SignedMeshGenesisPolicy) {
    let mesh_dir = home.join(".mesh-llm");
    std::fs::create_dir_all(&mesh_dir).expect("mesh dir");
    std::fs::write(
        mesh_dir.join("mesh-genesis-policy.json"),
        serde_json::to_vec_pretty(signed).expect("serialize policy"),
    )
    .expect("write policy");
}
