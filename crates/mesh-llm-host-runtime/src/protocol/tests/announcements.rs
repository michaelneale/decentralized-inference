use super::*;

#[test]
fn owner_fields_roundtrip_through_proto_announcement() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xAB; 32]).public());
    let ann = super::super::PeerAnnouncement {
        addr: iroh::EndpointAddr {
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
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        mesh_policy_hash: None,
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
        owner_attestation: Some(crate::crypto::SignedNodeOwnership {
            claim: crate::crypto::NodeOwnershipClaim {
                version: 1,
                cert_id: "cert-123".to_string(),
                owner_id: "owner-abc".to_string(),
                owner_sign_public_key: "11".repeat(32),
                node_endpoint_id: "22".repeat(32),
                issued_at_unix_ms: 10,
                expires_at_unix_ms: 20,
                node_label: Some("studio".to_string()),
                hostname_hint: Some("worker-01".to_string()),
            },
            signature: "33".repeat(64),
        }),
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
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
    };
    let proto_pa = local_ann_to_proto_ann(&ann);
    let skippy = proto_pa
        .subprotocols
        .iter()
        .find(|subprotocol| subprotocol.name == skippy_protocol::STAGE_SUBPROTOCOL_NAME)
        .expect("skippy-stage subprotocol should be advertised");
    assert_eq!(skippy.major, skippy_protocol::STAGE_SUBPROTOCOL_MAJOR);
    assert!(
        skippy
            .features
            .iter()
            .any(|feature| feature == skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER)
    );
    assert!(
        skippy
            .features
            .iter()
            .any(|feature| feature == skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST)
    );
    assert!(skippy.features.iter().any(|feature| feature
        == skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V7));
    assert_eq!(
        proto_pa
            .owner_attestation
            .as_ref()
            .map(|att| att.owner_id.as_str()),
        Some("owner-abc")
    );

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert!(roundtripped.artifact_transfer_supported);
    assert!(roundtripped.stage_status_list_supported);
    assert!(roundtripped.stage_protocol_generation_supported);
    let roundtripped = roundtripped
        .owner_attestation
        .expect("owner attestation must round-trip");
    assert_eq!(roundtripped.claim.owner_id, "owner-abc");
    assert_eq!(roundtripped.claim.cert_id, "cert-123");
    assert_eq!(roundtripped.claim.node_label.as_deref(), Some("studio"));
}

pub(crate) fn assert_mixed_version_peer_ignores_missing_release_attestation() {
    let proto = crate::proto::node::PeerAnnouncement {
        endpoint_id: vec![1; 32],
        role: crate::proto::node::NodeRole::Worker as i32,
        version: Some("0.66.0".into()),
        ..Default::default()
    };

    let (_addr, ann) = proto_ann_to_local(&proto).expect("announcement should decode");
    assert!(ann.release_attestation.is_none());

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xBC; 32]).public());
    let peer = crate::mesh::PeerInfo::from_announcement(
        peer_id,
        iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        &ann,
        crate::crypto::OwnershipSummary::default(),
    );
    assert_eq!(
        peer.release_attestation_summary.status,
        crate::ReleaseAttestationStatus::Missing
    );
    assert!(!peer.release_attestation_summary.verified);
}

#[test]
fn advertised_model_throughput_roundtrips_through_proto_announcement() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xAC; 32]).public());
    let expected_hints = vec![crate::network::metrics::ModelThroughputHint {
        model_name: "qwen".to_string(),
        avg_tokens_per_second_milli: 42_000,
        throughput_samples: 7,
    }];
    let salt = [0xC3; mesh_llm_routing::cache_inventory::CACHE_AFFINITY_SALT_BYTES];
    let prefix_hash = 0xfeed_beef;
    let ann = super::super::PeerAnnouncement {
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::super::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec!["qwen".to_string()],
        hosted_models: Some(vec!["qwen".to_string()]),
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        mesh_policy_hash: None,
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
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![
            expected_hints[0].clone(),
            crate::network::metrics::ModelThroughputHint {
                model_name: "ghost".to_string(),
                avg_tokens_per_second_milli: 250_000,
                throughput_samples: 99,
            },
        ],
        cache_affinity: Some(
            mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement {
                salt,
                epoch: 7,
                generated_at_unix_ms: crate::mesh::current_time_unix_ms(),
                ttl_ms: 10_000,
                entries: vec![
                    mesh_llm_routing::cache_inventory::CacheAffinityEntry {
                        model: "qwen".to_string(),
                        prefix_digest: mesh_llm_routing::cache_inventory::prefix_digest(
                            &salt,
                            "qwen",
                            prefix_hash,
                        ),
                        matched_tokens: 512,
                        suffix_prefill_tokens: 24,
                        tier: mesh_llm_routing::cache_inventory::CacheTier::L1,
                        restore_micros: 25,
                        queue_delay_micros: 50,
                        prefill_micros_per_token: 750,
                    },
                    mesh_llm_routing::cache_inventory::CacheAffinityEntry {
                        model: "ghost".to_string(),
                        prefix_digest: [0xEE; 16],
                        matched_tokens: 999,
                        suffix_prefill_tokens: 0,
                        tier: mesh_llm_routing::cache_inventory::CacheTier::L1,
                        restore_micros: 0,
                        queue_delay_micros: 0,
                        prefill_micros_per_token: 0,
                    },
                ],
            },
        ),
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
        inference_admission_state: None,
    };

    let mut proto_pa = local_ann_to_proto_ann(&ann);
    assert_eq!(proto_pa.advertised_model_throughput.len(), 1);
    assert_eq!(
        proto_pa
            .cache_affinity
            .as_ref()
            .expect("cache evidence")
            .entries
            .len(),
        1,
        "evidence for non-routable models must be removed"
    );
    assert_eq!(proto_pa.advertised_model_throughput[0].model_name, "qwen");
    assert_eq!(
        proto_pa.advertised_model_throughput[0].avg_tokens_per_second_milli,
        42_000
    );
    assert_eq!(
        proto_pa.advertised_model_throughput[0].throughput_samples,
        7
    );
    proto_pa
        .advertised_model_throughput
        .push(crate::proto::node::AdvertisedModelThroughput {
            model_name: "ghost".to_string(),
            avg_tokens_per_second_milli: 250_000,
            throughput_samples: 99,
        });

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.advertised_model_throughput, expected_hints);
    assert_eq!(
        roundtripped
            .cache_affinity
            .as_ref()
            .and_then(|advertisement| advertisement.probe(
                "qwen",
                prefix_hash,
                crate::mesh::current_time_unix_ms(),
            ))
            .map(|entry| (
                entry.matched_tokens,
                entry.restore_micros,
                entry.queue_delay_micros,
                entry.prefill_micros_per_token,
            )),
        Some((512, 25, 50, 750))
    );
}

#[test]
fn malformed_cache_affinity_is_dropped_at_the_gossip_boundary() {
    let proto = crate::proto::node::PeerAnnouncement {
        endpoint_id: SecretKey::from_bytes(&[0xAE; 32])
            .public()
            .as_bytes()
            .to_vec(),
        role: crate::proto::node::NodeRole::Host as i32,
        serving_models: vec!["qwen".to_string()],
        hosted_models: vec!["qwen".to_string()],
        hosted_models_known: Some(true),
        cache_affinity: Some(crate::proto::node::CacheAffinityAdvertisement {
            salt: vec![1; 31],
            epoch: 1,
            generated_at_unix_ms: crate::mesh::current_time_unix_ms(),
            ttl_ms: 10_000,
            entries: Vec::new(),
        }),
        ..Default::default()
    };

    let (_, announcement) = proto_ann_to_local(&proto).expect("peer announcement");
    assert!(announcement.cache_affinity.is_none());
}

#[test]
fn stale_and_far_future_cache_affinity_are_dropped_at_ingest() {
    use mesh_llm_routing::cache_inventory::CACHE_AFFINITY_MAX_FUTURE_SKEW_MS;

    let now = crate::mesh::current_time_unix_ms();
    let mut proto = crate::proto::node::PeerAnnouncement {
        endpoint_id: SecretKey::from_bytes(&[0xAF; 32])
            .public()
            .as_bytes()
            .to_vec(),
        role: crate::proto::node::NodeRole::Host as i32,
        serving_models: vec!["qwen".to_string()],
        hosted_models: vec!["qwen".to_string()],
        hosted_models_known: Some(true),
        cache_affinity: Some(crate::proto::node::CacheAffinityAdvertisement {
            salt: vec![1; 32],
            epoch: 1,
            generated_at_unix_ms: now.saturating_sub(10_001),
            ttl_ms: 10_000,
            entries: Vec::new(),
        }),
        ..Default::default()
    };

    let (_, stale) = proto_ann_to_local(&proto).expect("stale peer announcement");
    assert!(stale.cache_affinity.is_none());

    proto
        .cache_affinity
        .as_mut()
        .expect("cache advertisement")
        .generated_at_unix_ms = now
        .saturating_add(CACHE_AFFINITY_MAX_FUTURE_SKEW_MS)
        // Keep the fixture well beyond the boundary even if this test is
        // descheduled between capturing `now` and decoding the announcement.
        .saturating_add(60_000);
    let (_, future) = proto_ann_to_local(&proto).expect("future peer announcement");
    assert!(future.cache_affinity.is_none());
}

#[test]
fn inference_admission_state_roundtrips_through_proto_announcement() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xAD; 32]).public());
    let expected_state = crate::proto::node::InferenceAdmissionState::RemotePaused;
    let ann = super::super::PeerAnnouncement {
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::super::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec!["qwen".to_string()],
        hosted_models: Some(vec!["qwen".to_string()]),
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        mesh_policy_hash: None,
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
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![],
        cache_affinity: None,
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
        inference_admission_state: Some(expected_state),
    };

    let proto_pa = local_ann_to_proto_ann(&ann);
    assert_eq!(
        proto_pa.inference_admission_state,
        Some(expected_state as i32)
    );

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.inference_admission_state, Some(expected_state));
}

#[test]
fn proto_announcement_without_current_stage_generation_is_not_stage_compatible() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xCD; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Worker as i32,
        subprotocols: vec![crate::proto::node::MeshSubprotocol {
            name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
            major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            features: vec![
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string(),
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST.to_string(),
            ],
        }],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("proto announcement should decode");

    assert!(!ann.stage_protocol_generation_supported);
    assert!(ann.stage_status_list_supported);
}

#[test]
fn proto_announcement_without_required_generation_bundle_is_not_stage_compatible() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD0; 32]).public());
    for missing in [
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V7,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1,
    ] {
        let features = [
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL,
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V7,
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1,
        ]
        .into_iter()
        .filter(|feature| *feature != missing)
        .map(str::to_string)
        .collect();
        let proto_pa = crate::proto::node::PeerAnnouncement {
            endpoint_id: peer_id.as_bytes().to_vec(),
            role: crate::proto::node::NodeRole::Worker as i32,
            subprotocols: vec![crate::proto::node::MeshSubprotocol {
                name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
                major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
                features,
            }],
            ..Default::default()
        };

        let (_, ann) = proto_ann_to_local(&proto_pa).expect("proto announcement should decode");
        assert!(
            !ann.stage_protocol_generation_supported,
            "missing {missing} must reject the generation-7 bundle"
        );
    }
}

#[test]
fn partial_duplicate_stage_records_do_not_form_a_generation_bundle() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD1; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Worker as i32,
        subprotocols: vec![
            crate::proto::node::MeshSubprotocol {
                name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
                major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
                features: vec![
                    skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string(),
                    skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V7
                        .to_string(),
                ],
            },
            crate::proto::node::MeshSubprotocol {
                name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
                major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
                features: vec![
                    skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST.to_string(),
                    skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1.to_string(),
                ],
            },
        ],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("proto announcement should decode");

    assert!(!ann.stage_protocol_generation_supported);
}

#[test]
fn legacy_stage_announcement_does_not_gain_local_gguf_content_id_support() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xCF; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Worker as i32,
        subprotocols: vec![crate::proto::node::MeshSubprotocol {
            name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
            major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            features: vec![
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string(),
                // Frozen legacy wire token: the current protocol intentionally
                // exports only its V7 capability.
                "stage-generation-5".to_string(),
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST.to_string(),
            ],
        }],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("legacy announcement should decode");
    assert!(!ann.local_gguf_content_id_supported);

    let reencoded = local_ann_to_proto_ann(&ann);
    let stage = reencoded
        .subprotocols
        .iter()
        .find(|subprotocol| subprotocol.name == skippy_protocol::STAGE_SUBPROTOCOL_NAME)
        .expect("stage subprotocol should survive bridge re-encoding");
    assert!(!stage.features.iter().any(|feature| {
        feature == skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1
    }));
}

#[test]
fn local_gguf_content_id_support_roundtrips_through_announcement_bridge() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD0; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Worker as i32,
        subprotocols: vec![crate::proto::node::MeshSubprotocol {
            name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
            major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            features: vec![
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string(),
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1.to_string(),
            ],
        }],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("capability announcement should decode");
    assert!(ann.local_gguf_content_id_supported);

    let reencoded = local_ann_to_proto_ann(&ann);
    let stage = reencoded
        .subprotocols
        .iter()
        .find(|subprotocol| subprotocol.name == skippy_protocol::STAGE_SUBPROTOCOL_NAME)
        .expect("stage subprotocol should survive bridge re-encoding");
    assert_eq!(
        stage
            .features
            .iter()
            .filter(|feature| {
                feature.as_str()
                    == skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1
            })
            .count(),
        1
    );

    let (_, roundtripped) =
        proto_ann_to_local(&reencoded).expect("re-encoded capability should decode");
    assert!(roundtripped.local_gguf_content_id_supported);
}

#[test]
fn unknown_stage_feature_does_not_enable_local_gguf_content_id_support() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD1; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Worker as i32,
        subprotocols: vec![crate::proto::node::MeshSubprotocol {
            name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
            major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            features: vec![
                skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string(),
                "local-gguf-content-id-v999".to_string(),
            ],
        }],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("unknown feature should be ignored");
    assert!(!ann.local_gguf_content_id_supported);
}

#[test]
fn test_proto_round_trip_with_bandwidth_and_tflops() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xBC; 32]).public());
    let ann = super::super::PeerAnnouncement {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::super::NodeRole::Host { http_port: 3131 },
        first_joined_mesh_ts: None,
        models: vec!["Qwen".to_string()],
        vram_bytes: 48_000_000_000,
        model_source: Some("Qwen.gguf".to_string()),
        serving_models: vec!["Qwen".to_string()],
        hosted_models: Some(vec!["Qwen".to_string()]),
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()],
        version: Some("0.52.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("mesh-proto-roundtrip".to_string()),
        mesh_policy_hash: None,
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
        is_soc: Some(false),
        gpu_vram: Some("51539607552".to_string()),
        gpu_reserved_bytes: Some("1073741824".to_string()),
        memory: None,
        gpu_mem_bandwidth_gbps: Some("1948.70".to_string()),
        gpu_compute_tflops_fp32: Some("19.50".to_string()),
        gpu_compute_tflops_fp16: Some("312.00".to_string()),
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
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
    };

    let proto_pa = local_ann_to_proto_ann(&ann);
    let hardware = proto_pa
        .hardware
        .as_ref()
        .expect("hardware info must be present");
    assert_eq!(hardware.hostname.as_deref(), Some("worker-01"));
    assert_eq!(hardware.is_soc, Some(false));
    assert_eq!(hardware.gpus.len(), 1);
    assert_eq!(hardware.gpus[0].name.as_deref(), Some("NVIDIA A100"));
    assert_eq!(hardware.gpus[0].vram_bytes.as_deref(), Some("51539607552"));
    assert_eq!(
        hardware.gpus[0].reserved_bytes.as_deref(),
        Some("1073741824")
    );
    assert_eq!(
        hardware.gpus[0].mem_bandwidth_gbps.as_deref(),
        Some("1948.70")
    );
    assert_eq!(
        hardware.gpus[0].compute_tflops_fp32.as_deref(),
        Some("19.50")
    );
    assert_eq!(
        hardware.gpus[0].compute_tflops_fp16.as_deref(),
        Some("312.00")
    );

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(
        roundtripped.gpu_reserved_bytes.as_deref(),
        Some("1073741824")
    );
    assert_eq!(
        roundtripped.gpu_mem_bandwidth_gbps.as_deref(),
        Some("1948.70")
    );
    assert_eq!(
        roundtripped.gpu_compute_tflops_fp32.as_deref(),
        Some("19.50")
    );
    assert_eq!(
        roundtripped.gpu_compute_tflops_fp16.as_deref(),
        Some("312.00")
    );
    assert_eq!(
        roundtripped.explicit_model_interests,
        vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
    );
}

#[test]
fn test_proto_backward_compat_missing_tflops() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xCD; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        gpu_name: Some("NVIDIA A100".to_string()),
        gpu_vram: Some("51539607552".to_string()),
        hardware: Some(crate::proto::node::HardwareInfo {
            memory: None,
            is_soc: Some(false),
            hostname: None,
            gpus: vec![crate::proto::node::GpuInfo {
                name: Some("NVIDIA A100".to_string()),
                vram_bytes: Some("51539607552".to_string()),
                reserved_bytes: None,
                mem_bandwidth_gbps: Some("1948.70".to_string()),
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
        }),
        ..Default::default()
    };

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.gpu_reserved_bytes, None);
    assert_eq!(
        roundtripped.gpu_mem_bandwidth_gbps.as_deref(),
        Some("1948.70")
    );
    assert_eq!(roundtripped.gpu_compute_tflops_fp32, None);
    assert_eq!(roundtripped.gpu_compute_tflops_fp16, None);
}

#[test]
fn test_proto_gpu_info_preserves_legacy_fields_for_old_consumers() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xCE; 32]).public());
    let proto_pa = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        hardware: Some(crate::proto::node::HardwareInfo {
            memory: None,
            is_soc: Some(false),
            hostname: Some("worker-01".to_string()),
            gpus: vec![
                crate::proto::node::GpuInfo {
                    name: Some("NVIDIA A100".to_string()),
                    vram_bytes: Some("51539607552".to_string()),
                    reserved_bytes: Some("1073741824".to_string()),
                    mem_bandwidth_gbps: Some("1948.70".to_string()),
                    compute_tflops_fp32: Some("19.50".to_string()),
                    compute_tflops_fp16: Some("312.00".to_string()),
                },
                crate::proto::node::GpuInfo {
                    name: Some("NVIDIA A100".to_string()),
                    vram_bytes: Some("51539607552".to_string()),
                    reserved_bytes: None,
                    mem_bandwidth_gbps: Some("1948.70".to_string()),
                    compute_tflops_fp32: Some("19.50".to_string()),
                    compute_tflops_fp16: Some("312.00".to_string()),
                },
            ],
        }),
        ..Default::default()
    };

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.hostname.as_deref(), Some("worker-01"));
    assert_eq!(roundtripped.gpu_name.as_deref(), Some("2× NVIDIA A100"));
    assert_eq!(
        roundtripped.gpu_vram.as_deref(),
        Some("51539607552,51539607552")
    );
    assert_eq!(
        roundtripped.gpu_reserved_bytes.as_deref(),
        Some("1073741824,")
    );
    assert_eq!(
        roundtripped.gpu_mem_bandwidth_gbps.as_deref(),
        Some("1948.70,1948.70")
    );
    assert_eq!(
        roundtripped.gpu_compute_tflops_fp32.as_deref(),
        Some("19.50,19.50")
    );
    assert_eq!(
        roundtripped.gpu_compute_tflops_fp16.as_deref(),
        Some("312.00,312.00")
    );
    assert_eq!(roundtripped.is_soc, Some(false));
}

#[test]
fn advertised_memory_roundtrips_through_proto_announcement() {
    let memory = crate::mesh::AdvertisedMemory {
        total_bytes: 12_000_000_000,
        reserved_bytes: 500_000_000,
        platform_reserve_bytes: 0,
        configured_reserve_bytes: 2_000_000_000,
        usable_bytes: 9_500_000_000,
        system_ram_bytes: Some(32_000_000_000),
        ram_offload_bytes: 18_000_000_000,
    };
    // Start from a bare announcement with no GPU inventory, hostname or SoC
    // flag: the memory block alone must carry the hardware envelope onto the
    // wire.
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD0; 32]).public());
    let bare = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        vram_bytes: 9_500_000_000,
        ..Default::default()
    };
    let (_, mut ann) = proto_ann_to_local(&bare).expect("proto_ann_to_local must succeed");
    assert_eq!(ann.memory, None);
    ann.memory = Some(memory);

    let proto_pa = local_ann_to_proto_ann(&ann);
    let wire = proto_pa
        .hardware
        .as_ref()
        .and_then(|hardware| hardware.memory.as_ref())
        .expect("memory block must be on the wire");
    assert_eq!(wire.total_bytes, Some(12_000_000_000));
    assert_eq!(wire.reserved_bytes, Some(500_000_000));
    assert_eq!(wire.configured_reserve_bytes, Some(2_000_000_000));
    assert_eq!(wire.usable_bytes, Some(9_500_000_000));
    assert_eq!(wire.system_ram_bytes, Some(32_000_000_000));
    assert_eq!(wire.ram_offload_bytes, Some(18_000_000_000));
    assert_eq!(wire.platform_reserve_bytes, Some(0));

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.memory, Some(memory));
}

#[test]
fn platform_reserve_roundtrips_and_counts_in_the_partition() {
    // A Tegra-shaped block: 64 GB total, 6.4 GB kept back by the platform,
    // the rest usable, nothing configured by the owner.
    let memory = crate::mesh::AdvertisedMemory {
        total_bytes: 64_000_000_000,
        reserved_bytes: 0,
        platform_reserve_bytes: 6_400_000_000,
        configured_reserve_bytes: 0,
        usable_bytes: 57_600_000_000,
        system_ram_bytes: Some(64_000_000_000),
        ram_offload_bytes: 0,
    };
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD1; 32]).public());
    let bare = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        vram_bytes: 57_600_000_000,
        ..Default::default()
    };
    let (_, mut ann) = proto_ann_to_local(&bare).expect("proto_ann_to_local must succeed");
    ann.memory = Some(memory);

    let proto_pa = local_ann_to_proto_ann(&ann);
    let wire = proto_pa
        .hardware
        .as_ref()
        .and_then(|hardware| hardware.memory.as_ref())
        .expect("memory block must be on the wire");
    assert_eq!(wire.platform_reserve_bytes, Some(6_400_000_000));

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.memory, Some(memory));

    // The same block without the platform share no longer adds up and is
    // dropped, so a peer cannot quietly move that share into usable.
    let mut stripped = proto_pa.clone();
    if let Some(block) = stripped
        .hardware
        .as_mut()
        .and_then(|hardware| hardware.memory.as_mut())
    {
        block.platform_reserve_bytes = None;
    }
    let (_, dropped) = proto_ann_to_local(&stripped).expect("proto_ann_to_local must succeed");
    assert_eq!(dropped.memory, None);
}

#[test]
fn malformed_memory_blocks_are_dropped_at_ingest() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xCF; 32]).public());
    let announce = |memory: crate::proto::node::MemoryInfo| crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        vram_bytes: 12_000_000_000,
        hardware: Some(crate::proto::node::HardwareInfo {
            is_soc: Some(false),
            hostname: None,
            gpus: vec![],
            memory: Some(memory),
        }),
        ..Default::default()
    };

    let missing_total = announce(crate::proto::node::MemoryInfo {
        usable_bytes: Some(9_500_000_000),
        ..Default::default()
    });
    let (_, ann) = proto_ann_to_local(&missing_total).expect("proto_ann_to_local must succeed");
    assert_eq!(ann.memory, None, "a block without its total is dropped");

    let inverted = announce(crate::proto::node::MemoryInfo {
        total_bytes: Some(12_000_000_000),
        usable_bytes: Some(13_000_000_000),
        ..Default::default()
    });
    let (_, ann) = proto_ann_to_local(&inverted).expect("proto_ann_to_local must succeed");
    assert_eq!(ann.memory, None, "more usable than total is dropped");

    let unbalanced = announce(crate::proto::node::MemoryInfo {
        total_bytes: Some(12_000_000_000),
        reserved_bytes: Some(10_000_000_000),
        configured_reserve_bytes: Some(10_000_000_000),
        usable_bytes: Some(1_000_000_000),
        ..Default::default()
    });
    let (_, ann) = proto_ann_to_local(&unbalanced).expect("proto_ann_to_local must succeed");
    assert_eq!(
        ann.memory, None,
        "reserves and usable share that do not add up to the total are dropped"
    );

    let overflowing = announce(crate::proto::node::MemoryInfo {
        total_bytes: Some(u64::MAX),
        reserved_bytes: Some(u64::MAX),
        configured_reserve_bytes: Some(u64::MAX),
        usable_bytes: Some(u64::MAX),
        ..Default::default()
    });
    let (_, ann) = proto_ann_to_local(&overflowing).expect("proto_ann_to_local must succeed");
    assert_eq!(ann.memory, None, "a partition that overflows is dropped");

    let minimal = announce(crate::proto::node::MemoryInfo {
        total_bytes: Some(12_000_000_000),
        usable_bytes: Some(12_000_000_000),
        ..Default::default()
    });
    let (_, ann) = proto_ann_to_local(&minimal).expect("proto_ann_to_local must succeed");
    assert_eq!(
        ann.memory,
        Some(crate::mesh::AdvertisedMemory {
            total_bytes: 12_000_000_000,
            reserved_bytes: 0,
            platform_reserve_bytes: 0,
            configured_reserve_bytes: 0,
            usable_bytes: 12_000_000_000,
            system_ram_bytes: None,
            ram_offload_bytes: 0,
        }),
        "absent reserves count as zero when the partition still adds up"
    );
}

#[test]
fn memory_block_exceeding_the_placement_budget_is_dropped_at_ingest() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xD2; 32]).public());
    let announce = |vram_bytes: u64| crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        vram_bytes,
        hardware: Some(crate::proto::node::HardwareInfo {
            is_soc: Some(false),
            hostname: None,
            gpus: vec![],
            memory: Some(crate::proto::node::MemoryInfo {
                total_bytes: Some(12_000_000_000),
                usable_bytes: Some(12_000_000_000),
                ..Default::default()
            }),
        }),
        ..Default::default()
    };

    // The block explains the budget it travels with: more usable than the
    // announced budget is a contradiction, so the block is dropped and the
    // budget itself is kept.
    let (_, ann) =
        proto_ann_to_local(&announce(9_500_000_000)).expect("proto_ann_to_local must succeed");
    assert_eq!(ann.memory, None);
    assert_eq!(ann.vram_bytes, 9_500_000_000);

    let (_, ann) =
        proto_ann_to_local(&announce(12_000_000_000)).expect("proto_ann_to_local must succeed");
    assert!(
        ann.memory.is_some(),
        "usable equal to the budget is consistent"
    );
}
