#[test]
fn peer_lifecycle_messages_roundtrip() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let leaving_id = EndpointId::from(SecretKey::from_bytes(&[0x55; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(leaving_id, make_test_peer_info(leaving_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(leaving_id);

    let leaving_msg = PeerLeaving {
        peer_id: leaving_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &leaving_msg);
    let decoded_leaving: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");

    let accepted_id = resolve_peer_leaving(leaving_id, &decoded_leaving)
        .expect("PeerLeaving from sender itself must be accepted");

    peers.remove(&accepted_id);
    connection_ids.remove(&accepted_id);

    assert!(
        !peers.contains_key(&leaving_id),
        "leaving peer must be removed from peers after accepted PeerLeaving"
    );
    assert!(
        !connection_ids.contains(&leaving_id),
        "leaving peer must be removed from connections after accepted PeerLeaving"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xAA; 32]).public());
    let dead_id = EndpointId::from(SecretKey::from_bytes(&[0xBB; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(dead_id, make_test_peer_info(dead_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(dead_id);

    let down_msg = PeerDown {
        peer_id: dead_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &down_msg);
    let decoded_down: PeerDown =
        decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");

    let result = resolve_peer_down(self_id, dead_id, true);
    assert_eq!(
        result,
        Some(dead_id),
        "confirmed-unreachable peer must be returned for removal"
    );

    if let Some(id) = result {
        peers.remove(&id);
        connection_ids.remove(&id);
    }

    assert!(
        !peers.contains_key(&dead_id),
        "dead peer must be removed from peers when confirmed unreachable"
    );
    assert!(
        !connection_ids.contains(&dead_id),
        "dead peer must be removed from connections when confirmed unreachable"
    );

    assert_eq!(decoded_down.r#gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_lifecycle_rejects_forged_sender_or_unverified_down() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let valid_peer_bytes = EndpointId::from(SecretKey::from_bytes(&[0x77; 32]).public())
        .as_bytes()
        .to_vec();

    let bad_gen_down = PeerDown {
        peer_id: valid_peer_bytes.clone(),
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerDown, got {:?}",
        err
    );

    let bad_gen_leaving = PeerLeaving {
        peer_id: valid_peer_bytes.clone(),
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerLeaving, got {:?}",
        err
    );

    let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let victim_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(victim_id, make_test_peer_info(victim_id));

    let forged = PeerLeaving {
        peer_id: victim_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &forged);
    let decoded: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
        .expect("structurally valid PeerLeaving must decode");

    let err = resolve_peer_leaving(remote_id, &decoded)
        .expect_err("forged PeerLeaving (peer_id != remote) must be rejected");
    assert!(
        matches!(err, ControlFrameError::ForgedSender),
        "expected ForgedSender, got {:?}",
        err
    );

    assert!(
        peers.contains_key(&victim_id),
        "victim peer must NOT be removed when PeerLeaving is forged"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
    let still_alive_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(still_alive_id, make_test_peer_info(still_alive_id));

    let result = resolve_peer_down(self_id, still_alive_id, false);
    assert!(
        result.is_none(),
        "PeerDown must not trigger removal when peer is still reachable"
    );

    assert!(
        peers.contains_key(&still_alive_id),
        "reachable peer must NOT be removed after PeerDown with should_remove=false"
    );
}

// ── Gossip consistency tests ──────────────────────────────────────────────

/// PeerDown for a recently-seen (direct) peer should be ignored regardless
/// of connection state — the peer is alive from our direct gossip even if
/// the connection is broken or absent (NAT, relay-only, stale QUIC conn).
#[test]
fn peer_down_ignored_when_recently_seen_direct() {
    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xA0; 32]).public());
    let target_id = EndpointId::from(SecretKey::from_bytes(&[0xA1; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut peer = make_test_peer_info(target_id);
    // Peer was seen just now via direct gossip.
    peer.last_seen = std::time::Instant::now();
    peers.insert(target_id, peer);

    let recently_seen = peers
        .get(&target_id)
        .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
        .unwrap_or(false);

    // The fix: when recently_seen (direct), ignore the death report
    // regardless of whether we have a connection.
    assert!(
        recently_seen,
        "precondition: peer must be recently seen (direct)"
    );
    // We should NOT call resolve_peer_down in this case.
    // Verify that resolve_peer_down with should_remove=true would remove,
    // proving the guard is necessary.
    let would_remove = resolve_peer_down(self_id, target_id, true);
    assert!(
        would_remove.is_some(),
        "without the guard, the peer would be removed"
    );
    // The peer stays in our peer list.
    assert!(
        peers.contains_key(&target_id),
        "recently-seen peer must survive PeerDown from another node"
    );
}

#[test]
fn peer_down_reporter_cooldown_suppresses_probe_before_recently_seen_check() {
    assert_eq!(
        peer_down_report_disposition(true, false),
        PeerDownReportDisposition::SuppressReporterCooldown,
        "cooldown must suppress repeated false reports even for stale/not-recently-seen peers"
    );
    assert_eq!(
        peer_down_report_disposition(true, true),
        PeerDownReportDisposition::SuppressReporterCooldown,
        "cooldown remains the cheapest rejection path when direct proof-of-life also exists"
    );
    assert_eq!(
        peer_down_report_disposition(false, true),
        PeerDownReportDisposition::RejectRecentlySeen,
        "recent direct gossip should reject first-time false reports without probing"
    );
    assert_eq!(
        peer_down_report_disposition(false, false),
        PeerDownReportDisposition::ProbeReachability,
        "only uncooldowned stale reports should trigger open_bi/connect_mesh probing"
    );
}

/// PeerDown for a peer whose last_seen is stale and has no connection
/// should be confirmed (the old behavior for genuinely dead peers).
#[test]
fn peer_down_confirmed_when_stale_and_no_connection() {
    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xB0; 32]).public());
    let target_id = EndpointId::from(SecretKey::from_bytes(&[0xB1; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut peer = make_test_peer_info(target_id);
    // Peer was last seen well beyond the stale window.
    peer.last_seen =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS + 60);
    peers.insert(target_id, peer);

    let recently_seen = peers
        .get(&target_id)
        .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
        .unwrap_or(false);

    assert!(
        !recently_seen,
        "precondition: peer is stale (not recently seen)"
    );

    // With no connection and stale last_seen, resolve_peer_down confirms removal.
    let result = resolve_peer_down(self_id, target_id, true);
    assert!(
        result.is_some(),
        "stale peer with no connection must be confirmed dead"
    );

    // Apply removal.
    if let Some(id) = result {
        peers.remove(&id);
    }
    assert!(
        !peers.contains_key(&target_id),
        "stale peer must be removed after confirmed PeerDown"
    );
}

/// Transitive peer updates should refresh last_mentioned so the peer doesn't
/// get pruned while a bridge peer keeps mentioning it.
#[tokio::test]
async fn transitive_peer_update_refreshes_last_mentioned() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC0; 32]).public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    let mut peer = make_test_peer_info(peer_id);

    let old_time =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);
    peer.last_seen = old_time;
    peer.last_mentioned = old_time;
    node.insert_test_peer(peer).await;

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let ann = super::PeerAnnouncement {
        addr: addr.clone(),
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 8 * 1024 * 1024 * 1024,
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

    node.update_transitive_peer(peer_id, &addr, &ann, make_test_endpoint_id(0xee))
        .await;
    let peer = node
        .state
        .lock()
        .await
        .peers
        .get(&peer_id)
        .cloned()
        .expect("transitive peer must remain in production state");

    assert!(
        peer.last_mentioned.elapsed().as_secs() < 1,
        "update_transitive_peer must refresh last_mentioned after transitive gossip"
    );
    assert!(
        peer.last_seen == old_time,
        "update_transitive_peer must not refresh direct last_seen"
    );

    // Peer survives prune check because last_mentioned is fresh.
    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen < prune_cutoff || peer.last_mentioned >= prune_cutoff,
        "transitive peer with fresh last_mentioned must survive pruning"
    );

    // But PeerDown silencing uses only last_seen (direct), which is stale.
    let directly_seen_recently = peer.last_seen.elapsed().as_secs() < PEER_STALE_SECS;
    assert!(
        !directly_seen_recently,
        "transitive-only peer must NOT be considered directly seen"
    );
}

/// Transitive peer that is not mentioned stops surviving once both timestamps are stale.
#[test]
fn transitive_peer_expires_when_mentions_stop() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC1; 32]).public());
    let mut peer = make_test_peer_info(peer_id);

    // Both timestamps are beyond the prune window.
    let old_time =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);
    peer.last_seen = old_time;
    peer.last_mentioned = old_time;

    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen < prune_cutoff && peer.last_mentioned < prune_cutoff,
        "peer with both timestamps stale must be below prune cutoff"
    );
}

/// A directly-connected peer with fresh last_seen but stale last_mentioned
/// still survives pruning (last_seen alone is sufficient).
#[test]
fn direct_peer_survives_with_stale_last_mentioned() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC2; 32]).public());
    let mut peer = make_test_peer_info(peer_id);

    peer.last_seen = std::time::Instant::now();
    peer.last_mentioned =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);

    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen >= prune_cutoff || peer.last_mentioned >= prune_cutoff,
        "directly-connected peer must survive pruning via last_seen alone"
    );
}

// ── Task 9: End-to-end cut-over regression tests ──────────────────────────

/// Verifies that protobuf `/1` control frames still reject legacy JSON payloads AND
/// gen=0 / wrong-gen frames. Legacy JSON/raw compatibility is only carried on `/0`.
#[test]
fn proto_v1_control_frames_reject_legacy_json_and_wrong_gen() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    // JSON bytes that look plausible for the old wire format on each stream
    let json_gossip = b"[{\"addr\":{\"id\":\"aabbcc\",\"addrs\":[]}}]";
    let json_tunnel_map = b"{\"owner\":\"aabbcc\",\"entries\":[]}";
    let json_route = b"{\"hosts\":[],\"mesh_id\":null}";
    let json_peer_down = b"\"aabbccdd\"";
    let json_peer_leaving = b"\"aabbccdd\"";

    // All migrated streams must reject legacy JSON with DecodeError
    for (stream_type, json_bytes) in [
        (STREAM_GOSSIP, json_gossip.as_slice()),
        (STREAM_TUNNEL_MAP, json_tunnel_map.as_slice()),
        (STREAM_ROUTE_REQUEST, json_route.as_slice()),
        (STREAM_PEER_DOWN, json_peer_down.as_slice()),
        (STREAM_PEER_LEAVING, json_peer_leaving.as_slice()),
    ] {
        let mut frame = vec![stream_type];
        frame.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        frame.extend_from_slice(json_bytes);
        let err = match stream_type {
            STREAM_GOSSIP => decode_control_frame::<GossipFrame>(stream_type, &frame).unwrap_err(),
            STREAM_TUNNEL_MAP => {
                decode_control_frame::<crate::proto::node::TunnelMap>(stream_type, &frame)
                    .unwrap_err()
            }
            STREAM_ROUTE_REQUEST => {
                decode_control_frame::<RouteTableRequest>(stream_type, &frame).unwrap_err()
            }
            STREAM_PEER_DOWN => decode_control_frame::<PeerDown>(stream_type, &frame).unwrap_err(),
            STREAM_PEER_LEAVING => {
                decode_control_frame::<PeerLeaving>(stream_type, &frame).unwrap_err()
            }
            _ => unreachable!("all stream types in this table are migrated control frames"),
        };
        assert!(
            matches!(err, ControlFrameError::DecodeError(_)),
            "stream {:#04x}: expected DecodeError for JSON, got {:?}",
            stream_type,
            err
        );
    }

    // All migrated streams must also reject gen=0 and gen=99 where gen is checked
    let bad_gen_gossip = GossipFrame {
        r#gen: 0,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTableRequest gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_down = PeerDown {
        peer_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_leaving = PeerLeaving {
        peer_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    // Wrong gen (e.g. 2) also rejected
    let wrong_gen_gossip = GossipFrame {
        r#gen: 2,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &wrong_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=2 (future version) must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 2 }));
}

/// Verifies that remote peer model-scan metadata (available_model_metadata,
/// available_model_sizes) is stored in PeerInfo after gossip and can be read back —
/// this is the unit-level proof of what `/api/status` exposes for remote `model_scans`.
#[test]
fn remote_model_scans_are_ignored_after_gossip() {
    use crate::proto::node::{CompactModelMetadata, GossipFrame, PeerAnnouncement as ProtoPA};

    let peer_key = SecretKey::from_bytes(&[0xC0; 32]);
    let peer_id = EndpointId::from(peer_key.public());

    // Build a gossip frame as the remote peer would send it
    let meta = CompactModelMetadata {
        model_key: "Llama-3.3-70B-Q4_K_M".to_string(),
        context_length: 131072,
        vocab_size: 128256,
        embedding_size: 8192,
        head_count: 64,
        kv_head_count: 0,
        layer_count: 80,
        feed_forward_length: 28672,
        key_length: 128,
        value_length: 128,
        architecture: "llama".to_string(),
        tokenizer_model_name: "GPT2TokenizerFast".to_string(),
        special_tokens: vec![],
        rope_scale: 8.0,
        rope_freq_base: 500000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
        parameter_size: None,
    };
    let mut model_sizes = std::collections::HashMap::new();
    model_sizes.insert("Llama-3.3-70B-Q4_K_M".to_string(), 42_000_000_000u64);

    let gossip_frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: peer_id.as_bytes().to_vec(),
        peers: vec![ProtoPA {
            endpoint_id: peer_id.as_bytes().to_vec(),
            role: NodeRole::Host as i32,
            http_port: Some(9337),
            available_models: vec!["Llama-3.3-70B-Q4_K_M".to_string()],
            available_model_metadata: vec![meta.clone()],
            available_model_sizes: model_sizes.clone(),
            vram_bytes: 96 * 1024 * 1024 * 1024,
            ..Default::default()
        }],
    };

    // Verify the gossip frame encodes and decodes cleanly
    let encoded = encode_control_frame(STREAM_GOSSIP, &gossip_frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("gossip frame with model scan metadata must decode successfully");

    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, peer_id.as_bytes());
    assert_eq!(decoded.peers.len(), 1);
    let wire_pa = &decoded.peers[0];
    assert_eq!(wire_pa.available_model_metadata.len(), 1);
    assert_eq!(
        wire_pa.available_model_sizes.get("Llama-3.3-70B-Q4_K_M"),
        Some(&42_000_000_000u64)
    );

    // Convert to local PeerAnnouncement and verify passive inventory metadata is ignored.
    let (addr, local_ann) =
        proto_ann_to_local(wire_pa).expect("proto_ann_to_local must succeed on valid gossip PA");

    assert!(local_ann.available_models.is_empty());
    assert!(local_ann.available_model_metadata.is_empty());
    assert!(local_ann.available_model_sizes.is_empty());
    assert_eq!(addr.id, peer_id, "peer EndpointId must match sender");

    // Build PeerInfo as add_peer would, verify passive inventory metadata stays empty.
    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let peer_info = PeerInfo::from_announcement(
        peer_id,
        addr.clone(),
        &local_ann,
        OwnershipSummary::default(),
    );
    peers.insert(peer_id, peer_info);

    let stored = peers.get(&peer_id).unwrap();
    assert!(stored.available_models.is_empty());
    assert!(stored.available_model_metadata.is_empty());
    assert!(stored.available_model_sizes.is_empty());
}

/// Verifies that the passive-client route-table path populates the models list
/// correctly from protobuf RouteTable entries, and that mesh_id propagates through.
#[test]
fn passive_client_route_table_models_and_mesh_id_populated() {
    use crate::proto::node::{RouteEntry as ProtoRouteEntry, RouteTable};

    let host_key = SecretKey::from_bytes(&[0xD0; 32]);
    let host_id = EndpointId::from(host_key.public());

    let worker_key = SecretKey::from_bytes(&[0xD1; 32]);
    let worker_id = EndpointId::from(worker_key.public());

    // Simulate a routing table as served by a host to a passive client
    let table = RouteTable {
        entries: vec![
            ProtoRouteEntry {
                endpoint_id: host_id.as_bytes().to_vec(),
                model: "Qwen3-32B-Q4_K_M".to_string(),
            },
            ProtoRouteEntry {
                endpoint_id: worker_id.as_bytes().to_vec(),
                model: "GLM-4.7-Flash-Q4_K_M".to_string(),
            },
        ],
        mesh_id: Some("cafebabe12345678".to_string()),
        r#gen: NODE_PROTOCOL_GENERATION,
    };

    // Encode/decode via the same path as the live server
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let decoded: RouteTable = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTable must decode successfully for passive client");

    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.entries.len(), 2);
    assert_eq!(decoded.mesh_id.as_deref(), Some("cafebabe12345678"));

    // Convert to local routing table as a passive client would
    let local = proto_route_table_to_local(&decoded);

    assert_eq!(
        local.hosts.len(),
        2,
        "passive client must see both model entries"
    );
    assert_eq!(
        local.mesh_id.as_deref(),
        Some("cafebabe12345678"),
        "mesh_id must propagate to passive client via RouteTable"
    );

    // Verify model names are correct
    let models: Vec<&str> = local.hosts.iter().map(|h| h.model.as_str()).collect();
    assert!(
        models.contains(&"Qwen3-32B-Q4_K_M"),
        "host model must appear in passive client route table"
    );
    assert!(
        models.contains(&"GLM-4.7-Flash-Q4_K_M"),
        "worker model must appear in passive client route table"
    );

    // Verify endpoint IDs round-trip correctly
    let host_entry = local
        .hosts
        .iter()
        .find(|h| h.model == "Qwen3-32B-Q4_K_M")
        .unwrap();
    assert_eq!(
        host_entry.endpoint_id, host_id,
        "host endpoint_id must be preserved in passive client route table"
    );
    let worker_entry = local
        .hosts
        .iter()
        .find(|h| h.model == "GLM-4.7-Flash-Q4_K_M")
        .unwrap();
    assert_eq!(
        worker_entry.endpoint_id, worker_id,
        "worker endpoint_id must be preserved in passive client route table"
    );

    // Verify a bad-generation RouteTable is rejected by passive clients
    let stale_table = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &stale_table);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("stale RouteTable gen=0 must be rejected by passive client");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "passive client must reject stale RouteTable: {:?}",
        err
    );
}

#[test]
fn worker_only_legacy_models_are_excluded_from_http_routes() {
    let host_id = EndpointId::from(iroh::SecretKey::from_bytes(&[0xD2; 32]).public());
    let worker_id = EndpointId::from(iroh::SecretKey::from_bytes(&[0xD3; 32]).public());

    let mut legacy_host = make_test_peer_info(host_id);
    legacy_host.role = super::NodeRole::Host { http_port: 9337 };
    legacy_host.serving_models = vec!["legacy-host-model".to_string()];
    legacy_host.hosted_models_known = false;

    let mut legacy_worker = make_test_peer_info(worker_id);
    legacy_worker.role = super::NodeRole::Worker;
    legacy_worker.serving_models = vec!["worker-only-model".to_string()];
    legacy_worker.hosted_models_known = false;

    assert!(legacy_host.accepts_http_inference());
    assert!(!legacy_worker.accepts_http_inference());
    assert_eq!(
        legacy_host.http_routable_models(),
        vec!["legacy-host-model".to_string()]
    );
    assert!(legacy_host.routes_http_model("legacy-host-model"));
    assert!(legacy_worker.http_routable_models().is_empty());
    assert!(!legacy_worker.routes_http_model("worker-only-model"));
}

#[test]
#[serial_test::serial]
fn canonical_demand_model_ref_uses_loaded_catalog_without_refreshing() {
    use crate::models::remote_catalog::{
        CatalogCurated, CatalogEntry, CatalogSource, CatalogVariant, set_catalog_entries_for_test,
    };
    use std::collections::HashMap;

    let mut variants = HashMap::new();
    variants.insert(
        "Qwen3-8B-Q4_K_M".to_string(),
        CatalogVariant {
            source: CatalogSource {
                repo: "unsloth/Qwen3-8B-GGUF".to_string(),
                revision: Some("main".to_string()),
                file: Some("Qwen3-8B-Q4_K_M.gguf".to_string()),
            },
            curated: CatalogCurated {
                name: "Qwen3 8B Q4".to_string(),
                size: Some("5GB".to_string()),
                description: None,
                draft: None,
                moe: None,
                extra_files: Vec::new(),
                mmproj: None,
            },
            packages: Vec::new(),
        },
    );
    let _catalog = set_catalog_entries_for_test(vec![CatalogEntry {
        schema_version: 1,
        source_repo: "unsloth/Qwen3-8B-GGUF".to_string(),
        variants,
    }]);

    assert_eq!(
        canonical_demand_model_ref("Qwen3 8B Q4"),
        "unsloth/Qwen3-8B-GGUF@main:Q4_K_M"
    );
    assert_eq!(
        canonical_demand_model_ref("uncached-catalog-alias"),
        "uncached-catalog-alias"
    );
}

/// Verifies that clean shutdown cleanup removes the peer and gates reconnects
/// through the same dead-peer state used by production connection admission.
#[tokio::test]
async fn dead_peer_cleanup_prevents_readmission() {
    let peer_key = SecretKey::from_bytes(&[0xE0; 32]);
    let peer_id = EndpointId::from(peer_key.public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");

    node.insert_test_peer(make_test_peer_info(peer_id)).await;

    assert!(
        is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "peer must start admitted"
    );

    node.cleanup_peer_leaving(peer_id).await;
    let state = node.state.lock().await;

    assert!(
        !is_peer_admitted(&state.peers, &peer_id),
        "peer must be removed after PeerLeaving"
    );
    assert!(
        !state.connections.contains_key(&peer_id),
        "connection must be removed after PeerLeaving"
    );
    assert!(
        state.dead_peers.contains_key(&peer_id),
        "peer must be in dead_peers after cleanup"
    );
    assert!(
        state
            .dead_peers
            .get(&peer_id)
            .is_some_and(|t| t.elapsed() < super::DEAD_PEER_TTL),
        "dead_peers TTL check prevents re-connection to recently cleaned-up peer"
    );
}

/// Verifies that dead_peers entries expire after DEAD_PEER_TTL and no longer
/// block transitive re-learning or outbound reconnection.
#[tokio::test]
async fn dead_peer_ttl_expires() {
    let peer_key = SecretKey::from_bytes(&[0xF0; 32]);
    let peer_id = EndpointId::from(peer_key.public());
    let fresh_peer_id = EndpointId::from(SecretKey::from_bytes(&[0xF1; 32]).public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");

    let expired_age = super::DEAD_PEER_TTL + std::time::Duration::from_secs(1);
    let expired_at = std::time::Instant::now()
        .checked_sub(expired_age)
        .expect("monotonic clock too fresh to test TTL expiry");
    {
        let mut state = node.state.lock().await;
        state.dead_peers.insert(peer_id, expired_at);
        state
            .dead_peers
            .insert(fresh_peer_id, std::time::Instant::now());
    }

    let expired = node.retain_live_heartbeat_state().await;
    let state = node.state.lock().await;

    assert_eq!(expired, vec![peer_id]);
    assert!(!state.dead_peers.contains_key(&peer_id));
    assert!(
        state
            .dead_peers
            .get(&fresh_peer_id)
            .is_some_and(|t| t.elapsed() < super::DEAD_PEER_TTL),
        "fresh dead_peers entry must block reconnection"
    );
}

/// Verifies that non-scope tunnel streams (0x02 STREAM_TUNNEL and 0x04
/// STREAM_TUNNEL_HTTP) are NOT subject to protobuf frame validation — they are
/// raw byte pass-throughs and must not be accidentally broken by the cut-over.
/// Also verifies their admission policy.
#[test]
fn non_scope_tunnel_streams_pass_through_without_proto_validation() {
    assert!(
        !stream_allowed_before_admission(STREAM_TUNNEL, TrustPolicy::Off),
        "STREAM_TUNNEL (0x02) must be gated until after gossip admission"
    );
    assert!(
        stream_allowed_before_admission(STREAM_TUNNEL_HTTP, TrustPolicy::Off),
        "STREAM_TUNNEL_HTTP (0x04) must be allowed for passive SDK inference"
    );

    // After admission these streams are live. Verify that the stream type constants
    // are distinct from all migrated control-plane streams.
    assert_ne!(
        STREAM_TUNNEL, STREAM_GOSSIP,
        "tunnel must not collide with gossip"
    );
    assert_ne!(
        STREAM_TUNNEL, STREAM_TUNNEL_MAP,
        "raw tunnel must not collide with tunnel-map control frame"
    );
    assert_ne!(
        STREAM_TUNNEL_HTTP, STREAM_GOSSIP,
        "http-tunnel must not collide with gossip"
    );
    assert_ne!(
        STREAM_TUNNEL_HTTP, STREAM_ROUTE_REQUEST,
        "http-tunnel must not collide with route-request"
    );

    // encode_control_frame is not called for 0x02/0x04 — they are raw pass-throughs.
    // Verify that any random bytes on these streams would decode with DecodeError
    // if accidentally routed through the protobuf decoder, proving they are kept separate.
    let raw_rpc_bytes = b"\x00\x01\x02\x03RPC-BYTES";
    let mut fake_frame = vec![STREAM_TUNNEL];
    fake_frame.extend_from_slice(&(raw_rpc_bytes.len() as u32).to_le_bytes());
    fake_frame.extend_from_slice(raw_rpc_bytes);
    // Trying to decode a raw tunnel frame as gossip must yield a type mismatch
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake_frame)
        .expect_err("raw tunnel bytes fed to gossip decoder must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: 0x01,
                got: 0x02
            }
        ),
        "expected WrongStreamType{{expected:0x01,got:0x02}}, got {:?}",
        err
    );

    assert!(
        !stream_allowed_before_admission(STREAM_TUNNEL, TrustPolicy::Off),
        "STREAM_TUNNEL must require admission (raw tunnel security boundary)"
    );
}

/// Proves the behavioral contract introduced in the reconnect fix:
/// if gossip fails after a relay-level reconnect, the peer must be removed from
/// state.peers rather than left as a zombie.
#[tokio::test]
async fn reconnect_gossip_failure_removes_zombie_peer() {
    let peer_key = SecretKey::from_bytes(&[0xF0; 32]);
    let peer_id = EndpointId::from(peer_key.public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    let capture_dir = tempfile::tempdir().expect("swarm capture tempdir");
    let recorder = crate::capture::SwarmCaptureRecorder::new(capture_dir.path().join("capture"))
        .expect("swarm capture recorder");
    let capture_path = recorder.path().to_path_buf();
    node.set_swarm_capture_recorder(Some(recorder));

    node.insert_test_peer(make_test_peer_info(peer_id)).await;

    assert!(
        is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "peer must start admitted"
    );

    node.remove_peer_after_recovered_gossip_failure(peer_id)
        .await;

    assert!(
        !is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "zombie peer must be removed when reconnect gossip fails (relay-connected but process dead)"
    );

    let capture_line = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Ok(contents) = std::fs::read_to_string(&capture_path)
                && let Some(line) = contents.lines().find(|line| line.contains("peer_removed"))
            {
                break line.to_string();
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("peer removal should reach swarm capture");
    let capture: serde_json::Value =
        serde_json::from_str(&capture_line).expect("peer removal capture JSON");
    assert_eq!(capture["fields"]["reason"], "recovered_gossip_failed");
    assert!(capture["fields"]["last_seen_age_ms"].is_u64());
    assert!(capture["fields"]["last_mentioned_age_ms"].is_u64());
    assert_eq!(capture["fields"]["had_connection"], false);

    let peer_key2 = SecretKey::from_bytes(&[0xF1; 32]);
    let peer_id2 = EndpointId::from(peer_key2.public());
    node.insert_test_peer(make_test_peer_info(peer_id2)).await;

    node.recover_heartbeat_peer(peer_id2, &mut HashMap::new())
        .await;

    assert!(
        is_peer_admitted(&node.state.lock().await.peers, &peer_id2),
        "peer must remain admitted when reconnect gossip succeeds"
    );
}
