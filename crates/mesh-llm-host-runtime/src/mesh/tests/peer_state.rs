#[test]
fn relay_health_prefers_direct_paths_and_clears_relay_age() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(240),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );
    assert!(
        health.relay_since.is_some(),
        "relay age should start on relay path"
    );

    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Direct,
            rtt_ms: Some(18),
        },
        now,
    );
    assert!(
        health.relay_since.is_none(),
        "direct path should clear relay-only aging"
    );
}

#[test]
fn relay_health_reconnects_degraded_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
        },
        now - std::time::Duration::from_secs(30),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
            },
            false,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayRttDegraded)
    );
}

#[test]
fn relay_health_rescues_known_direct_candidate_after_short_grace() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(260),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_DIRECT_RESCUE_SECS + 1),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            true,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayOnlyTooLong)
    );
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            false,
            now,
            0,
            true,
        ),
        None,
        "relay-only peers without a direct candidate keep the longer fallback grace"
    );
}

#[test]
fn relay_health_reconnects_long_lived_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(260),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            false,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayOnlyTooLong)
    );
}

#[test]
fn relay_health_respects_cooldown_and_inflight_requests() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
        },
        now - std::time::Duration::from_secs(30),
    );
    health.last_reconnect_at =
        Some(now - std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS - 1));

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            0,
            true,
        ),
        None,
        "cooldown should suppress immediate retry"
    );

    health.last_reconnect_at = None;
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            1,
            true,
        ),
        None,
        "active requests should suppress relay refresh"
    );
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            0,
            false,
        ),
        None,
        "missing home relay should suppress churn"
    );
}

#[test]
fn relay_reconnect_controller_prioritizes_degraded_rtt_over_aged_relay() {
    let now = std::time::Instant::now();
    let degraded_peer = make_test_endpoint_id(21);
    let aged_peer = make_test_endpoint_id(22);
    let mut controller = RelayReconnectController::default();

    let initial = now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5);
    assert_eq!(
        controller.plan_reconnect(
            vec![
                RelayPeerObservation {
                    peer_id: aged_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
                RelayPeerObservation {
                    peer_id: degraded_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
            ],
            initial,
            0,
            true,
        ),
        None
    );

    assert_eq!(
        controller.plan_reconnect(
            vec![
                RelayPeerObservation {
                    peer_id: aged_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
                RelayPeerObservation {
                    peer_id: degraded_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 25),
                    },
                    has_direct_candidate: false,
                },
            ],
            now,
            0,
            true,
        ),
        Some((degraded_peer, RelayReconnectReason::RelayRttDegraded)),
        "high relay RTT should refresh before merely aged relay paths"
    );
}

#[test]
fn relay_reconnect_controller_tracks_home_relay_missing_and_restored_once() {
    let now = std::time::Instant::now();
    let mut controller = RelayReconnectController::default();

    assert_eq!(controller.observe_home_relay(true, now), None);
    assert_eq!(controller.observe_home_relay(false, now), None);
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS - 1),
        ),
        None,
        "home relay warning should wait for the grace period"
    );
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 2),
        ),
        Some(HomeRelayStatusTransition::Missing {
            missing_secs: RELAY_MISSING_GRACE_SECS + 2
        })
    );
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 10),
        ),
        None,
        "missing relay should not log on every monitor tick"
    );
    assert_eq!(
        controller.observe_home_relay(
            true,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 20),
        ),
        Some(HomeRelayStatusTransition::Restored)
    );
}

#[test]
fn relay_reconnect_controller_applies_cooldown_after_attempt_and_prunes_gone_peers() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(23);
    let other_peer = make_test_endpoint_id(24);
    let mut controller = RelayReconnectController::default();

    assert_eq!(
        controller.plan_reconnect(
            vec![RelayPeerObservation {
                peer_id: peer,
                snapshot: RelayPathSnapshot {
                    kind: SelectedPathKind::Relay,
                    rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
                },
                has_direct_candidate: false,
            }],
            now,
            0,
            true,
        ),
        Some((peer, RelayReconnectReason::RelayRttDegraded))
    );

    controller.record_reconnect_attempt(peer, RelayReconnectReason::RelayRttDegraded, now);
    assert_eq!(
        controller.plan_reconnect(
            vec![RelayPeerObservation {
                peer_id: peer,
                snapshot: RelayPathSnapshot {
                    kind: SelectedPathKind::Relay,
                    rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
                },
                has_direct_candidate: false,
            }],
            now + std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS - 1),
            0,
            true,
        ),
        None,
        "attempted reconnects should suppress immediate retry even before the next tick"
    );

    controller.plan_reconnect(
        vec![RelayPeerObservation {
            peer_id: other_peer,
            snapshot: RelayPathSnapshot {
                kind: SelectedPathKind::Direct,
                rtt_ms: Some(15),
            },
            has_direct_candidate: false,
        }],
        now + std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS + 1),
        0,
        true,
    );

    assert!(
        controller.peer_health(peer).is_none(),
        "controller should prune peers that are no longer active"
    );
}

fn peer_state_test_announcement(addr: EndpointAddr) -> super::PeerAnnouncement {
    super::PeerAnnouncement {
        addr,
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
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
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
    }
}

mod lan_join_target_tracking_tests {
    use super::*;

    #[tokio::test]
    async fn remember_join_target_updates_address_on_peer_rebind() {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let peer_id = make_test_endpoint_id(34);

        let mut first = EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        };
        first
            .addrs
            .insert(TransportAddr::Ip("192.168.1.50:47916".parse().unwrap()));
        node.remember_join_target(first).await;

        assert_eq!(
            node.join_target_lan_ipv4().await,
            vec!["192.168.1.50:47916".parse().unwrap()],
            "the first advertised LAN address should be recorded"
        );

        let mut rebound = EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        };
        rebound
            .addrs
            .insert(TransportAddr::Ip("192.168.1.50:51000".parse().unwrap()));
        node.remember_join_target(rebound).await;

        assert_eq!(
            node.join_target_lan_ipv4().await,
            vec!["192.168.1.50:51000".parse().unwrap()],
            "a rebind under the same peer id must replace the stale dial-back address"
        );
    }

    #[tokio::test]
    async fn join_target_lan_ipv4_keeps_only_lan_addresses() {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let peer_id = make_test_endpoint_id(35);
        let mut target = EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        };
        for addr in [
            "192.168.1.50:47916",
            "8.8.8.8:47916",
            "100.64.0.1:47916",
            "127.0.0.1:47916",
            "172.17.0.1:47916",
        ] {
            target
                .addrs
                .insert(TransportAddr::Ip(addr.parse().unwrap()));
        }
        node.remember_join_target(target).await;

        let lan_addrs: HashSet<_> = node
            .join_target_lan_ipv4()
            .await
            .into_iter()
            .map(|addr| addr.to_string())
            .collect();
        assert_eq!(
            lan_addrs,
            ["192.168.1.50:47916", "172.17.0.1:47916"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
    }

    #[tokio::test]
    async fn known_peer_lan_ipv4_keeps_only_lan_addresses() {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let peer_id = make_test_endpoint_id(36);
        let mut addr = EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        };
        for socket_addr in [
            "10.0.0.5:47916",
            "203.0.113.5:47916",
            "100.64.0.1:47916",
            "172.17.0.1:47916",
        ] {
            addr.addrs
                .insert(TransportAddr::Ip(socket_addr.parse().unwrap()));
        }
        let announcement = peer_state_test_announcement(addr.clone());

        node.add_peer(peer_id, addr, &announcement, Some(NODE_PROTOCOL_GENERATION))
            .await;

        let lan_addrs: HashSet<_> = node
            .known_peer_lan_ipv4()
            .await
            .into_iter()
            .map(|addr| addr.to_string())
            .collect();
        assert_eq!(
            lan_addrs,
            ["10.0.0.5:47916", "172.17.0.1:47916"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
    }

    #[tokio::test]
    async fn dial_peer_addr_clears_dead_peer_gate_before_connect() {
        let node = make_test_node(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let peer_id = make_test_endpoint_id(37);
        node.state
            .lock()
            .await
            .dead_peers
            .insert(peer_id, std::time::Instant::now());

        let _ = node
            .dial_peer_addr(EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            })
            .await;

        assert!(!node.state.lock().await.dead_peers.contains_key(&peer_id));
    }
}

#[test]
fn stale_dispatcher_cannot_remove_replacement_connection() {
    assert!(
        should_remove_connection(Some(7), 7),
        "matching stable id should remove tracked connection"
    );
    assert!(
        !should_remove_connection(Some(8), 7),
        "stale dispatcher must not remove a newer replacement connection"
    );
    assert!(
        !should_remove_connection(None, 7),
        "missing connection slot should be a no-op"
    );
}

#[test]
fn relay_only_peers_get_extra_heartbeat_grace() {
    // Relay-only peers get a higher failure threshold so transient
    // relay path-renegotiation (which can spike RTT to 10s+) doesn't
    // prematurely declare them dead and cause MoA reducer fallback.
    // See heartbeat_failure_policy_for_peer for the rationale.
    let peer = make_test_peer_info(make_test_endpoint_id(12));
    let local_descriptors = vec![];
    let local_runtime = vec![];

    let policy = heartbeat_failure_policy_for_peer(&local_descriptors, &local_runtime, &peer, true);

    assert_eq!(
        policy,
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 5,
        },
        "relay-only peers must have a noticeably higher grace than direct \
         (60s heartbeats × 5 = 5 min)"
    );
}

#[test]
fn is_relay_only_path_set_classifies_correctly() {
    use crate::mesh::heartbeat::is_relay_only_path_set;
    // Empty path set: be lenient (treat as relay-only). The connection
    // is brand-new or mid-failure; we don't want to declare the peer
    // dead prematurely.
    assert!(
        is_relay_only_path_set(std::iter::empty::<bool>()),
        "empty path set must default to relay-only (lenient)"
    );
    // All paths are non-IP (relay): relay-only.
    assert!(is_relay_only_path_set([false]));
    assert!(is_relay_only_path_set([false, false, false]));
    // Any IP path means NOT relay-only.
    assert!(!is_relay_only_path_set([true]));
    assert!(!is_relay_only_path_set([true, false]));
    assert!(!is_relay_only_path_set([false, true]));
    assert!(!is_relay_only_path_set([true, true, true]));
}

#[test]
fn classify_relay_only_defaults_to_strict_when_no_connection() {
    use crate::mesh::heartbeat::classify_relay_only_for_policy;
    // No Connection object at all (cleanly closed, QUIC idle-expired,
    // never opened): must default to STRICT, not lenient. Otherwise a
    // previously-direct peer that simply disconnected would silently
    // inherit the 5-min relay grace and keep stale model routes alive
    // an extra 3 min beyond what direct policy intends.
    assert!(
        !classify_relay_only_for_policy(None),
        "no Connection object must default to strict (not relay-only)"
    );
    // With a Connection: pass through whatever is_relay_only_connection
    // observed (i.e., classify by the connection's actual paths).
    assert!(
        classify_relay_only_for_policy(Some(true)),
        "a relay-only connection must keep its lenient classification"
    );
    assert!(
        !classify_relay_only_for_policy(Some(false)),
        "a connection with IP paths must remain strict (direct)"
    );
}

#[test]
fn direct_peers_use_strict_heartbeat_threshold() {
    let peer = make_test_peer_info(make_test_endpoint_id(13));
    let local_descriptors = vec![];
    let local_runtime = vec![];

    let policy =
        heartbeat_failure_policy_for_peer(&local_descriptors, &local_runtime, &peer, false);

    assert_eq!(
        policy.failure_threshold, 2,
        "direct paths stay at 2 misses — when the network is up at all, \
         two consecutive cycles of silence is a real failure signal"
    );
}

#[test]
fn peer_meaningfully_changed_detects_reserved_bytes_updates() {
    let peer_id = make_test_endpoint_id(12);
    let mut old_peer = make_test_peer_info(peer_id);
    let mut new_peer = old_peer.clone();

    old_peer.gpu_reserved_bytes = Some("1000".to_string());
    new_peer.gpu_reserved_bytes = Some("2000".to_string());

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[tokio::test]
async fn incoming_peer_promoted_after_valid_gossip() {
    use prost::Message as _;

    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xab; 32]).public());
    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let announcement = peer_state_test_announcement(addr);
    let frame = build_gossip_frame(&[announcement], peer_id);
    let decoded = decode_gossip_payload(ControlProtocol::ProtoV1, peer_id, &frame.encode_to_vec())
        .expect("valid gossip frame must decode through the production boundary");

    assert!(
        !is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "peer must NOT be admitted before gossip"
    );

    assert!(
        !stream_allowed_before_admission(STREAM_TUNNEL, TrustPolicy::Off),
        "raw tunnel streams must be gated until after admission"
    );
    assert!(
        stream_allowed_before_admission(STREAM_TUNNEL_HTTP, TrustPolicy::Off),
        "HTTP tunnel streams must be allowed for passive SDK clients"
    );

    assert!(
        stream_allowed_before_admission(STREAM_GOSSIP, TrustPolicy::Off),
        "STREAM_GOSSIP must always be allowed — it is the admission path"
    );

    node.apply_announced_peers(
        peer_id,
        &decoded,
        None,
        Some(NODE_PROTOCOL_GENERATION),
        false,
    )
    .await
    .expect("valid gossip must pass production admission");

    assert!(
        is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "peer must be admitted after production gossip admission completes"
    );
}

#[tokio::test]
async fn incoming_peer_rejected_on_legacy_or_malformed_gossip() {
    let malformed_payload = vec![0xFF_u8; 20];
    let mut bad_frame = vec![STREAM_GOSSIP];
    bad_frame.extend_from_slice(&(malformed_payload.len() as u32).to_le_bytes());
    bad_frame.extend_from_slice(&malformed_payload);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &bad_frame)
        .expect_err("malformed protobuf must be rejected");
    assert!(
        matches!(err, ControlFrameError::DecodeError(_)),
        "expected DecodeError for malformed payload, got {:?}",
        err
    );

    let bad_gen_frame = GossipFrame {
        r#gen: 0,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {:?}",
        err
    );

    for stream_type in [
        STREAM_TUNNEL,
        STREAM_TUNNEL_MAP,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_PLUGIN_CHANNEL,
        STREAM_PLUGIN_BULK_TRANSFER,
        STREAM_PLUGIN_MESH_STREAM,
    ] {
        assert!(
            !stream_allowed_before_admission(stream_type, TrustPolicy::Off),
            "stream {:#04x} must be quarantine-gated for unadmitted peers — if this fails, the gate is broken",
            stream_type
        );
    }

    assert!(
        stream_allowed_before_admission(STREAM_GOSSIP, TrustPolicy::Off),
        "STREAM_GOSSIP must bypass the gate (it is the admission handshake)"
    );
    assert!(
        stream_allowed_before_admission(STREAM_ROUTE_REQUEST, TrustPolicy::Off),
        "STREAM_ROUTE_REQUEST must bypass the gate (passive/client request-only path)"
    );
    assert!(
        stream_allowed_before_admission(STREAM_TUNNEL_HTTP, TrustPolicy::Off),
        "STREAM_TUNNEL_HTTP must bypass the gate (passive/client inference path)"
    );

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xcd; 32]).public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    assert!(
        !is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "a payload rejected by the production decoder must not admit a peer"
    );
}

#[tokio::test]
async fn passive_route_table_request_does_not_admit_peer() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xef; 32]).public());
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");

    assert!(
        !is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "passive caller must NOT be admitted before route request"
    );

    assert!(
        stream_allowed_before_admission(STREAM_ROUTE_REQUEST, TrustPolicy::Off),
        "STREAM_ROUTE_REQUEST must be allowed before admission (passive/client path)"
    );

    for &gated in &[
        STREAM_TUNNEL,
        STREAM_TUNNEL_MAP,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_PLUGIN_CHANNEL,
        STREAM_PLUGIN_BULK_TRANSFER,
        STREAM_PLUGIN_MESH_STREAM,
    ] {
        assert!(
            !stream_allowed_before_admission(gated, TrustPolicy::Off),
            "stream {:#04x} must remain gated after a route request — route request must not unlock other streams",
            gated
        );
    }

    let valid_req = RouteTableRequest {
        requester_id: vec![0xef_u8; 32],
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &valid_req);
    let decoded: RouteTableRequest = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTableRequest must decode successfully");
    assert_eq!(decoded.requester_id, vec![0xef_u8; 32]);
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);

    let bad_req = RouteTableRequest {
        requester_id: vec![0u8; 16],
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded_bad = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded_bad)
        .expect_err("route request with wrong-length requester_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    assert!(
        !is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "passive caller must NOT be admitted after route-table response"
    );

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let announcement = peer_state_test_announcement(addr.clone());
    node.add_peer(peer_id, addr, &announcement, Some(NODE_PROTOCOL_GENERATION))
        .await;
    assert!(
        is_peer_admitted(&node.state.lock().await.peers, &peer_id),
        "only the production gossip admission path should promote the peer"
    );
}

#[test]
fn control_frame_rejects_oversize_or_bad_generation() {
    let oversize_len = MAX_CONTROL_FRAME_BYTES + 1;
    let mut fake = vec![STREAM_GOSSIP];
    fake.extend_from_slice(&(oversize_len as u32).to_le_bytes());
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake)
        .expect_err("oversize frame must be rejected");
    assert!(
        matches!(err, ControlFrameError::OversizeFrame { .. }),
        "expected OversizeFrame, got {:?}",
        err
    );

    let bad_gen = GossipFrame {
        r#gen: 99,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("bad generation must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 99 }),
        "expected BadGeneration{{got:99}}, got {:?}",
        err
    );

    let bad_id = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 16],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_id);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("bad endpoint_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let valid = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &valid);
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("wrong stream type must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: 0x03,
                got: 0x01
            }
        ),
        "expected WrongStreamType, got {:?}",
        err
    );
}

#[test]
fn gossip_frame_roundtrip_preserves_scanned_model_metadata() {
    use crate::proto::node::{CompactModelMetadata, ExpertsSummary};

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    let peer_id_bytes = peer_id.as_bytes().to_vec();

    let meta = CompactModelMetadata {
        model_key: "Qwen3-8B-Q4_K_M".to_string(),
        context_length: 40960,
        vocab_size: 151936,
        embedding_size: 4096,
        head_count: 32,
        kv_head_count: 0,
        layer_count: 36,
        feed_forward_length: 14336,
        key_length: 128,
        value_length: 128,
        architecture: "qwen3".to_string(),
        tokenizer_model_name: "PreTrainedTokenizerFast".to_string(),
        special_tokens: vec![],
        rope_scale: 1.0,
        rope_freq_base: 1_000_000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
        parameter_size: None,
    };

    let mut model_sizes = HashMap::new();
    model_sizes.insert("Qwen3-8B-Q4_K_M".to_string(), 4_800_000_000u64);

    let experts = ExpertsSummary {
        total_experts: 64,
        expert_count_used: 8,
        top_expert_ids: vec![1, 5, 10],
    };

    let local_ann = super::PeerAnnouncement {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 8080 },
        first_joined_mesh_ts: None,
        models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        vram_bytes: 128 * 1024 * 1024 * 1024,
        model_source: Some("bartowski/Qwen3-8B-GGUF".to_string()),
        serving_models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        hosted_models: Some(vec!["Qwen3-8B-Q4_K_M".to_string()]),
        available_models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: Some("0.42.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("deadbeef12345678".to_string()),
        mesh_policy_hash: None,
        gpu_name: Some("Apple M4 Max".to_string()),
        hostname: Some("test-node".to_string()),
        is_soc: Some(true),
        gpu_vram: Some("128 GB".to_string()),
        gpu_reserved_bytes: None,
        memory: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![meta.clone()],
        experts_summary: Some(experts.clone()),
        available_model_sizes: model_sizes.clone(),
        served_model_descriptors: vec![ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: "Qwen3-8B-Q4_K_M".to_string(),
                is_primary: true,
                source_kind: ModelSourceKind::HuggingFace,
                canonical_ref: Some("hf/bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf".into()),
                repository: Some("bartowski/Qwen3-8B-GGUF".into()),
                revision: Some("main".into()),
                artifact: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                local_file_name: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                identity_hash: Some("identity-hash".into()),
            },
            capabilities_known: true,
            capabilities: crate::models::ModelCapabilities::default(),
            topology: None,
            metadata: None,
        }],
        served_model_runtime: vec![ModelRuntimeDescriptor {
            model_name: "Qwen3-8B-Q4_K_M".to_string(),
            identity_hash: Some("identity-hash".to_string()),
            context_length: Some(32768),
            ready: true,
        }],
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
        inference_admission_state: None,
    };

    let proto_pa = local_ann_to_proto_ann(&local_ann);
    assert_passive_model_metadata_stripped(&proto_pa);
    assert_descriptor_capability_provenance(&proto_pa);

    let (_, roundtripped) =
        proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed on valid proto PA");
    assert_local_gossip_restoration(&roundtripped);

    let frame = build_gossip_frame(&[local_ann], peer_id);
    assert_eq!(frame.sender_id, peer_id_bytes);
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("build_gossip_frame output must decode successfully");
    assert_eq!(decoded.peers.len(), 1);
    let wire_pa = &decoded.peers[0];
    assert_wire_gossip_preserves_model_runtime(wire_pa);
    let (_, final_local) =
        proto_ann_to_local(wire_pa).expect("final proto_ann_to_local must succeed");
    assert_local_gossip_restoration(&final_local);
}

fn assert_passive_model_metadata_stripped(proto_pa: &crate::proto::node::PeerAnnouncement) {
    assert_eq!(
        proto_pa.available_model_metadata.len(),
        0,
        "local_ann_to_proto_ann must strip passive available_model_metadata from gossip"
    );
    assert!(
        proto_pa.available_models.is_empty(),
        "local_ann_to_proto_ann must strip passive available_models from gossip"
    );
    assert_eq!(
        proto_pa.available_model_sizes.len(),
        0,
        "local_ann_to_proto_ann must strip passive available_model_sizes from gossip"
    );
    assert_eq!(
        proto_pa.experts_summary.as_ref().map(|e| e.total_experts),
        Some(64),
        "local_ann_to_proto_ann must carry experts_summary"
    );
}

fn assert_descriptor_capability_provenance(proto_pa: &crate::proto::node::PeerAnnouncement) {
    assert_eq!(
        proto_pa
            .served_model_descriptors
            .first()
            .and_then(|descriptor| descriptor.capabilities_known),
        Some(true),
        "gossip should preserve descriptor capability provenance"
    );
}

fn assert_local_gossip_restoration(roundtripped: &super::PeerAnnouncement) {
    assert_eq!(
        roundtripped.available_model_metadata.len(),
        0,
        "proto_ann_to_local must ignore passive available_model_metadata from gossip"
    );
    assert!(
        roundtripped.available_models.is_empty(),
        "proto_ann_to_local must ignore passive available_models from gossip"
    );
    assert!(roundtripped.available_model_sizes.is_empty());
    assert_eq!(
        roundtripped
            .experts_summary
            .as_ref()
            .map(|e| e.total_experts),
        Some(64),
        "proto_ann_to_local must restore experts_summary"
    );
    assert!(
        roundtripped
            .served_model_descriptors
            .first()
            .map(|descriptor| descriptor.capabilities_known)
            .unwrap_or(false),
        "proto_ann_to_local must restore descriptor capability provenance"
    );
    assert_eq!(
        roundtripped
            .served_model_runtime
            .first()
            .and_then(ModelRuntimeDescriptor::advertised_context_length),
        Some(32768),
        "proto_ann_to_local must preserve served model runtime context length"
    );
}

fn assert_wire_gossip_preserves_model_runtime(proto_pa: &crate::proto::node::PeerAnnouncement) {
    assert_eq!(
        proto_pa.available_model_metadata.len(),
        0,
        "build_gossip_frame must strip passive available_model_metadata from wire gossip"
    );
    assert!(proto_pa.available_models.is_empty());
    assert!(proto_pa.available_model_sizes.is_empty());
    assert_eq!(
        proto_pa
            .experts_summary
            .as_ref()
            .map(|e| e.top_expert_ids.as_slice()),
        Some([1u32, 5, 10].as_slice())
    );
    assert_eq!(
        proto_pa
            .served_model_runtime
            .first()
            .and_then(|runtime| runtime.context_length),
        Some(32768),
        "build_gossip_frame must preserve served model runtime context length"
    );
    assert_descriptor_capability_provenance(proto_pa);
}

#[test]
fn proto_ann_to_local_treats_missing_default_capability_provenance_as_unknown() {
    let peer_id = EndpointId::from(SecretKey::generate().public());
    let proto_pa = PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        served_model_descriptors: vec![crate::proto::node::ServedModelDescriptor {
            identity: Some(crate::proto::node::ServedModelIdentity {
                model_name: "Qwen3VL-2B-Instruct-Q4_K_M".to_string(),
                source_kind: crate::proto::node::ModelSourceKind::LocalGguf as i32,
                ..Default::default()
            }),
            capabilities: Some(crate::proto::node::ModelCapabilities::default()),
            capabilities_known: None,
            topology: None,
            metadata: None,
        }],
        ..Default::default()
    };

    let (_, ann) = proto_ann_to_local(&proto_pa).expect("valid proto announcement");
    let descriptor = ann
        .served_model_descriptors
        .first()
        .expect("descriptor should decode");
    assert!(!descriptor.capabilities_known);
}

#[test]
fn infer_remote_served_descriptors_marks_exactly_one_primary_for_duplicate_names() {
    let serving_models = vec![
        "Qwen3-8B-Q4_K_M".to_string(),
        "Qwen3-8B-Q4_K_M".to_string(),
        "Llama-3.2-3B-Q4_K_M".to_string(),
    ];

    let descriptors = infer_remote_served_descriptors(
        "Qwen3-8B-Q4_K_M",
        &serving_models,
        Some("Qwen/Qwen3-8B-GGUF@revabc/Qwen3-8B-Q4_K_M.gguf"),
    );

    assert_eq!(descriptors.len(), serving_models.len());
    assert_eq!(
        descriptors
            .iter()
            .filter(|descriptor| descriptor.identity.is_primary)
            .count(),
        1
    );
    assert!(descriptors[0].identity.is_primary);
    assert!(!descriptors[1].identity.is_primary);
}

#[test]
fn infer_remote_served_descriptors_leaves_primary_unknown_when_name_absent() {
    let serving_models = vec![
        "Qwen3-8B-Q4_K_M".to_string(),
        "Llama-3.2-3B-Q4_K_M".to_string(),
    ];

    let descriptors = infer_remote_served_descriptors(
        "Missing-Primary-Q4_K_M",
        &serving_models,
        Some("Qwen/Qwen3-8B-GGUF@revabc/Qwen3-8B-Q4_K_M.gguf"),
    );

    assert!(
        descriptors
            .iter()
            .all(|descriptor| !descriptor.identity.is_primary)
    );
    assert!(
        descriptors
            .iter()
            .all(|descriptor| descriptor.identity.source_kind == ModelSourceKind::Unknown)
    );
}

#[test]
fn public_model_id_from_identity_preserves_huggingface_revision() {
    let identity = ServedModelIdentity {
        model_name: "Qwen3-8B-Q4_K_M".to_string(),
        source_kind: ModelSourceKind::HuggingFace,
        repository: Some("Qwen/Qwen3-8B-GGUF".to_string()),
        revision: Some("revabc".to_string()),
        artifact: Some("Qwen3-8B-Q4_K_M.gguf".to_string()),
        ..Default::default()
    };

    assert_eq!(
        public_model_id_from_identity(&identity).as_deref(),
        Some("Qwen/Qwen3-8B-GGUF@revabc:Q4_K_M")
    );
}

#[test]
fn gossip_rejects_sender_id_mismatch_or_invalid_endpoint_len() {
    use prost::Message as _;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xaa; 32]).public());
    let peer_id_bytes = peer_id.as_bytes().to_vec();

    let invalid_sender_frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 16],
        peers: vec![PeerAnnouncement {
            endpoint_id: peer_id_bytes.clone(),
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &invalid_sender_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("16-byte sender_id must be rejected at decode time");
    assert!(
        matches!(err, ControlFrameError::InvalidSenderId { got: 16 }),
        "expected InvalidSenderId{{got:16}}, got {:?}",
        err
    );

    let impersonator_id = EndpointId::from(SecretKey::from_bytes(&[0xbb; 32]).public());
    let mismatch_frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: impersonator_id.as_bytes().to_vec(),
        peers: vec![PeerAnnouncement {
            endpoint_id: peer_id_bytes.clone(),
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let err = decode_gossip_payload(
        ControlProtocol::ProtoV1,
        peer_id,
        &mismatch_frame.encode_to_vec(),
    )
    .expect_err("the production gossip decoder must reject a forged sender identity");
    assert!(err.to_string().contains("sender_id mismatch"));

    let bad_endpoint_frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: peer_id_bytes.clone(),
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 20],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_endpoint_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("20-byte endpoint_id in peer must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 20 }),
        "expected InvalidEndpointId{{got:20}}, got {:?}",
        err
    );
}

#[test]
fn transitive_peer_update_refreshes_metadata_fields() {
    use crate::proto::node::CompactModelMetadata;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x10; 32]).public());
    let mut existing = make_test_peer_info(peer_id);
    existing.available_models = vec!["OldModel-Q4_K_M".to_string()];
    existing.models = vec!["OldModel-Q4_K_M".to_string()];
    existing.requested_models = vec!["OldModel-Q4_K_M".to_string()];

    let meta = CompactModelMetadata {
        model_key: "NewModel-Q4_K_M".to_string(),
        context_length: 8192,
        vocab_size: 32000,
        embedding_size: 4096,
        head_count: 32,
        kv_head_count: 0,
        layer_count: 32,
        feed_forward_length: 11008,
        key_length: 128,
        value_length: 128,
        architecture: "llama".to_string(),
        tokenizer_model_name: String::new(),
        special_tokens: vec![],
        rope_scale: 1.0,
        rope_freq_base: 10000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
        parameter_size: None,
    };

    let mut new_sizes = HashMap::new();
    new_sizes.insert("NewModel-Q4_K_M".to_string(), 4_800_000_000u64);

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let ann = super::PeerAnnouncement {
        addr: addr.clone(),
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec!["NewModel-Q4_K_M".to_string()],
        vram_bytes: 8 * 1024 * 1024 * 1024,
        model_source: Some("new-source".to_string()),
        serving_models: vec!["NewModel-Q4_K_M".to_string()],
        hosted_models: Some(vec!["NewModel-Q4_K_M".to_string()]),
        available_models: vec!["NewModel-Q4_K_M".to_string()],
        requested_models: vec!["NewModel-Q4_K_M".to_string()],
        explicit_model_interests: vec!["Org/NewModel-GGUF@main:Q4_K_M".to_string()],
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
        available_model_metadata: vec![meta],
        experts_summary: None,
        available_model_sizes: new_sizes,
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

    apply_transitive_ann(&mut existing, &addr, &ann, make_test_endpoint_id(0xee));

    assert!(
        existing.available_models.is_empty(),
        "remote available_models must be ignored during transitive gossip merge"
    );
    assert_eq!(
        existing.models,
        vec!["NewModel-Q4_K_M".to_string()],
        "models must be refreshed from transitive gossip"
    );
    assert_eq!(
        existing.requested_models,
        vec!["NewModel-Q4_K_M".to_string()],
        "requested_models must be refreshed from transitive gossip"
    );
    assert_eq!(
        existing.explicit_model_interests,
        vec!["Org/NewModel-GGUF@main:Q4_K_M".to_string()],
        "explicit_model_interests must be refreshed from transitive gossip"
    );
    assert!(existing.available_model_metadata.is_empty());
    assert!(existing.available_model_sizes.is_empty());
}

#[test]
fn transitive_peer_merge_preserves_richer_direct_address() {
    use iroh::TransportAddr;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let mut existing = make_test_peer_info(peer_id);

    let mut rich_addrs = std::collections::BTreeSet::new();
    rich_addrs.insert(TransportAddr::Ip("127.0.0.1:1000".parse().unwrap()));
    rich_addrs.insert(TransportAddr::Ip("192.168.1.1:1001".parse().unwrap()));
    rich_addrs.insert(TransportAddr::Ip("10.0.0.1:1002".parse().unwrap()));
    existing.addr = EndpointAddr {
        id: peer_id,
        addrs: rich_addrs,
    };

    let mut weak_addrs = std::collections::BTreeSet::new();
    weak_addrs.insert(TransportAddr::Ip("127.0.0.1:9999".parse().unwrap()));
    let weak_addr = EndpointAddr {
        id: peer_id,
        addrs: weak_addrs,
    };
    let ann = super::PeerAnnouncement {
        addr: weak_addr.clone(),
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 4 * 1024 * 1024 * 1024,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec!["SomeModel-Q4_K_M".to_string()],
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

    apply_transitive_ann(&mut existing, &weak_addr, &ann, make_test_endpoint_id(0xee));

    assert_eq!(
        existing.addr.addrs.len(),
        3,
        "rich direct address (3 paths) must not be overwritten by weaker transitive addr (1 path)"
    );
    assert!(
        existing.available_models.is_empty(),
        "remote available_models must still be ignored even when addr is preserved"
    );

    let mut richer_addrs = std::collections::BTreeSet::new();
    richer_addrs.insert(TransportAddr::Ip("127.0.0.1:1000".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("192.168.1.1:1001".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("10.0.0.1:1002".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("172.16.0.1:1003".parse().unwrap()));
    let richer_addr = EndpointAddr {
        id: peer_id,
        addrs: richer_addrs,
    };
    let ann2 = super::PeerAnnouncement {
        addr: richer_addr.clone(),
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 4 * 1024 * 1024 * 1024,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec!["SomeModel-Q4_K_M".to_string()],
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
    apply_transitive_ann(
        &mut existing,
        &richer_addr,
        &ann2,
        make_test_endpoint_id(0xee),
    );

    assert_eq!(
        existing.addr.addrs.len(),
        4,
        "richer transitive addr (4 paths) must replace existing (3 paths)"
    );
}

#[test]
fn tunnel_map_roundtrip_updates_remote_map() {
    use crate::proto::node::{TunnelEntry, TunnelMap};

    let owner_key = SecretKey::from_bytes(&[0x10; 32]);
    let owner_id = EndpointId::from(owner_key.public());
    let owner_bytes = owner_id.as_bytes().to_vec();

    let target_key = SecretKey::from_bytes(&[0x20; 32]);
    let target_id = EndpointId::from(target_key.public());
    let target_bytes = target_id.as_bytes().to_vec();

    let frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };

    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &frame);
    let decoded: TunnelMap = decode_control_frame(STREAM_TUNNEL_MAP, &encoded)
        .expect("valid TunnelMap must decode successfully");

    assert_eq!(decoded.owner_peer_id, owner_bytes);
    assert_eq!(decoded.entries.len(), 1);
    assert_eq!(decoded.entries[0].target_peer_id, target_bytes);
    assert_eq!(decoded.entries[0].tunnel_port, 50001);

    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    ingest_tunnel_map(owner_id, &decoded, &mut remote_tunnel_maps)
        .expect("valid tunnel map must ingest successfully");

    assert_eq!(remote_tunnel_maps.len(), 1);
    let inner = remote_tunnel_maps
        .get(&owner_id)
        .expect("owner must be present in remote_tunnel_maps");
    assert_eq!(inner.len(), 1);
    let port = inner
        .get(&target_id)
        .expect("target must be present in inner map");
    assert_eq!(*port, 50001u16);
}

#[test]
fn tunnel_map_rejects_owner_mismatch_or_bad_target_id() {
    use crate::proto::node::{TunnelEntry, TunnelMap};

    let owner_key = SecretKey::from_bytes(&[0x30; 32]);
    let owner_id = EndpointId::from(owner_key.public());
    let owner_bytes = owner_id.as_bytes().to_vec();

    let target_key = SecretKey::from_bytes(&[0x40; 32]);
    let target_id = EndpointId::from(target_key.public());
    let target_bytes = target_id.as_bytes().to_vec();

    let bad_owner_frame = TunnelMap {
        owner_peer_id: vec![0u8; 16],
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &bad_owner_frame);
    let err = decode_control_frame::<TunnelMap>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("bad owner_peer_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let bad_target_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: vec![0u8; 16],
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &bad_target_frame);
    let err = decode_control_frame::<TunnelMap>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("bad target_peer_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let different_key = SecretKey::from_bytes(&[0x50; 32]);
    let different_id = EndpointId::from(different_key.public());

    let mismatched_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    let result = ingest_tunnel_map(different_id, &mismatched_frame, &mut remote_tunnel_maps);
    assert!(result.is_err(), "owner mismatch must be rejected");
    assert!(
        remote_tunnel_maps.is_empty(),
        "mismatched owner must not populate remote_tunnel_maps"
    );

    let oversized_port_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 70000,
            relay_peer_id: None,
        }],
    };
    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    let result = ingest_tunnel_map(owner_id, &oversized_port_frame, &mut remote_tunnel_maps);
    assert!(result.is_err(), "tunnel_port > u16::MAX must be rejected");
    assert!(
        remote_tunnel_maps.is_empty(),
        "oversized tunnel_port must not populate remote_tunnel_maps"
    );
}

#[test]
fn route_table_request_roundtrip() {
    use crate::proto::node::{RouteEntry as ProtoRouteEntry, RouteTable};

    let peer_key = SecretKey::from_bytes(&[0x60; 32]);
    let peer_id = EndpointId::from(peer_key.public());
    let peer_bytes = peer_id.as_bytes().to_vec();

    let req = RouteTableRequest {
        requester_id: peer_bytes.clone(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &req);
    let decoded: RouteTableRequest = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTableRequest must decode successfully");
    assert_eq!(decoded.requester_id, peer_bytes);
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);

    let table = RouteTable {
        entries: vec![ProtoRouteEntry {
            endpoint_id: peer_bytes.clone(),
            model: "Qwen3-8B-Q4_K_M".to_string(),
        }],
        mesh_id: Some("test-mesh-0102030405060708".to_string()),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded_table = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let decoded_table: RouteTable = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded_table)
        .expect("valid RouteTable must decode successfully");
    assert_eq!(decoded_table.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded_table.entries.len(), 1);
    assert_eq!(decoded_table.entries[0].endpoint_id, peer_bytes);
    assert_eq!(decoded_table.entries[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(
        decoded_table.mesh_id.as_deref(),
        Some("test-mesh-0102030405060708")
    );

    let local = proto_route_table_to_local(&decoded_table);
    assert_eq!(local.hosts.len(), 1);
    assert_eq!(local.hosts[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(local.hosts[0].endpoint_id, peer_id);
    assert_eq!(local.mesh_id.as_deref(), Some("test-mesh-0102030405060708"));

    let round_tripped = routing_table_to_proto(&local);
    assert_eq!(round_tripped.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(round_tripped.entries.len(), 1);
    assert_eq!(round_tripped.entries[0].endpoint_id, peer_bytes);
    assert_eq!(round_tripped.entries[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(
        round_tripped.mesh_id.as_deref(),
        Some("test-mesh-0102030405060708")
    );
}

/// Verifies that remote passive inventory metadata is ignored on ingest.
#[test]
fn proto_v1_route_table_rejects_bad_generation_or_legacy_payload() {
    use crate::proto::node::RouteTable;

    let zero_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &zero_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {:?}",
        err
    );

    let wrong_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 99,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=99 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 99 }),
        "expected BadGeneration{{got:99}}, got {:?}",
        err
    );

    let bad_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for response, got {:?}",
        err
    );

    let wrong_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 42,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=42 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 42 }),
        "expected BadGeneration{{got:42}} for response, got {:?}",
        err
    );

    let legacy_json = b"{\"hosts\":[],\"mesh_id\":null}";
    let mut fake_frame = vec![STREAM_ROUTE_REQUEST];
    fake_frame.extend_from_slice(&(legacy_json.len() as u32).to_le_bytes());
    fake_frame.extend_from_slice(legacy_json);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &fake_frame)
        .expect_err("legacy JSON payload must be rejected");
    assert!(
        matches!(err, ControlFrameError::DecodeError(_)),
        "expected DecodeError for JSON payload, got {:?}",
        err
    );
}

#[tokio::test]
async fn connectivity_snapshot_does_not_treat_admitted_membership_as_connected() {
    let node = Node::new_for_tests(super::NodeRole::Worker).await.unwrap();
    node.insert_test_peer(make_test_peer_info(make_test_endpoint_id(42)))
        .await;

    assert_eq!(
        node.connectivity_snapshot().await,
        super::connectivity::MeshConnectivitySnapshot {
            admitted_peer_count: 1,
            connected_peer_count: 0,
        }
    );
}

#[test]
fn transitive_peer_update_refreshes_memory_only_when_advertised() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let advertised = crate::mesh::AdvertisedMemory {
        total_bytes: 12_000_000_000,
        reserved_bytes: 0,
        platform_reserve_bytes: 0,
        configured_reserve_bytes: 2_000_000_000,
        usable_bytes: 10_000_000_000,
        system_ram_bytes: None,
        ram_offload_bytes: 0,
    };
    let mut existing = make_test_peer_info(peer_id);
    existing.memory = Some(advertised);

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let mut ann = peer_state_test_announcement(addr.clone());
    apply_transitive_ann(&mut existing, &addr, &ann, make_test_endpoint_id(0xee));
    assert_eq!(
        existing.memory,
        Some(advertised),
        "a relay without the block keeps the last advertised one"
    );

    let refreshed = crate::mesh::AdvertisedMemory {
        configured_reserve_bytes: 3_000_000_000,
        usable_bytes: 9_000_000_000,
        ..advertised
    };
    ann.memory = Some(refreshed);
    apply_transitive_ann(&mut existing, &addr, &ann, make_test_endpoint_id(0xee));
    assert_eq!(existing.memory, Some(refreshed));
}

#[test]
fn transitive_peer_update_drops_the_cached_memory_when_the_capacity_moves() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x12; 32]).public());
    let advertised = crate::mesh::AdvertisedMemory {
        total_bytes: 12_000_000_000,
        reserved_bytes: 0,
        platform_reserve_bytes: 0,
        configured_reserve_bytes: 2_000_000_000,
        usable_bytes: 10_000_000_000,
        system_ram_bytes: None,
        ram_offload_bytes: 0,
    };
    let mut existing = make_test_peer_info(peer_id);
    existing.vram_bytes = 10_000_000_000;
    existing.memory = Some(advertised);

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    // An older relay strips the block and carries a new cap: the cached block
    // explained the old budget, so it must not travel with the new one.
    let mut ann = peer_state_test_announcement(addr.clone());
    ann.vram_bytes = 8_000_000_000;
    apply_transitive_ann(&mut existing, &addr, &ann, make_test_endpoint_id(0xee));
    assert_eq!(existing.vram_bytes, 8_000_000_000);
    assert_eq!(
        existing.memory, None,
        "a stale breakdown must not be paired with a new capacity"
    );

    // The same relay with the unchanged capacity keeps the block.
    existing.memory = Some(advertised);
    apply_transitive_ann(&mut existing, &addr, &ann, make_test_endpoint_id(0xee));
    assert_eq!(existing.memory, Some(advertised));
}

#[test]
fn peer_meaningfully_changed_detects_memory_updates() {
    let peer_id = make_test_endpoint_id(13);
    let mut old_peer = make_test_peer_info(peer_id);
    let mut new_peer = old_peer.clone();

    let advertised = crate::mesh::AdvertisedMemory {
        total_bytes: 12_000_000_000,
        reserved_bytes: 0,
        platform_reserve_bytes: 0,
        configured_reserve_bytes: 2_000_000_000,
        usable_bytes: 10_000_000_000,
        system_ram_bytes: None,
        ram_offload_bytes: 0,
    };
    old_peer.memory = Some(advertised);
    new_peer.memory = Some(crate::mesh::AdvertisedMemory {
        configured_reserve_bytes: 3_000_000_000,
        usable_bytes: 9_000_000_000,
        ..advertised
    });

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    assert!(!peer_meaningfully_changed(&old_peer, &old_peer.clone()));
}
