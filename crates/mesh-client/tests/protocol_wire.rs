// Integration tests for mesh-client::protocol wire types.
// These tests verify the portable protocol layer that is safe to use on mobile targets.

use mesh_client::proto::node::{
    DirectNodeAdmissionProof, GossipFrame, MeshGenesisPolicy, MeshRequirements, MeshSubprotocol,
    MeshSubprotocolOpen, NodeRole, NodeVersionBounds, OwnerControlEnvelope, OwnerControlErrorCode,
    OwnerControlGetConfigRequest, OwnerControlRequest, PeerAnnouncement, PeerDown, PeerLeaving,
    ProtocolGenerationBounds, ReleaseAttestationRequirement, ReleaseBuildAttestation, RouteTable,
    RouteTableRequest, SignedMeshGenesisPolicy,
};
use mesh_client::protocol::{
    ALPN_CONTROL_V1, ALPN_V0, ALPN_V1, ControlFrameError, ControlProtocol, MAX_CONTROL_FRAME_BYTES,
    NODE_PROTOCOL_GENERATION, STREAM_CONFIG_PUSH, STREAM_CONFIG_SUBSCRIBE, STREAM_GOSSIP,
    STREAM_PEER_DOWN, STREAM_PEER_LEAVING, STREAM_ROUTE_REQUEST, STREAM_SUBPROTOCOL,
    STREAM_TUNNEL_MAP, decode_control_frame, decode_legacy_tunnel_map_frame,
    decode_owner_control_envelope, encode_control_frame, encode_owner_control_envelope,
    owner_control_rejection_envelope,
};
use mesh_client::{
    ConfigTransportSelection, ControlPlaneBootstrapOptions, ControlPlaneRetryPolicy,
};
use prost::Message;

// ── ALPN constants ──────────────────────────────────────────────────────────

#[test]
fn alpn_v0_is_correct() {
    assert_eq!(ALPN_V0, b"mesh-llm/0");
}

#[test]
fn alpn_v1_is_correct() {
    assert_eq!(ALPN_V1, b"mesh-llm/1");
}

#[test]
fn control_alpn_is_correct() {
    assert_eq!(ALPN_CONTROL_V1, b"mesh-llm-control/1");
}

// ── ControlProtocol ─────────────────────────────────────────────────────────

#[test]
fn protocol_from_alpn_v1() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
}

#[test]
fn protocol_from_alpn_v0() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(protocol_from_alpn(ALPN_V0), ControlProtocol::JsonV0);
}

#[test]
fn protocol_from_alpn_unknown_defaults_to_v1() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}

// ── Wire constants sanity ────────────────────────────────────────────────────

#[test]
fn stream_type_constants_are_distinct() {
    let types = [
        STREAM_GOSSIP,
        STREAM_TUNNEL_MAP,
        STREAM_ROUTE_REQUEST,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_CONFIG_SUBSCRIBE,
        STREAM_CONFIG_PUSH,
    ];
    let mut seen = std::collections::HashSet::new();
    for t in &types {
        assert!(seen.insert(t), "duplicate stream type constant: {:#04x}", t);
    }
}

#[test]
fn node_protocol_generation_is_one() {
    assert_eq!(NODE_PROTOCOL_GENERATION, 1u32);
}

#[test]
fn max_control_frame_bytes_is_eight_mib() {
    assert_eq!(MAX_CONTROL_FRAME_BYTES, 8 * 1024 * 1024);
}

#[test]
fn config_stream_constants_remain_stable() {
    assert_eq!(STREAM_CONFIG_SUBSCRIBE, 0x0b);
    assert_eq!(STREAM_CONFIG_PUSH, 0x0c);
    assert_eq!(STREAM_SUBPROTOCOL, 0x0d);
}

#[test]
fn control_plane_bootstrap_requires_explicit_endpoint_by_default() {
    let err = ControlPlaneBootstrapOptions::new()
        .select_transport()
        .expect_err("new config clients should require explicit owner-control endpoints");

    assert_eq!(err.code, OwnerControlErrorCode::ControlEndpointRequired);
    assert!(!err.legacy_retry_allowed);
}

#[test]
fn control_plane_bootstrap_uses_explicit_control_endpoint() {
    let selection = ControlPlaneBootstrapOptions::new()
        .with_control_endpoint("https://control.example.test")
        .select_transport()
        .expect("configured control endpoint should stay on owner-control lane");

    assert_eq!(
        selection,
        ConfigTransportSelection::OwnerControl {
            endpoint: "https://control.example.test".to_string(),
            retry_policy: ControlPlaneRetryPolicy::NoSilentLegacyDowngrade,
        }
    );
}

// ── Control frame encode / decode roundtrip ──────────────────────────────────

fn make_valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

fn mesh_requirements_signed_policy_proto() -> SignedMeshGenesisPolicy {
    SignedMeshGenesisPolicy {
        version: 1,
        policy: Some(MeshGenesisPolicy {
            version: 1,
            origin_owner_id: "owner-123".into(),
            created_at_unix_ms: 1_717_171_717_000,
            requirements: Some(MeshRequirements {
                node_version: Some(NodeVersionBounds {
                    min: Some("0.65.0".into()),
                    max: Some("0.65.2".into()),
                }),
                protocol_generation: Some(ProtocolGenerationBounds {
                    min: Some(1),
                    max: Some(2),
                }),
                release_attestation: Some(ReleaseAttestationRequirement {
                    required: Some(true),
                    allowed_signer_keys: vec!["signer-a".into(), "signer-b".into()],
                }),
            }),
        }),
        origin_sign_public_key: vec![0x11; 32],
        signature_algorithm: "ed25519".into(),
        signature: vec![0x22; 64],
    }
}

fn mesh_requirements_release_attestation_proto() -> ReleaseBuildAttestation {
    ReleaseBuildAttestation {
        version: 1,
        node_version: "0.65.1".into(),
        build_id: "build-123".into(),
        commit: "abcdef123456".into(),
        target_triple: "aarch64-apple-darwin".into(),
        supported_protocol_generation_min: Some(1),
        supported_protocol_generation_max: Some(2),
        artifact_digest: Some("sha256:deadbeef".into()),
        signer_key_id: "signer-a".into(),
        signature_algorithm: "ed25519".into(),
        signature: vec![0x33; 64],
    }
}

fn sparse_release_attestation_proto() -> ReleaseBuildAttestation {
    ReleaseBuildAttestation {
        version: 1,
        node_version: "0.65.1".into(),
        build_id: "build-legacy".into(),
        commit: "abcdef123456".into(),
        target_triple: "aarch64-apple-darwin".into(),
        supported_protocol_generation_min: None,
        supported_protocol_generation_max: None,
        artifact_digest: None,
        signer_key_id: "signer-a".into(),
        signature_algorithm: "ed25519".into(),
        signature: vec![0x7F; 64],
    }
}

#[test]
fn gossip_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, vec![0u8; 32]);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}

#[test]
fn mesh_requirements_missing_optional_fields_remain_legacy_compatible() {
    let frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0x44; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0x55; 32],
            role: NodeRole::Worker as i32,
            version: Some("0.65.1".into()),
            mesh_id: Some("mesh-legacy".into()),
            ..Default::default()
        }],
    };

    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("legacy gossip frames without mesh-requirements proofs must still decode");

    let peer = &decoded.peers[0];
    assert_eq!(peer.version.as_deref(), Some("0.65.1"));
    assert_eq!(peer.mesh_id.as_deref(), Some("mesh-legacy"));
    assert_eq!(peer.mesh_policy_hash, None);
    assert!(peer.genesis_policy.is_none());
    assert!(peer.release_attestation.is_none());
}

#[test]
fn mesh_requirements_gossip_roundtrip_preserves_policy_and_attestation_fields() {
    let frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0x44; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0x55; 32],
            role: NodeRole::Worker as i32,
            version: Some("0.65.1".into()),
            mesh_id: Some("mesh-policy-a".into()),
            mesh_policy_hash: Some(
                "40a3e2b4d96294e47f443c74d0d8441bd3363efea1580eb82627253ae47363ee".into(),
            ),
            genesis_policy: Some(mesh_requirements_signed_policy_proto()),
            release_attestation: Some(mesh_requirements_release_attestation_proto()),
            ..Default::default()
        }],
    };

    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("mesh-requirements gossip fields must survive wire roundtrip");

    let peer = &decoded.peers[0];
    assert_eq!(peer.mesh_id.as_deref(), Some("mesh-policy-a"));
    assert_eq!(
        peer.mesh_policy_hash.as_deref(),
        Some("40a3e2b4d96294e47f443c74d0d8441bd3363efea1580eb82627253ae47363ee")
    );
    assert_eq!(
        peer.genesis_policy
            .as_ref()
            .and_then(|policy| policy.policy.as_ref())
            .map(|policy| policy.origin_owner_id.as_str()),
        Some("owner-123")
    );
    assert_eq!(
        peer.release_attestation
            .as_ref()
            .map(|attestation| attestation.signer_key_id.as_str()),
        Some("signer-a")
    );
}

#[test]
fn mesh_requirements_release_attestation_roundtrip_preserves_sparse_optional_fields() {
    let attestation = sparse_release_attestation_proto();
    let encoded = attestation.encode_to_vec();
    let decoded = ReleaseBuildAttestation::decode(encoded.as_slice())
        .expect("sparse release attestation protobuf should roundtrip");

    assert_eq!(decoded.supported_protocol_generation_min, None);
    assert_eq!(decoded.supported_protocol_generation_max, None);
    assert_eq!(decoded.artifact_digest, None);
    assert_eq!(decoded.signature, vec![0x7F; 64]);
}

#[test]
fn mesh_requirements_release_attestation_roundtrip_preserves_invalid_signature_bytes() {
    let mut attestation = mesh_requirements_release_attestation_proto();
    attestation.signature[0] ^= 0x01;

    let encoded = attestation.encode_to_vec();
    let decoded = ReleaseBuildAttestation::decode(encoded.as_slice())
        .expect("invalid release attestation protobuf should still roundtrip");

    assert_eq!(decoded.signature, attestation.signature);
    assert_eq!(decoded.signer_key_id, "signer-a");
}

#[test]
fn mesh_requirements_direct_proof_proto_roundtrip_preserves_fields() {
    let proof = DirectNodeAdmissionProof {
        version: 1,
        sender_id: vec![0x66; 32],
        mesh_id: "mesh-policy-a".into(),
        policy_hash: "policy-hash-a".into(),
        attestation_hash: "attestation-hash-a".into(),
        timestamp_unix_ms: 1_717_171_717_000,
        signature_algorithm: "ed25519".into(),
        signature: vec![0x77; 64],
    };

    let encoded = prost::Message::encode_to_vec(&proof);
    let decoded = DirectNodeAdmissionProof::decode(encoded.as_slice())
        .expect("direct proof protobuf should roundtrip");

    assert_eq!(decoded.mesh_id, "mesh-policy-a");
    assert_eq!(decoded.policy_hash, "policy-hash-a");
    assert_eq!(decoded.attestation_hash, "attestation-hash-a");
    assert_eq!(decoded.sender_id, vec![0x66; 32]);
}

#[test]
fn gossip_frame_bad_generation_rejected() {
    let mut frame = make_valid_gossip_frame();
    frame.r#gen = 0;
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gen=0 gossip frame must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {err:?}"
    );
}

#[test]
fn gossip_subprotocol_discovery_roundtrip_and_validation() {
    let frame = GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            subprotocols: vec![MeshSubprotocol {
                name: "skippy-stage".to_string(),
                major: 1,
                features: vec!["stage-control".to_string(), "artifact-transfer".to_string()],
            }],
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip subprotocol discovery must decode");
    assert_eq!(decoded.peers[0].subprotocols[0].name, "skippy-stage");
    assert_eq!(decoded.peers[0].subprotocols[0].major, 1);

    let mut invalid = decoded;
    invalid.peers[0].subprotocols[0].name.clear();
    let encoded = encode_control_frame(STREAM_GOSSIP, &invalid);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("invalid subprotocol discovery must be rejected");
    assert!(matches!(err, ControlFrameError::InvalidSubprotocol));
}

#[test]
fn mesh_subprotocol_open_validates_generic_envelope() {
    let open = MeshSubprotocolOpen {
        r#gen: NODE_PROTOCOL_GENERATION,
        name: "skippy-stage".to_string(),
        major: 1,
    };
    let encoded = encode_control_frame(STREAM_SUBPROTOCOL, &open);
    let decoded: MeshSubprotocolOpen = decode_control_frame(STREAM_SUBPROTOCOL, &encoded).unwrap();
    assert_eq!(decoded.name, "skippy-stage");
    assert_eq!(decoded.major, 1);

    let bad = MeshSubprotocolOpen {
        r#gen: NODE_PROTOCOL_GENERATION,
        name: " ".to_string(),
        major: 1,
    };
    let encoded = encode_control_frame(STREAM_SUBPROTOCOL, &bad);
    let err = decode_control_frame::<MeshSubprotocolOpen>(STREAM_SUBPROTOCOL, &encoded)
        .expect_err("empty subprotocol names must be rejected");
    assert!(matches!(err, ControlFrameError::InvalidSubprotocol));
}

#[test]
fn gossip_frame_invalid_sender_id_rejected() {
    let mut frame = make_valid_gossip_frame();
    frame.sender_id = vec![0u8; 16]; // 16 bytes instead of 32
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("short sender_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidSenderId { got: 16 }),
        "expected InvalidSenderId{{got:16}}, got {err:?}"
    );
}

#[test]
fn wrong_stream_type_rejected() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("wrong stream type must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: STREAM_TUNNEL_MAP,
                got: STREAM_GOSSIP,
            }
        ),
        "expected WrongStreamType, got {err:?}"
    );
}

// ── PeerDown / PeerLeaving roundtrip ─────────────────────────────────────────

#[test]
fn peer_down_roundtrip() {
    let msg = PeerDown {
        peer_id: vec![0xAB; 32],
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &msg);
    let decoded: PeerDown =
        decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");
    assert_eq!(decoded.peer_id, vec![0xAB; 32]);
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_leaving_roundtrip() {
    let msg = PeerLeaving {
        peer_id: vec![0xCD; 32],
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &msg);
    let decoded: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");
    assert_eq!(decoded.peer_id, vec![0xCD; 32]);
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_down_bad_generation_rejected() {
    let msg = PeerDown {
        peer_id: vec![0x77; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &msg);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

// ── RouteTable roundtrip ─────────────────────────────────────────────────────

#[test]
fn route_table_request_bad_generation_rejected() {
    let req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTableRequest gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

#[test]
fn route_table_bad_generation_rejected() {
    let table = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTable gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

// ── v0 legacy compatibility ───────────────────────────────────────────────────

#[test]
fn decode_legacy_tunnel_map_from_json() {
    // JSON: { "<hex-peer-id>": <port> }
    let peer_bytes = [0x42u8; 32];
    let hex_id = hex::encode(peer_bytes);
    let json = format!("{{\"{hex_id}\": 9337}}");

    let frame = decode_legacy_tunnel_map_frame(json.as_bytes())
        .expect("valid legacy tunnel map JSON must decode");

    assert_eq!(frame.entries.len(), 1);
    assert_eq!(frame.entries[0].target_peer_id, peer_bytes.to_vec());
    assert_eq!(frame.entries[0].tunnel_port, 9337);
}

#[test]
fn decode_legacy_tunnel_map_invalid_hex_ignored() {
    let json = b"{\"notvalidhex\": 9337}";
    let frame = decode_legacy_tunnel_map_frame(json)
        .expect("invalid hex entries should be silently ignored");
    assert_eq!(frame.entries.len(), 0);
}

// ── ControlFrameError Display ────────────────────────────────────────────────

#[test]
fn control_frame_error_display_bad_generation() {
    let err = ControlFrameError::BadGeneration { got: 99 };
    let s = err.to_string();
    assert!(s.contains("99"), "Display must mention the bad gen value");
}

#[test]
fn control_frame_error_implements_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(ControlFrameError::BadGeneration { got: 0 });
    assert!(err.to_string().contains("0"));
}

#[test]
fn owner_control_envelope_roundtrip() {
    let envelope = OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            request_id: 5,
            get_config: Some(OwnerControlGetConfigRequest {
                requester_node_id: vec![0x10; 32],
                target_node_id: vec![0x20; 32],
            }),
            watch_config: None,
            apply_config: None,
            refresh_inventory: None,
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        }),
        response: None,
        error: None,
    };
    let decoded = decode_owner_control_envelope(&encode_owner_control_envelope(&envelope))
        .expect("valid owner-control envelope must decode");
    assert_eq!(decoded.request.unwrap().request_id, 5);
}

#[test]
fn owner_control_unknown_command_rejects_with_structured_error() {
    let envelope = OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            request_id: 6,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: None,
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        }),
        response: None,
        error: None,
    };
    let bytes = encode_owner_control_envelope(&envelope);
    let err = decode_owner_control_envelope(&bytes)
        .expect_err("missing command variant must be rejected");
    let rejection = owner_control_rejection_envelope(&bytes, Some(6), &err);
    let error = rejection
        .error
        .expect("structured rejection must carry an error");
    assert_eq!(
        OwnerControlErrorCode::try_from(error.code).unwrap(),
        OwnerControlErrorCode::UnknownCommand
    );
}

#[test]
fn owner_control_legacy_json_rejects_with_structured_error() {
    let legacy_json = br#"{"request_id":6,"command":"GetConfig"}"#;
    let err = decode_owner_control_envelope(legacy_json)
        .expect_err("legacy json must be rejected on protobuf-only control plane");
    let rejection = owner_control_rejection_envelope(legacy_json, Some(6), &err);
    let error = rejection
        .error
        .expect("structured rejection must carry an error");
    assert_eq!(
        OwnerControlErrorCode::try_from(error.code).unwrap(),
        OwnerControlErrorCode::LegacyJsonUnsupported
    );
}
