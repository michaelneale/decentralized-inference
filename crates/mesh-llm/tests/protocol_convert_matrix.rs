//! Protocol Conversion Matrix Tests
//!
//! Covers the v0↔v1 conversion paths exposed by `mesh-client::protocol`:
//! `canonical_config_hash`, frame encode/decode round-trips, frame validation,
//! and the v0 legacy tunnel-map decode path.

use mesh_client::proto::node::{
    GossipFrame, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry, OwnerControlEnvelope,
    OwnerControlGetConfigRequest, OwnerControlRequest, PeerAnnouncement,
};
use mesh_client::protocol::convert::canonical_config_hash;
use mesh_client::protocol::{
    ControlFrameError, MAX_CONTROL_FRAME_BYTES, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP,
    STREAM_TUNNEL_MAP, decode_control_frame, decode_legacy_tunnel_map_frame,
    decode_owner_control_envelope, encode_control_frame, encode_owner_control_envelope,
    owner_control_rejection_envelope,
};
use mesh_llm_host_runtime::{
    BootstrapStatus, DirectPeerProofStatus, MeshGenesisPolicy, MeshRequirementDecision,
    MeshRequirementEvaluationInput, MeshRequirementRejectReason, MeshRequirements,
    NodeVersionBounds, PeerReleaseAttestationStatus, ProtocolGenerationBounds,
    ReleaseAttestationRequirement, ReleaseBuildAttestation, SignedBootstrapToken,
    SignedMeshGenesisPolicy, crypto::OwnerKeypair,
};

fn test_release_signing_key() -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[7u8; 32], &[8u8; 32]).expect("deterministic owner keypair")
}

fn test_release_signer_key_id() -> String {
    format!(
        "ed25519:{}",
        hex::encode(test_release_signing_key().verifying_key().as_bytes())
    )
}

fn signed_release_attestation() -> ReleaseBuildAttestation {
    let signing_key = test_release_signing_key();
    let mut attestation = ReleaseBuildAttestation {
        version: 1,
        node_version: "0.65.1".into(),
        build_id: "build-123".into(),
        commit: "abcdef123456".into(),
        target_triple: "aarch64-apple-darwin".into(),
        supported_protocol_generation_min: Some(1),
        supported_protocol_generation_max: Some(2),
        artifact_digest: Some("sha256:deadbeef".into()),
        signer_key_id: test_release_signer_key_id(),
        signature_algorithm: "ed25519".into(),
        signature: vec![0; 64],
    };
    attestation.signature = signing_key
        .sign_bytes(
            &attestation
                .canonical_bytes()
                .expect("canonical attestation bytes"),
        )
        .to_vec();
    attestation
}

fn restricted_requirements() -> MeshRequirements {
    MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.0".into()),
            max: Some("0.65.2".into()),
        },
        protocol_generation: ProtocolGenerationBounds {
            min: Some(1),
            max: Some(2),
        },
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["signer-b".into(), "signer-a".into()],
        },
    }
}

fn stable_requirement_input() -> MeshRequirementEvaluationInput {
    MeshRequirementEvaluationInput {
        advertised_node_version: Some("0.65.1".into()),
        negotiated_protocol_generation: Some(NODE_PROTOCOL_GENERATION),
        direct_proof: DirectPeerProofStatus::Verified,
        ..Default::default()
    }
}

fn signed_policy_and_owner() -> (SignedMeshGenesisPolicy, OwnerKeypair) {
    let owner = OwnerKeypair::generate();
    let policy = MeshGenesisPolicy::new(
        owner.owner_id(),
        1_717_171_717_000,
        restricted_requirements(),
    )
    .expect("policy should validate");
    let signed = SignedMeshGenesisPolicy::sign(policy, &owner).expect("signed policy");
    (signed, owner)
}

fn minimal_config() -> NodeConfigSnapshot {
    NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![NodeModelEntry {
            model: "Qwen3-8B".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            model_ref: None,
            mmproj_ref: None,
        }],
        plugins: vec![],
        config_toml: None,
        mesh_requirements: None,
        lifecycle_log_parser: None,
    }
}

fn valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0xAB; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: mesh_client::proto::node::NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

// ── canonical_config_hash ─────────────────────────────────────────────────────

#[test]
fn canonical_config_hash_output_is_32_bytes() {
    let hash = canonical_config_hash(&minimal_config());
    assert_eq!(hash.len(), 32);
}

#[test]
fn canonical_config_hash_is_deterministic() {
    let a = canonical_config_hash(&minimal_config());
    let b = canonical_config_hash(&minimal_config());
    assert_eq!(a, b);
}

#[test]
fn canonical_config_hash_differs_for_different_configs() {
    let config_a = minimal_config();
    let mut config_b = minimal_config();
    config_b.models.push(NodeModelEntry {
        model: "GLM-4.7-Flash".to_string(),
        mmproj: None,
        ctx_size: None,
        gpu_id: None,
        model_ref: None,
        mmproj_ref: None,
    });
    assert_ne!(
        canonical_config_hash(&config_a),
        canonical_config_hash(&config_b)
    );
}

// ── encode_control_frame + decode_control_frame round-trips ──────────────────

#[test]
fn gossip_frame_encodes_and_decodes_intact() {
    let frame = valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, vec![0xAB; 32]);
    assert_eq!(decoded.peers.len(), 1);
}

/// decode_control_frame must reject a buffer whose stream-type byte does not match.
#[test]
fn decode_rejects_wrong_stream_type() {
    let encoded = encode_control_frame(STREAM_GOSSIP, &valid_gossip_frame());
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("mismatched stream type must be rejected");
    assert!(matches!(
        err,
        ControlFrameError::WrongStreamType {
            expected: STREAM_TUNNEL_MAP,
            got: STREAM_GOSSIP
        }
    ));
}

/// A frame whose embedded length field exceeds MAX_CONTROL_FRAME_BYTES must be
/// rejected before any allocation attempt.
#[test]
fn decode_rejects_oversize_frame() {
    let oversize_len = (MAX_CONTROL_FRAME_BYTES + 1) as u32;
    let mut fake_frame = vec![STREAM_GOSSIP];
    fake_frame.extend_from_slice(&oversize_len.to_le_bytes());
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake_frame)
        .expect_err("frame claiming to exceed MAX_CONTROL_FRAME_BYTES must be rejected");
    assert!(matches!(err, ControlFrameError::OversizeFrame { .. }));
}

#[test]
fn decode_rejects_truncated_frame() {
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &[0x01, 0x02])
        .expect_err("truncated frame must be rejected");
    assert!(matches!(err, ControlFrameError::DecodeError(_)));
}

// ── v0 → v1: legacy tunnel-map JSON decode ───────────────────────────────────

#[test]
fn v0_tunnel_map_json_decode_roundtrip() {
    let mut map = std::collections::HashMap::new();
    map.insert(hex::encode([0xBB; 32]), 9337u16);
    let json = serde_json::to_vec(&map).unwrap();
    let tunnel_map = decode_legacy_tunnel_map_frame(&json)
        .expect("well-formed v0 tunnel map JSON must decode successfully");
    assert_eq!(tunnel_map.entries.len(), 1);
    assert_eq!(tunnel_map.entries[0].tunnel_port, 9337);
    assert_eq!(tunnel_map.entries[0].target_peer_id, vec![0xBB; 32]);
}

#[test]
fn v0_tunnel_map_empty_json_object_yields_no_entries() {
    let tunnel_map = decode_legacy_tunnel_map_frame(b"{}")
        .expect("empty v0 tunnel map must decode to zero entries");
    assert!(tunnel_map.entries.is_empty());
}

#[test]
fn v0_tunnel_map_rejects_invalid_hex_peer_id() {
    let mut map = std::collections::HashMap::new();
    map.insert("not-valid-hex".to_string(), 9337u16);
    let json = serde_json::to_vec(&map).unwrap();
    let tunnel_map = decode_legacy_tunnel_map_frame(&json)
        .expect("invalid hex entries must be silently skipped, not panic");
    assert!(
        tunnel_map.entries.is_empty(),
        "invalid hex peer ids must be filtered out"
    );
}

// ── Frame validation rejects bad generation ───────────────────────────────────

#[test]
fn gossip_frame_with_wrong_generation_is_rejected() {
    let bad_frame = GossipFrame {
        r#gen: 0,
        sender_id: vec![0u8; 32],
        peers: vec![],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gossip frame with gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));
}

#[test]
fn owner_control_get_config_roundtrip_works() {
    let envelope = OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            request_id: 77,
            get_config: Some(OwnerControlGetConfigRequest {
                requester_node_id: vec![0x11; 32],
                target_node_id: vec![0x22; 32],
            }),
            watch_config: None,
            apply_config: None,
            refresh_inventory: None,
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
        }),
        response: None,
        error: None,
    };
    let decoded = decode_owner_control_envelope(&encode_owner_control_envelope(&envelope))
        .expect("valid owner control envelope must decode");
    assert_eq!(decoded.request.unwrap().request_id, 77);
}

#[test]
fn owner_control_unknown_command_maps_to_structured_error() {
    let envelope = OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(OwnerControlRequest {
            request_id: 88,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: None,
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
        }),
        response: None,
        error: None,
    };
    let bytes = encode_owner_control_envelope(&envelope);
    let err = decode_owner_control_envelope(&bytes)
        .expect_err("missing command variant must fail validation");
    let rejection = owner_control_rejection_envelope(&bytes, Some(88), &err);
    let error = rejection
        .error
        .expect("rejection must carry structured error");
    assert_eq!(
        mesh_client::proto::node::OwnerControlErrorCode::try_from(error.code).unwrap(),
        mesh_client::proto::node::OwnerControlErrorCode::UnknownCommand
    );
}

#[test]
fn owner_control_legacy_json_maps_to_structured_error() {
    let legacy_json = br#"{"request_id":99,"command":"GetConfig"}"#;
    let err = decode_owner_control_envelope(legacy_json)
        .expect_err("legacy json must be rejected by protobuf-only owner-control lane");
    let rejection = owner_control_rejection_envelope(legacy_json, Some(99), &err);
    let error = rejection
        .error
        .expect("rejection must carry structured error");
    assert_eq!(
        mesh_client::proto::node::OwnerControlErrorCode::try_from(error.code).unwrap(),
        mesh_client::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
    );
}

#[test]
fn mesh_requirements_genesis_policy_roundtrips() {
    let policy = MeshGenesisPolicy::new(
        "owner-123",
        1_717_171_717_000,
        MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.0".into()),
                max: Some("0.65.9".into()),
            },
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(1),
            },
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["release-signer-a".into()],
            },
        },
    )
    .unwrap();
    let signed = SignedMeshGenesisPolicy {
        version: 1,
        policy,
        origin_sign_public_key: vec![0x11; 32],
        signature_algorithm: "ed25519".into(),
        signature: vec![0x22; 64],
    };

    let proto = signed.to_proto();
    let roundtripped = SignedMeshGenesisPolicy::from_proto(&proto).unwrap();

    assert_eq!(roundtripped, signed);
    assert_eq!(
        proto
            .policy
            .unwrap()
            .requirements
            .unwrap()
            .node_version
            .unwrap()
            .min
            .as_deref(),
        Some("0.65.0")
    );
}

#[test]
fn mesh_requirements_release_attestation_roundtrips() {
    let attestation = signed_release_attestation();

    let proto = attestation.to_proto();
    let roundtripped = ReleaseBuildAttestation::from_proto(&proto);

    assert_eq!(roundtripped, attestation);
    assert_eq!(proto.artifact_digest.as_deref(), Some("sha256:deadbeef"));
    assert_eq!(proto.supported_protocol_generation_max, Some(2));
    roundtripped.verify().expect("attestation should verify");
}

#[test]
fn mesh_requirements_policy_canonical_hash_matches_stable_fixture() {
    let policy = MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
        .expect("policy should validate");
    assert_eq!(
        policy.canonical_hash_hex().expect("hash should compute"),
        "40a3e2b4d96294e47f443c74d0d8441bd3363efea1580eb82627253ae47363ee"
    );
}

#[test]
fn mesh_requirements_policy_hash_derives_mesh_id() {
    let policy = MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
        .expect("policy should validate");
    assert_eq!(
        policy
            .policy_derived_mesh_id()
            .expect("mesh id should compute"),
        policy.canonical_hash_hex().expect("hash should compute")
    );
}

#[test]
fn mesh_requirements_policy_change_creates_distinct_mesh_id() {
    let policy_a =
        MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
            .expect("policy should validate");
    let mut requirements_b = restricted_requirements();
    requirements_b.release_attestation.allowed_signer_keys = vec!["signer-c".into()];
    let policy_b = MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, requirements_b)
        .expect("policy should validate");

    assert_ne!(
        policy_a
            .policy_derived_mesh_id()
            .expect("mesh id should compute"),
        policy_b
            .policy_derived_mesh_id()
            .expect("mesh id should compute")
    );
}

#[test]
fn mesh_requirements_signed_bootstrap_token_roundtrips_and_validates() {
    let (signed_policy, owner) = signed_policy_and_owner();
    let token = SignedBootstrapToken::sign(
        vec![br#"{\"id\":\"endpoint-a\"}"#.to_vec()],
        &signed_policy,
        Some(1_717_171_717_999),
        &owner,
    )
    .expect("token should sign");

    let proto = token.to_proto();
    let roundtripped = SignedBootstrapToken::from_proto(&proto).expect("token should roundtrip");
    roundtripped
        .verify_at(1_717_171_717_500)
        .expect("token should verify before expiry");
    assert_eq!(roundtripped, token);
}

#[test]
fn mesh_requirements_signed_bootstrap_token_rejects_invalid_signature() {
    let (signed_policy, owner) = signed_policy_and_owner();
    let mut token = SignedBootstrapToken::sign(
        vec![br#"{\"id\":\"endpoint-a\"}"#.to_vec()],
        &signed_policy,
        None,
        &owner,
    )
    .expect("token should sign");
    token.signature[0] ^= 0x01;

    assert_eq!(
        token.verify(),
        Err(MeshRequirementRejectReason::BootstrapTokenInvalid)
    );
}

#[test]
fn mesh_requirements_signed_bootstrap_token_rejects_expired_token() {
    let (signed_policy, owner) = signed_policy_and_owner();
    let token = SignedBootstrapToken::sign(
        vec![br#"{\"id\":\"endpoint-a\"}"#.to_vec()],
        &signed_policy,
        Some(100),
        &owner,
    )
    .expect("token should sign");

    assert_eq!(
        token.verify_at(101),
        Err(MeshRequirementRejectReason::BootstrapTokenExpired)
    );
}

#[test]
fn mesh_requirements_release_attestation_accepts_trusted_signer() {
    let requirements = MeshRequirements {
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["trusted-signer".into()],
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Present {
                signer_key: Some("trusted-signer".into()),
                attested_version: None,
            },
            ..Default::default()
        }),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_missing_signer_metadata() {
    let requirements = MeshRequirements {
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["trusted-signer".into()],
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Present {
                signer_key: None,
                attested_version: None,
            },
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BuildProofMissing)
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_missing_proof_when_required() {
    let requirements = MeshRequirements {
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["trusted-signer".into()],
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Unsigned,
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::CertifiedBinaryRequired)
    );
}

#[test]
fn mesh_requirements_release_attestation_allows_missing_proof_for_mixed_version_peers_when_optional()
 {
    assert_eq!(
        MeshRequirements::unrestricted().evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Unsigned,
            ..Default::default()
        }),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_garbled_proof() {
    let requirements = MeshRequirements {
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["trusted-signer".into()],
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Invalid,
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BuildProofInvalid)
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_untrusted_signer() {
    let requirements = MeshRequirements {
        release_attestation: ReleaseAttestationRequirement {
            required: true,
            allowed_signer_keys: vec!["trusted-signer".into()],
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            release_attestation: PeerReleaseAttestationStatus::Present {
                signer_key: Some("evil-signer".into()),
                attested_version: None,
            },
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::ReleaseSignerUntrusted)
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_mismatched_policy_hash() {
    let policy = MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
        .expect("policy should validate");
    assert_eq!(
        policy.evaluate(&MeshRequirementEvaluationInput {
            policy_hash: Some("deadbeef".into()),
            release_attestation: PeerReleaseAttestationStatus::Present {
                signer_key: Some("signer-a".into()),
                attested_version: None,
            },
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::MeshPolicyMismatch)
    );
}

#[test]
fn mesh_requirements_node_version_unset_accepts_legacy_version_inputs() {
    assert_eq!(
        MeshRequirements::unrestricted().evaluate(&MeshRequirementEvaluationInput::default()),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_node_version_min_only_rejects_lower_versions() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.1".into()),
            max: None,
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.0".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
    );
}

#[test]
fn mesh_requirements_node_version_max_only_rejects_higher_versions() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: None,
            max: Some("0.65.1".into()),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.2".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionAboveMaximum)
    );
}

#[test]
fn mesh_requirements_node_version_full_range_accepts_in_range() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.0".into()),
            max: Some("0.65.2".into()),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&stable_requirement_input()),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_node_version_exact_accepts_build_metadata() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.1".into()),
            max: Some("0.65.1".into()),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.1+build.99".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_node_version_exact_rejects_prerelease() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.1".into()),
            max: Some("0.65.1".into()),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.1-alpha.1".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
    );
}

#[test]
fn mesh_requirements_node_version_rejects_malformed_versions() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.1".into()),
            max: None,
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("not-a-version".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionMalformed)
    );
}

#[test]
fn mesh_requirements_node_version_accepts_single_leading_v_or_v_capitalized() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("v0.65.1".into()),
            max: Some("V0.65.1".into()),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            advertised_node_version: Some("V0.65.1".into()),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_release_attestation_rejects_invalid_signature() {
    let mut attestation = signed_release_attestation();
    attestation.signature[0] ^= 0x01;

    let error = attestation
        .verify()
        .expect_err("tampered release attestation signature must fail verification");
    assert!(error.to_string().contains("signature verification failed"));
}

#[test]
fn mesh_requirements_node_version_rejects_missing_versions_when_constrained() {
    let requirements = MeshRequirements {
        node_version: NodeVersionBounds {
            min: Some("0.65.1".into()),
            max: None,
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput::default()),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionMalformed)
    );
}

#[test]
fn mesh_requirements_protocol_generation_unset_accepts_unknown_generation() {
    assert_eq!(
        MeshRequirements::unrestricted().evaluate(&MeshRequirementEvaluationInput::default()),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_protocol_generation_min_only_rejects_lower_generations() {
    let requirements = MeshRequirements {
        protocol_generation: ProtocolGenerationBounds {
            min: Some(2),
            max: None,
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            negotiated_protocol_generation: Some(1),
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(
            MeshRequirementRejectReason::ProtocolGenerationBelowMinimum,
        )
    );
}

#[test]
fn mesh_requirements_protocol_generation_max_only_rejects_higher_generations() {
    let requirements = MeshRequirements {
        protocol_generation: ProtocolGenerationBounds {
            min: None,
            max: Some(1),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            negotiated_protocol_generation: Some(2),
            ..Default::default()
        }),
        MeshRequirementDecision::Rejected(
            MeshRequirementRejectReason::ProtocolGenerationAboveMaximum,
        )
    );
}

#[test]
fn mesh_requirements_protocol_generation_full_range_accepts_in_range() {
    let requirements = MeshRequirements {
        protocol_generation: ProtocolGenerationBounds {
            min: Some(1),
            max: Some(2),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput {
            negotiated_protocol_generation: Some(1),
            ..Default::default()
        }),
        MeshRequirementDecision::Accepted
    );
}

#[test]
fn mesh_requirements_protocol_generation_rejects_unknown_when_constrained() {
    let requirements = MeshRequirements {
        protocol_generation: ProtocolGenerationBounds {
            min: Some(1),
            max: Some(2),
        },
        ..MeshRequirements::unrestricted()
    };
    assert_eq!(
        requirements.evaluate(&MeshRequirementEvaluationInput::default()),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::ProtocolGenerationUnknown)
    );
}

#[test]
fn mesh_requirements_bootstrap_status_invalid_is_rejected_before_range_checks() {
    let policy = MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
        .expect("policy should validate");
    assert_eq!(
        policy.evaluate(&MeshRequirementEvaluationInput {
            bootstrap: BootstrapStatus::Invalid,
            negotiated_protocol_generation: Some(1),
            ..stable_requirement_input()
        }),
        MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BootstrapTokenInvalid)
    );
}
