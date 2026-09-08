use super::*;
use crate::crypto::OwnershipSummary;
use crate::mesh::{PeerInfo, resolve_peer_down, resolve_peer_leaving};
use crate::proto::node::{
    ConfiguredModelRef, GossipFrame, MeshSubprotocolOpen, NodeConfigSnapshot, NodeGpuConfig,
    NodeModelEntry, NodePluginEntry, NodeRole, OwnerControlError, OwnerControlErrorCode,
    OwnerControlHandshake, PeerAnnouncement, RouteTableRequest, SignedNodeOwnership,
};
use iroh::{EndpointAddr, EndpointId, SecretKey};
use std::collections::{HashMap, HashSet};

const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");

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

fn make_config_snapshot() -> NodeConfigSnapshot {
    NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Pinned as i32,
        }),
        models: vec![NodeModelEntry {
            model: "Qwen3-8B".to_string(),
            mmproj: Some("mmproj-cut".to_string()),
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".to_string()),
            model_ref: Some(ConfiguredModelRef {
                declared_ref: "Qwen3-8B".to_string(),
                source_kind: None,
                revision: None,
            }),
            mmproj_ref: Some(ConfiguredModelRef {
                declared_ref: "mmproj-cut".to_string(),
                source_kind: None,
                revision: None,
            }),
        }],
        plugins: vec![NodePluginEntry {
            name: "demo".to_string(),
            enabled: Some(true),
            command: Some("mesh-llm".to_string()),
            args: vec!["--plugin".to_string(), "demo".to_string()],
        }],
        config_toml: None,
        mesh_requirements: None,
        lifecycle_log_parser: None,
    }
}

fn make_nested_mesh_config() -> crate::plugin::MeshConfig {
    toml::from_str(
        r#"version = 1

[gpu]
assignment = "auto"
parallel = 2

[defaults.model_fit]
kv_unified = "auto"

[defaults.hardware]
gpu_layers = "auto"

[defaults.throughput]
parallel = 3

[defaults.skippy]

[defaults.speculative]
mode = "auto"

[defaults.request_defaults]
reasoning_budget = "auto"

[defaults.multimodal]
mmproj = "defaults-projector.gguf"

[defaults.advanced.server]
alias = "defaults-alias"

[[models]]
model = "Qwen3-8B.gguf"

[models.model_fit]
ctx_size = 16384

[models.hardware]
gpu_layers = 99

[models.throughput]
parallel = 4

[models.speculative]
draft_selection_policy = "auto"

[models.request_defaults]
top_p = 0.95

[models.multimodal]
mmproj = "model-projector.gguf"

[models.advanced.server]
alias = "model-alias"
"#,
    )
    .expect("nested mesh config should parse")
}

fn make_valid_owner_control_handshake() -> OwnerControlHandshake {
    OwnerControlHandshake {
        ownership: Some(SignedNodeOwnership {
            version: 1,
            cert_id: "cert-1".to_string(),
            owner_id: "owner-1".to_string(),
            owner_sign_public_key: vec![0x11; 32],
            node_endpoint_id: vec![0x22; 32],
            issued_at_unix_ms: 1,
            expires_at_unix_ms: 2,
            node_label: Some("node-01".to_string()),
            hostname_hint: Some("node-01".to_string()),
            signature: vec![0x33; 64],
        }),
    }
}
fn make_test_peer_info(peer_id: EndpointId) -> PeerInfo {
    PeerInfo {
        id: peer_id,
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: crate::mesh::NodeRole::Worker,
        first_joined_mesh_ts: None,
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

mod announcements;
mod config;
mod control_frames;
mod mesh_timestamps;
mod owner_control;

pub(crate) use announcements::assert_mixed_version_peer_ignores_missing_release_attestation;
pub(crate) use config::mesh_requirements_survive_owner_control_config_round_trip;
