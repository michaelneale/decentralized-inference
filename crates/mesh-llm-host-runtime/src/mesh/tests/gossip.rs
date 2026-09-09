use super::*;
use crate::crypto::OwnershipSummary;
use crate::mesh::connections::encode_endpoint_addr_token;
use crate::mesh::now_secs;
use iroh::SecretKey;
use mesh_llm_types::mesh::DEMAND_TTL_SECS;
use serial_test::serial;
use std::collections::HashMap;

pub(crate) fn test_endpoint_id(seed: u8) -> EndpointId {
    EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
}

pub(crate) fn test_addr(seed: u8) -> EndpointAddr {
    EndpointAddr {
        id: test_endpoint_id(seed),
        addrs: Default::default(),
    }
}

pub(crate) fn test_announcement(ts: Option<u64>) -> PeerAnnouncement {
    PeerAnnouncement {
        addr: test_addr(0x11),
        role: NodeRole::Worker,
        first_joined_mesh_ts: ts,
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

pub(crate) fn test_peer(ts: Option<u64>) -> PeerInfo {
    PeerInfo::from_announcement(
        test_endpoint_id(0x22),
        test_addr(0x22),
        &test_announcement(ts),
        OwnershipSummary::default(),
    )
}

mod merge_and_refresh {
    use super::*;

    include!("gossip/merge_and_refresh.rs");
}

mod admission {
    use super::*;

    include!("gossip/admission.rs");
}

mod discovery {
    use super::*;

    include!("gossip/discovery.rs");
}
