use super::*;

#[test]
fn test_peer_announcement_first_joined_mesh_ts_roundtrip() {
    use iroh::SecretKey;
    use std::collections::HashMap;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xEF; 32]).public());

    let ann_with_timestamp = super::super::PeerAnnouncement {
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::super::NodeRole::Worker,
        first_joined_mesh_ts: Some(1_700_000_000_000u64),
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
    };

    let proto_pa = local_ann_to_proto_ann(&ann_with_timestamp);
    assert_eq!(proto_pa.first_joined_mesh_ts, Some(1_700_000_000_000u64));

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(
        roundtripped.first_joined_mesh_ts,
        Some(1_700_000_000_000u64)
    );

    let ann_without_timestamp = super::super::PeerAnnouncement {
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

    let proto_pa = local_ann_to_proto_ann(&ann_without_timestamp);
    assert_eq!(proto_pa.first_joined_mesh_ts, None);

    let (_, roundtripped) = proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
    assert_eq!(roundtripped.first_joined_mesh_ts, None);
}
