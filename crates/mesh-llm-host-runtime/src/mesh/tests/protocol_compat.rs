#[test]
fn stage_load_proto_roundtrip_preserves_source_model_bytes() {
    let load = stage_load_request();
    let proto = stage_load_to_proto(load.clone());
    assert_encoded_stage_load(&proto);

    let decoded = stage_load_from_proto(proto).unwrap();
    assert_decoded_stage_load(&decoded);
}

fn assert_encoded_stage_load(proto: &skippy_stage_proto::LoadStage) {
    assert_eq!(proto.source_model_bytes, Some(123_456_789));
    assert_eq!(proto.continuous_batching, Some(false));
    assert_eq!(proto.mmap, Some(false));
    assert_eq!(proto.mlock, Some(true));
    assert_eq!(proto.projector_use_gpu, Some(false));
    assert_eq!(proto.media_marker.as_deref(), Some("<media>"));
    assert_eq!(proto.image_min_tokens, Some(32));
    assert_eq!(proto.image_max_tokens, Some(2048));
    assert_eq!(proto.batch_max_tokens, Some(512));
    assert_eq!(proto.glm_dsa_policy, 1);
    assert_eq!(proto.generation_signal_window, Some(24));
}

fn assert_decoded_stage_load(decoded: &crate::inference::skippy::StageLoadRequest) {
    assert_eq!(decoded.source_model_bytes, Some(123_456_789));
    assert!(!decoded.continuous_batching);
    assert_eq!(decoded.model_path.as_deref(), Some("/models/demo.gguf"));
    assert_eq!(decoded.mmap, Some(false));
    assert!(decoded.mlock);
    assert_eq!(decoded.projector_use_gpu, Some(false));
    assert_eq!(decoded.media_marker.as_deref(), Some("<media>"));
    assert_eq!(decoded.image_min_tokens, Some(32));
    assert_eq!(decoded.image_max_tokens, Some(2048));
    assert_eq!(decoded.batch_max_tokens, Some(512));
    assert_eq!(decoded.glm_dsa_policy, skippy_protocol::GlmDsaPolicy::V1);
    assert_eq!(decoded.generation_signal_window, Some(24));
}

#[test]
fn stage_load_proto_missing_continuous_batching_keeps_legacy_enabled_default() {
    let mut proto = skippy_stage_proto::LoadStage::default();
    proto.admission = Some(crate::inference::skippy::test_stage_admission(0, 1).into());
    let decoded = stage_load_from_proto(proto).unwrap();

    assert!(decoded.continuous_batching);
}

#[test]
fn stage_control_request_timeout_uses_stage_load_floor() {
    let mut load = stage_load_request();
    load.source_model_bytes = None;
    assert_eq!(
        Node::stage_control_request_timeout(&crate::inference::skippy::StageControlRequest::Load(
            load.clone()
        )),
        std::time::Duration::from_secs(900)
    );

    load.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    assert_eq!(
        Node::stage_control_request_timeout(&crate::inference::skippy::StageControlRequest::Load(
            load
        )),
        std::time::Duration::from_secs(1360)
    );

    let mut prepare_load = stage_load_request();
    prepare_load.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    assert_eq!(
        Node::stage_control_request_timeout(
            &crate::inference::skippy::StageControlRequest::Prepare(
                crate::inference::skippy::StagePrepareRequest {
                    load: prepare_load,
                    coordinator_id: None,
                },
            )
        ),
        std::time::Duration::from_secs(1360)
    );
}

#[test]
fn test_merge_demand_takes_max() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );
    ours.insert(
        "Hermes".into(),
        ModelDemand {
            last_active: 200,
            request_count: 10,
        },
    );

    let mut theirs = HashMap::new();
    theirs.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 150,
            request_count: 30,
        },
    );
    theirs.insert(
        "Qwen".into(),
        ModelDemand {
            last_active: 300,
            request_count: 5,
        },
    );

    merge_demand(&mut ours, &theirs);

    // GLM: max(100,150)=150 for last_active, max(50,30)=50 for count
    assert_eq!(ours["GLM"].last_active, 150);
    assert_eq!(ours["GLM"].request_count, 50);
    // Hermes: unchanged (not in theirs)
    assert_eq!(ours["Hermes"].last_active, 200);
    assert_eq!(ours["Hermes"].request_count, 10);
    // Qwen: new entry from theirs
    assert_eq!(ours["Qwen"].last_active, 300);
    assert_eq!(ours["Qwen"].request_count, 5);
}

#[test]
fn test_merge_demand_empty_maps() {
    let mut ours = HashMap::new();
    let theirs = HashMap::new();
    merge_demand(&mut ours, &theirs);
    assert!(ours.is_empty());

    let mut theirs2 = HashMap::new();
    theirs2.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 1,
        },
    );
    merge_demand(&mut ours, &theirs2);
    assert_eq!(ours.len(), 1);
    assert_eq!(ours["GLM"].request_count, 1);
}

#[test]
fn test_merge_demand_idempotent() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );

    let theirs = ours.clone();
    merge_demand(&mut ours, &theirs);

    assert_eq!(ours["GLM"].last_active, 100);
    assert_eq!(ours["GLM"].request_count, 50);
}

#[test]
fn test_demand_ttl_filtering() {
    let now = now_secs();
    let mut demand = HashMap::new();

    // Recent — should survive
    demand.insert(
        "Recent".into(),
        ModelDemand {
            last_active: now - 60, // 1 min ago
            request_count: 10,
        },
    );
    // Stale — should be filtered
    demand.insert(
        "Stale".into(),
        ModelDemand {
            last_active: now - DEMAND_TTL_SECS - 100, // past TTL
            request_count: 100,
        },
    );

    let filtered: HashMap<String, ModelDemand> = demand
        .into_iter()
        .filter(|(_, d)| (now - d.last_active) < DEMAND_TTL_SECS)
        .collect();

    assert_eq!(filtered.len(), 1);
    assert!(filtered.contains_key("Recent"));
    assert!(!filtered.contains_key("Stale"));
}

#[test]
fn test_demand_serialization_roundtrip() {
    let mut demand: HashMap<String, ModelDemand> = HashMap::new();
    demand.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 1772309000,
            request_count: 42,
        },
    );

    let json = serde_json::to_string(&demand).unwrap();
    let decoded: HashMap<String, ModelDemand> = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded["GLM"].last_active, 1772309000);
    assert_eq!(decoded["GLM"].request_count, 42);
}

#[test]
fn test_demand_deserialization_missing_field() {
    // Simulate old gossip message without model_demand field
    // Just verify ModelDemand defaults work
    let d = ModelDemand::default();
    assert_eq!(d.last_active, 0);
    assert_eq!(d.request_count, 0);

    // Verify HashMap<String, ModelDemand> defaults to empty
    let empty: HashMap<String, ModelDemand> = Default::default();
    assert!(empty.is_empty());

    // The real test: serde default on a struct with model_demand
    #[derive(Deserialize, Default)]
    struct TestStruct {
        #[serde(default)]
        model_demand: HashMap<String, ModelDemand>,
        #[serde(default)]
        requested_models: Vec<String>,
    }
    let parsed: TestStruct = serde_json::from_str("{}").unwrap();
    assert!(parsed.model_demand.is_empty());
    assert!(parsed.requested_models.is_empty());
}

#[test]
fn proto_announcement_defaults_missing_legacy_hardware_fields() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xBC; 32]).public());
    let proto = crate::proto::node::PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        ..Default::default()
    };

    let (_, decoded) = proto_ann_to_local(&proto).expect("production proto adapter must decode");

    assert_eq!(decoded.gpu_name, None);
    assert_eq!(decoded.hostname, None);
    assert_eq!(decoded.is_soc, None);
    assert_eq!(decoded.gpu_vram, None);
    assert_eq!(decoded.gpu_mem_bandwidth_gbps, None);
}

#[test]
fn peer_info_announcement_adapter_preserves_legacy_hardware_fields() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xBD; 32]).public());
    let mut peer = make_test_peer_info(peer_id);
    peer.gpu_name = Some("NVIDIA A100".to_string());
    peer.hostname = Some("worker-01".to_string());
    peer.is_soc = Some(false);
    peer.gpu_vram = Some("51539607552".to_string());
    peer.gpu_mem_bandwidth_gbps = Some("1948.70".to_string());

    let ann = Node::announcement_from_peer(&peer);
    let proto = local_ann_to_proto_ann(&ann);
    let hardware = proto.hardware.as_ref().expect("hardware must be encoded");
    let (_, decoded) = proto_ann_to_local(&proto).expect("production proto adapter must decode");

    assert_eq!(hardware.hostname.as_deref(), Some("worker-01"));
    assert_eq!(hardware.gpus[0].name.as_deref(), Some("NVIDIA A100"));
    assert_eq!(decoded.gpu_name.as_deref(), Some("NVIDIA A100"));
    assert_eq!(decoded.hostname.as_deref(), Some("worker-01"));
    assert_eq!(decoded.is_soc, Some(false));
    assert_eq!(decoded.gpu_vram.as_deref(), Some("51539607552"));
    assert_eq!(decoded.gpu_mem_bandwidth_gbps.as_deref(), Some("1948.70"));
}

#[tokio::test]
async fn local_announcement_uses_enumerate_host_for_host_fields_only() {
    let mut node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    node.gpu_name = Some("NVIDIA RTX 5090".to_string());
    node.hostname = Some("carrack".to_string());
    node.is_soc = Some(true);
    node.gpu_vram = Some("34359738368".to_string());
    *node.gpu_mem_bandwidth_gbps.lock().await = Some(vec![1792.0]);

    node.enumerate_host = false;
    let private_ann = node.build_local_announcement(node.snapshot_local_announcement_data().await);
    assert_eq!(private_ann.gpu_name, None);
    assert_eq!(private_ann.hostname, None);
    assert_eq!(private_ann.gpu_vram, None);
    assert_eq!(private_ann.is_soc, Some(true));
    assert_eq!(
        private_ann.gpu_mem_bandwidth_gbps.as_deref(),
        Some("1792.00")
    );

    node.enumerate_host = true;
    let public_ann = node.build_local_announcement(node.snapshot_local_announcement_data().await);
    assert_eq!(public_ann.gpu_name.as_deref(), Some("NVIDIA RTX 5090"));
    assert_eq!(public_ann.hostname.as_deref(), Some("carrack"));
    assert_eq!(public_ann.gpu_vram.as_deref(), Some("34359738368"));
    assert_eq!(public_ann.is_soc, Some(true));
    assert_eq!(
        public_ann.gpu_mem_bandwidth_gbps.as_deref(),
        Some("1792.00")
    );
}

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

#[test]
fn protocol_from_alpn_defaults_to_v1() {
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}

#[test]
fn identity_from_model_source_treats_absolute_gguf_as_local() {
    let identity =
        identity_from_model_source("/home/jdumay/models/smollm2-a.gguf").expect("identity");

    assert_eq!(identity.source_kind, ModelSourceKind::LocalGguf);
    assert_eq!(identity.local_file_name.as_deref(), Some("smollm2-a.gguf"));
    assert_eq!(identity.repository, None);
}

#[test]
fn parse_hf_ref_parts_rejects_absolute_paths() {
    assert!(parse_hf_ref_parts("/home/jdumay/models/smollm2-a.gguf").is_none());
}

#[test]
fn identity_from_model_source_keeps_huggingface_refs() {
    let identity =
        identity_from_model_source("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M").expect("identity");

    assert_eq!(identity.source_kind, ModelSourceKind::HuggingFace);
    assert_eq!(
        identity.canonical_ref.as_deref(),
        Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M")
    );
}

#[test]
fn control_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}
