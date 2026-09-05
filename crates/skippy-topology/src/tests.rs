use super::*;
use serde::Deserialize;

fn nodes(count: u32) -> Vec<NodeSpec> {
    (0..count)
        .map(|index| NodeSpec {
            node_id: format!("node-{index}"),
            cached_slice_bytes: 0,
            vram_bytes: 0,
        })
        .collect()
}

fn compact_identity(value: &str) -> String {
    value.to_ascii_lowercase().replace(['_', '-', '/', ' '], "")
}

fn weighted_node(node_id: &str, vram_bytes: u64) -> NodeSpec {
    NodeSpec {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        vram_bytes,
    }
}

fn placement_signal(node_id: &str) -> NodePlacementSignal {
    NodePlacementSignal {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        missing_artifact_bytes: 0,
        rtt_ms: None,
        artifact_transfer_supported: false,
        availability_score: 0,
    }
}

fn edge(source: &str, target: &str, rtt_ms: u32) -> StageEdgeSignal {
    StageEdgeSignal {
        source_node_id: source.to_string(),
        target_node_id: target.to_string(),
        rtt_ms: Some(rtt_ms),
        large_frame_bytes_per_sec: None,
        direct_prediction_return_supported: true,
    }
}

fn stage_layout(plan: &TopologyPlan) -> Vec<(&str, u32, u32)> {
    plan.stages
        .iter()
        .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
        .collect()
}

fn role_layout(plan: &TopologyPlan) -> Vec<Vec<StageRole>> {
    plan.stages
        .iter()
        .map(|stage| stage.roles.clone())
        .collect()
}

#[test]
fn transport_aware_plan_orders_same_nodes_by_stage_edge_cost() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("node-a", 30),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[],
        &[
            edge("node-a", "node-b", 200),
            edge("node-b", "node-c", 200),
            edge("node-a", "node-c", 5),
            edge("node-c", "node-b", 5),
        ],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("node-a", 0, 3), ("node-c", 3, 6), ("node-b", 6, 9)]
    );
    assert!(plan.diagnostics.iter().any(|diagnostic| diagnostic.code
        == PlanReasonCode::NetworkPipelineCost
        && diagnostic.message.contains("node-a -> node-c")));
}

#[test]
fn transport_aware_plan_orders_two_stages_by_edge_cost() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 30), weighted_node("node-b", 30)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[],
        &[edge("node-a", "node-b", 200), edge("node-b", "node-a", 5)],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("node-b", 0, 3), ("node-a", 3, 6)]
    );
}

#[test]
fn transport_aware_plan_preserves_package_order_when_edges_tie() {
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("cold-a", 30),
            weighted_node("warm", 30),
            weighted_node("cold-b", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[warm],
        &[
            edge("warm", "cold-a", 10),
            edge("warm", "cold-b", 10),
            edge("cold-a", "warm", 10),
            edge("cold-a", "cold-b", 10),
            edge("cold-b", "warm", 10),
            edge("cold-b", "cold-a", 10),
        ],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("warm", 0, 3), ("cold-a", 3, 6), ("cold-b", 6, 9)]
    );
}

#[test]
fn dense_attention_plan_allows_costed_kv_migration() {
    let request = TopologyPlanRequest {
        topology_id: "dense".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.stages.len(), 3);
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::AttentionKv)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::CostedKv)
    );
    assert!(plan.diagnostics.is_empty());
}

#[test]
fn split_topology_labels_driver_embedding_intermediate_and_readout_roles() {
    let request = TopologyPlanRequest {
        topology_id: "roles".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(9, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        role_layout(&plan),
        vec![
            vec![StageRole::Driver, StageRole::Embedding],
            vec![StageRole::Intermediate],
            vec![StageRole::Readout],
        ]
    );
}

#[test]
fn single_stage_topology_labels_combined_driver_embedding_and_readout() {
    let request = TopologyPlanRequest {
        topology_id: "single-roles".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(2, 10),
        nodes: nodes(1),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        role_layout(&plan),
        vec![vec![
            StageRole::Driver,
            StageRole::Embedding,
            StageRole::Readout,
        ]]
    );
}

#[test]
fn weighted_contiguous_plan_uses_node_vram_for_layer_spans() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 6), ("node-b", 6, 9), ("node-c", 9, 12)]
    );
}

#[test]
fn weighted_contiguous_plan_falls_back_to_even_without_weights() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 0), weighted_node("node-b", 0)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 3), ("node-b", 3, 6)]
    );
}

#[test]
fn package_aware_plan_matches_weighted_without_package_signals() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let weighted_plan = plan_weighted_contiguous(&request).expect("weighted plan");
    let package_plan =
        plan_package_aware_contiguous_with_signals(&request, &[]).expect("package-aware plan");

    assert_eq!(stage_layout(&package_plan), stage_layout(&weighted_plan));
}

#[test]
fn package_aware_plan_prefers_cached_peer_for_equal_capacity() {
    let mut cold = placement_signal("cold");
    cold.missing_artifact_bytes = 32;
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    warm.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("cold", 40), weighted_node("warm", 40)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, warm]).expect("plan");

    assert_eq!(stage_layout(&plan), vec![("warm", 0, 4), ("cold", 4, 8)]);
    assert!(
        plan.stages[0]
            .reason_codes
            .contains(&PlanReasonCode::CacheLocalityPreferred)
    );
    assert!(
        plan.stages[1]
            .reason_codes
            .contains(&PlanReasonCode::ArtifactTransferPenalty)
    );
}

#[test]
fn package_aware_plan_reports_cold_start_artifact_totals() {
    let mut transfer_ready = placement_signal("transfer-ready");
    transfer_ready.missing_artifact_bytes = 32;
    transfer_ready.artifact_transfer_supported = true;
    let mut remote_fallback = placement_signal("remote-fallback");
    remote_fallback.missing_artifact_bytes = 16;
    remote_fallback.artifact_transfer_supported = false;
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("transfer-ready", 30),
            weighted_node("remote-fallback", 30),
            weighted_node("warm", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(
        &request,
        &[transfer_ready, remote_fallback, warm],
    )
    .expect("plan");

    let diagnostic = plan
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == PlanReasonCode::ArtifactTransferPenalty)
        .expect("artifact diagnostic");
    assert!(diagnostic.message.contains("cached=30 bytes"));
    assert!(diagnostic.message.contains("missing=46 bytes"));
    assert!(
        diagnostic
            .message
            .contains("peer-transfer-eligible=30 bytes")
    );
    assert!(
        diagnostic
            .message
            .contains("remote-download-fallback=16 bytes")
    );
}

#[test]
fn weighted_plan_without_artifact_signals_stays_diagnostic_quiet() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 30), weighted_node("node-b", 30)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert!(plan.diagnostics.is_empty());
}

#[test]
fn package_aware_plan_penalizes_missing_untransferable_artifacts() {
    let mut cold = placement_signal("cold-high-vram");
    cold.missing_artifact_bytes = 64;
    cold.artifact_transfer_supported = false;
    let mut ready = placement_signal("ready-lower-vram");
    ready.cached_slice_bytes = 16;
    ready.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![
            weighted_node("cold-high-vram", 100),
            weighted_node("ready-lower-vram", 80),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, ready]).expect("plan");

    assert_eq!(plan.stages[0].node_id, "ready-lower-vram");
    assert!(
        plan.stages[1]
            .reason_codes
            .contains(&PlanReasonCode::ArtifactTransferPenalty)
    );
}

#[test]
fn package_aware_plan_treats_high_rtt_as_cost_not_exclusion() {
    let mut distant = placement_signal("distant");
    distant.rtt_ms = Some(250);
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("distant", 100), weighted_node("nearby", 80)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[distant]).expect("plan");

    assert!(plan.stages.iter().any(|stage| stage.node_id == "distant"));
    let distant_stage = plan
        .stages
        .iter()
        .find(|stage| stage.node_id == "distant")
        .expect("distant stage");
    assert!(
        distant_stage
            .reason_codes
            .contains(&PlanReasonCode::NetworkPipelineCost)
    );
}

#[test]
fn package_aware_plan_can_promote_extra_cached_peer() {
    let mut cached_extra = placement_signal("cached-extra");
    cached_extra.cached_slice_bytes = 100;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(2, 10),
        nodes: vec![
            weighted_node("cold-a", 40),
            weighted_node("cold-b", 40),
            weighted_node("cached-extra", 40),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cached_extra]).expect("plan");

    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.stages[0].node_id, "cached-extra");
    assert!(
        plan.stages
            .iter()
            .any(|stage| stage.node_id == "cold-a" || stage.node_id == "cold-b")
    );
}

#[test]
fn falcon_h1_marks_every_stage_as_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 4), (4, 6)]
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::Mixed)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner)
    );
    assert_eq!(plan.diagnostics.len(), 3);
}

#[test]
fn qwen3next_mixed_layers_only_make_recurrent_ranges_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3next".to_string(),
        model_id: "qwen3next".to_string(),
        layers: qwen3next_layers(8, [2, 3, 6], 10),
        nodes: nodes(4),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.state_affinity)
            .collect::<Vec<_>>(),
        vec![
            StateAffinity::AttentionKv,
            StateAffinity::Recurrent,
            StateAffinity::AttentionKv,
            StateAffinity::Mixed
        ]
    );
    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.migration_policy)
            .collect::<Vec<_>>(),
        vec![
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner,
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner
        ]
    );
    assert_eq!(plan.diagnostics.len(), 2);
}

#[test]
fn explicit_recurrent_transfer_policy_is_loud() {
    let request = TopologyPlanRequest {
        topology_id: "transfer".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(2, 10),
        nodes: nodes(1),
        family: None,
        policy: PlannerPolicy {
            allow_recurrent_state_transfer: true,
        },
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages[0].migration_policy,
        MigrationPolicy::RecurrentStateTransferAllowed
    );
    assert_eq!(plan.diagnostics[0].severity, DiagnosticSeverity::Warning);
}

#[test]
fn qwen3_family_uses_f32_wire_payloads() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3-wire".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(28, 10),
        nodes: nodes(2),
        family: Some(qwen3_dense_capability(28, 1024)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("qwen3_dense"));
    assert_eq!(plan.boundaries.len(), 1);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 4096);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 4096);
}

#[test]
fn accepted_dense_families_emit_exact_state_mobility_reason() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3".to_string(),
        model_id: "gemma3".to_string(),
        layers: dense_attention_layers(26, 10),
        nodes: nodes(2),
        family: Some(gemma3_capability(26, 1152)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("gemma3"));
    assert!(plan.stages.iter().all(|stage| {
        stage
            .reason_codes
            .contains(&PlanReasonCode::ExactStateMobilityAccepted)
    }));
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityAccepted)
    );
}

#[test]
fn dense_family_uses_f32_without_split_constraints() {
    let request = TopologyPlanRequest {
        topology_id: "olmo".to_string(),
        model_id: "olmo".to_string(),
        layers: dense_attention_layers(32, 10),
        nodes: nodes(2),
        family: Some(olmo_capability(32, 4096)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("olmo"));
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 16384);
}

#[test]
fn dense_family_f32_wire_bytes_are_recorded() {
    let families = [
        (gemma2_capability(26, 2304), 9216),
        (gemma3_capability(26, 1152), 4608),
        (glm4_capability(40, 4096), 16384),
    ];

    for (family, expected_f32_wire_bytes) in families {
        let request = TopologyPlanRequest {
            topology_id: family.family_id.clone(),
            model_id: family.family_id.clone(),
            layers: dense_attention_layers(family.layer_count, 10),
            nodes: nodes(2),
            family: Some(family),
            policy: PlannerPolicy::default(),
        };

        let plan = plan_even_contiguous(&request).expect("plan");
        assert_eq!(
            plan.boundaries[0].wire_payload_bytes_per_token,
            expected_f32_wire_bytes
        );
    }
}

#[test]
fn falcon_family_capability_marks_attention_layers_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon-family".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: dense_attention_layers(24, 10),
        nodes: nodes(2),
        family: Some(falcon_h1_capability(24, 2048)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::Mixed)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner)
    );
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityRejected)
    );
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::RecurrentOwnerSticky)
    );
}

#[test]
fn gemma4_e4b_accepts_validated_boundary_with_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-valid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(2),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 21);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 10240);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 10240);
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::TokenSidebandRequired)
    );
}

#[test]
fn rwkv7_boundary_accounts_for_v_first_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "rwkv7-sideband".to_string(),
        model_id: "rwkv7-191m".to_string(),
        layers: falcon_h1_layers(12, 4),
        nodes: nodes(3),
        family: Some(rwkv7_capability(12, 768)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 4);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 6144);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 6144);
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::ActivationSidebandRequired)
    );
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::RecurrentOwnerSticky)
    );
}

#[test]
fn gemma4_e4b_rejects_known_bad_shared_kv_boundaries() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-invalid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (14, BoundaryDecision::Rejected),
            (28, BoundaryDecision::Rejected)
        ]
    );
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::SharedKvRegionCut)
    );
}

#[test]
fn gemma3n_requires_altup_sideband_and_reviewed_kv_boundary() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3n".to_string(),
        model_id: "gemma3n".to_string(),
        layers: dense_attention_layers(30, 10),
        nodes: nodes(3),
        family: Some(gemma3n_capability(30, 2048)),
        policy: PlannerPolicy::default(),
    };

    let even_plan = plan_even_contiguous(&request).expect("even plan");
    assert_eq!(
        even_plan
            .boundaries
            .iter()
            .map(|boundary| {
                (
                    boundary.layer_boundary,
                    boundary.decision,
                    boundary.raw_activation_bytes_per_token,
                    boundary.wire_payload_bytes_per_token,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted, 32768, 32768),
            (20, BoundaryDecision::Rejected, 32768, 32768)
        ]
    );
    assert!(even_plan.boundaries.iter().all(|boundary| {
        boundary
            .reason_codes
            .contains(&PlanReasonCode::ActivationSidebandRequired)
    }));

    let reviewed_plan = plan_contiguous_with_splits(&request, &[10, 15]).expect("reviewed plan");
    assert_eq!(
        reviewed_plan
            .boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted),
            (15, BoundaryDecision::Accepted)
        ]
    );
}

#[test]
fn explicit_splits_return_reasoned_boundary_decisions() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-explicit".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_contiguous_with_splits(&request, &[12, 24]).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 12), (12, 24), (24, 42)]
    );
    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (12, BoundaryDecision::Rejected),
            (24, BoundaryDecision::Rejected)
        ]
    );
}

#[test]
fn infers_known_family_capabilities_from_model_identity() {
    let reviewed = reviewed_capability_records();
    assert!(reviewed.len() >= 13);

    let llama = infer_family_capability(
        "/Volumes/External/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        16,
        2048,
    )
    .expect("reviewed llama");
    assert_eq!(llama.family_id, "llama");
    assert_eq!(llama.exact_state_mobility, ExactStateMobility::Accepted);
    let reviewed_width = llama.activation_width;
    let mismatched_package_width = infer_family_capability(
        "/Volumes/External/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        16,
        reviewed_width * 4,
    )
    .expect("reviewed llama with mismatched package estimate");
    assert_eq!(
        mismatched_package_width.activation_width, reviewed_width,
        "package metadata must not replace a reviewed pre-load estimate"
    );

    let laguna = infer_family_capability(
        "poolside/Laguna-S-2.1-GGUF@edd093522473dc7313b0738d8b4116b7f8b9745f/laguna-s-2.1-Q4_K_M.gguf",
        48,
        3072,
    )
    .expect("reviewed Poolside Laguna S 2.1 Q4_K_M");
    assert_eq!(laguna.family_id, "laguna");

    let gemma4_e4b = infer_family_capability(
            "unsloth/gemma-4-E4B-it-GGUF@315e03409eb1cdde302488d66e586dea1e82aad1/gemma-4-E4B-it-Q4_K_M.gguf",
            42,
            2560,
        )
        .expect("reviewed gemma4 e4b");
    assert_eq!(gemma4_e4b.family_id, "gemma4_e4b");
    assert!(!gemma4_e4b.split_constraints.is_empty());
    assert!(!gemma4_e4b.sidebands.is_empty());

    assert_eq!(
        infer_family_capability("meshllm/gemma-4-e4b-it", 42, 2560)
            .expect("gemma")
            .family_id,
        "gemma4_e4b"
    );
    assert_eq!(
        infer_family_capability("tiiuae/Falcon-H1-1.5B", 24, 2048)
            .expect("falcon")
            .family_id,
        "falcon_h1"
    );
    assert_eq!(
        infer_family_capability("Qwen/Qwen3-Coder-Next", 48, 2048)
            .expect("qwen3next")
            .family_id,
        "qwen3next"
    );
    let inkling =
        infer_family_capability("meshllm/inkling-UD-Q2_K_XL-layers", 66, 6144).expect("inkling");
    assert_eq!(inkling.family_id, "inkling");
    assert_eq!(
        inkling.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(
        inkling.recurrent_ranges,
        vec![LayerRange { start: 0, end: 66 }]
    );

    let rwkv6 =
        infer_family_capability("latestissue/rwkv-6-finch-1b6-gguf:Q4_K", 24, 2048).expect("rwkv6");
    assert_eq!(rwkv6.family_id, "rwkv6");
    assert_eq!(
        rwkv6.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    for (identity, expected_family) in [
        ("bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M", "jamba"),
        ("meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M", "lfm2"),
        ("mradermacher/mamba-130m-hf-GGUF:Q4_K_M", "mamba"),
        ("mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M", "mamba2"),
        ("Mungert/rwkv7-191M-world-GGUF:Q4_K", "rwkv7"),
    ] {
        let capability = infer_family_capability(identity, 12, 768)
            .unwrap_or_else(|| panic!("failed to infer {identity}"));
        assert_eq!(capability.family_id, expected_family, "{identity}");
        assert!(
            !capability.recurrent_ranges.is_empty(),
            "{identity} should be treated as recurrent"
        );
    }
    for (identity, expected_family) in [
        ("mradermacher/Maincoder-1B-GGUF:Q2_K", "maincoder"),
        ("LiteLLMs/OpenELM-270M-GGUF:Q2_K", "openelm"),
        (
            "RichardErkhov/smallcloudai_-_Refact-1_6B-fim-gguf:Q2_K",
            "refact",
        ),
        ("s3nh/MiniCPM-2B-dpo-fp32-GGUF:Q3_K_S", "minicpm"),
        (
            "duyntnet/MiniCPM-3B-OpenHermes-2.5-v2-imatrix-GGUF:IQ2_XXS",
            "minicpm3",
        ),
        ("mmnga-o/plamo-3-nict-2b-base-gguf:IQ3_M", "plamo3"),
        ("StatPan/42dot_LLM-PLM-1.3B_GGUF:q3_k_m", "plm"),
        (
            "bartowski/SmallThinker-3B-Preview-GGUF:IQ2_M",
            "smallthinker",
        ),
        ("bartowski/HuggingFaceTB_SmolLM3-3B-GGUF:IQ2_M", "smollm3"),
    ] {
        let capability = infer_family_capability(identity, 24, 2048)
            .unwrap_or_else(|| panic!("failed to infer {identity}"));
        assert_eq!(capability.family_id, expected_family, "{identity}");
        assert!(
            capability.recurrent_ranges.is_empty(),
            "{identity} should be treated as dense"
        );
    }
    assert_eq!(
        infer_family_capability("Qwen/Qwen3-0.6B", 28, 1024)
            .expect("qwen3")
            .family_id,
        "qwen3_dense"
    );
    let qwen2moe = infer_family_capability("mradermacher/Qwen2-1.5B-2x-MoE-GGUF:Q4_K_S", 28, 1536)
        .expect("qwen2moe");
    assert_eq!(qwen2moe.family_id, "qwen2moe");
    let qwen3moe = infer_family_capability(
        "mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-GGUF:Q4_K_M",
        28,
        1024,
    )
    .expect("qwen3moe");
    assert_eq!(qwen3moe.family_id, "qwen3moe");
    for identity in [
        "laguna",
        "poolside/Laguna-XS-2.1-GGUF:Q4_K_M",
        "poolside/Laguna-S-2.1-GGUF:Q4_K_M",
        "poolside/Laguna-XS.2-GGUF:Q4_K_M",
        "poolside/Laguna-M.1-GGUF:Q4_K_M",
    ] {
        let laguna = infer_family_capability(identity, 48, 3072)
            .unwrap_or_else(|| panic!("failed to infer {identity}"));
        assert_eq!(laguna.family_id, "laguna", "{identity}");
        assert_eq!(laguna.exact_state_mobility, ExactStateMobility::Untested);
        assert!(laguna.recurrent_ranges.is_empty());
    }
    let openai_moe =
        infer_family_capability("ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4", 24, 2880)
            .expect("openai_moe/gpt-oss");
    assert_eq!(openai_moe.family_id, "openai_moe");
    assert_eq!(
        openai_moe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let llama4 = infer_family_capability(
        "ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M",
        48,
        5120,
    )
    .expect("llama4 package");
    assert_eq!(llama4.family_id, "llama4");
    assert_eq!(llama4.exact_state_mobility, ExactStateMobility::Untested);
    let mistral4 = infer_family_capability(
        "bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF:IQ2_XXS",
        36,
        4096,
    )
    .expect("mistral4 package");
    assert_eq!(mistral4.family_id, "mistral4");
    assert_eq!(mistral4.exact_state_mobility, ExactStateMobility::Untested);
    let qwen3_coder_package = infer_family_capability(
        "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL",
        62,
        6144,
    )
    .expect("qwen3 coder package");
    assert_eq!(qwen3_coder_package.family_id, "qwen3moe");
    let qwen3_coder_30b =
        infer_family_capability("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M", 48, 2048)
            .expect("qwen3 coder 30b");
    assert_eq!(qwen3_coder_30b.family_id, "qwen3moe");
    let exaone_moe_package =
        infer_family_capability("LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF:Q4_K_M", 49, 6144)
            .expect("exaone-moe package");
    assert_eq!(exaone_moe_package.family_id, "exaone_moe");
    assert_eq!(
        exaone_moe_package.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    let gemma3n =
        infer_family_capability("lmstudio-community/gemma-3n-E2B-it-GGUF:Q4_K_M", 30, 2048)
            .expect("gemma3n");
    assert_eq!(gemma3n.family_id, "gemma3n");
    assert_eq!(gemma3n.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(gemma3n.sidebands[0].kind, SidebandKind::Gemma3nAltup);
    let qwen2vl = infer_family_capability("bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M", 28, 1536)
        .expect("qwen2vl");
    assert_eq!(qwen2vl.family_id, "qwen2vl");
    assert_eq!(qwen2vl.exact_state_mobility, ExactStateMobility::Untested);
    let qwen3vl = infer_family_capability("Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M", 28, 2048)
        .expect("qwen3vl");
    assert_eq!(qwen3vl.family_id, "qwen3vl");
    assert_eq!(qwen3vl.exact_state_mobility, ExactStateMobility::Untested);
    let deepseek2ocr =
        infer_family_capability("ggml-org/DeepSeek-OCR-GGUF:Q8_0", 12, 1280).expect("deepseek2ocr");
    assert_eq!(deepseek2ocr.family_id, "deepseek2ocr");
    assert_eq!(
        deepseek2ocr.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let hunyuan_vl =
        infer_family_capability("ggml-org/HunyuanOCR-GGUF:Q8_0", 24, 1024).expect("hunyuan_vl");
    assert_eq!(hunyuan_vl.family_id, "hunyuan_vl");
    assert_eq!(
        hunyuan_vl.exact_state_mobility,
        ExactStateMobility::Untested
    );
    let qwen3vlmoe = infer_family_capability(
        "noctrex/Qwen3-VL-30B-A3B-Instruct-1M-MXFP4_MOE-GGUF:MXFP4_MOE",
        48,
        2048,
    )
    .expect("qwen3vlmoe");
    assert_eq!(qwen3vlmoe.family_id, "qwen3vlmoe");
    assert_eq!(
        qwen3vlmoe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let apertus =
        infer_family_capability("unsloth/Apertus-8B-Instruct-2509-GGUF:UD-IQ2_M", 32, 4096)
            .expect("apertus");
    assert_eq!(apertus.family_id, "apertus");
    let bitnet =
        infer_family_capability("Sarverott/bitnet_b1_58-large-Q4_K_M-GGUF:Q4_K_M", 24, 1536)
            .expect("bitnet");
    assert_eq!(bitnet.family_id, "bitnet");
    assert_eq!(bitnet.exact_state_mobility, ExactStateMobility::Accepted);
    let plamo =
        infer_family_capability("QuantFactory/plamo-13b-GGUF:Q2_K", 40, 5120).expect("plamo");
    assert_eq!(plamo.family_id, "plamo");
    assert_eq!(plamo.exact_state_mobility, ExactStateMobility::Accepted);
    let starcoder = infer_family_capability(
        "RichardErkhov/bigcode_-_tiny_starcoder_py-gguf:Q2_K",
        20,
        768,
    )
    .expect("starcoder");
    assert_eq!(starcoder.family_id, "starcoder");
    assert_eq!(starcoder.exact_state_mobility, ExactStateMobility::Accepted);
    let llada =
        infer_family_capability("mradermacher/LLaDA-1.5-Tiny-GGUF:Q2_K", 6, 512).expect("llada");
    assert_eq!(llada.family_id, "llada");
    assert_eq!(llada.exact_state_mobility, ExactStateMobility::Untested);
    let plamo2 = infer_family_capability("mmnga/plamo-2-1b-gguf:Q4_K_M", 16, 2048).expect("plamo2");
    assert_eq!(plamo2.family_id, "plamo2");
    assert_eq!(plamo2.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(plamo2.recurrent_ranges.len(), 1);
    let ernie4_5 =
        infer_family_capability("lmstudio-community/ERNIE-4.5-0.3B-GGUF:Q4_K_M", 18, 1024)
            .expect("ernie4_5");
    assert_eq!(ernie4_5.family_id, "ernie4_5");
    assert_eq!(ernie4_5.exact_state_mobility, ExactStateMobility::Accepted);
    let ernie4_5_moe = infer_family_capability(
        "lmstudio-community/ERNIE-4.5-21B-A3B-PT-GGUF:Q4_K_M",
        28,
        2560,
    )
    .expect("ernie4_5_moe");
    assert_eq!(ernie4_5_moe.family_id, "ernie4_5_moe");
    assert_eq!(
        ernie4_5_moe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let qwen =
        infer_family_capability("zhangtao103239/Qwen-1.8B-GGUF:q5_k_m", 24, 2048).expect("qwen");
    assert_eq!(qwen.family_id, "qwen");
    assert_eq!(qwen.exact_state_mobility, ExactStateMobility::Accepted);
    let jais = infer_family_capability("mradermacher/Jais-family-256m-GGUF:Q4_K_M", 14, 1088)
        .expect("jais");
    assert_eq!(jais.family_id, "jais");
    assert_eq!(jais.exact_state_mobility, ExactStateMobility::Accepted);
    let jais2 =
        infer_family_capability("mradermacher/JAIS2-IT-0.3-GGUF:Q4_K_M", 32, 3328).expect("jais2");
    assert_eq!(jais2.family_id, "jais2");
    assert_eq!(jais2.exact_state_mobility, ExactStateMobility::Accepted);
    let nemotron_h =
        infer_family_capability("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M", 42, 3136)
            .expect("nemotron_h");
    assert_eq!(nemotron_h.family_id, "nemotron_h");
    assert_eq!(
        nemotron_h.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(nemotron_h.recurrent_ranges.len(), 1);
    let nemotron_h_moe = infer_family_capability(
        "lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:Q4_K_M",
        52,
        2688,
    )
    .expect("nemotron_h_moe package");
    assert_eq!(nemotron_h_moe.family_id, "nemotron_h_moe");
    assert_eq!(
        nemotron_h_moe.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(nemotron_h_moe.recurrent_ranges.len(), 1);
    let llada_moe = infer_family_capability(
        "mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF:IQ2_XS",
        16,
        2048,
    )
    .expect("llada_moe");
    assert_eq!(llada_moe.family_id, "llada_moe");
    assert_eq!(llada_moe.exact_state_mobility, ExactStateMobility::Untested);
    let dream = infer_family_capability("mradermacher/DreamOn-v0-7B-i1-GGUF:IQ2_XS", 28, 3584)
        .expect("dream");
    assert_eq!(dream.family_id, "dream");
    assert_eq!(dream.exact_state_mobility, ExactStateMobility::Untested);
    let nemotron = infer_family_capability(
        "mradermacher/nemotron-3-8b-chat-4k-sft-hf-i1-GGUF:IQ2_XS",
        32,
        4096,
    )
    .expect("nemotron");
    assert_eq!(nemotron.family_id, "nemotron");
    assert_eq!(nemotron.exact_state_mobility, ExactStateMobility::Accepted);
    let seed_oss = infer_family_capability(
        "lmstudio-community/Seed-OSS-36B-Instruct-GGUF:Q4_K_M",
        64,
        5120,
    )
    .expect("seed_oss package");
    assert_eq!(seed_oss.family_id, "seed_oss");
    assert_eq!(seed_oss.exact_state_mobility, ExactStateMobility::Untested);
    let lfm2moe =
        infer_family_capability("noctrex/LFM2-8B-A1B-MXFP4_MOE-GGUF", 24, 2048).expect("lfm2moe");
    assert_eq!(lfm2moe.family_id, "lfm2moe");
    assert_eq!(lfm2moe.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(lfm2moe.recurrent_ranges.len(), 1);
    let kimi = infer_family_capability(
        "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF:IQ2_XXS",
        27,
        2304,
    )
    .expect("kimi_linear");
    assert_eq!(kimi.family_id, "kimi_linear");
    assert_eq!(
        kimi.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(
        kimi.recurrent_ranges,
        vec![
            LayerRange { start: 0, end: 3 },
            LayerRange { start: 4, end: 7 },
            LayerRange { start: 8, end: 11 },
            LayerRange { start: 12, end: 15 },
            LayerRange { start: 16, end: 19 },
            LayerRange { start: 20, end: 23 },
            LayerRange { start: 24, end: 26 },
        ]
    );
    assert_eq!(
        infer_family_capability("meta/Llama-3.2-1B-Instruct", 16, 2048)
            .expect("llama")
            .family_id,
        "llama"
    );
    assert_eq!(
        infer_family_capability("DeepSeek-Coder-V2-Lite-Instruct", 27, 2048)
            .expect("deepseek")
            .family_id,
        "deepseek2"
    );
    let deepseek3 = infer_family_capability("unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL", 61, 7168)
        .expect("reviewed deepseek3");
    assert_eq!(deepseek3.family_id, "deepseek3");
    assert_eq!(deepseek3.exact_state_mobility, ExactStateMobility::Accepted);
    let qwen35moe = infer_family_capability("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL", 40, 2048)
        .expect("reviewed qwen35moe");
    assert_eq!(qwen35moe.family_id, "qwen35moe");
    assert_eq!(
        qwen35moe.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(qwen35moe.recurrent_ranges.len(), 1);
    let generic_deepseek3 = infer_family_capability("unsloth/DeepSeek-V3.2-GGUF:Q4_K_M", 61, 7168)
        .expect("generic deepseek3");
    assert_eq!(generic_deepseek3.family_id, "deepseek3");
    assert_eq!(
        generic_deepseek3.exact_state_mobility,
        ExactStateMobility::Untested
    );
    assert_eq!(
        infer_family_capability("unsloth/GLM-4.7-Flash-GGUF", 47, 2048)
            .expect("glm47")
            .family_id,
        "glm47_flash"
    );
    assert_eq!(
        infer_family_capability("meshllm/glm-4-9b-0414", 40, 4096)
            .expect("glm4")
            .family_id,
        "glm4"
    );
    assert_eq!(
        infer_family_capability("bartowski/gemma-2-2b-it", 26, 2304)
            .expect("gemma2")
            .family_id,
        "gemma2"
    );
    assert_eq!(
        infer_family_capability("ggml-org/gemma-3-1b-it", 26, 1152)
            .expect("gemma3")
            .family_id,
        "gemma3"
    );
    let gemma =
        infer_family_capability("ggml-org/gemma-3-270m-it-GGUF:Q8_0", 18, 640).expect("gemma");
    assert_eq!(gemma.family_id, "gemma");
    assert_eq!(
        infer_family_capability("google-gemma-4-26B-A4B-it", 30, 2816)
            .expect("gemma4a4b")
            .family_id,
        "gemma4_a4b"
    );
    assert_eq!(
        infer_family_capability("meshllm/olmo-7b-instruct", 32, 4096)
            .expect("olmo")
            .family_id,
        "olmo"
    );
    assert_eq!(
        infer_family_capability("unsloth/MiniMax-M2.7-GGUF", 62, 3072)
            .expect("minimax")
            .family_id,
        "minimax_m27"
    );
    assert!(infer_family_capability("unknown", 1, 1).is_none());
    assert!(
        infer_family_capability("glm-dsa", 78, 7168).is_none(),
        "GLM-DSA must remain unadvertised until its native staged graph patch is restored"
    );
}

#[test]
fn stage_runtime_test_catalog_has_unique_architectures() {
    for (index, expected) in TEST_LLAMA_ARCHITECTURE_CATALOG.iter().enumerate() {
        assert!(
            TEST_LLAMA_ARCHITECTURE_CATALOG[index + 1..]
                .iter()
                .all(|other| other.llama_architecture != expected.llama_architecture),
            "duplicate architecture in test-only coverage catalog: {}",
            expected.llama_architecture
        );
        assert!(!expected.family_id.is_empty());
    }
}

#[test]
#[allow(deprecated)]
fn legacy_runtime_family_registry_stays_empty() {
    assert!(
        STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS.is_empty(),
        "production family rows must not regain KV or topology authority"
    );
}

#[derive(Debug, Deserialize)]
struct ParityCandidateManifest {
    candidates: Vec<ParityCandidate>,
}

#[derive(Debug, Deserialize)]
struct ParityCandidate {
    llama_model: String,
    status: String,
}

#[test]
fn parity_candidate_manifest_covers_stage_runtime_architectures() {
    let manifest: ParityCandidateManifest = serde_json::from_str(include_str!(
        "../../../docs/skippy/llama-parity-candidates.json"
    ))
    .expect("parity candidate manifest must parse");
    let candidates: Vec<String> = manifest
        .candidates
        .into_iter()
        .map(|candidate| compact_identity(&candidate.llama_model))
        .collect();

    for expected in TEST_LLAMA_ARCHITECTURE_CATALOG {
        assert!(
            candidates.contains(&compact_identity(expected.llama_architecture)),
            "missing parity candidate row for {}",
            expected.llama_architecture
        );
    }
}

#[test]
fn parity_candidate_manifest_uses_known_statuses() {
    let manifest: ParityCandidateManifest = serde_json::from_str(include_str!(
        "../../../docs/skippy/llama-parity-candidates.json"
    ))
    .expect("parity candidate manifest must parse");

    for candidate in manifest.candidates {
        assert!(
            matches!(
                candidate.status.as_str(),
                "candidate"
                    | "candidate_stateful"
                    | "candidate_multimodal"
                    | "certified"
                    | "certified_package_only"
                    | "implementation_base"
                    | "needs_candidate"
                    | "needs_runtime_slice_support"
                    | "no_public_gguf_candidate"
                    | "non_causal_aux"
                    | "package_or_remote_only"
            ),
            "{} has unknown status {}",
            candidate.llama_model,
            candidate.status
        );
    }
}

#[test]
fn reviewed_supported_families_smoke_plan_with_expected_policy_signals() {
    let reviewed = reviewed_capability_records();
    assert!(reviewed.len() >= 13);

    for record in reviewed {
        let identity = record
            .model_id
            .as_deref()
            .or(record.canonical_ref.as_deref())
            .or(record.source_repo.as_deref())
            .or(record.distribution_id.as_deref())
            .expect("reviewed family record has an identity");
        let expected = record.capability;
        let family =
            infer_family_capability(identity, expected.layer_count, expected.activation_width)
                .unwrap_or_else(|| panic!("failed to infer reviewed family for {identity}"));

        assert_eq!(
            family.family_id, expected.family_id,
            "family id mismatch for {identity}"
        );
        assert_eq!(
            family.layer_count, expected.layer_count,
            "layer count mismatch for {identity}"
        );
        assert_eq!(
            family.activation_width, expected.activation_width,
            "activation width mismatch for {identity}"
        );

        let request = TopologyPlanRequest {
            topology_id: format!("smoke-{}", family.family_id),
            model_id: identity.to_string(),
            layers: dense_attention_layers(family.layer_count, 10),
            nodes: nodes(2),
            family: Some(family.clone()),
            policy: PlannerPolicy::default(),
        };
        let plan = plan_even_contiguous(&request)
            .unwrap_or_else(|error| panic!("failed to plan {identity}: {error}"));

        assert_eq!(plan.family_id.as_deref(), Some(family.family_id.as_str()));
        assert_eq!(
            plan.stages.len(),
            2,
            "unexpected stage count for {identity}"
        );
        assert_eq!(
            plan.boundaries.len(),
            1,
            "unexpected boundary count for {identity}"
        );
        assert_eq!(
            plan.boundaries[0].wire_payload_bytes_per_token,
            plan.boundaries[0].raw_activation_bytes_per_token,
            "f32 wire payload mismatch for {identity}"
        );

        if family.recurrent_ranges.is_empty() {
            assert!(
                plan.stages.iter().all(|stage| {
                    stage.migration_policy != MigrationPolicy::StickyRecurrentOwner
                }),
                "dense family unexpectedly sticky: {identity}"
            );
        } else {
            assert!(
                plan.stages.iter().any(|stage| {
                    stage.migration_policy == MigrationPolicy::StickyRecurrentOwner
                }),
                "recurrent family was not sticky: {identity}"
            );
            assert!(
                plan.boundaries[0]
                    .reason_codes
                    .contains(&PlanReasonCode::RecurrentOwnerSticky),
                "missing recurrent boundary signal for {identity}"
            );
            match family.exact_state_mobility {
                ExactStateMobility::Accepted => assert!(
                    plan.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == PlanReasonCode::ExactStateMobilityAccepted
                    }),
                    "missing accepted recurrent state mobility diagnostic for {identity}"
                ),
                ExactStateMobility::RejectedTooLarge => assert!(
                    plan.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == PlanReasonCode::ExactStateMobilityRejected
                    }),
                    "missing rejected recurrent state mobility diagnostic for {identity}"
                ),
                ExactStateMobility::Untested => {}
            }
        }

        if !family.sidebands.is_empty() {
            let sideband_reason_codes: Vec<_> = family
                .sidebands
                .iter()
                .map(|sideband| match sideband.kind {
                    SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
                    SidebandKind::Rwkv7VFirst | SidebandKind::Gemma3nAltup => {
                        PlanReasonCode::ActivationSidebandRequired
                    }
                })
                .collect();
            assert!(
                sideband_reason_codes
                    .iter()
                    .any(|code| plan.boundaries[0].reason_codes.contains(code)),
                "missing sideband signal for {identity}"
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FrozenSupportedFamilySplit {
    docs_family: &'static str,
    family_id: &'static str,
    layer_count: u32,
    activation_width: u32,
    boundaries: Option<&'static [u32]>,
    wire_multiplier: u64,
}

const FROZEN_SUPPORTED_FAMILY_SPLITS: &[FrozenSupportedFamilySplit] = &[
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3 dense",
        family_id: "qwen3_dense",
        layer_count: 28,
        activation_width: 1024,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen2",
        family_id: "qwen2",
        layer_count: 24,
        activation_width: 896,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen2-VL",
        family_id: "qwen2vl",
        layer_count: 28,
        activation_width: 1536,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3-VL",
        family_id: "qwen3vl",
        layer_count: 28,
        activation_width: 2048,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Llama",
        family_id: "llama",
        layer_count: 16,
        activation_width: 2048,
        boundaries: Some(&[5, 10]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "DeepSeek2",
        family_id: "deepseek2",
        layer_count: 27,
        activation_width: 2048,
        boundaries: Some(&[7, 14]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "DeepSeek LLM",
        family_id: "deepseek",
        layer_count: 30,
        activation_width: 4096,
        boundaries: Some(&[10, 20]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "DeepSeek3",
        family_id: "deepseek3",
        layer_count: 61,
        activation_width: 7168,
        boundaries: None,
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "GLM-4.7 Flash",
        family_id: "glm47_flash",
        layer_count: 47,
        activation_width: 2048,
        boundaries: Some(&[15, 31]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "GLM4-MoE",
        family_id: "glm4_moe",
        layer_count: 47,
        activation_width: 2048,
        boundaries: Some(&[15, 31]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "GLM4 9B",
        family_id: "glm4",
        layer_count: 40,
        activation_width: 4096,
        boundaries: Some(&[13, 27]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Baichuan",
        family_id: "baichuan",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Bloom",
        family_id: "bloom",
        layer_count: 24,
        activation_width: 1024,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "GPT2",
        family_id: "gpt2",
        layer_count: 12,
        activation_width: 768,
        boundaries: Some(&[4, 8]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "GPT-NeoX",
        family_id: "gptneox",
        layer_count: 6,
        activation_width: 512,
        boundaries: Some(&[2, 4]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Gemma4 A4B",
        family_id: "gemma4_a4b",
        layer_count: 30,
        activation_width: 2816,
        boundaries: Some(&[8, 15]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Gemma4 E4B",
        family_id: "gemma4_e4b",
        layer_count: 42,
        activation_width: 2560,
        boundaries: Some(&[21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Gemma3",
        family_id: "gemma3",
        layer_count: 26,
        activation_width: 1152,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Gemma3n",
        family_id: "gemma3n",
        layer_count: 30,
        activation_width: 2048,
        boundaries: Some(&[10, 15]),
        wire_multiplier: 4,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Gemma2",
        family_id: "gemma2",
        layer_count: 26,
        activation_width: 2304,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Phi2",
        family_id: "phi2",
        layer_count: 32,
        activation_width: 2560,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Granite",
        family_id: "granite",
        layer_count: 40,
        activation_width: 2048,
        boundaries: Some(&[13, 26]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Granite-Hybrid",
        family_id: "granite_hybrid",
        layer_count: 32,
        activation_width: 768,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Granite-MoE",
        family_id: "granite_moe",
        layer_count: 6,
        activation_width: 64,
        boundaries: Some(&[2, 4]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Hunyuan-Dense",
        family_id: "hunyuan_dense",
        layer_count: 32,
        activation_width: 2048,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Hunyuan-MoE",
        family_id: "hunyuan_moe",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Hunyuan-VL / HunyuanOCR",
        family_id: "hunyuan_vl",
        layer_count: 24,
        activation_width: 1024,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "LFM2",
        family_id: "lfm2",
        layer_count: 16,
        activation_width: 1024,
        boundaries: Some(&[5, 10]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Jamba",
        family_id: "jamba",
        layer_count: 28,
        activation_width: 2560,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Kimi Linear",
        family_id: "kimi_linear",
        layer_count: 27,
        activation_width: 2304,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Laguna S 2.1",
        family_id: "laguna",
        layer_count: 48,
        activation_width: 3072,
        boundaries: Some(&[25, 39]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Mamba",
        family_id: "mamba",
        layer_count: 24,
        activation_width: 768,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Mamba2",
        family_id: "mamba2",
        layer_count: 64,
        activation_width: 2560,
        boundaries: Some(&[21, 42]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "RWKV6",
        family_id: "rwkv6",
        layer_count: 24,
        activation_width: 2048,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "RWKV7",
        family_id: "rwkv7",
        layer_count: 12,
        activation_width: 768,
        boundaries: Some(&[4, 8]),
        wire_multiplier: 2,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Falcon-H1",
        family_id: "falcon_h1",
        layer_count: 24,
        activation_width: 2048,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Falcon",
        family_id: "falcon",
        layer_count: 32,
        activation_width: 4544,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "InternLM2",
        family_id: "internlm2",
        layer_count: 24,
        activation_width: 2048,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Phi3",
        family_id: "phi",
        layer_count: 32,
        activation_width: 3072,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "PhiMoE",
        family_id: "phimoe",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "OLMo",
        family_id: "olmo",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "OLMo2",
        family_id: "olmo2",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "OLMoE",
        family_id: "olmoe",
        layer_count: 16,
        activation_width: 2048,
        boundaries: Some(&[5, 10]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Mistral3",
        family_id: "mistral",
        layer_count: 26,
        activation_width: 3072,
        boundaries: Some(&[8, 17]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen2-MoE",
        family_id: "qwen2moe",
        layer_count: 28,
        activation_width: 1536,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3-MoE",
        family_id: "qwen3moe",
        layer_count: 28,
        activation_width: 1024,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3-VL-MoE",
        family_id: "qwen3vlmoe",
        layer_count: 48,
        activation_width: 2048,
        boundaries: Some(&[16, 32]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "EXAONE",
        family_id: "exaone",
        layer_count: 30,
        activation_width: 2560,
        boundaries: Some(&[10, 20]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "EXAONE4",
        family_id: "exaone4",
        layer_count: 30,
        activation_width: 2048,
        boundaries: Some(&[10, 20]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "EXAONE-MoE",
        family_id: "exaone_moe",
        layer_count: 49,
        activation_width: 6144,
        boundaries: None,
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Cohere2",
        family_id: "cohere2",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "MiniMax M2.7",
        family_id: "minimax_m27",
        layer_count: 62,
        activation_width: 3072,
        boundaries: Some(&[20, 41]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3Next",
        family_id: "qwen3next",
        layer_count: 48,
        activation_width: 2048,
        boundaries: Some(&[16, 32]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Arcee",
        family_id: "arcee",
        layer_count: 36,
        activation_width: 2560,
        boundaries: Some(&[12, 24]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "ChatGLM",
        family_id: "chatglm",
        layer_count: 28,
        activation_width: 4096,
        boundaries: Some(&[9, 18]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "CodeShell",
        family_id: "codeshell",
        layer_count: 42,
        activation_width: 4096,
        boundaries: Some(&[14, 28]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Deci",
        family_id: "deci",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Qwen3.5 recurrent",
        family_id: "qwen35",
        layer_count: 32,
        activation_width: 2560,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "XVerse",
        family_id: "xverse",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Maincoder",
        family_id: "maincoder",
        layer_count: 32,
        activation_width: 1536,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "OpenELM",
        family_id: "openelm",
        layer_count: 16,
        activation_width: 1280,
        boundaries: Some(&[5, 10]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "MiniCPM",
        family_id: "minicpm",
        layer_count: 40,
        activation_width: 2304,
        boundaries: Some(&[13, 26]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "MiniCPM3",
        family_id: "minicpm3",
        layer_count: 40,
        activation_width: 2304,
        boundaries: Some(&[13, 26]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Plamo3",
        family_id: "plamo3",
        layer_count: 24,
        activation_width: 2560,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "PLM",
        family_id: "plm",
        layer_count: 24,
        activation_width: 2048,
        boundaries: Some(&[8, 16]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "Refact",
        family_id: "refact",
        layer_count: 32,
        activation_width: 2048,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "SmallThinker",
        family_id: "smallthinker",
        layer_count: 36,
        activation_width: 2048,
        boundaries: Some(&[12, 24]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "SmolLM3",
        family_id: "smollm3",
        layer_count: 36,
        activation_width: 2048,
        boundaries: Some(&[12, 24]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "StableLM",
        family_id: "stablelm",
        layer_count: 32,
        activation_width: 2560,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "StarCoder2",
        family_id: "starcoder2",
        layer_count: 30,
        activation_width: 3072,
        boundaries: Some(&[10, 20]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "MPT",
        family_id: "mpt",
        layer_count: 32,
        activation_width: 4096,
        boundaries: Some(&[10, 21]),
        wire_multiplier: 1,
    },
    FrozenSupportedFamilySplit {
        docs_family: "DeepSeek-OCR",
        family_id: "deepseek2ocr",
        layer_count: 12,
        activation_width: 1280,
        boundaries: Some(&[4, 8]),
        wire_multiplier: 1,
    },
];

fn reviewed_capability_for_frozen_family(
    spec: FrozenSupportedFamilySplit,
) -> FamilyCapabilityRecord {
    let mut matches = reviewed_capability_records().into_iter().filter(|record| {
        record.capability.family_id == spec.family_id
            && record.capability.layer_count == spec.layer_count
            && record.capability.activation_width == spec.activation_width
    });
    let record = matches.next().unwrap_or_else(|| {
        panic!(
            "missing reviewed capability for {} ({})",
            spec.docs_family, spec.family_id
        )
    });
    assert!(
        matches.next().is_none(),
        "multiple reviewed capabilities for {} ({})",
        spec.docs_family,
        spec.family_id
    );
    record.capability
}

#[test]
fn frozen_supported_family_split_contract_matches_planner() {
    assert_eq!(FROZEN_SUPPORTED_FAMILY_SPLITS.len(), 72);
    let mut explicit_boundary_rows = 0;
    let mut no_boundary_rows = 0;

    for spec in FROZEN_SUPPORTED_FAMILY_SPLITS {
        let family = reviewed_capability_for_frozen_family(*spec);
        let request = TopologyPlanRequest {
            topology_id: format!("frozen-{}", spec.family_id),
            model_id: spec.family_id.to_string(),
            layers: dense_attention_layers(spec.layer_count, 10),
            nodes: nodes(3),
            family: Some(family),
            policy: PlannerPolicy::default(),
        };

        let Some(boundaries) = spec.boundaries else {
            no_boundary_rows += 1;
            let plan = plan_contiguous_with_splits(&request, &[])
                .unwrap_or_else(|error| panic!("{} no-cut plan failed: {error}", spec.docs_family));
            assert_eq!(plan.stages.len(), 1, "{}", spec.docs_family);
            assert_eq!(
                (plan.stages[0].layer_start, plan.stages[0].layer_end),
                (0, spec.layer_count)
            );
            assert!(
                plan.boundaries.is_empty(),
                "{} invented a split",
                spec.docs_family
            );
            continue;
        };

        explicit_boundary_rows += 1;
        let plan = plan_contiguous_with_splits(&request, boundaries)
            .unwrap_or_else(|error| panic!("{} split plan failed: {error}", spec.docs_family));
        assert_eq!(
            plan.family_id.as_deref(),
            Some(spec.family_id),
            "{}",
            spec.docs_family
        );
        let mut expected_edges = Vec::with_capacity(boundaries.len() + 1);
        expected_edges.push(0);
        expected_edges.extend_from_slice(boundaries);
        expected_edges.push(spec.layer_count);
        let expected_ranges = expected_edges
            .windows(2)
            .map(|edge| (edge[0], edge[1]))
            .collect::<Vec<_>>();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            expected_ranges,
            "{} stage ranges",
            spec.docs_family
        );
        assert!(
            plan.boundaries
                .iter()
                .all(|boundary| boundary.decision == BoundaryDecision::Accepted),
            "{} has an unexpected rejected documented boundary",
            spec.docs_family
        );
        let expected_payload_bytes = u64::from(spec.activation_width) * 4 * spec.wire_multiplier;
        assert!(
            plan.boundaries.iter().all(|boundary| {
                boundary.raw_activation_bytes_per_token == expected_payload_bytes
                    && boundary.wire_payload_bytes_per_token == expected_payload_bytes
            }),
            "{} F32 payload bytes",
            spec.docs_family
        );
    }

    assert_eq!(explicit_boundary_rows, 70);
    assert_eq!(no_boundary_rows, 2);
}

#[test]
fn frozen_supported_family_split_exceptions_remain_enforced() {
    let gemma4_spec = FROZEN_SUPPORTED_FAMILY_SPLITS
        .iter()
        .find(|spec| spec.family_id == "gemma4_e4b")
        .copied()
        .expect("Gemma4 E4B contract");
    let gemma4_request = TopologyPlanRequest {
        topology_id: "frozen-gemma4-e4b-invalid".to_string(),
        model_id: gemma4_spec.family_id.to_string(),
        layers: dense_attention_layers(gemma4_spec.layer_count, 10),
        nodes: nodes(2),
        family: Some(reviewed_capability_for_frozen_family(gemma4_spec)),
        policy: PlannerPolicy::default(),
    };
    for cut in [12, 14, 24, 28] {
        let plan = plan_contiguous_with_splits(&gemma4_request, &[cut]).expect("Gemma4 plan");
        assert_eq!(
            plan.boundaries[0].decision,
            BoundaryDecision::Rejected,
            "cut {cut}"
        );
        assert!(
            plan.boundaries[0]
                .reason_codes
                .contains(&PlanReasonCode::SharedKvRegionCut)
        );
    }

    let gemma3n_spec = FROZEN_SUPPORTED_FAMILY_SPLITS
        .iter()
        .find(|spec| spec.family_id == "gemma3n")
        .copied()
        .expect("Gemma3n contract");
    let gemma3n_request = TopologyPlanRequest {
        topology_id: "frozen-gemma3n".to_string(),
        model_id: gemma3n_spec.family_id.to_string(),
        layers: dense_attention_layers(gemma3n_spec.layer_count, 10),
        nodes: nodes(3),
        family: Some(reviewed_capability_for_frozen_family(gemma3n_spec)),
        policy: PlannerPolicy::default(),
    };
    let reviewed_chain =
        plan_contiguous_with_splits(&gemma3n_request, &[10, 15]).expect("Gemma3n chain");
    assert_eq!(
        reviewed_chain
            .stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 10), (10, 15), (15, 30)]
    );
    assert!(reviewed_chain.boundaries.iter().all(|boundary| {
        boundary.decision == BoundaryDecision::Accepted
            && boundary
                .reason_codes
                .contains(&PlanReasonCode::ActivationSidebandRequired)
            && boundary.raw_activation_bytes_per_token == 32_768
            && boundary.wire_payload_bytes_per_token == 32_768
    }));
    let rejected_final_slice =
        plan_contiguous_with_splits(&gemma3n_request, &[10, 20]).expect("Gemma3n rejected chain");
    assert_eq!(
        rejected_final_slice.boundaries[0].decision,
        BoundaryDecision::Accepted
    );
    assert_eq!(
        rejected_final_slice.boundaries[1].decision,
        BoundaryDecision::Rejected
    );
    assert!(
        rejected_final_slice.boundaries[1]
            .reason_codes
            .contains(&PlanReasonCode::SharedKvRegionCut)
    );
}

#[test]
fn qwen35_series_inference_covers_qwen36_release_names() {
    // Qwen3.6 and Qwen3.8 load as llama.cpp `qwen35`/`qwen35moe`; there is no
    // `qwen36` or `qwen38` arch.
    // Every quant and uploader must resolve to the recurrent series, not qwen3moe.
    for identity in [
        "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        "bartowski/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        "Qwen/Qwen3.6-35B-A3B-Instruct-GGUF:Q5_K_M",
        "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-Q1_0",
        "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-IQ2_XXS",
        "meshllm/Qwen3.8-2.4T-A95B-UD-Q1_0-layers",
        "qwen35moe",
        "qwen36moe",
        "qwen38moe",
    ] {
        let family = infer_family_capability(identity, 40, 2048)
            .unwrap_or_else(|| panic!("expected qwen35moe capability for {identity}"));
        assert_eq!(family.family_id, "qwen35moe", "wrong family for {identity}");
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 40 }],
            "qwen35moe must expose a recurrent range for {identity}"
        );
        assert_eq!(
            family.exact_state_mobility,
            ExactStateMobility::RejectedTooLarge,
            "qwen35moe full-state handoff must stay rejected for {identity}"
        );
    }

    for identity in [
        "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
        "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        "qwen35",
        "qwen36",
        "qwen38",
    ] {
        let family = infer_family_capability(identity, 32, 2560)
            .unwrap_or_else(|| panic!("expected qwen35 capability for {identity}"));
        assert_eq!(family.family_id, "qwen35", "wrong family for {identity}");
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 32 }],
            "qwen35 must expose a recurrent range for {identity}"
        );
    }
}

#[test]
fn qwen4exp_flash_next_has_its_own_fail_closed_hybrid_policy() {
    for identity in [
        "qwen4exp",
        "qwen4_exp",
        "Qwen/Qwen3.8-Flash-Next",
        "unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S",
    ] {
        let family = infer_family_capability(identity, 48, 2560)
            .unwrap_or_else(|| panic!("expected qwen4exp capability for {identity}"));
        assert_eq!(family.family_id, "qwen4exp", "wrong family for {identity}");
        assert_eq!(family.activation_width, 2560);
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 48 }],
            "QWEN4EXP state ownership stays sticky until per-layer certification"
        );
        assert_eq!(
            family.exact_state_mobility,
            ExactStateMobility::RejectedTooLarge,
            "QWEN4EXP must not advertise recurrent/indexer state mobility"
        );
        assert_eq!(family.sidebands.len(), 1);
        assert_eq!(family.sidebands[0].kind, SidebandKind::TokenIds);
        assert_eq!(family.sidebands[0].first_required_layer, 1);
    }

    let legacy = infer_family_capability("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", 32, 2560)
        .expect("legacy Qwen3.8 capability");
    assert_eq!(legacy.family_id, "qwen35");
}

#[test]
fn unknown_qwen3_point_releases_resolve_to_no_family() {
    // A Qwen3 point release we have no evidence for must not be guessed into
    // the non-recurrent `qwen3moe`/`qwen3_dense` families. Qwen3.5, 3.6 and
    // 3.8 are all hybrid, so a wrong guess advertises a non-recurrent policy
    // for a probably-recurrent model. No capability is the safe answer: it
    // surfaces the gap at onboarding instead of at runtime.
    for identity in [
        "Qwen/Qwen3.9-40B-A3B-GGUF:Q4_K_M",
        "unsloth/Qwen3.7-27B-GGUF:Q4_K_M",
        "qwen39",
        "qwen3.7",
        // A dotted release whose version is not a single digit must not be
        // narrowed to its first digit: `Qwen3.50` is not `Qwen3.5`.
        "Qwen/Qwen3.50-40B-A3B-GGUF:Q4_K_M",
        "Qwen/Qwen3.58-40B-GGUF:Q4_K_M",
        "qwen3.50",
    ] {
        assert!(
            infer_family_capability(identity, 40, 2048).is_none(),
            "{identity} must not resolve to a guessed family"
        );
    }
}

#[test]
fn qwen3_parameter_sizes_are_not_mistaken_for_qwen35_series() {
    // `Qwen3-5B` compacts to `qwen35b`: the digit is a parameter count, not a
    // series number, so these must stay on the non-recurrent Qwen3 families.
    for (identity, expected) in [
        ("Qwen/Qwen3-5B-GGUF:Q4_K_M", "qwen3_dense"),
        ("Qwen/Qwen3-6B-GGUF:Q4_K_M", "qwen3_dense"),
        ("Qwen/Qwen3-8B-GGUF:Q4_K_M", "qwen3_dense"),
        // Fractional sizes compact to `qwen30.6b`: the digit after the dot is
        // a size, not a point release.
        ("Qwen/Qwen3-0.6B-GGUF:Q8_0", "qwen3_dense"),
        // Multi-digit runs are parameter counts, not point releases.
        ("Qwen/Qwen3-235B-A22B-GGUF:Q4_K_M", "qwen3moe"),
        ("Qwen/Qwen3-0.6B:Q8_0", "qwen3_dense"),
        ("Qwen/Qwen3-35B-A3B-GGUF:Q4_K_M", "qwen3moe"),
        ("Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M", "qwen3moe"),
    ] {
        let family = infer_family_capability(identity, 40, 2048)
            .unwrap_or_else(|| panic!("expected capability for {identity}"));
        assert_eq!(family.family_id, expected, "wrong family for {identity}");
        assert!(
            family.recurrent_ranges.is_empty(),
            "{identity} must not be treated as recurrent"
        );
    }
}
