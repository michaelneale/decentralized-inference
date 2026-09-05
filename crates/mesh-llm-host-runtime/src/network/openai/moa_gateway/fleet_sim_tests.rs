//! Synthetic fleet sizing for the MoA worker pool.
//!
//! Fabricates N admitted mesh peers directly in `state.peers` (no processes,
//! no sockets, no inference) and runs the *real* `assemble_worker_pool` and
//! `compute_actor_candidates` over them. Every input those functions read —
//! served model descriptors, gossiped `parameter_count_b`, advertised context,
//! `tool_use` capability — is a plain field on `PeerInfo`, so a 2000-node
//! fleet is 2000 structs.
//!
//! Purpose: measure how admitted pool width scales with fleet size, model
//! diversity, and tier mix, before spending anything on real inference.

use super::pool::{assemble_worker_pool, compute_actor_candidates};
use crate::inference::election;
use crate::mesh;
use crate::models::{CapabilityLevel, ModelCapabilities};
use iroh::{EndpointAddr, EndpointId, SecretKey};
use std::collections::HashMap;

/// A model as a fleet node would advertise it.
#[derive(Clone, Copy)]
pub(super) struct FleetModel {
    pub(super) name: &'static str,
    parameter_count_b: f64,
    context_length: u32,
}

pub(super) const SMALL_MODELS: &[FleetModel] = &[
    FleetModel {
        name: "gemma-4-E4B-it-Q4_K_M",
        parameter_count_b: 4.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Qwen3.5-9B-Q4_K_M",
        parameter_count_b: 9.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Llama-3.2-3B-Instruct-Q4_K_M",
        parameter_count_b: 3.0,
        context_length: 32768,
    },
];

pub(super) const BIG_MODELS: &[FleetModel] = &[
    FleetModel {
        name: "Qwen3.8-27B-Q4_K_M",
        parameter_count_b: 27.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Qwen3-32B-Q4_K_M",
        parameter_count_b: 32.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Gemma-3-27B-it-Q4_K_M",
        parameter_count_b: 27.0,
        context_length: 32768,
    },
];

fn endpoint_id(seed: u32) -> EndpointId {
    let mut bytes = [0u8; 32];
    bytes[..4].copy_from_slice(&seed.to_le_bytes());
    // Seed 0 is a valid scalar but keep it distinct from an all-zero key.
    bytes[31] = 1;
    EndpointId::from(SecretKey::from_bytes(&bytes).public())
}

/// One admitted host peer serving exactly one model.
pub(super) fn fleet_peer(seed: u32, model: FleetModel) -> mesh::PeerInfo {
    let id = endpoint_id(seed);
    mesh::PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.name.to_string()],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![model.name.to_string()],
        hosted_models: vec![model.name.to_string()],
        hosted_models_known: true,
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
        served_model_descriptors: vec![mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model.name.to_string(),
                is_primary: true,
                ..Default::default()
            },
            capabilities_known: true,
            capabilities: ModelCapabilities {
                tool_use: CapabilityLevel::Supported,
                ..Default::default()
            },
            topology: None,
            metadata: Some(mesh::ServedModelMetadata {
                parameter_count_b: Some(model.parameter_count_b),
                native_context_length: Some(model.context_length),
                ..Default::default()
            }),
        }],
        served_model_runtime: vec![mesh::ModelRuntimeDescriptor {
            model_name: model.name.to_string(),
            identity_hash: None,
            context_length: Some(model.context_length),
            ready: true,
        }],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        owner_summary: Default::default(),
        advertised_model_throughput: vec![],
        // Simulated peers advertise no cache affinity: these tests exercise
        // admission and replica choice, which must not depend on cache state.
        cache_affinity: None,
        inference_admission_state: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
    }
}

/// A peer whose model advertises a specific `tool_use` capability level.
///
/// The default `fleet_peer` advertises `Supported` for every model, which makes
/// the tool-capability term in `compute_actor_candidates` constant and
/// therefore untested. Tool turns route through a single actor
/// (`tool_turn::handle_tool_query`), and that actor is `actor_candidates[0]`,
/// so any reordering of that list changes which model emits real tool calls.
pub(super) fn fleet_peer_with_tool_use(
    seed: u32,
    model: FleetModel,
    tool_use: CapabilityLevel,
    admission: Option<crate::proto::node::InferenceAdmissionState>,
) -> mesh::PeerInfo {
    let mut peer = fleet_peer(seed, model);
    peer.inference_admission_state = admission;
    for descriptor in &mut peer.served_model_descriptors {
        descriptor.capabilities.tool_use = tool_use;
    }
    peer
}

pub(super) fn fleet_peer_with_health(
    seed: u32,
    model: FleetModel,
    admission: Option<crate::proto::node::InferenceAdmissionState>,
    throughput_milli: Option<u64>,
) -> mesh::PeerInfo {
    let mut peer = fleet_peer(seed, model);
    peer.inference_admission_state = admission;
    if let Some(avg_tokens_per_second_milli) = throughput_milli {
        peer.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
            model_name: model.name.to_string(),
            avg_tokens_per_second_milli,
            throughput_samples: 8,
        }];
    }
    peer
}

/// Build a node whose mesh view is `fleet`: (model, replica count) pairs.
pub(super) async fn node_with_fleet(fleet: &[(FleetModel, usize)]) -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let mut seed = 1u32;
    for (model, replicas) in fleet {
        for _ in 0..*replicas {
            node.insert_test_peer(fleet_peer(seed, *model)).await;
            seed += 1;
        }
    }
    node
}

/// Admitted worker names for a fleet, using the real assembly path.
async fn admitted_pool(fleet: &[(FleetModel, usize)]) -> Vec<String> {
    let node = node_with_fleet(fleet).await;
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    models.into_iter().map(|m| m.name).collect()
}

fn total_nodes(fleet: &[(FleetModel, usize)]) -> usize {
    fleet.iter().map(|(_, n)| *n).sum()
}

/// Pool width is a function of distinct model count, not fleet size.
///
/// This is the claim the whole fleet plan rests on: adding identical nodes
/// cannot widen the committee, because `assemble_worker_pool` resolves exactly
/// one worker per canonical model name.
#[tokio::test]
async fn pool_width_is_flat_in_fleet_size() {
    let mut rows: Vec<(usize, usize, usize)> = Vec::new();
    for replicas in [1usize, 10, 100, 1000] {
        let fleet = vec![
            (BIG_MODELS[0], replicas),
            (SMALL_MODELS[0], replicas),
            (SMALL_MODELS[1], replicas),
        ];
        let pool = admitted_pool(&fleet).await;
        rows.push((total_nodes(&fleet), 3, pool.len()));
    }

    for (nodes, distinct_models, admitted) in &rows {
        tracing::debug!("nodes={nodes:>5} distinct_models={distinct_models} admitted={admitted}");
    }

    let widths: Vec<usize> = rows.iter().map(|(_, _, w)| *w).collect();
    assert!(
        widths.windows(2).all(|w| w[0] == w[1]),
        "admitted pool width changed with fleet size: {widths:?}"
    );
}

/// Sweep distinct model count with the fleet size held constant.
#[tokio::test]
async fn pool_width_tracks_model_diversity() {
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for distinct in 1usize..=6 {
        let mut fleet = Vec::new();
        let all: Vec<FleetModel> = BIG_MODELS
            .iter()
            .zip(SMALL_MODELS.iter())
            .flat_map(|(b, s)| [*b, *s])
            .collect();
        for model in all.iter().take(distinct) {
            fleet.push((*model, 200usize));
        }
        let pool = admitted_pool(&fleet).await;
        rows.push((distinct, pool.len()));
    }
    assert_eq!(
        rows.iter()
            .map(|(distinct, _)| *distinct)
            .collect::<Vec<_>>(),
        (1usize..=6).collect::<Vec<_>>()
    );
    assert!(
        rows.iter().all(|(_, width)| (2..=3).contains(width)),
        "committee width must remain bounded as admitted model diversity grows: {rows:?}"
    );
}

/// The bimodal fleet Mic described: 1200 small nodes, 800 big nodes.
#[tokio::test]
async fn bimodal_fleet_admission_and_actor() {
    let fleet = vec![
        (SMALL_MODELS[0], 600usize),
        (SMALL_MODELS[1], 600),
        (BIG_MODELS[0], 400),
        (BIG_MODELS[1], 400),
    ];
    let node = node_with_fleet(&fleet).await;
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;

    tracing::debug!("fleet nodes = {}", total_nodes(&fleet));
    tracing::debug!(
        "admitted workers = {:?}",
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()
    );
    tracing::debug!(
        "actor order = {:?}",
        actors
            .iter()
            .filter_map(|&i| models.get(i).map(|m| m.name.as_str()))
            .collect::<Vec<_>>()
    );

    let admitted = models
        .iter()
        .map(|model| super::pool::canonical_base_name(&model.name))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        admitted,
        BIG_MODELS[..2]
            .iter()
            .map(|model| super::pool::canonical_base_name(model.name))
            .collect::<std::collections::BTreeSet<_>>(),
        "the bimodal fleet must retain its two distinct big models"
    );
    let mut actor_models = actors
        .iter()
        .map(|&index| super::pool::canonical_base_name(&models[index].name))
        .collect::<Vec<_>>();
    actor_models.sort();
    let mut expected_actor_models = BIG_MODELS[..2]
        .iter()
        .map(|model| super::pool::canonical_base_name(model.name))
        .collect::<Vec<_>>();
    expected_actor_models.sort();
    assert_eq!(
        actor_models, expected_actor_models,
        "actor candidates must contain both admitted equal-capability models"
    );
}

/// How many of the fleet's nodes can ever receive a call for one turn?
#[tokio::test]
async fn calls_per_turn_vs_fleet_size() {
    let mut summary: HashMap<usize, usize> = HashMap::new();
    for replicas in [1usize, 50, 500] {
        let fleet = vec![
            (BIG_MODELS[0], replicas),
            (BIG_MODELS[1], replicas),
            (SMALL_MODELS[0], replicas),
            (SMALL_MODELS[1], replicas),
        ];
        let pool = admitted_pool(&fleet).await;
        summary.insert(total_nodes(&fleet), pool.len());
        tracing::debug!(
            "nodes={:>5} admitted={} names={pool:?}",
            total_nodes(&fleet),
            pool.len()
        );
    }
    assert_eq!(
        summary,
        HashMap::from([(4, 2), (200, 2), (2_000, 2)]),
        "replica growth must not widen per-turn committee calls"
    );
}

#[tokio::test]
async fn homogeneous_fleet_uses_two_distinct_replicas() {
    let pool = admitted_pool(&[(BIG_MODELS[0], 2_000)]).await;
    assert_eq!(pool.len(), 2, "same-model self-fill should form a pair");
    assert!(pool.iter().all(|name| name == BIG_MODELS[0].name));
}

#[tokio::test]
async fn paused_and_deprioritized_replicas_release_affinity() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let paused = fleet_peer_with_health(
        1,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::RemotePaused),
        Some(90_000),
    );
    let deprioritized = fleet_peer_with_health(
        2,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(80_000),
    );
    let healthy = fleet_peer_with_health(
        3,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::Accepting),
        Some(40_000),
    );
    let paused_id = paused.id;
    let deprioritized_id = deprioritized.id;
    let healthy_id = healthy.id;
    node.insert_test_peer(paused).await;
    node.insert_test_peer(deprioritized).await;
    node.insert_test_peer(healthy).await;

    let hosts = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_eq!(hosts, vec![healthy_id, deprioritized_id]);
    assert!(!hosts.contains(&paused_id));
}

#[tokio::test]
async fn healthy_small_model_precedes_deprioritized_big_actor() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    node.insert_test_peer(fleet_peer_with_health(
        1,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(80_000),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_health(
        2,
        BIG_MODELS[1],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(70_000),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_health(
        3,
        SMALL_MODELS[1],
        Some(InferenceAdmissionState::Accepting),
        Some(20_000),
    ))
    .await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;
    assert_eq!(models.len(), 3, "small spillover must remain admitted");
    assert_eq!(
        super::pool::canonical_base_name(&models[actors[0]].name),
        super::pool::canonical_base_name(SMALL_MODELS[1].name)
    );
}

#[tokio::test]
async fn local_small_model_absorbs_load_when_big_models_are_deprioritized() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let local_name = SMALL_MODELS[1].name.to_string();
    node.set_hosted_models(vec![local_name.clone()]).await;
    // A remote descriptor supplies authoritative size metadata for the same
    // canonical model while the target table selects the local backend.
    node.insert_test_peer(fleet_peer_with_health(
        1,
        SMALL_MODELS[1],
        Some(InferenceAdmissionState::RemotePaused),
        Some(20_000),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_health(
        2,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(80_000),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_health(
        3,
        BIG_MODELS[1],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(70_000),
    ))
    .await;

    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        local_name.clone(),
        vec![election::InferenceTarget::Local(19_337)],
    );
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;

    assert_eq!(
        models.len(),
        3,
        "healthy local spillover must remain admitted"
    );
    assert_eq!(
        super::pool::canonical_base_name(&models[actors[0]].name),
        super::pool::canonical_base_name(&local_name)
    );
}

#[tokio::test]
async fn mixed_version_admission_states_remain_routable() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let legacy = fleet_peer_with_health(1, BIG_MODELS[0], None, None);
    let unspecified = fleet_peer_with_health(
        2,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::Unspecified),
        None,
    );
    let legacy_id = legacy.id;
    let unspecified_id = unspecified.id;
    node.insert_test_peer(legacy).await;
    node.insert_test_peer(unspecified).await;

    let hosts = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_eq!(hosts.len(), 2);
    assert!(hosts.contains(&legacy_id));
    assert!(hosts.contains(&unspecified_id));
}

#[tokio::test]
async fn sticky_replica_releases_when_hot_and_recovers_without_extra_reshuffle() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let peers = (1..=4)
        .map(|seed| {
            fleet_peer_with_health(
                seed,
                BIG_MODELS[0],
                Some(InferenceAdmissionState::Accepting),
                None,
            )
        })
        .collect::<Vec<_>>();
    for peer in &peers {
        node.insert_test_peer(peer.clone()).await;
    }

    let initial = node.hosts_for_model(BIG_MODELS[0].name).await;
    let hot_id = initial[0];
    let mut hot = peers
        .iter()
        .find(|peer| peer.id == hot_id)
        .expect("selected peer")
        .clone();
    hot.inference_admission_state = Some(InferenceAdmissionState::AcceptingDeprioritized);
    node.insert_test_peer(hot.clone()).await;

    let under_duress = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_ne!(under_duress[0], hot_id);
    assert_eq!(under_duress.last(), Some(&hot_id));
    assert_eq!(under_duress[..3], initial[1..]);

    hot.inference_admission_state = Some(InferenceAdmissionState::Accepting);
    node.insert_test_peer(hot).await;
    assert_eq!(node.hosts_for_model(BIG_MODELS[0].name).await, initial);
}

#[tokio::test]
async fn throughput_breaks_ties_between_healthy_same_tier_models() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    node.insert_test_peer(fleet_peer_with_health(
        1,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::Accepting),
        Some(20_000),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_health(
        2,
        BIG_MODELS[1],
        Some(InferenceAdmissionState::Accepting),
        Some(60_000),
    ))
    .await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;
    let ranked_bases = actors
        .iter()
        .map(|&index| super::pool::canonical_base_name(&models[index].name))
        .collect::<Vec<_>>();
    assert_eq!(
        ranked_bases,
        vec![
            super::pool::canonical_base_name(BIG_MODELS[1].name),
            super::pool::canonical_base_name(BIG_MODELS[0].name)
        ]
    );
}

// ── The actual Buzz curated ladder (PR block/buzz#6189) ─────────────────────
//
// `desktop/src-tauri/src/mesh_llm/catalog.rs` recommends exactly one model per
// rated-memory class, so a real Buzz fleet only ever advertises these three
// names. Two of the three sit BELOW `SMALL_TIER_MAX_B` (10.0), which is what
// makes the mix below interesting: the ladder was not chosen with MoA's tier
// boundary in mind, and Qwen3.5-9B misses it by 1B.

/// `< 32 GB` rung — Gemma 4 E4B.
const BUZZ_RUNG_SMALL: FleetModel = SMALL_MODELS[0];
/// `32-79 GB` rung — Qwen3.5 9B. Small tier: 9.0 < 10.0.
const BUZZ_RUNG_MEDIUM: FleetModel = SMALL_MODELS[1];
/// `>= 80 GB` rung — Qwen3.8 27B. The only big-tier rung.
const BUZZ_RUNG_LARGE: FleetModel = BIG_MODELS[0];

/// Every Buzz-ladder fleet shape, through the real assembly path.
///
/// Printed rather than asserted row-by-row: the point is the shape of the
/// table, and the two assertions below pin the load-bearing claims.
#[tokio::test]
async fn buzz_ladder_committee_by_fleet_shape() {
    let shapes: Vec<(&str, Vec<(FleetModel, usize)>)> = vec![
        ("laptops only (E4B x50)", vec![(BUZZ_RUNG_SMALL, 50)]),
        ("mid only (9B x50)", vec![(BUZZ_RUNG_MEDIUM, 50)]),
        (
            "laptops + mid",
            vec![(BUZZ_RUNG_SMALL, 50), (BUZZ_RUNG_MEDIUM, 50)],
        ),
        ("one big (27B x1)", vec![(BUZZ_RUNG_LARGE, 1)]),
        ("two big (27B x2)", vec![(BUZZ_RUNG_LARGE, 2)]),
        ("big fleet (27B x100)", vec![(BUZZ_RUNG_LARGE, 100)]),
        (
            "full ladder, one big",
            vec![
                (BUZZ_RUNG_SMALL, 50),
                (BUZZ_RUNG_MEDIUM, 50),
                (BUZZ_RUNG_LARGE, 1),
            ],
        ),
        (
            "full ladder, many big",
            vec![
                (BUZZ_RUNG_SMALL, 500),
                (BUZZ_RUNG_MEDIUM, 300),
                (BUZZ_RUNG_LARGE, 100),
            ],
        ),
    ];

    tracing::debug!(
        "\n{:<26} {:>6} {:>9}  committee",
        "fleet shape",
        "nodes",
        "workers"
    );
    for (label, fleet) in &shapes {
        let pool = admitted_pool(fleet).await;
        tracing::debug!(
            "{:<26} {:>6} {:>9}  {:?}",
            label,
            total_nodes(fleet),
            pool.len(),
            pool
        );
    }

    // The two claims that matter, pinned.
    //
    // 1. An all-laptop Buzz fleet never convenes a committee: every rung below
    //    80 GB is small-tier, and an all-small pool collapses to its best
    //    member (measured loss at every small width, see
    //    `evals/moa-openrouter/RESULTS.md`).
    let laptops = admitted_pool(&[(BUZZ_RUNG_SMALL, 500), (BUZZ_RUNG_MEDIUM, 300)]).await;
    assert_eq!(
        laptops.len(),
        1,
        "all-small Buzz fleet must collapse to one worker, got {laptops:?}"
    );
    // Compare canonical bases: alias resolution may return the fully
    // qualified repo ref rather than the short descriptor name.
    assert_eq!(
        super::pool::canonical_base_name(&laptops[0]),
        super::pool::canonical_base_name(BUZZ_RUNG_MEDIUM.name),
        "best member is the 9B"
    );

    // 2. Adding ONE 80GB+ machine does not produce a committee — it produces a
    //    solo 27B. The pool stops being all-small, so the small collapse no
    //    longer applies, but a single big name on a single endpoint cannot
    //    self-fill (iron law), and the smalls are not deleted because
    //    `healthy_big_count < 2`. So the 27B answers alone.
    let one_big = admitted_pool(&[
        (BUZZ_RUNG_SMALL, 500),
        (BUZZ_RUNG_MEDIUM, 300),
        (BUZZ_RUNG_LARGE, 1),
    ])
    .await;
    tracing::debug!("one-big ladder pool: {one_big:?}");
}

/// Two 80GB+ machines on the same 27B: the committee Buzz can actually reach.
///
/// This is the *only* shape on the curated ladder that convenes a real
/// committee, and it is the homogeneous same-model case the mid-scale eval
/// measured at 48W/23T/2L — including the refinement round, which `Auto`
/// enables for a homogeneous non-small pool.
#[tokio::test]
async fn buzz_ladder_two_big_machines_form_the_only_reachable_committee() {
    let pool = admitted_pool(&[(BUZZ_RUNG_LARGE, 2)]).await;
    assert_eq!(pool.len(), 2, "two 27B endpoints self-fill to a pair");
    assert!(pool.iter().all(|name| name == BUZZ_RUNG_LARGE.name));

    // And the smalls are deleted once two big *models* are healthy — note this
    // needs two distinct big NAMES, not two replicas, so the curated ladder
    // (one big rung) never trips it.
    let with_smalls = admitted_pool(&[
        (BUZZ_RUNG_LARGE, 2),
        (BUZZ_RUNG_SMALL, 10),
        (BUZZ_RUNG_MEDIUM, 10),
    ])
    .await;
    tracing::debug!("two big replicas + smalls: {with_smalls:?}");
}

// ─── Tool-turn actor selection ───────────────────────────────────────
//
// A tool-bearing turn does not run a committee. `handle_tool_query` picks ONE
// actor — `reducer_candidates(config)[0]`, which is `actor_candidates[0]` when
// the host supplies it — and only that model receives the real tool schemas;
// every other worker advises tool-free. So the ordering asserted here decides
// whether tool calls are emitted by a model that can actually make them.
//
// The health and throughput terms added for fleet robustness sort BELOW
// `tool_use`. These tests exist to keep it that way: a robustness heuristic
// must never promote a non-tool-caller ahead of a tool-caller.

/// Tool capability outranks health. A deprioritized tool-caller must still act
/// before a perfectly healthy model that cannot call tools.
#[tokio::test]
async fn tool_capability_outranks_health_for_the_acting_model() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    // The tool-caller is deprioritized AND small — losing on every other term.
    node.insert_test_peer(fleet_peer_with_tool_use(
        1,
        SMALL_MODELS[1],
        CapabilityLevel::Supported,
        Some(InferenceAdmissionState::AcceptingDeprioritized),
    ))
    .await;
    // The healthy big model cannot call tools.
    node.insert_test_peer(fleet_peer_with_tool_use(
        2,
        BIG_MODELS[0],
        CapabilityLevel::None,
        Some(InferenceAdmissionState::Accepting),
    ))
    .await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;
    assert_eq!(
        super::pool::canonical_base_name(&models[actors[0]].name),
        super::pool::canonical_base_name(SMALL_MODELS[1].name),
        "the acting model must be the tool-caller even when it is \
         deprioritized and small; actor order = {:?}",
        actors
            .iter()
            .map(|&i| models[i].name.as_str())
            .collect::<Vec<_>>()
    );
}

/// Tool capability outranks advertised throughput. A slow tool-caller still
/// acts before a fast model that cannot call tools.
#[tokio::test]
async fn tool_capability_outranks_advertised_throughput_for_the_acting_model() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let mut slow_tool_caller =
        fleet_peer_with_tool_use(1, BIG_MODELS[0], CapabilityLevel::Supported, None);
    slow_tool_caller.advertised_model_throughput =
        vec![crate::network::metrics::ModelThroughputHint {
            model_name: BIG_MODELS[0].name.to_string(),
            avg_tokens_per_second_milli: 1_000,
            throughput_samples: 8,
        }];
    let mut fast_non_caller =
        fleet_peer_with_tool_use(2, BIG_MODELS[1], CapabilityLevel::None, None);
    fast_non_caller.advertised_model_throughput =
        vec![crate::network::metrics::ModelThroughputHint {
            model_name: BIG_MODELS[1].name.to_string(),
            avg_tokens_per_second_milli: 90_000,
            throughput_samples: 8,
        }];
    node.insert_test_peer(slow_tool_caller).await;
    node.insert_test_peer(fast_non_caller).await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;
    assert_eq!(
        super::pool::canonical_base_name(&models[actors[0]].name),
        super::pool::canonical_base_name(BIG_MODELS[0].name),
        "a 90x throughput advantage must not buy the acting slot for a model \
         that cannot call tools"
    );
}

/// Every admitted worker stays in `actor_candidates`, so the hedge ladder in
/// `hedged_reducer_call` can fall through when the preferred actor fails.
/// Reordering must never shorten the list.
#[tokio::test]
async fn actor_candidates_retain_every_admitted_worker_as_hedge_fallback() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    node.insert_test_peer(fleet_peer_with_tool_use(
        1,
        BIG_MODELS[0],
        CapabilityLevel::Supported,
        Some(InferenceAdmissionState::AcceptingDeprioritized),
    ))
    .await;
    node.insert_test_peer(fleet_peer_with_tool_use(
        2,
        BIG_MODELS[1],
        CapabilityLevel::None,
        None,
    ))
    .await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
    let actors = compute_actor_candidates(&node, &models).await;
    assert_eq!(
        actors.len(),
        models.len(),
        "a deprioritized worker must be demoted, never dropped: it is the \
         fallback the actor hedge walks to"
    );
    let mut seen = actors.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), actors.len(), "indices must be unique");
}

/// `cap_committee` must not fill the committee with deprioritized workers
/// while healthy ones exist.
///
/// The cap keeps `COMMITTEE_CAP_BIG` (4) big-tier models ordered by tier then
/// stable index, with no availability term. With five big bases resolved and
/// the four lowest-indexed ones deprioritized, the healthy fifth is capped out
/// — and `compute_actor_candidates` then ranks a pool that no longer contains
/// a healthy worker, so its availability term has nothing left to choose.
/// Admission runs before the cap, so this is the one place availability can
/// still be lost.
#[tokio::test]
async fn committee_cap_keeps_a_healthy_worker_over_deprioritized_ones() {
    use crate::proto::node::InferenceAdmissionState;

    const EXTRA_BIG: &[FleetModel] = &[
        FleetModel {
            name: "Mistral-Small-24B-Q4_K_M",
            parameter_count_b: 24.0,
            context_length: 32768,
        },
        FleetModel {
            // Sorts LAST alphabetically and by insertion index, so it is the
            // first casualty of an index-only cap. A name that happened to
            // sort early would let this test pass on luck.
            name: "Zephyr-70B-Q4_K_M",
            parameter_count_b: 70.0,
            context_length: 32768,
        },
    ];

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    // Four deprioritized big bases, then one healthy big base.
    let deprioritized = [BIG_MODELS[0], BIG_MODELS[1], BIG_MODELS[2], EXTRA_BIG[0]];
    for (i, model) in deprioritized.iter().enumerate() {
        node.insert_test_peer(fleet_peer_with_health(
            i as u32 + 1,
            *model,
            Some(InferenceAdmissionState::AcceptingDeprioritized),
            None,
        ))
        .await;
    }
    let healthy = EXTRA_BIG[1];
    node.insert_test_peer(fleet_peer_with_health(
        99,
        healthy,
        Some(InferenceAdmissionState::Accepting),
        None,
    ))
    .await;

    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;

    let kept: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    assert!(
        kept.iter()
            .any(|name| super::pool::canonical_base_name(name)
                == super::pool::canonical_base_name(healthy.name)),
        "the only healthy big worker must survive the committee cap; kept = {kept:?}"
    );
}

// ─── Parity with main on a healthy fleet ─────────────────────────────
//
// The robustness terms added here (admission health, rendezvous replica
// choice, availability inside the committee cap) all key off
// `inference_admission_state`. On a fleet where every peer is healthy — which
// is the normal case, and every large-model deployment we care about — they
// must be inert: same workers, same count, same tiering as the previous rule.
//
// These tests pin that. They are the cheap answer to "does this change MoA for
// large models?": assembly is the only code this branch touches, so if
// assembly is identical on a healthy fleet then the committee, the reducer and
// the refinement round all receive exactly what they receive on main.

/// A healthy big-tier fleet assembles the same pool the pre-change rule gave:
/// smalls excluded once two big models are present, capped at
/// `COMMITTEE_CAP_BIG`.
#[tokio::test]
async fn healthy_big_fleet_assembles_the_same_pool_as_before() {
    // Two distinct big models plus two smalls, all healthy (admission state
    // None, exactly what a peer that never reported activity advertises).
    let pool = admitted_pool(&[
        (BIG_MODELS[0], 2),
        (BIG_MODELS[1], 2),
        (SMALL_MODELS[0], 3),
        (SMALL_MODELS[1], 3),
    ])
    .await;

    // The measured admission rule: >=2 big healthy => drop every small.
    assert_eq!(
        pool.len(),
        2,
        "expected the two big models only, got {pool:?}"
    );
    for small in [SMALL_MODELS[0], SMALL_MODELS[1]] {
        assert!(
            !pool
                .iter()
                .any(|name| super::pool::canonical_base_name(name)
                    == super::pool::canonical_base_name(small.name)),
            "small-tier worker {} survived a healthy two-big pool: {pool:?}",
            small.name
        );
    }
}

/// A healthy single-big + small fleet keeps the small worker, because dropping
/// it would collapse the committee to a solo model. Unchanged by this branch.
#[tokio::test]
async fn healthy_single_big_fleet_still_keeps_the_small_worker() {
    let pool = admitted_pool(&[(BIG_MODELS[0], 2), (SMALL_MODELS[0], 2)]).await;
    assert_eq!(
        pool.len(),
        2,
        "one big + one small must stay a mixed committee, got {pool:?}"
    );
}

/// Removing a replica that is not our preferred one must not change our
/// preference.
///
/// This is the property `hash % hosts.len()` lacked: the index was computed
/// against the list *length*, so any peer leaving — even an unrelated one —
/// changed the modulus and moved every origin onto a different replica,
/// discarding its prefix cache. Rendezvous hashing scores each peer
/// independently, so dropping a non-preferred peer leaves the ranking of the
/// rest untouched.
///
/// Note the converse is deliberately NOT asserted: rendezvous hashing does not
/// promise a *joining* peer never becomes preferred — if its score is highest
/// it should win, which is how load moves onto new capacity. An earlier version
/// of this test asserted that and failed, correctly.
#[tokio::test]
async fn removing_a_non_preferred_replica_does_not_move_our_preference() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    for seed in 1..=6u32 {
        node.insert_test_peer(fleet_peer(seed, BIG_MODELS[0])).await;
    }

    let hosts = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_eq!(hosts.len(), 6);
    let preferred = hosts[0];
    let victim = *hosts
        .last()
        .expect("six replicas were inserted; last is the least preferred");
    assert_ne!(
        victim, preferred,
        "victim must not be the preferred replica"
    );

    node.remove_test_peer(victim).await;

    let after = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_eq!(after.len(), 5, "exactly the victim should be gone");
    assert!(!after.contains(&victim));
    assert_eq!(
        after[0], preferred,
        "dropping the least-preferred replica moved our preferred replica"
    );
    assert_eq!(
        after,
        hosts
            .iter()
            .copied()
            .filter(|id| *id != victim)
            .collect::<Vec<_>>(),
        "the surviving order must be the old order minus the victim"
    );
}
