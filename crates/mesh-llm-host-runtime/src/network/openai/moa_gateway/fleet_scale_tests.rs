//! Thousand-scale robustness for MoA pool assembly.
//!
//! `fleet_sim_tests.rs` measures admitted *width*; `fleet_fairness_tests.rs`
//! measures how scarce big-tier capacity is *shared*. This file measures what
//! happens when the fleet is large and things go wrong:
//!
//! - Assembly is on the **per-request** path (`assemble_worker_pool` runs for
//!   every `model=mesh` turn), and it walks every peer at least twice
//!   (`models_being_served`, `gossiped_sizes`, `model_routing_hints`). So its
//!   cost in fleet size is a serving property, not a test curiosity.
//! - Degradation shapes: every peer paused, every peer deprioritized, a fleet
//!   with no big tier, and a fleet where the only big replicas are unreachable.
//!
//! These are measurements plus a small number of load-bearing assertions. The
//! timings are wall-clock on one machine and are recorded for *shape* (linear
//! vs superlinear), not as a performance budget — a threshold assertion here
//! would be flaky on CI hardware.

use super::fleet_sim_tests::{
    BIG_MODELS, FleetModel, SMALL_MODELS, fleet_peer, fleet_peer_with_health,
};
use super::pool::{assemble_worker_pool, canonical_base_name, compute_actor_candidates};
use crate::inference::election;
use crate::mesh;
use std::time::Instant;

/// A fleet of `count` peers spread evenly over `models`, as distinct endpoints.
async fn node_with_n_peers(models: &[FleetModel], count: u32) -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    for seed in 1..=count {
        let model = models[(seed as usize) % models.len()];
        node.insert_test_peer(fleet_peer(seed, model)).await;
    }
    node
}

async fn assemble(node: &mesh::Node) -> Vec<String> {
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(node, Some(&targets), Some(13_000), &http, None).await;
    models.into_iter().map(|m| m.name).collect()
}

/// How does per-request pool assembly scale with fleet size?
///
/// Assembly walks every peer, so cost grows with the fleet even though the
/// admitted committee does not. This records the curve up to 4000 peers. The
/// assertion is on the *result* (width stays capped), not on the timing.
#[tokio::test]
async fn assembly_cost_and_width_at_thousand_scale() {
    let models = [
        BIG_MODELS[0],
        BIG_MODELS[1],
        SMALL_MODELS[0],
        SMALL_MODELS[1],
    ];
    tracing::debug!(
        "\n{:>7} {:>12} {:>8}  committee",
        "peers",
        "assemble_ms",
        "workers"
    );
    let mut widths = Vec::new();
    for count in [10u32, 100, 1_000, 2_000, 4_000] {
        let node = node_with_n_peers(&models, count).await;
        // One warm pass, then measure: the first call populates nothing cached
        // today, but this keeps the numbers comparable if that ever changes.
        let _ = assemble(&node).await;
        let started = Instant::now();
        let pool = assemble(&node).await;
        let elapsed = started.elapsed();
        tracing::debug!(
            "{count:>7} {:>12.1} {:>8}  {pool:?}",
            elapsed.as_secs_f64() * 1000.0,
            pool.len()
        );
        widths.push(pool.len());
    }
    assert!(
        widths.windows(2).all(|w| w[0] == w[1]),
        "admitted width must not grow with fleet size: {widths:?}"
    );
}

/// Actor ranking sorts the admitted pool, not the fleet — confirm it stays
/// bounded work and returns a full ranking at scale.
#[tokio::test]
async fn actor_ranking_is_bounded_at_thousand_scale() {
    let models = [BIG_MODELS[0], BIG_MODELS[1], SMALL_MODELS[0]];
    let node = node_with_n_peers(&models, 3_000).await;
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, pool) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;

    let started = Instant::now();
    let actors = compute_actor_candidates(&node, &pool).await;
    tracing::debug!(
        "3000 peers: pool={} rank_ms={:.1} actors={:?}",
        pool.len(),
        started.elapsed().as_secs_f64() * 1000.0,
        actors
    );
    assert_eq!(
        actors.len(),
        pool.len(),
        "ranking must cover every admitted worker"
    );
}

/// Every peer in a 1000-node fleet is paused: MoA must admit nobody rather
/// than route to a peer that will refuse.
///
/// This is the shape that matters for robustness: the gateway's own fallback
/// (`degrade_to_single_model`) can only work if assembly reports honestly.
#[tokio::test]
async fn fully_paused_thousand_node_fleet_admits_nobody() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    for seed in 1..=1_000u32 {
        node.insert_test_peer(fleet_peer_with_health(
            seed,
            BIG_MODELS[(seed as usize) % BIG_MODELS.len()],
            Some(InferenceAdmissionState::AllPaused),
            None,
        ))
        .await;
    }

    let pool = assemble(&node).await;
    tracing::debug!("1000 paused peers -> committee {pool:?}");
    assert!(
        pool.is_empty(),
        "a fully paused fleet must admit no workers, got {pool:?}"
    );
}

/// Every peer deprioritized (the "whole fleet is busy" shape).
///
/// Deprioritized is spillover capacity, not a refusal, so the committee must
/// still form — shedding load must never shed the last route.
#[tokio::test]
async fn fully_deprioritized_thousand_node_fleet_still_serves() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    for seed in 1..=1_000u32 {
        node.insert_test_peer(fleet_peer_with_health(
            seed,
            BIG_MODELS[(seed as usize) % BIG_MODELS.len()],
            Some(InferenceAdmissionState::AcceptingDeprioritized),
            None,
        ))
        .await;
    }

    let pool = assemble(&node).await;
    tracing::debug!("1000 deprioritized peers -> committee {pool:?}");
    assert!(
        !pool.is_empty(),
        "deprioritized peers are spillover capacity and must still be admitted"
    );
}

/// A thousand small nodes and no big tier: the measured all-small collapse must
/// still hold at scale, i.e. exactly one worker.
#[tokio::test]
async fn thousand_small_nodes_still_collapse_to_one_worker() {
    let node = node_with_n_peers(&[SMALL_MODELS[0], SMALL_MODELS[1], SMALL_MODELS[2]], 1_000).await;
    let pool = assemble(&node).await;
    tracing::debug!("1000 small peers, 3 distinct models -> committee {pool:?}");
    assert_eq!(
        pool.len(),
        1,
        "all-small collapse must hold at fleet scale, got {pool:?}"
    );
}

/// The asymmetric fleet Mic asked about, at scale: ~1000 small nodes and a
/// handful of big ones. Records what the committee is and who acts first.
#[tokio::test]
async fn few_big_many_small_at_thousand_scale() {
    use crate::proto::node::InferenceAdmissionState;

    for big in [1u32, 2, 4] {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
            .await
            .expect("test node");
        let mut seed = 1u32;
        for _ in 0..big {
            node.insert_test_peer(fleet_peer_with_health(
                seed,
                BIG_MODELS[0],
                Some(InferenceAdmissionState::Accepting),
                None,
            ))
            .await;
            seed += 1;
        }
        for _ in 0..1_000 {
            node.insert_test_peer(fleet_peer(seed, SMALL_MODELS[(seed as usize) % 3]))
                .await;
            seed += 1;
        }

        let targets = election::ModelTargets::default();
        let http = reqwest::Client::new();
        let (_b, pool) =
            assemble_worker_pool(&node, Some(&targets), Some(13_000), &http, None).await;
        let actors = compute_actor_candidates(&node, &pool).await;
        let first = actors
            .first()
            .and_then(|&i| pool.get(i))
            .map(|m| canonical_base_name(&m.name))
            .unwrap_or_default();
        tracing::debug!(
            "big={big} + 1000 small -> workers={} committee={:?} actor_first={first}",
            pool.len(),
            pool.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()
        );
        assert!(
            !pool.is_empty(),
            "an asymmetric fleet must always produce a servable pool"
        );
    }
}

/// Replica count does not substitute for model count in admission control.
///
/// `apply_admission_control` counts big-tier *entries in the pool*, and
/// assembly resolves exactly one worker per canonical model name. So 400 big
/// replicas of ONE model contribute `big_count == 1`, the `>= 2` gate never
/// fires, and small workers are retained beside the big one — the same outcome
/// as a single big machine.
///
/// This is consistent with the measured rule (32B + 8B keeps the 8B, because
/// dropping it collapses to solo), but it is worth pinning explicitly: on a
/// fleet where the big tier is deep but uniform — which is the likely real
/// shape, since a curated ladder recommends one model per memory class — the
/// small-exclusion path is unreachable no matter how many big machines join.
#[tokio::test]
async fn deep_uniform_big_tier_does_not_exclude_smalls() {
    use crate::proto::node::InferenceAdmissionState;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let mut seed = 1u32;
    for _ in 0..400 {
        node.insert_test_peer(fleet_peer_with_health(
            seed,
            BIG_MODELS[0],
            Some(InferenceAdmissionState::Accepting),
            None,
        ))
        .await;
        seed += 1;
    }
    for _ in 0..600 {
        node.insert_test_peer(fleet_peer(seed, SMALL_MODELS[0]))
            .await;
        seed += 1;
    }

    let pool = assemble(&node).await;
    tracing::debug!("400 uniform big + 600 small -> committee {pool:?}");
    let has_small = pool
        .iter()
        .any(|n| canonical_base_name(n) == canonical_base_name(SMALL_MODELS[0].name));
    assert!(
        has_small,
        "one distinct big model cannot trip the >=2-big exclusion, however many \
         replicas serve it: {pool:?}"
    );

    // Two distinct big NAMES do trip it, at the same fleet size.
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let mut seed = 1u32;
    for _ in 0..200 {
        node.insert_test_peer(fleet_peer_with_health(
            seed,
            BIG_MODELS[0],
            Some(InferenceAdmissionState::Accepting),
            None,
        ))
        .await;
        seed += 1;
        node.insert_test_peer(fleet_peer_with_health(
            seed,
            BIG_MODELS[1],
            Some(InferenceAdmissionState::Accepting),
            None,
        ))
        .await;
        seed += 1;
    }
    for _ in 0..600 {
        node.insert_test_peer(fleet_peer(seed, SMALL_MODELS[0]))
            .await;
        seed += 1;
    }
    let pool = assemble(&node).await;
    tracing::debug!("200+200 distinct big + 600 small -> committee {pool:?}");
    assert!(
        !pool
            .iter()
            .any(|n| canonical_base_name(n) == canonical_base_name(SMALL_MODELS[0].name)),
        "two distinct healthy big models must exclude verified smalls: {pool:?}"
    );
}
