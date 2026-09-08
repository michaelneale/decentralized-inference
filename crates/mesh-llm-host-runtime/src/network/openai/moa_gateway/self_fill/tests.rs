use super::super::fleet_sim_tests::{BIG_MODELS, fleet_peer_with_health};
use super::super::pool::assemble_worker_pool;
use super::*;
use std::collections::HashSet;
use tokio::sync::{Barrier, mpsc};

async fn fleet(count: u32) -> (mesh::Node, Vec<InferenceTarget>) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let mut candidates = Vec::new();
    for seed in 1..=count {
        let peer = fleet_peer_with_health(seed, BIG_MODELS[0], None, Some(100_000));
        candidates.push(InferenceTarget::Remote(peer.id));
        node.insert_test_peer(peer).await;
    }
    (node, candidates)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_committees_spread_across_twenty_clones() {
    let (node, candidates) = fleet(20).await;
    let affinity = Arc::new(AffinityRouter::new());
    let start = Arc::new(Barrier::new(10));
    let mut tasks = Vec::new();
    for _ in 0..10 {
        let (node, candidates, affinity, start) = (
            node.clone(),
            candidates.clone(),
            affinity.clone(),
            start.clone(),
        );
        tasks.push(tokio::spawn(async move {
            start.wait().await;
            select_clones(
                &node,
                BIG_MODELS[0].name,
                Some(13_000),
                candidates,
                Some(&affinity),
            )
            .await
        }));
    }
    // JoinHandle outputs retain every guard until all selections have finished.
    let mut committees = Vec::new();
    for task in tasks {
        committees.push(task.await.unwrap());
    }
    let mut unique = HashSet::new();
    for committee in &committees {
        assert_eq!(committee.len(), 2);
        assert_ne!(committee[0].0, committee[1].0);
        for (target, _) in committee {
            assert!(
                unique.insert(format!("{target:?}")),
                "clone reused while idle equivalents remain"
            );
        }
    }
    assert_eq!(unique.len(), 20);
    assert_eq!(affinity.stats_snapshot().reservation_active, 20);
    drop(committees);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn pressure_does_not_promote_slow_or_unknown_context_clones() {
    let (node, mut candidates) = fleet(2).await;
    let slow = fleet_peer_with_health(3, BIG_MODELS[0], None, Some(1_000));
    candidates.push(InferenceTarget::Remote(slow.id));
    node.insert_test_peer(slow).await;
    let mut unknown = fleet_peer_with_health(4, BIG_MODELS[0], None, Some(100_000));
    unknown.served_model_runtime.clear();
    unknown.served_model_descriptors[0].metadata = None;
    candidates.push(InferenceTarget::Remote(unknown.id));
    node.insert_test_peer(unknown).await;
    let affinity = AffinityRouter::new();
    let mut committees = Vec::new();
    for _ in 0..5 {
        let selected = select_clones(
            &node,
            BIG_MODELS[0].name,
            Some(13_000),
            candidates.clone(),
            Some(&affinity),
        )
        .await;
        assert_eq!(selected.len(), 2);
        assert!(
            selected
                .iter()
                .all(|(target, _)| candidates[..2].contains(target))
        );
        committees.push(selected);
    }
    assert_eq!(affinity.stats_snapshot().reservation_active, 10);
    drop(committees);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn cancelling_assembled_turn_releases_backend_reservations() {
    let (node, _) = fleet(20).await;
    let affinity = Arc::new(AffinityRouter::new());
    let (ready_tx, mut ready_rx) = mpsc::channel(1);
    let task_affinity = affinity.clone();
    let task = tokio::spawn(async move {
        let (backends, models) = assemble_worker_pool(
            &node,
            None,
            Some(13_000),
            &reqwest::Client::new(),
            Some(&task_affinity),
        )
        .await;
        assert_eq!(models.len(), 2);
        // The real gateway clones backend Arcs for worker/reducer calls.
        let worker_reference = backends[0].clone();
        drop(backends);
        assert_eq!(task_affinity.stats_snapshot().reservation_active, 1);
        ready_tx.send(()).await.unwrap();
        std::future::pending::<()>().await;
        drop(worker_reference);
    });
    tokio::time::timeout(std::time::Duration::from_secs(10), ready_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(affinity.stats_snapshot().reservation_active, 1);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn singleton_and_context_filtered_fleet_do_not_leak_reservations() {
    let (node, candidates) = fleet(1).await;
    let affinity = AffinityRouter::new();
    let (backends, models) = assemble_worker_pool(
        &node,
        None,
        Some(13_000),
        &reqwest::Client::new(),
        Some(&affinity),
    )
    .await;
    assert_eq!(models.len(), 1);
    drop(backends);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    let selected = select_clones(
        &node,
        BIG_MODELS[0].name,
        Some(100_000),
        candidates,
        Some(&affinity),
    )
    .await;
    assert!(selected.is_empty());
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn healthy_clones_beat_faster_deprioritized_clone_even_under_pressure() {
    use crate::proto::node::InferenceAdmissionState;

    let (node, healthy) = fleet(2).await;
    let hot = fleet_peer_with_health(
        3,
        BIG_MODELS[0],
        Some(InferenceAdmissionState::AcceptingDeprioritized),
        Some(200_000),
    );
    node.insert_test_peer(hot).await;
    let candidates: Vec<_> = node
        .hosts_for_model(BIG_MODELS[0].name)
        .await
        .into_iter()
        .map(InferenceTarget::Remote)
        .collect();
    let affinity = AffinityRouter::new();
    let mut committees = Vec::new();
    // Also preserve health priority when reservation accounting is disabled.
    for router in [None, Some(&affinity), Some(&affinity), Some(&affinity)] {
        let selected = select_clones(
            &node,
            BIG_MODELS[0].name,
            Some(13_000),
            candidates.clone(),
            router,
        )
        .await;
        assert_eq!(selected.len(), 2);
        assert_ne!(selected[0].0, selected[1].0);
        assert!(selected.iter().all(|(target, _)| healthy.contains(target)));
        committees.push(selected);
    }
    assert_eq!(affinity.stats_snapshot().reservation_active, 6);
    drop(committees);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[tokio::test]
async fn depleted_healthy_tier_uses_and_spreads_spillover() {
    use crate::proto::node::InferenceAdmissionState;

    // With one healthy clone it owns slot one; with none, both slots may
    // use spillover. Equal throughput across tiers must not merge their
    // reservation windows, even after the healthy clone is reserved.
    for healthy_count in [0, 1] {
        let (node, healthy) = fleet(healthy_count).await;
        let mut spillover = Vec::new();
        for seed in 2..=5 {
            let peer = fleet_peer_with_health(
                seed,
                BIG_MODELS[0],
                Some(InferenceAdmissionState::AcceptingDeprioritized),
                Some(100_000),
            );
            spillover.push(InferenceTarget::Remote(peer.id));
            node.insert_test_peer(peer).await;
        }
        let candidates = node
            .hosts_for_model(BIG_MODELS[0].name)
            .await
            .into_iter()
            .map(InferenceTarget::Remote)
            .collect::<Vec<_>>();
        let affinity = AffinityRouter::new();
        let mut committees = Vec::new();
        let mut used_spillover = HashSet::new();
        for _ in 0..(4 / (2 - healthy_count)) {
            let selected = select_clones(
                &node,
                BIG_MODELS[0].name,
                Some(13_000),
                candidates.clone(),
                Some(&affinity),
            )
            .await;
            assert_eq!(selected.len(), 2);
            assert_ne!(selected[0].0, selected[1].0);
            if healthy_count == 1 {
                assert_eq!(selected[0].0, healthy[0]);
            }
            for (target, _) in &selected {
                if spillover.contains(target) {
                    assert!(used_spillover.insert(format!("{target:?}")));
                }
            }
            committees.push(selected);
        }
        assert_eq!(used_spillover.len(), 4);
        assert_eq!(
            affinity.stats_snapshot().reservation_active,
            committees.len() * 2
        );
        drop(committees);
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }
}

#[tokio::test]
async fn context_ineligible_healthy_clones_do_not_block_spillover() {
    use crate::proto::node::InferenceAdmissionState;

    let (node, _) = fleet(2).await;
    let mut spillover = Vec::new();
    for seed in 3..=4 {
        let mut peer = fleet_peer_with_health(
            seed,
            BIG_MODELS[0],
            Some(InferenceAdmissionState::AcceptingDeprioritized),
            Some(200_000),
        );
        // Existing healthy clones cannot fit 100k; these can.
        for runtime in &mut peer.served_model_runtime {
            runtime.context_length = Some(200_000);
        }
        spillover.push(InferenceTarget::Remote(peer.id));
        node.insert_test_peer(peer).await;
    }
    let candidates = node
        .hosts_for_model(BIG_MODELS[0].name)
        .await
        .into_iter()
        .map(InferenceTarget::Remote)
        .collect();
    let affinity = AffinityRouter::new();
    let selected = select_clones(
        &node,
        BIG_MODELS[0].name,
        Some(100_000),
        candidates,
        Some(&affinity),
    )
    .await;
    assert_eq!(selected.len(), 2);
    assert_ne!(selected[0].0, selected[1].0);
    assert!(
        selected
            .iter()
            .all(|(target, _)| spillover.contains(target))
    );
    drop(selected);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}
