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
