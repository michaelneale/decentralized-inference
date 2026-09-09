use super::*;
use crate::protocol::NODE_PROTOCOL_GENERATION;
use iroh::{EndpointId, SecretKey};

fn endpoint_id(seed: u8) -> EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    EndpointId::from(SecretKey::from_bytes(&bytes).public())
}

fn lifecycle_envelope(request_id: u64, marker: &str) -> crate::proto::node::OwnerControlEnvelope {
    crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: Some(crate::proto::node::OwnerControlResponse {
            request_id,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: None,
            load_model: Some(crate::proto::node::OwnerControlLoadModelResponse {
                intent_id: marker.to_string(),
                accepted_state: "queued".to_string(),
                target: None,
            }),
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        }),
        error: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_same_key_reservations_elect_one_leader_and_replay_exact_envelope() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x31);
    let request_id = 42;
    let task_count = 8;
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(task_count));
    let mut handles = Vec::new();

    for _ in 0..task_count {
        let cache = cache.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            cache.reserve(requester, request_id)
        }));
    }

    let mut leader = None;
    let mut followers = Vec::new();
    for handle in handles {
        match handle.await.expect("reservation task should not panic") {
            OwnerLifecycleResponseReservation::Leader(candidate) => {
                assert!(
                    leader.replace(candidate).is_none(),
                    "only one leader should be elected"
                );
            }
            OwnerLifecycleResponseReservation::Follower(follower) => followers.push(follower),
            OwnerLifecycleResponseReservation::Ready(envelope) => {
                panic!("first concurrent wave should not see ready envelope: {envelope:?}");
            }
        }
    }

    let leader = leader.expect("same-key wave should elect one leader");
    assert_eq!(followers.len(), task_count - 1);
    let envelope = lifecycle_envelope(request_id, "leader-response");
    leader.publish(envelope.clone());

    for follower in followers {
        assert_eq!(follower.wait().await, envelope);
    }
    match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, envelope),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("completed same-key request should replay from cache");
        }
    }
}

#[tokio::test]
async fn dropped_leader_releases_follower_with_control_unavailable_envelope() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x32);
    let request_id = 43;
    let leader = match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("first reservation should lead");
        }
    };
    let follower = match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Follower(follower) => follower,
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("second same-key reservation should follow");
        }
    };

    drop(leader);

    let envelope = tokio::time::timeout(std::time::Duration::from_secs(1), follower.wait())
        .await
        .expect("dropped leader should not strand follower");
    let error = envelope
        .error
        .expect("dropped leader should publish an error");
    assert_eq!(error.request_id, Some(request_id));
    assert_eq!(
        error.code,
        crate::proto::node::OwnerControlErrorCode::ControlUnavailable as i32
    );
}

#[tokio::test]
async fn expired_pending_reservation_cannot_complete_its_replacement() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x39);
    let request_id = 44;
    let stale_leader = match cache.reserve_with_fallback(
        requester,
        request_id,
        lifecycle_unavailable_envelope(request_id),
        Duration::ZERO,
    ) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        _ => panic!("first reservation should lead"),
    };
    let replacement = match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        _ => panic!("expired pending reservation should be replaced"),
    };

    stale_leader.publish(lifecycle_envelope(request_id, "stale"));
    let follower = match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Follower(follower) => follower,
        _ => panic!("stale completion must not replace the current pending reservation"),
    };
    let expected = lifecycle_envelope(request_id, "replacement");
    replacement.publish(expected.clone());
    assert_eq!(follower.wait().await, expected);
}

#[tokio::test]
async fn pending_lifecycle_reservation_survives_cache_saturation() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x34);
    let pending_request_id = 900;
    let pending_leader = match cache.reserve(requester, pending_request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("first pending reservation should lead");
        }
    };
    let initial_follower = match cache.reserve(requester, pending_request_id) {
        OwnerLifecycleResponseReservation::Follower(follower) => follower,
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("same pending reservation should follow");
        }
    };

    for request_id in 0..MAX_CACHED_LIFECYCLE_RESPONSES as u64 {
        let leader = match cache.reserve(requester, request_id) {
            OwnerLifecycleResponseReservation::Leader(leader) => leader,
            OwnerLifecycleResponseReservation::Follower(_)
            | OwnerLifecycleResponseReservation::Ready(_) => {
                panic!("distinct request should reserve a leader");
            }
        };
        leader.publish(lifecycle_envelope(request_id, "saturating"));
    }

    let retry_follower = match cache.reserve(requester, pending_request_id) {
        OwnerLifecycleResponseReservation::Follower(follower) => Some(follower),
        OwnerLifecycleResponseReservation::Ready(_) => None,
        OwnerLifecycleResponseReservation::Leader(_) => {
            panic!("pending reservation must not be evicted while its leader is in flight");
        }
    };
    let envelope = lifecycle_envelope(pending_request_id, "pending-complete");
    pending_leader.publish(envelope.clone());

    assert_eq!(initial_follower.wait().await, envelope);
    if let Some(retry_follower) = retry_follower {
        assert_eq!(retry_follower.wait().await, envelope);
    }
}

#[tokio::test]
async fn all_pending_lifecycle_cache_rejects_new_admission_without_stranding_followers() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x35);
    let mut leaders = Vec::new();

    for request_id in 0..MAX_CACHED_LIFECYCLE_RESPONSES_PER_REQUESTER as u64 {
        let leader = match cache.reserve(requester, request_id) {
            OwnerLifecycleResponseReservation::Leader(leader) => leader,
            OwnerLifecycleResponseReservation::Follower(_)
            | OwnerLifecycleResponseReservation::Ready(_) => {
                panic!("distinct pending request should reserve a leader");
            }
        };
        leaders.push((request_id, leader));
    }

    let follower = match cache.reserve(requester, 0) {
        OwnerLifecycleResponseReservation::Follower(follower) => follower,
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("existing pending request should still follow");
        }
    };
    let rejected_request_id = MAX_CACHED_LIFECYCLE_RESPONSES_PER_REQUESTER as u64;
    let rejected = cache.reserve(requester, rejected_request_id);
    let rejected_error = match rejected {
        OwnerLifecycleResponseReservation::Ready(envelope) => envelope
            .error
            .expect("all-pending admission should return unavailable"),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("all-pending cache should not admit a new tracked reservation");
        }
    };
    assert_eq!(rejected_error.request_id, Some(rejected_request_id));

    let (first_request_id, first_leader) = leaders.remove(0);
    let envelope = lifecycle_envelope(first_request_id, "first-pending-complete");
    first_leader.publish(envelope.clone());

    assert_eq!(follower.wait().await, envelope);
}

#[test]
fn one_requester_cannot_evict_another_requesters_replay() {
    let cache = OwnerLifecycleResponseCache::default();
    let protected_requester = endpoint_id(0x37);
    let noisy_requester = endpoint_id(0x38);
    let protected_request_id = 7;
    let protected = lifecycle_envelope(protected_request_id, "protected");

    match cache.reserve(protected_requester, protected_request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader.publish(protected.clone()),
        _ => panic!("first reservation should lead"),
    }
    for request_id in 0..MAX_CACHED_LIFECYCLE_RESPONSES as u64 {
        match cache.reserve(noisy_requester, request_id) {
            OwnerLifecycleResponseReservation::Leader(leader) => {
                leader.publish(lifecycle_envelope(request_id, "noisy"));
            }
            _ => panic!("distinct request should reserve a leader"),
        }
    }

    match cache.reserve(protected_requester, protected_request_id) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, protected),
        _ => panic!("another requester's traffic must not evict the protected replay"),
    }
}

#[test]
fn completed_lifecycle_responses_are_bounded() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x33);

    for request_id in 0..=MAX_CACHED_LIFECYCLE_RESPONSES as u64 {
        let leader = match cache.reserve(requester, request_id) {
            OwnerLifecycleResponseReservation::Leader(leader) => leader,
            OwnerLifecycleResponseReservation::Follower(_)
            | OwnerLifecycleResponseReservation::Ready(_) => {
                panic!("new request id should reserve a leader");
            }
        };
        leader.publish(lifecycle_envelope(request_id, "bounded"));
    }

    let evicted = cache.reserve(requester, 0);
    assert!(
        matches!(evicted, OwnerLifecycleResponseReservation::Leader(_)),
        "oldest completed response should be evicted after cache reaches its bound"
    );
    let newest = cache.reserve(requester, MAX_CACHED_LIFECYCLE_RESPONSES as u64);
    assert!(
        matches!(newest, OwnerLifecycleResponseReservation::Ready(_)),
        "newest completed response should remain cached"
    );
}

#[tokio::test]
async fn completed_response_replays_before_command_deadline() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x36);
    let request_id = 901;
    let leader = match cache.reserve_with_fallback(
        requester,
        request_id,
        lifecycle_unavailable_envelope(request_id),
        std::time::Duration::from_secs(60),
    ) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        _ => panic!("first reservation should lead"),
    };
    let envelope = lifecycle_envelope(request_id, "before-deadline");
    leader.publish(envelope.clone());

    match cache.reserve_with_fallback(
        requester,
        request_id,
        lifecycle_unavailable_envelope(request_id),
        std::time::Duration::from_secs(60),
    ) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, envelope),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("completed same-key request should replay before its deadline");
        }
    }
}

#[test]
fn completed_response_expires_with_command_deadline() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x37);
    let request_id = 902;
    let leader = match cache.reserve_with_fallback(
        requester,
        request_id,
        lifecycle_unavailable_envelope(request_id),
        std::time::Duration::ZERO,
    ) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => panic!("first reservation should lead"),
    };
    leader.publish(lifecycle_envelope(request_id, "expired"));

    match cache.reserve_with_fallback(
        requester,
        request_id,
        lifecycle_unavailable_envelope(request_id),
        std::time::Duration::from_secs(60),
    ) {
        OwnerLifecycleResponseReservation::Leader(_) => {}
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("expired completed response should elect a new leader");
        }
    }
}

#[test]
fn same_request_id_from_different_requesters_elects_independent_leaders() {
    let cache = OwnerLifecycleResponseCache::default();
    let first_requester = endpoint_id(0x38);
    let second_requester = endpoint_id(0x39);
    let request_id = 903;

    let first_leader = match cache.reserve(first_requester, request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("first requester should lead its request id");
        }
    };
    let second_leader = match cache.reserve(second_requester, request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("different requester should lead the same request id independently");
        }
    };

    let first_envelope = lifecycle_envelope(request_id, "first-requester");
    let second_envelope = lifecycle_envelope(request_id, "second-requester");
    first_leader.publish(first_envelope.clone());
    second_leader.publish(second_envelope.clone());

    match cache.reserve(first_requester, request_id) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, first_envelope),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("first requester should replay its own completed response");
        }
    }
    match cache.reserve(second_requester, request_id) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, second_envelope),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("second requester should replay its own completed response");
        }
    }
}

#[test]
fn complete_key_only_transitions_pending_and_preserves_authoritative_ready() {
    let cache = OwnerLifecycleResponseCache::default();
    let requester = endpoint_id(0x3a);
    let request_id = 904;
    let leader = match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Leader(leader) => leader,
        OwnerLifecycleResponseReservation::Follower(_)
        | OwnerLifecycleResponseReservation::Ready(_) => {
            panic!("first reservation should lead");
        }
    };
    let authoritative = lifecycle_envelope(request_id, "authoritative");
    leader.publish(authoritative.clone());

    let key = OwnerLifecycleCacheKey {
        requester,
        request_id,
    };
    let late_fallback = lifecycle_envelope(request_id, "late-fallback");
    let (unrelated_reservation, _) = watch::channel(None);
    assert!(!cache.complete_key(key, &unrelated_reservation, late_fallback));

    match cache.reserve(requester, request_id) {
        OwnerLifecycleResponseReservation::Ready(cached) => assert_eq!(cached, authoritative),
        OwnerLifecycleResponseReservation::Leader(_)
        | OwnerLifecycleResponseReservation::Follower(_) => {
            panic!("ready response should remain authoritative");
        }
    }
}
