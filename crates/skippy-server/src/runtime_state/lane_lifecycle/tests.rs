use super::*;
use anyhow::{Result, bail};
use std::sync::Mutex as StdMutex;

struct RecordingSessionObserver(StdMutex<Vec<crate::runtime_state::SessionLifecycleEvent>>);

impl Default for RecordingSessionObserver {
    fn default() -> Self {
        Self(StdMutex::new(Vec::new()))
    }
}

impl crate::runtime_state::SessionLifecycleObserver for RecordingSessionObserver {
    fn observe(&self, event: crate::runtime_state::SessionLifecycleEvent) {
        self.0.lock().unwrap().push(event);
    }
}

#[test]
fn dropping_an_absent_session_notifies_nothing() {
    let observer = Arc::new(RecordingSessionObserver::default());
    let runtime =
        RuntimeState::new_modelless_for_test(1).with_session_lifecycle_observer(observer.clone());
    let mut runtime = runtime;

    let stats = runtime.drop_session_timed("never-existed").unwrap();

    assert!(!stats.reset_session);
    assert!(!stats.lane_discarded);
    assert!(
        observer.0.lock().unwrap().is_empty(),
        "no real reset/reclaim decision was made, so nothing should be notified"
    );
}

/// `drop_session_timed`'s real reset/discard branches require a
/// native `StageSession` (the same model-backed requirement
/// `evict_resident_prefix_for_tokens`'s native drop hit in the KV
/// producer tests), unavailable in this environment. This proves the
/// notify WIRING at `RuntimeState::notify_session_lifecycle` -- the
/// exact call the real reset/discard branches make -- reaches the
/// observer with the right event shape.
#[test]
fn notify_wiring_reaches_the_attached_observer() {
    let observer = Arc::new(RecordingSessionObserver::default());
    let runtime =
        RuntimeState::new_modelless_for_test(1).with_session_lifecycle_observer(observer.clone());

    runtime.notify_session_lifecycle(crate::runtime_state::SessionLifecycleEvent::SessionReset {
        reset_ms: 1.5,
    });
    runtime.notify_session_lifecycle(crate::runtime_state::SessionLifecycleEvent::SessionReclaimed);

    assert_eq!(
        *observer.0.lock().unwrap(),
        vec![
            crate::runtime_state::SessionLifecycleEvent::SessionReset { reset_ms: 1.5 },
            crate::runtime_state::SessionLifecycleEvent::SessionReclaimed,
        ]
    );
}

#[test]
fn create_indexed_lane_resource_keeps_index_available_when_creation_fails() {
    let mut next_lane_index = 0;
    let mut free_lane_indices: Vec<usize> = Vec::new();

    let error = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        2,
        || -> Result<()> { bail!("transient session creation failure") },
    )
    .expect_err("failed creation should propagate the original error");

    assert_eq!(error.to_string(), "transient session creation failure");
    assert_eq!(next_lane_index, 0);
    assert!(free_lane_indices.is_empty());

    let (index, resource) =
        create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
            Ok("lane")
        })
        .expect("successful retry should reuse the unconsumed lane index");

    assert_eq!(index, 0);
    assert_eq!(resource, "lane");
    assert_eq!(next_lane_index, 1);
}

#[test]
fn create_indexed_lane_resource_reuses_freed_indices_before_growing() {
    // Simulate the wedge scenario: all lanes allocated, one lane
    // freed via the discard path, next allocation must reuse the
    // freed index rather than bailing with "all execution lanes
    // are busy".
    let mut next_lane_index = 0;
    let mut free_lane_indices: Vec<usize> = Vec::new();
    let lane_count = 2;

    // Allocate both lanes.
    let (a_idx, _) = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        lane_count,
        || Ok("a"),
    )
    .expect("first allocation should succeed");
    let (b_idx, _) = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        lane_count,
        || Ok("b"),
    )
    .expect("second allocation should succeed");
    assert_eq!(a_idx, 0);
    assert_eq!(b_idx, 1);
    assert_eq!(next_lane_index, 2);

    // Pool is full at the high-water mark. A third allocation must
    // fail.
    let error = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        lane_count,
        || Ok("c"),
    )
    .expect_err("allocating past lane_count should fail when no slots are free");
    assert!(error.to_string().contains("all execution lanes are busy"));

    // Discard one lane: the caller pushes its freed index onto the
    // free list (this is what drop_session_timed does on the
    // discard branch).
    free_lane_indices.push(a_idx);

    // The next allocation MUST reuse the freed index instead of
    // bailing. This is the wedge regression: previously
    // next_lane_index stayed at lane_count and every allocation
    // failed forever.
    let (reused_idx, _) = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        lane_count,
        || Ok("c"),
    )
    .expect("allocation must reuse a freed index, not stay wedged");
    assert_eq!(reused_idx, 0);
    assert_eq!(next_lane_index, 2);
    assert!(free_lane_indices.is_empty());
}

#[test]
fn create_indexed_lane_resource_returns_freed_index_on_create_failure() {
    // If create() fails while consuming a freed index, the index
    // must go back onto the free list so a retry can use it.
    let mut next_lane_index = 1;
    let mut free_lane_indices: Vec<usize> = vec![0];

    let error = create_indexed_lane_resource(
        &mut next_lane_index,
        &mut free_lane_indices,
        2,
        || -> Result<()> { bail!("create failed mid-reuse") },
    )
    .expect_err("failed creation should propagate");
    assert_eq!(error.to_string(), "create failed mid-reuse");
    assert_eq!(next_lane_index, 1);
    assert_eq!(free_lane_indices, vec![0]);

    // A retry should now succeed using the same freed index.
    let (idx, _) =
        create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
            Ok("retry")
        })
        .expect("retry should succeed");
    assert_eq!(idx, 0);
    assert_eq!(next_lane_index, 1);
    assert!(free_lane_indices.is_empty());
}

#[test]
fn capped_target_idle_sessions_clamps_to_the_configured_bound() {
    assert_eq!(capped_target_idle_sessions(10, Some(2)), 2);
    assert_eq!(capped_target_idle_sessions(1, Some(2)), 1);
}

#[test]
fn capped_target_idle_sessions_is_unbounded_when_unset() {
    assert_eq!(capped_target_idle_sessions(10, None), 10);
}
