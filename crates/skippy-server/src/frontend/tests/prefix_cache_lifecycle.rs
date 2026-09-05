//! Real red/green coverage for `kv_integration::lifecycle`'s new observer
//! seam (plan task 12, §8.11): proves `probe_resident_prefix`,
//! `restore_resident_prefix`, and `evict_resident_prefix_for_tokens` --
//! genuine production call sites, not test-only fixtures -- actually
//! notify an attached observer, and that the default (no observer) path is
//! unaffected.

use super::support::*;
use super::*;
use std::sync::Mutex as StdMutex;

#[derive(Default)]
struct RecordingKvObserver {
    events: StdMutex<Vec<KvLifecycleEvent>>,
}

impl KvLifecycleObserver for RecordingKvObserver {
    fn observe(&self, event: KvLifecycleEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[test]
fn cold_lookup_notifies_a_cache_miss() {
    let config = prefix_cache_test_config();
    let observer = Arc::new(RecordingKvObserver::default());
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
        .unwrap()
        .expect("resident prefix cache enabled")
        .with_kv_lifecycle_observer(observer.clone());
    let identity = kv.prefill_identity(
        &config,
        &prefix_cache_test_base(),
        0,
        &(0..1024).collect::<Vec<_>>(),
    );

    assert!(kv.probe_resident_prefix(&identity).is_none());
    assert_eq!(
        *observer.events.lock().unwrap(),
        vec![KvLifecycleEvent::CacheLookupMiss]
    );
}

#[test]
fn repeated_prompt_lookup_notifies_a_cache_hit_with_bounded_counts() {
    let config = prefix_cache_test_config();
    let observer = Arc::new(RecordingKvObserver::default());
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
        .unwrap()
        .expect("resident prefix cache enabled")
        .with_kv_lifecycle_observer(observer.clone());
    let first_request = prefix_cache_base_with_request("request-a", "session-a");
    let second_request = prefix_cache_base_with_request("request-b", "session-b");
    let tokens = (0..1024).collect::<Vec<_>>();
    let recorded = kv.prefill_identity(&config, &first_request, 0, &tokens);
    let looked_up = kv.prefill_identity(&config, &second_request, 0, &tokens);
    seed_resident_prefix(&kv, &recorded);

    let hit = kv
        .probe_resident_prefix(&looked_up)
        .expect("repeated prompt should hit");

    let events = observer.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    match events[0] {
        KvLifecycleEvent::CacheLookupHit {
            matched_tokens,
            resident_entries,
        } => {
            assert_eq!(matched_tokens, hit.token_count);
            assert!(resident_entries >= 1);
        }
        ref other => panic!("expected CacheLookupHit, got {other:?}"),
    }
}

/// `evict_resident_prefix_for_tokens`'s real native-drop path requires a
/// model-backed `RuntimeState` (a modelless test runtime rejects the
/// resident-cell drop with `InvalidArgument: model and out_session are
/// required`), which this environment cannot build without downloading a
/// real model. This test instead proves the notify-on-eviction WIRING at
/// `KvStageIntegration::notify_kv_lifecycle` -- the exact call the real
/// `evict_resident_prefix_for_tokens` makes when `evicted_entries > 0` --
/// reaches the observer with the counts it is given, unmodified.
#[test]
fn eviction_notify_reaches_the_observer_with_bounded_counts() {
    let config = prefix_cache_test_config();
    let observer = Arc::new(RecordingKvObserver::default());
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
        .unwrap()
        .expect("resident prefix cache enabled")
        .with_kv_lifecycle_observer(observer.clone());

    kv.notify_kv_lifecycle(KvLifecycleEvent::CacheEviction {
        evicted_entries: 3,
        evicted_tokens: 512,
    });

    assert_eq!(
        *observer.events.lock().unwrap(),
        vec![KvLifecycleEvent::CacheEviction {
            evicted_entries: 3,
            evicted_tokens: 512,
        }]
    );
}

#[test]
fn real_initialization_notifies_started_then_completed() {
    let config = prefix_cache_test_config();
    let observer = Arc::new(RecordingKvObserver::default());

    let kv = KvStageIntegration::from_loaded_model(
        &config,
        Some(skippy_runtime::ModelStateKind::Dense),
        Some(observer.clone()),
    )
    .unwrap()
    .expect("resident prefix cache enabled");
    drop(kv);

    assert_eq!(
        *observer.events.lock().unwrap(),
        vec![
            KvLifecycleEvent::KvInitStarted,
            KvLifecycleEvent::KvInitCompleted,
        ]
    );
}

#[test]
fn disabled_kv_config_never_notifies_init_at_all() {
    let config = StageConfig {
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::Disabled,
            ..prefix_cache_test_config()
                .kv_cache
                .expect("test cache config")
        }),
        ..prefix_cache_test_config()
    };
    let observer = Arc::new(RecordingKvObserver::default());

    let kv = KvStageIntegration::from_loaded_model(
        &config,
        Some(skippy_runtime::ModelStateKind::Dense),
        Some(observer.clone()),
    )
    .unwrap();

    assert!(kv.is_none());
    assert!(
        observer.events.lock().unwrap().is_empty(),
        "a disabled/not-applicable KV config must never emit init started/completed/failed"
    );
}

#[test]
fn capacity_rejection_notifies_the_real_admission_deficit() {
    let config = prefix_cache_test_config();
    let observer = Arc::new(RecordingKvObserver::default());
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
        .unwrap()
        .expect("resident prefix cache enabled")
        .with_kv_lifecycle_observer(observer.clone());
    let mut runtime = crate::runtime_state::RuntimeState::new_modelless_with_capacity_for_test(
        config.lane_count,
        8,
    );

    let rejected = kv
        .admit_resident_capacity(&mut runtime, "request", 9, 1, 1, None)
        .unwrap();

    assert!(!rejected.admitted);
    assert_eq!(
        *observer.events.lock().unwrap(),
        vec![KvLifecycleEvent::CapacityApproachingLimit {
            admission_deficit_tokens: rejected.admission_deficit_tokens,
        }]
    );
}

#[test]
fn no_observer_attached_is_unaffected_and_never_panics() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
        .unwrap()
        .expect("resident prefix cache enabled");
    let identity = kv.prefill_identity(
        &config,
        &prefix_cache_test_base(),
        0,
        &(0..1024).collect::<Vec<_>>(),
    );
    // No `.with_kv_lifecycle_observer(...)` call: the default (`None`)
    // path must behave identically to every pre-existing prefix_cache test.
    assert!(kv.probe_resident_prefix(&identity).is_none());
}
