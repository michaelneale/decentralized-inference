use super::*;
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

/// Real wrapper-method-level proof, not just direct `notify_session_lifecycle`:
/// on a lane-exhausted runtime (`lane_count: 0`), `self.session(session_id)?`
/// bails with "all execution lanes are busy" BEFORE `export_full_state`/
/// `import_full_state`/etc ever reach `notify_export_outcome`/
/// `notify_import_outcome` -- exactly like `drop_session_timed`'s own
/// absent-session no-op: a caller-side precondition failure is not a native
/// export/import failure and must not be misreported as one.
///
/// A genuine `Ok` or native-level `Err` from these six methods requires
/// a real loaded model (`RuntimeState::model` on `new_modelless_for_test`
/// is a null dummy handle per its own doc comment, so calling through to
/// a real `StageSession` here would be undefined behavior, not a safe
/// test). That reachable positive/native-failure path remains
/// unexecuted in this crate's test suite; `notify_export_outcome`/
/// `notify_import_outcome`'s own outcome-to-event mapping is proven
/// directly below instead.
#[test]
fn export_and_import_wrappers_on_a_lane_exhausted_runtime_notify_nothing() {
    let observer = Arc::new(RecordingSessionObserver::default());
    let mut runtime =
        RuntimeState::new_modelless_for_test(0).with_session_lifecycle_observer(observer.clone());

    assert!(runtime.export_full_state("never-existed").is_err());
    assert!(runtime.import_full_state("never-existed", &[]).is_err());
    assert!(
        runtime
            .import_full_state_for_token_count("never-existed", &[], 0)
            .is_err()
    );
    assert!(
        runtime
            .import_state_for_token_count("never-existed", &[], 0)
            .is_err()
    );
    assert!(runtime.export_recurrent_state("never-existed").is_err());
    assert!(
        runtime
            .import_recurrent_state_for_token_count("never-existed", &[], 0)
            .is_err()
    );

    assert!(
        observer.0.lock().unwrap().is_empty(),
        "a lane-exhaustion precondition failure is not a native export/\
         import result and must not be reported as RuntimeStateExportFailed/\
         RuntimeStateImportFailed"
    );
}

/// Direct proof of `notify_export_outcome`/`notify_import_outcome`'s own
/// outcome-to-event mapping (the code every one of the six wrapper
/// methods calls), independent of the native-session-unavailable
/// limitation above.
#[test]
fn notify_export_and_import_outcome_map_ok_and_err_to_the_right_event() {
    let observer = Arc::new(RecordingSessionObserver::default());
    let runtime =
        RuntimeState::new_modelless_for_test(1).with_session_lifecycle_observer(observer.clone());

    runtime.notify_export_outcome(&Ok::<(), anyhow::Error>(()));
    runtime.notify_export_outcome(&Err::<(), anyhow::Error>(anyhow::anyhow!("export failed")));
    runtime.notify_import_outcome(&Ok::<(), anyhow::Error>(()));
    runtime.notify_import_outcome(&Err::<(), anyhow::Error>(anyhow::anyhow!("import failed")));

    assert_eq!(
        *observer.0.lock().unwrap(),
        vec![
            crate::runtime_state::SessionLifecycleEvent::RuntimeStateExportCompleted,
            crate::runtime_state::SessionLifecycleEvent::RuntimeStateExportFailed,
            crate::runtime_state::SessionLifecycleEvent::RuntimeStateImportCompleted,
            crate::runtime_state::SessionLifecycleEvent::RuntimeStateImportFailed,
        ]
    );
}

#[test]
fn prefix_restore_moves_tracked_position_backwards() {
    let mut token_counts = std::collections::BTreeMap::from([("lane-a".to_string(), 3_535)]);

    record_restored_session_token_count(&mut token_counts, "lane-a", 3_530);

    assert_eq!(token_counts.get("lane-a"), Some(&3_530));
}
