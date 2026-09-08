use std::io;
use std::sync::{Arc, Mutex as StdMutex};

use mesh_llm_events::{OutputEvent, OutputSink, clear_output_sink, set_output_sink};

use super::*;

#[derive(Default)]
struct RecordingOutputSink {
    events: StdMutex<Vec<OutputEvent>>,
}

impl RecordingOutputSink {
    fn take_events(&self) -> Vec<OutputEvent> {
        std::mem::take(&mut *self.events.lock().expect("recording sink mutex poisoned"))
    }
}

impl OutputSink for RecordingOutputSink {
    fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.events
            .lock()
            .expect("recording sink mutex poisoned")
            .push(event);
        Ok(())
    }
}

struct OutputSinkResetGuard;

impl Drop for OutputSinkResetGuard {
    fn drop(&mut self) {
        clear_output_sink();
    }
}

#[test]
fn native_model_open_finished_translates_to_info_without_readiness_events() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFinished,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("finished event should produce output visibility");

    match translated {
        OutputEvent::Info { message, context } => {
            assert!(message.contains("waiting for Rust runtime readiness"));
            assert!(
                context
                    .as_deref()
                    .is_some_and(|value| value.contains("sequence=7"))
            );
        }
        other => panic!("expected info event, got {other:?}"),
    }
}

#[test]
fn native_model_open_progress_translates_to_percentage_visibility() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenProgress,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("progress event should produce output visibility");

    match translated {
        OutputEvent::Info { message, .. } => {
            assert!(message.contains("Opening native model 50%"));
        }
        other => panic!("expected info event, got {other:?}"),
    }
}

#[test]
fn native_model_open_handled_failure_translates_to_warning_without_readiness_events() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
            sequence: 8,
            status: "Err",
            emitter: "OpenThread",
            progress_current: 0,
            progress_total: 0,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("handled failure should still produce output visibility");

    match translated {
        OutputEvent::Warning { message, context } => {
            assert!(message.contains("handled model-open failure"));
            assert!(
                context
                    .as_deref()
                    .is_some_and(|value| value.contains("sequence=8"))
            );
        }
        other => panic!("expected warning event, got {other:?}"),
    }
}

#[test]
#[serial_test::serial]
fn native_model_open_reporter_emits_visibility_only_events() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let mut reporter =
        skippy_native_model_open_event_reporter("/private/models/model-a.gguf".to_string(), None);
    for kind in [
        SkippyNativeRuntimeEventKind::ModelOpenStarted,
        SkippyNativeRuntimeEventKind::ModelOpenProgress,
        SkippyNativeRuntimeEventKind::ModelOpenFinished,
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
    ] {
        reporter(SkippyNativeRuntimeEvent {
            abi_version: 1,
            struct_size: 144,
            reserved0: 0,
            reserved1: 0,
            category: skippy_runtime::RuntimeEventCategory::ModelOpen,
            kind,
            sequence: 1,
            emitter: skippy_runtime::RuntimeEventEmitterKind::OpenThread,
            timestamp_mono_ns: 10,
            model_id: 11,
            stage_id: 0,
            session_id: 0,
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            failure_code: if kind == SkippyNativeRuntimeEventKind::ModelOpenFailedHandled {
                skippy_runtime::RuntimeEventFailureCode::ModelError
            } else {
                skippy_runtime::RuntimeEventFailureCode::None
            },
            status: skippy_runtime::Status::Ok,
            detail_bytes: b"prompt=private native detail".to_vec(),
            // Only ModelOpenProgress has a natural numeric summary to
            // report (e.g. bytes-processed alongside progress_current/
            // progress_total); Started/Finished/FailedHandled legitimately
            // carry none here, mirroring the existing per-kind
            // `failure_code` branch above. This exercises both the
            // `Some(..)` and `None` construction paths of the widened
            // struct within one test, proving neither shape breaks the
            // reporter/translate pipeline (translate does not read these
            // fields yet -- that is Task 9's producer-migration scope).
            numeric_summary_0: (kind == SkippyNativeRuntimeEventKind::ModelOpenProgress)
                .then_some(4096),
            numeric_summary_1: (kind == SkippyNativeRuntimeEventKind::ModelOpenProgress)
                .then_some(8192),
            numeric_summary_2: (kind == SkippyNativeRuntimeEventKind::ModelOpenProgress)
                .then_some(0),
            numeric_summary_3: (kind == SkippyNativeRuntimeEventKind::ModelOpenProgress)
                .then_some(0),
        });
    }

    let events = sink.take_events();
    let model_events = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                OutputEvent::Info { .. } | OutputEvent::Warning { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        model_events.len(),
        4,
        "every native callback should stay visible"
    );
    assert!(model_events.iter().all(|event| {
        matches!(
            event,
            OutputEvent::Info { .. } | OutputEvent::Warning { .. }
        )
    }));
    assert!(events.iter().all(|event| {
        !matches!(
            event,
            OutputEvent::LaunchPlan { .. }
                | OutputEvent::ApiReady { .. }
                | OutputEvent::WebserverReady { .. }
                | OutputEvent::ModelLoading { .. }
                | OutputEvent::ModelLoaded { .. }
                | OutputEvent::ModelReady { .. }
                | OutputEvent::RuntimeReady { .. }
        )
    }));
    let serialized = format!("{events:?}");
    for raw_value in [
        "/private/models/model-a.gguf",
        "prompt=private native detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "native presentation must not include {raw_value}"
        );
    }
}

#[test]
fn native_model_open_callbacks_map_only_static_operational_transitions() {
    assert_eq!(
        [
            SkippyNativeRuntimeEventKind::ModelOpenStarted,
            SkippyNativeRuntimeEventKind::ModelOpenProgress,
            SkippyNativeRuntimeEventKind::BackendDeviceSelected,
            SkippyNativeRuntimeEventKind::ModelOpenFinished,
            SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
        ]
        .into_iter()
        .filter_map(native_skippy_operational_event)
        .collect::<Vec<_>>(),
        vec![
            NativeSkippyOperationalEvent::ModelOpenStarted,
            NativeSkippyOperationalEvent::ModelOpenFinished,
            NativeSkippyOperationalEvent::ModelOpenFailed,
        ]
    );
}

struct FixedStoreClock(&'static str);

impl mesh_llm_log_store::Clock for FixedStoreClock {
    fn now(&self) -> String {
        self.0.to_string()
    }
}

#[tokio::test]
#[serial_test::serial]
async fn native_reporter_keeps_rich_presentation_while_audit_stays_static() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let clock: Arc<dyn mesh_llm_log_store::Clock> =
        Arc::new(FixedStoreClock("2026-08-07T12:00:00Z"));
    crate::initialize_logging_foundation_with_store_clock_for_test(
        &mesh_llm_config::LoggingConfig {
            application_state_root: Some(temporary_directory.path().join("logging")),
            ..Default::default()
        },
        clock,
    )
    .await;

    let service = crate::runtime::run_auto::start_run_auto_logging_service()
        .await
        .expect("startable logging service");

    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let mut reporter =
        skippy_native_model_open_event_reporter("/private/models/model-a.gguf".to_string(), None);
    for kind in [
        SkippyNativeRuntimeEventKind::ModelOpenStarted,
        SkippyNativeRuntimeEventKind::ModelOpenProgress,
        SkippyNativeRuntimeEventKind::ModelOpenFinished,
    ] {
        reporter(SkippyNativeRuntimeEvent {
            abi_version: 1,
            struct_size: 144,
            reserved0: 0,
            reserved1: 0,
            category: skippy_runtime::RuntimeEventCategory::ModelOpen,
            kind,
            sequence: 1,
            emitter: skippy_runtime::RuntimeEventEmitterKind::OpenThread,
            timestamp_mono_ns: 10,
            model_id: 11,
            stage_id: 0,
            session_id: 0,
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            failure_code: skippy_runtime::RuntimeEventFailureCode::None,
            status: skippy_runtime::Status::Ok,
            detail_bytes: b"prompt=private native detail".to_vec(),
            // None here, deliberately: this test's own concern is that the
            // audit trail stays static while presentation is rich -- an
            // invariant about rich vs. static rendering, not about the
            // struct extension. The extension's `Some(..)` construction
            // path is already exercised (for ModelOpenProgress) by the
            // sibling test above; duplicating that here would not add
            // coverage, only noise unrelated to what this test asserts.
            numeric_summary_0: None,
            numeric_summary_1: None,
            numeric_summary_2: None,
            numeric_summary_3: None,
        });
    }

    let presentation = sink.take_events();
    assert_eq!(presentation.len(), 3);
    let serialized_presentation = format!("{presentation:?}");
    for rich_context in [
        "sequence=1",
        "status=Ok",
        "emitter=OpenThread",
        "Opening native model 50%",
    ] {
        assert!(
            serialized_presentation.contains(rich_context),
            "presentation must keep {rich_context}"
        );
    }

    let audits = service
        .bus_ref()
        .drain()
        .into_iter()
        .map(|entry| {
            let audit: serde_json::Value =
                serde_json::from_str(&entry.payload).expect("audit payload");
            serde_json::json!({
                "kind": "audit",
                "level": audit["severity"],
                "message": audit["code"],
            })
        })
        .filter(|entry| {
            entry["message"]
                .as_str()
                // The audit bus is shared with startup/runtime events. This
                // test asserts the model-open callback's ordered pair, so
                // select that family without discarding any model-open
                // observation.
                .is_some_and(|code| code.starts_with("skippy_native_model_open_"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        audits,
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "skippy_native_model_open_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "skippy_native_model_open_finished",
            }),
        ]
    );
    let serialized_audits = format!("{audits:?}");
    for raw_value in [
        "model-a.gguf",
        "prompt=private native detail",
        "OpenThread",
        "sequence=",
    ] {
        assert!(
            !serialized_audits.contains(raw_value),
            "audit payloads must not include {raw_value}"
        );
    }

    assert!(service.shutdown().await);
}

/// D2 fix (event-system-fixes deferral): proves the actual native-event
/// wiring this file owns -- given a bound `progress_ingress`, a real
/// `ModelOpenProgress` native callback event produces a coalesced
/// `ModelLoadProgress` fact through the reporter closure (not just through
/// `submit_load_progress` called directly, which
/// `model_lifecycle::events::tests` already covers).
#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn native_model_open_progress_with_a_bound_ingress_submits_a_model_load_progress_fact() {
    use crate::runtime::model_lifecycle::LoadOperation;
    use crate::runtime_events::config::PROGRESS_EXPORT_INTERVAL;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::{ModelLoadingEventKind, RuntimeFact};

    clear_runtime_event_engine();
    let engine = RuntimeEventEngine::new();
    install_runtime_event_engine(engine.clone());

    let op = LoadOperation::begin("org/model");
    // Flush the four pre-resolution facts `begin()` submits so only the
    // progress fact below is pending when the flush is asserted.
    engine.drain();
    let ingress = op
        .progress_ingress()
        .expect("load reservation must still be live before native completion");
    #[cfg(feature = "dynamic-native-runtime")]
    let load_scope = ingress.scope();

    let mut reporter =
        skippy_native_model_open_event_reporter("org/model".to_string(), Some(ingress));
    reporter(SkippyNativeRuntimeEvent {
        abi_version: 1,
        struct_size: 144,
        reserved0: 0,
        reserved1: 0,
        category: skippy_runtime::RuntimeEventCategory::ModelOpen,
        kind: SkippyNativeRuntimeEventKind::ModelOpenProgress,
        sequence: 1,
        emitter: skippy_runtime::RuntimeEventEmitterKind::OpenThread,
        timestamp_mono_ns: 10,
        model_id: 11,
        stage_id: 0,
        session_id: 0,
        progress_current: 500,
        progress_total: 1000,
        progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        failure_code: skippy_runtime::RuntimeEventFailureCode::None,
        status: skippy_runtime::Status::Ok,
        detail_bytes: Vec::new(),
        numeric_summary_0: None,
        numeric_summary_1: None,
        numeric_summary_2: None,
        numeric_summary_3: None,
    });

    let flushed = engine.drain_up_to_at(None, std::time::Instant::now() + PROGRESS_EXPORT_INTERVAL);
    assert_eq!(
        flushed.applied, 1,
        "the ModelOpenProgress native event must produce exactly one coalesced ModelLoadProgress fact"
    );
    let has_progress = engine.replay().snapshot().into_iter().any(|frame| {
        matches!(
            frame.fact.as_ref(),
            RuntimeFact::ModelLoading(fact) if *fact.kind() == ModelLoadingEventKind::ModelLoadProgress
        )
    });
    assert!(
        has_progress,
        "ModelLoadProgress fact must have been published"
    );

    #[cfg(feature = "dynamic-native-runtime")]
    {
        let frames = engine.replay().snapshot();
        let frame = frames
            .iter()
            .find(|frame| frame.fact.kind_id() == "model_load_progress")
            .expect("native progress published");
        assert_eq!(frame.scope, load_scope);
        assert_eq!(
            frame
                .fact
                .data()
                .scope
                .model_id
                .as_ref()
                .map(|id| id.as_str()),
            Some("org/model")
        );
        let metadata = frame.fact.metadata().expect("native provenance retained");
        assert_eq!(
            metadata.producer,
            mesh_llm_runtime_event_contracts::ProducerSource::Native
        );
        let source = metadata.native_source.as_ref().expect("native source");
        assert_eq!(source.sequence, 1);
        assert_eq!(source.timestamp_mono_ns, 10);
        assert_eq!(source.model_id, 11);
        assert_eq!(source.struct_size, 144);
        assert!(metadata.wall_clock_unix_ns.is_some());
        assert!(metadata.process_monotonic_time.is_some());
        let wire = std::str::from_utf8(&frame.wire_bytes).expect("UTF-8 frame");
        assert!(wire.contains("\"producer\":\"native\""));
        assert!(wire.contains("\"native_sequence\":1"));
    }

    clear_runtime_event_engine();
}
