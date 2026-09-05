use super::*;

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn duplicate_started_for_the_same_key_cascades_the_superseded_generation() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    assert_eq!(
        engine.occupied_count(),
        2,
        "first generation's root + prefill"
    );

    // A second `Started` for the SAME (request_id, session_id) key, before
    // the first ever resolved. The chosen behavior: the stale tracking is
    // replaced and its reservations drop, synthesizing correct
    // `terminal_not_delivered` terminals for the superseded generation --
    // never a silent leak, and never an incorrect "successful completion"
    // synthesized for a generation that never actually finished.
    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    engine.drain();

    let kinds = generation_kinds(&engine);
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == GenerationEventKind::GenerationFailed)
            .count(),
        1,
        "the superseded generation must synthesize exactly one terminal_not_delivered terminal, not silently vanish"
    );
    // The new generation's own root+prefill are still open (not resolved).
    assert_eq!(engine.occupied_count(), 2);

    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert_eq!(
        engine.occupied_count(),
        0,
        "the NEW generation resolves cleanly, unaffected by the superseded one"
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn composite_ingress_delivers_to_both_sinks_when_one_fails() {
    struct FailingSink;
    impl GenerationLifecycleIngress for FailingSink {
        fn try_submit(&self, _observation: GenerationLifecycleObservation) -> Result<()> {
            Err(anyhow::anyhow!("plugin sink unavailable"))
        }
    }

    struct RecordingSink(std::sync::Mutex<usize>);
    impl GenerationLifecycleIngress for RecordingSink {
        fn try_submit(&self, _observation: GenerationLifecycleObservation) -> Result<()> {
            *self.0.lock().unwrap() += 1;
            Ok(())
        }
    }

    let recording = Arc::new(RecordingSink(std::sync::Mutex::new(0)));
    let composite = skippy_server::frontend::CompositeGenerationLifecycleIngress::new(vec![
        Arc::new(FailingSink),
        recording.clone(),
    ]);
    let result = composite.try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)));
    assert!(result.is_err(), "the failing sink's error surfaces");
    assert_eq!(
        *recording.0.lock().unwrap(),
        1,
        "the other sink must still receive the observation"
    );
}
