use super::*;
use mesh_llm_runtime_event_contracts::Outcome;

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn session_becomes_active_on_start_and_idle_on_completion() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    assert!(
        engine.state_lane_kinds().contains(&"session_active"),
        "session must become active on generation start"
    );
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert!(
        session_kinds(&engine).contains(&SessionEventKind::SessionIdle),
        "session must become idle once the generation resolves"
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn prefill_completed_carries_the_real_time_to_first_token_signal() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    // `GenerationReceipt::test_fixture` sets `request_to_first_token_us: Some(1)`.
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    let kinds = prefill_kinds(&engine);
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == PrefillEventKind::PrefillCompleted)
            .count(),
        1,
        "exactly one prefill terminal per generation"
    );
    clear_runtime_event_engine();
}

/// Plan task 12 round 8: `GenerationReceipt::full_state` no longer drives a
/// second `RuntimeStateExportCompleted` fact from this adapter (round 7 had
/// derived one from a populated digest). `RuntimeState::export_full_state`'s
/// own `SessionLifecycleObserver` wiring (`runtime_events/session.rs`) is
/// now the single source, for every caller including ones that never touch
/// `GenerationReceipt` -- so a receipt carrying a full-state digest must not
/// ALSO cause the generation adapter itself to emit that kind.
#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn a_receipt_carrying_a_full_state_digest_does_not_double_emit_from_the_generation_adapter() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(5, 6, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(
            GenerationReceipt::test_fixture_with_full_state(
                5,
                6,
                GenerationTermination::MaxTokens,
                4096,
            ),
        ))
        .unwrap();
    engine.drain();
    assert!(
        !engine
            .state_lane_kinds()
            .contains(&"runtime_state_export_completed"),
        "the generation adapter must not derive an export fact from \
         GenerationReceipt::full_state any more -- that is now the \
         session/runtime-state observer's job, at the real export call"
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn generation_completed_reports_success_and_cancelled_receipt_reports_cancelled() {
    let completed = receipt_terminal_fact(&receipt(1, 2, GenerationTermination::MaxTokens));
    let RuntimeFact::Generation(completed) = completed else {
        panic!("expected a Generation fact");
    };
    assert_eq!(completed.data().outcome, Some(Outcome::Success));

    // The load-bearing fix: `GenerationTermination::Cancelled` on the
    // RECEIPT path (a mid-stream cancellation that still produced a
    // target-authoritative receipt) must report `Outcome::Cancelled`, the
    // same disposition the explicit `GenerationAbort` path already reports
    // -- never a blanket `Success` for a request that was cancelled.
    let cancelled = receipt_terminal_fact(&receipt(1, 2, GenerationTermination::Cancelled));
    let RuntimeFact::Generation(cancelled) = cancelled else {
        panic!("expected a Generation fact");
    };
    assert_eq!(cancelled.data().outcome, Some(Outcome::Cancelled));

    let RuntimeFact::Generation(explicit_abort) = abort_terminal_fact() else {
        panic!("expected a Generation fact");
    };
    assert_eq!(
        explicit_abort.data().outcome,
        cancelled.data().outcome,
        "the explicit-abort and cancelled-receipt paths must agree on outcome"
    );
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn first_token_produced_fires_exactly_once_on_the_first_commit() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    for count in [1_usize, 2, 3] {
        adapter
            .try_submit(GenerationLifecycleObservation::Committed(
                GenerationCommit {
                    request_id: 1,
                    session_id: 2,
                    generated_token_count: count,
                    token_ids: vec![count as i32].into_boxed_slice(),
                },
            ))
            .unwrap();
    }
    assert_eq!(
        engine
            .state_lane_kinds()
            .iter()
            .filter(|kind| **kind == "first_token_produced")
            .count(),
        1,
        "first_token_produced must fire exactly once, not once per commit"
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn stop_condition_reached_co_emits_only_for_callback_stop_termination() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();
    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::CallbackStop,
        )))
        .unwrap();
    engine.drain();
    assert!(
        generation_kinds(&engine).contains(&GenerationEventKind::StopConditionReached),
        "CallbackStop termination must co-emit stop_condition_reached"
    );
    clear_runtime_event_engine();

    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();
    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(3, 4, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            3,
            4,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert!(
        !generation_kinds(&engine).contains(&GenerationEventKind::StopConditionReached),
        "budget-exhaustion (MaxTokens) termination must NOT report a stop condition"
    );
    clear_runtime_event_engine();
}
