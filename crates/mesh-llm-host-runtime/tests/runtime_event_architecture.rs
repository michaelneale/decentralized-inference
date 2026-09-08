use std::time::Duration;

use mesh_llm_host_runtime::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::{
    FactData, FactMetadata, FamilyFact, ModelLoadingEventKind, NativeRuntimeEventKind, OperationId,
    OperationScope, Outcome, ProducerSource, ReasonCode, RuntimeEventIngress, RuntimeFact,
    Severity, SubmitOutcome,
};

fn terminal() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::with_data(
        NativeRuntimeEventKind::RuntimeStopped,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

fn state() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

#[test]
fn a_terminal_does_not_discard_its_own_earlier_accepted_state() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine.reserve_root(OperationId::new(), terminal).unwrap();
    assert_eq!(
        reservation.ingress().try_submit(state()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        reservation.ingress().try_submit(terminal()),
        SubmitOutcome::Accepted
    );
    engine.drain();
    let frames = engine.replay().snapshot();
    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.sequence.get())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(frames[0].fact.kind_id(), "runtime_initialized");
    assert_eq!(frames[1].fact.kind_id(), "runtime_stopped");
}

#[test]
fn cancelled_state_cannot_join_a_new_reservation_reusing_the_scope() {
    let engine = RuntimeEventEngine::with_capacity(1);
    let scope = OperationId::new();
    let first = engine.reserve_root(scope, terminal).unwrap();
    assert_eq!(first.ingress().try_submit(state()), SubmitOutcome::Accepted);
    first.cancel();
    let second = engine.reserve_root(scope, terminal).unwrap();
    assert_eq!(
        second
            .ingress()
            .try_submit(RuntimeFact::NativeRuntime(FamilyFact::new(
                NativeRuntimeEventKind::NativeLibraryLoaded,
            ))),
        SubmitOutcome::Accepted,
    );
    assert_eq!(
        second.ingress().try_submit(terminal()),
        SubmitOutcome::Accepted
    );
    engine.drain();
    assert_eq!(
        engine
            .replay()
            .snapshot()
            .iter()
            .map(|frame| frame.sequence.get())
            .collect::<Vec<_>>(),
        vec![2, 3],
    );
}

#[test]
fn delayed_progress_does_not_regress_replay() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine.reserve_root(OperationId::new(), terminal).unwrap();
    engine.drain();
    assert_eq!(
        reservation
            .ingress()
            .try_submit(RuntimeFact::ModelLoading(FamilyFact::new(
                ModelLoadingEventKind::ModelLoadProgress,
            ))),
        SubmitOutcome::Coalesced,
    );
    engine
        .unreserved_ingress(OperationScope::root_only(OperationId::new()))
        .try_submit(state());
    engine.drain();
    std::thread::sleep(Duration::from_millis(120));
    engine.drain();
    let frames = engine.replay().snapshot();
    assert!(!frames.is_empty());
    assert!(
        frames
            .windows(2)
            .all(|pair| pair[0].sequence.get() < pair[1].sequence.get()),
        "a deferred observation must not move a reconnect cursor backwards"
    );
}

#[test]
fn shutdown_settles_a_reservation_whose_guard_is_still_alive() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine.reserve_root(OperationId::new(), terminal).unwrap();
    let report = engine.shutdown(None);
    assert_eq!(report.remaining_after_deadline, 0);
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].scope, reservation.scope());
    assert_eq!(frames[0].fact.data().outcome, Some(Outcome::Unknown));
    assert_eq!(
        frames[0].fact.data().reason,
        Some(ReasonCode::TerminalNotDelivered)
    );
    assert_eq!(
        reservation.ingress().try_submit(terminal()),
        SubmitOutcome::RejectedShuttingDown
    );
    drop(reservation);
    engine.drain();
    assert_eq!(engine.replay().snapshot().len(), 1);
}

#[test]
fn full_state_lane_preserves_accepted_inputs_and_rejects_the_new_key() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..4096 {
        assert_eq!(
            engine
                .unreserved_ingress(OperationScope::root_only(OperationId::new()))
                .try_submit(state()),
            SubmitOutcome::Accepted,
        );
    }
    assert_eq!(
        engine
            .unreserved_ingress(OperationScope::root_only(OperationId::new()))
            .try_submit(state()),
        SubmitOutcome::RejectedCapacity,
    );
    assert_eq!(engine.drain().applied, 4096);
}

#[test]
fn partial_terminal_drains_cannot_publish_ahead_of_an_older_queued_terminal() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let first = engine.reserve_root(OperationId::new(), terminal).unwrap();
    let second = engine.reserve_root(OperationId::new(), terminal).unwrap();
    assert_eq!(
        first.ingress().try_submit(terminal()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        second.ingress().try_submit(terminal()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .unreserved_ingress(OperationScope::root_only(OperationId::new()))
            .try_submit(state()),
        SubmitOutcome::Accepted
    );
    engine.drain_up_to(Some(1));
    engine.drain();
    let sequences: Vec<_> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences, vec![1, 2, 3]);
}

#[test]
fn shutdown_chunks_preserve_order_across_state_and_terminal_lanes() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..65 {
        assert_eq!(
            engine
                .unreserved_ingress(OperationScope::root_only(OperationId::new()))
                .try_submit(state()),
            SubmitOutcome::Accepted
        );
    }
    let reservation = engine.reserve_root(OperationId::new(), terminal).unwrap();
    assert_eq!(
        reservation.ingress().try_submit(terminal()),
        SubmitOutcome::Accepted
    );
    assert_eq!(engine.shutdown(None).remaining_after_deadline, 0);
    let sequences: Vec<_> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences, (1..=66).collect::<Vec<_>>());
}

#[test]
fn shutdown_reports_pending_unreserved_state_when_its_budget_is_zero() {
    let engine = RuntimeEventEngine::with_capacity(4);
    assert_eq!(
        engine
            .unreserved_ingress(OperationScope::root_only(OperationId::new()))
            .try_submit(state()),
        SubmitOutcome::Accepted
    );
    let report = engine.shutdown(Some(0));
    assert!(report.remaining_after_deadline > 0);
    assert!(engine.health().snapshot().shutdown_degraded > 0);
}

#[tokio::test]
async fn attachment_before_the_first_drain_receives_the_first_event() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine
        .unreserved_ingress(OperationScope::root_only(OperationId::new()))
        .try_submit(state());
    let mut attachment = engine.attach().expect("attach");
    assert_eq!(attachment.published_frontier, 0);
    assert!(attachment.replay.is_empty());
    engine.drain();
    let frame = tokio::time::timeout(Duration::from_secs(2), attachment.subscription.recv())
        .await
        .expect("first live event must arrive")
        .expect("subscription remains live");
    assert_eq!(frame.sequence.get(), 1);
}

#[test]
fn producer_metadata_survives_the_real_ingress_and_publication_path() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let metadata = FactMetadata {
        producer: ProducerSource::Reconciled,
        severity: Severity::Warning,
        wall_clock_unix_ns: Some(123456789),
        process_monotonic_time: Some(Duration::from_millis(7)),
        native_source: None,
        native_sequence: None,
    };
    engine
        .unreserved_ingress(OperationScope::root_only(OperationId::new()))
        .try_submit(state().with_metadata(metadata.clone()));
    engine.drain();
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].fact.metadata(), Some(&metadata));
    let wire = std::str::from_utf8(&frames[0].wire_bytes).unwrap();
    assert!(wire.contains("\"producer\":\"reconciled\""));
    assert!(wire.contains("\"severity\":\"warning\""));
    assert!(wire.contains("\"wall_clock_unix_ns\":123456789"));
    assert!(wire.contains("\"process_monotonic_ns\":7000000"));
}
