use std::time::{Duration, Instant};

use crate::{
    DiagnosticEventKind, GenerationEventKind, RequestEventKind, RuntimeEventIngress, SubmitOutcome,
};

use super::support::{SaturatingIngress, diagnostic_fact, generation_fact, request_fact};

const SATURATED_SUBMISSION_DEADLINE: Duration = Duration::from_millis(50);

fn assert_object_safe(_: &dyn RuntimeEventIngress) {}

#[test]
fn ingress_contract_is_object_safe() {
    // Given
    let (ingress, _receiver) = SaturatingIngress::connected(1);

    // When
    let erased: &dyn RuntimeEventIngress = &ingress;

    // Then
    assert_object_safe(erased);
}

#[test]
fn accepted_submission_reports_accepted() {
    // Given
    let (ingress, _receiver) = SaturatingIngress::connected(1);

    // When
    let outcome = ingress.try_submit(request_fact(RequestEventKind::RequestReceived));

    // Then
    assert_eq!(outcome, SubmitOutcome::Accepted);
}

#[test]
fn saturated_progress_submission_returns_without_a_consumer() {
    // Given
    let (ingress, _receiver) = SaturatingIngress::connected(1);
    assert_eq!(
        ingress.try_submit(request_fact(RequestEventKind::RequestReceived)),
        SubmitOutcome::Accepted
    );

    // When
    let started = Instant::now();
    let outcome = ingress.try_submit(generation_fact(GenerationEventKind::GenerationProgress));
    let elapsed = started.elapsed();

    // Then
    assert_eq!(outcome, SubmitOutcome::DroppedProgress);
    assert!(
        elapsed < SATURATED_SUBMISSION_DEADLINE,
        "elapsed: {elapsed:?}"
    );
}

#[test]
fn saturated_class_outcomes_are_explicit() {
    // Given
    let (ingress, _receiver) = SaturatingIngress::connected(1);
    assert_eq!(
        ingress.try_submit(request_fact(RequestEventKind::RequestReceived)),
        SubmitOutcome::Accepted
    );

    // When
    let outcomes = [
        ingress.try_submit(request_fact(RequestEventKind::RequestQueued)),
        ingress.try_submit(diagnostic_fact(DiagnosticEventKind::WarningRaised)),
        ingress.try_submit(request_fact(RequestEventKind::RequestCompleted)),
    ];

    // Then
    assert_eq!(
        outcomes,
        [
            SubmitOutcome::RejectedCapacity,
            SubmitOutcome::DroppedDiagnostic,
            SubmitOutcome::TerminalDeliveryFailed,
        ]
    );
}

#[test]
fn disconnected_ingress_reports_shutdown() {
    // Given
    let (ingress, receiver) = SaturatingIngress::connected(1);
    drop(receiver);

    // When
    let outcome = ingress.try_submit(request_fact(RequestEventKind::RequestReceived));

    // Then
    assert_eq!(outcome, SubmitOutcome::RejectedShuttingDown);
}
