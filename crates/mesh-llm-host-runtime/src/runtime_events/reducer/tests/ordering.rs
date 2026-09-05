use mesh_llm_runtime_event_contracts::{
    FactData, FactMetadata, FamilyFact, NativeEmitter, NativeRuntimeEventKind,
    NativeSequenceDomain, NativeSequenceEvidence, NativeSequenceObservation, Outcome,
    ProducerSource, RuntimeFact, Severity,
};

use super::fixtures::{input, input_with_native, progress_fact, scope, terminal_fact};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, RejectReason, apply};

fn native_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

fn process_native_fact(sequence: u64, evidence: NativeSequenceEvidence) -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::with_metadata(
        NativeRuntimeEventKind::RuntimeStopped,
        FactData::default(),
        FactMetadata {
            producer: ProducerSource::Native,
            severity: Severity::Info,
            wall_clock_unix_ns: None,
            process_monotonic_time: None,
            native_source: None,
            native_sequence: Some(NativeSequenceObservation::new(
                NativeSequenceDomain::Process,
                NativeEmitter::new(2),
                sequence,
                evidence,
            )),
        },
    ))
}

fn operation_native_fact(sequence: u64, evidence: NativeSequenceEvidence) -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::with_metadata(
        NativeRuntimeEventKind::RuntimeStopped,
        FactData::default(),
        FactMetadata {
            producer: ProducerSource::Native,
            severity: Severity::Info,
            wall_clock_unix_ns: None,
            process_monotonic_time: None,
            native_source: None,
            native_sequence: Some(NativeSequenceObservation::new(
                NativeSequenceDomain::Operation,
                NativeEmitter::new(2),
                sequence,
                evidence,
            )),
        },
    ))
}

#[test]
fn process_sequence_map_is_shared_until_a_process_observation_mutates_it() {
    let scope_a = scope();
    let scope_b = scope();
    let snapshot = ReducerSnapshot::empty();
    let initial_map = snapshot.process_native_sequences_ptr();

    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope_a, 0, native_fact()))
    else {
        panic!("ordinary facts must apply");
    };
    assert_eq!(snapshot.process_native_sequences_ptr(), initial_map);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_b,
            1,
            operation_native_fact(1, NativeSequenceEvidence::First),
        ),
    ) else {
        panic!("operation-domain observations must apply");
    };
    assert_eq!(snapshot.process_native_sequences_ptr(), initial_map);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_a,
            2,
            process_native_fact(10, NativeSequenceEvidence::First),
        ),
    ) else {
        panic!("process-domain observations must apply");
    };
    let first_process_map = snapshot.process_native_sequences_ptr();
    assert_ne!(first_process_map, initial_map);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_b,
            3,
            process_native_fact(10, NativeSequenceEvidence::Unchecked),
        ),
    ) else {
        panic!("a duplicate process-domain observation must still apply");
    };
    assert_eq!(
        snapshot.process_native_sequences_ptr(),
        first_process_map,
        "a duplicate process-domain observation must not clone an unchanged map"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_b,
            4,
            process_native_fact(11, NativeSequenceEvidence::Contiguous),
        ),
    ) else {
        panic!("a second process-domain observation must apply");
    };
    assert_ne!(
        snapshot.process_native_sequences_ptr(),
        first_process_map,
        "each process-domain mutation must remain isolated from its predecessor"
    );
}

#[test]
fn duplicate_fact_is_rejected_not_applied_twice() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(10)))
    else {
        panic!("first apply must succeed");
    };

    let outcome = apply(&snapshot, input(scope, 0, progress_fact(10)));
    assert!(matches!(
        outcome,
        ReduceOutcome::Rejected(RejectReason::Duplicate)
    ));
}

#[test]
fn stale_progress_does_not_regress_published_state() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(80)))
    else {
        panic!("first apply must succeed");
    };

    let outcome = apply(&snapshot, input(scope, 1, progress_fact(50)));
    assert!(matches!(
        outcome,
        ReduceOutcome::Rejected(RejectReason::StaleProgress)
    ));
    assert_eq!(
        snapshot
            .operation(scope)
            .expect("operation state")
            .last_progress_current,
        Some(80),
        "rejected regression must leave the last-valid value untouched"
    );
}

#[test]
fn native_sequence_gap_is_flagged_without_reordering() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 0, 10, native_fact()))
    else {
        panic!("first apply must succeed");
    };

    // Native sequence jumps from 10 to 15 (a gap of 4); ingress sequence is
    // still the next in order (1), so this must still apply.
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 1, 15, native_fact()))
    else {
        panic!("gapped native sequence must still be accepted");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.native_gap_count, 1);
    assert_eq!(
        state.last_native_sequence,
        Some(15),
        "ordering stays ingress-sequence order; native sequence is data only"
    );
}

#[test]
fn native_sequence_high_water_mark_survives_out_of_order_input() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 0, 10, native_fact()))
    else {
        panic!("first apply must succeed");
    };

    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 1, 5, native_fact()))
    else {
        panic!("out-of-order native input must still be accepted");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.last_native_sequence, Some(10));
    assert_eq!(state.native_gap_count, 0);
}

#[test]
fn native_sequence_maximum_is_overflow_safe_and_remains_high_water_mark() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input_with_native(scope, 0, u64::MAX, native_fact()),
    ) else {
        panic!("first apply must succeed");
    };

    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 1, 0, native_fact()))
    else {
        panic!("a regressing native sequence must still be accepted");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.last_native_sequence, Some(u64::MAX));
    assert_eq!(state.native_gap_count, 0);
}

#[test]
fn process_native_gap_evidence_survives_interleaved_operation_scopes() {
    let scope_a = scope();
    let scope_b = scope();
    let snapshot = ReducerSnapshot::empty();

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_a,
            0,
            process_native_fact(10, NativeSequenceEvidence::First),
        ),
    ) else {
        panic!("first process callback must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_b,
            1,
            process_native_fact(11, NativeSequenceEvidence::Contiguous),
        ),
    ) else {
        panic!("interleaved process callback must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope_a,
            2,
            process_native_fact(13, NativeSequenceEvidence::Gap),
        ),
    ) else {
        panic!("explicitly gapped process callback must apply");
    };

    assert_eq!(
        snapshot
            .operation(scope_a)
            .expect("scope a state")
            .native_gap_count,
        1
    );
    assert_eq!(
        snapshot
            .operation(scope_b)
            .expect("scope b state")
            .native_gap_count,
        0,
        "the process high-water mark must not turn another scope's callback into a gap"
    );
}

#[test]
fn process_sequence_jump_without_adapter_evidence_is_not_a_gap() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope,
            0,
            process_native_fact(20, NativeSequenceEvidence::First),
        ),
    ) else {
        panic!("first process callback must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope,
            1,
            process_native_fact(24, NativeSequenceEvidence::Unchecked),
        ),
    ) else {
        panic!("coalesced process callback must apply");
    };
    assert_eq!(
        snapshot
            .operation(scope)
            .expect("scope state")
            .native_gap_count,
        0,
        "the reducer must not infer drops from surviving reduced facts"
    );
}

#[test]
fn mixed_native_and_rust_facts_apply_in_ingress_sequence_order() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, native_fact())) else {
        panic!("native-origin fact must apply first");
    };
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 1, progress_fact(20)))
    else {
        panic!("rust-origin fact must apply second, by ingress sequence");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.last_ingress_sequence, 1);
    assert_eq!(state.last_progress_current, Some(20));

    // Same two facts, sequence numbers swapped: a fact bearing the LOWER
    // ingress sequence but arriving SECOND is stale, proving order is
    // ingress-sequence only, never arrival order or origin.
    let out_of_order = apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)));
    assert!(matches!(
        out_of_order,
        ReduceOutcome::Rejected(RejectReason::Duplicate)
    ));
}

#[test]
fn wall_clock_skew_never_changes_ordering_or_acceptance() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let mut early = input(scope, 0, progress_fact(10));
    early.wall_clock_hint = Some(1_000);
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, early) else {
        panic!("first apply must succeed");
    };

    // Wall clock goes BACKWARDS relative to the previous fact, but the
    // ingress sequence still advances normally; acceptance must be
    // unaffected because wall_clock_hint is inert data, never consulted.
    let mut later = input(scope, 1, progress_fact(20));
    later.wall_clock_hint = Some(500);
    let outcome = apply(&snapshot, later);
    assert!(matches!(outcome, ReduceOutcome::Applied(_)));
}
