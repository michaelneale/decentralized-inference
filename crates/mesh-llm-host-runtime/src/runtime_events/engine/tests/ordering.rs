use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn concurrent_terminal_writes_apply_in_ingress_sequence_order() {
    let engine = RuntimeEventEngine::with_capacity(64);
    let workers = 16;
    let barrier = Arc::new(Barrier::new(workers));
    let reservations: Vec<_> = (0..workers)
        .map(|_| {
            engine
                .reserve_root(OperationId::new(), synthetic_unknown)
                .expect("reserve")
        })
        .collect();

    thread::scope(|scope| {
        for reservation in &reservations {
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                reservation.ingress().try_submit(terminal_success())
            });
        }
    });

    let report = engine.drain();
    assert_eq!(report.applied, workers);

    let sequences: Vec<u64> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    let mut sorted = sequences.clone();
    sorted.sort_unstable();
    assert_eq!(
        sequences, sorted,
        "drain must apply wake entries in strictly increasing ingress-sequence order"
    );
}

#[test]
fn drain_up_to_a_budget_leaves_the_remainder_queued_for_the_next_pass() {
    let engine = RuntimeEventEngine::with_capacity(8);
    for _ in 0..4 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        reservation.ingress().try_submit(terminal_success());
    }

    let first = engine.drain_up_to(Some(2));
    assert_eq!(first.applied, 2);
    assert_eq!(first.left_queued, 2);

    let second = engine.drain_up_to(Some(2));
    assert_eq!(second.applied, 2);
    assert_eq!(second.left_queued, 0);
}
