//! Review defect D8 (`.omo/plans/event-system-fixes.md` task 5), exercised
//! through the real [`SkippyGenerationRuntimeEventAdapter`] rather than the
//! bare engine: an OpenAI request-level root settling while this adapter's
//! generation/prefill children are still occupied must never lose a
//! `generation_completed` terminal, across many concurrent roots.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, OperationId, Outcome, RequestEventKind, RuntimeEventIngress, RuntimeFact,
};

use super::*;

fn request_completed_fact() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

fn synthetic_request_terminal() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(mesh_llm_runtime_event_contracts::ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn no_generation_completed_terminal_is_lost_across_concurrent_roots_with_children() {
    const ROOTS: usize = 8;
    let engine = install_test_engine();
    let adapter = Arc::new(SkippyGenerationRuntimeEventAdapter::new());

    let stop = Arc::new(AtomicBool::new(false));
    let drainer = {
        let engine = engine.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                engine.drain();
                thread::yield_now();
            }
        })
    };

    let barrier = Arc::new(Barrier::new(ROOTS));
    let workers: Vec<_> = (0..ROOTS)
        .map(|i| {
            let engine = engine.clone();
            let adapter = adapter.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                let mut root_bytes = [0_u8; 16];
                root_bytes[0..8].copy_from_slice(&(i as u64).to_le_bytes());
                let root_id = OperationId::from_bytes(root_bytes);
                let root = engine
                    .reserve_root(root_id, synthetic_request_terminal)
                    .expect("reserve root");
                barrier.wait();

                adapter
                    .try_submit(GenerationLifecycleObservation::Started(start(
                        i as u64,
                        i as u64,
                        Some(root_bytes),
                    )))
                    .unwrap();

                // The request-level root settles NOW, potentially while the
                // generation/prefill children this adapter manages are
                // still occupied -- review defect D8's exact scenario.
                let _ = root.ingress().try_submit(request_completed_fact());

                adapter
                    .try_submit(GenerationLifecycleObservation::Completed(receipt(
                        i as u64,
                        i as u64,
                        GenerationTermination::MaxTokens,
                    )))
                    .unwrap();
            })
        })
        .collect();

    for worker in workers {
        worker.join().expect("worker thread panicked");
    }
    stop.store(true, Ordering::Relaxed);
    drainer.join().expect("drainer thread panicked");

    // A safety net, not a requirement: everything above settles well
    // inside the real grace window since no real sleep ever happens, but
    // simulate grace expiry deterministically in case a worker was
    // starved, rather than depending on wall-clock luck.
    for _ in 0..8 {
        engine.drain();
    }
    let far_future =
        std::time::Instant::now() + crate::runtime_events::config::CHILD_SETTLE_GRACE * 2;
    engine.drain_up_to_at(None, far_future);
    engine.drain_up_to_at(None, far_future);

    assert_eq!(
        engine.occupied_count(),
        0,
        "every reservation must eventually settle"
    );
    assert_eq!(
        engine.health().snapshot().terminal_delivery_failed,
        0,
        "no child terminal should ever be lost to a premature root release"
    );
    assert_eq!(adapter.delivery_failures(), 0);
    let completed = generation_kinds(&engine)
        .into_iter()
        .filter(|kind| *kind == GenerationEventKind::GenerationCompleted)
        .count();
    assert_eq!(
        completed, ROOTS,
        "every generation_completed terminal must be published"
    );
    clear_runtime_event_engine();
}
