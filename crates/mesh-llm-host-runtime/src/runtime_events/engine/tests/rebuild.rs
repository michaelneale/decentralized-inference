use mesh_llm_runtime_event_contracts::OperationId;

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::RuntimeEventIngress;

#[test]
fn rebuild_bumps_generation_and_evicts_retained_replay() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    reservation.ingress().try_submit(terminal_success());
    engine.drain();
    assert_eq!(engine.replay().len(), 1);
    assert_eq!(engine.health().snapshot().rebuild_generation, 0);

    let generation = engine.rebuild();

    assert_eq!(generation, 1);
    assert_eq!(engine.health().snapshot().rebuild_generation, 1);
    assert!(
        engine.replay().is_empty(),
        "rebuild evicts replay for a coherent fresh window"
    );
    assert_eq!(engine.health().snapshot().replay_evicted, 1);
}

#[test]
fn ingress_sequence_continues_monotonically_across_a_rebuild() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let first = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    first.ingress().try_submit(terminal_success());
    engine.drain();
    let before = engine.wake().next_ingress_sequence();

    engine.rebuild();

    let after = engine.wake().next_ingress_sequence();
    assert!(after > before, "sequence must not reset on rebuild");
}
