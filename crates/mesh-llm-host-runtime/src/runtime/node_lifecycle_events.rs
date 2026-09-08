//! Runtime-event producer wiring for whole-node §8.14 lifecycle facts
//! (plan task 10, `.omo/plans/event-system.md` line 278/280):
//! `node_starting`, `node_accepting_requests`, `node_draining`,
//! `node_stopped`.
//!
//! Unlike stage/topology operations, there is no single bounded
//! "node lifecycle operation" object spanning the whole process for these
//! four facts to reserve against -- each is a StateTransition-class,
//! unreserved, fire-and-forget emission at an already-existing one-shot
//! call site (engine install, the existing one-shot `RuntimeReady` output
//! event, the existing `broadcast_leaving()`/shutdown sequence). Local-only:
//! never gossiped, matching every other producer this task added.

use mesh_llm_runtime_event_contracts::{
    FactData, NodeAvailabilityEventKind, NodeAvailabilityFact, OperationId, OperationScope,
    Outcome, ReasonCode, RuntimeEventIngress, RuntimeFact,
};

use crate::runtime_events::runtime_event_engine;

/// StateTransition-class kinds route through `submit_state_transition`,
/// which ignores the reservation handle entirely -- `unreserved_ingress`
/// is correct and sufficient for these three.
fn emit_state_transition(kind: NodeAvailabilityEventKind) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
    let _ = ingress.try_submit(RuntimeFact::NodeAvailability(
        NodeAvailabilityFact::with_data(kind, FactData::default()),
    ));
}

/// `NodeStopped` is the one Terminal-class kind in this family
/// (delivery/execution.rs). A Terminal-class fact submitted through
/// `unreserved_ingress` always reports `TerminalDeliveryFailed` and never
/// reaches any consumer -- confirmed by a real, caught red (see
/// `.omo/evidence/event-system/task-10/red2.txt`). Reserve-and-
/// immediately-resolve, mirroring `model_lifecycle::events::
/// reconcile_process_crash`'s established pattern for a terminal with no
/// live bounded operation to attach to.
fn emit_node_stopped_terminal() {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    let synthetic: fn() -> RuntimeFact = || {
        RuntimeFact::NodeAvailability(NodeAvailabilityFact::with_data(
            NodeAvailabilityEventKind::NodeStopped,
            FactData {
                outcome: Some(Outcome::Unknown),
                reason: Some(ReasonCode::TerminalNotDelivered),
                ..FactData::default()
            },
        ))
    };
    let Some(root) = engine.reserve_root(OperationId::new(), synthetic) else {
        return;
    };
    let _ = root.ingress().try_submit(RuntimeFact::NodeAvailability(
        NodeAvailabilityFact::with_data(
            NodeAvailabilityEventKind::NodeStopped,
            FactData {
                outcome: Some(Outcome::Success),
                ..FactData::default()
            },
        ),
    ));
}

/// Call immediately after installing the runtime-event engine (the
/// earliest point a fact can even be delivered).
pub(crate) fn emit_node_starting() {
    emit_state_transition(NodeAvailabilityEventKind::NodeStarting);
}

/// Call at the same one-shot moment the existing `OutputEvent::RuntimeReady`
/// fires (both `--auto` and `--local-model-only` paths already compute this
/// moment exactly once; see `startup_handles.rs::mark_ready_and_maybe_emit`
/// and `local_model_only.rs::run_loaded_local_model`).
pub(crate) fn emit_node_accepting_requests() {
    emit_state_transition(NodeAvailabilityEventKind::NodeAcceptingRequests);
}

/// Call at the start of the existing graceful-shutdown sequence, alongside
/// `mesh::Node::broadcast_leaving()`.
pub(crate) fn emit_node_draining() {
    emit_state_transition(NodeAvailabilityEventKind::NodeDraining);
}

/// Call at the end of the existing graceful-shutdown sequence, once every
/// mesh-facing subsystem has stopped.
pub(crate) fn emit_node_stopped() {
    emit_node_stopped_terminal();
}

#[cfg(test)]
mod tests {
    use super::{
        emit_node_accepting_requests, emit_node_draining, emit_node_starting, emit_node_stopped,
    };
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn emits_all_four_node_lifecycle_kinds() {
        let engine = install_test_engine();
        emit_node_starting();
        emit_node_accepting_requests();
        emit_node_draining();
        emit_node_stopped();
        engine.drain();
        // Every delivery class in this family -- the three
        // StateTransition kinds and the one Terminal kind -- reaches
        // replay() once drained.
        let kinds = engine
            .replay()
            .snapshot()
            .into_iter()
            .map(|frame| frame.fact.kind_id().to_string())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&"node_starting".to_string()));
        assert!(kinds.contains(&"node_accepting_requests".to_string()));
        assert!(kinds.contains(&"node_draining".to_string()));
        assert!(kinds.contains(&"node_stopped".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    fn absent_engine_never_panics() {
        clear_runtime_event_engine();
        emit_node_starting();
        emit_node_accepting_requests();
        emit_node_draining();
        emit_node_stopped();
    }
}
