//! Runtime-event producer wiring for split-topology lifecycle (plan task
//! 10, `.omo/plans/event-system.md` line 278; spec §8.6 topology/connection
//! bullets, §8.14 node-degraded/unavailable bullets).
//!
//! Two independently-bounded operations:
//! - `TopologyAssemblyOperation` wraps one `load_split_runtime_generation`
//!   call: `TopologyAssembling` -> terminal (`TopologyReady` /
//!   `TopologyUnavailable`).
//! - `TopologyWithdrawalOperation` wraps one withdrawal of an active
//!   generation: begin (no start event -- withdrawal is only ever a
//!   terminal decision) -> terminal `TopologyUnavailable`.
//!
//! `report_recovery_decision` is a StateTransition-only, unreserved
//! emission (no bounded operation -- it observes the coordinator's own
//! already-computed `SplitLossRecoveryDecision` on every health tick /
//! peer-change reaction) covering `TopologyDegraded`,
//! `StageConnectionLost`, `StageConnectionRecovered`, and the co-derived
//! §8.14 `NodeDegraded`/`NodeUnavailable` facts. Best-effort throughout,
//! matching the established degrade-on-absent-engine contract.

use mesh_llm_runtime_event_contracts::{
    FactData, NodeAvailabilityEventKind, NodeAvailabilityFact, OperationId, OperationScope,
    Outcome, ReasonCode, RuntimeEventIngress, RuntimeFact, ScopeIdentities, StageTopologyEventKind,
    StageTopologyFact, TopologyId,
};

use crate::runtime_events::engine::{OperationReservation, RuntimeEventEngine};
use crate::runtime_events::runtime_event_engine;

fn topology_scope(topology_id: &str) -> FactData {
    FactData {
        scope: ScopeIdentities {
            topology_id: TopologyId::new(topology_id).ok(),
            ..ScopeIdentities::default()
        },
        ..FactData::default()
    }
}

fn terminal_not_delivered() -> FactData {
    FactData {
        outcome: Some(Outcome::Unknown),
        reason: Some(ReasonCode::TerminalNotDelivered),
        ..FactData::default()
    }
}

fn synthetic_unavailable_terminal() -> RuntimeFact {
    RuntimeFact::StageTopology(StageTopologyFact::with_data(
        StageTopologyEventKind::TopologyUnavailable,
        terminal_not_delivered(),
    ))
}

fn submit(reservation: &OperationReservation, kind: StageTopologyEventKind, data: FactData) {
    let _ =
        reservation
            .ingress()
            .try_submit(RuntimeFact::StageTopology(StageTopologyFact::with_data(
                kind, data,
            )));
}

fn emit_state(engine: &std::sync::Arc<RuntimeEventEngine>, fact: RuntimeFact) {
    let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
    let _ = ingress.try_submit(fact);
}

/// One full split-topology assembly attempt (`load_split_runtime_generation`).
pub(super) struct TopologyAssemblyOperation {
    root: Option<OperationReservation>,
    topology_id: String,
}

impl Drop for TopologyAssemblyOperation {
    fn drop(&mut self) {
        let Some(root) = self.root.take() else {
            return;
        };
        // Keep the fallback terminal in the same topology scope as the
        // assembly attempt.  The generic reservation guard only knows the
        // family, so its synthetic terminal would otherwise lose topology
        // identity and make the dropped operation indistinguishable from a
        // different topology.
        let mut data = terminal_not_delivered();
        data.scope = topology_scope(&self.topology_id).scope;
        submit(&root, StageTopologyEventKind::TopologyUnavailable, data);
    }
}

impl TopologyAssemblyOperation {
    pub(super) fn begin(topology_id: &str) -> Self {
        let root = runtime_event_engine().and_then(|engine| {
            engine.reserve_root(OperationId::new(), synthetic_unavailable_terminal)
        });
        if let Some(root) = &root {
            submit(
                root,
                StageTopologyEventKind::TopologyAssembling,
                topology_scope(topology_id),
            );
        }
        Self {
            root,
            topology_id: topology_id.to_string(),
        }
    }

    pub(super) fn ready(mut self) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::TopologyReady,
                FactData {
                    outcome: Some(Outcome::Success),
                    ..topology_scope(&self.topology_id)
                },
            );
        }
    }

    pub(super) fn unavailable(mut self, reason: ReasonCode) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::TopologyUnavailable,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..topology_scope(&self.topology_id)
                },
            );
        }
    }
}

/// One withdrawal of an active topology generation.
pub(super) struct TopologyWithdrawalOperation {
    root: Option<OperationReservation>,
    topology_id: String,
}

impl Drop for TopologyWithdrawalOperation {
    fn drop(&mut self) {
        let Some(root) = self.root.take() else {
            return;
        };
        let mut data = terminal_not_delivered();
        data.scope = topology_scope(&self.topology_id).scope;
        submit(&root, StageTopologyEventKind::TopologyUnavailable, data);
    }
}

impl TopologyWithdrawalOperation {
    pub(super) fn begin(topology_id: &str) -> Self {
        let root = runtime_event_engine().and_then(|engine| {
            engine.reserve_root(OperationId::new(), synthetic_unavailable_terminal)
        });
        Self {
            root,
            topology_id: topology_id.to_string(),
        }
    }

    pub(super) fn withdrawn(mut self) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::TopologyUnavailable,
                FactData {
                    outcome: Some(Outcome::Success),
                    reason: Some(ReasonCode::StageUnavailable),
                    ..topology_scope(&self.topology_id)
                },
            );
        }
        if let Some(engine) = runtime_event_engine() {
            emit_state(
                &engine,
                RuntimeFact::NodeAvailability(NodeAvailabilityFact::with_data(
                    NodeAvailabilityEventKind::NodeUnavailable,
                    topology_scope(&self.topology_id),
                )),
            );
        }
    }
}

/// §8.6 `stage connection established` -- call once a downstream stage in a
/// `load_downstream_split_runtime_stages` iteration accepts its load and
/// becomes this coordinator's live downstream peer (the same point
/// `loading.rs` already records the peer into `ready_by_stage`/
/// `*downstream`). StateTransition-class, unreserved, local-only.
pub(super) fn emit_stage_connection_established(
    topology_id: &str,
    stage_id: &str,
    stage_index: u32,
) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    emit_state(
        &engine,
        RuntimeFact::StageTopology(StageTopologyFact::with_data(
            StageTopologyEventKind::StageConnectionEstablished,
            {
                let mut data = topology_scope(topology_id);
                data.scope.stage = mesh_llm_runtime_event_contracts::StageId::new(stage_id)
                    .ok()
                    .map(|id| {
                        mesh_llm_runtime_event_contracts::StageIdentity::new(id, stage_index)
                    });
                data
            },
        )),
    );
}

/// Newly-lost node ids (present in `unavailable_now` but not in
/// `previously_unavailable`) and newly-recovered node ids (present in
/// `previously_unavailable` but absent from `unavailable_now`). A pure
/// function so `handle_loss_recovery`'s flap/recovery classification is
/// unit-testable independent of the async coordinator.
pub(super) fn classify_connection_changes(
    previously_unavailable: &[iroh::EndpointId],
    unavailable_now: &[iroh::EndpointId],
) -> (Vec<iroh::EndpointId>, Vec<iroh::EndpointId>) {
    let lost = unavailable_now
        .iter()
        .filter(|node| !previously_unavailable.contains(node))
        .copied()
        .collect();
    let recovered = previously_unavailable
        .iter()
        .filter(|node| !unavailable_now.contains(node))
        .copied()
        .collect();
    (lost, recovered)
}

fn node_scope(topology_id: &str, node_id: iroh::EndpointId) -> FactData {
    let mut data = topology_scope(topology_id);
    data.summary = mesh_llm_runtime_event_contracts::HumanSummary::new(&format!(
        "node={}",
        node_id.fmt_short()
    ))
    .ok();
    data
}

/// Observes one `handle_loss_recovery` classification and emits every
/// StateTransition-class fact it implies: `StageConnectionLost`/
/// `StageConnectionRecovered` per node, `TopologyDegraded` when the
/// coordinator is actively recovering, and the co-derived §8.14
/// `NodeDegraded`/`NodeUnavailable` (a split-serving node's own degraded/
/// unavailable state IS this node's contribution to §8.14, since this
/// coordinator only exists while this node is the local stage-0 owner).
pub(super) fn report_recovery_decision(
    topology_id: &str,
    healthy: bool,
    withdrawing: bool,
    previously_unavailable: &[iroh::EndpointId],
    unavailable_now: &[iroh::EndpointId],
    local_capacity_available: bool,
) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    emit_state(
        &engine,
        RuntimeFact::NodeAvailability(NodeAvailabilityFact::with_data(
            NodeAvailabilityEventKind::ResourcePressureChanged,
            FactData {
                outcome: Some(if local_capacity_available {
                    Outcome::Success
                } else {
                    Outcome::Failure
                }),
                ..topology_scope(topology_id)
            },
        )),
    );
    let (lost, recovered) = classify_connection_changes(previously_unavailable, unavailable_now);
    for node in &lost {
        emit_state(
            &engine,
            RuntimeFact::StageTopology(StageTopologyFact::with_data(
                StageTopologyEventKind::StageConnectionLost,
                node_scope(topology_id, *node),
            )),
        );
    }
    for node in &recovered {
        emit_state(
            &engine,
            RuntimeFact::StageTopology(StageTopologyFact::with_data(
                StageTopologyEventKind::StageConnectionRecovered,
                node_scope(topology_id, *node),
            )),
        );
    }
    if healthy {
        return;
    }
    emit_state(
        &engine,
        RuntimeFact::StageTopology(StageTopologyFact::with_data(
            StageTopologyEventKind::TopologyDegraded,
            topology_scope(topology_id),
        )),
    );
    let node_kind = if withdrawing {
        NodeAvailabilityEventKind::NodeUnavailable
    } else {
        NodeAvailabilityEventKind::NodeDegraded
    };
    emit_state(
        &engine,
        RuntimeFact::NodeAvailability(NodeAvailabilityFact::with_data(
            node_kind,
            topology_scope(topology_id),
        )),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    fn node(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        iroh::SecretKey::from_bytes(&bytes).public()
    }

    #[test]
    fn classify_connection_changes_detects_lost_and_recovered() {
        let a = node(1);
        let b = node(2);
        let c = node(3);
        // a was already unavailable, b newly lost, c not involved.
        let (lost, recovered) = classify_connection_changes(&[a], &[a, b]);
        assert_eq!(lost, vec![b]);
        assert!(recovered.is_empty());

        // a recovers (no longer unavailable), b remains lost.
        let (lost2, recovered2) = classify_connection_changes(&[a, b], &[b]);
        assert!(lost2.is_empty());
        assert_eq!(recovered2, vec![a]);
        let _ = c;
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn topology_assembly_success_emits_assembling_then_ready() {
        let engine = install_test_engine();
        let op = TopologyAssemblyOperation::begin("topo-a");
        assert_eq!(engine.occupied_count(), 1);
        op.ready();
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let kinds = engine
            .replay()
            .snapshot()
            .into_iter()
            .map(|frame| frame.fact.kind_id().to_string())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&"topology_ready".to_string()));
        assert!(kinds.contains(&"topology_assembling".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn topology_assembly_failure_resolves_topology_unavailable() {
        let engine = install_test_engine();
        let op = TopologyAssemblyOperation::begin("topo-a");
        op.unavailable(ReasonCode::StageUnavailable);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let kinds = engine
            .replay()
            .snapshot()
            .into_iter()
            .map(|frame| frame.fact.kind_id().to_string())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&"topology_unavailable".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_an_assembly_operation_synthesizes_topology_unavailable() {
        let engine = install_test_engine();
        {
            let op = TopologyAssemblyOperation::begin("topo-dropped");
            assert_eq!(engine.occupied_count(), 1);
            drop(op);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let unavailable = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::StageTopology(fact)
                    if *fact.kind() == StageTopologyEventKind::TopologyUnavailable =>
                {
                    Some(
                        fact.data()
                            .scope
                            .topology_id
                            .as_ref()
                            .map(|topology_id| topology_id.as_str().to_string()),
                    )
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(unavailable, vec![Some("topo-dropped".to_string())]);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_a_withdrawal_operation_preserves_topology_identity() {
        let engine = install_test_engine();
        {
            let op = TopologyWithdrawalOperation::begin("topo-withdrawal-dropped");
            assert_eq!(engine.occupied_count(), 1);
            drop(op);
        }
        engine.drain();
        let unavailable = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::StageTopology(fact)
                    if *fact.kind() == StageTopologyEventKind::TopologyUnavailable =>
                {
                    Some(
                        fact.data()
                            .scope
                            .topology_id
                            .as_ref()
                            .map(|topology_id| topology_id.as_str().to_string()),
                    )
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            unavailable,
            vec![Some("topo-withdrawal-dropped".to_string())]
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn withdrawal_resolves_topology_unavailable_and_node_unavailable() {
        let engine = install_test_engine();
        let op = TopologyWithdrawalOperation::begin("topo-a");
        op.withdrawn();
        engine.drain();
        let kinds = engine
            .replay()
            .snapshot()
            .into_iter()
            .map(|frame| frame.fact.kind_id().to_string())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&"topology_unavailable".to_string()));
        assert!(kinds.contains(&"node_unavailable".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn report_recovery_decision_healthy_emits_nothing_but_connection_recovery() {
        let engine = install_test_engine();
        let a = node(1);
        // a recovers, decision is healthy (NoActiveStageLoss) -- only the
        // per-node connection-recovered fact fires, no TopologyDegraded /
        // NodeDegraded.
        report_recovery_decision("topo-a", true, false, &[a], &[], true);
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"stage_connection_recovered"));
        assert!(!kinds.contains(&"topology_degraded"));
        assert!(!kinds.contains(&"node_degraded"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn report_recovery_decision_degraded_emits_lost_topology_degraded_and_node_degraded() {
        let engine = install_test_engine();
        let a = node(1);
        report_recovery_decision("topo-a", false, false, &[], &[a], true);
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"stage_connection_lost"));
        assert!(kinds.contains(&"topology_degraded"));
        assert!(kinds.contains(&"node_degraded"));
        assert!(!kinds.contains(&"node_unavailable"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn report_recovery_decision_withdrawing_emits_node_unavailable_not_node_degraded() {
        let engine = install_test_engine();
        let a = node(1);
        report_recovery_decision("topo-a", false, true, &[], &[a], true);
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"node_unavailable"));
        assert!(!kinds.contains(&"node_degraded"));
        clear_runtime_event_engine();
    }

    /// §8.14 `resource_pressure_changed` -- fires every tick regardless of
    /// health/withdrawal state, carrying the existing `local_model_fits()`
    /// boolean as its outcome (Success = capacity available, Failure =
    /// pressure present). The state lane's own latest-value-wins
    /// coalescing (`lanes.rs::submit_state_transition`) is what keeps this
    /// from becoming an event storm, not any dedup logic here.
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn report_recovery_decision_emits_resource_pressure_changed_both_ways() {
        let engine = install_test_engine();
        report_recovery_decision("topo-a", true, false, &[], &[], false);
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"resource_pressure_changed"),
            "must fire even on an otherwise-healthy tick"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn emit_stage_connection_established_reaches_the_state_lane() {
        let engine = install_test_engine();
        emit_stage_connection_established("topo-a", "stage-1", 3);
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"stage_connection_established")
        );
        engine.drain();
        let stage_index = engine
            .replay()
            .snapshot()
            .into_iter()
            .find_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::StageTopology(fact)
                    if *fact.kind() == StageTopologyEventKind::StageConnectionEstablished =>
                {
                    fact.data().scope.stage.as_ref().map(|stage| stage.index)
                }
                _ => None,
            })
            .expect("connection fact must carry stage identity");
        assert_eq!(stage_index, 3);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_never_panics() {
        clear_runtime_event_engine();
        let op = TopologyAssemblyOperation::begin("topo-a");
        op.ready();
        let withdraw = TopologyWithdrawalOperation::begin("topo-a");
        withdraw.withdrawn();
        report_recovery_decision("topo-a", false, false, &[], &[node(1)], true);
    }
}
