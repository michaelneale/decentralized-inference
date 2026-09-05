//! Runtime-event producer wiring for stage lifecycle (plan task 10,
//! `.omo/plans/event-system.md` line 278; spec §8.6 stage bullets).
//!
//! One reservation per stage-load attempt: `StageStarting` -> `StageLoading`
//! -> exactly one terminal (`StageReady` / `StageFailed`). Stop is a second,
//! independent reservation: `StageStopping` -> terminal `StageStopped`.
//! `available_stage_set_changed`/`request_capacity_changed`/
//! `lane_capacity_changed` (spec §8.14, `NodeAvailabilityEventKind`) are
//! StateTransition-class facts co-emitted alongside the stage-set change
//! they observe, reusing the same `lane_count` field the stage load/stop
//! request already carries -- no new wire field, no new heuristic.
//!
//! Best-effort and never blocking: an absent engine or an exhausted
//! reservation table degrades every call here to a silent no-op, matching
//! `runtime::model_lifecycle::events`'s established contract (task 9).

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, FactData, HumanSummary, NodeAvailabilityEventKind,
    NodeAvailabilityFact, NumericSummary, NumericSummaryKey, NumericValue, OperationId, Outcome,
    ReasonCode, RuntimeEventIngress, RuntimeFact, ScopeIdentities, StageId, StageIdentity,
    StageTopologyEventKind, StageTopologyFact, TopologyId,
};

use crate::runtime_events::engine::OperationReservation;
use crate::runtime_events::runtime_event_engine;

fn stage_scope(topology_id: &str, stage_id: &str, layer_start: u32) -> FactData {
    FactData {
        scope: ScopeIdentities {
            topology_id: TopologyId::new(topology_id).ok(),
            stage: StageId::new(stage_id)
                .ok()
                .map(|id| StageIdentity::new(id, layer_start)),
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

fn synthetic_load_terminal() -> RuntimeFact {
    RuntimeFact::StageTopology(StageTopologyFact::with_data(
        StageTopologyEventKind::StageFailed,
        terminal_not_delivered(),
    ))
}

fn synthetic_stop_terminal() -> RuntimeFact {
    RuntimeFact::StageTopology(StageTopologyFact::with_data(
        StageTopologyEventKind::StageStopped,
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

/// Fire-and-forget `NodeAvailability` StateTransition emission with no
/// reservation lifecycle -- matches the engine's own documented contract
/// for `unreserved_ingress` (state transitions coalesce by resource key in
/// the reducer's state lane and never need a bounded slot). Never gossiped:
/// this is a purely local, process-internal fact.
fn emit_node_availability(
    engine: &std::sync::Arc<crate::runtime_events::engine::RuntimeEventEngine>,
    kind: NodeAvailabilityEventKind,
    data: FactData,
) {
    let ingress = engine.unreserved_ingress(
        mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new()),
    );
    let _ = ingress.try_submit(RuntimeFact::NodeAvailability(
        NodeAvailabilityFact::with_data(kind, data),
    ));
}

fn lane_count_summary(lane_count: u32) -> BoundedNumericSummaries {
    NumericSummaryKey::new("lane_count")
        .ok()
        .and_then(|key| {
            BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Unsigned(u64::from(lane_count)),
            )])
            .ok()
        })
        .unwrap_or_default()
}

/// Co-emits the two §8.14 `NodeAvailability` facts a stage-set change
/// always carries at this layer: the stage set itself changed, and the
/// node's request/lane capacity moved by `lane_count` (the only capacity
/// signal a `StageLoadRequest`/`StageStopRequest` already carries -- no
/// separate session-capacity signal exists at this layer, so
/// `SessionCapacityChanged` is deliberately not emitted here).
fn emit_stage_set_and_capacity_changed(
    topology_id: &str,
    stage_id: &str,
    layer_start: u32,
    lane_count: u32,
) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    let scope = stage_scope(topology_id, stage_id, layer_start);
    emit_node_availability(
        &engine,
        NodeAvailabilityEventKind::AvailableStageSetChanged,
        scope.clone(),
    );
    let summary = FactData {
        numeric_summaries: lane_count_summary(lane_count),
        ..scope.clone()
    };
    emit_node_availability(
        &engine,
        NodeAvailabilityEventKind::RequestCapacityChanged,
        summary.clone(),
    );
    emit_node_availability(
        &engine,
        NodeAvailabilityEventKind::LaneCapacityChanged,
        summary,
    );
}

/// One stage-load operation. Reserved before the binary stage process
/// starts; resolves with exactly one terminal (`StageReady`/`StageFailed`).
pub(crate) struct StageLoadOperation {
    root: Option<OperationReservation>,
    topology_id: String,
    stage_id: String,
    layer_start: u32,
}

impl StageLoadOperation {
    pub(crate) fn begin(
        topology_id: &str,
        stage_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> Self {
        let root = runtime_event_engine()
            .and_then(|engine| engine.reserve_root(OperationId::new(), synthetic_load_terminal));
        if let Some(root) = &root {
            let mut data = stage_scope(topology_id, stage_id, layer_start);
            data.summary = HumanSummary::new(&format!("layers {layer_start}-{layer_end}")).ok();
            submit(root, StageTopologyEventKind::StageStarting, data.clone());
            submit(root, StageTopologyEventKind::StageLoading, data);
        }
        Self {
            root,
            topology_id: topology_id.to_string(),
            stage_id: stage_id.to_string(),
            layer_start,
        }
    }

    /// §8.6 `stage ready` -- also co-emits the §8.14 stage-set/capacity
    /// facts, since a newly-ready stage is exactly what changes the node's
    /// available stage set and lane capacity.
    pub(crate) fn ready(mut self, lane_count: u32) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::StageReady,
                FactData {
                    outcome: Some(Outcome::Success),
                    ..stage_scope(&self.topology_id, &self.stage_id, self.layer_start)
                },
            );
        }
        emit_stage_set_and_capacity_changed(
            &self.topology_id,
            &self.stage_id,
            self.layer_start,
            lane_count,
        );
    }

    pub(crate) fn failed(mut self, reason: ReasonCode) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::StageFailed,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..stage_scope(&self.topology_id, &self.stage_id, self.layer_start)
                },
            );
        }
    }
}

impl Drop for StageLoadOperation {
    fn drop(&mut self) {
        let Some(root) = self.root.as_ref() else {
            return;
        };
        let mut data = terminal_not_delivered();
        data.scope = stage_scope(&self.topology_id, &self.stage_id, self.layer_start).scope;
        submit(root, StageTopologyEventKind::StageFailed, data);
    }
}

/// One stage-stop operation. Reserved before shutdown begins.
pub(crate) struct StageStopOperation {
    root: Option<OperationReservation>,
    topology_id: String,
    stage_id: String,
}

impl StageStopOperation {
    pub(crate) fn begin(topology_id: &str, stage_id: &str) -> Self {
        let root = runtime_event_engine()
            .and_then(|engine| engine.reserve_root(OperationId::new(), synthetic_stop_terminal));
        if let Some(root) = &root {
            submit(
                root,
                StageTopologyEventKind::StageStopping,
                stage_scope(topology_id, stage_id, 0),
            );
        }
        Self {
            root,
            topology_id: topology_id.to_string(),
            stage_id: stage_id.to_string(),
        }
    }

    /// §8.6 `stage stopped` -- also co-emits the §8.14 stage-set/capacity
    /// facts with `lane_count: 0`, since the stopped stage no longer
    /// contributes lane capacity.
    pub(crate) fn stopped(mut self) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                StageTopologyEventKind::StageStopped,
                FactData {
                    outcome: Some(Outcome::Success),
                    ..stage_scope(&self.topology_id, &self.stage_id, 0)
                },
            );
        }
        emit_stage_set_and_capacity_changed(&self.topology_id, &self.stage_id, 0, 0);
    }
}

impl Drop for StageStopOperation {
    fn drop(&mut self) {
        let Some(root) = self.root.as_ref() else {
            return;
        };
        let mut data = terminal_not_delivered();
        data.scope = stage_scope(&self.topology_id, &self.stage_id, 0).scope;
        submit(root, StageTopologyEventKind::StageStopped, data);
    }
}

#[cfg(test)]
mod tests {
    use super::{StageLoadOperation, StageStopOperation, emit_stage_set_and_capacity_changed};
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::{
        Outcome, ReasonCode, RuntimeEventIngress, RuntimeFact, StageTopologyEventKind,
    };

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    fn replay_kinds(engine: &RuntimeEventEngine) -> Vec<String> {
        engine
            .replay()
            .snapshot()
            .into_iter()
            .map(|frame| frame.fact.kind_id().to_string())
            .collect()
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn stage_load_success_emits_starting_loading_ready_and_capacity_facts() {
        let engine = install_test_engine();
        let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        assert_eq!(engine.occupied_count(), 1);
        op.ready(4);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        // Every delivery class -- terminal and state-transition alike --
        // reaches replay() once drained.
        let kinds = replay_kinds(&engine);
        assert!(kinds.contains(&"stage_ready".to_string()));
        assert!(kinds.contains(&"stage_starting".to_string()));
        assert!(kinds.contains(&"stage_loading".to_string()));
        assert!(kinds.contains(&"available_stage_set_changed".to_string()));
        assert!(kinds.contains(&"request_capacity_changed".to_string()));
        assert!(kinds.contains(&"lane_capacity_changed".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn stage_load_failure_resolves_one_terminal_without_capacity_facts() {
        let engine = install_test_engine();
        let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        op.failed(ReasonCode::ModelFormatOrLoadFailure);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let kinds = replay_kinds(&engine);
        assert!(kinds.contains(&"stage_failed".to_string()));
        assert!(
            !kinds.contains(&"available_stage_set_changed".to_string()),
            "a failed load must not report a stage-set change"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn capacity_facts_preserve_nonzero_stage_layer_start() {
        let engine = install_test_engine();
        emit_stage_set_and_capacity_changed("topo-a", "stage-1", 16, 4);
        engine.drain();
        let stage_indices = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| match frame.fact.as_ref() {
                RuntimeFact::NodeAvailability(fact) => {
                    fact.data().scope.stage.as_ref().map(|stage| stage.index)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stage_indices, vec![16, 16, 16]);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn stage_stop_emits_stopping_stopped_and_zeroed_capacity_facts() {
        let engine = install_test_engine();
        let op = StageStopOperation::begin("topo-a", "stage-0");
        assert_eq!(engine.occupied_count(), 1);
        op.stopped();
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let kinds = replay_kinds(&engine);
        assert!(kinds.contains(&"stage_stopped".to_string()));
        assert!(kinds.contains(&"stage_stopping".to_string()));
        assert!(kinds.contains(&"available_stage_set_changed".to_string()));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_a_load_operation_without_a_terminal_synthesizes_one() {
        let engine = install_test_engine();
        {
            let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
            assert_eq!(engine.occupied_count(), 1);
            drop(op);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let frames = engine.replay().snapshot();
        let terminals = frames
            .iter()
            .filter(|frame| frame.fact.kind_id() == "stage_failed")
            .collect::<Vec<_>>();
        assert_eq!(
            terminals.len(),
            1,
            "drop must synthesize exactly one terminal"
        );
        let data = terminals[0].fact.data();
        assert_eq!(data.outcome, Some(Outcome::Unknown));
        assert_eq!(data.reason, Some(ReasonCode::TerminalNotDelivered));
        assert_eq!(
            data.scope.topology_id.as_ref().map(|id| id.as_str()),
            Some("topo-a")
        );
        assert_eq!(
            data.scope.stage.as_ref().map(|stage| stage.id.as_str()),
            Some("stage-0")
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_a_stop_operation_synthesizes_one_scoped_terminal() {
        let engine = install_test_engine();
        {
            let op = StageStopOperation::begin("topo-a", "stage-0");
            assert_eq!(engine.occupied_count(), 1);
            drop(op);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let terminals = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter(|frame| frame.fact.kind_id() == "stage_stopped")
            .collect::<Vec<_>>();
        assert_eq!(
            terminals.len(),
            1,
            "drop must synthesize exactly one terminal"
        );
        let data = terminals[0].fact.data();
        assert_eq!(data.outcome, Some(Outcome::Unknown));
        assert_eq!(data.reason, Some(ReasonCode::TerminalNotDelivered));
        assert_eq!(
            data.scope.topology_id.as_ref().map(|id| id.as_str()),
            Some("topo-a")
        );
        assert_eq!(
            data.scope.stage.as_ref().map(|stage| stage.id.as_str()),
            Some("stage-0")
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn shutdown_synthesizes_held_load_terminal_with_scope() {
        let engine = install_test_engine();
        let op = StageLoadOperation::begin("topo-a", "stage-1", 16, 24);
        let report = engine.shutdown(None);
        assert_eq!(report.remaining_after_deadline, 0);

        let terminals = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter(|frame| frame.fact.kind_id() == "stage_failed")
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1);
        let data = terminals[0].fact.data();
        assert_eq!(data.outcome, Some(Outcome::Unknown));
        assert_eq!(data.reason, Some(ReasonCode::TerminalNotDelivered));
        assert_eq!(
            data.scope.topology_id.as_ref().map(|id| id.as_str()),
            Some("topo-a")
        );
        assert_eq!(
            data.scope.stage.as_ref().map(|stage| stage.id.as_str()),
            Some("stage-1")
        );
        assert_eq!(data.scope.stage.as_ref().map(|stage| stage.index), Some(16));

        drop(op);
        assert_eq!(
            engine
                .replay()
                .snapshot()
                .into_iter()
                .filter(|frame| frame.fact.kind_id() == "stage_failed")
                .count(),
            1,
            "shutdown synthesis must settle the held guard without a duplicate drop terminal"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn shutdown_synthesizes_held_stop_terminal_with_scope() {
        let engine = install_test_engine();
        let op = StageStopOperation::begin("topo-a", "stage-1");
        let report = engine.shutdown(None);
        assert_eq!(report.remaining_after_deadline, 0);

        let terminals = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter(|frame| frame.fact.kind_id() == "stage_stopped")
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1);
        let data = terminals[0].fact.data();
        assert_eq!(data.outcome, Some(Outcome::Unknown));
        assert_eq!(data.reason, Some(ReasonCode::TerminalNotDelivered));
        assert_eq!(
            data.scope.topology_id.as_ref().map(|id| id.as_str()),
            Some("topo-a")
        );
        assert_eq!(
            data.scope.stage.as_ref().map(|stage| stage.id.as_str()),
            Some("stage-1")
        );
        assert_eq!(data.scope.stage.as_ref().map(|stage| stage.index), Some(0));

        drop(op);
        assert_eq!(
            engine
                .replay()
                .snapshot()
                .into_iter()
                .filter(|frame| frame.fact.kind_id() == "stage_stopped")
                .count(),
            1,
            "shutdown synthesis must settle the held guard without a duplicate drop terminal"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_degrades_to_no_op_and_never_panics() {
        clear_runtime_event_engine();
        let load = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        load.ready(4);
        let stop = StageStopOperation::begin("topo-a", "stage-0");
        stop.stopped();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reservation_exhaustion_degrades_load_to_a_no_op_without_failing() {
        let engine = RuntimeEventEngine::with_capacity(0);
        clear_runtime_event_engine();
        install_runtime_event_engine(engine.clone());
        let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        op.ready(4);
        assert!(engine.health().snapshot().reservation_exhausted > 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn stage_ready_does_not_emit_session_capacity_changed() {
        // Documents the deliberate scope boundary: this layer has no
        // distinct session-capacity signal (owned by Task 12's
        // skippy-server lane/session tracking), so
        // `SessionCapacityChanged` is never emitted from here.
        let engine = install_test_engine();
        let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        op.ready(4);
        engine.drain();
        assert!(
            !engine
                .state_lane_kinds()
                .contains(&"session_capacity_changed"),
            "this layer has no distinct session-capacity signal (owned by Task 12)"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn stage_load_terminal_class_is_write_once_per_slot() {
        // Mutation-style structural proof: StageReady/StageFailed are
        // Terminal-class (delivery/lifecycle.rs); a second terminal write
        // to the same slot after `ready()` has already resolved it must
        // not panic and must not double-release.
        let engine = install_test_engine();
        let op = StageLoadOperation::begin("topo-a", "stage-0", 0, 12);
        let root = op.root.as_ref().expect("reserved");
        let _ = root.ingress().try_submit(RuntimeFact::StageTopology(
            mesh_llm_runtime_event_contracts::StageTopologyFact::with_data(
                StageTopologyEventKind::StageReady,
                mesh_llm_runtime_event_contracts::FactData::default(),
            ),
        ));
        drop(op);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let frames = engine.replay().snapshot();
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.fact.kind_id() == "stage_ready")
                .count(),
            1,
            "a real terminal must not be duplicated by the operation drop"
        );
        assert!(
            !frames
                .iter()
                .any(|frame| frame.fact.kind_id() == "stage_failed")
        );
        clear_runtime_event_engine();
    }
}
