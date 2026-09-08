//! Task 6 (`.omo/plans/event-system-fixes.md`, defect D6) acceptance tests:
//! the reducer's `operations` map stays bounded by the reservation table
//! capacity even under sustained settled-operation churn, and bounded
//! per-category domain state (models/stages/sessions/requests/devices/cache)
//! is actually populated from applied facts instead of discarded.

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, FactData, FamilyFact, KvRuntimeStateEventKind, LogicalModelId,
    ModelAvailabilityEventKind, ModelUnloadingEventKind, OperationId, OperationScope, Outcome,
    RequestEventKind, RequestId, ResourceHealthEventKind, RuntimeFact, ScopeIdentities,
    SessionEventKind, SessionId, StageIdentity, StageTopologyEventKind, StateName, StateTransition,
};

use super::fixtures::{input, terminal_fact};
use crate::runtime_events::config::{REQUEST_ROOT_BOUND, RESERVATION_TABLE_CAPACITY};
use crate::runtime_events::reducer::{
    ReduceOutcome, ReducerSnapshot, apply,
    rebuild::{self, RebuildOutcome},
};

fn root() -> OperationScope {
    OperationScope::root_only(OperationId::new())
}

fn model_fact(kind: ModelAvailabilityEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelAvailability(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn unload_fact(kind: ModelUnloadingEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelUnloading(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn stage_fact(kind: StageTopologyEventKind, stage_id: &str, index: u32) -> RuntimeFact {
    RuntimeFact::StageTopology(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                stage: Some(StageIdentity::new(
                    mesh_llm_runtime_event_contracts::StageId::new(stage_id)
                        .expect("valid stage id"),
                    index,
                )),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn session_fact(kind: SessionEventKind, session_id: &str) -> RuntimeFact {
    RuntimeFact::Session(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                session_id: Some(SessionId::new(session_id).expect("valid session id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn request_fact(kind: RequestEventKind, request_id: &str) -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                request_id: Some(RequestId::new(request_id).expect("valid request id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn device_fact(kind: ResourceHealthEventKind, device_id: &str) -> RuntimeFact {
    RuntimeFact::ResourceHealth(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                device_id: Some(
                    mesh_llm_runtime_event_contracts::DeviceId::new(device_id)
                        .expect("valid device id"),
                ),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn cache_fact(kind: KvRuntimeStateEventKind) -> RuntimeFact {
    RuntimeFact::KvRuntimeState(FamilyFact::new(kind))
}

fn model_load_phase_fact(model_id: &str, phase: &str) -> RuntimeFact {
    RuntimeFact::ModelLoading(FamilyFact::with_data(
        mesh_llm_runtime_event_contracts::ModelLoadingEventKind::ModelLoadPhaseChanged,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            state: Some(StateTransition::new(
                None,
                StateName::new(phase).expect("valid state name"),
            )),
            ..FactData::default()
        },
    ))
}

#[test]
fn settled_operations_are_evicted_to_stay_within_reservation_capacity() {
    let mut snapshot = ReducerSnapshot::empty();
    for sequence in 0..10_000u64 {
        let scope = root();
        let fact = terminal_fact(Outcome::Success);
        let ReduceOutcome::Applied(next) = apply(&snapshot, input(scope, sequence, fact)) else {
            panic!("every distinct root scope must apply");
        };
        snapshot = next;
    }
    assert!(
        snapshot.operation_count() <= RESERVATION_TABLE_CAPACITY,
        "operations map must never exceed the reservation table capacity ({RESERVATION_TABLE_CAPACITY}), got {}",
        snapshot.operation_count()
    );
}

#[test]
fn a_loaded_model_becomes_available_and_unload_removes_it() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_fact(ModelAvailabilityEventKind::ModelAvailable, "demo-model"),
        ),
    ) else {
        panic!("model_available must apply");
    };

    let models = snapshot.domain().models();
    let demo = models
        .iter()
        .find(|model| model.id == "demo-model")
        .expect("model must appear in state.models after model_available");
    assert_eq!(demo.availability.as_deref(), Some("available"));

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            unload_fact(ModelUnloadingEventKind::UnloadCompleted, "demo-model"),
        ),
    ) else {
        panic!("unload_completed must apply");
    };
    assert!(
        !snapshot
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "demo-model"),
        "unload must remove the model from state.models"
    );
}

#[test]
fn model_load_phase_changed_carries_the_producer_supplied_phase_name() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_load_phase_fact("phase-model", "downloading_weights"),
        ),
    ) else {
        panic!("model_load_phase_changed must apply");
    };
    let model = snapshot
        .domain()
        .models()
        .into_iter()
        .find(|model| model.id == "phase-model")
        .expect("model must be tracked");
    assert_eq!(model.load_phase.as_deref(), Some("downloading_weights"));
}

/// F2 fix (event-system-fixes, live-sampling finding): a single load
/// operation's model identity legitimately changes mid-flight (a
/// pre-resolution provisional id, superseded by the resolved canonical id
/// once source resolution completes). Before this fix the provisional
/// row was an ORPHANED PHANTOM -- stuck at whatever `load_phase` its last
/// fact set, forever, because no later fact ever referenced that id again
/// to transition or evict it (reproduced live: a stale
/// `"load_phase":"loading"` row survived 15+ minutes and 3 reconnects).
/// `reconcile_model_root_identity` correlates by the fact's ROOT
/// operation (stable across the whole operation, unlike model_id) and
/// evicts the stale row the moment the SAME root reports a different id.
#[test]
fn a_root_operations_provisional_model_row_is_superseded_not_orphaned_on_resolution() {
    let snapshot = ReducerSnapshot::empty();
    let scope = root();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope,
            0,
            model_load_phase_fact("pending-resolution/op-1", "loading"),
        ),
    ) else {
        panic!("the provisional (pre-resolution) fact must apply");
    };
    assert!(
        snapshot
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "pending-resolution/op-1"),
        "the provisional row must exist right after the pre-resolution fact"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(scope, 1, model_load_phase_fact("resolved-model", "loading")),
    ) else {
        panic!("the resolved-identity fact for the SAME root must apply");
    };

    let models = snapshot.domain().models();
    assert!(
        !models
            .iter()
            .any(|model| model.id == "pending-resolution/op-1"),
        "the provisional row must be SUPERSEDED once the SAME root reports \
         its resolved identity, not left an orphaned phantom stuck in \
         \"loading\" forever"
    );
    assert!(
        models.iter().any(|model| model.id == "resolved-model"),
        "the resolved identity's own row must exist after supersession"
    );
}

#[test]
fn stages_track_latest_topology_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            stage_fact(StageTopologyEventKind::StageReady, "stage-0", 0),
        ),
    ) else {
        panic!("stage_ready must apply");
    };
    let stage = snapshot
        .domain()
        .stages()
        .into_iter()
        .find(|stage| stage.id == "stage-0")
        .expect("stage must be tracked in state.stages");
    assert_eq!(stage.state.as_deref(), Some("ready"));
}

#[test]
fn sessions_track_active_count_and_bounded_recent() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            session_fact(SessionEventKind::SessionCreated, "sess-1"),
        ),
    ) else {
        panic!("session_created must apply");
    };
    assert_eq!(snapshot.domain().sessions_active_count(), 1);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            session_fact(SessionEventKind::SessionClosed, "sess-1"),
        ),
    ) else {
        panic!("session_closed must apply");
    };
    assert_eq!(
        snapshot.domain().sessions_active_count(),
        0,
        "a closed session must leave the active count"
    );
    let recent = snapshot.domain().sessions_recent();
    assert!(
        recent
            .iter()
            .any(|entry| entry.id == "sess-1" && entry.state == "closed"),
        "a closed session must appear in the bounded recent list"
    );
}

#[test]
fn requests_are_in_flight_only_and_bounded() {
    let mut snapshot = ReducerSnapshot::empty();
    for index in 0..(REQUEST_ROOT_BOUND + 50) {
        let request_id = format!("req-{index}");
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(
                root(),
                index as u64,
                request_fact(RequestEventKind::RequestReceived, &request_id),
            ),
        ) else {
            panic!("request_received must apply for {request_id}");
        };
        snapshot = next;
    }
    assert!(
        snapshot.domain().requests().len() <= REQUEST_ROOT_BOUND,
        "in-flight requests must stay bounded by REQUEST_ROOT_BOUND"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            (REQUEST_ROOT_BOUND + 50) as u64,
            request_fact(RequestEventKind::RequestCompleted, "req-0"),
        ),
    ) else {
        panic!("request_completed must apply even for an evicted-from-domain request");
    };
    assert!(
        !snapshot
            .domain()
            .requests()
            .iter()
            .any(|request| request.id == "req-0"),
        "a completed request must never appear as in-flight"
    );
}

#[test]
fn a_child_terminal_does_not_remove_its_streaming_root_request() {
    let root_operation = OperationId::new();
    let root_scope = OperationScope::root_only(root_operation);
    let child_scope = OperationScope::with_child(root_operation, ChildOperationId::new());

    let ReduceOutcome::Applied(snapshot) = apply(
        &ReducerSnapshot::empty(),
        input(
            root_scope,
            0,
            request_fact(RequestEventKind::RequestReceived, "streaming-request"),
        ),
    ) else {
        panic!("request_received must apply");
    };

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            1,
            request_fact(
                RequestEventKind::RequestExecutionStarted,
                "streaming-request",
            ),
        ),
    ) else {
        panic!("a backend child progress fact must apply");
    };
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .find(|request| request.id == "streaming-request")
            .and_then(|request| request.state.as_deref()),
        Some("received"),
        "child observations must not overwrite the root request status"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            2,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("a backend child terminal must apply");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .any(|request| request.id == "streaming-request"),
        "a child terminal must not hide the still-streaming root request"
    );
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .find(|request| request.id == "streaming-request")
            .and_then(|request| request.state.as_deref()),
        Some("received"),
        "a child terminal must not overwrite the root request status"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root_scope,
            3,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("the root terminal must apply");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .all(|request| request.id != "streaming-request"),
        "only the root terminal may remove the request row"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            4,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("a late child terminal must still apply to the child operation");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .all(|request| request.id != "streaming-request"),
        "a child terminal after root completion must not resurrect the request row"
    );
}

#[test]
fn devices_track_the_latest_resource_health_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            device_fact(ResourceHealthEventKind::DeviceReady, "gpu-0"),
        ),
    ) else {
        panic!("device_ready must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            device_fact(ResourceHealthEventKind::DeviceDegraded, "gpu-0"),
        ),
    ) else {
        panic!("device_degraded must apply");
    };
    let device = snapshot
        .domain()
        .devices()
        .into_iter()
        .find(|device| device.id == "gpu-0")
        .expect("device must be tracked in state.devices");
    assert_eq!(device.state.as_deref(), Some("degraded"));
}

#[test]
fn cache_tracks_the_latest_pressure_and_capacity_signal() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            cache_fact(KvRuntimeStateEventKind::CachePressureCrossed),
        ),
    ) else {
        panic!("cache_pressure_crossed must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            cache_fact(KvRuntimeStateEventKind::ContextExhausted),
        ),
    ) else {
        panic!("context_exhausted must apply");
    };
    let cache = snapshot.domain().cache();
    assert_eq!(cache.pressure.as_deref(), Some("pressure"));
    assert_eq!(cache.capacity_state.as_deref(), Some("exhausted"));
}

#[test]
fn rebuild_preserves_last_valid_domain_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_fact(ModelAvailabilityEventKind::ModelAvailable, "durable-model"),
        ),
    ) else {
        panic!("model_available must apply");
    };

    let RebuildOutcome::Rebuilt(rebuilt) = rebuild::rebuild(&snapshot, 1) else {
        panic!("rebuild to a higher generation must succeed");
    };
    assert!(
        rebuilt
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "durable-model"),
        "domain state must survive a rebuild, not be discarded"
    );
}

/// Task 6 defect C (verifier follow-up, `.omo/plans/event-system-fixes.md`):
/// the missing safety-critical invariant -- an UNSETTLED (in-flight)
/// operation is never evicted, even once the map is driven well past
/// `RESERVATION_TABLE_CAPACITY` with settled churn. A known subset of
/// scopes receives a non-terminal fact (tracked, but never settled) and
/// is held open across the whole run; the size-triggered capacity sweep
/// must skip them every time and evict progressively newer SETTLED
/// entries instead.
#[test]
fn unsettled_operations_survive_the_capacity_sweep_while_settled_ones_are_evicted() {
    let mut snapshot = ReducerSnapshot::empty();
    let mut sequence = 0u64;

    let in_flight: Vec<OperationScope> = (0..8).map(|_| root()).collect();
    for &scope in &in_flight {
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(
                scope,
                sequence,
                model_load_phase_fact("in-flight-model", "loading"),
            ),
        ) else {
            panic!("a non-terminal fact must apply for an in-flight scope");
        };
        snapshot = next;
        sequence += 1;
    }

    let first_settled_scope = root();
    let settled_total = RESERVATION_TABLE_CAPACITY + 200;
    for index in 0..settled_total {
        let scope = if index == 0 {
            first_settled_scope
        } else {
            root()
        };
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(scope, sequence, terminal_fact(Outcome::Success)),
        ) else {
            panic!("every distinct settled root scope must apply");
        };
        snapshot = next;
        sequence += 1;
    }

    assert!(
        snapshot.operation_count() <= RESERVATION_TABLE_CAPACITY,
        "operations map must stay bounded even with an in-flight subset held open, got {}",
        snapshot.operation_count()
    );
    for &scope in &in_flight {
        let state = snapshot.operation(scope).unwrap_or_else(|| {
            panic!("an in-flight (unsettled) operation must never be evicted by the capacity sweep")
        });
        assert!(
            !state.settled,
            "the in-flight scope must still report unsettled"
        );
    }
    assert!(
        snapshot.operation(first_settled_scope).is_none(),
        "the earliest-settled operation must be evicted once the map exceeds capacity"
    );
}

/// Task 6 defect D (verifier follow-up, `.omo/plans/event-system-fixes.md`):
/// `models()`/`stages()`/`requests()`/`devices()` used to return
/// `HashMap` iteration order (nondeterministic per process). The
/// `*_order` `VecDeque`s `DomainState` already maintains for bounded
/// eviction must ALSO drive output order, so repeated reads of the same
/// state are stable and match insertion/touch order.
#[test]
fn category_arrays_preserve_deterministic_insertion_order_not_hashmap_order() {
    let mut snapshot = ReducerSnapshot::empty();
    let facts: Vec<RuntimeFact> = vec![
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-a"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-b"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-c"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-d"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-e"),
        stage_fact(StageTopologyEventKind::StageReady, "stage-a", 0),
        stage_fact(StageTopologyEventKind::StageReady, "stage-b", 1),
        stage_fact(StageTopologyEventKind::StageReady, "stage-c", 2),
        stage_fact(StageTopologyEventKind::StageReady, "stage-d", 3),
        stage_fact(StageTopologyEventKind::StageReady, "stage-e", 4),
        request_fact(RequestEventKind::RequestReceived, "req-a"),
        request_fact(RequestEventKind::RequestReceived, "req-b"),
        request_fact(RequestEventKind::RequestReceived, "req-c"),
        request_fact(RequestEventKind::RequestReceived, "req-d"),
        request_fact(RequestEventKind::RequestReceived, "req-e"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-a"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-b"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-c"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-d"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-e"),
        // `sessions.recent` is a `VecDeque` already (not a `HashMap`), but
        // the brief for this defect calls it out explicitly ("must be
        // deterministic too -- check it"), so it is proven here rather than
        // left as an unstated assumption.
        session_fact(SessionEventKind::SessionCreated, "sess-a"),
        session_fact(SessionEventKind::SessionClosed, "sess-a"),
        session_fact(SessionEventKind::SessionCreated, "sess-b"),
        session_fact(SessionEventKind::SessionClosed, "sess-b"),
        session_fact(SessionEventKind::SessionCreated, "sess-c"),
        session_fact(SessionEventKind::SessionClosed, "sess-c"),
        session_fact(SessionEventKind::SessionCreated, "sess-d"),
        session_fact(SessionEventKind::SessionClosed, "sess-d"),
        session_fact(SessionEventKind::SessionCreated, "sess-e"),
        session_fact(SessionEventKind::SessionClosed, "sess-e"),
    ];
    for (sequence, fact) in facts.into_iter().enumerate() {
        let ReduceOutcome::Applied(next) = apply(&snapshot, input(root(), sequence as u64, fact))
        else {
            panic!("every distinct-category fact must apply");
        };
        snapshot = next;
    }

    assert_eq!(
        snapshot
            .domain()
            .models()
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
        vec!["model-a", "model-b", "model-c", "model-d", "model-e"],
        "state.models must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .stages()
            .iter()
            .map(|stage| stage.id.clone())
            .collect::<Vec<_>>(),
        vec!["stage-a", "stage-b", "stage-c", "stage-d", "stage-e"],
        "state.stages must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .map(|request| request.id.clone())
            .collect::<Vec<_>>(),
        vec!["req-a", "req-b", "req-c", "req-d", "req-e"],
        "state.requests must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .devices()
            .iter()
            .map(|device| device.id.clone())
            .collect::<Vec<_>>(),
        vec!["device-a", "device-b", "device-c", "device-d", "device-e"],
        "state.devices must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .sessions_recent()
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>(),
        vec!["sess-a", "sess-b", "sess-c", "sess-d", "sess-e"],
        "sessions.recent must preserve closure order (it already does, via a \
         VecDeque -- proven here rather than left unstated)"
    );
}
