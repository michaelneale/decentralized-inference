//! `runtime_state` domain projection.
//!
//! Defect D6 (`.omo/plans/event-system-fixes.md` task 6): the six category
//! arrays used to be structurally present but always empty, because the
//! reducer's `ReducerSnapshot` discarded every domain fact after folding it
//! into generic operation health. Task 6 landed
//! `runtime_events::reducer::DomainState` (bounded per-category state); this
//! module projects it into EXPLICIT `Serialize` structs -- no
//! `serde_json::Value` bags -- so every field this module emits is real,
//! reducer-backed data. `node` was already populated from genuinely
//! available reducer/health data and is unchanged.
//!
//! Inner-key naming here is Rust-side only (task 6). Pinning these keys
//! into `fixtures/runtime_events_v1/frames.json` with sample frames is
//! task 7's job, not this module's.

use serde::Serialize;

#[cfg(test)]
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::reducer::{
    CacheDomainState, DeviceDomainState, DomainState, ModelDomainState, RequestDomainState,
    SessionRecentEntry, StageDomainState,
};

#[derive(Debug, Serialize)]
pub(crate) struct NodeProjection {
    pub(crate) rebuild_generation: u64,
    pub(crate) tracked_operation_count: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) availability: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<String>,
}

impl From<ModelDomainState> for ModelProjection {
    fn from(model: ModelDomainState) -> Self {
        Self {
            id: model.id,
            availability: model.availability,
            load_phase: model.load_phase,
            last_outcome: model.last_outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StageProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<String>,
}

impl From<StageDomainState> for StageProjection {
    fn from(stage: StageDomainState) -> Self {
        Self {
            id: stage.id,
            index: stage.index,
            state: stage.state,
            last_outcome: stage.last_outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionRecentProjection {
    pub(crate) id: String,
    pub(crate) state: String,
}

impl From<SessionRecentEntry> for SessionRecentProjection {
    fn from(entry: SessionRecentEntry) -> Self {
        Self {
            id: entry.id,
            state: entry.state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionsProjection {
    pub(crate) active_count: usize,
    pub(crate) recent: Vec<SessionRecentProjection>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
}

impl From<RequestDomainState> for RequestProjection {
    fn from(request: RequestDomainState) -> Self {
        Self {
            id: request.id,
            state: request.state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DeviceProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
}

impl From<DeviceDomainState> for DeviceProjection {
    fn from(device: DeviceDomainState) -> Self {
        Self {
            id: device.id,
            state: device.state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct CacheProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pressure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capacity_state: Option<String>,
}

impl From<CacheDomainState> for CacheProjection {
    fn from(cache: CacheDomainState) -> Self {
        Self {
            pressure: cache.pressure,
            capacity_state: cache.capacity_state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StateProjection {
    pub(crate) node: NodeProjection,
    pub(crate) models: Vec<ModelProjection>,
    pub(crate) stages: Vec<StageProjection>,
    pub(crate) sessions: SessionsProjection,
    pub(crate) requests: Vec<RequestProjection>,
    pub(crate) devices: Vec<DeviceProjection>,
    pub(crate) cache: CacheProjection,
}

fn build_from_domain(node: NodeProjection, domain: &DomainState) -> StateProjection {
    StateProjection {
        node,
        models: domain
            .models()
            .into_iter()
            .map(ModelProjection::from)
            .collect(),
        stages: domain
            .stages()
            .into_iter()
            .map(StageProjection::from)
            .collect(),
        sessions: SessionsProjection {
            active_count: domain.sessions_active_count(),
            recent: domain
                .sessions_recent()
                .into_iter()
                .map(SessionRecentProjection::from)
                .collect(),
        },
        requests: domain
            .requests()
            .into_iter()
            .map(RequestProjection::from)
            .collect(),
        devices: domain
            .devices()
            .into_iter()
            .map(DeviceProjection::from)
            .collect(),
        cache: CacheProjection::from(domain.cache()),
    }
}

#[cfg(test)]
pub(crate) fn build(engine: &RuntimeEventEngine) -> StateProjection {
    let snapshot = engine.reducer_snapshot();
    build_from_snapshot(&snapshot)
}

/// Project an already-captured reducer snapshot. Runtime-event stream
/// attachment uses this form so `runtime_state` is built from the same
/// publication boundary as its cursor and initial health frame.
pub(crate) fn build_from_snapshot(
    snapshot: &crate::runtime_events::reducer::ReducerSnapshot,
) -> StateProjection {
    let node = NodeProjection {
        rebuild_generation: snapshot.rebuild_generation,
        tracked_operation_count: snapshot.operation_count(),
    };
    build_from_domain(node, snapshot.domain())
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{
        FactData, FamilyFact, LogicalModelId, ModelAvailabilityEventKind, RuntimeEventIngress,
        ScopeIdentities,
    };

    use super::*;

    fn model_available_fact(model_id: &str) -> mesh_llm_runtime_event_contracts::RuntimeFact {
        mesh_llm_runtime_event_contracts::RuntimeFact::ModelAvailability(FamilyFact::with_data(
            ModelAvailabilityEventKind::ModelAvailable,
            FactData {
                scope: ScopeIdentities {
                    model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                    ..ScopeIdentities::default()
                },
                ..FactData::default()
            },
        ))
    }

    #[test]
    fn a_model_available_fact_populates_the_models_category() {
        let engine = RuntimeEventEngine::new();
        let reservation = engine
            .reserve_root(mesh_llm_runtime_event_contracts::OperationId::new(), || {
                mesh_llm_runtime_event_contracts::RuntimeFact::NativeRuntime(FamilyFact::new(
                    mesh_llm_runtime_event_contracts::NativeRuntimeEventKind::RuntimeStopped,
                ))
            })
            .expect("reserve");
        reservation
            .ingress()
            .try_submit(model_available_fact("qa-model"));
        engine.drain();

        let projection = build(&engine);
        let model = projection
            .models
            .iter()
            .find(|model| model.id == "qa-model")
            .expect("model must be projected onto state.models");
        assert_eq!(model.availability.as_deref(), Some("available"));

        let value = serde_json::to_value(model).expect("serializable");
        let object = value
            .as_object()
            .expect("model projection is a JSON object");
        let keys: std::collections::BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["id", "availability"].into_iter().collect(),
            "unset load_phase/last_outcome must be omitted, not null"
        );
    }

    #[test]
    fn sessions_projection_has_exactly_active_count_and_recent() {
        let engine = RuntimeEventEngine::new();
        let projection = build(&engine);
        let value = serde_json::to_value(&projection.sessions).expect("serializable");
        let object = value.as_object().expect("sessions is a JSON object");
        let keys: std::collections::BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["active_count", "recent"].into_iter().collect());
    }

    #[test]
    fn cache_projection_omits_unset_fields() {
        let engine = RuntimeEventEngine::new();
        let projection = build(&engine);
        let value = serde_json::to_value(&projection.cache).expect("serializable");
        let object = value.as_object().expect("cache is a JSON object");
        assert!(object.is_empty(), "a fresh engine has no cache signal yet");
    }
}
