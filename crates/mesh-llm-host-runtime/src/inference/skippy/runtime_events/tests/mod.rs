//! Shared fixtures for `runtime_events` tests, split by responsibility:
//! `generation_lifecycle` (correlation, cleanup, composite fan-out) and
//! `session_prefill_kv` (the co-derived session/prefill/KV facts and
//! privacy proofs).

mod concurrent_roots;
mod edge_cases;
mod generation_lifecycle;
mod privacy;
mod session_prefill_kv;

use super::*;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
use mesh_llm_runtime_event_contracts::{
    GenerationEventKind, NumericValue, PrefillEventKind, SessionEventKind,
};
use skippy_server::frontend::GenerationTermination;
use std::sync::Arc;
use std::sync::Arc as StdArc;

fn install_test_engine() -> StdArc<RuntimeEventEngine> {
    clear_runtime_event_engine();
    let engine = RuntimeEventEngine::new();
    install_runtime_event_engine(engine.clone());
    engine
}

fn start(
    request_id: u64,
    session_id: u64,
    frontend_request_id: Option<[u8; 16]>,
) -> GenerationStart {
    GenerationStart {
        request_id,
        session_id,
        agent_session_id: None,
        prompt_token_ids: Arc::from([1, 2, 3]),
        frontend_request_id,
    }
}

fn receipt(
    request_id: u64,
    session_id: u64,
    termination: GenerationTermination,
) -> GenerationReceipt {
    GenerationReceipt::test_fixture(request_id, session_id, termination)
}

fn generation_kinds(engine: &RuntimeEventEngine) -> Vec<GenerationEventKind> {
    engine
        .replay()
        .snapshot()
        .into_iter()
        .filter_map(|frame| match frame.fact.as_ref() {
            RuntimeFact::Generation(fact) => Some(*fact.kind()),
            _ => None,
        })
        .collect()
}

fn prefill_kinds(engine: &RuntimeEventEngine) -> Vec<PrefillEventKind> {
    engine
        .replay()
        .snapshot()
        .into_iter()
        .filter_map(|frame| match frame.fact.as_ref() {
            RuntimeFact::Prefill(fact) => Some(*fact.kind()),
            _ => None,
        })
        .collect()
}

fn session_kinds(engine: &RuntimeEventEngine) -> Vec<SessionEventKind> {
    engine
        .replay()
        .snapshot()
        .into_iter()
        .filter_map(|frame| match frame.fact.as_ref() {
            RuntimeFact::Session(fact) => Some(*fact.kind()),
            _ => None,
        })
        .collect()
}
