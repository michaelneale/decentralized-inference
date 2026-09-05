//! Module-wiring tests: connection-shape classification and wire-payload
//! allowlist/required-field properties. Black-box HTTP/SSE byte tests live
//! under `crate::api::tests::runtime_events_v1` (they need the full
//! `MeshApi` + TCP harness already established there).

use std::sync::Arc;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, EventSequence, FactData, FamilyFact, OperationId, OperationScope, Outcome,
    ProcessInstanceId, ReasonCode, RequestEventKind, RequestId, RuntimeEventIngress, RuntimeFact,
    ScopeIdentities,
};

use crate::runtime_events::config::{EngineConfig, REPLAY_MAX_AGE};
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::replay::ReplayFrame;

use super::cursor::Cursor;
use super::frames::{EVENT_PROJECTION_ALLOWLIST, REQUIRED_ENVELOPE_KEYS, STATE_TOP_LEVEL_KEYS};
use super::recovery::{ConnectionShape, Gap, GapReason, classify_attachment};

fn synthetic_unknown() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

fn terminal_success() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

fn submit_and_drain(engine: &std::sync::Arc<RuntimeEventEngine>, count: usize) {
    for _ in 0..count {
        let reservation = engine
            .reserve_root(
                mesh_llm_runtime_event_contracts::OperationId::new(),
                synthetic_unknown,
            )
            .expect("reserve");
        reservation.ingress().try_submit(terminal_success());
    }
    engine.drain();
}

fn classify_current(
    engine: &std::sync::Arc<RuntimeEventEngine>,
    requested: Option<Cursor>,
) -> Result<ConnectionShape, super::cursor::CursorError> {
    let attachment = engine.attach().expect("attach");
    classify_attachment(&attachment, engine.process_instance(), requested)
}

#[test]
fn no_cursor_classifies_as_no_cursor() {
    let engine = RuntimeEventEngine::new();
    assert!(matches!(
        classify_current(&engine, None).expect("classify"),
        ConnectionShape::NoCursor
    ));
}

#[test]
fn in_window_cursor_returns_all_published_frames_strictly_after_it() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 3);

    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::InWindow { frames } =
        classify_current(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected in-window shape");
    };
    let sequences: Vec<u64> = frames.iter().map(|frame| frame.sequence.get()).collect();
    assert_eq!(sequences, vec![1, 2, 3]);
}

#[test]
fn future_sequence_for_current_instance_is_rejected() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 1);

    let cursor = Cursor::new(engine.process_instance(), 5);
    assert!(classify_current(&engine, Some(cursor)).is_err());
}

#[test]
fn cursor_before_any_event_for_a_fresh_engine_is_in_window_and_empty() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::InWindow { frames } =
        classify_current(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected in-window shape");
    };
    assert!(frames.is_empty());
}

#[test]
fn stale_instance_is_a_gap() {
    let engine = RuntimeEventEngine::new();
    let foreign = mesh_llm_runtime_event_contracts::ProcessInstanceId::new();
    let cursor = Cursor::new(foreign, 0);
    let ConnectionShape::Gap(gap) = classify_current(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected gap shape");
    };
    assert_eq!(gap.reason, GapReason::StaleInstance);
}

#[test]
fn evicted_after_rebuild_is_a_gap() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 2);
    engine.rebuild();

    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::Gap(gap) = classify_current(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected gap shape");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
}

#[test]
fn cursor_at_the_rebuild_frontier_requires_a_fresh_snapshot() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 1);
    engine.rebuild();
    let attachment = engine.attach().expect("attach");
    let cursor = Cursor::new(engine.process_instance(), attachment.published_frontier);

    let ConnectionShape::Gap(gap) =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("a cursor at the invalidated rebuild frontier must be a gap");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
}

#[test]
fn attachment_classification_reports_a_gap_for_a_read_time_expired_published_frame() {
    let engine = RuntimeEventEngine::new();
    let mut attachment = engine.attach().expect("attach");
    attachment.replay = vec![
        replay_frame_at(5, Instant::now() - REPLAY_MAX_AGE - Duration::from_secs(1)),
        replay_frame_at(7, Instant::now()),
    ];
    attachment.published_frontier = 7;
    attachment.replay_evicted_through = None;

    let cursor = Cursor::new(engine.process_instance(), 4);
    let ConnectionShape::Gap(gap) =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("a cursor before an expired published frame must be a gap");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
    assert_eq!(gap.oldest_available, Some(7));
    assert_eq!(gap.latest, Some(7));
}

#[test]
fn attachment_classification_keeps_published_sequence_holes_in_window() {
    let engine = RuntimeEventEngine::new();
    let mut attachment = engine.attach().expect("attach");
    attachment.replay = vec![
        replay_frame_at(5, Instant::now()),
        replay_frame_at(7, Instant::now()),
    ];
    attachment.published_frontier = 7;
    attachment.replay_evicted_through = None;

    let cursor = Cursor::new(engine.process_instance(), 4);
    let ConnectionShape::InWindow { frames } =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("rejected or coalesced sequence holes must remain in-window");
    };
    let sequences: Vec<u64> = frames.iter().map(|frame| frame.sequence.get()).collect();
    assert_eq!(sequences, vec![5, 7]);
}

#[test]
fn attachment_classification_preserves_eviction_watermark_when_replay_is_empty() {
    let engine = RuntimeEventEngine::new();
    let mut attachment = engine.attach().expect("attach");
    attachment.published_frontier = 7;
    attachment.replay.clear();
    attachment.replay_evicted_through = Some(7);

    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::Gap(gap) =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("an empty replay with an evicted published frontier must be a gap");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
    assert_eq!(gap.oldest_available, None);
    assert_eq!(gap.latest, None);
}

#[test]
fn attachment_classification_keeps_pristine_empty_replay_in_window() {
    let engine = RuntimeEventEngine::new();
    let attachment = engine.attach().expect("attach");
    let cursor = Cursor::new(engine.process_instance(), 0);

    let ConnectionShape::InWindow { frames } =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("a pristine empty replay must remain in-window");
    };
    assert!(frames.is_empty());
}

#[test]
fn attachment_classification_all_stale_frames_is_in_window_when_caught_up() {
    let engine = RuntimeEventEngine::new();
    let mut attachment = engine.attach().expect("attach");
    attachment.replay = vec![replay_frame_at(
        5,
        Instant::now() - REPLAY_MAX_AGE - Duration::from_secs(1),
    )];
    // Keep the captured frontier ahead of the cursor so this exercises the
    // read-time lookup instead of the already-caught-up frontier shortcut.
    attachment.published_frontier = 6;

    let cursor = Cursor::new(engine.process_instance(), 5);
    let ConnectionShape::InWindow { frames } =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("a caught-up cursor must remain in-window after all frames stale");
    };
    assert!(frames.is_empty());
}

#[test]
fn attachment_classification_all_stale_frames_is_gap_when_behind() {
    let engine = RuntimeEventEngine::new();
    let mut attachment = engine.attach().expect("attach");
    attachment.replay = vec![replay_frame_at(
        5,
        Instant::now() - REPLAY_MAX_AGE - Duration::from_secs(1),
    )];
    attachment.published_frontier = 6;

    let cursor = Cursor::new(engine.process_instance(), 4);
    let ConnectionShape::Gap(gap) =
        classify_attachment(&attachment, engine.process_instance(), Some(cursor))
            .expect("classify attachment")
    else {
        panic!("a behind cursor must report all-stale replay as a gap");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
    assert_eq!(gap.oldest_available, None);
    assert_eq!(gap.latest, Some(5));
}

#[test]
fn state_top_level_keys_match_the_frozen_set() {
    let engine = RuntimeEventEngine::new();
    let projection = crate::runtime_event_api::state_projection::build(&engine);
    let value = serde_json::to_value(&projection).expect("serializable");
    let object = value.as_object().expect("state is a JSON object");
    let keys: Vec<&str> = object.keys().map(String::as_str).collect();
    for required in STATE_TOP_LEVEL_KEYS {
        assert!(keys.contains(required), "missing state key {required}");
    }
    assert_eq!(keys.len(), STATE_TOP_LEVEL_KEYS.len());
}

#[test]
fn event_projection_keys_are_a_subset_of_the_allowlist_for_every_submitted_kind() {
    let facts = [terminal_success(), synthetic_unknown()];
    let scope = OperationScope::root_only(OperationId::new());
    for fact in facts {
        let projection = super::frames::event_projection(&fact, scope);
        let value = serde_json::to_value(&projection).expect("serializable");
        let object = value.as_object().expect("event is a JSON object");
        for key in object.keys() {
            assert!(
                EVENT_PROJECTION_ALLOWLIST.contains(&key.as_str()),
                "key {key} is not in the projected-key allowlist"
            );
        }
    }
}

#[test]
fn envelope_frame_carries_every_required_key() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::state_frame(&engine, cursor);
    let data_line = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line present");
    let value: serde_json::Value = serde_json::from_str(data_line).expect("valid JSON");
    let object = value.as_object().expect("envelope is a JSON object");
    for required in REQUIRED_ENVELOPE_KEYS {
        assert!(object.contains_key(*required), "missing {required}");
    }
}

#[test]
fn every_frame_is_exactly_id_event_data_blank_line() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::health_frame(&engine, cursor);
    let mut lines = frame.split('\n');
    assert!(lines.next().unwrap().starts_with("id: rt1:"));
    assert_eq!(lines.next().unwrap(), "event: runtime_health");
    assert!(lines.next().unwrap().starts_with("data: "));
    assert_eq!(lines.next().unwrap(), "");
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

#[test]
fn keepalive_frame_has_no_id_or_data() {
    assert_eq!(super::frames::KEEPALIVE_FRAME, ": keepalive\n\n");
}

// ─── Task 8: versioned health with bounds and a nullable ingress p99 ───

#[test]
fn health_projection_bounds_equal_the_frozen_engine_config() {
    let engine = RuntimeEventEngine::new();
    let projection: super::frames::HealthProjection = engine.health().snapshot().into();
    let bounds = EngineConfig::FROZEN;
    assert_eq!(
        projection.bounds.reservation_table_capacity,
        bounds.reservation_table_capacity
    );
    assert_eq!(
        projection.bounds.state_transition_lane_depth,
        bounds.state_transition_lane_depth
    );
    assert_eq!(
        projection.bounds.diagnostic_lane_depth,
        bounds.diagnostic_lane_depth
    );
    assert_eq!(projection.bounds.wake_list_depth, bounds.wake_list_depth);
    assert_eq!(
        projection.bounds.replay_max_frames,
        bounds.replay_max_frames
    );
    assert_eq!(
        projection.bounds.subscriber_lag_max_frames,
        bounds.subscriber_lag_max_frames
    );
    assert_eq!(
        projection.bounds.max_concurrent_subscribers,
        bounds.max_concurrent_subscribers
    );
}

#[test]
fn health_frame_wire_json_carries_version_bounds_and_a_null_ingress_p99() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::health_frame(&engine, cursor);
    let data_line = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line present");
    let value: serde_json::Value = serde_json::from_str(data_line).expect("valid JSON");
    let health = value["health"].as_object().expect("health is an object");

    assert!(health.contains_key("version"), "version must be present");
    let bounds = health["bounds"]
        .as_object()
        .expect("bounds is a present object");
    assert!(bounds.contains_key("reservation_table_capacity"));
    assert!(bounds.contains_key("max_concurrent_subscribers"));
    assert_eq!(health["state_transition_rejected"], 0);
    assert_eq!(health["cancelled_reservation_rejected"], 0);
    assert_eq!(health["state_degraded"], false);
    assert_eq!(health["rebuild_required"], false);
    assert!(
        health["ingress_p99_us"].is_null(),
        "task 8 lands this field as always-nullable until task 13"
    );
}

#[test]
fn health_frame_wire_json_populates_ingress_p99_us_after_the_minimum_sample_threshold() {
    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeFact,
    };

    let engine = RuntimeEventEngine::new();
    for _ in 0..100 {
        let scope = OperationScope::root_only(OperationId::new());
        let fact =
            RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized));
        let _ = engine.unreserved_ingress(scope).try_submit(fact);
    }

    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::health_frame(&engine, cursor);
    let data_line = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line present");
    let value: serde_json::Value = serde_json::from_str(data_line).expect("valid JSON");
    let health = value["health"].as_object().expect("health is an object");

    assert!(
        health["ingress_p99_us"].is_u64(),
        "after >=100 submissions, ingress_p99_us must be a populated number, got: {:?}",
        health["ingress_p99_us"]
    );
}

// ─── Shared Rust/TS fixture round-trip ─────────────────────────────────

const FRAMES_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/frames.json"
);
const CURSORS_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/cursors.json"
);
const RECOVERY_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/recovery.json"
);

#[test]
fn frames_fixture_required_keys_match_the_rust_constants() {
    let fixture: serde_json::Value = serde_json::from_str(FRAMES_FIXTURE).expect("valid JSON");
    let required: Vec<&str> = fixture["required_envelope_keys"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(required, REQUIRED_ENVELOPE_KEYS);

    let state_keys: Vec<&str> = fixture["state_top_level_keys"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(state_keys, STATE_TOP_LEVEL_KEYS);

    let allowlist: Vec<&str> = fixture["event_projected_key_allowlist"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(allowlist, EVENT_PROJECTION_ALLOWLIST);

    assert_eq!(fixture["keepalive_frame"], super::frames::KEEPALIVE_FRAME);
}

#[test]
fn cursors_fixture_examples_parse_exactly_as_declared() {
    let fixture: serde_json::Value = serde_json::from_str(CURSORS_FIXTURE).expect("valid JSON");
    for value in fixture["valid"].as_array().expect("array") {
        let text = value.as_str().expect("string");
        assert!(Cursor::parse(text).is_ok(), "must parse: {text}");
    }
    for value in fixture["invalid"].as_array().expect("array") {
        let text = value.as_str().expect("string");
        assert!(Cursor::parse(text).is_err(), "must reject: {text}");
    }
}

#[test]
fn recovery_fixture_gap_reasons_match_the_rust_enum() {
    let fixture: serde_json::Value = serde_json::from_str(RECOVERY_FIXTURE).expect("valid JSON");
    let reasons = fixture["replay_gap_reasons"].as_object().expect("object");
    assert!(reasons.contains_key(GapReason::StaleInstance.as_str()));
    assert!(reasons.contains_key(GapReason::Evicted.as_str()));
}

// ─── Task 7: operation identity + always-present base keys ─────────────

#[test]
fn producer_and_severity_are_always_present_on_every_event_projection() {
    let scope = OperationScope::root_only(OperationId::new());
    for fact in [terminal_success(), synthetic_unknown()] {
        let value = serde_json::to_value(super::frames::event_projection(&fact, scope))
            .expect("serializable");
        let object = value.as_object().expect("event is a JSON object");
        assert!(
            object.contains_key("producer"),
            "producer must always be present"
        );
        assert!(
            object.contains_key("severity"),
            "severity must always be present"
        );
    }
}

#[test]
fn root_and_child_operation_scopes_are_distinguishable_on_the_wire() {
    let root_id = OperationId::new();
    let root_scope = OperationScope::root_only(root_id);
    let child_scope = OperationScope::with_child(root_id, ChildOperationId::new());

    let root_value = serde_json::to_value(super::frames::event_projection(
        &terminal_success(),
        root_scope,
    ))
    .expect("serializable");
    let child_value = serde_json::to_value(super::frames::event_projection(
        &terminal_success(),
        child_scope,
    ))
    .expect("serializable");

    let root_operation = root_value["operation"]
        .as_object()
        .expect("operation present");
    assert_eq!(root_operation["root"], root_id.to_string());
    assert!(
        !root_operation.contains_key("child"),
        "a root scope must not carry a child id"
    );

    let child_operation = child_value["operation"]
        .as_object()
        .expect("operation present");
    assert_eq!(child_operation["root"], root_id.to_string());
    assert!(
        child_operation.contains_key("child"),
        "a child scope must carry a child id"
    );
}

// ─── Task 7: byte-exact sample frames for all four frame types ─────────

const SAMPLE_INSTANCE_UUID: &str = "0195f000-0000-7000-8000-000000000001";
const SAMPLE_ROOT_UUID: &str = "0195f000-0000-7000-8000-0000000000aa";
const SAMPLE_CHILD_UUID: &str = "0195f000-0000-7000-8000-0000000000bb";

fn parse_fixture_uuid(text: &str) -> uuid::Uuid {
    uuid::Uuid::parse_str(text).expect("valid fixture uuid")
}

fn sample_instance() -> ProcessInstanceId {
    ProcessInstanceId::from_uuid(parse_fixture_uuid(SAMPLE_INSTANCE_UUID))
}

fn sample_scope() -> OperationScope {
    OperationScope::with_child(
        OperationId::from_uuid(parse_fixture_uuid(SAMPLE_ROOT_UUID)),
        ChildOperationId::from_uuid(parse_fixture_uuid(SAMPLE_CHILD_UUID)),
    )
}

fn sample_request_completed_fact() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            scope: ScopeIdentities {
                request_id: Some(RequestId::new(SAMPLE_ROOT_UUID).expect("valid request id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn replay_frame_at(sequence: u64, recorded_at: Instant) -> ReplayFrame {
    ReplayFrame {
        sequence: EventSequence::new(sequence),
        rebuild_generation: 0,
        scope: sample_scope(),
        fact: Arc::new(sample_request_completed_fact()),
        recorded_at,
        wire_bytes: Arc::from(Vec::<u8>::new()),
    }
}

fn sample_gap() -> Gap {
    Gap {
        reason: GapReason::Evicted,
        requested: Cursor::new(sample_instance(), 5),
        oldest_available: Some(10),
        latest: Some(20),
    }
}

/// One canonical, deterministic sample per frame type, pinned byte-exact
/// against `fixtures/runtime_events_v1/frames.json`'s `sample_frames` and
/// consumed identically by `use-runtime-events-v1.test.tsx` on the
/// TypeScript side (parsed through `parseSseBlock`/`parseRuntimeEventsV1Frame`).
/// Every input is a fixed constant (instance/root/child UUIDs, a fresh
/// zero-state engine, a hand-built fact/gap) so re-running this test can
/// never produce a different byte sequence.
#[test]
fn sample_frames_fixture_is_byte_exact_for_every_frame_type() {
    let fixture: serde_json::Value = serde_json::from_str(FRAMES_FIXTURE).expect("valid JSON");
    let samples = fixture["sample_frames"]
        .as_object()
        .expect("sample_frames object present in the fixture");

    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(sample_instance(), 0);

    assert_eq!(
        super::frames::state_frame(&engine, cursor),
        samples["runtime_state"]
            .as_str()
            .expect("runtime_state sample")
    );
    assert_eq!(
        super::frames::health_frame(&engine, cursor),
        samples["runtime_health"]
            .as_str()
            .expect("runtime_health sample")
    );

    let frame = ReplayFrame {
        sequence: EventSequence::new(1),
        rebuild_generation: 0,
        scope: sample_scope(),
        fact: std::sync::Arc::new(sample_request_completed_fact()),
        recorded_at: std::time::Instant::now(),
        // `event_frame_at` (called below) never reads `wire_bytes` --
        // added by task 9, `.omo/plans/event-system-fixes.md` -- so an
        // empty placeholder does not affect this byte-exactness assertion.
        wire_bytes: std::sync::Arc::from(Vec::new()),
    };
    assert_eq!(
        super::frames::event_frame_at(sample_instance(), &frame),
        samples["runtime_event"]
            .as_str()
            .expect("runtime_event sample")
    );

    assert_eq!(
        super::frames::replay_gap_frame_at(sample_instance(), 0, &sample_gap()),
        samples["runtime_replay_gap"]
            .as_str()
            .expect("runtime_replay_gap sample")
    );
}
