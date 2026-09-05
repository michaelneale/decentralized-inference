use std::collections::BTreeMap;
use std::time::Duration;

use crate::{
    BoundedNumericSummaries, CarrierKind, CarrierLocation, ChildOperationId, DeviceId, EventId,
    EventSequence, FactData, GenerationEventKind, GenerationFact, HumanSummary, LogicalModelId,
    NumericSummary, NumericSummaryKey, NumericValue, OperationId, OperationScope, Outcome,
    ProcessInstanceId, ProducerSource, Progress, ProgressUnit, ReasonCode, RequestId,
    RuntimeEventEnvelope, RuntimeEventSchemaVersion, RuntimeFact, ScopeIdentities, SessionId,
    Severity, StageId, StageIdentity, StateName, StateTransition, TopologyId, UnknownReasonCode,
};

use super::native::unknown_native_source;

#[test]
fn carrier_location_matrix_is_independently_total() {
    // Given
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/inventory/runtime_events.toml"
    ))
    .expect("inventory should be readable");
    let inventory = toml::from_str::<toml::Value>(&manifest).expect("inventory should parse");
    let recorded = inventory["carriers"]
        .as_array()
        .expect("carrier array")
        .iter()
        .map(|item| {
            (
                item["name"].as_str().expect("carrier name"),
                item["location"].as_str().expect("carrier location"),
            )
        })
        .collect::<BTreeMap<_, _>>();

    // When
    let implemented = CarrierKind::ALL
        .iter()
        .map(|carrier| (carrier.as_str(), carrier.location().as_str()))
        .collect::<BTreeMap<_, _>>();

    // Then
    assert_eq!(implemented, recorded);
    assert_eq!(
        CarrierKind::NativeSequence.location(),
        CarrierLocation::NativeSourceEnvelope
    );
}

#[test]
fn complete_domain_and_native_carriers_survive_envelope_ownership() {
    // Given
    let data = complete_fact_data();
    let native_source = unknown_native_source();
    let process = ProcessInstanceId::from_bytes([1; 16]);
    let root = OperationId::from_bytes([2; 16]);
    let child = ChildOperationId::from_bytes([3; 16]);
    let envelope = RuntimeEventEnvelope {
        schema_version: RuntimeEventSchemaVersion::CURRENT,
        event_id: EventId::new(process, EventSequence::new(9)),
        operation: OperationScope::with_child(root, child),
        producer: ProducerSource::Native,
        severity: Severity::Warning,
        wall_clock_unix_ns: 1_725_148_800_000_000_000,
        process_monotonic_time: Duration::from_secs(7),
        native_source: Some(native_source.clone()),
        fact: RuntimeFact::Generation(GenerationFact::with_data(
            GenerationEventKind::GenerationProgress,
            data.clone(),
        )),
    };

    // When
    let (recovered_fact, recovered_native) = envelope.into_parts();

    // Then
    assert_eq!(recovered_fact.data(), &data);
    assert_eq!(recovered_native, Some(native_source));
}

fn complete_fact_data() -> FactData {
    FactData {
        scope: ScopeIdentities {
            model_id: Some(LogicalModelId::new("model").expect("model id")),
            topology_id: Some(TopologyId::new("topology").expect("topology id")),
            stage: Some(StageIdentity::new(
                StageId::new("stage").expect("stage id"),
                0,
            )),
            session_id: Some(SessionId::new("session").expect("session id")),
            request_id: Some(RequestId::new("request").expect("request id")),
            device_id: Some(DeviceId::new("device").expect("device id")),
        },
        state: Some(StateTransition::new(
            Some(StateName::new("queued").expect("previous state")),
            StateName::new("running").expect("current state"),
        )),
        progress: Some(Progress::new(3, Some(10), ProgressUnit::Steps)),
        outcome: Some(Outcome::Success),
        reason: Some(ReasonCode::Unknown(
            UnknownReasonCode::new("future_reason").expect("unknown reason"),
        )),
        duration: Some(Duration::from_millis(12)),
        numeric_summaries: BoundedNumericSummaries::new(vec![NumericSummary::new(
            NumericSummaryKey::new("tokens").expect("summary key"),
            NumericValue::Unsigned(3),
        )])
        .expect("bounded summaries"),
        summary: Some(HumanSummary::new("bounded summary").expect("summary")),
    }
}
