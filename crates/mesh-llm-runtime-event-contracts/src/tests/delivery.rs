use std::collections::BTreeSet;

use crate::{
    DeliveryClass, DiagnosticEventKind, DiagnosticFact, GenerationEventKind, GenerationFact,
    ModelAvailabilityEventKind, ModelAvailabilityFact, ModelPreparationEventKind,
    ModelPreparationFact, RuntimeFact,
};

#[test]
fn delivery_class_is_derived_only_from_fact_kind() {
    // Given
    let cases = [
        (
            RuntimeFact::ModelPreparation(ModelPreparationFact::new(
                ModelPreparationEventKind::ModelPreparationCompleted,
            )),
            DeliveryClass::Terminal,
        ),
        (
            RuntimeFact::Generation(GenerationFact::new(GenerationEventKind::GenerationProgress)),
            DeliveryClass::Progress,
        ),
        (
            RuntimeFact::Diagnostic(DiagnosticFact::new(DiagnosticEventKind::WarningRaised)),
            DeliveryClass::Diagnostic,
        ),
        (
            RuntimeFact::ModelAvailability(ModelAvailabilityFact::new(
                ModelAvailabilityEventKind::ModelCapacityChanged,
            )),
            DeliveryClass::StateTransition,
        ),
    ];

    // When / Then
    for (fact, expected) in cases {
        assert_eq!(fact.delivery_class(), expected);
    }
}

#[test]
fn all_fifteen_typed_family_facts_have_one_runtime_sum_variant() {
    // Given / When
    let facts = crate::tests::support::one_fact_per_family();

    // Then
    assert_eq!(facts.len(), 15);
    assert_eq!(facts[0].kind_id(), "runtime_resolution_started");
    assert_eq!(facts[14].kind_id(), "ingress_queue_pressure");
}

#[test]
fn every_inventory_kind_has_its_exact_derived_delivery_class() {
    // Given
    let inventory_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/inventory/runtime_events.toml"
    ))
    .expect("inventory should be readable");
    let inventory = toml::from_str::<toml::Value>(&inventory_text).expect("inventory should parse");
    let terminal_ids = catalog_ids(&inventory, "terminal_event_ids");
    let progress_ids = catalog_ids(&inventory, "progress_event_ids");
    let diagnostic_ids = DiagnosticEventKind::ALL
        .iter()
        .map(|kind| kind.as_str())
        .filter(|id| !terminal_ids.contains(id))
        .collect::<BTreeSet<_>>();

    // When
    let classified = crate::tests::support::all_facts()
        .into_iter()
        .map(|fact| (fact.kind_id(), fact.delivery_class()))
        .collect::<Vec<_>>();

    // Then
    assert_eq!(classified.len(), 184);
    assert_eq!(
        classified
            .iter()
            .map(|(id, _)| id)
            .collect::<BTreeSet<_>>()
            .len(),
        184
    );
    for (id, actual) in classified {
        let expected = if terminal_ids.contains(id) {
            DeliveryClass::Terminal
        } else if progress_ids.contains(id) {
            DeliveryClass::Progress
        } else if diagnostic_ids.contains(id) {
            DeliveryClass::Diagnostic
        } else {
            DeliveryClass::StateTransition
        };
        assert_eq!(actual, expected, "delivery mismatch for {id}");
    }
}

fn catalog_ids<'a>(inventory: &'a toml::Value, key: &str) -> BTreeSet<&'a str> {
    inventory["catalog"][key]
        .as_array()
        .expect("catalog list")
        .iter()
        .map(|value| value.as_str().expect("event id"))
        .collect()
}
