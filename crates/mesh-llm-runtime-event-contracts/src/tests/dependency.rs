use std::collections::BTreeSet;

#[test]
fn contract_crate_has_no_mesh_or_presentation_dependency_edge() {
    // Given
    let manifest = parse_manifest(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));

    // When
    let dependencies = manifest["dependencies"]
        .as_table()
        .expect("dependencies table")
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();

    // Then
    assert_eq!(dependencies, BTreeSet::from(["uuid"]));
}

#[test]
fn mesh_client_does_not_depend_on_runtime_event_contracts() {
    // Given
    let manifest = parse_manifest(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../mesh-client/Cargo.toml"
    ));

    // When
    let has_contract_edge = manifest["dependencies"]
        .as_table()
        .expect("dependencies table")
        .contains_key("mesh-llm-runtime-event-contracts");

    // Then
    assert!(!has_contract_edge);
}

fn parse_manifest(path: &str) -> toml::Value {
    let text = std::fs::read_to_string(path).expect("manifest should be readable");
    toml::from_str(&text).expect("manifest should parse")
}
