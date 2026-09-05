use super::*;
use mesh_llm_runtime_event_contracts::SessionEventKind;

/// Real content-boundary proof at the redacted PROJECTION path (not just
/// the `FactData` shape): seed a `GenerationReceipt` whose UNUSED fields
/// carry unmistakable prohibited markers -- a fake prompt-derived digest,
/// distinctive token IDs, and an agent-session string that would never
/// legitimately appear in a numeric-summary/enum-only fact -- run it
/// through every fact-construction function this adapter has, and prove
/// the marker bytes/strings are absent from the resulting facts' `Debug`
/// representation (the closest thing to "serialized output" available:
/// `FactData` deliberately does not implement `Serialize`, per this
/// session's own constraint, so `Debug` is the real text projection a
/// human or log line would ever see this fact through).
#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn redacted_projection_never_leaks_seeded_prohibited_markers() {
    const PROHIBITED_AGENT_SESSION: &str = "PROHIBITED-AGENT-SESSION-MARKER-Q7X9";
    const PROHIBITED_TOKEN_ID: i32 = 0x5EED_71D0; // unmistakable, not a plausible real token id
    let mut poisoned = GenerationReceipt::test_fixture_with_full_state(
        1,
        2,
        GenerationTermination::MaxTokens,
        4096,
    );
    poisoned.agent_session_id = Some(PROHIBITED_AGENT_SESSION.into());
    poisoned.prompt_token_digest = [0xAB; 32];
    poisoned.prompt_token_ids = std::sync::Arc::from([PROHIBITED_TOKEN_ID; 4]);
    poisoned.generated_token_ids = vec![PROHIBITED_TOKEN_ID; 4].into_boxed_slice();

    let generation = format!("{:?}", receipt_terminal_fact(&poisoned));
    let prefill = format!("{:?}", prefill_terminal_fact(&poisoned));
    let session = format!("{:?}", session_active_fact());
    let combined = format!("{generation}{prefill}{session}");

    assert!(
        !combined.contains(PROHIBITED_AGENT_SESSION),
        "agent session id must never reach a fact"
    );
    assert!(
        !combined.contains(&PROHIBITED_TOKEN_ID.to_string()),
        "token IDs must never reach a fact"
    );
    assert!(
        !combined.contains("ab, ab, ab") && !combined.contains("171, 171, 171"),
        "the prompt token digest bytes must never reach a fact"
    );
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn no_prohibited_fields_reach_the_fact_data_shape() {
    // FactData structurally has no field capable of carrying token IDs,
    // digests, or prompt content -- scope, state, progress, outcome,
    // reason, duration, bounded numeric summaries, and a bounded human
    // summary only. This adapter never sets `summary`, and its numeric
    // summaries carry only counts/microseconds/byte lengths.
    let fact = receipt_terminal_fact(&receipt(1, 2, GenerationTermination::MaxTokens));
    let RuntimeFact::Generation(fact) = fact else {
        panic!("expected a Generation fact");
    };
    assert!(fact.data().summary.is_none());
    assert_eq!(
        fact.data().scope,
        mesh_llm_runtime_event_contracts::ScopeIdentities::default()
    );
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn session_fact_kinds_carry_no_scope_identifiers() {
    let fact = session_active_fact();
    let RuntimeFact::Session(fact) = fact else {
        panic!("expected a Session fact");
    };
    assert_eq!(*fact.kind(), SessionEventKind::SessionActive);
    assert_eq!(
        fact.data().scope,
        mesh_llm_runtime_event_contracts::ScopeIdentities::default()
    );
}
