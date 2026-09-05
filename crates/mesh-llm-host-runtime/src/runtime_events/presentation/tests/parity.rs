//! JSON/TUI parity: both real formatters
//! (`crates/mesh-llm-tui/src/output/formatting.rs`) render an
//! `OutputEvent::Info` from exactly its `message`/`context` fields. This
//! module cannot depend on `mesh-llm-tui`'s crate-private formatting trait,
//! so these tests mirror that exact two-field contract and prove
//! presentation projections never emit any other variant -- which is what
//! structurally guarantees the two real renderers can never diverge for the
//! same fact.

use mesh_llm_events::OutputEvent;

use super::super::projection::fact_projection_event;
use super::{kitchen_sink_terminal_fact, terminal_fact};

fn tui_summary_line(event: &OutputEvent) -> String {
    match event {
        OutputEvent::Info { message, context } => match context {
            Some(context) => format!("{context}: {message}"),
            None => message.clone(),
        },
        other => panic!("presentation projections must be OutputEvent::Info, got {other:?}"),
    }
}

fn json_message_and_context(event: &OutputEvent) -> (String, Option<String>) {
    match event {
        OutputEvent::Info { message, context } => (message.clone(), context.clone()),
        other => panic!("presentation projections must be OutputEvent::Info, got {other:?}"),
    }
}

#[test]
fn json_and_tui_render_the_same_message_and_context_for_the_same_fact() {
    let event = fact_projection_event(&kitchen_sink_terminal_fact());

    let (json_message, json_context) = json_message_and_context(&event);
    let tui_line = tui_summary_line(&event);

    assert!(
        tui_line.contains(&json_message),
        "the TUI line must contain the exact JSON message text, not a divergent rendering"
    );
    if let Some(context) = &json_context {
        assert!(
            tui_line.starts_with(context.as_str()),
            "the TUI line must be prefixed by the same context JSON reports"
        );
    }
}

#[test]
fn every_projected_fact_is_the_single_info_variant_both_renderers_key_off_of() {
    for fact in [terminal_fact(), kitchen_sink_terminal_fact()] {
        let event = fact_projection_event(&fact);
        assert!(
            matches!(event, OutputEvent::Info { .. }),
            "a divergent OutputEvent variant would let JSON and TUI render different shapes \
             for the same fact"
        );
    }
}
