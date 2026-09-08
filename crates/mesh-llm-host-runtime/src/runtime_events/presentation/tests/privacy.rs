//! Privacy: presentation output is deny-by-default. Every projected
//! fragment key must be in the fixed allowlist, an empty fact projects
//! nothing beyond category/kind, and the bounded human summary passes
//! through unmodified with nothing else leaking alongside it.

use mesh_llm_events::OutputEvent;

use super::super::projection::{
    PROJECTED_FRAGMENT_KEYS, fact_projection_event, projected_fragments,
};
use super::{kitchen_sink_terminal_fact, terminal_fact};

#[test]
fn every_projected_fragment_key_is_in_the_deny_by_default_allowlist() {
    for fact in [terminal_fact(), kitchen_sink_terminal_fact()] {
        for (key, _value) in projected_fragments(&fact) {
            assert!(
                PROJECTED_FRAGMENT_KEYS.contains(&key),
                "projected fragment key {key:?} is not in the deny-by-default allowlist"
            );
        }
    }
}

#[test]
fn a_fact_with_no_optional_data_projects_only_the_category_and_kind() {
    let fragments = projected_fragments(&terminal_fact());
    let keys: Vec<&str> = fragments.iter().map(|(key, _)| *key).collect();
    assert_eq!(
        keys,
        vec!["category", "kind"],
        "an empty FactData must not manufacture any other fragment"
    );
}

#[test]
fn a_bounded_human_summary_passes_through_unmodified() {
    let fragments = projected_fragments(&kitchen_sink_terminal_fact());
    let summary = fragments
        .iter()
        .find(|(key, _)| *key == "summary")
        .map(|(_, value)| value.clone());
    assert_eq!(summary.as_deref(), Some("bounded summary"));
}

#[test]
fn the_privacy_safe_message_never_contains_a_path_separator_or_url_scheme() {
    let OutputEvent::Info { message, .. } = fact_projection_event(&kitchen_sink_terminal_fact())
    else {
        panic!("expected Info");
    };
    assert!(
        !message.contains("://"),
        "no URL scheme may appear in a presentation message"
    );
    assert!(
        !message.contains('/'),
        "no path fragment may appear in a presentation message"
    );
}
