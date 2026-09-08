//! The plan's eleven-row field matrix, at producer-write-method
//! granularity: `RuntimeDataProducer`'s non-`#[cfg(test)]` write entry
//! points in `producers.rs`, each mapped to the `FieldId` dirty scope it
//! publishes under. `tests::producer_write_methods_match_producers_source`
//! mechanically re-derives the method-name set from `producers.rs` itself
//! (not a hand-copied allowlist) and asserts it equals this list, so a
//! rename or a new write method desyncs the test instead of staying green.

use super::matrix::FieldId;

/// One producer write entry point and the dirty scope it publishes under.
pub(crate) struct ProducerWriteMethod {
    pub method: &'static str,
    pub field: FieldId,
}

/// The eleven non-test `RuntimeDataProducer` write methods, each mapped to
/// its `FieldId`. Kept as a hand-authored field mapping (mechanical set
/// extraction alone can't infer WHICH dirty scope a method delegates to
/// across `producers.rs`/`collector.rs`); its NAME set is what `tests`
/// mechanically verifies against the real source.
pub(crate) const PRODUCER_WRITE_METHODS: [ProducerWriteMethod; 11] = [
    ProducerWriteMethod {
        method: "mark_status_dirty",
        field: FieldId::Status,
    },
    ProducerWriteMethod {
        method: "publish_runtime_status",
        field: FieldId::Status,
    },
    ProducerWriteMethod {
        method: "publish_local_processes",
        field: FieldId::Processes,
    },
    ProducerWriteMethod {
        method: "replace_local_instances_snapshot",
        field: FieldId::Inventory,
    },
    ProducerWriteMethod {
        method: "publish_routing_snapshot",
        field: FieldId::Routing,
    },
    ProducerWriteMethod {
        method: "publish_llama_slots_snapshot",
        field: FieldId::Runtime,
    },
    ProducerWriteMethod {
        method: "publish_plugin_summary",
        field: FieldId::Plugins,
    },
    ProducerWriteMethod {
        method: "publish_plugin_manifest",
        field: FieldId::Plugins,
    },
    ProducerWriteMethod {
        method: "publish_plugin_providers",
        field: FieldId::Plugins,
    },
    ProducerWriteMethod {
        method: "publish_plugin_endpoint",
        field: FieldId::Plugins,
    },
    ProducerWriteMethod {
        method: "clear_plugin_reports",
        field: FieldId::Plugins,
    },
];

/// Mechanically extract every non-`#[cfg(test)]` `pub(crate) fn` in
/// `source` whose name starts with a write-verb prefix already used
/// consistently throughout `producers.rs` (`mark_`, `publish_`,
/// `replace_`, `clear_`, `update_`). This is the real cross-check for
/// `PRODUCER_WRITE_METHODS`: a rename, deletion, or new write method
/// changes this extracted set and desyncs the equality test in `tests`.
pub(crate) fn extract_non_test_write_method_names(source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let mut names = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix("pub(crate) fn ") else {
            continue;
        };
        let name = rest.split(['(', '<']).next().unwrap_or("").to_string();
        let is_write = name.starts_with("mark_")
            || name.starts_with("publish_")
            || name.starts_with("replace_")
            || name.starts_with("clear_")
            || name.starts_with("update_");
        let previous_line_is_cfg_test = index > 0 && lines[index - 1].trim() == "#[cfg(test)]";
        if is_write && !previous_line_is_cfg_test {
            names.push(name);
        }
    }
    names
}
