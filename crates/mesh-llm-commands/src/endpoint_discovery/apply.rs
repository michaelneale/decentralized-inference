//! Turn discovered local servers into an `openai-endpoint` plugin entry in
//! `~/.mesh-llm/config.toml`.
//!
//! Only one endpoint can be published today: `resolve_plugins` rejects a second
//! `[[plugin]]` block with the same name (`plugin/config.rs`), and
//! `PluginConfigEntry.url` holds a single URL. The plan therefore selects one
//! endpoint and reports the rest as unpublishable rather than silently
//! dropping them.

use super::probe::{DiscoveredEndpoint, OPENAI_ENDPOINT_PLUGIN, discover_local_endpoints};
use crate::terminal::{style_muted, style_ok, style_warn};
use anyhow::{Context, Result};
use mesh_llm_config::ConfigStore;
use std::path::Path;

/// The `openai-endpoint` entry already present in the config, if any.
///
/// The `enabled` flag matters independently of `url`: an entry the operator
/// disabled on purpose must not be silently re-enabled.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExistingEndpointEntry {
    pub url: Option<String>,
    pub enabled: bool,
}

/// What `discover` intends to do about the servers it found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EndpointPublishStatus {
    /// Nothing answered a probe.
    NothingFound,
    /// `openai-endpoint` is already pointed at this URL; nothing to change.
    AlreadyConfigured { base_url: String },
    /// `openai-endpoint` is configured for a different URL. Left alone.
    ConfiguredElsewhere { configured: String },
    /// The operator disabled `openai-endpoint`. Treated as an opt-out.
    DisabledByOperator,
    /// Servers answered, but none is serving a model, so there is nothing to
    /// publish yet.
    NoModelsServed,
    /// Write `selected` into the config.
    Publish { selected: DiscoveredEndpoint },
}

/// A full plan: what to publish, and what had to be left out.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointPublishPlan {
    pub status: EndpointPublishStatus,
    pub found: Vec<DiscoveredEndpoint>,
    /// Endpoints that were found but cannot be published alongside the
    /// selected one, because only one external endpoint is expressible.
    pub unpublishable: Vec<DiscoveredEndpoint>,
}

/// Decide what to do given the probe results and the `openai-endpoint` entry
/// already in the config (`None` when the plugin has no entry at all).
///
/// Selection prefers the endpoint serving the most models, breaking ties by
/// lowest port so the choice is deterministic. Any pre-existing entry — a
/// configured URL, or an entry the operator disabled — wins over publishing.
pub fn plan_endpoint_publish(
    found: Vec<DiscoveredEndpoint>,
    existing: Option<&ExistingEndpointEntry>,
) -> EndpointPublishPlan {
    if found.is_empty() {
        return EndpointPublishPlan {
            status: EndpointPublishStatus::NothingFound,
            found,
            unpublishable: Vec::new(),
        };
    }

    if found.iter().all(|endpoint| endpoint.models.is_empty()) {
        return EndpointPublishPlan {
            status: EndpointPublishStatus::NoModelsServed,
            found,
            unpublishable: Vec::new(),
        };
    }

    let selected_index = found
        .iter()
        .enumerate()
        .max_by_key(|(_, endpoint)| (endpoint.models.len(), std::cmp::Reverse(endpoint.port)))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let selected = found[selected_index].clone();

    if let Some(existing) = existing
        && let Some(status) = existing_entry_status(existing, &found)
    {
        return EndpointPublishPlan {
            status,
            found,
            unpublishable: Vec::new(),
        };
    }

    let unpublishable = found
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected_index)
        .map(|(_, endpoint)| endpoint.clone())
        .collect();
    EndpointPublishPlan {
        status: EndpointPublishStatus::Publish { selected },
        found,
        unpublishable,
    }
}

/// Decide whether a pre-existing entry blocks publishing. `None` means the
/// entry carries no operator intent (present but empty and enabled), so
/// publishing may proceed.
fn existing_entry_status(
    existing: &ExistingEndpointEntry,
    found: &[DiscoveredEndpoint],
) -> Option<EndpointPublishStatus> {
    if !existing.enabled {
        return Some(EndpointPublishStatus::DisabledByOperator);
    }
    let configured = existing
        .url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())?;
    Some(
        if found
            .iter()
            .any(|endpoint| urls_match(&endpoint.base_url, configured))
        {
            EndpointPublishStatus::AlreadyConfigured {
                base_url: configured.to_string(),
            }
        } else {
            EndpointPublishStatus::ConfiguredElsewhere {
                configured: configured.to_string(),
            }
        },
    )
}

fn open_store(config_path: Option<&Path>) -> Result<ConfigStore> {
    match config_path {
        Some(path) => Ok(ConfigStore::open(path)),
        None => ConfigStore::default_path(),
    }
}

fn urls_match(left: &str, right: &str) -> bool {
    left.trim_end_matches('/') == right.trim_end_matches('/')
}

/// `mesh-llm plugins discover`: probe, report, and optionally write the config
/// entry. Writing requires `apply`; without it this only prints findings.
pub async fn run_discover_endpoints(config_path: Option<&Path>, apply: bool) -> Result<()> {
    let store = open_store(config_path)?;
    // A config we cannot read must abort rather than be treated as "nothing
    // configured", which would publish over the operator's own settings.
    let existing = existing_endpoint_entry(&store).with_context(|| {
        format!(
            "cannot read existing plugin configuration from {}",
            store.path().display()
        )
    })?;
    let found = discover_local_endpoints().await;
    let plan = plan_endpoint_publish(found, existing.as_ref());

    for endpoint in &plan.found {
        eprintln!("{} Found {}", style_ok("✓"), endpoint.describe());
        for model in &endpoint.models {
            eprintln!("    {}", style_muted(model));
        }
    }

    match &plan.status {
        EndpointPublishStatus::NothingFound => {
            eprintln!(
                "{}",
                style_muted(
                    "No local OpenAI-compatible server found on the well-known loopback ports."
                )
            );
            return Ok(());
        }
        EndpointPublishStatus::AlreadyConfigured { base_url } => {
            eprintln!(
                "{} {OPENAI_ENDPOINT_PLUGIN} already publishes {base_url}",
                style_ok("✓")
            );
            return Ok(());
        }
        EndpointPublishStatus::ConfiguredElsewhere { configured } => {
            eprintln!(
                "{} {OPENAI_ENDPOINT_PLUGIN} is configured for {configured}; leaving it alone",
                style_warn("!")
            );
            return Ok(());
        }
        EndpointPublishStatus::NoModelsServed => {
            eprintln!(
                "{} No model is loaded on those servers yet; nothing to publish",
                style_warn("!")
            );
            return Ok(());
        }
        EndpointPublishStatus::DisabledByOperator => {
            eprintln!(
                "{} {OPENAI_ENDPOINT_PLUGIN} is disabled in your config; leaving it alone",
                style_warn("!")
            );
            eprintln!("  Re-enable it with: mesh-llm plugins enable {OPENAI_ENDPOINT_PLUGIN}");
            return Ok(());
        }
        EndpointPublishStatus::Publish { selected } => {
            for endpoint in &plan.unpublishable {
                eprintln!(
                    "{} {} cannot be published at the same time: mesh-llm supports one {OPENAI_ENDPOINT_PLUGIN} entry",
                    style_warn("!"),
                    endpoint.describe()
                );
            }
            if !apply {
                eprintln!();
                eprintln!(
                    "Run `mesh-llm plugins discover --apply` to publish {} to the mesh.",
                    selected.describe()
                );
                return Ok(());
            }
            publish_endpoint(&store, selected)?;
            eprintln!(
                "{} Publishing {} via {OPENAI_ENDPOINT_PLUGIN}",
                style_ok("✓"),
                selected.describe()
            );
            eprintln!(
                "  Install the plugin if you have not already: mesh-llm plugins install {OPENAI_ENDPOINT_PLUGIN}"
            );
            eprintln!("  Then restart mesh-llm; its models join your /v1/models list.");
        }
    }
    Ok(())
}

/// Read the existing `openai-endpoint` entry from the config at `config_path`
/// (or the default location). Errors on an unreadable or invalid config.
pub fn existing_endpoint_entry_at(
    config_path: Option<&Path>,
) -> Result<Option<ExistingEndpointEntry>> {
    existing_endpoint_entry(&open_store(config_path)?)
}

/// Read the existing `openai-endpoint` entry. Errors on an unreadable or
/// invalid config so callers can refuse to write.
fn existing_endpoint_entry(store: &ConfigStore) -> Result<Option<ExistingEndpointEntry>> {
    let config = store.load()?;
    Ok(config
        .plugins
        .iter()
        .find(|entry| entry.name == OPENAI_ENDPOINT_PLUGIN)
        .map(|entry| ExistingEndpointEntry {
            url: entry.url.clone(),
            // An absent `enabled` key means enabled, matching `resolve_plugins`.
            enabled: entry.enabled.unwrap_or(true),
        }))
}

/// Point the single `openai-endpoint` entry at this endpoint. Delegates to
/// `ConfigStore`, which preserves the operator's formatting, comments, and any
/// `startup.optional` value they already chose.
fn publish_endpoint(store: &ConfigStore, endpoint: &DiscoveredEndpoint) -> Result<()> {
    store
        .upsert_plugin_url(OPENAI_ENDPOINT_PLUGIN, &endpoint.base_url)
        .with_context(|| {
            format!(
                "failed to write {OPENAI_ENDPOINT_PLUGIN} entry to {}",
                store.path().display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(product: &str, port: u16, models: &[&str]) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            product: product.to_string(),
            port,
            base_url: format!("http://127.0.0.1:{port}/v1"),
            models: models.iter().map(|model| (*model).to_string()).collect(),
        }
    }

    fn configured(url: &str) -> ExistingEndpointEntry {
        ExistingEndpointEntry {
            url: Some(url.to_string()),
            enabled: true,
        }
    }

    #[test]
    fn nothing_found_reports_nothing_found() {
        let plan = plan_endpoint_publish(Vec::new(), None);

        assert_eq!(plan.status, EndpointPublishStatus::NothingFound);
    }

    #[test]
    fn selects_the_endpoint_serving_the_most_models() {
        let lm_studio = endpoint("LM Studio", 1234, &["a"]);
        let ollama = endpoint("Ollama", 11434, &["a", "b", "c"]);

        let plan = plan_endpoint_publish(vec![lm_studio.clone(), ollama.clone()], None);

        assert_eq!(
            plan.status,
            EndpointPublishStatus::Publish { selected: ollama }
        );
        assert_eq!(plan.unpublishable, vec![lm_studio]);
    }

    #[test]
    fn ties_break_to_the_lowest_port_deterministically() {
        let low = endpoint("LM Studio", 1234, &["a"]);
        let high = endpoint("Ollama", 11434, &["b"]);

        let plan = plan_endpoint_publish(vec![low.clone(), high], None);

        assert_eq!(
            plan.status,
            EndpointPublishStatus::Publish { selected: low }
        );
    }

    #[test]
    fn existing_matching_config_is_left_alone() {
        let ollama = endpoint("Ollama", 11434, &["a"]);

        let plan = plan_endpoint_publish(
            vec![ollama],
            Some(&configured("http://127.0.0.1:11434/v1/")),
        );

        assert_eq!(
            plan.status,
            EndpointPublishStatus::AlreadyConfigured {
                base_url: "http://127.0.0.1:11434/v1/".to_string()
            }
        );
    }

    #[test]
    fn existing_unrelated_config_is_never_overwritten() {
        let ollama = endpoint("Ollama", 11434, &["a"]);

        let plan = plan_endpoint_publish(vec![ollama], Some(&configured("http://gpu-box:8000/v1")));

        assert_eq!(
            plan.status,
            EndpointPublishStatus::ConfiguredElsewhere {
                configured: "http://gpu-box:8000/v1".to_string()
            }
        );
        assert!(
            plan.unpublishable.is_empty(),
            "no publish is planned, so nothing is displaced"
        );
    }

    #[test]
    fn publish_writes_an_enabled_optional_plugin_entry() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");
        std::fs::write(&path, "").expect("seed config");
        let store = ConfigStore::open(&path);

        publish_endpoint(&store, &endpoint("Ollama", 11434, &["a"])).expect("publish");

        let config = store.load().expect("reload config");
        let entry = config
            .plugins
            .iter()
            .find(|entry| entry.name == OPENAI_ENDPOINT_PLUGIN)
            .expect("plugin entry written");
        assert_eq!(entry.enabled, Some(true));
        assert_eq!(entry.url.as_deref(), Some("http://127.0.0.1:11434/v1"));
        assert!(
            entry.startup.optional,
            "a discovered endpoint must not block startup when the server goes away"
        );
    }

    #[test]
    fn publish_is_idempotent_and_updates_the_url_in_place() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");
        std::fs::write(&path, "").expect("seed config");
        let store = ConfigStore::open(&path);

        publish_endpoint(&store, &endpoint("Ollama", 11434, &["a"])).expect("first publish");
        publish_endpoint(&store, &endpoint("LM Studio", 1234, &["a"])).expect("second publish");

        let config = store.load().expect("reload config");
        let entries = config
            .plugins
            .iter()
            .filter(|entry| entry.name == OPENAI_ENDPOINT_PLUGIN)
            .collect::<Vec<_>>();
        assert_eq!(
            entries.len(),
            1,
            "a duplicate entry would make resolve_plugins reject the whole config"
        );
        assert_eq!(entries[0].url.as_deref(), Some("http://127.0.0.1:1234/v1"));
    }
}

#[cfg(test)]
mod preservation_tests {
    use super::*;

    fn endpoint(port: u16) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            product: "Ollama".into(),
            port,
            base_url: format!("http://127.0.0.1:{port}/v1"),
            models: vec!["llama3:8b".into()],
        }
    }

    #[test]
    fn publish_preserves_comments_and_unrelated_keys_verbatim() {
        let original = "# my hand-written mesh config\n\
                        [runtime]\n\
                        # keep this comment\n\
                        listen_all = true\n\
                        \n\
                        [[models]]\n\
                        model = \"Qwen3-8B-Q4_K_M\"\n";
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");
        std::fs::write(&path, original).expect("seed config");
        let store = ConfigStore::open(&path);

        publish_endpoint(&store, &endpoint(11434)).expect("publish");

        let written = std::fs::read_to_string(&path).expect("read back");
        assert!(
            written.starts_with(original),
            "existing bytes must be untouched; got:\n{written}"
        );
        assert!(written.contains("# keep this comment"));
        assert!(written.contains("name = \"openai-endpoint\""));
        assert!(written.contains("url = \"http://127.0.0.1:11434/v1\""));
    }

    #[test]
    fn publish_leaves_an_unrelated_plugin_block_alone() {
        let original = "[[plugin]]\nname = \"metrics\"\nenabled = true\n";
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");
        std::fs::write(&path, original).expect("seed config");
        let store = ConfigStore::open(&path);

        publish_endpoint(&store, &endpoint(11434)).expect("publish");

        let config = store.load().expect("reload");
        let names = config
            .plugins
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["metrics", OPENAI_ENDPOINT_PLUGIN]);
    }
}

#[cfg(test)]
mod existing_entry_tests {
    use super::*;

    fn ollama() -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            product: "Ollama".into(),
            port: 11434,
            base_url: "http://127.0.0.1:11434/v1".into(),
            models: vec!["llama3:8b".into()],
        }
    }

    fn seeded_store(contents: &str) -> (tempfile::TempDir, ConfigStore) {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");
        std::fs::write(&path, contents).expect("seed config");
        let store = ConfigStore::open(&path);
        (temp, store)
    }

    #[test]
    fn a_disabled_entry_is_an_opt_out_not_an_empty_slot() {
        let existing = ExistingEndpointEntry {
            url: None,
            enabled: false,
        };

        let plan = plan_endpoint_publish(vec![ollama()], Some(&existing));

        assert_eq!(plan.status, EndpointPublishStatus::DisabledByOperator);
    }

    #[test]
    fn a_disabled_entry_with_a_url_is_still_an_opt_out() {
        let existing = ExistingEndpointEntry {
            url: Some("http://127.0.0.1:11434/v1".into()),
            enabled: false,
        };

        let plan = plan_endpoint_publish(vec![ollama()], Some(&existing));

        assert_eq!(plan.status, EndpointPublishStatus::DisabledByOperator);
    }

    #[test]
    fn absent_enabled_key_reads_as_enabled_like_resolve_plugins_does() {
        let (_temp, store) =
            seeded_store("[[plugin]]\nname = \"openai-endpoint\"\nurl = \"http://x:1/v1\"\n");

        let existing = existing_endpoint_entry(&store)
            .expect("read entry")
            .expect("entry present");

        assert!(existing.enabled);
        assert_eq!(existing.url.as_deref(), Some("http://x:1/v1"));
    }

    #[test]
    fn disabled_entry_is_read_off_disk_as_disabled() {
        let (_temp, store) =
            seeded_store("[[plugin]]\nname = \"openai-endpoint\"\nenabled = false\n");

        let existing = existing_endpoint_entry(&store)
            .expect("read entry")
            .expect("entry present");

        assert!(!existing.enabled);
    }

    #[test]
    fn an_unreadable_config_is_an_error_not_an_empty_config() {
        let (_temp, store) = seeded_store("this is not = = valid toml\n");

        let error = existing_endpoint_entry(&store)
            .expect_err("invalid TOML must not read as 'nothing configured'");

        assert!(
            !format!("{error:#}").is_empty(),
            "the failure must surface so callers refuse to overwrite"
        );
    }

    #[test]
    fn publish_does_not_flip_an_explicit_startup_optional_false() {
        let (_temp, store) = seeded_store(
            "[[plugin]]\nname = \"openai-endpoint\"\nenabled = true\n\n[plugin.startup]\noptional = false\n",
        );

        publish_endpoint(&store, &ollama()).expect("publish");

        let config = store.load().expect("reload");
        let entry = config
            .plugins
            .iter()
            .find(|entry| entry.name == OPENAI_ENDPOINT_PLUGIN)
            .expect("entry");
        assert!(
            !entry.startup.optional,
            "an operator's explicit startup.optional = false must survive"
        );
        assert_eq!(entry.url.as_deref(), Some("http://127.0.0.1:11434/v1"));
    }

    #[test]
    fn publish_errors_rather_than_panics_on_a_non_table_startup_value() {
        let (_temp, store) = seeded_store("[[plugin]]\nname = \"openai-endpoint\"\nstartup = 5\n");

        let error = publish_endpoint(&store, &ollama())
            .expect_err("a malformed startup value must be an error, not a panic");

        assert!(format!("{error:#}").contains("startup"), "got: {error:#}");
    }

    #[test]
    fn already_configured_wins_even_when_a_richer_server_appears() {
        let big = DiscoveredEndpoint {
            product: "LM Studio".into(),
            port: 1234,
            base_url: "http://127.0.0.1:1234/v1".into(),
            models: vec!["a".into(), "b".into(), "c".into()],
        };
        let existing = ExistingEndpointEntry {
            url: Some("http://127.0.0.1:11434/v1".into()),
            enabled: true,
        };

        let plan = plan_endpoint_publish(vec![big, ollama()], Some(&existing));

        assert_eq!(
            plan.status,
            EndpointPublishStatus::AlreadyConfigured {
                base_url: "http://127.0.0.1:11434/v1".to_string()
            },
            "an existing choice must not be silently upgraded"
        );
    }
}

#[cfg(test)]
mod empty_server_tests {
    use super::*;

    #[test]
    fn a_server_with_no_models_is_not_published() {
        let empty = DiscoveredEndpoint {
            product: "Ollama".into(),
            port: 11434,
            base_url: "http://127.0.0.1:11434/v1".into(),
            models: Vec::new(),
        };

        let plan = plan_endpoint_publish(vec![empty], None);

        assert_eq!(
            plan.status,
            EndpointPublishStatus::NoModelsServed,
            "publishing a provider that serves nothing would advertise phantom capacity"
        );
    }

    #[test]
    fn one_empty_server_does_not_block_a_populated_one() {
        let empty = DiscoveredEndpoint {
            product: "LM Studio".into(),
            port: 1234,
            base_url: "http://127.0.0.1:1234/v1".into(),
            models: Vec::new(),
        };
        let populated = DiscoveredEndpoint {
            product: "Ollama".into(),
            port: 11434,
            base_url: "http://127.0.0.1:11434/v1".into(),
            models: vec!["llama3:8b".into()],
        };

        let plan = plan_endpoint_publish(vec![empty, populated.clone()], None);

        assert_eq!(
            plan.status,
            EndpointPublishStatus::Publish {
                selected: populated
            }
        );
    }
}
