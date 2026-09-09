use super::{PluginManager, proto};
use anyhow::{Result, bail};

/// Deliver a host-originated channel message to every loaded plugin that
/// declares `channel` in its manifest — the mirror of
/// [`PluginManager::broadcast_mesh_event`] for `PluginMeshEvent::Channel`, needed
/// because `PluginManager::dispatch_channel_message` is addressed to one
/// already-known `plugin_id` (its normal caller is a mesh-relayed message
/// that already names its target), not a broadcast from inside the host
/// runtime itself.
impl PluginManager {
    pub async fn broadcast_channel_message(
        &self,
        channel: &str,
        content_type: &str,
        body: Vec<u8>,
        correlation_id: &str,
    ) -> Result<()> {
        let mut results = Vec::new();
        for (plugin_id, plugin) in &self.inner.plugins {
            if !self.plugin_declares_mesh_channel(plugin_id, channel).await {
                continue;
            }
            let result = plugin
                .try_send_channel_message(proto::ChannelMessage {
                    channel: channel.to_string(),
                    source_peer_id: String::new(),
                    target_peer_id: plugin_id.clone(),
                    content_type: content_type.to_string(),
                    body: body.clone(),
                    message_kind: String::new(),
                    correlation_id: correlation_id.to_string(),
                    metadata_json: String::new(),
                })
                .await;
            results.push((plugin_id.clone(), result));
        }
        aggregate_broadcast_results(results)
    }

    /// Whether an already-running `plugin_name` declares `channel` in its
    /// manifest. Gates delivery paths (`broadcast_channel_message`,
    /// `dispatch_channel_message`, `dispatch_bulk_transfer_message`) that
    /// must never lazily start a stopped plugin just to ask it — see
    /// [`PluginManager::manifest_snapshot`].
    pub async fn plugin_declares_mesh_channel(&self, plugin_name: &str, channel: &str) -> bool {
        self.manifest_snapshot(plugin_name)
            .await
            .is_some_and(|manifest| manifest_declares_mesh_channel(&manifest, channel))
    }
}

/// Turn per-plugin delivery outcomes from [`PluginManager::broadcast_channel_message`]
/// into one aggregate result. A single plugin rejecting or losing its channel
/// must never suppress delivery to the other declaring plugins — this only
/// summarizes failures after every recipient has already been attempted.
fn aggregate_broadcast_results(results: Vec<(String, Result<()>)>) -> Result<()> {
    let attempted = results.len();
    let failures: Vec<String> = results
        .into_iter()
        .filter_map(|(plugin_id, result)| result.err().map(|error| format!("{plugin_id}: {error}")))
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        bail!(
            "failed to deliver channel message to {} of {} declaring plugin(s): {}",
            failures.len(),
            attempted,
            failures.join("; ")
        )
    }
}

fn manifest_declares_mesh_channel(manifest: &proto::PluginManifest, channel: &str) -> bool {
    manifest
        .mesh_channels
        .iter()
        .any(|entry| entry.name == channel)
}

#[cfg(test)]
mod tests {
    use super::super::{PluginHostMode, ResolvedPlugins};
    use super::*;
    use mesh_llm_plugin::MeshVisibility;
    use tokio::sync::mpsc;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    #[test]
    fn manifest_declares_mesh_channel_matches_exact_name_only() {
        let manifest = proto::PluginManifest {
            mesh_channels: vec![proto::MeshChannelManifest {
                name: "openai.exchange.v1".into(),
            }],
            ..proto::PluginManifest::default()
        };

        assert!(manifest_declares_mesh_channel(
            &manifest,
            "openai.exchange.v1"
        ));
        assert!(!manifest_declares_mesh_channel(&manifest, "other.channel"));
    }

    /// One declaring plugin failing must not stop delivery to the rest —
    /// every recipient's result is collected before an aggregate error names
    /// only the ones that actually failed. Regression for the coderbot
    /// finding: the old loop used `?` on each send and returned on the first
    /// failure, silently skipping later plugins in the `BTreeMap`.
    #[test]
    fn aggregate_broadcast_results_reports_every_failure_without_dropping_successes() {
        let results = vec![
            ("alpha".to_string(), Ok(())),
            ("beta".to_string(), Err(anyhow::anyhow!("channel closed"))),
            ("gamma".to_string(), Err(anyhow::anyhow!("send timed out"))),
            ("delta".to_string(), Ok(())),
        ];

        let error = aggregate_broadcast_results(results)
            .expect_err("some plugins failed")
            .to_string();

        assert!(error.contains("2 of 4"), "error was: {error}");
        assert!(error.contains("beta: channel closed"), "error was: {error}");
        assert!(
            error.contains("gamma: send timed out"),
            "error was: {error}"
        );
        assert!(!error.contains("alpha"), "error was: {error}");
        assert!(!error.contains("delta"), "error was: {error}");
    }

    #[test]
    fn aggregate_broadcast_results_is_ok_when_every_plugin_succeeds() {
        let results = vec![("alpha".to_string(), Ok(())), ("beta".to_string(), Ok(()))];

        assert!(aggregate_broadcast_results(results).is_ok());
    }

    #[tokio::test]
    async fn broadcast_channel_message_is_a_no_op_with_no_plugins_loaded() {
        let specs = ResolvedPlugins {
            shared_endpoint: None,
            externals: Vec::new(),
            inactive: Vec::new(),
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
            .await
            .expect("empty plugin set starts cleanly");

        manager
            .broadcast_channel_message(
                "openai.exchange.v1",
                "application/json",
                b"{}".to_vec(),
                "corr-1",
            )
            .await
            .expect("broadcasting with no plugins loaded is a no-op, not an error");

        manager.shutdown().await;
    }
}
