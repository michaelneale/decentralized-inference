use super::shared_endpoint::{SHARED_ENDPOINT_ID, SHARED_ENDPOINT_NAME};
use super::{
    InferenceEndpointRoute, PluginCapabilityProvider, PluginEndpointSummary, PluginManager,
    PluginSummary, endpoint_kind_name, endpoint_transport_kind_name, plugin_manifest_overview,
    proto,
};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use url::Url;

pub(super) const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;
const ENDPOINT_STARTUP_GRACE_SECS: u64 = 30;
const ENDPOINT_FAILURE_THRESHOLD: u32 = 2;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct EndpointHealthRecord {
    pub(super) state: String,
    pub(super) available: bool,
    pub(super) detail: Option<String>,
    pub(super) models: Vec<String>,
}

#[derive(Clone, Debug)]
pub(super) struct EndpointHealthState {
    record: EndpointHealthRecord,
    first_checked_at: Instant,
    consecutive_failures: u32,
}

impl PluginManager {
    pub(super) async fn refresh_plugin_endpoints(&self, plugin_name: &str) -> Result<()> {
        let summary = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.summary().await
        } else if let Some(summary) = self.inner.inactive.get(plugin_name) {
            summary.clone()
        } else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };
        self.publish_plugin_summary(&summary);

        let manifest = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.manifest_snapshot().await
        } else {
            self.manifest(plugin_name).await.ok().flatten()
        };
        let Some(manifest) = manifest else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            self.publish_plugin_summary(&summary);
            self.publish_plugin_providers(plugin_name, Vec::new());
            return Ok(());
        };

        let now = Instant::now();
        let prefix = format!("{plugin_name}:");
        let previous = self
            .inner
            .endpoint_health
            .lock()
            .await
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|endpoint_id| (endpoint_id.to_string(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let plugin_default = endpoint_record_from_plugin_status(&summary);
        let mut providers = manifest
            .capabilities
            .iter()
            .map(|capability| PluginCapabilityProvider {
                capability: capability.clone(),
                plugin_name: summary.name.clone(),
                plugin_status: summary.status.clone(),
                endpoint_id: None,
                available: plugin_default.available,
                detail: plugin_default.detail.clone(),
            })
            .collect::<Vec<_>>();
        let mut endpoint_states = BTreeMap::new();
        let mut endpoint_summaries = Vec::new();
        for endpoint in &manifest.endpoints {
            let key = endpoint.endpoint_id.clone();
            let health =
                endpoint_health_for_summary(&summary, endpoint, previous.get(&key), now).await;
            for capability in endpoint_declared_capabilities(endpoint) {
                providers.push(PluginCapabilityProvider {
                    capability,
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: Some(endpoint.endpoint_id.clone()),
                    available: health.record.available,
                    detail: health.record.detail.clone(),
                });
            }
            endpoint_summaries.push(PluginEndpointSummary {
                plugin_name: summary.name.clone(),
                plugin_status: summary.status.clone(),
                endpoint_id: endpoint.endpoint_id.clone(),
                state: health.record.state.clone(),
                available: health.record.available,
                kind: endpoint_kind_name(endpoint.kind).to_string(),
                transport_kind: endpoint_transport_kind_name(endpoint.transport_kind).to_string(),
                protocol: endpoint.protocol.clone(),
                address: endpoint.address.clone(),
                args: endpoint.args.clone(),
                namespace: endpoint.namespace.clone(),
                supports_streaming: endpoint.supports_streaming,
                managed_by_plugin: endpoint.managed_by_plugin,
                detail: health.record.detail.clone(),
                models: health.record.models.clone(),
            });
            endpoint_states.insert(endpoint_key(plugin_name, &key), health);
        }

        self.clear_plugin_endpoint_health(plugin_name).await;
        self.publish_plugin_summary(&summary);
        self.publish_plugin_manifest(plugin_name, Some(plugin_manifest_overview(&manifest)));
        self.publish_plugin_providers(plugin_name, providers);
        for endpoint_summary in endpoint_summaries {
            self.plugin_endpoint_producer(plugin_name, &endpoint_summary.endpoint_id)
                .publish_plugin_endpoint(endpoint_summary);
        }

        let mut registry = self.inner.endpoint_health.lock().await;
        registry.extend(endpoint_states);
        Ok(())
    }

    async fn clear_plugin_endpoint_health(&self, plugin_name: &str) {
        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
        drop(registry);
        self.plugin_summary_producer(plugin_name)
            .clear_plugin_reports(plugin_name);
    }

    pub async fn inference_endpoints(&self) -> Result<Vec<InferenceEndpointRoute>> {
        let mut endpoints = self.plugin_inference_endpoints().await?;
        endpoints.extend(self.shared_endpoint_inference_route().await);
        Ok(endpoints)
    }

    /// The operator-shared upstream as a route, when its health record says it
    /// is currently available.
    ///
    /// This is a process-free source in the same inventory: it is not a plugin,
    /// does not appear in the plugin list, and its address never leaves this
    /// node. Only the model names it reports are advertised to peers.
    async fn shared_endpoint_inference_route(&self) -> Option<InferenceEndpointRoute> {
        let address = self.inner.shared_endpoint.clone()?;
        let record = self
            .inner
            .endpoint_health
            .lock()
            .await
            .get(&endpoint_key(SHARED_ENDPOINT_NAME, SHARED_ENDPOINT_ID))
            .map(|state| state.record.clone())?;
        record.available.then(|| InferenceEndpointRoute {
            // The upstream is the operator's own server, reached over loopback
            // or their LAN. A caller's bearer token authenticates them to this
            // node and must not be replayed onward to it.
            strip_caller_credentials: true,
            plugin_name: SHARED_ENDPOINT_NAME.into(),
            endpoint_id: SHARED_ENDPOINT_ID.into(),
            address,
            models: record.models,
        })
    }

    /// Probe the shared upstream once at startup, seeding its health record.
    ///
    /// Startup failure must be loud: `mesh-llm share <url>` against a server
    /// that is not running should report that, not join the mesh advertising
    /// nothing. Later probes go through the ordinary supervisor tick, where the
    /// same grace/threshold policy as plugin endpoints applies.
    pub(crate) async fn start_shared_endpoint(&self) -> Result<Vec<String>> {
        let address = self
            .inner
            .shared_endpoint
            .clone()
            .context("no shared endpoint was configured for this node")?;
        let record = probe_models_endpoint(&address, ProbePolicy::Strict).await;
        anyhow::ensure!(
            record.available,
            "cannot reach the upstream models API at {address}{}",
            record
                .detail
                .as_deref()
                .map(|detail| format!(": {detail}"))
                .unwrap_or_default()
        );
        // An empty `data` array is valid OpenAI-compatible data (the probe
        // layer accepts it, and on refresh it is a *successful* probe that
        // simply withdraws the models). At startup it is still useless: the
        // node would join advertising nothing, which is the same unhelpful
        // outcome the unreachable-upstream check exists to prevent.
        anyhow::ensure!(
            !record.models.is_empty(),
            "the upstream at {address} reports no models; \
             load a model on it (for example `ollama run <model>`) and retry"
        );
        let models = record.models.clone();
        self.inner.endpoint_health.lock().await.insert(
            endpoint_key(SHARED_ENDPOINT_NAME, SHARED_ENDPOINT_ID),
            EndpointHealthState {
                record,
                first_checked_at: Instant::now(),
                consecutive_failures: 0,
            },
        );
        Ok(models)
    }

    /// Re-probe the shared upstream on the supervisor tick.
    ///
    /// Reuses `apply_endpoint_probe`, so an upstream that goes away is
    /// tolerated for one tick as `degraded` and only then withdrawn, and a
    /// recovered upstream is re-advertised — identical to a plugin endpoint.
    pub(super) async fn refresh_shared_endpoint(&self) {
        let Some(address) = self.inner.shared_endpoint.clone() else {
            return;
        };
        let key = endpoint_key(SHARED_ENDPOINT_NAME, SHARED_ENDPOINT_ID);
        let previous = self.inner.endpoint_health.lock().await.get(&key).cloned();
        let probe = probe_models_endpoint(&address, ProbePolicy::Strict).await;
        let state = apply_endpoint_probe(previous.as_ref(), probe, Instant::now());
        self.inner.endpoint_health.lock().await.insert(key, state);
    }

    async fn plugin_inference_endpoints(&self) -> Result<Vec<InferenceEndpointRoute>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_inference_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let endpoint_summaries = self.endpoints().await?;
        let mut endpoints = Vec::new();
        for endpoint in endpoint_summaries {
            if endpoint.kind != "inference" || !endpoint.available {
                continue;
            }
            let Some(address) = endpoint.address.clone() else {
                continue;
            };
            endpoints.push(InferenceEndpointRoute {
                strip_caller_credentials: false,
                plugin_name: endpoint.plugin_name,
                endpoint_id: endpoint.endpoint_id,
                address,
                models: endpoint.models,
            });
        }
        Ok(endpoints)
    }
}

pub(super) fn endpoint_record_from_plugin_status(summary: &PluginSummary) -> EndpointHealthRecord {
    if !summary.enabled || summary.status == "disabled" {
        return EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        };
    }

    match summary.status.as_str() {
        "running" => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        },
        "starting" | "restarting" => EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        "degraded" => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        _ => EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
    }
}

fn endpoint_state_from_plugin_status(summary: &PluginSummary, now: Instant) -> EndpointHealthState {
    EndpointHealthState {
        record: endpoint_record_from_plugin_status(summary),
        first_checked_at: now,
        consecutive_failures: 0,
    }
}

fn endpoint_key(plugin_name: &str, endpoint_id: &str) -> String {
    format!("{plugin_name}:{endpoint_id}")
}

pub(super) fn endpoint_declared_capabilities(endpoint: &proto::EndpointManifest) -> Vec<String> {
    match proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => {
            let mut capabilities = vec!["endpoint:inference".into()];
            if let Some(protocol) = endpoint.protocol.as_deref() {
                capabilities.push(format!("endpoint:inference/{protocol}"));
            }
            capabilities
        }
        proto::EndpointKind::Mcp => {
            let mut capabilities = vec!["endpoint:mcp".into()];
            if let Some(namespace) = endpoint.namespace.as_deref() {
                capabilities.push(format!("endpoint:mcp/{namespace}"));
            }
            capabilities
        }
        proto::EndpointKind::Unspecified => Vec::new(),
    }
}

async fn endpoint_health_for_summary(
    summary: &PluginSummary,
    endpoint: &proto::EndpointManifest,
    previous: Option<&EndpointHealthState>,
    now: Instant,
) -> EndpointHealthState {
    if summary.status != "running" {
        return endpoint_state_from_plugin_status(summary, now);
    }

    let probe = probe_endpoint(endpoint)
        .await
        .unwrap_or(EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        });
    apply_endpoint_probe(previous, probe, now)
}

fn apply_endpoint_probe(
    previous: Option<&EndpointHealthState>,
    probe: EndpointHealthRecord,
    now: Instant,
) -> EndpointHealthState {
    let first_checked_at = previous.map(|state| state.first_checked_at).unwrap_or(now);

    if probe.available {
        return EndpointHealthState {
            record: probe,
            first_checked_at,
            consecutive_failures: 0,
        };
    }

    let failure_streak = previous
        .map(|state| state.consecutive_failures.saturating_add(1))
        .unwrap_or(1);
    let within_startup_grace =
        now.duration_since(first_checked_at) < Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS);
    let was_available = previous
        .map(|state| state.record.available)
        .unwrap_or(false);

    let record = if !was_available && within_startup_grace {
        EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else if was_available && failure_streak < ENDPOINT_FAILURE_THRESHOLD {
        EndpointHealthRecord {
            state: "degraded".into(),
            available: true,
            detail: probe.detail,
            // A tolerated tick must stay *routable*: dropping the model list
            // here leaves `available = true` but removes every model from
            // `inference_models` / `inference_endpoint_for_model`, so the route
            // survives while nothing can be dispatched to it. Carry the last
            // known-good inventory until the failure threshold withdraws the
            // endpoint outright.
            models: previous
                .map(|state| state.record.models.clone())
                .unwrap_or_default(),
        }
    } else {
        EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    };

    EndpointHealthState {
        record,
        first_checked_at,
        consecutive_failures: failure_streak,
    }
}

async fn probe_endpoint(endpoint: &proto::EndpointManifest) -> Option<EndpointHealthRecord> {
    match (
        proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified),
        proto::EndpointTransportKind::try_from(endpoint.transport_kind)
            .unwrap_or(proto::EndpointTransportKind::Unspecified),
    ) {
        (proto::EndpointKind::Inference, proto::EndpointTransportKind::EndpointTransportHttp) => {
            let protocol = endpoint.protocol.as_deref().unwrap_or_default();
            if protocol.eq_ignore_ascii_case("openai_compatible") {
                return Some(
                    probe_openai_compatible_http_endpoint(endpoint.address.as_deref()?).await,
                );
            }
            None
        }
        _ => None,
    }
}

/// How strictly a models probe treats the transport and the response body.
///
/// Configured plugins keep the long-standing lenient policy: any 2xx is
/// healthy, and an unparseable body yields an empty model list. The shared
/// upstream in `mesh-llm share` is operator-supplied and unvetted, so it gets
/// the strict policy — see [`ProbePolicy::Strict`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProbePolicy {
    /// Existing configured-plugin behaviour, unchanged.
    Lenient,
    /// Sharing policy: the client follows no redirects and ignores environment
    /// proxies, so what the probe validated is the same server inference will
    /// connect to; the body is read with a bound; and a 2xx whose body is not
    /// an OpenAI model list is a failed probe rather than an empty inventory.
    Strict,
}

/// Largest models response the strict policy will read.
const STRICT_PROBE_BODY_LIMIT_BYTES: usize = 1 << 20;

async fn probe_openai_compatible_http_endpoint(address: &str) -> EndpointHealthRecord {
    probe_models_endpoint(address, ProbePolicy::Lenient).await
}

pub(super) async fn probe_models_endpoint(
    address: &str,
    policy: ProbePolicy,
) -> EndpointHealthRecord {
    let models_url = match endpoint_models_url(address) {
        Some(url) => url,
        None => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("invalid endpoint address '{address}'")),
                models: Vec::new(),
            };
        }
    };

    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(3));
    if policy == ProbePolicy::Strict {
        // A redirect could validate a different server (or an HTTPS one) while
        // inference still connects to the original HTTP address, and an
        // environment proxy could do the same silently.
        builder = builder
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy();
    }
    let client = match builder.build() {
        Ok(client) => client,
        Err(err) => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("build health probe client: {err}")),
                models: Vec::new(),
            };
        }
    };

    match client.get(models_url.clone()).send().await {
        Ok(response) if response.status().is_success() => {
            let status = response.status();
            match (policy, parse_models_response(response, policy).await) {
                (_, Ok(models)) => EndpointHealthRecord {
                    state: "healthy".into(),
                    available: true,
                    detail: Some(format!("GET {models_url} -> {status}")),
                    models,
                },
                (ProbePolicy::Lenient, Err(_)) => EndpointHealthRecord {
                    state: "healthy".into(),
                    available: true,
                    detail: Some(format!("GET {models_url} -> {status}")),
                    models: Vec::new(),
                },
                // A 200 that is not an OpenAI model list means this is not an
                // OpenAI-compatible server. Treat it as a failed probe so
                // startup refuses it and a refresh does not count it as a
                // recovery.
                (ProbePolicy::Strict, Err(err)) => EndpointHealthRecord {
                    state: "unhealthy".into(),
                    available: false,
                    detail: Some(format!(
                        "GET {models_url} -> {status} but the response is not an \
                         OpenAI-compatible model list: {err}"
                    )),
                    models: Vec::new(),
                },
            }
        }
        Ok(response) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: Vec::new(),
        },
        Err(err) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} failed: {}", models_url, err)),
            models: Vec::new(),
        },
    }
}

async fn parse_models_response(
    response: reqwest::Response,
    policy: ProbePolicy,
) -> Result<Vec<String>> {
    let body = match policy {
        ProbePolicy::Lenient => response.json::<Value>().await?,
        ProbePolicy::Strict => {
            let bytes = read_bounded_body(response, STRICT_PROBE_BODY_LIMIT_BYTES).await?;
            serde_json::from_slice::<Value>(&bytes).context("response body is not JSON")?
        }
    };
    if policy == ProbePolicy::Strict {
        // An OpenAI models response must have a `data` array. An empty array is
        // a valid (if unhelpful) inventory; anything else is a different API.
        let entries = body
            .get("data")
            .and_then(|value| value.as_array())
            .context("response has no OpenAI `data` array")?;
        anyhow::ensure!(
            entries.iter().all(|entry| entry
                .get("id")
                .and_then(|id| id.as_str())
                .is_some_and(|id| !id.trim().is_empty())),
            "response `data` contains an entry without a model id"
        );
    }
    let models = body
        .get("data")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    Ok(models)
}

/// Read at most `limit` bytes of a response body, failing if it is larger.
///
/// An unvetted upstream must not be able to make the probe allocate without
/// bound.
async fn read_bounded_body(response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    use futures_util::StreamExt;

    let mut collected = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading response body")?;
        anyhow::ensure!(
            collected.len() + chunk.len() <= limit,
            "response body exceeds {limit} bytes"
        );
        collected.extend_from_slice(&chunk);
    }
    Ok(collected)
}

fn endpoint_models_url(address: &str) -> Option<Url> {
    let mut url = Url::parse(address).ok()?;
    let mut path = url.path().trim_end_matches('/').to_string();
    if path.is_empty() {
        path = "/v1".into();
    }
    if !path.ends_with("/models") {
        if path.ends_with("/v1") || path.ends_with("/api/v1") {
            path.push_str("/models");
        } else {
            path.push_str("/v1/models");
        }
    }
    url.set_path(&path);
    url.set_query(None);
    Some(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginWebUiState;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn running_summary() -> PluginSummary {
        PluginSummary {
            name: "demo".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: None,
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: PluginWebUiState::default(),
            startup: None,
            error: None,
        }
    }

    async fn spawn_fake_models_server(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_seen = requests.clone();
        let handle = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await.unwrap();
                requests_seen.fetch_add(1, Ordering::SeqCst);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (format!("http://{addr}/api/v1"), handle, requests)
    }

    #[tokio::test]
    async fn shared_endpoint_startup_publishes_models_as_a_route() {
        let (address, server, _) = spawn_fake_models_server(vec![(
            "200 OK",
            r#"{"data":[{"id":"llama3.2"},{"id":"qwen3"}]}"#,
        )])
        .await;
        let manager = PluginManager::for_test_shared_endpoint(&address);

        let models = manager.start_shared_endpoint().await.unwrap();
        assert_eq!(models, vec!["llama3.2".to_string(), "qwen3".to_string()]);

        let routes = manager.inference_endpoints().await.unwrap();
        assert_eq!(routes.len(), 1, "the upstream is the only inference source");
        assert_eq!(routes[0].plugin_name, SHARED_ENDPOINT_NAME);
        assert_eq!(routes[0].endpoint_id, SHARED_ENDPOINT_ID);
        assert_eq!(routes[0].address, address);
        assert_eq!(routes[0].models, models);
        assert!(
            routes[0].strip_caller_credentials,
            "a caller's token must not be replayed to the operator's own server"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn shared_endpoint_startup_fails_loudly_when_the_upstream_is_unreachable() {
        // Bind then drop, so the port is closed but well-formed.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}/v1", listener.local_addr().unwrap());
        drop(listener);
        let manager = PluginManager::for_test_shared_endpoint(&address);

        let error = manager
            .start_shared_endpoint()
            .await
            .expect_err("sharing a server that is not running must fail startup");
        assert!(error.to_string().contains("cannot reach the upstream"));
        assert!(
            manager.inference_endpoints().await.unwrap().is_empty(),
            "a failed probe must advertise nothing"
        );
    }

    #[tokio::test]
    async fn shared_endpoint_survives_one_failure_then_withdraws_and_recovers() {
        let (address, server, _) = spawn_fake_models_server(vec![
            ("200 OK", r#"{"data":[{"id":"llama3.2"}]}"#),
            ("503 Service Unavailable", "{}"),
            ("503 Service Unavailable", "{}"),
            ("200 OK", r#"{"data":[{"id":"llama3.2"}]}"#),
        ])
        .await;
        let manager = PluginManager::for_test_shared_endpoint(&address);
        manager.start_shared_endpoint().await.unwrap();

        // Same grace/threshold policy as a plugin endpoint: one bad tick is
        // tolerated as degraded, so a brief upstream blip does not flap the
        // model list across the mesh.
        manager.refresh_shared_endpoint().await;
        assert_eq!(
            manager.inference_endpoints().await.unwrap().len(),
            1,
            "a single failed probe must not withdraw the models"
        );

        manager.refresh_shared_endpoint().await;
        assert!(
            manager.inference_endpoints().await.unwrap().is_empty(),
            "a sustained outage must withdraw the models"
        );

        manager.refresh_shared_endpoint().await;
        assert_eq!(
            manager.inference_endpoints().await.unwrap().len(),
            1,
            "recovery must re-advertise the upstream"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn shared_endpoint_stays_routable_through_one_failed_probe() {
        let (address, server, _) = spawn_fake_models_server(vec![
            ("200 OK", r#"{"data":[{"id":"llama3.2"}]}"#),
            ("503 Service Unavailable", "{}"),
            ("503 Service Unavailable", "{}"),
        ])
        .await;
        let manager = PluginManager::for_test_shared_endpoint(&address);
        manager.start_shared_endpoint().await.unwrap();

        manager.refresh_shared_endpoint().await;
        // Route count alone would pass even with an emptied model list, which
        // is exactly the bug: assert the models and the lookup used to dispatch.
        assert_eq!(
            manager.inference_models().await.unwrap(),
            vec!["llama3.2".to_string()],
            "a tolerated failed tick must keep the last known-good models"
        );
        assert!(
            manager
                .inference_endpoint_for_model("llama3.2")
                .await
                .unwrap()
                .is_some(),
            "the degraded route must still resolve the model it advertises"
        );

        manager.refresh_shared_endpoint().await;
        assert!(
            manager.inference_models().await.unwrap().is_empty(),
            "a sustained outage must withdraw the models"
        );
        assert!(
            manager
                .inference_endpoint_for_model("llama3.2")
                .await
                .unwrap()
                .is_none()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn shared_endpoint_startup_rejects_a_200_that_is_not_a_model_list() {
        for body in [
            "<html><body>hello</body></html>",
            r#"{"models":[{"name":"llama3.2"}]}"#,
            r#"{"data":{"id":"llama3.2"}}"#,
            r#"{"data":[{"name":"llama3.2"}]}"#,
            "{",
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let owned = body.to_string();
            let server = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await.unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{owned}",
                    owned.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            });
            let manager = PluginManager::for_test_shared_endpoint(&format!("http://{addr}/v1"));
            let error = manager.start_shared_endpoint().await.unwrap_err();
            assert!(
                error.to_string().contains("cannot reach the upstream"),
                "unexpected error for {body}: {error}"
            );
            assert!(manager.inference_endpoints().await.unwrap().is_empty());
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn shared_endpoint_startup_rejects_a_valid_but_empty_inventory() {
        let (address, server, _) =
            spawn_fake_models_server(vec![("200 OK", r#"{"data":[]}"#)]).await;
        let manager = PluginManager::for_test_shared_endpoint(&address);

        let error = manager.start_shared_endpoint().await.unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("reports no models"),
            "an empty inventory must be rejected with its own actionable error, not \
             the malformed-data error: {message}"
        );
        assert!(
            !message.contains("cannot reach"),
            "valid empty data is distinct from an unreachable or malformed upstream: {message}"
        );
        assert!(
            manager.inference_endpoints().await.unwrap().is_empty(),
            "startup must not join the mesh advertising nothing"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_valid_empty_inventory_on_refresh_withdraws_models_without_a_failed_tick() {
        let (address, server, _) = spawn_fake_models_server(vec![
            ("200 OK", r#"{"data":[{"id":"llama3.2"}]}"#),
            ("200 OK", r#"{"data":[]}"#),
        ])
        .await;
        let manager = PluginManager::for_test_shared_endpoint(&address);
        manager.start_shared_endpoint().await.unwrap();

        manager.refresh_shared_endpoint().await;

        let key = endpoint_key(SHARED_ENDPOINT_NAME, SHARED_ENDPOINT_ID);
        let state = manager
            .inner
            .endpoint_health
            .lock()
            .await
            .get(&key)
            .cloned()
            .expect("the shared endpoint is tracked");
        assert_eq!(
            state.record.state, "healthy",
            "a valid empty inventory is a successful probe, not a failed tick"
        );
        assert_eq!(state.consecutive_failures, 0);
        assert!(state.record.models.is_empty());
        assert!(
            manager.inference_models().await.unwrap().is_empty(),
            "the models are withdrawn while the upstream stays reachable"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn invalid_data_on_refresh_is_a_failed_tick_not_a_recovery() {
        let (address, server, _) = spawn_fake_models_server(vec![
            ("200 OK", r#"{"data":[{"id":"llama3.2"}]}"#),
            ("503 Service Unavailable", "{}"),
            ("200 OK", "<html>not openai</html>"),
        ])
        .await;
        let manager = PluginManager::for_test_shared_endpoint(&address);
        manager.start_shared_endpoint().await.unwrap();

        manager.refresh_shared_endpoint().await;
        manager.refresh_shared_endpoint().await;
        assert!(
            manager.inference_models().await.unwrap().is_empty(),
            "a 200 with invalid data must count as a failure, not a recovery"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_lenient_plugin_probe_still_accepts_an_unparseable_200() {
        // Configured-plugin compatibility: the strict policy is scoped to
        // sharing and must not change this long-standing behaviour.
        let (address, server, _) =
            spawn_fake_models_server(vec![("200 OK", "<html>not openai</html>")]).await;

        let health = probe_models_endpoint(&address, ProbePolicy::Lenient).await;
        assert!(health.available);
        assert_eq!(health.state, "healthy");
        assert!(health.models.is_empty());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn the_strict_probe_does_not_follow_a_redirect_to_another_server() {
        let (valid, valid_server, _) =
            spawn_fake_models_server(vec![("200 OK", r#"{"data":[{"id":"somewhere-else"}]}"#)])
                .await;
        let redirect_target = format!("{valid}/models");

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let redirector = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await.unwrap();
            let response = format!(
                "HTTP/1.1 302 Found\r\nLocation: {redirect_target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            let _ = stream.shutdown().await;
        });

        let health = probe_models_endpoint(&format!("http://{addr}/v1"), ProbePolicy::Strict).await;
        assert!(
            !health.available,
            "a redirect must not validate a different server than inference will use"
        );
        redirector.await.unwrap();
        valid_server.abort();
    }

    #[tokio::test]
    async fn a_manager_with_no_shared_endpoint_adds_no_route_and_refresh_is_inert() {
        let manager = PluginManager::for_test_summaries(Vec::new());
        manager.refresh_shared_endpoint().await;
        assert!(manager.inference_endpoints().await.unwrap().is_empty());
        assert!(manager.start_shared_endpoint().await.is_err());
    }

    #[test]
    fn running_plugin_endpoints_are_healthy() {
        let summary = running_summary();
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn restarting_plugin_endpoints_are_not_available() {
        let summary = PluginSummary {
            status: "restarting".into(),
            error: Some("timed out".into()),
            ..running_summary()
        };
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "starting".into(),
                available: false,
                detail: Some("timed out".into()),
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn first_probe_failure_stays_in_startup_grace() {
        let now = Instant::now();
        let state = apply_endpoint_probe(
            None,
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("GET /models failed".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(state.record.state, "starting");
        assert!(!state.record.available);
        assert_eq!(state.consecutive_failures, 1);
    }

    #[test]
    fn healthy_endpoint_degrades_before_becoming_unhealthy() {
        let now = Instant::now();
        let healthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: 0,
        };

        let degraded = apply_endpoint_probe(
            Some(&healthy),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(degraded.record.state, "degraded");
        assert!(degraded.record.available);
        assert_eq!(degraded.consecutive_failures, 1);

        let unhealthy = apply_endpoint_probe(
            Some(&degraded),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now + Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS),
        );
        assert_eq!(unhealthy.record.state, "unhealthy");
        assert!(!unhealthy.record.available);
        assert_eq!(unhealthy.consecutive_failures, 2);
    }

    #[test]
    fn unhealthy_endpoint_recovers_immediately_on_success() {
        let now = Instant::now();
        let unhealthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: ENDPOINT_FAILURE_THRESHOLD,
        };

        let recovered = apply_endpoint_probe(
            Some(&unhealthy),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            now,
        );
        assert_eq!(recovered.record.state, "healthy");
        assert!(recovered.record.available);
        assert_eq!(recovered.record.models, vec!["demo".to_string()]);
        assert_eq!(recovered.consecutive_failures, 0);
    }

    #[test]
    fn models_probe_url_extends_openai_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/v1/models");
    }

    #[test]
    fn models_probe_url_extends_api_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/api/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/api/v1/models");
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_extracts_models_from_fake_server() {
        let (address, handle, requests) = spawn_fake_models_server(vec![(
            "200 OK",
            r#"{"data":[{"id":"lemonade-small"},{"id":"lemonade-large"}]}"#,
        )])
        .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(health.available);
        assert_eq!(health.state, "healthy");
        assert_eq!(
            health.models,
            vec!["lemonade-small".to_string(), "lemonade-large".to_string()]
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_marks_503_unavailable() {
        let (address, handle, requests) =
            spawn_fake_models_server(vec![("503 Service Unavailable", r#"{"error":"warming"}"#)])
                .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!health.available);
        assert_eq!(health.state, "unhealthy");
        assert!(health.models.is_empty());
        assert!(
            health
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("503 Service Unavailable")
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_recovers_when_fake_server_recovers() {
        let (address, handle, requests) = spawn_fake_models_server(vec![
            ("503 Service Unavailable", r#"{"error":"warming"}"#),
            ("200 OK", r#"{"data":[{"id":"lemonade-recovered"}]}"#),
        ])
        .await;

        let first = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!first.available);
        assert_eq!(first.state, "unhealthy");

        let second = probe_openai_compatible_http_endpoint(&address).await;
        assert!(second.available);
        assert_eq!(second.state, "healthy");
        assert_eq!(second.models, vec!["lemonade-recovered".to_string()]);
        assert_eq!(requests.load(Ordering::SeqCst), 2);

        handle.await.unwrap();
    }

    #[test]
    fn endpoint_declares_inference_capabilities() {
        let endpoint = proto::EndpointManifest {
            endpoint_id: "demo".into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some("http://localhost:8000/api/v1".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        };
        assert_eq!(
            endpoint_declared_capabilities(&endpoint),
            vec![
                "endpoint:inference".to_string(),
                "endpoint:inference/openai_compatible".to_string()
            ]
        );
    }
}
