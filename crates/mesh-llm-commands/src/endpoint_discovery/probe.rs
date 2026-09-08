//! Detect OpenAI-compatible LLM servers that are already running on this
//! machine (Ollama, LM Studio, LiteLLM, vLLM, TGI, llama.cpp, Lemonade) so
//! `mesh setup` and `mesh-llm plugins discover` can offer to publish their
//! models to the mesh through the `openai-endpoint` plugin.
//!
//! Probes are loopback-only and short-timeout. Nothing here writes config;
//! callers decide what to do with the findings.

use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

/// Plugin that fronts an already-running OpenAI-compatible server.
pub const OPENAI_ENDPOINT_PLUGIN: &str = "openai-endpoint";

const PROBE_TIMEOUT: Duration = Duration::from_millis(700);

/// A well-known local server we know how to look for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndpointCandidate {
    /// Product name shown to the operator.
    pub product: &'static str,
    /// Port the product listens on by default.
    pub port: u16,
    /// Base URL handed to the `openai-endpoint` plugin as `url`.
    pub base_url: &'static str,
    /// Endpoint probed to list models.
    pub models_url: &'static str,
}

/// Loopback candidates, most specific first. Ollama is probed on its native
/// `/api/tags` because that shape is unique to Ollama; everything else is
/// identified only as "an OpenAI-compatible server on port N".
pub const CANDIDATES: &[EndpointCandidate] = &[
    EndpointCandidate {
        product: "Ollama",
        port: 11434,
        base_url: "http://127.0.0.1:11434/v1",
        models_url: "http://127.0.0.1:11434/api/tags",
    },
    EndpointCandidate {
        product: "LM Studio",
        port: 1234,
        base_url: "http://127.0.0.1:1234/v1",
        models_url: "http://127.0.0.1:1234/v1/models",
    },
    EndpointCandidate {
        product: "LiteLLM proxy",
        port: 4000,
        base_url: "http://127.0.0.1:4000/v1",
        models_url: "http://127.0.0.1:4000/v1/models",
    },
    EndpointCandidate {
        product: "OpenAI-compatible server",
        port: 8000,
        base_url: "http://127.0.0.1:8000/v1",
        models_url: "http://127.0.0.1:8000/v1/models",
    },
    EndpointCandidate {
        product: "OpenAI-compatible server",
        port: 8080,
        base_url: "http://127.0.0.1:8080/v1",
        models_url: "http://127.0.0.1:8080/v1/models",
    },
];

/// One server that answered a probe.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DiscoveredEndpoint {
    pub product: String,
    pub port: u16,
    pub base_url: String,
    pub models: Vec<String>,
}

impl DiscoveredEndpoint {
    /// Short operator-facing description, e.g. `Ollama on :11434 (3 models)`.
    pub fn describe(&self) -> String {
        let count = self.models.len();
        format!(
            "{} on :{} ({} model{})",
            self.product,
            self.port,
            count,
            if count == 1 { "" } else { "s" }
        )
    }
}

/// Probe every known loopback candidate concurrently.
pub async fn discover_local_endpoints() -> Vec<DiscoveredEndpoint> {
    let Ok(client) = probe_client() else {
        return Vec::new();
    };
    let mut probes = tokio::task::JoinSet::new();
    for candidate in CANDIDATES {
        let client = client.clone();
        let candidate = *candidate;
        probes.spawn(async move { probe_candidate(&client, candidate).await });
    }
    let mut found = Vec::new();
    while let Some(result) = probes.join_next().await {
        if let Ok(Some(endpoint)) = result {
            found.push(endpoint);
        }
    }
    found.sort_by_key(|endpoint| endpoint.port);
    found
}

/// Loopback probes must never traverse a proxy: `HTTP_PROXY`/`ALL_PROXY` would
/// send them off-box and make a live local server undetectable.
fn probe_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .no_proxy()
        .build()
}

async fn probe_candidate(
    client: &reqwest::Client,
    candidate: EndpointCandidate,
) -> Option<DiscoveredEndpoint> {
    let response = client.get(candidate.models_url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body = response.json::<Value>().await.ok()?;
    let models = parse_model_ids(&body)?;
    Some(DiscoveredEndpoint {
        product: candidate.product.to_string(),
        port: candidate.port,
        base_url: candidate.base_url.to_string(),
        models,
    })
}

/// Extract model ids from either an OpenAI `/v1/models` body (`data[].id`) or
/// an Ollama `/api/tags` body (`models[].name`). Returns `None` when the body
/// has neither shape, so an unrelated JSON service on the port is not reported
/// as an LLM server.
pub fn parse_model_ids(body: &Value) -> Option<Vec<String>> {
    if let Some(entries) = body.get("data").and_then(Value::as_array) {
        return Some(collect_field(entries, "id"));
    }
    if let Some(entries) = body.get("models").and_then(Value::as_array) {
        return Some(collect_field(entries, "name"));
    }
    None
}

fn collect_field(entries: &[Value], field: &str) -> Vec<String> {
    let mut ids = entries
        .iter()
        .filter_map(|entry| entry.get(field).and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_models_response() {
        let body = json!({"object": "list", "data": [{"id": "b"}, {"id": "a"}, {"id": "a"}]});

        assert_eq!(
            parse_model_ids(&body),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn parses_ollama_tags_response() {
        let body = json!({"models": [{"name": "llama3:8b", "size": 1}]});

        assert_eq!(parse_model_ids(&body), Some(vec!["llama3:8b".to_string()]));
    }

    #[test]
    fn rejects_unrelated_json_service() {
        let body = json!({"status": "ok"});

        assert_eq!(parse_model_ids(&body), None);
    }

    #[test]
    fn empty_model_list_is_still_a_server() {
        let body = json!({"data": []});

        assert_eq!(parse_model_ids(&body), Some(Vec::new()));
    }

    #[test]
    fn describe_singularises_one_model() {
        let endpoint = DiscoveredEndpoint {
            product: "Ollama".into(),
            port: 11434,
            base_url: "http://127.0.0.1:11434/v1".into(),
            models: vec!["llama3:8b".into()],
        };

        assert_eq!(endpoint.describe(), "Ollama on :11434 (1 model)");
    }

    #[test]
    fn candidate_ports_are_unique() {
        let mut ports = CANDIDATES.iter().map(|c| c.port).collect::<Vec<_>>();
        ports.sort_unstable();
        let before = ports.len();
        ports.dedup();

        assert_eq!(
            ports.len(),
            before,
            "duplicate candidate ports would double-report one server"
        );
    }
}

#[cfg(test)]
mod live_probe_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Serve one JSON body on an ephemeral loopback port.
    async fn spawn_json_server(body: &'static str) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await.expect("read request");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            let _ = stream.shutdown().await;
        });
        (port, handle)
    }

    #[tokio::test]
    async fn probe_reports_a_real_openai_compatible_server() {
        let (port, server) =
            spawn_json_server(r#"{"object":"list","data":[{"id":"qwen3-8b"}]}"#).await;
        let models_url = format!("http://127.0.0.1:{port}/v1/models").leak();
        let base_url = format!("http://127.0.0.1:{port}/v1").leak();
        let candidate = EndpointCandidate {
            product: "Test server",
            port,
            base_url,
            models_url,
        };
        let client = probe_client().expect("client");

        let found = probe_candidate(&client, candidate)
            .await
            .expect("a live server must be detected");

        assert_eq!(found.models, vec!["qwen3-8b".to_string()]);
        assert_eq!(found.base_url.as_str(), base_url);
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn probe_ignores_a_non_llm_json_service_on_a_candidate_port() {
        let (port, server) = spawn_json_server(r#"{"status":"ok"}"#).await;
        let candidate = EndpointCandidate {
            product: "Test server",
            port,
            base_url: format!("http://127.0.0.1:{port}/v1").leak(),
            models_url: format!("http://127.0.0.1:{port}/v1/models").leak(),
        };
        let client = probe_client().expect("client");

        assert!(
            probe_candidate(&client, candidate).await.is_none(),
            "an unrelated JSON service must not be published as an LLM endpoint"
        );
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn probe_of_a_closed_port_returns_none_without_hanging() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);
        let candidate = EndpointCandidate {
            product: "Test server",
            port,
            base_url: format!("http://127.0.0.1:{port}/v1").leak(),
            models_url: format!("http://127.0.0.1:{port}/v1/models").leak(),
        };
        let client = probe_client().expect("client");

        assert!(probe_candidate(&client, candidate).await.is_none());
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn a_server_that_accepts_and_never_replies_is_cut_off_by_the_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();
        // Accept the connection and hold it open without ever responding.
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(stream);
        });
        let candidate = EndpointCandidate {
            product: "Test server",
            port,
            base_url: format!("http://127.0.0.1:{port}/v1").leak(),
            models_url: format!("http://127.0.0.1:{port}/v1/models").leak(),
        };
        let client = probe_client().expect("client");

        let started = std::time::Instant::now();
        let found = probe_candidate(&client, candidate).await;
        let elapsed = started.elapsed();

        server.abort();
        assert!(found.is_none());
        assert!(
            elapsed < PROBE_TIMEOUT * 4,
            "a hanging local service must not stall `mesh-llm setup`; took {elapsed:?}"
        );
    }
}
