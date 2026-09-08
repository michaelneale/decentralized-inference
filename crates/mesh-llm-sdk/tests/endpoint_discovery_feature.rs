//! The `endpoint-discovery` SDK feature must be usable on its own — an
//! embedder wanting to list local LLM servers should not have to pull in a
//! client transport, an API server, or a native runtime.

#![cfg(feature = "endpoint-discovery")]

use mesh_llm_sdk::endpoint_discovery::{DiscoveredEndpoint, discover_local_endpoints};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[test]
fn discovered_endpoint_is_publicly_constructible_and_describable() {
    let endpoint = DiscoveredEndpoint {
        product: "Ollama".to_string(),
        port: 11434,
        base_url: "http://127.0.0.1:11434/v1".to_string(),
        models: vec!["llama3:8b".to_string(), "qwen3:4b".to_string()],
    };

    assert_eq!(endpoint.describe(), "Ollama on :11434 (2 models)");
    assert_eq!(endpoint.models.len(), 2);
}

#[tokio::test]
async fn discovery_is_callable_through_the_sdk_and_returns_without_a_local_server() {
    // No assertion on contents: the developer machine running this test may or
    // may not have a real LLM server on a well-known port. What matters is
    // that the SDK entry point is reachable and completes promptly.
    let started = std::time::Instant::now();
    let found = discover_local_endpoints().await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "discovery must not stall an embedding application; took {elapsed:?}"
    );
    for endpoint in &found {
        assert!(
            !endpoint.base_url.is_empty(),
            "a reported endpoint must carry a usable base URL"
        );
    }
}

#[tokio::test]
async fn a_server_on_a_well_known_port_is_reported_through_the_sdk() {
    // Bind the LM Studio port so this exercises the real candidate list rather
    // than an injected address. Skip when something already owns it.
    let Ok(listener) = TcpListener::bind("127.0.0.1:1234").await else {
        return;
    };
    let server = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let body = r#"{"object":"list","data":[{"id":"sdk-test-model"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });

    let found = discover_local_endpoints().await;

    server.abort();
    let reported = found.iter().find(|endpoint| endpoint.port == 1234);
    let reported = reported.expect("a live server on a well-known port must be reported");
    assert_eq!(reported.models, vec!["sdk-test-model".to_string()]);
    assert_eq!(reported.base_url, "http://127.0.0.1:1234/v1");
}
