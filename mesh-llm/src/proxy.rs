//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::{election, mesh, router, tunnel};
use anyhow::Result;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

// ── Request parsing ──

/// Peek at an HTTP request without consuming it. Returns bytes peeked and optional model name.
pub async fn peek_request(stream: &TcpStream, buf: &mut [u8]) -> Result<(usize, Option<String>)> {
    let n = stream.peek(buf).await?;
    if n == 0 {
        anyhow::bail!("Empty request");
    }
    let model = extract_model_from_http(&buf[..n]);
    Ok((n, model))
}

/// Extract `"model"` field from a JSON POST body in an HTTP request.
pub fn extract_model_from_http(buf: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(buf).ok()?;
    let body_start = s.find("\r\n\r\n")? + 4;
    let body = &s[body_start..];
    let model_key = "\"model\"";
    let pos = body.find(model_key)?;
    let after_key = &body[pos + model_key.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    let after_quote = after_ws.strip_prefix('"')?;
    let end = after_quote.find('"')?;
    Some(after_quote[..end].to_string())
}

/// Extract a session hint from an HTTP request for MoE sticky routing.
/// Looks for "user" or "session_id" in the JSON body. Falls back to None.
pub fn extract_session_hint(buf: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(buf).ok()?;
    let body_start = s.find("\r\n\r\n")? + 4;
    let body = &s[body_start..];
    // Try "user" field first (standard OpenAI parameter)
    for key in &["\"user\"", "\"session_id\""] {
        if let Some(pos) = body.find(key) {
            let after_key = &body[pos + key.len()..];
            let after_colon = after_key.trim_start().strip_prefix(':')?;
            let after_ws = after_colon.trim_start();
            let after_quote = after_ws.strip_prefix('"')?;
            let end = after_quote.find('"')?;
            return Some(after_quote[..end].to_string());
        }
    }
    None
}

/// Compute a sticky routing key from a peeked HTTP request.
///
/// Tries (in order):
/// 1. Explicit `"user"` or `"session_id"` field (cooperative clients)
/// 2. Conversation fingerprint: hash of system prompt + first user message
///    (this matches what llama.cpp's prompt cache keys on, so same conversation
///    prefix → same backend → KV cache hit)
///
/// Returns None only for requests with no parseable body (e.g. GET /v1/models).
pub fn sticky_key(buf: &[u8]) -> Option<u64> {
    // Fast path: explicit session hint
    if let Some(hint) = extract_session_hint(buf) {
        return Some(hash_bytes(hint.as_bytes()));
    }
    // Slow path: fingerprint the conversation prefix
    conversation_fingerprint(buf)
}

/// Hash the system prompt + first user message from a chat completion request.
/// This is cheap — we already peeked the buffer, just scan for the text.
fn conversation_fingerprint(buf: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(buf).ok()?;
    let body_start = s.find("\r\n\r\n")? + 4;
    let body = &s[body_start..];

    // We need system + first user message. Do lightweight JSON scanning
    // rather than a full parse — these fields are always near the start.
    let mut hasher: u64 = 0;
    let mut found = false;

    // Find "messages" array and extract role/content pairs
    // We hash: all system messages + the first user message content
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(messages) = parsed.get("messages").and_then(|m| m.as_array()) {
            for msg in messages {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                match role {
                    "system" => {
                        if let Some(content) = msg_content_text(msg) {
                            hasher = hash_combine(hasher, hash_bytes(content.as_bytes()));
                            found = true;
                        }
                    }
                    "user" => {
                        if let Some(content) = msg_content_text(msg) {
                            hasher = hash_combine(hasher, hash_bytes(content.as_bytes()));
                            found = true;
                        }
                        break; // Only first user message — that's the KV prefix key
                    }
                    _ => {}
                }
            }
        }
    }

    if found {
        Some(hasher)
    } else {
        None
    }
}

/// Extract text content from a message (handles both string and Anthropic-style content blocks).
fn msg_content_text(msg: &serde_json::Value) -> Option<String> {
    // Simple string content
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return Some(s.to_string());
    }
    // Anthropic-style: [{"type":"text","text":"..."}]
    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for b in blocks {
            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                out.push_str(t);
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
        (acc ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn hash_combine(a: u64, b: u64) -> u64 {
    a.wrapping_mul(31).wrapping_add(b)
}

/// Try to parse the JSON body from a peeked HTTP request buffer.
pub fn extract_body_json(buf: &[u8]) -> Option<serde_json::Value> {
    let s = std::str::from_utf8(buf).ok()?;
    let body_start = s.find("\r\n\r\n")? + 4;
    let body = &s[body_start..];
    serde_json::from_str(body).ok()
}

pub fn is_models_list_request(buf: &[u8]) -> bool {
    let s = String::from_utf8_lossy(buf);
    s.starts_with("GET ")
        && (s.contains("/v1/models") || s.contains("/models"))
        && !s.contains("/v1/models/")
}

pub fn is_drop_request(buf: &[u8]) -> bool {
    let s = String::from_utf8_lossy(buf);
    s.starts_with("POST ") && s.contains("/mesh/drop")
}

// ── Model-aware tunnel routing ──

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(node: mesh::Node, tcp_stream: TcpStream, track_demand: bool) {
    let mut buf = vec![0u8; 32768];
    let (n, model_name) = match peek_request(&tcp_stream, &mut buf).await {
        Ok(v) => v,
        Err(_) => return,
    };

    // Handle /v1/models
    if is_models_list_request(&buf[..n]) {
        let served = node.models_being_served().await;
        let _ = send_models_list(tcp_stream, &served).await;
        return;
    }

    // Demand tracking for rebalancing (done after routing so we track the actual model used)
    // We'll track below after routing resolves the effective model

    // Smart routing: if no model specified (or model="auto"), classify and pick
    let routed_model = if model_name.is_none() || model_name.as_deref() == Some("auto") {
        if let Some(body_json) = extract_body_json(&buf[..n]) {
            let cl = router::classify(&body_json);
            let served = node.models_being_served().await;
            let available: Vec<(&str, f64)> =
                served.iter().map(|name| (name.as_str(), 0.0)).collect();
            let picked = router::pick_model_classified(&cl, &available);
            if let Some(name) = picked {
                tracing::info!(
                    "router: {:?}/{:?} tools={} → {name}",
                    cl.category,
                    cl.complexity,
                    cl.needs_tools
                );
                Some(name.to_string())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let effective_model = routed_model.or(model_name);

    // Demand tracking for rebalancing
    if track_demand {
        if let Some(ref name) = effective_model {
            node.record_request(name);
        }
    }

    // Resolve target hosts by model name, fall back to any host.
    // Use conversation fingerprint to put the sticky-preferred host first
    // so the same multi-turn conversation hits the same backend (KV cache affinity).
    let mut target_hosts = if let Some(ref name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        vec![]
    };
    let target_hosts = if target_hosts.is_empty() {
        match node.any_host().await {
            Some(p) => vec![p.id],
            None => {
                let _ = send_503(tcp_stream).await;
                return;
            }
        }
    } else {
        // Sticky routing: rotate the host list so the conversation-preferred
        // host is first. Falls back to the existing hash-of-node-id ordering.
        if target_hosts.len() > 1 {
            if let Some(key) = sticky_key(&buf[..n]) {
                let idx = (key as usize) % target_hosts.len();
                target_hosts.rotate_left(idx);
            }
        }
        target_hosts
    };

    // Try each host in order — if tunnel fails, retry with next.
    // On first failure, trigger background gossip refresh so future requests
    // have a fresh routing table (doesn't block the retry loop).
    let mut last_err = None;
    let mut refreshed = false;
    for target_host in &target_hosts {
        match node.open_http_tunnel(*target_host).await {
            Ok((quic_send, quic_recv)) => {
                if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                    tracing::debug!("HTTP tunnel relay ended: {e}");
                }
                return;
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to tunnel to host {}: {e}, trying next",
                    target_host.fmt_short()
                );
                last_err = Some(e);
                // Background refresh on first failure — non-blocking
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move {
                        refresh_node.gossip_one_peer().await;
                    });
                    refreshed = true;
                }
            }
        }
    }
    // All hosts failed
    if let Some(e) = last_err {
        tracing::warn!("All hosts failed for model {:?}: {e}", effective_model);
    }
    let _ = send_503(tcp_stream).await;
}

/// Route a request to a known inference target (local llama-server or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: TcpStream,
    target: election::InferenceTarget,
) {
    match target {
        election::InferenceTarget::Local(port) | election::InferenceTarget::MoeLocal(port) => {
            match TcpStream::connect(format!("127.0.0.1:{port}")).await {
                Ok(upstream) => {
                    let _inflight = node.begin_inflight_request();
                    let _ = upstream.set_nodelay(true);
                    if let Err(e) = tunnel::relay_tcp_streams(tcp_stream, upstream).await {
                        tracing::debug!("API proxy (local) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't reach llama-server on {port}: {e}");
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::Remote(host_id)
        | election::InferenceTarget::MoeRemote(host_id) => {
            match node.open_http_tunnel(host_id).await {
                Ok((quic_send, quic_recv)) => {
                    if let Err(e) =
                        tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await
                    {
                        tracing::debug!("API proxy (remote) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "API proxy: can't tunnel to host {}: {e}",
                        host_id.fmt_short()
                    );
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::None => {
            let _ = send_503(tcp_stream).await;
        }
    }
}

// ── Response helpers ──

pub async fn send_models_list(mut stream: TcpStream, models: &[String]) -> std::io::Result<()> {
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let has_vision = crate::download::MODEL_CATALOG
                .iter()
                .find(|c| {
                    c.name == m.as_str()
                        || c.file.strip_suffix(".gguf").unwrap_or(c.file) == m.as_str()
                })
                .map(|c| c.mmproj.is_some())
                .unwrap_or(false);
            let mut caps = vec!["text"];
            if has_vision {
                caps.push("vision");
            }
            serde_json::json!({
                "id": m,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
            })
        })
        .collect();
    let body = serde_json::json!({ "object": "list", "data": data }).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_json_ok(mut stream: TcpStream, data: &serde_json::Value) -> std::io::Result<()> {
    let body = data.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_400(mut stream: TcpStream, msg: &str) -> std::io::Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_503(mut stream: TcpStream) -> std::io::Result<()> {
    let body = r#"{"error":"No inference server available — election in progress"}"#;
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Pipeline-aware HTTP proxy for local targets.
///
/// Instead of TCP tunneling, this:
/// 1. Parses the HTTP request body
/// 2. Calls the planner model for a pre-plan
/// 3. Injects the plan into the request
/// 4. Forwards to the strong model via HTTP
/// 5. Streams the response back to the client
pub async fn pipeline_proxy_local(
    mut client_stream: TcpStream,
    buf: &[u8],
    n: usize,
    planner_port: u16,
    planner_model: &str,
    strong_port: u16,
    node: &mesh::Node,
) {
    // Parse request body
    let mut body = match extract_body_json(&buf[..n]) {
        Some(b) => b,
        None => {
            let _ = send_400(client_stream, "invalid JSON body").await;
            return;
        }
    };

    // Extract whether this is a streaming request
    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Pre-plan: ask the small model
    let http_client = reqwest::Client::new();
    let planner_url = format!("http://127.0.0.1:{planner_port}");
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    match crate::pipeline::pre_plan(&http_client, &planner_url, planner_model, &messages).await {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::pipeline::inject_plan(&mut body, &plan);
        }
        Err(e) => {
            tracing::warn!("pipeline: pre-plan failed ({e}), proceeding without plan");
        }
    }

    // Forward to strong model — use reqwest for full HTTP handling
    let strong_url = format!("http://127.0.0.1:{strong_port}/v1/chat/completions");

    let _inflight = node.begin_inflight_request();

    if is_streaming {
        // Streaming: forward SSE chunks to client
        match http_client.post(&strong_url).json(&body).send().await {
            Ok(resp) => {
                let status = resp.status();
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("text/event-stream")
                    .to_string();

                // Send HTTP response headers
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
                );
                if client_stream.write_all(header.as_bytes()).await.is_err() {
                    return;
                }

                // Stream body chunks
                use tokio_stream::StreamExt;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            // HTTP chunked encoding
                            let chunk_header = format!("{:x}\r\n", bytes.len());
                            if client_stream
                                .write_all(chunk_header.as_bytes())
                                .await
                                .is_err()
                            {
                                break;
                            }
                            if client_stream.write_all(&bytes).await.is_err() {
                                break;
                            }
                            if client_stream.write_all(b"\r\n").await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("pipeline: stream error: {e}");
                            break;
                        }
                    }
                }
                // Terminal chunk
                let _ = client_stream.write_all(b"0\r\n\r\n").await;
                let _ = client_stream.shutdown().await;
            }
            Err(e) => {
                tracing::warn!("pipeline: strong model request failed: {e}");
                let _ = send_503(client_stream).await;
            }
        }
    } else {
        // Non-streaming: simple request/response
        match http_client.post(&strong_url).json(&body).send().await {
            Ok(resp) => {
                let status = resp.status();
                match resp.bytes().await {
                    Ok(resp_bytes) => {
                        let header = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                            resp_bytes.len()
                        );
                        let _ = client_stream.write_all(header.as_bytes()).await;
                        let _ = client_stream.write_all(&resp_bytes).await;
                        let _ = client_stream.shutdown().await;
                    }
                    Err(e) => {
                        tracing::warn!("pipeline: response read failed: {e}");
                        let _ = send_503(client_stream).await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("pipeline: strong model request failed: {e}");
                let _ = send_503(client_stream).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_session_hint_user_field() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"qwen\",\"user\":\"alice\",\"messages\":[]}";
        assert_eq!(extract_session_hint(req), Some("alice".to_string()));
    }

    #[test]
    fn test_extract_session_hint_session_id() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"session_id\":\"sess-42\"}";
        assert_eq!(extract_session_hint(req), Some("sess-42".to_string()));
    }

    #[test]
    fn test_extract_session_hint_user_preferred_over_session_id() {
        // "user" appears before "session_id" in our search order
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"user\":\"bob\",\"session_id\":\"sess-1\"}";
        assert_eq!(extract_session_hint(req), Some("bob".to_string()));
    }

    #[test]
    fn test_extract_session_hint_none() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"messages\":[]}";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_no_body() {
        let req = b"GET /v1/models HTTP/1.1\r\n\r\n";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_no_headers_end() {
        let req = b"POST /v1/chat body without proper headers";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_whitespace_variants() {
        // Extra whitespace around colon and value
        let req = b"POST / HTTP/1.1\r\n\r\n{\"user\" : \"charlie\" }";
        assert_eq!(extract_session_hint(req), Some("charlie".to_string()));
    }

    #[test]
    fn test_extract_session_hint_empty_value() {
        let req = b"POST / HTTP/1.1\r\n\r\n{\"user\":\"\"}";
        assert_eq!(extract_session_hint(req), Some("".to_string()));
    }

    #[test]
    fn test_extract_model_from_http_basic() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"Qwen3-30B\"}";
        assert_eq!(extract_model_from_http(req), Some("Qwen3-30B".to_string()));
    }

    // ── Sticky routing tests ──

    #[test]
    fn test_sticky_key_uses_session_hint_when_present() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        let key = sticky_key(req);
        assert!(key.is_some());

        // Same user → same key
        let req2 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"different message\"}]}";
        assert_eq!(sticky_key(req), sticky_key(req2));

        // Different user → different key
        let req3 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"user\":\"bob\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        assert_ne!(sticky_key(req), sticky_key(req3));
    }

    #[test]
    fn test_sticky_key_fingerprints_conversation_without_user() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"messages\":[{\"role\":\"system\",\"content\":\"You are helpful.\"},{\"role\":\"user\",\"content\":\"Hello\"}]}";
        let key = sticky_key(req);
        assert!(key.is_some());

        // Same system + first user message → same key (multi-turn stickiness)
        let req_turn2 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"messages\":[{\"role\":\"system\",\"content\":\"You are helpful.\"},{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hi!\"},{\"role\":\"user\",\"content\":\"How are you?\"}]}";
        assert_eq!(sticky_key(req), sticky_key(req_turn2));
    }

    #[test]
    fn test_sticky_key_different_system_prompt_different_key() {
        let req1 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"system\",\"content\":\"You are a coder.\"},{\"role\":\"user\",\"content\":\"Hello\"}]}";
        let req2 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"system\",\"content\":\"You are a poet.\"},{\"role\":\"user\",\"content\":\"Hello\"}]}";
        assert_ne!(sticky_key(req1), sticky_key(req2));
    }

    #[test]
    fn test_sticky_key_different_first_user_message_different_key() {
        let req1 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"user\",\"content\":\"Write Python code\"}]}";
        let req2 = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"user\",\"content\":\"Write Rust code\"}]}";
        assert_ne!(sticky_key(req1), sticky_key(req2));
    }

    #[test]
    fn test_sticky_key_none_for_get_request() {
        let req = b"GET /v1/models HTTP/1.1\r\n\r\n";
        assert!(sticky_key(req).is_none());
    }

    #[test]
    fn test_sticky_key_none_for_empty_messages() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"messages\":[]}";
        assert!(sticky_key(req).is_none());
    }

    #[test]
    fn test_sticky_key_works_with_anthropic_content_blocks() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello world\"}]}]}";
        let key = sticky_key(req);
        assert!(key.is_some());
    }

    #[test]
    fn test_conversation_fingerprint_stability() {
        // The fingerprint must be deterministic — same input, same output, always.
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"messages\":[{\"role\":\"system\",\"content\":\"sys\"},{\"role\":\"user\",\"content\":\"hello\"}]}";
        let k1 = sticky_key(req);
        let k2 = sticky_key(req);
        assert_eq!(k1, k2);
    }
}
