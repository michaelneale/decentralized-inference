//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::{election, mesh, tunnel};
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

pub fn is_models_list_request(buf: &[u8]) -> bool {
    let s = String::from_utf8_lossy(buf);
    s.starts_with("GET ") && (s.contains("/v1/models") || s.contains("/models"))
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

    // Demand tracking for rebalancing
    if track_demand {
        if let Some(ref name) = model_name {
            node.record_request(name);
        }
    }

    // Resolve target hosts by model name, fall back to any host
    let target_hosts = if let Some(ref name) = model_name {
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
                tracing::warn!("Failed to tunnel to host {}: {e}, trying next", target_host.fmt_short());
                last_err = Some(e);
                // Background refresh on first failure — non-blocking
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move { refresh_node.gossip_one_peer().await; });
                    refreshed = true;
                }
            }
        }
    }
    // All hosts failed
    if let Some(e) = last_err {
        tracing::warn!("All hosts failed for model {:?}: {e}", model_name);
    }
    let _ = send_503(tcp_stream).await;
}

/// Route a request to a known inference target (local llama-server or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(node: mesh::Node, tcp_stream: TcpStream, target: election::InferenceTarget) {
    match target {
        election::InferenceTarget::Local(llama_port) => {
            match TcpStream::connect(format!("127.0.0.1:{llama_port}")).await {
                Ok(upstream) => {
                    let _ = upstream.set_nodelay(true);
                    if let Err(e) = tunnel::relay_tcp_streams(tcp_stream, upstream).await {
                        tracing::debug!("API proxy (local) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't reach llama-server on {llama_port}: {e}");
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::Remote(host_id) => {
            match node.open_http_tunnel(host_id).await {
                Ok((quic_send, quic_recv)) => {
                    if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                        tracing::debug!("API proxy (remote) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't tunnel to host {}: {e}", host_id.fmt_short());
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
            serde_json::json!({
                "id": m,
                "object": "model",
                "owned_by": "mesh-llm",
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
        body.len(), body
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
