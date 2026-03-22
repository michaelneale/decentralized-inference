//! Mixture-of-Agents (MoA) — client-side multi-model strategies.
//!
//! Supports multiple algorithms selected by model name:
//!
//! - `moa` — Fan-out + synthesize (MoA paper, arxiv 2406.04692)
//!   All models answer → aggregator synthesizes into refined response.
//!
//! - `best-of-n` — Fan-out + pick best (NSED-inspired)
//!   All models answer → aggregator picks the best response verbatim.
//!   Cheaper than synthesis; good for tool routing where one model nails it.
//!
//! - `moa-2` — Two-layer MoA (advanced-moa.py pattern)
//!   Layer 1: all models answer. Layer 2: all models refine seeing Layer 1.
//!   Then aggregator synthesizes Layer 2. Higher quality, ~3x latency.
//!
//! All strategies run entirely on the client. Hosts are unaware.
//! Requests go through the local proxy which routes via QUIC to hosts.

use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

// ── Strategy enum ──

#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    /// Fan-out to all models, aggregator synthesizes all responses.
    Synthesize,
    /// Fan-out to all models, aggregator picks the best response verbatim.
    BestOfN,
    /// Two-layer MoA: all models answer, then all refine, then aggregator synthesizes.
    TwoLayer,
}

/// Parse model name into a strategy (if it's a MoA request).
pub fn parse_strategy(model: &str) -> Option<Strategy> {
    match model {
        "moa" | "mixture" => Some(Strategy::Synthesize),
        "best-of-n" | "bestofn" | "bon" => Some(Strategy::BestOfN),
        "moa-2" | "moa2" => Some(Strategy::TwoLayer),
        _ => None,
    }
}

/// Check if a model name is a MoA request.
pub fn is_moa_request(model: &str) -> bool {
    parse_strategy(model).is_some()
}

/// Filter models to only include tier 3+ (strong models).
/// Weak models drag down MoA with slow responses and poor quality.
pub fn filter_strong_models(models: &[String]) -> Vec<String> {
    let strong: Vec<String> = models.iter()
        .filter(|m| {
            crate::router::profile_for(m)
                .map(|p| p.tier >= 3)
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    // Fall back to all models if filtering leaves <2
    if strong.len() >= 2 { strong } else { models.to_vec() }
}

// ── Prompts ──

const SYNTHESIZE_SYSTEM: &str = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:";

const BEST_OF_N_SYSTEM: &str = "You have been provided with a set of responses from various models to the latest user query. Your task is to select the BEST response — the one that most accurately, completely, and helpfully answers the query. Output ONLY the selected response, verbatim, with no modifications, preamble, or commentary. Do not synthesize or merge responses. Pick the single best one and reproduce it exactly.\n\nResponses from models:";

// ── Aggregator selection ──

/// Pick the strongest model as aggregator (highest tier from router profiles).
fn pick_aggregator(models: &[String]) -> String {
    let mut best: Option<(&str, u8)> = None;
    for m in models {
        let tier = crate::router::profile_for(m)
            .map(|p| p.tier)
            .unwrap_or(1);
        if best.is_none() || tier > best.unwrap().1 {
            best = Some((m, tier));
        }
    }
    best.map(|(n, _)| n.to_string())
        .unwrap_or_else(|| models[0].clone())
}

// ── Reference injection (MoA pattern) ──

fn inject_references(messages: &[Value], references: &[(String, String)], system_prompt: &str) -> Vec<Value> {
    let mut result = messages.to_vec();
    let mut ref_text = system_prompt.to_string();
    for (i, (model_name, response)) in references.iter().enumerate() {
        ref_text.push_str(&format!("\n\n{}. [{}]:\n{}", i + 1, model_name, response));
    }

    if result.first()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str()) == Some("system")
    {
        if let Some(first) = result.first_mut() {
            if let Some(content) = first.get("content").and_then(|c| c.as_str()) {
                first["content"] = json!(format!("{}\n\n{}", content, ref_text));
            }
        }
    } else {
        result.insert(0, json!({"role": "system", "content": ref_text}));
    }

    result
}

// ── Model response collection ──

async fn collect_model_response(
    client: &Client,
    proxy_url: &str,
    model: &str,
    messages: &[Value],
    max_tokens: u64,
    temperature: f64,
) -> (String, Result<String, String>, u64) {
    let start = Instant::now();
    let body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true,
    });

    let resp = match client.post(format!("{proxy_url}/v1/chat/completions"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let ms = start.elapsed().as_millis() as u64;
            return (model.to_string(), Err(format!("request failed: {e}")), ms);
        }
    };

    // Collect SSE stream — capture both content and reasoning_content.
    // Thinking models put reasoning in reasoning_content and final answer in content.
    // For MoA references, we want both — reasoning IS the useful part.
    let mut content = String::new();
    let mut reasoning = String::new();
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            let ms = start.elapsed().as_millis() as u64;
            return (model.to_string(), Err(format!("read failed: {e}")), ms);
        }
    };

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" { break; }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = chunk.pointer("/choices/0/delta") {
                    if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                        content.push_str(c);
                    }
                    if let Some(r) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                        reasoning.push_str(r);
                    }
                }
            }
        }
    }

    // Use content if substantial, otherwise fall back to reasoning_content
    let final_text = if content.len() > reasoning.len() / 4 {
        content
    } else if !reasoning.is_empty() {
        reasoning
    } else {
        content
    };

    let ms = start.elapsed().as_millis() as u64;
    if final_text.is_empty() {
        (model.to_string(), Err("empty response".into()), ms)
    } else {
        (model.to_string(), Ok(final_text), ms)
    }
}

/// Fan out to all models in parallel. Returns vec of (model_name, response_text).
async fn fan_out(
    client: &Client,
    proxy_url: &str,
    models: &[String],
    messages: &[Value],
    max_tokens: u64,
    temperature: f64,
) -> (Vec<(String, String)>, u64) {
    let start = Instant::now();
    let mut handles = Vec::new();
    for model in models {
        let client = client.clone();
        let url = proxy_url.to_string();
        let model = model.clone();
        let msgs = messages.to_vec();
        handles.push(tokio::spawn(async move {
            collect_model_response(&client, &url, &model, &msgs, max_tokens, temperature).await
        }));
    }

    let mut references: Vec<(String, String)> = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((model_name, Ok(text), ms)) => {
                tracing::info!("moa: {model_name} → {} chars in {ms}ms", text.len());
                references.push((model_name, text));
            }
            Ok((model_name, Err(e), ms)) => {
                tracing::warn!("moa: {model_name} failed ({ms}ms): {e}");
            }
            Err(e) => {
                tracing::warn!("moa: task join error: {e}");
            }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    (references, elapsed)
}

// ── Main handler ──

pub async fn handle_moa_request(
    mut client_stream: TcpStream,
    buf: &[u8],
    n: usize,
    proxy_port: u16,
    models: Vec<String>,
    node: &crate::mesh::Node,
) {
    let proxy_url = format!("http://127.0.0.1:{proxy_port}");
    let http_client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(300))
        .pool_max_idle_per_host(0)
        .build()
        .unwrap_or_default();

    let body = match crate::proxy::extract_body_json(&buf[..n]) {
        Some(b) => b,
        None => {
            let _ = crate::proxy::send_400(client_stream, "invalid JSON body").await;
            return;
        }
    };

    let model_name = body.get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("moa");
    let strategy = parse_strategy(model_name).unwrap_or(Strategy::Synthesize);

    let messages = body.get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let max_tokens = body.get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(4096);

    let temperature = body.get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.7);

    let is_streaming = body.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Filter to strong models only (tier 3+) — weak models drag down the ensemble
    let models = filter_strong_models(&models);

    if models.len() < 2 {
        let _ = crate::proxy::send_400(client_stream, "MoA requires 2+ strong models").await;
        return;
    }

    let aggregator = pick_aggregator(&models);

    tracing::info!(
        "moa: strategy={:?}, {} models [{}], aggregator: {}",
        strategy, models.len(), models.join(", "), aggregator
    );

    let start = Instant::now();

    // ── Layer 1: fan out to all models ──
    let (references, fanout_ms) = fan_out(
        &http_client, &proxy_url, &models, &messages, max_tokens, temperature,
    ).await;

    tracing::info!("moa: fan-out done — {}/{} responded in {fanout_ms}ms",
        references.len(), models.len());

    if references.is_empty() {
        let _ = crate::proxy::send_503(client_stream).await;
        return;
    }

    // Single response — no aggregation needed
    if references.len() == 1 {
        send_plain_response(&mut client_stream, &references[0].1).await;
        return;
    }

    // ── Layer 2 (two-layer MoA only): each model refines seeing Layer 1 ──
    let final_refs = if matches!(strategy, Strategy::TwoLayer) {
        let refined_messages = inject_references(&messages, &references, SYNTHESIZE_SYSTEM);
        let (layer2_refs, layer2_ms) = fan_out(
            &http_client, &proxy_url, &models, &refined_messages, max_tokens, temperature,
        ).await;
        tracing::info!("moa: layer 2 done — {}/{} responded in {layer2_ms}ms",
            layer2_refs.len(), models.len());
        if layer2_refs.is_empty() { references } else { layer2_refs }
    } else {
        references
    };

    // ── Aggregation: inject references, send to strongest model ──
    let system_prompt = match strategy {
        Strategy::BestOfN => BEST_OF_N_SYSTEM,
        _ => SYNTHESIZE_SYSTEM,
    };
    let agg_messages = inject_references(&messages, &final_refs, system_prompt);
    let agg_body = json!({
        "model": aggregator,
        "messages": agg_messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": is_streaming,
    });

    let _inflight = node.begin_inflight_request();

    tracing::info!("moa: aggregating with {aggregator} (strategy={:?})", strategy);

    match http_client.post(format!("{proxy_url}/v1/chat/completions"))
        .json(&agg_body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            if is_streaming {
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
                );
                if client_stream.write_all(header.as_bytes()).await.is_err() {
                    return;
                }
                use tokio_stream::StreamExt;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            let hdr = format!("{:x}\r\n", bytes.len());
                            if client_stream.write_all(hdr.as_bytes()).await.is_err() { break; }
                            if client_stream.write_all(&bytes).await.is_err() { break; }
                            if client_stream.write_all(b"\r\n").await.is_err() { break; }
                        }
                        Err(e) => {
                            tracing::debug!("moa: stream error: {e}");
                            break;
                        }
                    }
                }
                let _ = client_stream.write_all(b"0\r\n\r\n").await;
                let _ = client_stream.shutdown().await;
            } else {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
                            bytes.len()
                        );
                        let _ = client_stream.write_all(header.as_bytes()).await;
                        let _ = client_stream.write_all(&bytes).await;
                        let _ = client_stream.shutdown().await;
                    }
                    Err(e) => {
                        tracing::warn!("moa: aggregation read failed: {e}");
                        let _ = crate::proxy::send_503(client_stream).await;
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!("moa: aggregation request failed: {e}");
            let _ = crate::proxy::send_503(client_stream).await;
        }
    }

    let total_ms = start.elapsed().as_millis();
    tracing::info!("moa: complete in {total_ms}ms (strategy={:?}, {} refs → {aggregator})",
        strategy, final_refs.len());
}

async fn send_plain_response(stream: &mut TcpStream, text: &str) {
    let resp_body = json!({
        "id": "moa-single",
        "object": "chat.completion",
        "model": "moa",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    });
    let body = resp_body.to_string();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).await;
    let _ = stream.write_all(body.as_bytes()).await;
    let _ = stream.shutdown().await;
}
