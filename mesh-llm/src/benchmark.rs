use crate::{mesh, telemetry};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkProbe {
    pub name: &'static str,
    pub prompt: &'static str,
    pub settings: BenchmarkSettings,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSettings {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: u32,
    pub stream: bool,
}

#[allow(dead_code)]
pub fn analysis_report_v1() -> BenchmarkProbe {
    BenchmarkProbe {
        name: "analysis_report_v1",
        prompt: ANALYSIS_REPORT_V1_PROMPT,
        settings: BenchmarkSettings {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 300,
            stream: false,
        },
    }
}

#[allow(dead_code)]
pub fn analysis_report_v1_stream() -> BenchmarkProbe {
    BenchmarkProbe {
        name: "analysis_report_v1_stream",
        prompt: ANALYSIS_REPORT_V1_PROMPT,
        settings: BenchmarkSettings {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 300,
            stream: true,
        },
    }
}

pub const ANALYSIS_REPORT_V1_PROMPT: &str = r#"You are a senior AI systems analyst.

Read the following internal engineering report and produce:

1. A concise executive summary (5 bullet points)
2. A list of the top 5 technical risks
3. Three concrete performance improvement recommendations
4. A final 2–3 sentence conclusion

Be precise, structured, and avoid generic language.

---

INTERNAL REPORT:

System Overview:
The LLM inference service currently runs a 13B parameter transformer model behind an HTTP API. The deployment uses tensor parallelism across 2 GPUs with dynamic batching enabled. Requests are routed through a queue that aggregates requests for up to 15ms before dispatch. The average request context length is 1,200 input tokens with a mean generation length of 220 output tokens.

Observed Performance:
Over the past 7 days, median latency has remained stable at 820ms. However, p95 latency increased from 1.4s to 2.1s during peak traffic windows (14:00–18:00 UTC). GPU utilization averages 72% during off-peak hours and reaches 96–98% during peak hours. KV cache usage frequently exceeds 85% during sustained load. Occasional batch size reductions have been observed when memory pressure increases.

Infrastructure Notes:
Each node has 2x A100 40GB GPUs. CPU utilization remains below 55% under load. Network I/O does not appear saturated. No significant disk I/O is observed. Horizontal scaling is currently manual. Autoscaling is under consideration but not implemented.

Recent Changes:
A new feature was deployed enabling longer context windows (up to 8k tokens). After deployment, memory fragmentation increased and average prefill time rose by 18%. Decode token throughput per GPU decreased from 145 tokens/sec to 123 tokens/sec under peak concurrency.

Error Metrics:
HTTP 5xx errors remain below 0.3%. However, request timeouts increased by 1.1% during peak hours. No GPU OOM crashes have been recorded, but soft memory allocation retries increased.

Operational Constraints:
Latency SLO: p95 < 1.5s
Target throughput: 30 requests/sec sustained
Cost sensitivity: moderate — GPU overprovisioning should be avoided if possible.

---

Produce your response now."#;

pub fn spawn_probe_loop(
    node: mesh::Node,
    telemetry: telemetry::Telemetry,
    api_port: u16,
    interval_secs: u64,
    model_override: Option<String>,
) {
    let interval_secs = interval_secs.max(10);
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(55))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Benchmark runner disabled: failed to build HTTP client: {e}");
                return;
            }
        };
        let probe_nonstream = analysis_report_v1();
        let probe_stream = analysis_report_v1_stream();
        let mut run_stream_next = false;
        loop {
            let probe = if run_stream_next { &probe_stream } else { &probe_nonstream };
            if let Err(e) = run_probe_once(&client, &node, &telemetry, api_port, probe, model_override.as_deref()).await {
                tracing::debug!("Benchmark probe failed to record: {e}");
            }
            run_stream_next = !run_stream_next;
            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
        }
    });
}

async fn run_probe_once(
    client: &reqwest::Client,
    node: &mesh::Node,
    telemetry: &telemetry::Telemetry,
    api_port: u16,
    probe: &BenchmarkProbe,
    model_override: Option<&str>,
) -> anyhow::Result<()> {
    let ts = unix_ts_now();
    let mesh_id = node.mesh_id().await;
    let model = match model_override {
        Some(m) => m.to_string(),
        None => match discover_model(client, api_port).await {
            Some(m) => m,
            None => {
                tracing::debug!("Benchmark probe {} skipped: no model available from /v1/models yet", probe.name);
                return Ok(());
            }
        },
    };

    let url = format!("http://127.0.0.1:{api_port}/v1/chat/completions");
    let body = if probe.settings.stream {
        json!({
            "model": model,
            "messages": [
                {"role": "user", "content": probe.prompt}
            ],
            "temperature": probe.settings.temperature,
            "top_p": probe.settings.top_p,
            "max_tokens": probe.settings.max_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
        })
    } else {
        json!({
            "model": model,
            "messages": [
                {"role": "user", "content": probe.prompt}
            ],
            "temperature": probe.settings.temperature,
            "top_p": probe.settings.top_p,
            "max_tokens": probe.settings.max_tokens,
            "stream": false,
        })
    };

    if probe.settings.stream {
        return run_stream_probe_once(client, telemetry, probe, ts, mesh_id, model, url, body).await;
    }

    let started = Instant::now();
    let resp = client.post(&url).json(&body).send().await;
    let latency_ms = started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;

    match resp {
        Ok(resp) => {
            let status = resp.status();
            let status_code = Some(status.as_u16());
            let text = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                telemetry
                    .insert_benchmark_run(telemetry::BenchmarkRunInput {
                        ts,
                        mesh_id,
                        target_node_id: None,
                        model,
                        probe_name: probe.name.into(),
                        probe_type: "nonstream".into(),
                        stream: probe.settings.stream,
                        temperature: probe.settings.temperature,
                        top_p: probe.settings.top_p,
                        max_tokens: probe.settings.max_tokens,
                        prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
                        route_kind: None,
                        success: false,
                        status_code,
                        latency_ms: Some(latency_ms),
                        ttft_ms: None,
                        prompt_tokens: None,
                        completion_tokens: None,
                        tokens_per_sec: None,
                        error_kind: Some("http_error".into()),
                        error_message: Some(truncate(&text, 300)),
                        response_shape_ok: None,
                    })
                    .await?;
                return Ok(());
            }

            let parsed: Option<ChatCompletionResponse> = serde_json::from_str(&text).ok();
            let (prompt_tokens, completion_tokens, response_shape_ok) = if let Some(ref p) = parsed {
                let content = p.choices.first().and_then(|c| c.message.content.as_deref()).unwrap_or("");
                (
                    p.usage.as_ref().map(|u| u.prompt_tokens),
                    p.usage.as_ref().map(|u| u.completion_tokens),
                    Some(validate_analysis_report_shape(content)),
                )
            } else {
                (None, None, None)
            };
            let tps = completion_tokens.and_then(|n| {
                if latency_ms == 0 {
                    None
                } else {
                    Some(n as f64 / (latency_ms as f64 / 1000.0))
                }
            });

            telemetry
                .insert_benchmark_run(telemetry::BenchmarkRunInput {
                    ts,
                    mesh_id,
                    target_node_id: None,
                    model,
                    probe_name: probe.name.into(),
                    probe_type: "nonstream".into(),
                    stream: probe.settings.stream,
                    temperature: probe.settings.temperature,
                    top_p: probe.settings.top_p,
                    max_tokens: probe.settings.max_tokens,
                    prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
                    route_kind: None,
                    success: true,
                    status_code,
                    latency_ms: Some(latency_ms),
                    ttft_ms: None,
                    prompt_tokens,
                    completion_tokens,
                    tokens_per_sec: tps,
                    error_kind: None,
                    error_message: None,
                    response_shape_ok,
                })
                .await?;
        }
        Err(e) => {
            telemetry
                .insert_benchmark_run(telemetry::BenchmarkRunInput {
                    ts,
                    mesh_id,
                    target_node_id: None,
                    model,
                    probe_name: probe.name.into(),
                    probe_type: "nonstream".into(),
                    stream: probe.settings.stream,
                    temperature: probe.settings.temperature,
                    top_p: probe.settings.top_p,
                    max_tokens: probe.settings.max_tokens,
                    prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
                    route_kind: None,
                    success: false,
                    status_code: None,
                    latency_ms: Some(latency_ms),
                    ttft_ms: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    tokens_per_sec: None,
                    error_kind: Some(if e.is_timeout() { "timeout" } else { "request_error" }.into()),
                    error_message: Some(truncate(&e.to_string(), 300)),
                    response_shape_ok: None,
                })
                .await?;
        }
    }

    Ok(())
}

async fn run_stream_probe_once(
    client: &reqwest::Client,
    telemetry: &telemetry::Telemetry,
    probe: &BenchmarkProbe,
    ts: i64,
    mesh_id: Option<String>,
    model: String,
    url: String,
    body: serde_json::Value,
) -> anyhow::Result<()> {
    let started = Instant::now();
    let resp = client.post(&url).json(&body).send().await;

    let mut resp = match resp {
        Ok(r) => r,
        Err(e) => {
            let latency_ms = started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
            telemetry.insert_benchmark_run(telemetry::BenchmarkRunInput {
                ts,
                mesh_id,
                target_node_id: None,
                model,
                probe_name: probe.name.into(),
                probe_type: "stream".into(),
                stream: true,
                temperature: probe.settings.temperature,
                top_p: probe.settings.top_p,
                max_tokens: probe.settings.max_tokens,
                prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
                route_kind: None,
                success: false,
                status_code: None,
                latency_ms: Some(latency_ms),
                ttft_ms: None,
                prompt_tokens: None,
                completion_tokens: None,
                tokens_per_sec: None,
                error_kind: Some(if e.is_timeout() { "timeout" } else { "request_error" }.into()),
                error_message: Some(truncate(&e.to_string(), 300)),
                response_shape_ok: None,
            }).await?;
            return Ok(());
        }
    };

    let status = resp.status();
    let status_code = Some(status.as_u16());
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let latency_ms = started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
        telemetry.insert_benchmark_run(telemetry::BenchmarkRunInput {
            ts,
            mesh_id,
            target_node_id: None,
            model,
            probe_name: probe.name.into(),
            probe_type: "stream".into(),
            stream: true,
            temperature: probe.settings.temperature,
            top_p: probe.settings.top_p,
            max_tokens: probe.settings.max_tokens,
            prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
            route_kind: None,
            success: false,
            status_code,
            latency_ms: Some(latency_ms),
            ttft_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
            tokens_per_sec: None,
            error_kind: Some("http_error".into()),
            error_message: Some(truncate(&text, 300)),
            response_shape_ok: None,
        }).await?;
        return Ok(());
    }

    let mut pending = String::new();
    let mut saw_done = false;
    let mut parse_error: Option<String> = None;
    let mut content = String::new();
    let mut ttft_ms: Option<u32> = None;
    let mut prompt_tokens: Option<u32> = None;
    let mut completion_tokens: Option<u32> = None;

    while let Some(chunk) = resp.chunk().await? {
        pending.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = pending.find("\n\n") {
            let event = pending[..idx].to_string();
            pending.drain(..idx + 2);

            for line in event.lines() {
                let Some(data) = line.strip_prefix("data: ") else { continue };
                let payload = data.trim();
                if payload == "[DONE]" {
                    saw_done = true;
                    continue;
                }
                match serde_json::from_str::<ChatCompletionStreamChunk>(payload) {
                    Ok(chunk) => {
                        if let Some(usage) = chunk.usage {
                            prompt_tokens = Some(usage.prompt_tokens);
                            completion_tokens = Some(usage.completion_tokens);
                        }
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(delta) = &choice.delta {
                                if let Some(piece) = &delta.content {
                                    if !piece.is_empty() {
                                        if ttft_ms.is_none() {
                                            ttft_ms = Some(started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32);
                                        }
                                        content.push_str(piece);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if parse_error.is_none() {
                            parse_error = Some(truncate(&e.to_string(), 200));
                        }
                    }
                }
            }
        }
    }

    let latency_ms = started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
    let success = parse_error.is_none() && saw_done;
    let response_shape_ok = if content.is_empty() { None } else { Some(validate_analysis_report_shape(&content)) };
    let tokens_per_sec = completion_tokens.and_then(|n| {
        if latency_ms == 0 { None } else { Some(n as f64 / (latency_ms as f64 / 1000.0)) }
    });

    telemetry.insert_benchmark_run(telemetry::BenchmarkRunInput {
        ts,
        mesh_id,
        target_node_id: None,
        model,
        probe_name: probe.name.into(),
        probe_type: "stream".into(),
        stream: true,
        temperature: probe.settings.temperature,
        top_p: probe.settings.top_p,
        max_tokens: probe.settings.max_tokens,
        prompt_hash: Some(fnv1a_hex(probe.prompt.as_bytes())),
        route_kind: None,
        success,
        status_code,
        latency_ms: Some(latency_ms),
        ttft_ms,
        prompt_tokens,
        completion_tokens,
        tokens_per_sec,
        error_kind: if success { None } else if parse_error.is_some() { Some("stream_parse_error".into()) } else { Some("stream_incomplete".into()) },
        error_message: parse_error,
        response_shape_ok,
    }).await?;

    Ok(())
}

async fn discover_model(client: &reqwest::Client, api_port: u16) -> Option<String> {
    let url = format!("http://127.0.0.1:{api_port}/v1/models");
    let resp = client.get(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().await.ok()?;
    let parsed: ModelsListResponse = serde_json::from_str(&text).ok()?;
    parsed.data.into_iter().map(|m| m.id).next()
}

fn validate_analysis_report_shape(content: &str) -> bool {
    let c = content.to_lowercase();
    (c.contains("executive summary") || c.contains("summary"))
        && c.contains("risk")
        && c.contains("recommend")
        && c.contains("conclusion")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        s[..max].to_string()
    }
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn unix_ts_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<ModelListItem>,
}

#[derive(Debug, Deserialize)]
struct ModelListItem {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChunk {
    #[serde(default)]
    choices: Vec<ChatStreamChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamChoice {
    #[serde(default)]
    delta: Option<ChatStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamDelta {
    #[serde(default)]
    content: Option<String>,
}
