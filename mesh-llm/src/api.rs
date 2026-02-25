//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — console HTML dashboard
//!
//! The console is read-only — shows status, topology, models.
//! All mutations happen via CLI flags (--join, --model, --auto).

use crate::{download, election, mesh, nostr, telemetry};
use include_dir::{include_dir, Dir};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");

// ── Shared state ──

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
}

struct ApiInner {
    node: mesh::Node,
    telemetry: telemetry::Telemetry,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    mesh_name: Option<String>,
    nostr_relays: Vec<String>,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Serialize)]
struct StatusPayload {
    node_id: String,
    token: String,
    node_status: String,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    my_vram_gb: f64,
    model_size_gb: f64,
    peers: Vec<PeerPayload>,
    launch_pi: Option<String>,
    launch_goose: Option<String>,
    mesh_models: Vec<MeshModelPayload>,
    /// Mesh identity (for matching against discovered meshes)
    mesh_id: Option<String>,
    /// Human-readable mesh name (from Nostr publishing)
    mesh_name: Option<String>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    vram_gb: f64,
    serving: Option<String>,
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    status: String,
    node_count: usize,
    size_gb: f64,
}

#[derive(Serialize)]
struct TelemetryEventsPayload {
    live: telemetry::LiveSnapshot,
    nodes: Vec<telemetry::NodeMetricRow>,
    rollup: Vec<telemetry::RollupMetricRow>,
    node_history: Vec<telemetry::NodeMetricRow>,
    benchmarks: Vec<telemetry::BenchmarkRunRow>,
}

#[derive(serde::Deserialize)]
struct ChatSampleIngest {
    #[serde(default)]
    ttft_ms: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    tokens_per_sec: Option<f64>,
}

impl MeshApi {
    pub fn new(
        node: mesh::Node,
        telemetry: telemetry::Telemetry,
        model_name: String,
        api_port: u16,
        model_size_bytes: u64,
    ) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                telemetry,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                nostr_relays: nostr::DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect(),
                sse_clients: Vec::new(),
            })),

        }
    }


    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    pub async fn set_client(&self, is_client: bool) {
        self.inner.lock().await.is_client = is_client;
    }

    pub async fn set_mesh_name(&self, name: String) {
        self.inner.lock().await.mesh_name = Some(name);
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
        }
        self.push_status().await;
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        self.inner.lock().await.llama_port = port;
    }

    async fn status(&self) -> StatusPayload {
        let inner = self.inner.lock().await;
        let node = &inner.node;
        let node_id = node.id().fmt_short().to_string();
        let token = node.invite_token();
        let my_vram_gb = node.vram_bytes() as f64 / 1e9;

        let all_peers = node.peers().await;
        let peers: Vec<PeerPayload> = all_peers.iter().map(|p| PeerPayload {
            id: p.id.fmt_short().to_string(),
            role: match p.role {
                mesh::NodeRole::Worker => "Worker".into(),
                mesh::NodeRole::Host { http_port } => format!("Host (:{http_port})"),
                mesh::NodeRole::Client => "Client".into(),
            },
            models: p.models.clone(),
            vram_gb: p.vram_bytes as f64 / 1e9,
            serving: p.serving.clone(),
        }).collect();

        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let my_serving = inner.model_name.clone();
        let mesh_models: Vec<MeshModelPayload> = catalog.iter().map(|name| {
            let is_warm = served.contains(name);
            let node_count = if is_warm {
                let peer_count = all_peers.iter()
                    .filter(|p| p.serving.as_deref() == Some(name.as_str()))
                    .count();
                let me = if *name == my_serving { 1 } else { 0 };
                peer_count + me
            } else {
                0
            };
            // Model size: use local knowledge if it's our model, otherwise catalog
            let size_gb = if *name == my_serving && inner.model_size_bytes > 0 {
                inner.model_size_bytes as f64 / 1e9
            } else {
                download::parse_size_gb(
                    download::MODEL_CATALOG.iter()
                        .find(|m| m.file.strip_suffix(".gguf").unwrap_or(m.file) == name.as_str()
                            || m.name == name.as_str())
                        .map(|m| m.size)
                        .unwrap_or("0")
                )
            };
            MeshModelPayload {
                name: name.clone(),
                status: if is_warm { "warm".into() } else { "cold".into() },
                node_count,
                size_gb,
            }
        }).collect();

        let (launch_pi, launch_goose) = if inner.llama_ready {
            let name = &inner.model_name;
            let port = inner.api_port;
            (
                Some(format!("pi --provider mesh --model {name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{port} OPENAI_API_KEY=mesh GOOSE_MODEL={name} goose session")),
            )
        } else { (None, None) };

        let mesh_id = node.mesh_id().await;

        // Derive node status for display
        let node_status = if inner.is_client {
            "Client".to_string()
        } else if inner.is_host && inner.llama_ready {
            // Check if any peers are workers in our split (serving same model, role=Worker)
            let has_split_workers = all_peers.iter().any(|p|
                matches!(p.role, mesh::NodeRole::Worker) &&
                p.serving.as_deref() == Some(inner.model_name.as_str())
            );
            if has_split_workers {
                "Serving (split)".to_string()
            } else {
                "Serving".to_string()
            }
        } else if !inner.is_host && inner.model_name != "(idle)" && inner.model_name != "" {
            // We have a model assigned but aren't host — we're a worker in someone's split
            "Worker (split)".to_string()
        } else if inner.model_name == "(idle)" || inner.model_name == "" {
            if all_peers.is_empty() {
                "Idle".to_string()
            } else {
                "Standby".to_string()
            }
        } else {
            "Standby".to_string()
        };

        StatusPayload {
            node_id,
            token,
            node_status,
            is_host: inner.is_host,
            is_client: inner.is_client,
            llama_ready: inner.llama_ready,
            model_name: inner.model_name.clone(),
            draft_name: inner.draft_name.clone(),
            api_port: inner.api_port,
            my_vram_gb,
            model_size_gb: inner.model_size_bytes as f64 / 1e9,
            peers,
            launch_pi,
            launch_goose,
            mesh_models,
            mesh_id,
            mesh_name: inner.mesh_name.clone(),
        }
    }

    async fn push_status(&self) {
        let status = self.status().await;
        if let Ok(json) = serde_json::to_string(&status) {
            let event = format!("data: {json}\n\n");
            let mut inner = self.inner.lock().await;
            inner.sse_clients.retain(|tx| !tx.is_closed());
            for tx in &inner.sse_clients {
                let _ = tx.send(event.clone());
            }
        }
    }
}

// ── Server ──

/// Start the mesh management API server.
pub async fn start(
    port: u16,
    state: MeshApi,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
    listen_all: bool,
) {
    // Watch election target changes
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            if target_rx.changed().await.is_err() { break; }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_) => {
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None;
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
            state2.push_status().await;
        }
    });

    // Periodic status push (picks up peer changes)
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            state3.push_status().await;
        }
    });

    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Management API: failed to bind :{port}: {e}");
            return;
        }
    };
    tracing::info!("Management API on http://localhost:{port}");

    loop {
        let Ok((stream, _)) = listener.accept().await else { continue };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_request(stream, &state).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

async fn handle_request(mut stream: TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let path_only = path.split('?').next().unwrap_or(path);

    match path_only {
        // ── Console HTML ──
        "/" => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Console bundle missing").await?;
            }
        }

        // ── Frontend static assets ──
        p if p.starts_with("/assets/") => {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Discover meshes via Nostr ──
        "/api/discover" => {
            let relays = state.inner.lock().await.nostr_relays.clone();
            let filter = nostr::MeshFilter::default();
            match nostr::discover(&relays, &filter).await {
                Ok(meshes) => {
                    if let Ok(json) = serde_json::to_string(&meshes) {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(), json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    } else {
                        respond_error(&mut stream, 500, "Failed to serialize").await?;
                    }
                }
                Err(e) => {
                    respond_error(&mut stream, 500, &format!("Discovery failed: {e}")).await?;
                }
            }
        }

        // ── Live status ──
        "/api/status" => {
            let status = state.status().await;
            let json = serde_json::to_string(&status)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(), json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── Telemetry: local live snapshot ──
        "/api/metrics/live" => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let json = serde_json::to_string(&telemetry.snapshot())?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(), json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── Telemetry: ingest chat generation sample (console client) ──
        "/api/metrics/chat-sample" => {
            if !req.starts_with("POST ") {
                respond_error(&mut stream, 405, "Method not allowed").await?;
            } else if let Some(body) = http_body(&req) {
                let telemetry = state.inner.lock().await.telemetry.clone();
                match serde_json::from_str::<ChatSampleIngest>(body) {
                    Ok(sample) => {
                        telemetry.record_generation_metrics(sample.ttft_ms, sample.completion_tokens, sample.tokens_per_sec);
                        let resp = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n";
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Err(_) => respond_error(&mut stream, 400, "Invalid JSON").await?,
                }
            } else {
                respond_error(&mut stream, 400, "Missing body").await?;
            }
        }

        // ── Telemetry SSE stream (full package for UI) ──
        p if p.starts_with("/api/metrics/events") => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;
            let telemetry = state.inner.lock().await.telemetry.clone();
            let minutes = query_param_u32(path, "minutes").unwrap_or(180).clamp(1, 7 * 24 * 60);
            let node_id = query_param(path, "id").map(|s| s.to_string());
            let limit = query_param_u32(path, "limit").unwrap_or(300).clamp(1, 2000);
            loop {
                let node_history = if let Some(id) = node_id.clone() {
                    telemetry.node_history_for(id, minutes).await
                } else {
                    telemetry.node_history(minutes).await
                };
                let payload = match (
                    telemetry.all_nodes_latest(minutes).await,
                    telemetry.rollup_history(minutes).await,
                    node_history,
                    telemetry.benchmark_history(minutes, limit).await,
                ) {
                    (Ok(nodes), Ok(rollup), Ok(node_history), Ok(benchmarks)) => TelemetryEventsPayload {
                        live: telemetry.snapshot(),
                        nodes,
                        rollup,
                        node_history,
                        benchmarks,
                    },
                    _ => {
                        if stream.write_all(b"event: error\ndata: {\"error\":\"telemetry query failed\"}\n\n").await.is_err() {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                };
                let json = serde_json::to_string(&payload)?;
                if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }

        // ── Telemetry: local node history (SQLite) ──
        p if p.starts_with("/api/metrics/node") => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let minutes = query_param_u32(path, "minutes").unwrap_or(60).clamp(1, 24 * 60);
            let node_id = query_param(path, "id").map(|s| s.to_string());
            let result = if let Some(id) = node_id {
                telemetry.node_history_for(id, minutes).await
            } else {
                telemetry.node_history(minutes).await
            };
            match result {
                Ok(rows) => {
                    let json = serde_json::to_string(&rows)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(), json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => respond_error(&mut stream, 500, &format!("Telemetry query failed: {e}")).await?,
            }
        }

        // ── Telemetry: latest row per known node (SQLite, local DB) ──
        p if p.starts_with("/api/metrics/nodes") => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let minutes = query_param_u32(path, "minutes").unwrap_or(180).clamp(1, 7 * 24 * 60);
            match telemetry.all_nodes_latest(minutes).await {
                Ok(rows) => {
                    let json = serde_json::to_string(&rows)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(), json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => respond_error(&mut stream, 500, &format!("Telemetry query failed: {e}")).await?,
            }
        }

        // ── Telemetry: rollup history across rows in local DB ──
        p if p.starts_with("/api/metrics/rollup") => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let minutes = query_param_u32(path, "minutes").unwrap_or(60).clamp(1, 7 * 24 * 60);
            match telemetry.rollup_history(minutes).await {
                Ok(rows) => {
                    let json = serde_json::to_string(&rows)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(), json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => respond_error(&mut stream, 500, &format!("Telemetry query failed: {e}")).await?,
            }
        }

        // ── Telemetry: benchmark run history (raw local rows, recent only) ──
        p if p.starts_with("/api/metrics/benchmarks") => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let minutes = query_param_u32(path, "minutes").unwrap_or(60).clamp(1, 7 * 24 * 60);
            let limit = query_param_u32(path, "limit").unwrap_or(200).clamp(1, 2000);
            match telemetry.benchmark_history(minutes, limit).await {
                Ok(rows) => {
                    let json = serde_json::to_string(&rows)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(), json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => respond_error(&mut stream, 500, &format!("Telemetry query failed: {e}")).await?,
            }
        }

        // ── SSE event stream ──
        "/api/events" => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            let status = state.status().await;
            if let Ok(json) = serde_json::to_string(&status) {
                stream.write_all(format!("data: {json}\n\n").as_bytes()).await?;
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            state.inner.lock().await.sse_clients.push(tx);

            while let Some(event) = rx.recv().await {
                if stream.write_all(event.as_bytes()).await.is_err() {
                    break;
                }
            }
        }

        // ── Chat proxy (routes through inference API port) ──
        p if p.starts_with("/api/chat") => {
            let telemetry = state.inner.lock().await.telemetry.clone();
            let span = telemetry.start_request(telemetry::RouteKind::Local);
            let inner = state.inner.lock().await;
            if !inner.llama_ready && !inner.is_client {
                drop(inner);
                let res = respond_error(&mut stream, 503, "LLM not ready").await;
                span.finish(false);
                return res;
            }
            let port = inner.api_port;
            drop(inner);
            let target = format!("127.0.0.1:{port}");
            if let Ok(mut upstream) = TcpStream::connect(&target).await {
                let rewritten = req.replacen("/api/chat", "/v1/chat/completions", 1);
                upstream.write_all(rewritten.as_bytes()).await?;
                let result = tokio::io::copy_bidirectional(&mut stream, &mut upstream).await;
                span.finish(result.is_ok());
                result?;
            } else {
                let res = respond_error(&mut stream, 502, "Cannot reach LLM server").await;
                span.finish(false);
                res?;
            }
        }

        _ => {
            respond_error(&mut stream, 404, "Not found").await?;
        }
    }
    Ok(())
}

fn query_param_u32(path: &str, key: &str) -> Option<u32> {
    query_param(path, key)?.parse::<u32>().ok()
}

fn query_param<'a>(path: &'a str, key: &str) -> Option<&'a str> {
    let (_, query) = path.split_once('?')?;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=')?;
        if k == key {
            return Some(v);
        }
    }
    None
}

fn http_body(req: &str) -> Option<&str> {
    req.split_once("\r\n\r\n").map(|(_, b)| b)
}

async fn respond_error(stream: &mut TcpStream, code: u16, msg: &str) -> anyhow::Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let status = match code {
        400 => "Bad Request",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Not Found",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_console_index(stream: &mut TcpStream) -> anyhow::Result<bool> {
    if let Some(file) = CONSOLE_DIST.get_file("index.html") {
        respond_bytes(stream, 200, "OK", "text/html; charset=utf-8", file.contents()).await?;
        return Ok(true);
    }
    Ok(false)
}

async fn respond_console_asset(stream: &mut TcpStream, path: &str) -> anyhow::Result<bool> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return Ok(false);
    }
    let Some(file) = CONSOLE_DIST.get_file(rel) else {
        return Ok(false);
    };
    let content_type = match rel.rsplit('.').next().unwrap_or("") {
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    };
    respond_bytes(stream, 200, "OK", content_type, file.contents()).await?;
    Ok(true)
}

async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}
