//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is read-only — shows status, topology, models.
//! All mutations happen via CLI flags (--join, --model, --auto).

use crate::{download, election, mesh, nostr, plugin};
use include_dir::{include_dir, Dir};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");
const MESH_LLM_VERSION: &str = crate::VERSION;

// ── Shared state ──

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
}

struct ApiInner {
    node: mesh::Node,
    plugin_manager: plugin::PluginManager,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    mesh_name: Option<String>,
    latest_version: Option<String>,
    nostr_relays: Vec<String>,
    nostr_discovery: bool,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Serialize)]
struct GpuEntry {
    name: String,
    vram_bytes: u64,
}

fn build_gpus(gpu_name: Option<&str>, gpu_vram: Option<&str>) -> Vec<GpuEntry> {
    let names: Vec<&str> = gpu_name
        .map(|s| s.split(", ").collect())
        .unwrap_or_default();
    if names.is_empty() {
        return vec![];
    }
    let vrams: Vec<u64> = gpu_vram
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse::<u64>().ok())
                .collect()
        })
        .unwrap_or_default();
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name: name.to_string(),
            vram_bytes: vrams.get(i).copied().unwrap_or(0),
        })
        .collect()
}

#[derive(Serialize)]
struct StatusPayload {
    version: String,
    latest_version: Option<String>,
    node_id: String,
    token: String,
    node_status: String,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    model_name: String,
    serving_models: Vec<String>,
    draft_name: Option<String>,
    api_port: u16,
    my_vram_gb: f64,
    model_size_gb: f64,
    peers: Vec<PeerPayload>,
    launch_pi: Option<String>,
    launch_goose: Option<String>,
    mesh_models: Vec<MeshModelPayload>,
    inflight_requests: u64,
    /// Mesh identity (for matching against discovered meshes)
    mesh_id: Option<String>,
    /// Human-readable mesh name (from Nostr publishing)
    mesh_name: Option<String>,
    /// true when this node found the mesh via Nostr discovery (community/public mesh)
    nostr_discovery: bool,
    my_hostname: Option<String>,
    my_is_soc: Option<bool>,
    gpus: Vec<GpuEntry>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    vram_gb: f64,
    serving: Option<String>,
    serving_models: Vec<String>,
    rtt_ms: Option<u32>,
    hostname: Option<String>,
    is_soc: Option<bool>,
    gpus: Vec<GpuEntry>,
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    status: String,
    node_count: usize,
    size_gb: f64,
    /// Whether this model supports vision/image input
    vision: bool,
    /// Total requests seen across the mesh (from demand map)
    #[serde(skip_serializing_if = "Option::is_none")]
    request_count: Option<u64>,
    /// Seconds since last request or declaration (None if no demand data)
    #[serde(skip_serializing_if = "Option::is_none")]
    last_active_secs_ago: Option<u64>,
}

impl MeshApi {
    pub fn new(
        node: mesh::Node,
        model_name: String,
        api_port: u16,
        model_size_bytes: u64,
        plugin_manager: plugin::PluginManager,
    ) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                nostr_discovery: false,
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

    pub async fn set_nostr_discovery(&self, v: bool) {
        self.inner.lock().await.nostr_discovery = v;
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
        // Snapshot inner fields and drop the lock before any async node queries.
        // This prevents deadlock: if node.peers() etc. block on node.state.lock(),
        // we don't hold inner.lock() hostage, so other handlers can still proceed.
        let (node, node_id, token, my_vram_gb, inflight_requests,
             model_name, model_size_bytes, llama_ready, is_host, is_client,
             api_port, draft_name, mesh_name, latest_version, nostr_discovery) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.llama_ready,
                inner.is_host,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.nostr_discovery,
            )
        }; // inner lock dropped here

        let all_peers = node.peers().await;
        let peers: Vec<PeerPayload> = all_peers
            .iter()
            .map(|p| PeerPayload {
                id: p.id.fmt_short().to_string(),
                role: match p.role {
                    mesh::NodeRole::Worker => "Worker".into(),
                    mesh::NodeRole::Host { .. } => "Host".into(),
                    mesh::NodeRole::Client => "Client".into(),
                },
                models: p.models.clone(),
                vram_gb: p.vram_bytes as f64 / 1e9,
                serving: p.serving.clone(),
                serving_models: p.serving_models.clone(),
                rtt_ms: p.rtt_ms,
                hostname: p.hostname.clone(),
                is_soc: p.is_soc,
                gpus: build_gpus(p.gpu_name.as_deref(), p.gpu_vram.as_deref()),
            })
            .collect();

        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let active_demand = node.active_demand().await;
        let my_serving_models = node.serving_models().await;
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mesh_models: Vec<MeshModelPayload> = catalog.iter().map(|name| {
            let is_warm = served.contains(name);
            let node_count = if is_warm {
                let peer_count = all_peers.iter()
                    .filter(|p| {
                        p.serving_models.iter().any(|s| s == name)
                            || p.serving.as_deref() == Some(name.as_str())
                    })
                    .count();
                // Count self: check all serving models, fall back to primary model_name
                let me = if my_serving_models.iter().any(|s| s == name) || *name == model_name { 1 } else { 0 };
                peer_count + me
            } else {
                0
            };
            let size_gb = if *name == model_name && model_size_bytes > 0 {
                model_size_bytes as f64 / 1e9
            } else {
                download::parse_size_gb(
                    download::MODEL_CATALOG.iter()
                        .find(|m| m.file.strip_suffix(".gguf").unwrap_or(m.file) == name.as_str()
                            || m.name == name.as_str())
                        .map(|m| m.size)
                        .unwrap_or("0")
                )
            };
            let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                Some(d) => (
                    Some(d.request_count),
                    Some(now_ts.saturating_sub(d.last_active)),
                ),
                None => (None, None),
            };
            let vision = download::MODEL_CATALOG.iter()
                .find(|m| m.name == name.as_str()
                    || m.file.strip_suffix(".gguf").unwrap_or(m.file) == name.as_str())
                .map(|m| m.mmproj.is_some())
                .unwrap_or(false);
            MeshModelPayload {
                name: name.clone(),
                status: if is_warm { "warm".into() } else { "cold".into() },
                node_count,
                size_gb,
                vision,
                request_count,
                last_active_secs_ago,
            }
        }).collect();

        let (launch_pi, launch_goose) = if llama_ready {
            (
                Some(format!("pi --provider mesh --model {model_name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name} goose session")),
            )
        } else {
            (None, None)
        };

        let mesh_id = node.mesh_id().await;

        // Derive node status for display
        let node_status = if is_client {
            "Client".to_string()
        } else if is_host && llama_ready {
            let has_split_workers = all_peers.iter().any(|p|
                matches!(p.role, mesh::NodeRole::Worker) &&
                p.serving.as_deref() == Some(model_name.as_str())
            );
            if has_split_workers {
                "Serving (split)".to_string()
            } else {
                "Serving".to_string()
            }
        } else if !is_host && model_name != "(idle)" && !model_name.is_empty() {
            "Worker (split)".to_string()
        } else if model_name == "(idle)" || model_name.is_empty() {
            if all_peers.is_empty() {
                "Idle".to_string()
            } else {
                "Standby".to_string()
            }
        } else {
            "Standby".to_string()
        };

        StatusPayload {
            version: MESH_LLM_VERSION.to_string(),
            latest_version,
            node_id,
            token,
            node_status,
            is_host,
            is_client,
            llama_ready,
            model_name,
            serving_models: my_serving_models,
            draft_name,
            api_port,
            my_vram_gb,
            model_size_gb: model_size_bytes as f64 / 1e9,
            peers,
            launch_pi,
            launch_goose,
            mesh_models,
            inflight_requests,
            mesh_id,
            mesh_name,
            nostr_discovery,
            my_hostname: node.hostname.clone(),
            my_is_soc: node.is_soc,
            gpus: build_gpus(node.gpu_name.as_deref(), node.gpu_vram.as_deref()),
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
            if target_rx.changed().await.is_err() {
                break;
            }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port)
                | election::InferenceTarget::MoeLocal(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_)
                | election::InferenceTarget::MoeRemote(_) => {
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

    // Push status when peers join/leave.
    let mut peer_rx = {
        let inner = state.inner.lock().await;
        inner.node.peer_change_rx.clone()
    };
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            if peer_rx.changed().await.is_err() {
                break;
            }
            state3.push_status().await;
        }
    });

    // Push status when in-flight request count changes.
    let mut inflight_rx = {
        let inner = state.inner.lock().await;
        inner.node.inflight_change_rx()
    };
    let state4 = state.clone();
    tokio::spawn(async move {
        loop {
            if inflight_rx.changed().await.is_err() {
                break;
            }
            state4.push_status().await;
        }
    });

    // One-shot check for newer public release (for UI footer indicator).
    let state5 = state.clone();
    tokio::spawn(async move {
        let Some(latest) = crate::latest_release_version().await else {
            return;
        };
        if !crate::version_newer(&latest, crate::VERSION) {
            return;
        }
        {
            let mut inner = state5.inner.lock().await;
            inner.latest_version = Some(latest);
        }
        state5.push_status().await;
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
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
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
    let n = match tokio::time::timeout(std::time::Duration::from_secs(5), stream.read(&mut buf)).await {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return Err(e.into()),
        Err(_) => return Ok(()), // read timeout — health check probe, just close
    };
    if n == 0 {
        return Ok(()); // empty read — closed before sending
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let method = req.split_whitespace().next().unwrap_or("GET");
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let path_only = path.split('?').next().unwrap_or(path);

    match (method, path_only) {
        // ── Dashboard UI ──
        ("GET", "/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", "/dashboard") | ("GET", "/chat") | ("GET", "/dashboard/") | ("GET", "/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", p) if p.starts_with("/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        // ── Frontend static assets (bundled UI dist) ──
        ("GET", p) if p.starts_with("/assets/") || matches!(p.rsplit('.').next(), Some("png" | "ico" | "webmanifest")) || (p.ends_with(".json") && !p.starts_with("/api/")) => {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Discover meshes via Nostr ──
        ("GET", "/api/discover") => {
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
        ("GET", "/api/status") => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
                Ok(status) => {
                    let json = serde_json::to_string(&status)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(),
                        json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(_) => {
                    respond_error(&mut stream, 503, "Status temporarily unavailable").await?;
                }
            }
        }

        // ── SSE event stream ──
        ("GET", "/api/events") => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            let status = state.status().await;
            if let Ok(json) = serde_json::to_string(&status) {
                stream
                    .write_all(format!("data: {json}\n\n").as_bytes())
                    .await?;
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            state.inner.lock().await.sse_clients.push(tx);

            loop {
                tokio::select! {
                    event = rx.recv() => {
                        match event {
                            Some(data) => {
                                if stream.write_all(data.as_bytes()).await.is_err() {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                        // SSE keepalive comment to prevent proxy/browser timeout
                        if stream.write_all(b": keepalive\n\n").await.is_err() {
                            break;
                        }
                    }
                }
            }
        }

        // ── Plugins ──
        ("GET", "/api/plugins") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            let plugins = plugin_manager.list().await;
            let json = serde_json::to_string(&plugins)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            let rest = &p["/api/plugins/".len()..];
            let plugin_name = rest.trim_end_matches("/tools");
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            match plugin_manager.tools(plugin_name).await {
                Ok(tools) => {
                    let json = serde_json::to_string(&tools)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(),
                        json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 404, &e.to_string()).await?;
                }
            }
        }

        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            let rest = &p["/api/plugins/".len()..];
            if let Some((plugin_name, tool_name)) = rest.split_once("/tools/") {
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("{}");
                let payload = if body.trim().is_empty() { "{}" } else { body };
                let plugin_manager = state.inner.lock().await.plugin_manager.clone();
                match plugin_manager.call_tool(plugin_name, tool_name, payload).await {
                    Ok(result) if !result.is_error => {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            result.content_json.len(),
                            result.content_json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => {
                        respond_error(&mut stream, 502, &e.to_string()).await?;
                    }
                }
            } else {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Blackboard ──
        ("GET", "/api/blackboard/feed") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            if !plugin_manager.is_enabled(plugin::BLACKBOARD_PLUGIN_ID).await {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let query_str = path.split('?').nth(1).unwrap_or("");
                let params: Vec<(&str, &str)> = query_str
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .collect();
                let request = crate::blackboard::FeedRequest {
                    from: params.iter().find(|(k, _)| *k == "from").map(|(_, v)| (*v).to_string()),
                    limit: params
                        .iter()
                        .find(|(k, _)| *k == "limit")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(20),
                    since: params
                        .iter()
                        .find(|(k, _)| *k == "since")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0),
                };
                match plugin_manager
                    .call_tool(
                        plugin::BLACKBOARD_PLUGIN_ID,
                        "feed",
                        &serde_json::to_string(&request)?,
                    )
                    .await
                {
                    Ok(result) if !result.is_error => {
                        let items: Vec<crate::blackboard::BlackboardItem> =
                            serde_json::from_str(&result.content_json).unwrap_or_default();
                        let json = serde_json::to_string(&items).unwrap_or_else(|_| "[]".into());
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => respond_error(&mut stream, 502, &e.to_string()).await?,
                }
            }
        }

        ("GET", "/api/blackboard/search") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            if !plugin_manager.is_enabled(plugin::BLACKBOARD_PLUGIN_ID).await {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let query_str = path.split('?').nth(1).unwrap_or("");
                let params: Vec<(&str, &str)> = query_str
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .collect();
                let request = crate::blackboard::SearchRequest {
                    query: params
                        .iter()
                        .find(|(k, _)| *k == "q")
                        .map(|(_, v)| (*v).replace('+', " ").replace("%20", " "))
                        .unwrap_or_default(),
                    limit: params
                        .iter()
                        .find(|(k, _)| *k == "limit")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(20),
                    since: params
                        .iter()
                        .find(|(k, _)| *k == "since")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0),
                };
                match plugin_manager
                    .call_tool(
                        plugin::BLACKBOARD_PLUGIN_ID,
                        "search",
                        &serde_json::to_string(&request)?,
                    )
                    .await
                {
                    Ok(result) if !result.is_error => {
                        let items: Vec<crate::blackboard::BlackboardItem> =
                            serde_json::from_str(&result.content_json).unwrap_or_default();
                        let json = serde_json::to_string(&items).unwrap_or_else(|_| "[]".into());
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => respond_error(&mut stream, 502, &e.to_string()).await?,
                }
            }
        }

        ("POST", "/api/blackboard/post") => {
            let (node, plugin_manager) = {
                let inner = state.inner.lock().await;
                (inner.node.clone(), inner.plugin_manager.clone())
            };
            if !plugin_manager.is_enabled(plugin::BLACKBOARD_PLUGIN_ID).await {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);
                match parsed {
                    Ok(val) => {
                        let text = val["text"].as_str().unwrap_or("").to_string();
                        if text.is_empty() {
                            respond_error(&mut stream, 400, "Missing 'text' field").await?;
                        } else {
                            let request = crate::blackboard::PostRequest {
                                text,
                                from: node.peer_name().await,
                                peer_id: node.id().fmt_short().to_string(),
                            };
                            match plugin_manager
                                .call_tool(
                                    plugin::BLACKBOARD_PLUGIN_ID,
                                    "post",
                                    &serde_json::to_string(&request)?,
                                )
                                .await
                            {
                                Ok(result) if !result.is_error => {
                                    let json = result.content_json;
                                    let resp = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                        json.len(),
                                        json
                                    );
                                    stream.write_all(resp.as_bytes()).await?;
                                }
                                Ok(result) => {
                                    let status = if result.content_json.contains("Rate limited") {
                                        429
                                    } else {
                                        400
                                    };
                                    respond_error(&mut stream, status, &result.content_json).await?;
                                }
                                Err(e) => {
                                    let msg = e.to_string();
                                    let status = if msg.contains("Rate limited") { 429 } else { 400 };
                                    respond_error(&mut stream, status, &msg).await?;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        respond_error(&mut stream, 400, "Invalid JSON body").await?;
                    }
                }
            }
        }

        // ── Chat proxy (routes through inference API port) ──
        (m, p) if m != "POST" && p.starts_with("/api/chat") => {
            respond_error(&mut stream, 405, "Method Not Allowed").await?;
        }
        ("POST", p) if p.starts_with("/api/chat") => {
            let inner = state.inner.lock().await;
            if !inner.llama_ready && !inner.is_client {
                drop(inner);
                return respond_error(&mut stream, 503, "LLM not ready").await;
            }
            let port = inner.api_port;
            drop(inner);
            let target = format!("127.0.0.1:{port}");
            if let Ok(mut upstream) = TcpStream::connect(&target).await {
                let rewritten = req.replacen("/api/chat", "/v1/chat/completions", 1);
                upstream.write_all(rewritten.as_bytes()).await?;
                tokio::io::copy_bidirectional(&mut stream, &mut upstream).await?;
            } else {
                respond_error(&mut stream, 502, "Cannot reach LLM server").await?;
            }
        }

        _ => {
            respond_error(&mut stream, 404, "Not found").await?;
        }
    }
    Ok(())
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
        respond_bytes(
            stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            file.contents(),
        )
        .await?;
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
    // Hashed asset filenames (Vite output) are immutable — cache forever.
    // Non-hashed assets (favicon, manifest) get short cache.
    let cache_control = if rel.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    };
    respond_bytes_cached(stream, 200, "OK", content_type, cache_control, file.contents()).await?;
    Ok(true)
}

async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    respond_bytes_cached(stream, code, status, content_type, "no-cache", body).await
}

async fn respond_bytes_cached(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    cache_control: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: {cache_control}\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_gpus_both_none() {
        let result = build_gpus(None, None);
        assert!(result.is_empty(), "expected empty vec when no gpu_name");
    }

    #[test]
    fn test_build_gpus_single_no_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 0);
    }

    #[test]
    fn test_build_gpus_single_with_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), Some("34359738368")); // 32 GiB in bytes
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
    }

    #[test]
    fn test_build_gpus_multi_full_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(result[1].name, "NVIDIA RTX 3080");
        assert_eq!(result[1].vram_bytes, 10_737_418_240);
    }

    #[test]
    fn test_build_gpus_multi_partial_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(result[1].vram_bytes, 0, "missing VRAM entry should default to 0");
    }

    #[test]
    fn test_build_gpus_vram_no_gpu_name() {
        let result = build_gpus(None, Some("34359738368"));
        assert!(result.is_empty(), "no gpu_name means no entries even if vram present");
    }

    #[test]
    fn test_build_gpus_vram_whitespace_trimmed() {
        let result = build_gpus(Some("NVIDIA RTX 4090"), Some(" 25769803776 "));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vram_bytes, 25_769_803_776);
    }
}
