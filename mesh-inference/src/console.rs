//! Web console for interactive mesh management.
//!
//! Serves a single-page dashboard at http://localhost:PORT with:
//! - Live mesh status via SSE
//! - Join mesh, select model, auto-elect host
//! - Built-in chat to test the LLM

use crate::{election, launch, mesh, tunnel};
use anyhow::{Context, Result};
use mesh::NodeRole;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

static HTML: &str = include_str!("console.html");

/// Catalog of models available for auto-download.
struct CatalogModel {
    name: &'static str,
    file: &'static str,
    url: &'static str,
    size: &'static str,
    description: &'static str,
}

const MODEL_CATALOG: &[CatalogModel] = &[
    CatalogModel {
        name: "Qwen2.5-3B-Instruct-Q4_K_M",
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        size: "2GB",
        description: "Small & fast general chat",
    },
    CatalogModel {
        name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        size: "4.7GB",
        description: "Code generation & completion",
    },
    CatalogModel {
        name: "GLM-4.7-Flash-Q4_K_M",
        file: "GLM-4.7-Flash-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
        size: "17GB",
        description: "Large general chat with reasoning",
    },
];

/// Shared state for the console.
#[derive(Clone)]
struct ConsoleState {
    inner: Arc<Mutex<ConsoleInner>>,
}

struct ConsoleInner {
    node: Option<mesh::Node>,
    tunnel_mgr: Option<tunnel::Manager>,
    role: CurrentRole,
    /// True if this node is currently the elected host
    is_host: bool,
    llama_port: Option<u16>,
    llama_ready: bool,
    client_port: Option<u16>,
    rpc_port: Option<u16>,
    model_path: Option<PathBuf>,
    download_progress: Option<DownloadProgress>,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Clone, Serialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CurrentRole {
    Idle,
    /// Node is active: rpc-server running, participating in election
    Active,
    /// Lite client — no GPU, no model
    JustConnect,
}

#[derive(Clone, Serialize)]
struct DownloadProgress {
    url: String,
    downloaded: u64,
    total: Option<u64>,
}

/// JSON payload sent to SSE clients and GET /api/status.
#[derive(Serialize)]
struct StatusPayload {
    node_id: Option<String>,
    token: Option<String>,
    role: CurrentRole,
    role_label: String,
    is_host: bool,
    peers: Vec<PeerPayload>,
    models: Vec<ModelInfo>,
    model_path: Option<String>,
    model_name: Option<String>,
    llama_port: Option<u16>,
    llama_ready: bool,
    client_port: Option<u16>,
    download: Option<DownloadProgress>,
    has_binaries: bool,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    vram_gb: f64,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    path: String,
    size: String,
}

impl ConsoleState {
    fn new() -> Self {
        ConsoleState {
            inner: Arc::new(Mutex::new(ConsoleInner {
                node: None,
                tunnel_mgr: None,
                role: CurrentRole::Idle,
                is_host: false,
                llama_port: None,
                llama_ready: false,
                client_port: None,
                rpc_port: None,
                model_path: None,
                download_progress: None,
                sse_clients: Vec::new(),
            })),
        }
    }

    async fn status(&self) -> StatusPayload {
        let inner = self.inner.lock().await;
        let (node_id, token, peers) = if let Some(node) = &inner.node {
            let id = node.id().fmt_short().to_string();
            let token = node.invite_token();
            let peers = node.peers().await.into_iter().map(|p| PeerPayload {
                id: p.id.fmt_short().to_string(),
                role: match p.role {
                    NodeRole::Worker => "GPU Worker".into(),
                    NodeRole::Host { http_port } => format!("LLM Host (:{http_port})"),
                    NodeRole::Client => "Client".into(),
                },
                models: p.models,
                vram_gb: p.vram_bytes as f64 / 1e9,
            }).collect();
            (Some(id), Some(token), peers)
        } else {
            (None, None, Vec::new())
        };

        let role_label = match (&inner.role, inner.is_host) {
            (CurrentRole::Idle, _) => "Idle".into(),
            (CurrentRole::Active, true) => if inner.llama_ready { "Host (LLM ready)".into() } else { "Host (starting...)".into() },
            (CurrentRole::Active, false) => "Worker".into(),
            (CurrentRole::JustConnect, _) => "Client".into(),
        };
        let model_name = inner.model_path.as_ref().map(|p| {
            p.file_stem().unwrap_or_default().to_string_lossy().to_string()
        });

        StatusPayload {
            node_id,
            token,
            role: inner.role.clone(),
            role_label,
            is_host: inner.is_host,
            peers,
            models: scan_models(),
            model_path: inner.model_path.as_ref().map(|p| p.display().to_string()),
            model_name,
            llama_port: inner.llama_port,
            llama_ready: inner.llama_ready,
            client_port: inner.client_port,
            download: inner.download_progress.clone(),
            has_binaries: detect_binaries().is_some(),
        }
    }

    async fn broadcast_sse(&self, event: &str) {
        let mut inner = self.inner.lock().await;
        inner.sse_clients.retain(|tx| !tx.is_closed());
        for tx in &inner.sse_clients {
            let _ = tx.send(event.to_string());
        }
    }

    async fn push_status(&self) {
        let status = self.status().await;
        if let Ok(json) = serde_json::to_string(&status) {
            let event = format!("data: {json}\n\n");
            self.broadcast_sse(&event).await;
        }
    }
}

/// Scan ~/.models/ for GGUF files.
fn scan_models() -> Vec<ModelInfo> {
    let models_dir = dirs::home_dir()
        .map(|h| h.join(".models"))
        .unwrap_or_default();
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let size = std::fs::metadata(&path)
                    .map(|m| {
                        let gb = m.len() as f64 / (1024.0 * 1024.0 * 1024.0);
                        format!("{gb:.1}GB")
                    })
                    .unwrap_or_default();
                models.push(ModelInfo {
                    name,
                    path: path.display().to_string(),
                    size,
                });
            }
        }
    }
    models
}

/// Find llama.cpp binaries.
fn detect_binaries() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    if dir.join("rpc-server").exists() && dir.join("llama-server").exists() {
        return Some(dir.to_path_buf());
    }
    let dev = dir.join("../llama.cpp/build/bin");
    if dev.join("rpc-server").exists() && dev.join("llama-server").exists() {
        return Some(dev.canonicalize().ok()?);
    }
    let cargo = dir.join("../../../llama.cpp/build/bin");
    if cargo.join("rpc-server").exists() && cargo.join("llama-server").exists() {
        return Some(cargo.canonicalize().ok()?);
    }
    None
}

// Election logic is in election.rs (shared with CLI)

// ─── Main entry ─────────────────────────────────────────────────────

pub async fn run(port: u16, initial_join: Vec<String>) -> Result<()> {
    let state = ConsoleState::new();

    // Start mesh node + tunnel manager once — never restart.
    {
        let (node, channels) = mesh::Node::start(NodeRole::Client, &[], None).await?;
        let tunnel_mgr = tunnel::Manager::start(
            node.clone(),
            0, // no rpc-server yet
            channels.rpc,
            channels.http,
        ).await?;

        let model_names: Vec<String> = scan_models().iter().map(|m| m.name.clone()).collect();
        node.set_models(model_names).await;

        let mut inner = state.inner.lock().await;
        inner.node = Some(node.clone());
        inner.tunnel_mgr = Some(tunnel_mgr);
    }

    for token in &initial_join {
        let inner = state.inner.lock().await;
        if let Some(node) = &inner.node {
            match node.join(token).await {
                Ok(()) => eprintln!("Joined mesh via initial token"),
                Err(e) => eprintln!("Failed to join initial token: {e}"),
            }
        }
    }

    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).await
        .with_context(|| format!("Failed to bind console to port {port}"))?;

    eprintln!("┌─────────────────────────────────────────┐");
    eprintln!("│  mesh-inference console                  │");
    eprintln!("│  http://localhost:{port:<21}│", port = port);
    eprintln!("└─────────────────────────────────────────┘");

    let state_bg = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            state_bg.push_status().await;
        }
    });

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_http(stream, state).await {
                        tracing::debug!("HTTP handler error: {e}");
                    }
                });
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nShutting down...");
                break;
            }
        }
    }

    Ok(())
}

// ─── HTTP routing ───────────────────────────────────────────────────

async fn handle_http(mut stream: TcpStream, state: ConsoleState) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let first_line = request.lines().next().unwrap_or("");
    let (method, path) = parse_request_line(first_line);
    let body = request.split("\r\n\r\n").nth(1).unwrap_or("").to_string();

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => {
            respond_html(&mut stream, HTML).await?;
        }
        ("GET", "/api/status") => {
            let status = state.status().await;
            respond_json(&mut stream, &status).await?;
        }
        ("GET", "/api/events") => {
            handle_sse(stream, state).await?;
        }
        ("POST", "/api/join") => {
            handle_join(&mut stream, &state, &body).await?;
        }
        ("POST", "/api/download-model") => {
            handle_download_model(&mut stream, &state, &body).await?;
        }
        ("GET", "/api/catalog") => {
            let catalog: Vec<serde_json::Value> = MODEL_CATALOG.iter().map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "file": m.file,
                    "size": m.size,
                    "description": m.description,
                })
            }).collect();
            respond_json(&mut stream, &catalog).await?;
        }
        ("POST", "/api/start") => {
            handle_start(&mut stream, &state, &body).await?;
        }
        ("POST", "/api/just-connect") => {
            handle_just_connect(&mut stream, &state).await?;
        }
        ("POST", "/api/stop") => {
            handle_stop(&mut stream, &state).await?;
        }
        _ => {
            let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
            stream.write_all(resp.as_bytes()).await?;
        }
    }

    Ok(())
}

fn parse_request_line(line: &str) -> (String, String) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        (parts[0].to_string(), parts[1].to_string())
    } else {
        ("GET".to_string(), "/".to_string())
    }
}

async fn respond_html(stream: &mut TcpStream, html: &str) -> Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
        html.len(), html
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_json<T: Serialize>(stream: &mut TcpStream, data: &T) -> Result<()> {
    let json = serde_json::to_string(data)?;
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
        json.len(), json
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_ok(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let json = serde_json::json!({ "ok": true, "message": msg });
    respond_json(stream, &json).await
}

async fn respond_err(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let json = serde_json::json!({ "ok": false, "error": msg });
    let body = serde_json::to_string(&json)?;
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

// ─── SSE ────────────────────────────────────────────────────────────

async fn handle_sse(mut stream: TcpStream, state: ConsoleState) -> Result<()> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    let status = state.status().await;
    if let Ok(json) = serde_json::to_string(&status) {
        stream.write_all(format!("data: {json}\n\n").as_bytes()).await?;
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    {
        let mut inner = state.inner.lock().await;
        inner.sse_clients.push(tx);
    }

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                if stream.write_all(event.as_bytes()).await.is_err() { break; }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() { break; }
            }
        }
    }

    Ok(())
}

// ─── Handlers ───────────────────────────────────────────────────────

async fn handle_join(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let token = if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        v.get("token").and_then(|t| t.as_str()).unwrap_or("").to_string()
    } else {
        body.trim().trim_matches('"').to_string()
    };
    let token = token.trim();
    if token.is_empty() {
        return respond_err(stream, "No token provided").await;
    }

    let inner = state.inner.lock().await;
    let node = inner.node.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?;
    match node.join(token).await {
        Ok(()) => {
            drop(inner);
            state.push_status().await;
            respond_ok(stream, "Joined mesh").await
        }
        Err(e) => respond_err(stream, &format!("Failed to join: {e}")).await,
    }
}

/// POST /api/start — select model, start rpc-server, enter election loop.
async fn handle_start(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let model_path = req.get("model").and_then(|v| v.as_str()).unwrap_or("");
    let llama_port: u16 = req.get("port").and_then(|v| v.as_u64()).unwrap_or(8090) as u16;

    if model_path.is_empty() {
        return respond_err(stream, "No model selected").await;
    }

    let model = PathBuf::from(model_path);
    if !model.exists() {
        return respond_err(stream, "Model file not found").await;
    }

    let bin_dir = match detect_binaries() {
        Some(d) => d,
        None => return respond_err(stream, "llama.cpp binaries not found").await,
    };

    let mut inner = state.inner.lock().await;
    if inner.role != CurrentRole::Idle {
        return respond_err(stream, "Already running — stop first").await;
    }

    let node = inner.node.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?.clone();
    let tunnel_mgr = inner.tunnel_mgr.as_ref().ok_or_else(|| anyhow::anyhow!("Tunnel manager not started"))?.clone();

    // Start rpc-server — every active node contributes GPU
    let rpc_port = match launch::start_rpc_server(&bin_dir, None, Some(&model)).await {
        Ok(p) => p,
        Err(e) => return respond_err(stream, &format!("Failed to start rpc-server: {e}")).await,
    };
    tunnel_mgr.set_rpc_port(rpc_port);

    // Start as Worker; election will promote to Host if appropriate
    node.set_role(NodeRole::Worker).await;

    inner.rpc_port = Some(rpc_port);
    inner.model_path = Some(model.clone());
    inner.role = CurrentRole::Active;
    inner.llama_port = Some(llama_port);
    drop(inner);

    state.push_status().await;
    respond_ok(stream, &format!("Started — rpc-server on port {rpc_port}, entering election")).await?;

    // Spawn the election loop
    spawn_election(node, tunnel_mgr, rpc_port, bin_dir, model, state.clone());

    Ok(())
}

/// Spawn the election loop backed by the shared election module,
/// wired to update ConsoleState on every host/ready change.
fn spawn_election(
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    rpc_port: u16,
    bin_dir: PathBuf,
    model: PathBuf,
    state: ConsoleState,
) {
    let (target_tx, _target_rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
    // TODO: console should use target_rx for its own API proxying
    tokio::spawn(async move {
        let state2 = state.clone();
        election::election_loop(
            node, tunnel_mgr, rpc_port, bin_dir, model, None, 8, target_tx,
            move |is_host, llama_ready| {
                let state3 = state2.clone();
                tokio::spawn(async move {
                    {
                        let mut inner = state3.inner.lock().await;
                        inner.is_host = is_host;
                        inner.llama_ready = llama_ready;
                    }
                    state3.push_status().await;
                });
            },
        ).await;
    });
}

// ─── Download ───────────────────────────────────────────────────────

async fn handle_download_model(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let model_name = if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        v.get("model").and_then(|t| t.as_str()).unwrap_or("").to_string()
    } else {
        String::new()
    };

    let catalog_entry = if model_name.is_empty() {
        MODEL_CATALOG.first()
    } else {
        MODEL_CATALOG.iter().find(|m| m.name == model_name)
    };

    let entry = match catalog_entry {
        Some(e) => e,
        None => return respond_err(stream, &format!("Unknown model: {model_name}")).await,
    };

    let models_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("No home directory"))?
        .join(".models");
    let dest = models_dir.join(entry.file);

    if dest.exists() {
        return respond_ok(stream, "Model already exists").await;
    }

    let url = entry.url.to_string();
    let name = entry.name.to_string();

    let state2 = state.clone();
    tokio::spawn(async move {
        if let Err(e) = download_model_task(&state2, &models_dir, &dest, &url, &name).await {
            eprintln!("Model download failed: {e}");
            let mut inner = state2.inner.lock().await;
            inner.download_progress = None;
        }
        state2.push_status().await;
    });

    respond_ok(stream, &format!("Downloading {}", entry.name)).await
}

async fn download_model_task(state: &ConsoleState, dir: &std::path::Path, dest: &std::path::Path, url: &str, name: &str) -> Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    let tmp = dest.with_extension("gguf.part");

    eprintln!("Downloading {name} from {url}");

    let mut child = tokio::process::Command::new("curl")
        .args(["-L", "-o", tmp.to_str().unwrap(), "--progress-bar", url])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    {
        let mut inner = state.inner.lock().await;
        inner.download_progress = Some(DownloadProgress {
            url: url.to_string(),
            downloaded: 0,
            total: None,
        });
    }
    state.push_status().await;

    let tmp2 = tmp.clone();
    let state2 = state.clone();
    let progress_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let size = tokio::fs::metadata(&tmp2).await.map(|m| m.len()).unwrap_or(0);
            let mut inner = state2.inner.lock().await;
            if let Some(ref mut p) = inner.download_progress {
                p.downloaded = size;
            }
            drop(inner);
            state2.push_status().await;
        }
    });

    let status = child.wait().await?;
    progress_task.abort();

    if !status.success() {
        let _ = tokio::fs::remove_file(&tmp).await;
        anyhow::bail!("curl failed with status {status}");
    }

    tokio::fs::rename(&tmp, dest).await?;
    eprintln!("Model downloaded to {}", dest.display());

    {
        let mut inner = state.inner.lock().await;
        inner.download_progress = None;
    }
    state.push_status().await;
    Ok(())
}

// ─── Just Connect (lite client) ─────────────────────────────────────

async fn handle_just_connect(stream: &mut TcpStream, state: &ConsoleState) -> Result<()> {
    let inner = state.inner.lock().await;
    let node = inner.node.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?.clone();

    if inner.role != CurrentRole::Idle {
        return respond_err(stream, "Already running — stop first").await;
    }

    let peers = node.peers().await;
    let host = peers.iter().find(|p| matches!(p.role, NodeRole::Host { .. }));
    if host.is_none() {
        return respond_err(stream, "No host found in the mesh yet").await;
    }
    let host_id = host.unwrap().id;
    drop(inner);

    let local_port = 8080u16;
    let listener = match TcpListener::bind(format!("127.0.0.1:{local_port}")).await {
        Ok(l) => l,
        Err(_) => TcpListener::bind("127.0.0.1:0").await?,
    };
    let actual_port = listener.local_addr()?.port();

    {
        let mut inner = state.inner.lock().await;
        inner.role = CurrentRole::JustConnect;
        inner.client_port = Some(actual_port);
    }
    state.push_status().await;

    let node2 = node.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((tcp_stream, _)) => {
                    let _ = tcp_stream.set_nodelay(true);
                    let node = node2.clone();
                    tokio::spawn(async move {
                        match node.open_http_tunnel(host_id).await {
                            Ok((quic_send, quic_recv)) => {
                                let _ = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await;
                            }
                            Err(e) => {
                                tracing::warn!("HTTP tunnel failed: {e}");
                            }
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("Proxy accept error: {e}");
                    break;
                }
            }
        }
    });

    respond_ok(stream, &format!("Connected! API at http://localhost:{actual_port}")).await
}

// ─── Stop ───────────────────────────────────────────────────────────

async fn handle_stop(stream: &mut TcpStream, state: &ConsoleState) -> Result<()> {
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", "rpc-server"])
        .output().await;
    launch::kill_llama_server().await;

    let mut inner = state.inner.lock().await;
    if let Some(tunnel_mgr) = &inner.tunnel_mgr {
        tunnel_mgr.set_rpc_port(0);
    }
    if let Some(node) = &inner.node {
        node.set_role(NodeRole::Client).await;
    }
    inner.role = CurrentRole::Idle;
    inner.is_host = false;
    inner.llama_port = None;
    inner.llama_ready = false;
    inner.client_port = None;
    inner.rpc_port = None;
    drop(inner);

    state.push_status().await;
    respond_ok(stream, "Stopped").await
}
