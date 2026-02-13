//! Web console for interactive mesh management.
//!
//! Serves a single-page dashboard at http://localhost:PORT with:
//! - Live mesh status via SSE
//! - Join mesh, download model, choose role
//! - Built-in chat to test the LLM

use crate::{launch, mesh, tunnel};
use anyhow::{Context, Result};
use mesh::NodeRole;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

static HTML: &str = include_str!("console.html");

/// Models we know how to auto-download.
const DEFAULT_MODEL: &str = "GLM-4.7-Flash-Q4_K_M";
const DEFAULT_MODEL_FILE: &str = "GLM-4.7-Flash-Q4_K_M.gguf";
const DEFAULT_MODEL_URL: &str = "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf";
const DEFAULT_MODEL_SIZE: &str = "17GB";

/// Shared state for the console.
#[derive(Clone)]
struct ConsoleState {
    inner: Arc<Mutex<ConsoleInner>>,
}

struct ConsoleInner {
    node: Option<mesh::Node>,
    tunnel_mgr: Option<tunnel::Manager>,
    role: CurrentRole,
    llama_port: Option<u16>,
    llama_ready: bool,
    client_port: Option<u16>,
    rpc_port: Option<u16>,
    model_path: Option<PathBuf>,
    download_progress: Option<DownloadProgress>,
    /// SSE subscribers — each gets a sender
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Clone, Serialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CurrentRole {
    Idle,
    ShareGpu,
    RunLlm,
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
            }).collect();
            (Some(id), Some(token), peers)
        } else {
            (None, None, Vec::new())
        };

        let role_label = match &inner.role {
            CurrentRole::Idle => "Idle".into(),
            CurrentRole::ShareGpu => "GPU Worker".into(),
            CurrentRole::RunLlm => if inner.llama_ready { "LLM Host".into() } else { "Starting...".into() },
            CurrentRole::JustConnect => "Client".into(),
        };
        let model_name = inner.model_path.as_ref().map(|p| {
            p.file_stem().unwrap_or_default().to_string_lossy().to_string()
        });

        StatusPayload {
            node_id,
            token,
            role: inner.role.clone(),
            role_label,
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

/// Find llama.cpp binaries. Checks:
/// 1. Next to the mesh-inference binary (bundle/deploy layout)
/// 2. ../llama.cpp/build/bin/ relative to the binary (dev layout)
/// 3. ../../llama.cpp/build/bin/ relative to the binary (cargo target/release/ layout)
fn detect_binaries() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;

    // 1. Same directory (bundle)
    if dir.join("rpc-server").exists() && dir.join("llama-server").exists() {
        return Some(dir.to_path_buf());
    }

    // 2. Relative to binary: ../llama.cpp/build/bin/
    let dev = dir.join("../llama.cpp/build/bin");
    if dev.join("rpc-server").exists() && dev.join("llama-server").exists() {
        return Some(dev.canonicalize().ok()?);
    }

    // 3. From cargo target/release/: ../../../llama.cpp/build/bin/
    let cargo = dir.join("../../../llama.cpp/build/bin");
    if cargo.join("rpc-server").exists() && cargo.join("llama-server").exists() {
        return Some(cargo.canonicalize().ok()?);
    }

    None
}

pub async fn run(port: u16, initial_join: Vec<String>) -> Result<()> {
    let state = ConsoleState::new();

    // Start mesh node + tunnel manager once — never restart.
    // rpc_port starts at 0 (no rpc-server yet), updated when user picks a role.
    {
        let (node, channels) = mesh::Node::start(NodeRole::Client).await?;
        let tunnel_mgr = tunnel::Manager::start(
            node.clone(),
            0, // no rpc-server yet
            channels.rpc,
            None, // no llama-server HTTP yet
            channels.http,
        ).await?;

        // Set available models from disk so gossip broadcasts them
        let model_names: Vec<String> = scan_models().iter().map(|m| m.name.clone()).collect();
        node.set_models(model_names).await;

        let mut inner = state.inner.lock().await;
        inner.node = Some(node.clone());
        inner.tunnel_mgr = Some(tunnel_mgr);
    }

    // Join any initial tokens
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

    // Background: push status updates every 2s
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

/// Parse an HTTP request and route it.
async fn handle_http(mut stream: TcpStream, state: ConsoleState) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let first_line = request.lines().next().unwrap_or("");
    let (method, path) = parse_request_line(first_line);

    // Extract body for POST requests
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
            handle_download_model(&mut stream, &state).await?;
        }
        ("POST", "/api/share-gpu") => {
            handle_share_gpu(&mut stream, &state, &body).await?;
        }
        ("POST", "/api/run-llm") => {
            handle_run_llm(&mut stream, &state, &body).await?;
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
        html.len(),
        html
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_json<T: Serialize>(stream: &mut TcpStream, data: &T) -> Result<()> {
    let json = serde_json::to_string(data)?;
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
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
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

/// SSE: keep connection open, push status events.
async fn handle_sse(mut stream: TcpStream, state: ConsoleState) -> Result<()> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    // Send initial status
    let status = state.status().await;
    if let Ok(json) = serde_json::to_string(&status) {
        stream.write_all(format!("data: {json}\n\n").as_bytes()).await?;
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    {
        let mut inner = state.inner.lock().await;
        inner.sse_clients.push(tx);
    }

    // Stream events until client disconnects
    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                if stream.write_all(event.as_bytes()).await.is_err() {
                    break;
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {
                // Keep-alive
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn handle_join(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let token = body.trim().trim_matches('"');
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

async fn handle_download_model(stream: &mut TcpStream, state: &ConsoleState) -> Result<()> {
    let models_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("No home directory"))?
        .join(".models");
    let dest = models_dir.join(DEFAULT_MODEL_FILE);

    if dest.exists() {
        return respond_ok(stream, "Model already exists").await;
    }

    // Start download in background
    let state2 = state.clone();
    tokio::spawn(async move {
        if let Err(e) = download_model_task(&state2, &models_dir, &dest).await {
            eprintln!("Model download failed: {e}");
            let mut inner = state2.inner.lock().await;
            inner.download_progress = None;
        }
        state2.push_status().await;
    });

    respond_ok(stream, "Download started").await
}

async fn download_model_task(state: &ConsoleState, dir: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    let tmp = dest.with_extension("gguf.part");

    eprintln!("Downloading {DEFAULT_MODEL} from {DEFAULT_MODEL_URL}");

    // Use curl for the download (simpler than adding reqwest dependency)
    let mut child = tokio::process::Command::new("curl")
        .args(["-L", "-o", tmp.to_str().unwrap(), "--progress-bar", DEFAULT_MODEL_URL])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    // Poll file size for progress
    {
        let mut inner = state.inner.lock().await;
        inner.download_progress = Some(DownloadProgress {
            url: DEFAULT_MODEL_URL.to_string(),
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

async fn handle_share_gpu(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let model_path = req.get("model").and_then(|v| v.as_str()).unwrap_or("");
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

    // Start rpc-server
    let rpc_port = match launch::start_rpc_server(&bin_dir, None, Some(&model)).await {
        Ok(p) => p,
        Err(e) => return respond_err(stream, &format!("Failed to start rpc-server: {e}")).await,
    };

    // Update tunnel manager to forward inbound streams to rpc-server
    if let Some(tunnel_mgr) = &inner.tunnel_mgr {
        tunnel_mgr.set_rpc_port(rpc_port);
    }

    // Update gossip role — no node restart needed
    if let Some(node) = &inner.node {
        node.set_role(NodeRole::Worker).await;
    }

    inner.rpc_port = Some(rpc_port);
    inner.model_path = Some(model);
    inner.role = CurrentRole::ShareGpu;
    drop(inner);

    state.push_status().await;
    respond_ok(stream, &format!("Sharing GPU — rpc-server on port {rpc_port}")).await
}

async fn handle_run_llm(stream: &mut TcpStream, state: &ConsoleState, body: &str) -> Result<()> {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let model_path = req.get("model").and_then(|v| v.as_str()).unwrap_or("");
    let llama_port: u16 = req.get("port").and_then(|v| v.as_u64()).unwrap_or(8090) as u16;
    let tensor_split = req.get("tensor_split").and_then(|v| v.as_str()).map(|s| s.to_string());

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

    // Update role on existing node — no restart
    let node = inner.node.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?.clone();
    let tunnel_mgr = inner.tunnel_mgr.as_ref().ok_or_else(|| anyhow::anyhow!("Tunnel manager not started"))?.clone();
    node.set_role(NodeRole::Host { http_port: llama_port }).await;

    inner.model_path = Some(model.clone());
    inner.role = CurrentRole::RunLlm;
    inner.llama_port = Some(llama_port);
    drop(inner);

    state.push_status().await;
    respond_ok(stream, "Starting LLM...").await?;

    // Start llama-server in background — use any existing worker peers
    let state2 = state.clone();
    tokio::spawn(async move {
        let existing_peers = node.peers().await;
        let has_workers = existing_peers.iter().any(|p| matches!(p.role, NodeRole::Worker));
        if has_workers {
            eprintln!("Found worker(s) in mesh, waiting for tunnels...");
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tunnel_mgr.wait_for_peers(1),
            ).await;
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        } else {
            eprintln!("No workers in mesh — starting with local GPU only");
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        // Only use tunnel ports for Worker peers
        let worker_peers = node.peers().await;
        let worker_ids: Vec<_> = worker_peers.iter()
            .filter(|p| matches!(p.role, NodeRole::Worker))
            .map(|p| p.id)
            .collect();
        let all_ports = tunnel_mgr.peer_ports_map().await;
        let tunnel_ports: Vec<u16> = worker_ids.iter()
            .filter_map(|id| all_ports.get(id).copied())
            .collect();
        let n_peers = tunnel_ports.len();

        if n_peers > 0 {
            eprintln!("Got {n_peers} GPU worker(s), starting llama-server");
            let my_map = tunnel_mgr.peer_ports_map().await;
            let _ = node.broadcast_tunnel_map(my_map).await;
            let _ = node.wait_for_tunnel_maps(n_peers, std::time::Duration::from_secs(15)).await;
            let remote_maps = node.all_remote_tunnel_maps().await;
            tunnel_mgr.update_rewrite_map(&remote_maps).await;
        } else {
            eprintln!("No workers found — starting llama-server with local GPU only");
        }

        match launch::start_llama_server(
            &bin_dir, &model, llama_port, &tunnel_ports, tensor_split.as_deref(),
        ).await {
            Ok(()) => {
                eprintln!("llama-server ready: http://localhost:{llama_port}");
                let mut inner = state2.inner.lock().await;
                inner.llama_ready = true;
            }
            Err(e) => {
                eprintln!("Failed to start llama-server: {e}");
                let mut inner = state2.inner.lock().await;
                inner.role = CurrentRole::Idle;
                inner.llama_port = None;
                inner.llama_ready = false;
            }
        }
        state2.push_status().await;
    });

    Ok(())
}

async fn handle_just_connect(stream: &mut TcpStream, state: &ConsoleState) -> Result<()> {
    let inner = state.inner.lock().await;
    let node = inner.node.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?;
    let node = node.clone();

    if inner.role != CurrentRole::Idle {
        return respond_err(stream, "Already running — stop first").await;
    }

    // Check if there's a host in the mesh
    let peers = node.peers().await;
    let host = peers.iter().find(|p| matches!(p.role, NodeRole::Host { .. }));
    if host.is_none() {
        return respond_err(stream, "No host found in the mesh — someone needs to be running an LLM first").await;
    }
    let host = host.unwrap().clone();
    let host_id = host.id;
    drop(inner);

    // Bind local proxy port
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

    // Start tunnel proxy in background
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

async fn handle_stop(stream: &mut TcpStream, state: &ConsoleState) -> Result<()> {
    // Kill child processes
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", "rpc-server"])
        .output().await;
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", "llama-server"])
        .output().await;

    let mut inner = state.inner.lock().await;
    inner.role = CurrentRole::Idle;
    inner.llama_port = None;
    inner.llama_ready = false;
    inner.client_port = None;
    inner.rpc_port = None;
    inner.tunnel_mgr = None;
    drop(inner);

    state.push_status().await;
    respond_ok(stream, "Stopped").await
}
