//! Web console — thin status viewer bolted onto the running CLI process.
//!
//! Serves a single-page dashboard with:
//! - Live mesh status via SSE (peers, VRAM, roles)
//! - Cluster visualization
//! - Agent launch commands (pi, goose)
//! - Built-in chat proxy to the running LLM

use crate::{election, mesh};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static HTML: &str = include_str!("console.html");

/// Shared live state — written by the main process, read by the console.
#[derive(Clone)]
pub struct ConsoleState {
    inner: Arc<Mutex<ConsoleInner>>,
}

struct ConsoleInner {
    node: mesh::Node,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Serialize)]
struct StatusPayload {
    node_id: String,
    token: String,
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
    /// All models in the mesh with their status
    mesh_models: Vec<MeshModelPayload>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    vram_gb: f64,
    /// What this peer is currently serving
    serving: Option<String>,
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    /// "warm" = loaded and serving, "cold" = available on disk but not loaded
    status: String,
    /// Number of nodes serving this model
    node_count: usize,
}

impl ConsoleState {
    pub fn new(node: mesh::Node, model_name: String, api_port: u16, model_size_bytes: u64) -> Self {
        ConsoleState {
            inner: Arc::new(Mutex::new(ConsoleInner {
                node,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                draft_name: None,
                api_port,
                model_size_bytes,
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

        // Build mesh model list: warm (being served) and cold (on disk only)
        // Note: mesh_catalog and models_being_served need the node's inner locks,
        // but we already hold inner (ConsoleInner). The Node locks are separate, so this is fine.
        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let my_serving = inner.model_name.clone(); // what this node is serving
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
            MeshModelPayload {
                name: name.clone(),
                status: if is_warm { "warm".into() } else { "cold".into() },
                node_count,
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

        StatusPayload {
            node_id,
            token,
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

/// Start the console web server. Call this from run_auto().
pub async fn start(
    port: u16,
    state: ConsoleState,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
) {
    // Watch target changes to update llama_port and readiness
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
                    // Worker with host available — API proxy works
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None; // no local llama, chat goes through api_port
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
            state2.push_status().await;
        }
    });

    // Also push status every 2s for peer changes
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            state3.push_status().await;
        }
    });

    let listener = match TcpListener::bind(format!("127.0.0.1:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Console: failed to bind :{port}: {e}");
            return;
        }
    };
    tracing::info!("Console listening on http://localhost:{port}");

    loop {
        let Ok((stream, _)) = listener.accept().await else { continue };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, &state).await {
                tracing::debug!("Console connection error: {e}");
            }
        });
    }
}

async fn handle_connection(mut stream: TcpStream, state: &ConsoleState) -> anyhow::Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req.split_whitespace().nth(1).unwrap_or("/");

    match path {
        "/" => {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                HTML.len(), HTML
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        "/api/status" => {
            let status = state.status().await;
            let json = serde_json::to_string(&status)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(), json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        "/api/events" => {
            // SSE
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            // Send current state immediately
            let status = state.status().await;
            if let Ok(json) = serde_json::to_string(&status) {
                let event = format!("data: {json}\n\n");
                stream.write_all(event.as_bytes()).await?;
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            state.inner.lock().await.sse_clients.push(tx);

            while let Some(event) = rx.recv().await {
                if stream.write_all(event.as_bytes()).await.is_err() {
                    break;
                }
            }
        }
        p if p.starts_with("/api/chat") => {
            // Always route through the API port — it handles model-based
            // routing (local llama-server OR remote via QUIC tunnel)
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
