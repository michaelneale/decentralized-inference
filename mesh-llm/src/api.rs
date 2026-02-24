//! Mesh management API — serves on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/join      — join a mesh by invite token
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — console HTML (optional UI)
//!
//! The console HTML is a thin client that calls these endpoints.
//! All functionality works without the HTML — use curl, scripts, etc.

use crate::{election, mesh, nostr};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_HTML: &str = include_str!("console.html");

// ── Shared state ──

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
    /// Channel for /api/join to signal the main loop.
    /// None if runtime joining isn't supported (already serving).
    pub join_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
}

struct ApiInner {
    node: mesh::Node,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    mesh_name: Option<String>,
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
}

impl MeshApi {
    pub fn new(node: mesh::Node, model_name: String, api_port: u16, model_size_bytes: u64) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                sse_clients: Vec::new(),
            })),
            join_tx: None,
        }
    }

    pub async fn set_model_name(&self, name: String) {
        self.inner.lock().await.model_name = name;
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

        let mesh_id = node.mesh_id().await;

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

    let listener = match TcpListener::bind(format!("127.0.0.1:{port}")).await {
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

    match path {
        // ── Console HTML ──
        "/" => {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                CONSOLE_HTML.len(), CONSOLE_HTML
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── Discover meshes via Nostr ──
        "/api/discover" => {
            let relays: Vec<String> = nostr::DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect();
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

        // ── Join a mesh ──
        "/api/join" => {
            if req.starts_with("GET") {
                return respond_error(&mut stream, 405, "Method Not Allowed").await;
            }
            let body = if let Some(pos) = req.find("\r\n\r\n") {
                req[pos + 4..].to_string()
            } else {
                String::new()
            };

            #[derive(serde::Deserialize)]
            struct JoinReq { token: String }

            match serde_json::from_str::<JoinReq>(&body) {
                Ok(jr) => {
                    if let Some(ref tx) = state.join_tx {
                        if tx.send(jr.token).is_ok() {
                            let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 24\r\n\r\n{\"ok\":true,\"joining\":true}";
                            stream.write_all(resp.as_bytes()).await?;
                        } else {
                            respond_error(&mut stream, 500, "Join channel closed").await?;
                        }
                    } else {
                        // Already serving — direct join (adds peer, no model reassignment)
                        let inner = state.inner.lock().await;
                        let node = inner.node.clone();
                        drop(inner);
                        if let Err(e) = node.join(&jr.token).await {
                            respond_error(&mut stream, 500, &format!("Failed to join: {e}")).await?;
                        } else {
                            let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"ok\":true}";
                            stream.write_all(resp.as_bytes()).await?;
                        }
                    }
                }
                Err(_) => respond_error(&mut stream, 400, "Invalid JSON").await?,
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
