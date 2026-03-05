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

use crate::{download, election, mesh, nostr};
use include_dir::{include_dir, Dir};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

mod hub;

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
    hub_first_time_onboarding: bool,
    hub: hub::HubState,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
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
    draft_name: Option<String>,
    api_port: u16,
    my_vram_gb: f64,
    model_size_gb: f64,
    peers: Vec<PeerPayload>,
    launch_pi: Option<String>,
    launch_goose: Option<String>,
    mesh_models: Vec<MeshModelPayload>,
    top_catalog_models: Vec<CatalogTopModelPayload>,
    inflight_requests: u64,
    /// Mesh identity (for matching against discovered meshes)
    mesh_id: Option<String>,
    /// Human-readable mesh name (from Nostr publishing)
    mesh_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_auth_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_avatar_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_profile_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_meshes_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_mesh_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_mesh_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_mesh_slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_mesh_visibility: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh_link_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    membership_enforcement: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_default_mesh_selector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_default_invite_configured: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_first_time_onboarding: Option<bool>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    vram_gb: f64,
    serving: Option<String>,
    rtt_ms: Option<u32>,
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    status: String,
    node_count: usize,
    size_gb: f64,
    /// Total requests seen across the mesh (from demand map)
    #[serde(skip_serializing_if = "Option::is_none")]
    request_count: Option<u64>,
    /// Seconds since last request or declaration (None if no demand data)
    #[serde(skip_serializing_if = "Option::is_none")]
    last_active_secs_ago: Option<u64>,
}

#[derive(Serialize)]
struct CatalogTopModelPayload {
    name: String,
    size_gb: f64,
    description: String,
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
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                hub_first_time_onboarding: false,
                hub: hub::load_state(),
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

    async fn request_catalog_model(&self, requested: &str) -> anyhow::Result<String> {
        let trimmed = requested.trim();
        if trimmed.is_empty() {
            anyhow::bail!("model is required");
        }
        let entry = download::find_model(trimmed)
            .ok_or_else(|| anyhow::anyhow!("model not found in mesh-llm catalog: {trimmed}"))?;
        let canonical = entry.name.to_string();
        let node = self.inner.lock().await.node.clone();
        node.record_request(&canonical);
        Ok(canonical)
    }

    async fn status(&self, include_demand: bool) -> StatusPayload {
        // Snapshot inner fields and drop the lock before any async node queries.
        // This prevents deadlock: if node.peers() etc. block on node.state.lock(),
        // we don't hold inner.lock() hostage, so other handlers can still proceed.
        let (
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            model_name,
            model_size_bytes,
            llama_ready,
            is_host,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            hub_first_time_onboarding,
            hub_state,
        ) = {
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
                inner.hub_first_time_onboarding,
                inner.hub.clone(),
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
                vram_gb: p.vram_bytes as f64 / 1e9,
                serving: p.serving.clone(),
                rtt_ms: p.rtt_ms,
            })
            .collect();

        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let active_demand = if include_demand {
            Some(node.active_demand().await)
        } else {
            None
        };
        let now_ts = if include_demand {
            Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
        } else {
            None
        };
        let mesh_models: Vec<MeshModelPayload> = catalog
            .iter()
            .map(|name| {
                let is_warm = served.contains(name);
                let node_count = if is_warm {
                    let peer_count = all_peers
                        .iter()
                        .filter(|p| p.serving.as_deref() == Some(name.as_str()))
                        .count();
                    let me = if *name == model_name { 1 } else { 0 };
                    peer_count + me
                } else {
                    0
                };
                let size_gb = if *name == model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    download::parse_size_gb(
                        download::MODEL_CATALOG
                            .iter()
                            .find(|m| {
                                m.file.strip_suffix(".gguf").unwrap_or(m.file) == name.as_str()
                                    || m.name == name.as_str()
                            })
                            .map(|m| m.size)
                            .unwrap_or("0"),
                    )
                };
                let (request_count, last_active_secs_ago) =
                    match active_demand.as_ref().and_then(|d| d.get(name)) {
                        Some(d) => (
                            Some(d.request_count),
                            Some(now_ts.unwrap_or_default().saturating_sub(d.last_active)),
                        ),
                        None => (None, None),
                    };
                MeshModelPayload {
                    name: name.clone(),
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    size_gb,
                    request_count,
                    last_active_secs_ago,
                }
            })
            .collect();
        let draft_model_names: HashSet<&str> = download::MODEL_CATALOG
            .iter()
            .filter_map(|m| m.draft)
            .collect();
        let mut top_catalog_models: Vec<CatalogTopModelPayload> = download::MODEL_CATALOG
            .iter()
            .filter(|m| !draft_model_names.contains(m.name))
            .filter_map(|m| {
                let size_gb = download::parse_size_gb(m.size);
                let fits_vram = my_vram_gb <= 0.0 || size_gb <= (my_vram_gb * 1.1);
                if !fits_vram {
                    return None;
                }
                Some(CatalogTopModelPayload {
                    name: m.name.to_string(),
                    size_gb,
                    description: m.description.to_string(),
                })
            })
            .take(8)
            .collect();
        if top_catalog_models.is_empty() {
            top_catalog_models = download::MODEL_CATALOG
                .iter()
                .filter(|m| !draft_model_names.contains(m.name))
                .take(8)
                .map(|m| CatalogTopModelPayload {
                    name: m.name.to_string(),
                    size_gb: download::parse_size_gb(m.size),
                    description: m.description.to_string(),
                })
                .collect();
        }

        let (launch_pi, launch_goose) = if llama_ready {
            (
                Some(format!("pi --provider mesh --model {model_name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name} goose session")),
            )
        } else {
            (None, None)
        };

        let mesh_id = node.mesh_id().await;
        let hub_profile_url = format!("{}/profile", hub_state.base_url);
        let hub_meshes_url = format!("{}/my-meshes", hub_state.base_url);
        let hub_mesh_url = if let Some(slug) = hub_state
            .linked_mesh_slug
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            Some(format!("{}/meshes/{}", hub_state.base_url, slug))
        } else if hub_state.linked_mesh_id.is_some() {
            Some(format!("{}/my-meshes", hub_state.base_url))
        } else {
            None
        };
        let hub_auth_state = if hub_state.access_token.is_some() {
            "logged_in".to_string()
        } else if hub_state.auth_pending {
            "auth_pending".to_string()
        } else {
            "logged_out".to_string()
        };
        let status_token = if hub_state.membership_enforcement == "hub_enforced" {
            String::new()
        } else {
            token
        };

        // Derive node status for display
        let node_status = if is_client {
            "Client".to_string()
        } else if is_host && llama_ready {
            let has_split_workers = all_peers.iter().any(|p| {
                matches!(p.role, mesh::NodeRole::Worker)
                    && p.serving.as_deref() == Some(model_name.as_str())
            });
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
            token: status_token,
            node_status,
            is_host,
            is_client,
            llama_ready,
            model_name,
            draft_name,
            api_port,
            my_vram_gb,
            model_size_gb: model_size_bytes as f64 / 1e9,
            peers,
            launch_pi,
            launch_goose,
            mesh_models,
            top_catalog_models,
            inflight_requests,
            mesh_id,
            mesh_name,
            hub_base_url: Some(hub_state.base_url),
            hub_auth_state: Some(hub_auth_state),
            hub_user_name: hub_state.profile.as_ref().and_then(|p| p.name.clone()),
            hub_user_handle: hub_state.profile.as_ref().and_then(|p| p.handle.clone()),
            hub_user_avatar_url: hub_state
                .profile
                .as_ref()
                .and_then(|p| p.avatar_url.clone()),
            hub_profile_url: Some(hub_profile_url),
            hub_meshes_url: Some(hub_meshes_url),
            hub_mesh_url,
            hub_mesh_name: hub_state.linked_mesh_name,
            hub_mesh_slug: hub_state.linked_mesh_slug,
            hub_mesh_visibility: hub_state.linked_mesh_visibility,
            mesh_link_state: Some(hub_state.link_state),
            membership_enforcement: Some(hub_state.membership_enforcement),
            hub_default_mesh_selector: hub_state.default_mesh_selector,
            hub_default_invite_configured: Some(hub_state.default_invite_token.is_some()),
            hub_first_time_onboarding: Some(hub_first_time_onboarding),
        }
    }

    async fn push_status(&self) {
        let status = self.status(false).await;
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
                election::InferenceTarget::Remote(_) | election::InferenceTarget::MoeRemote(_) => {
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

    hub::initialize(&state).await;
    hub::spawn_background_tasks(&state);

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
    let n = stream.read(&mut buf).await?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let method = req.split_whitespace().next().unwrap_or("GET");
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let path_only = path.split('?').next().unwrap_or(path);
    if hub::handle_route(state, method, path_only, &req, &mut stream).await? {
        return Ok(());
    }

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

        // ── Frontend static assets ──
        ("GET", p) if p.starts_with("/assets/") => {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Discover meshes via Nostr ──
        ("GET", "/api/discover") => {
            if hub::is_hub_enforced_mode(state).await {
                respond_error(
                    &mut stream,
                    403,
                    "Local discovery is disabled for InferenceHub-linked meshes",
                )
                .await?;
                return Ok(());
            }
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

        ("POST", "/api/models/request") => {
            let body = parse_json_body(&req).unwrap_or_default();
            let requested_model = body
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .unwrap_or("");
            let model = match state.request_catalog_model(requested_model).await {
                Ok(name) => name,
                Err(e) => {
                    respond_error(&mut stream, 400, &e.to_string()).await?;
                    return Ok(());
                }
            };
            state.push_status().await;
            respond_json_value(
                &mut stream,
                200,
                &serde_json::json!({
                    "ok": true,
                    "model": model,
                }),
            )
            .await?;
        }

        // ── Live status ──
        ("GET", "/api/status") => {
            let include_demand = query_flag(path, "include_demand");
            let status = state.status(include_demand).await;
            let json = serde_json::to_string(&status)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── SSE event stream ──
        ("GET", "/api/events") => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            let status = state.status(false).await;
            if let Ok(json) = serde_json::to_string(&status) {
                stream
                    .write_all(format!("data: {json}\n\n").as_bytes())
                    .await?;
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
        401 => "Unauthorized",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
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

fn parse_json_body(req: &str) -> Option<serde_json::Value> {
    let (_head, body) = req.split_once("\r\n\r\n")?;
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(trimmed).ok()
}

fn query_flag(path: &str, key: &str) -> bool {
    let Some((_, query)) = path.split_once('?') else {
        return false;
    };
    query.split('&').any(|pair| {
        let mut parts = pair.splitn(2, '=');
        let name = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("1").trim();
        name == key && !matches!(value, "0" | "false" | "False" | "FALSE")
    })
}

async fn respond_json_value(
    stream: &mut TcpStream,
    code: u16,
    body: &serde_json::Value,
) -> anyhow::Result<()> {
    let payload = serde_json::to_string(body)?;
    respond_json_raw(stream, code, &payload).await
}

async fn respond_json_raw(stream: &mut TcpStream, code: u16, body: &str) -> anyhow::Result<()> {
    let status = match code {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let payload = if body.trim().is_empty() { "{}" } else { body };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload,
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
