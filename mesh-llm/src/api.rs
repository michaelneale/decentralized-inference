//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   GET  /api/models    — list curated model metadata
//!   GET  /api/models/search?q=... — search Hugging Face or curated metadata
//!   GET  /api/models/show?ref=... — inspect one exact model ref
//!   POST /api/models/download     — start or resume a background model download
//!   GET  /api/models/download?ref=... — inspect download progress
//!   DELETE /api/models/download?ref=... — cancel a background download
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is mostly read-only — status, topology, and models.
//! Mesh mutations happen via CLI flags (--join, --model, --auto); model acquisition can also be driven via HTTP.

use crate::{affinity, election, mesh, models, nostr, plugin, proxy};
use include_dir::{include_dir, Dir};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");
const MESH_LLM_VERSION: &str = crate::VERSION;

// ── Shared state ──

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
    downloads: Arc<StdMutex<HashMap<String, ModelDownloadJob>>>,
}

struct ApiInner {
    node: mesh::Node,
    plugin_manager: plugin::PluginManager,
    affinity_router: affinity::AffinityRouter,
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

#[derive(Clone)]
struct ModelDownloadJob {
    model_ref: String,
    include_draft: bool,
    status: String,
    current_file: Option<String>,
    files_completed: usize,
    files_total: usize,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    path: Option<String>,
    draft_path: Option<String>,
    error: Option<String>,
    abort_handle: Option<tokio::task::AbortHandle>,
}

#[derive(Serialize)]
struct GpuEntry {
    name: String,
    vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bandwidth_gbps: Option<f64>,
}

fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_bandwidth: Option<&str>,
) -> Vec<GpuEntry> {
    let names: Vec<&str> = gpu_name
        .map(|s| s.split(", ").collect())
        .unwrap_or_default();
    if names.is_empty() {
        return vec![];
    }
    let vrams: Vec<Option<u64>> = gpu_vram
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let bandwidths: Vec<Option<f64>> = gpu_bandwidth
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name: name.to_string(),
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            bandwidth_gbps: bandwidths.get(i).copied().flatten(),
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
    routing_affinity: affinity::AffinityStatsSnapshot,
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

#[derive(Serialize)]
struct CuratedModelPayload {
    id: String,
    file: String,
    size: String,
    description: String,
    draft: Option<String>,
    vision: bool,
    download_url: String,
}

#[derive(Serialize)]
struct ModelSearchHitPayload {
    exact_ref: String,
    downloads: Option<u64>,
    likes: Option<u64>,
    curated: Option<CuratedModelPayload>,
}

#[derive(Serialize)]
struct ModelDetailsPayload {
    name: String,
    exact_ref: String,
    source: String,
    download_url: String,
    size: Option<String>,
    description: Option<String>,
    draft: Option<String>,
    vision: bool,
    moe: Option<crate::models::MoeConfig>,
}

#[derive(Deserialize)]
struct ModelDownloadRequest {
    #[serde(rename = "ref")]
    model_ref: String,
    #[serde(default)]
    draft: bool,
}

#[derive(Serialize)]
struct ModelDownloadPayload {
    model_ref: String,
    draft: bool,
    status: String,
    current_file: Option<String>,
    files_completed: usize,
    files_total: usize,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    path: Option<String>,
    draft_path: Option<String>,
    error: Option<String>,
}

fn curated_model_payload(model: &models::CuratedModel) -> CuratedModelPayload {
    CuratedModelPayload {
        id: model.id.to_string(),
        file: model.file.to_string(),
        size: model.size.to_string(),
        description: model.description.to_string(),
        draft: model.draft.map(str::to_string),
        vision: model.mmproj.is_some(),
        download_url: model.url.to_string(),
    }
}

fn model_details_payload(details: models::ModelDetails) -> ModelDetailsPayload {
    ModelDetailsPayload {
        name: details.display_name,
        exact_ref: details.exact_ref,
        source: details.source.to_string(),
        download_url: details.download_url,
        size: details.size_label,
        description: details.description,
        draft: details.draft,
        vision: details.vision,
        moe: details.moe,
    }
}

fn download_job_key(model_ref: &str, include_draft: bool) -> String {
    format!("{}|draft={include_draft}", model_ref.trim())
}

fn download_job_payload(job: &ModelDownloadJob) -> ModelDownloadPayload {
    ModelDownloadPayload {
        model_ref: job.model_ref.clone(),
        draft: job.include_draft,
        status: job.status.clone(),
        current_file: job.current_file.clone(),
        files_completed: job.files_completed,
        files_total: job.files_total,
        downloaded_bytes: job.downloaded_bytes,
        total_bytes: job.total_bytes,
        path: job.path.clone(),
        draft_path: job.draft_path.clone(),
        error: job.error.clone(),
    }
}

fn set_download_job<F>(
    downloads: &Arc<StdMutex<HashMap<String, ModelDownloadJob>>>,
    key: &str,
    f: F,
) where
    F: FnOnce(&mut ModelDownloadJob),
{
    if let Ok(mut jobs) = downloads.lock() {
        if let Some(job) = jobs.get_mut(key) {
            f(job);
        }
    }
}

fn download_assets_for_model(model: &models::CuratedModel) -> Vec<(String, String)> {
    let mut assets = vec![(model.file.to_string(), model.url.to_string())];
    for asset in model.extra_files {
        assets.push((asset.file.to_string(), asset.url.to_string()));
    }
    if let Some(asset) = &model.mmproj {
        assets.push((asset.file.to_string(), asset.url.to_string()));
    }
    assets
}

fn parse_query_params(path: &str) -> HashMap<String, String> {
    path.split('?')
        .nth(1)
        .unwrap_or("")
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (key, value) = part.split_once('=').unwrap_or((part, ""));
            (decode_url_component(key), decode_url_component(value))
        })
        .collect()
}

fn decode_url_component(input: &str) -> String {
    let mut decoded = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                decoded.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = &input[i + 1..i + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    decoded.push(byte);
                    i += 3;
                } else {
                    decoded.push(bytes[i]);
                    i += 1;
                }
            }
            byte => {
                decoded.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

impl MeshApi {
    pub fn new(
        node: mesh::Node,
        model_name: String,
        api_port: u16,
        model_size_bytes: u64,
        plugin_manager: plugin::PluginManager,
        affinity_router: affinity::AffinityRouter,
    ) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                affinity_router,
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
            downloads: Arc::new(StdMutex::new(HashMap::new())),
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
        let (
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            routing_affinity,
            model_name,
            model_size_bytes,
            llama_ready,
            is_host,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            nostr_discovery,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.affinity_router.stats_snapshot(),
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
                gpus: build_gpus(
                    p.gpu_name.as_deref(),
                    p.gpu_vram.as_deref(),
                    p.gpu_bandwidth_gbps.as_deref(),
                ),
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
        let mesh_models: Vec<MeshModelPayload> = catalog
            .iter()
            .map(|name| {
                let is_warm = served.contains(name);
                let node_count = if is_warm {
                    let peer_count = all_peers
                        .iter()
                        .filter(|p| {
                            p.serving_models.iter().any(|s| s == name)
                                || p.serving.as_deref() == Some(name.as_str())
                        })
                        .count();
                    // Count self: check all serving models, fall back to primary model_name
                    let me = if my_serving_models.iter().any(|s| s == name) || *name == model_name {
                        1
                    } else {
                        0
                    };
                    peer_count + me
                } else {
                    0
                };
                let size_gb = if *name == model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    crate::models::metadata_for_model_name(name)
                        .map(|m| crate::models::parse_size_gb(m.size))
                        .unwrap_or(0.0)
                };
                let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                    Some(d) => (
                        Some(d.request_count),
                        Some(now_ts.saturating_sub(d.last_active)),
                    ),
                    None => (None, None),
                };
                let vision = crate::models::metadata_for_model_name(name)
                    .map(|m| m.mmproj.is_some())
                    .unwrap_or(false);
                MeshModelPayload {
                    name: name.clone(),
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    size_gb,
                    vision,
                    request_count,
                    last_active_secs_ago,
                }
            })
            .collect();

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
            gpus: {
                let bw = node.gpu_bandwidth_gbps.lock().await;
                let bw_str = bw.as_ref().map(|v| {
                    v.iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                });
                build_gpus(
                    node.gpu_name.as_deref(),
                    node.gpu_vram.as_deref(),
                    bw_str.as_deref(),
                )
            },
            routing_affinity,
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

async fn spawn_model_download_job(
    state: MeshApi,
    request: ModelDownloadRequest,
) -> anyhow::Result<ModelDownloadPayload> {
    let key = download_job_key(&request.model_ref, request.draft);
    {
        let mut jobs = state.downloads.lock().unwrap();
        if let Some(existing) = jobs.get(&key) {
            let status = existing.status.as_str();
            if matches!(status, "queued" | "downloading" | "completed") {
                return Ok(download_job_payload(existing));
            }
        }
        jobs.insert(
            key.clone(),
            ModelDownloadJob {
                model_ref: request.model_ref.clone(),
                include_draft: request.draft,
                status: "queued".to_string(),
                current_file: None,
                files_completed: 0,
                files_total: 0,
                downloaded_bytes: 0,
                total_bytes: None,
                path: None,
                draft_path: None,
                error: None,
                abort_handle: None,
            },
        );
    }

    let job_state = state.clone();
    let job_key = key.clone();
    let request_for_task = request;
    let handle = tokio::spawn(async move {
        if let Err(error) =
            run_model_download_job(job_state.clone(), &job_key, request_for_task).await
        {
            set_download_job(&job_state.downloads, &job_key, |job| {
                if job.status != "cancelled" {
                    job.status = "failed".to_string();
                    job.error = Some(error.to_string());
                    job.current_file = None;
                }
            });
        }
    });
    let abort_handle = handle.abort_handle();
    set_download_job(&state.downloads, &key, |job| {
        job.abort_handle = Some(abort_handle);
    });

    let jobs = state.downloads.lock().unwrap();
    Ok(download_job_payload(
        jobs.get(&key).expect("download job exists"),
    ))
}

async fn run_model_download_job(
    state: MeshApi,
    key: &str,
    request: ModelDownloadRequest,
) -> anyhow::Result<()> {
    set_download_job(&state.downloads, key, |job| {
        job.status = "downloading".to_string();
        job.error = None;
    });

    let exact_ref = models::parse_exact_model_ref(&request.model_ref)?;
    let model_dir = models::primary_models_dir();
    tokio::fs::create_dir_all(&model_dir).await?;

    let (path, mut assets, draft_path) = match &exact_ref {
        models::ExactModelRef::Curated(model) => {
            let mut assets = download_assets_for_model(model);
            let draft_path = if request.draft {
                model
                    .draft
                    .and_then(models::find_curated_model_exact)
                    .map(|draft_model| {
                        assets.extend(download_assets_for_model(draft_model));
                        model_dir.join(draft_model.file).display().to_string()
                    })
            } else {
                None
            };
            (
                model_dir.join(model.file).display().to_string(),
                assets,
                draft_path,
            )
        }
        models::ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let mut assets = models::huggingface_download_assets(repo, revision.as_deref(), file)?;
            let filename = assets
                .first()
                .map(|(file, _)| file.clone())
                .ok_or_else(|| anyhow::anyhow!("Cannot extract filename from {file}"))?;
            let draft_path = if request.draft {
                models::show_exact_model(&request.model_ref)
                    .await
                    .ok()
                    .and_then(|details| details.draft)
                    .and_then(|draft| models::find_curated_model_exact(&draft))
                    .map(|draft_model| {
                        assets.extend(download_assets_for_model(draft_model));
                        model_dir.join(draft_model.file).display().to_string()
                    })
            } else {
                None
            };
            (
                model_dir.join(filename).display().to_string(),
                assets,
                draft_path,
            )
        }
        models::ExactModelRef::Url { url, filename } => {
            let mut assets = vec![(filename.clone(), url.clone())];
            let draft_path = if request.draft {
                models::show_exact_model(&request.model_ref)
                    .await
                    .ok()
                    .and_then(|details| details.draft)
                    .and_then(|draft| models::find_curated_model_exact(&draft))
                    .map(|draft_model| {
                        assets.extend(download_assets_for_model(draft_model));
                        model_dir.join(draft_model.file).display().to_string()
                    })
            } else {
                None
            };
            (
                model_dir.join(filename).display().to_string(),
                assets,
                draft_path,
            )
        }
    };

    set_download_job(&state.downloads, key, |job| {
        job.files_total = assets.len();
        job.path = Some(path.clone());
        job.draft_path = draft_path.clone();
    });

    let mut completed = 0usize;
    for (file, url) in assets.drain(..) {
        let dest = model_dir.join(&file);
        if tokio::fs::metadata(&dest)
            .await
            .map(|meta| meta.len() > 1_000_000)
            .unwrap_or(false)
        {
            completed += 1;
            let existing_bytes = std::fs::metadata(&dest).map(|meta| meta.len()).unwrap_or(0);
            set_download_job(&state.downloads, key, |job| {
                job.files_completed = completed;
                job.current_file = Some(file.clone());
                job.downloaded_bytes = existing_bytes;
                job.total_bytes = Some(job.downloaded_bytes);
            });
            continue;
        }

        set_download_job(&state.downloads, key, |job| {
            job.current_file = Some(file.clone());
            job.downloaded_bytes = 0;
            job.total_bytes = None;
        });
        let downloads = state.downloads.clone();
        let progress_key = key.to_string();
        models::download_url_with_progress(&url, &dest, move |downloaded, total| {
            set_download_job(&downloads, &progress_key, |job| {
                job.downloaded_bytes = downloaded;
                job.total_bytes = total;
            });
        })
        .await?;
        completed += 1;
        set_download_job(&state.downloads, key, |job| {
            job.files_completed = completed;
            job.downloaded_bytes = job.total_bytes.unwrap_or(job.downloaded_bytes);
        });
    }

    set_download_job(&state.downloads, key, |job| {
        if job.status != "cancelled" {
            job.status = "completed".to_string();
            job.current_file = None;
            job.abort_handle = None;
        }
    });
    Ok(())
}

fn get_model_download_job(
    state: &MeshApi,
    model_ref: &str,
    include_draft: bool,
) -> Option<ModelDownloadPayload> {
    let key = download_job_key(model_ref, include_draft);
    let jobs = state.downloads.lock().unwrap();
    jobs.get(&key).map(download_job_payload)
}

fn cancel_model_download_job(
    state: &MeshApi,
    model_ref: &str,
    include_draft: bool,
) -> Option<ModelDownloadPayload> {
    let key = download_job_key(model_ref, include_draft);
    let mut jobs = state.downloads.lock().unwrap();
    let job = jobs.get_mut(&key)?;
    if let Some(handle) = job.abort_handle.take() {
        handle.abort();
    }
    job.status = "cancelled".to_string();
    job.error = None;
    job.current_file = None;
    Some(download_job_payload(job))
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
    let request = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        proxy::read_http_request(&mut stream),
    )
    .await
    {
        Ok(Ok(request)) => request,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Ok(()), // read timeout — health check probe, just close
    };
    let req = String::from_utf8_lossy(&request.raw);
    let method = request.method.as_str();
    let path = request.path.as_str();
    let path_only = path.split('?').next().unwrap_or(path);
    let body = http_body_text(&request.raw);

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
        ("GET", p)
            if p.starts_with("/assets/")
                || matches!(p.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
                || (p.ends_with(".json") && !p.starts_with("/api/")) =>
        {
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
                    respond_json(&mut stream, &status).await?;
                }
                Err(_) => {
                    respond_error(&mut stream, 503, "Status temporarily unavailable").await?;
                }
            }
        }

        // ── Model management ──
        ("GET", "/api/models") => {
            let payload: Vec<CuratedModelPayload> = models::CURATED_MODELS
                .iter()
                .map(curated_model_payload)
                .collect();
            respond_json(&mut stream, &payload).await?;
        }

        ("GET", "/api/models/search") => {
            let params = parse_query_params(path);
            let query = params.get("q").map(|q| q.trim()).unwrap_or("");
            if query.is_empty() {
                respond_error(&mut stream, 400, "Missing required query param 'q'").await?;
            } else {
                let limit = params
                    .get("limit")
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(10)
                    .clamp(1, 50);
                let curated_only = params
                    .get("curated")
                    .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
                    .unwrap_or(false);
                if curated_only {
                    let payload: Vec<CuratedModelPayload> = models::search_curated_models(query)
                        .into_iter()
                        .take(limit)
                        .map(curated_model_payload)
                        .collect();
                    respond_json(&mut stream, &payload).await?;
                } else {
                    let payload: Vec<ModelSearchHitPayload> =
                        models::search_huggingface(query, limit)
                            .await?
                            .into_iter()
                            .map(|hit| ModelSearchHitPayload {
                                exact_ref: hit.exact_ref,
                                downloads: hit.downloads,
                                likes: hit.likes,
                                curated: hit.curated.map(curated_model_payload),
                            })
                            .collect();
                    respond_json(&mut stream, &payload).await?;
                }
            }
        }

        ("GET", "/api/models/show") => {
            let params = parse_query_params(path);
            let Some(model_ref) = params.get("ref").map(|value| value.trim()) else {
                respond_error(&mut stream, 400, "Missing required query param 'ref'").await?;
                return Ok(());
            };
            if model_ref.is_empty() {
                respond_error(&mut stream, 400, "Missing required query param 'ref'").await?;
            } else {
                let details = models::show_exact_model(model_ref).await?;
                respond_json(&mut stream, &model_details_payload(details)).await?;
            }
        }

        ("POST", "/api/models/download") => {
            let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
            let payload: ModelDownloadRequest = match serde_json::from_str(body) {
                Ok(payload) => payload,
                Err(_) => {
                    respond_error(&mut stream, 400, "Invalid JSON body").await?;
                    return Ok(());
                }
            };
            if payload.model_ref.trim().is_empty() {
                respond_error(&mut stream, 400, "Missing required field 'ref'").await?;
                return Ok(());
            }
            let payload = spawn_model_download_job(state.clone(), payload).await?;
            respond_json_status(&mut stream, 202, "Accepted", &payload).await?;
        }

        ("GET", "/api/models/download") => {
            let params = parse_query_params(path);
            let Some(model_ref) = params.get("ref").map(|value| value.trim()) else {
                respond_error(&mut stream, 400, "Missing required query param 'ref'").await?;
                return Ok(());
            };
            let include_draft = params
                .get("draft")
                .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false);
            if let Some(payload) = get_model_download_job(&state, model_ref, include_draft) {
                respond_json(&mut stream, &payload).await?;
            } else {
                respond_error(&mut stream, 404, "Download job not found").await?;
            }
        }

        ("DELETE", "/api/models/download") => {
            let params = parse_query_params(path);
            let Some(model_ref) = params.get("ref").map(|value| value.trim()) else {
                respond_error(&mut stream, 400, "Missing required query param 'ref'").await?;
                return Ok(());
            };
            let include_draft = params
                .get("draft")
                .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false);
            if let Some(payload) = cancel_model_download_job(&state, model_ref, include_draft) {
                respond_json(&mut stream, &payload).await?;
            } else {
                respond_error(&mut stream, 404, "Download job not found").await?;
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
                let payload = if body.trim().is_empty() { "{}" } else { body };
                let plugin_manager = state.inner.lock().await.plugin_manager.clone();
                match plugin_manager
                    .call_tool(plugin_name, tool_name, payload)
                    .await
                {
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
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let query_str = path.split('?').nth(1).unwrap_or("");
                let params: Vec<(&str, &str)> = query_str
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .collect();
                let request = crate::blackboard::FeedRequest {
                    from: params
                        .iter()
                        .find(|(k, _)| *k == "from")
                        .map(|(_, v)| (*v).to_string()),
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
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
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
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
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
                                    respond_error(&mut stream, status, &result.content_json)
                                        .await?;
                                }
                                Err(e) => {
                                    let msg = e.to_string();
                                    let status = if msg.contains("Rate limited") {
                                        429
                                    } else {
                                        400
                                    };
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

fn http_body_text(raw: &[u8]) -> &str {
    let body_start = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(raw.len());
    std::str::from_utf8(&raw[body_start..]).unwrap_or("")
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

async fn respond_json<T: Serialize>(stream: &mut TcpStream, value: &T) -> anyhow::Result<()> {
    respond_json_status(stream, 200, "OK", value).await
}

async fn respond_json_status<T: Serialize>(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    value: &T,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(value)?;
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
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
    respond_bytes_cached(
        stream,
        200,
        "OK",
        content_type,
        cache_control,
        file.contents(),
    )
    .await?;
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
    use mesh_llm_plugin::MeshVisibility;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    #[test]
    fn test_build_gpus_both_none() {
        let result = build_gpus(None, None, None);
        assert!(result.is_empty(), "expected empty vec when no gpu_name");
    }

    #[test]
    fn test_build_gpus_single_no_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 0);
    }

    #[test]
    fn test_build_gpus_single_with_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), Some("34359738368"), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
    }

    #[test]
    fn test_build_gpus_multi_full_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
            None,
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
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(
            result[1].vram_bytes, 0,
            "missing VRAM entry should default to 0"
        );
    }

    #[test]
    fn test_build_gpus_vram_no_gpu_name() {
        let result = build_gpus(None, Some("34359738368"), None);
        assert!(
            result.is_empty(),
            "no gpu_name means no entries even if vram present"
        );
    }

    #[test]
    fn test_build_gpus_vram_whitespace_trimmed() {
        let result = build_gpus(Some("NVIDIA RTX 4090"), Some(" 25769803776 "), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vram_bytes, 25_769_803_776);
    }

    #[test]
    fn test_build_gpus_with_bandwidth() {
        let result = build_gpus(
            Some("NVIDIA A100, NVIDIA A6000"),
            Some("85899345920,51539607552"),
            Some("1948.70,780.10"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].bandwidth_gbps, Some(1948.70));
        assert_eq!(result[1].bandwidth_gbps, Some(780.10));
    }

    #[test]
    fn test_build_gpus_unparsable_vram_preserves_index() {
        let result = build_gpus(Some("GPU0, GPU1, GPU2"), Some("100,foo,300"), None);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].vram_bytes, 100);
        assert_eq!(
            result[1].vram_bytes, 0,
            "unparsable vram should default to 0, not shift indices"
        );
        assert_eq!(result[2].vram_bytes, 300);
    }

    #[test]
    fn test_build_gpus_unparsable_bandwidth_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,200,300"),
            Some("1.0,bad,3.0"),
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].bandwidth_gbps, Some(1.0));
        assert_eq!(
            result[1].bandwidth_gbps, None,
            "unparsable bandwidth should be None, not shift indices"
        );
        assert_eq!(result[2].bandwidth_gbps, Some(3.0));
    }

    #[test]
    fn test_decode_url_component_decodes_spaces_and_percent_escapes() {
        assert_eq!(decode_url_component("qwen+coder+30b"), "qwen coder 30b");
        assert_eq!(
            decode_url_component("Qwen%2FQwen3-Coder-Next-GGUF"),
            "Qwen/Qwen3-Coder-Next-GGUF"
        );
    }

    #[test]
    fn test_parse_query_params_decodes_values() {
        let params = parse_query_params("/api/models/search?q=qwen+coder&limit=5");
        assert_eq!(params.get("q"), Some(&"qwen coder".to_string()));
        assert_eq!(params.get("limit"), Some(&"5".to_string()));
    }

    #[test]
    fn test_http_body_text_extracts_body() {
        let raw = b"POST /api/plugins/x/tools/y HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        assert_eq!(http_body_text(raw), "{\"a\":1}");
    }

    async fn build_test_mesh_api() -> MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let resolved_plugins = plugin::ResolvedPlugins {
            externals: vec![],
            inactive: vec![],
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let plugin_manager = plugin::PluginManager::start(
            &resolved_plugins,
            plugin::PluginHostMode {
                mesh_visibility: MeshVisibility::Private,
            },
            mesh_tx,
        )
        .await
        .unwrap();
        MeshApi::new(
            node,
            "test-model".to_string(),
            3131,
            0,
            plugin_manager,
            affinity::AffinityRouter::default(),
        )
    }

    async fn spawn_management_test_server(
        state: MeshApi,
    ) -> (
        std::net::SocketAddr,
        tokio::task::JoinHandle<anyhow::Result<()>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_request(stream, &state).await
        });
        (addr, handle)
    }

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    async fn read_until_contains(
        stream: &mut TcpStream,
        needle: &[u8],
        timeout: Duration,
    ) -> Vec<u8> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut response = Vec::new();
        while !contains_bytes(&response, needle) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out waiting for {:?} in response: {}",
                String::from_utf8_lossy(needle),
                String::from_utf8_lossy(&response)
            );
            let mut chunk = [0u8; 4096];
            let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
                .await
                .expect("timed out waiting for response bytes")
                .unwrap();
            assert!(n > 0, "unexpected EOF while waiting for response bytes");
            response.extend_from_slice(&chunk[..n]);
        }
        response
    }

    #[tokio::test]
    async fn test_management_request_parser_handles_fragmented_post_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = br#"{"text":"fragmented"}"#;
        let headers = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                proxy::read_http_request(&mut stream),
            )
            .await
            .unwrap()
            .unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(&headers.as_bytes()[..45]).await.unwrap();
            stream.write_all(&headers.as_bytes()[45..]).await.unwrap();
            stream.write_all(&body[..8]).await.unwrap();
            stream.write_all(&body[8..]).await.unwrap();
            let mut sink = [0u8; 1];
            let _ = stream.read(&mut sink).await;
        });

        client.await.unwrap();
        let request = server.await.unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/blackboard/post");
        assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
    }

    #[tokio::test]
    async fn test_api_events_sends_initial_payload_and_updates() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state.clone()).await;

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();

        let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
        let initial_text = String::from_utf8_lossy(&initial);
        assert!(initial_text.contains("HTTP/1.1 200 OK"));
        assert!(initial_text.contains("Content-Type: text/event-stream"));
        assert!(initial_text.contains("\"llama_ready\":false"));

        state.update(true, true).await;
        let updated =
            read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
        let updated_text = String::from_utf8_lossy(&updated);
        assert!(updated_text.contains("\"llama_ready\":true"));
        assert!(updated_text.contains("\"is_host\":true"));

        drop(stream);
        handle.abort();
    }
}
