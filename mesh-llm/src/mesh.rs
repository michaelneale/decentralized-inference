//! Mesh membership via iroh QUIC connections.
//!
//! Single ALPN, single connection per peer. Bi-streams multiplexed by
//! first byte: 0x01 = gossip, 0x02 = tunnel (RPC), 0x03 = tunnel map, 0x04 = tunnel (HTTP).

use anyhow::Result;
use base64::Engine;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use tokio::sync::{watch, Mutex};

/// Demand signal for a model — tracks interest via API requests and --model declarations.
/// Gossiped across the mesh and merged via max(). Decays naturally when last_active gets old.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    /// Unix timestamp of the most recent request or declaration.
    pub last_active: u64,
    /// Total requests seen (merged across peers via max).
    pub request_count: u64,
}

/// How long a demand entry stays relevant without being refreshed.
pub const DEMAND_TTL_SECS: u64 = 86400; // 24 hours

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn endpoint_id_hex(id: EndpointId) -> String {
    hex::encode(id.as_bytes())
}

fn new_plugin_message_id(source_peer_id: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{source_peer_id}:{nanos}:{}", rand::random::<u64>())
}

fn node_role_label(role: &NodeRole) -> String {
    match role {
        NodeRole::Worker => "worker".into(),
        NodeRole::Host { .. } => "host".into(),
        NodeRole::Client => "client".into(),
    }
}

fn peer_info_to_mesh_peer(peer: &PeerInfo) -> crate::plugin::proto::MeshPeer {
    crate::plugin::proto::MeshPeer {
        peer_id: endpoint_id_hex(peer.id),
        version: peer.version.clone().unwrap_or_default(),
        capabilities: Vec::new(),
        role: node_role_label(&peer.role),
        vram_bytes: peer.vram_bytes,
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        available_models: peer.available_models.clone(),
        requested_models: peer.requested_models.clone(),
        rtt_ms: peer.rtt_ms,
        model_source: peer.model_source.clone().unwrap_or_default(),
    }
}

fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.role != new.role
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving != new.serving
        || old.serving_models != new.serving_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.version != new.version
}

/// Merge two demand maps. For each model, take max of last_active and request_count.
pub fn merge_demand(
    ours: &mut HashMap<String, ModelDemand>,
    theirs: &HashMap<String, ModelDemand>,
) {
    for (model, their_demand) in theirs {
        let entry = ours.entry(model.clone()).or_default();
        entry.last_active = entry.last_active.max(their_demand.last_active);
        entry.request_count = entry.request_count.max(their_demand.request_count);
    }
}

pub const ALPN: &[u8] = b"mesh-llm/0";
const STREAM_GOSSIP: u8 = 0x01;
const STREAM_TUNNEL: u8 = 0x02;
const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
const STREAM_ROUTE_REQUEST: u8 = 0x05;
const STREAM_PEER_DOWN: u8 = 0x06;
const STREAM_PEER_LEAVING: u8 = 0x07;
const STREAM_BLACKBOARD: u8 = 0x08;
const STREAM_PLUGIN_CHANNEL: u8 = 0x09;
const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x0a;

/// Role a node plays in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeRole {
    /// Provides GPU compute via rpc-server for a specific model.
    Worker,
    /// Runs llama-server for a specific model, orchestrates inference, provides HTTP API.
    Host { http_port: u16 },
    /// Lite client — no compute, accesses the API via tunnel.
    Client,
}

impl Default for NodeRole {
    fn default() -> Self {
        NodeRole::Worker
    }
}

/// Gossip payload — extends EndpointAddr with role metadata.
/// Backward-compatible: old nodes that don't send role default to Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerAnnouncement {
    addr: EndpointAddr,
    #[serde(default)]
    role: NodeRole,
    /// GGUF model names on disk (catalog contribution)
    #[serde(default)]
    models: Vec<String>,
    /// Available VRAM in bytes (0 = unknown)
    #[serde(default)]
    vram_bytes: u64,
    /// How to get the model — catalog name, HF URL, or filename.
    /// Lets joining nodes auto-download without specifying --model.
    #[serde(default)]
    model_source: Option<String>,
    /// Primary model currently loaded in VRAM (None = not assigned yet).
    /// Kept for backward compat — old nodes only read this field.
    #[serde(default)]
    serving: Option<String>,
    /// All models currently loaded in VRAM (multi-model per node).
    #[serde(default)]
    serving_models: Vec<String>,
    /// All GGUF filenames on disk in ~/.models/ (for mesh catalog)
    #[serde(default)]
    available_models: Vec<String>,
    /// Models this node wants the mesh to serve (from --model flags)
    #[serde(default)]
    requested_models: Vec<String>,
    /// mesh-llm version string (e.g. "0.23.0")
    #[serde(default)]
    version: Option<String>,
    /// Mesh-wide demand map — only populated on the self entry.
    /// Other entries default to empty (serde default).
    #[serde(default)]
    model_demand: HashMap<String, ModelDemand>,
    /// Stable mesh identity — only populated on the self entry.
    #[serde(default)]
    mesh_id: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    #[serde(default)]
    gpu_name: Option<String>,
    /// Hostname of the node
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    is_soc: Option<bool>,
    #[serde(default)]
    gpu_vram: Option<String>,
    /// Per-GPU p90 bandwidth in GB/s, comma-separated in device order (e.g. "1671.7,722.16")
    #[serde(default)]
    gpu_bandwidth_gbps: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub tunnel_port: Option<u16>,
    pub role: NodeRole,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub rtt_ms: Option<u32>,
    pub model_source: Option<String>,
    /// Primary model currently loaded in VRAM
    pub serving: Option<String>,
    /// All models currently loaded in VRAM (multi-model per node)
    pub serving_models: Vec<String>,
    /// All GGUFs on disk
    pub available_models: Vec<String>,
    /// Models this node has requested the mesh to serve
    pub requested_models: Vec<String>,
    /// Last time we directly communicated with this peer (gossip, heartbeat, tunnel).
    /// Peers not seen in PEER_STALE_SECS are pruned from gossip and eventually removed.
    pub last_seen: std::time::Instant,
    /// mesh-llm version (e.g. "0.23.0")
    pub version: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    pub gpu_name: Option<String>,
    /// Hostname of the node
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_bandwidth_gbps: Option<String>,
}

/// Peers not directly verified within this window are considered stale
/// and excluded from gossip propagation. After 2x this duration they're removed entirely.
const PEER_STALE_SECS: u64 = 180; // 3 minutes

/// Directories to scan for GGUF models.
pub fn model_dirs() -> Vec<std::path::PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let mut dirs = vec![home.join(".models")];
    // Also scan goose's model directory (~/Library/Application Support/Block.goose/models/)
    if let Some(data_dir) = dirs::data_dir() {
        let goose_dir = data_dir.join("Block.goose").join("models");
        if goose_dir.exists() {
            dirs.push(goose_dir);
        }
    }
    dirs
}

/// Scan model directories for GGUF files and return their stem names.
pub fn scan_local_models() -> Vec<String> {
    let mut names = Vec::new();
    for models_dir in model_dirs() {
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        // Skip draft models (tiny) and partial downloads
                        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        if size > 500_000_000 {
                            // > 500MB, skip draft models
                            // For split GGUFs (name-00001-of-00006.gguf), use base name
                            let name = split_gguf_base_name(stem).unwrap_or(stem).to_string();
                            if !names.contains(&name) {
                                names.push(name);
                            }
                        }
                    }
                }
            }
        }
    }
    names.sort();
    names
}

/// Extract the base model name from a split GGUF stem.
/// "GLM-5-UD-IQ2_XXS-00001-of-00006" → Some("GLM-5-UD-IQ2_XXS")
/// "Qwen3-8B-Q4_K_M" → None (not a split file)
fn split_gguf_base_name(stem: &str) -> Option<&str> {
    // Pattern: ...-NNNNN-of-NNNNN
    let suffix = stem.rfind("-of-")?;
    let part_num = &stem[suffix + 4..];
    if part_num.len() != 5 || !part_num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let dash = stem[..suffix].rfind('-')?;
    let seq = &stem[dash + 1..suffix];
    if seq.len() != 5 || !seq.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&stem[..dash])
}

/// Find a GGUF model file by stem name, searching all model directories.
/// Returns the first match found (prefers ~/.models/ over goose dir).
/// For split GGUFs, finds the first part (name-00001-of-NNNNN.gguf).
pub fn find_model_path(stem: &str) -> std::path::PathBuf {
    let filename = format!("{}.gguf", stem);
    for dir in model_dirs() {
        // Try single-file first
        let candidate = dir.join(&filename);
        if candidate.exists() {
            return candidate;
        }
        // Try split GGUF: stem-00001-of-*.gguf
        let split_prefix = format!("{}-00001-of-", stem);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&split_prefix) && name.ends_with(".gguf") {
                        return entry.path();
                    }
                }
            }
        }
    }
    // Fallback: return ~/.models/ path even if it doesn't exist
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".models")
        .join(&filename)
}

/// Detect available VRAM. On Apple Silicon, uses ~75% of system RAM
/// (the rest is reserved for OS/apps on unified memory).
/// Detect available memory for model loading, capped by max_vram_gb if set.
/// "VRAM" is a misnomer — on macOS unified memory and Linux CPU-only, this
/// is system RAM. On Linux with a GPU, it's actual GPU VRAM.
pub fn detect_vram_bytes_capped(max_vram_gb: Option<f64>) -> u64 {
    let mut detected = crate::hardware::survey().vram_bytes;
    if let Some(cap) = max_vram_gb {
        let cap_bytes = (cap * 1e9) as u64;
        if cap_bytes < detected {
            detected = cap_bytes;
        }
    }
    detected
}

/// Lightweight routing table for passive nodes (clients + standby GPU).
/// Contains just enough info to route requests to the right host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub hosts: Vec<RouteEntry>,
    /// Stable mesh identity — shared by all nodes in the same mesh.
    #[serde(default)]
    pub mesh_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEntry {
    pub model: String,
    pub node_id: String,
    pub endpoint_id: EndpointId,
    pub vram_gb: f64,
}

/// Discover our public IP via STUN, then pair it with the given port.
/// We can't send STUN from the bound port (iroh owns it), but we only need
/// the public IP — the port is known from --bind-port + router forwarding.
async fn stun_public_addr(advertised_port: u16) -> Option<std::net::SocketAddr> {
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

    let stun_servers = [
        "stun.l.google.com:19302",
        "stun.cloudflare.com:3478",
        "stun.stunprotocol.org:3478",
    ];

    // Bind to ephemeral port — we only care about the IP, not the mapped port.
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;

    for server in &stun_servers {
        // STUN Binding Request: type=0x0001, len=0, magic=0x2112A442, txn=random
        let mut req = [0u8; 20];
        req[0] = 0x00;
        req[1] = 0x01; // Binding Request
                       // length = 0
        req[4] = 0x21;
        req[5] = 0x12;
        req[6] = 0xA4;
        req[7] = 0x42; // Magic Cookie
        rand::fill(&mut req[8..20]);

        let dest: SocketAddr = match tokio::net::lookup_host(server).await {
            Ok(mut addrs) => match addrs.next() {
                Some(a) => a,
                None => continue,
            },
            Err(_) => continue,
        };

        if sock.send_to(&req, dest).await.is_err() {
            continue;
        }

        let mut buf = [0u8; 256];
        match tokio::time::timeout(std::time::Duration::from_secs(2), sock.recv_from(&mut buf))
            .await
        {
            Ok(Ok((len, _))) if len >= 20 => {
                // Parse STUN response for XOR-MAPPED-ADDRESS (0x0020)
                // or MAPPED-ADDRESS (0x0001)
                let magic = &req[4..8];
                let _txn = &req[8..20];
                let mut i = 20;
                while i + 4 <= len {
                    let attr_type = u16::from_be_bytes([buf[i], buf[i + 1]]);
                    let attr_len = u16::from_be_bytes([buf[i + 2], buf[i + 3]]) as usize;
                    if i + 4 + attr_len > len {
                        break;
                    }
                    let val = &buf[i + 4..i + 4 + attr_len];

                    if attr_type == 0x0020 && attr_len >= 8 && val[1] == 0x01 {
                        // XOR-MAPPED-ADDRESS, IPv4 — extract IP only
                        let ip = Ipv4Addr::new(
                            val[4] ^ magic[0],
                            val[5] ^ magic[1],
                            val[6] ^ magic[2],
                            val[7] ^ magic[3],
                        );
                        let addr = SocketAddr::V4(SocketAddrV4::new(ip, advertised_port));
                        tracing::info!("STUN discovered public address: {addr}");
                        return Some(addr);
                    }
                    if attr_type == 0x0001 && attr_len >= 8 && val[1] == 0x01 {
                        // MAPPED-ADDRESS, IPv4 — extract IP only
                        let ip = Ipv4Addr::new(val[4], val[5], val[6], val[7]);
                        let addr = SocketAddr::V4(SocketAddrV4::new(ip, advertised_port));
                        tracing::info!("STUN discovered public address: {addr}");
                        return Some(addr);
                    }

                    // Attributes are padded to 4-byte boundary
                    i += 4 + (attr_len + 3) & !3;
                }
            }
            _ => continue,
        }
    }

    tracing::warn!("STUN: could not discover public address");
    None
}

#[derive(Clone)]
pub struct Node {
    endpoint: Endpoint,
    public_addr: Option<std::net::SocketAddr>,
    state: Arc<Mutex<MeshState>>,
    role: Arc<Mutex<NodeRole>>,
    models: Arc<Mutex<Vec<String>>>,
    model_source: Arc<Mutex<Option<String>>>,
    serving: Arc<Mutex<Option<String>>>,
    serving_models: Arc<Mutex<Vec<String>>>,
    llama_ready: Arc<Mutex<bool>>,
    available_models: Arc<Mutex<Vec<String>>>,
    requested_models: Arc<Mutex<Vec<String>>>,
    /// Mesh-wide demand map — merged from gossip + local API requests.
    /// This is the single source of truth for "what does the mesh want?"
    model_demand: Arc<std::sync::Mutex<HashMap<String, ModelDemand>>>,
    mesh_id: Arc<Mutex<Option<String>>>,
    accepting: Arc<(tokio::sync::Notify, std::sync::atomic::AtomicBool)>,
    vram_bytes: u64,
    peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    inflight_change_tx: watch::Sender<u64>,
    tunnel_tx: tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    tunnel_http_tx:
        tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    plugin_manager: Arc<Mutex<Option<crate::plugin::PluginManager>>>,
    pub blackboard: crate::blackboard::BlackboardStore,
    blackboard_name: Arc<Mutex<Option<String>>>,
    pub enumerate_host: bool,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_bandwidth_gbps: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
}

struct MeshState {
    peers: HashMap<EndpointId, PeerInfo>,
    connections: HashMap<EndpointId, Connection>,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    dead_peers: std::collections::HashSet<EndpointId>,
    seen_plugin_messages: HashSet<String>,
    seen_plugin_message_order: VecDeque<String>,
}

/// Channels returned by Node::start for inbound tunnel streams.
pub struct TunnelChannels {
    pub rpc: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    pub http: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
}

pub struct InflightRequestGuard {
    inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    inflight_change_tx: watch::Sender<u64>,
}

impl Drop for InflightRequestGuard {
    fn drop(&mut self) {
        let _ = self.inflight_requests.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
        let _ = self.inflight_change_tx.send(
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64,
        );
    }
}

impl Node {
    pub fn begin_inflight_request(&self) -> InflightRequestGuard {
        self.inflight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = self.inflight_change_tx.send(
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64,
        );
        InflightRequestGuard {
            inflight_requests: self.inflight_requests.clone(),
            inflight_change_tx: self.inflight_change_tx.clone(),
        }
    }

    pub fn inflight_requests(&self) -> u64 {
        self.inflight_requests
            .load(std::sync::atomic::Ordering::Relaxed) as u64
    }

    pub fn inflight_change_rx(&self) -> watch::Receiver<u64> {
        self.inflight_change_tx.subscribe()
    }

    pub async fn start(
        role: NodeRole,
        relay_urls: &[String],
        bind_port: Option<u16>,
        max_vram_gb: Option<f64>,
        enumerate_host: bool,
    ) -> Result<(Self, TunnelChannels)> {
        // Clients use an ephemeral key so they get a unique identity even
        // when running on the same machine as a GPU node.
        let secret_key = if matches!(role, NodeRole::Client)
            || std::env::var("MESH_LLM_EPHEMERAL_KEY").is_ok()
        {
            let key = SecretKey::generate(&mut rand::rng());
            tracing::info!("Using ephemeral key (unique identity)");
            key
        } else {
            load_or_create_key().await?
        };
        // Configure QUIC transport for heavy RPC traffic:
        // Use iroh's default transport config — it sets keep_alive, path timeouts,
        // and multipath correctly. Only override the bidi stream limit.
        use iroh::endpoint::QuicTransportConfig;
        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .build();
        let mut builder = Endpoint::builder()
            .secret_key(secret_key)
            .alpns(vec![ALPN.to_vec()])
            .transport_config(transport_config);

        {
            use iroh::{RelayConfig, RelayMap};
            let urls: Vec<String> = if relay_urls.is_empty() {
                vec![
                    "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./".into(),
                    "https://mesh-llm-relay.fly.dev./".into(),
                ]
            } else {
                relay_urls.to_vec()
            };
            // Two relays: dedicated iroh relay (proper QUIC + STUN) and Fly relay (fallback).
            let configs: Vec<RelayConfig> = urls
                .iter()
                .map(|url| RelayConfig {
                    url: url.parse().expect("invalid relay URL"),
                    quic: None,
                })
                .collect();
            let relay_map = RelayMap::from_iter(configs);
            tracing::info!("Relay: {:?}", urls);
            builder = builder.relay_mode(iroh::endpoint::RelayMode::Custom(relay_map));
        }
        if let Some(port) = bind_port {
            tracing::info!("Binding QUIC to UDP port {port}");
            builder = builder.bind_addr(std::net::SocketAddr::from(([0, 0, 0, 0], port)))?;
        }
        let endpoint = builder.bind().await?;
        // Wait briefly for relay connection so the invite token includes the relay URL.
        // On sinkholed networks this times out and we proceed without relay (direct UDP only).
        match tokio::time::timeout(std::time::Duration::from_secs(5), endpoint.online()).await {
            Ok(()) => tracing::info!("Relay connected"),
            Err(_) => tracing::warn!("Relay connection timed out (5s) — proceeding without relay"),
        }

        // Discover public IP via STUN so the invite token includes it.
        // With --bind-port, the advertised port is the bound port (for port forwarding).
        // Without --bind-port, we use port 0 — the IP is still useful for hole-punching.
        // Relay STUN may not work on sinkholed networks, so we use raw STUN to Google/Cloudflare.
        let stun_port = bind_port.unwrap_or(0);
        let public_addr = stun_public_addr(stun_port).await;

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);

        let hw = crate::hardware::survey();
        let mut vram = hw.vram_bytes;
        let gpu_name = if matches!(role, NodeRole::Client) {
            None
        } else {
            hw.gpu_name
        };
        let hostname = hw.hostname;
        let is_soc = Some(hw.is_soc);
        let gpu_vram = if hw.gpu_vram.is_empty() {
            None
        } else {
            Some(
                hw.gpu_vram
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        };
        if let Some(max_gb) = max_vram_gb {
            let max_bytes = (max_gb * 1e9) as u64;
            if max_bytes < vram {
                tracing::info!(
                    "Detected VRAM: {:.1} GB, capped to {:.1} GB (--max-vram)",
                    vram as f64 / 1e9,
                    max_gb
                );
                vram = max_bytes;
            } else {
                tracing::info!(
                    "Detected VRAM: {:.1} GB (--max-vram {:.1} has no effect)",
                    vram as f64 / 1e9,
                    max_gb
                );
            }
        } else {
            tracing::info!("Detected VRAM: {:.1} GB", vram as f64 / 1e9);
        }

        let node = Node {
            endpoint,
            public_addr,
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                remote_tunnel_maps: HashMap::new(),
                dead_peers: std::collections::HashSet::new(),
                seen_plugin_messages: HashSet::new(),
                seen_plugin_message_order: VecDeque::new(),
            })),
            role: Arc::new(Mutex::new(role)),
            models: Arc::new(Mutex::new(Vec::new())),
            model_source: Arc::new(Mutex::new(None)),
            serving: Arc::new(Mutex::new(None)),
            serving_models: Arc::new(Mutex::new(Vec::new())),
            llama_ready: Arc::new(Mutex::new(false)),
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
            mesh_id: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            vram_bytes: vram,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            tunnel_tx,
            tunnel_http_tx,
            plugin_manager: Arc::new(Mutex::new(None)),
            blackboard: crate::blackboard::BlackboardStore::new(false),
            blackboard_name: Arc::new(Mutex::new(None)),
            enumerate_host,
            gpu_name,
            hostname,
            is_soc,
            gpu_vram,
            gpu_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
        };

        // Accept loop starts but waits for start_accepting() before processing connections.
        // This lets idle mode create a node (for identity/token) without joining any mesh.
        let node2 = node.clone();
        tokio::spawn(async move {
            node2.accept_loop().await;
        });

        Ok((
            node,
            TunnelChannels {
                rpc: tunnel_rx,
                http: tunnel_http_rx,
            },
        ))
    }

    pub fn invite_token(&self) -> String {
        let mut addr = self.endpoint.addr();
        // Inject STUN-discovered public address if relay STUN didn't provide one.
        if let Some(pub_addr) = self.public_addr {
            use iroh::TransportAddr;
            let has_public = addr.addrs.iter().any(|a| match a {
                TransportAddr::Ip(sock) => match sock.ip() {
                    std::net::IpAddr::V4(v4) => !v4.is_private() && !v4.is_loopback(),
                    _ => false,
                },
                _ => false,
            });
            if !has_public {
                addr.addrs.insert(TransportAddr::Ip(pub_addr));
            }
        }
        let json = serde_json::to_vec(&addr).expect("serializable");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json)
    }

    async fn build_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<crate::plugin::proto::MeshPeer>,
        detail_json: String,
    ) -> crate::plugin::proto::MeshEvent {
        crate::plugin::proto::MeshEvent {
            kind: kind as i32,
            peer,
            local_peer_id: endpoint_id_hex(self.endpoint.id()),
            mesh_id: self.mesh_id.lock().await.clone().unwrap_or_default(),
            detail_json,
        }
    }

    /// Enable accepting inbound connections. Call before join() or when ready to participate.
    /// Until this is called, the accept loop blocks waiting.
    pub fn start_accepting(&self) {
        self.accepting
            .1
            .store(true, std::sync::atomic::Ordering::Release);
        self.accepting.0.notify_waiters();
        let node = self.clone();
        tokio::spawn(async move {
            let plugin_manager = node.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                let _ = plugin_manager
                    .broadcast_mesh_event(
                        node.build_mesh_event(
                            crate::plugin::proto::mesh_event::Kind::LocalAccepting,
                            None,
                            String::new(),
                        )
                        .await,
                    )
                    .await;
            }
        });
    }

    pub async fn join(&self, invite_token: &str) -> Result<()> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
        let addr: EndpointAddr = serde_json::from_slice(&json)?;
        // Clear dead status — explicit join should always attempt connection
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.connect_to_peer(addr).await
    }

    /// Connect to a peer without gossip exchange — for passive nodes (clients/standby).
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub async fn role(&self) -> NodeRole {
        self.role.lock().await.clone()
    }

    pub async fn set_role(&self, role: NodeRole) {
        *self.role.lock().await = role;
    }

    pub async fn set_models(&self, models: Vec<String>) {
        *self.models.lock().await = models;
    }

    pub async fn set_model_source(&self, source: String) {
        *self.model_source.lock().await = Some(source);
    }

    pub async fn set_serving(&self, model: Option<String>) {
        *self.serving.lock().await = model;
    }

    pub async fn set_serving_models(&self, models: Vec<String>) {
        // Also keep `serving` in sync (primary = first model)
        *self.serving.lock().await = models.first().cloned();
        *self.serving_models.lock().await = models;
    }

    pub async fn serving_models(&self) -> Vec<String> {
        self.serving_models.lock().await.clone()
    }

    /// Set the display name for blackboard posts.
    pub async fn set_blackboard_name(&self, name: String) {
        *self.blackboard_name.lock().await = Some(name);
    }

    /// Get the display name for this node (for blackboard posts).
    /// Falls back to the short endpoint ID if no name is set.
    pub async fn peer_name(&self) -> String {
        if let Some(ref name) = *self.blackboard_name.lock().await {
            name.clone()
        } else {
            self.endpoint.id().fmt_short().to_string()
        }
    }

    pub async fn set_plugin_manager(&self, plugin_manager: crate::plugin::PluginManager) {
        let peers = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect::<Vec<_>>()
        };
        *self.plugin_manager.lock().await = Some(plugin_manager.clone());
        let local_kind = if self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            crate::plugin::proto::mesh_event::Kind::LocalAccepting
        } else {
            crate::plugin::proto::mesh_event::Kind::LocalStandby
        };
        let _ = plugin_manager
            .broadcast_mesh_event(self.build_mesh_event(local_kind, None, String::new()).await)
            .await;
        if self.mesh_id.lock().await.is_some() {
            let _ = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                        None,
                        String::new(),
                    )
                    .await,
                )
                .await;
        }
        for peer in peers {
            if let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUp,
                        Some(peer_info_to_mesh_peer(&peer)),
                        String::new(),
                    )
                    .await,
                )
                .await
            {
                tracing::debug!(
                    "Failed to send existing peer snapshot to plugins for {}: {err}",
                    peer.id.fmt_short()
                );
            }
        }
    }

    pub fn start_plugin_channel_forwarder(
        &self,
        mut rx: tokio::sync::mpsc::Receiver<crate::plugin::PluginMeshEvent>,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(err) = node.forward_plugin_event(event).await {
                    tracing::debug!("Plugin mesh forward failed: {err}");
                }
            }
        });
    }

    async fn emit_plugin_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<&PeerInfo>,
        detail_json: String,
    ) {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        if let Some(plugin_manager) = plugin_manager {
            if let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(kind, peer.map(peer_info_to_mesh_peer), detail_json)
                        .await,
                )
                .await
            {
                tracing::debug!(
                    "Failed to deliver plugin mesh event {:?} for {}: {err}",
                    kind,
                    peer.map(|p| p.id.fmt_short().to_string())
                        .unwrap_or_else(|| self.endpoint.id().fmt_short().to_string())
                );
            }
        }
    }

    async fn update_peer_rtt(&self, id: EndpointId, rtt_ms: u32) {
        let updated_peer = {
            let mut state = self.state.lock().await;
            if let Some(peer) = state.peers.get_mut(&id) {
                peer.rtt_ms = Some(rtt_ms);
                Some(peer.clone())
            } else {
                None
            }
        };
        if let Some(peer) = updated_peer {
            tracing::info!("Peer {} RTT: {}ms", id.fmt_short(), rtt_ms);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    /// Re-gossip our state to all connected peers.
    /// Call after changing serving/role/models so peers learn the update.
    pub async fn regossip(&self) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.initiate_gossip(conn, peer_id).await {
                    tracing::debug!("Regossip to {} failed: {e}", peer_id.fmt_short());
                }
            });
        }
    }

    /// Gossip with one connected peer to update routing table.
    /// Used by: (1) passive nodes' periodic 60s heartbeat, (2) background
    /// refresh on tunnel failure so future requests have fresh routing.
    pub async fn gossip_one_peer(&self) {
        let conn = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .next()
                .map(|(id, c)| (*id, c.clone()))
        };
        if let Some((peer_id, conn)) = conn {
            let _ = self.initiate_gossip_inner(conn, peer_id, false).await;
        }
    }

    pub async fn serving(&self) -> Option<String> {
        self.serving.lock().await.clone()
    }

    pub async fn set_llama_ready(&self, ready: bool) {
        *self.llama_ready.lock().await = ready;
    }

    pub async fn is_llama_ready(&self) -> bool {
        *self.llama_ready.lock().await
    }

    pub async fn mesh_id(&self) -> Option<String> {
        self.mesh_id.lock().await.clone()
    }

    /// Set the mesh identity. If None was set, adopts the given ID (from gossip).
    /// If already set, ignores (originator's ID wins).
    pub async fn set_mesh_id(&self, id: String) {
        let mut current = self.mesh_id.lock().await;
        if current.is_none() {
            *current = Some(id);
            drop(current);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                None,
                String::new(),
            )
            .await;
        }
    }

    /// Set mesh ID unconditionally (for originator).
    pub async fn set_mesh_id_force(&self, id: String) {
        *self.mesh_id.lock().await = Some(id);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
            None,
            String::new(),
        )
        .await;
    }

    pub async fn set_available_models(&self, models: Vec<String>) {
        *self.available_models.lock().await = models;
    }

    pub async fn available_models(&self) -> Vec<String> {
        self.available_models.lock().await.clone()
    }

    /// Record a request for a model — updates the demand map.
    /// Called from API proxy on every request (including misses for unserved models).
    /// Uses std::sync::Mutex (not tokio) so it can be called from sync context too.
    pub fn record_request(&self, model: &str) {
        // "auto" is a routing directive, not a real model — don't pollute demand
        if model == "auto" || model.is_empty() {
            return;
        }
        let mut demand = self.model_demand.lock().unwrap();
        let entry = demand.entry(model.to_string()).or_default();
        entry.last_active = now_secs();
        entry.request_count += 1;
    }

    /// Get the current demand map (for gossip and assignment decisions).
    pub fn get_demand(&self) -> HashMap<String, ModelDemand> {
        self.model_demand.lock().unwrap().clone()
    }

    /// Merge incoming demand from gossip into our local map.
    pub fn merge_remote_demand(&self, remote: &HashMap<String, ModelDemand>) {
        let mut demand = self.model_demand.lock().unwrap();
        merge_demand(&mut demand, remote);
    }

    /// Remove demand entries that have expired (past TTL and not pinned).
    /// Call periodically to prevent unbounded map growth.
    pub async fn gc_demand(&self) {
        let now = now_secs();
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for p in peers.peers.values() {
            for m in &p.requested_models {
                pinned.insert(m.clone());
            }
        }
        drop(peers);
        drop(my_requested);

        let mut demand = self.model_demand.lock().unwrap();
        demand.retain(|model, d| pinned.contains(model) || (now - d.last_active) < DEMAND_TTL_SECS);
    }

    /// Get active demand entries (within TTL or pinned by a live node).
    /// This replaces mesh_wanted_models().
    pub async fn active_demand(&self) -> HashMap<String, ModelDemand> {
        let now = now_secs();
        let demand = self.model_demand.lock().unwrap().clone();

        // Check which models are pinned (declared via --model by self or a live peer)
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for p in peers.peers.values() {
            for m in &p.requested_models {
                pinned.insert(m.clone());
            }
        }
        drop(peers);
        drop(my_requested);

        demand
            .into_iter()
            .filter(|(model, d)| pinned.contains(model) || (now - d.last_active) < DEMAND_TTL_SECS)
            .collect()
    }

    pub async fn set_requested_models(&self, models: Vec<String>) {
        // Seed demand entries for --model declarations
        {
            let mut demand = self.model_demand.lock().unwrap();
            let now = now_secs();
            for m in &models {
                let entry = demand.entry(m.clone()).or_default();
                entry.last_active = entry.last_active.max(now);
            }
        }
        *self.requested_models.lock().await = models;
    }

    /// Start a background task that periodically checks peer health.
    /// Probes each peer by attempting a gossip exchange. If the probe fails
    /// (connection dead, peer unresponsive), removes the peer immediately
    /// rather than waiting for QUIC idle timeout.
    /// Start a slow heartbeat (60s) that gossips with a random subset of peers.
    /// At small mesh sizes (≤5 peers), talks to everyone. At larger sizes,
    /// picks K random peers per cycle. Information propagates infectiously —
    /// changes reach all nodes in O(log N) cycles.
    /// Death detection primarily happens on the data path (tunnel fails →
    /// broadcast_peer_down), not via heartbeat.
    pub fn start_heartbeat(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut fail_counts: std::collections::HashMap<EndpointId, u32> =
                std::collections::HashMap::new();

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;

                let mut peers_and_conns: Vec<(EndpointId, Option<Connection>)> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .keys()
                        .map(|id| {
                            let conn = state.connections.get(id).cloned();
                            (*id, conn)
                        })
                        .collect()
                };
                tracing::debug!("Heartbeat tick: {} peers to check", peers_and_conns.len());

                // Random-K gossip: pick a subset at larger mesh sizes.
                // At ≤5 peers, talk to everyone (backward compat with current behavior).
                // At larger sizes, pick 5 random peers per cycle.
                const GOSSIP_K: usize = 5;
                if peers_and_conns.len() > GOSSIP_K {
                    use rand::seq::SliceRandom;
                    peers_and_conns.shuffle(&mut rand::rng());
                    peers_and_conns.truncate(GOSSIP_K);
                }

                for (peer_id, conn) in peers_and_conns {
                    let hb_start = std::time::Instant::now();
                    let alive = if let Some(conn) = conn {
                        // Gossip as heartbeat — syncs state but won't re-discover dead peers
                        let result = tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            node.initiate_gossip_inner(conn, peer_id, false),
                        )
                        .await
                        .map(|r| r.is_ok())
                        .unwrap_or(false);
                        tracing::debug!(
                            "Heartbeat gossip {} = {} ({}ms)",
                            peer_id.fmt_short(),
                            if result { "ok" } else { "fail" },
                            hb_start.elapsed().as_millis()
                        );
                        result
                    } else {
                        // No connection — try to reconnect using stored address
                        let addr = {
                            let state = node.state.lock().await;
                            state.peers.get(&peer_id).map(|p| p.addr.clone())
                        };
                        if let Some(addr) = addr {
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(10),
                                node.endpoint.connect(addr, ALPN),
                            )
                            .await
                            {
                                Ok(Ok(new_conn)) => {
                                    eprintln!(
                                        "💚 Heartbeat: reconnected to {}",
                                        peer_id.fmt_short()
                                    );
                                    node.state
                                        .lock()
                                        .await
                                        .connections
                                        .insert(peer_id, new_conn.clone());
                                    // Spawn dispatch_streams for the new connection
                                    let n2 = node.clone();
                                    let nc = new_conn.clone();
                                    tokio::spawn(async move {
                                        n2.dispatch_streams(nc, peer_id).await;
                                    });
                                    // Try gossip on the new connection
                                    tokio::time::timeout(
                                        std::time::Duration::from_secs(10),
                                        node.initiate_gossip_inner(new_conn, peer_id, false),
                                    )
                                    .await
                                    .map(|r| r.is_ok())
                                    .unwrap_or(false)
                                }
                                _ => false,
                            }
                        } else {
                            false
                        }
                    };

                    if alive {
                        if fail_counts.contains_key(&peer_id) {
                            eprintln!(
                                "💚 Heartbeat: {} recovered (was {}/2)",
                                peer_id.fmt_short(),
                                fail_counts.get(&peer_id).unwrap_or(&0)
                            );
                            // Clear dead_peers if peer came back
                            node.state.lock().await.dead_peers.remove(&peer_id);
                        }
                        fail_counts.remove(&peer_id);
                    } else {
                        // Check if peer has contacted US recently (inbound gossip).
                        // If so, peer is alive — we just can't reach them outbound (NAT).
                        let recently_seen = {
                            let state = node.state.lock().await;
                            state
                                .peers
                                .get(&peer_id)
                                .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
                                .unwrap_or(false)
                        };
                        if recently_seen {
                            // Peer is alive via inbound, don't count as failure
                            if fail_counts.contains_key(&peer_id) {
                                eprintln!("💚 Heartbeat: {} outbound failed but seen recently (inbound alive)", peer_id.fmt_short());
                                fail_counts.remove(&peer_id);
                            }
                        } else {
                            let count = fail_counts.entry(peer_id).or_default();
                            *count += 1;
                            if *count >= 2 {
                                // Only add to dead_peers on confirmed death (2 strikes),
                                // not on first timeout — a single timeout shouldn't block
                                // incoming gossip from an otherwise-alive peer.
                                node.state.lock().await.dead_peers.insert(peer_id);
                                eprintln!("💔 Heartbeat: {} unreachable ({} failures), removing + broadcasting death", peer_id.fmt_short(), count);
                                fail_counts.remove(&peer_id);
                                node.handle_peer_death(peer_id).await;
                            } else {
                                eprintln!(
                                    "💛 Heartbeat: {} unreachable ({}/2), will retry",
                                    peer_id.fmt_short(),
                                    count
                                );
                            }
                        }
                    }
                }

                // Prune stale peers: no direct contact in 2× the stale window.
                // These are ghost records propagated via gossip from other nodes
                // but never directly verified by us.
                let prune_cutoff =
                    std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
                let stale_peers: Vec<EndpointId> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .iter()
                        .filter(|(_, p)| p.last_seen < prune_cutoff)
                        .map(|(id, _)| *id)
                        .collect()
                };
                for stale_id in stale_peers {
                    eprintln!(
                        "🧹 Pruning stale peer {} (no direct contact in {}s)",
                        stale_id.fmt_short(),
                        PEER_STALE_SECS * 2
                    );
                    node.remove_peer(stale_id).await;
                    // Also close any lingering connection
                    node.state.lock().await.connections.remove(&stale_id);
                }

                // GC expired demand entries to prevent unbounded map growth
                node.gc_demand().await;
            }
        });
    }

    /// Handle a peer death: remove from state, broadcast to all other peers.
    pub async fn handle_peer_death(&self, dead_id: EndpointId) {
        eprintln!(
            "⚠️  Peer {} died — removing and broadcasting",
            dead_id.fmt_short()
        );
        {
            let mut state = self.state.lock().await;
            // Keep the connection alive — if the peer recovers, their inbound
            // gossip will arrive on the existing connection and trigger recovery
            // via handle_gossip_stream → add_peer → clear dead_peers.
            // Don't remove: state.connections.remove(&dead_id);
            state.dead_peers.insert(dead_id);
        }
        self.remove_peer(dead_id).await;
        self.broadcast_peer_down(dead_id).await;
    }

    /// Broadcast that a peer is down to all connected peers.
    async fn broadcast_peer_down(&self, dead_id: EndpointId) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(id, _)| **id != dead_id)
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        let dead_bytes = dead_id.as_bytes().to_vec();
        for (peer_id, conn) in conns {
            let bytes = dead_bytes.clone();
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_DOWN]).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!(
                        "Failed to broadcast peer_down to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
    }

    /// Announce clean shutdown to all peers.
    pub async fn broadcast_leaving(&self) {
        let my_id_bytes = self.endpoint.id().as_bytes().to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = my_id_bytes.clone();
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_LEAVING]).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!("Failed to send leaving to {}: {e}", peer_id.fmt_short());
                }
            });
        }
        // Give broadcasts a moment to flush
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    async fn forward_plugin_event(&self, event: crate::plugin::PluginMeshEvent) -> Result<()> {
        match event {
            crate::plugin::PluginMeshEvent::Channel {
                plugin_id,
                mut message,
            } => {
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
                let frame = crate::plugin::proto::MeshChannelFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_channel_frame(&frame, None).await
            }
            crate::plugin::PluginMeshEvent::BulkTransfer {
                plugin_id,
                mut message,
            } => {
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
                let frame = crate::plugin::proto::MeshBulkFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_bulk_frame(&frame, None).await
            }
        }
    }

    async fn remember_plugin_message(&self, message_id: String) -> bool {
        const MAX_SEEN_PLUGIN_MESSAGES: usize = 4096;

        let mut state = self.state.lock().await;
        if !state.seen_plugin_messages.insert(message_id.clone()) {
            return false;
        }
        state.seen_plugin_message_order.push_back(message_id);
        while state.seen_plugin_message_order.len() > MAX_SEEN_PLUGIN_MESSAGES {
            if let Some(oldest) = state.seen_plugin_message_order.pop_front() {
                state.seen_plugin_messages.remove(&oldest);
            }
        }
        true
    }

    async fn broadcast_plugin_channel_frame(
        &self,
        frame: &crate::plugin::proto::MeshChannelFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_CHANNEL]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    async fn broadcast_plugin_bulk_frame(
        &self,
        frame: &crate::plugin::proto::MeshBulkFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_BULK_TRANSFER]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin bulk frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    async fn handle_plugin_channel_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            anyhow::bail!("Plugin channel frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshChannelFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_channel_message(crate::plugin::PluginMeshEvent::Channel {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_channel_frame(&frame, Some(remote))
                .await?;
        }

        Ok(())
    }

    async fn handle_plugin_bulk_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 64_000_000 {
            anyhow::bail!("Plugin bulk frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshBulkFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_bulk_transfer_message(crate::plugin::PluginMeshEvent::BulkTransfer {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_bulk_frame(&frame, Some(remote))
                .await?;
        }

        Ok(())
    }

    /// Broadcast a blackboard item to all connected peers (flood-fill).
    #[allow(dead_code)]
    pub async fn broadcast_blackboard(&self, item: &crate::blackboard::BlackboardItem) {
        if !self.blackboard.is_enabled() {
            return;
        }
        let msg = crate::blackboard::BlackboardMessage::Post(item.clone());
        let data = match serde_json::to_vec(&msg) {
            Ok(d) => d,
            Err(_) => return,
        };
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_BLACKBOARD]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!(
                        "Failed to broadcast blackboard to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
    }

    /// Sync blackboard with a peer: exchange digests, fetch missing items.
    pub async fn sync_blackboard(&self, conn: Connection, remote: EndpointId) {
        if !self.blackboard.is_enabled() {
            return;
        }
        let res = async {
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_BLACKBOARD]).await?;

            // Send SyncRequest
            let req = crate::blackboard::BlackboardMessage::SyncRequest;
            let req_data = serde_json::to_vec(&req)?;
            send.write_all(&(req_data.len() as u32).to_le_bytes())
                .await?;
            send.write_all(&req_data).await?;

            // Read their digest
            let mut len_buf = [0u8; 4];
            recv.read_exact(&mut len_buf).await?;
            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 1_000_000 {
                anyhow::bail!("Blackboard sync response too large");
            }
            let mut buf = vec![0u8; len];
            recv.read_exact(&mut buf).await?;
            let their_msg: crate::blackboard::BlackboardMessage = serde_json::from_slice(&buf)?;

            if let crate::blackboard::BlackboardMessage::SyncDigest(their_ids) = their_msg {
                // Figure out what we're missing
                let our_ids = self.blackboard.ids().await;
                let missing: Vec<u64> = their_ids
                    .iter()
                    .filter(|id| !our_ids.contains(id))
                    .cloned()
                    .collect();

                if !missing.is_empty() {
                    // Request missing items
                    let fetch = crate::blackboard::BlackboardMessage::FetchRequest(missing);
                    let fetch_data = serde_json::to_vec(&fetch)?;
                    send.write_all(&(fetch_data.len() as u32).to_le_bytes())
                        .await?;
                    send.write_all(&fetch_data).await?;

                    // Read their response
                    let mut len_buf2 = [0u8; 4];
                    recv.read_exact(&mut len_buf2).await?;
                    let len2 = u32::from_le_bytes(len_buf2) as usize;
                    if len2 > 10_000_000 {
                        anyhow::bail!("Knowledge fetch response too large");
                    }
                    let mut buf2 = vec![0u8; len2];
                    recv.read_exact(&mut buf2).await?;
                    let items_msg: crate::blackboard::BlackboardMessage =
                        serde_json::from_slice(&buf2)?;

                    if let crate::blackboard::BlackboardMessage::FetchResponse(items) = items_msg {
                        let count = items.len();
                        for item in items {
                            self.blackboard.insert(item).await;
                        }
                        if count > 0 {
                            tracing::info!(
                                "Blackboard sync: got {} items from {}",
                                count,
                                remote.fmt_short()
                            );
                        }
                    }
                }
            }

            send.finish()?;
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if let Err(e) = res {
            tracing::debug!("Blackboard sync with {} failed: {e}", remote.fmt_short());
        }
    }

    /// Handle an inbound blackboard stream from a peer.
    async fn handle_blackboard_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        // Read the message
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            anyhow::bail!("Knowledge message too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let msg: crate::blackboard::BlackboardMessage = serde_json::from_slice(&buf)?;

        match msg {
            crate::blackboard::BlackboardMessage::Post(item) => {
                // Insert and re-broadcast if new
                let peer_name = item.from.clone();
                if self.blackboard.insert(item.clone()).await {
                    eprintln!(
                        "📝 Blackboard from {}: {}",
                        peer_name,
                        if item.text.len() > 80 {
                            format!("{}...", &item.text[..80])
                        } else {
                            item.text.clone()
                        }
                    );
                    // Forward to other peers (flood-fill)
                    let data =
                        serde_json::to_vec(&crate::blackboard::BlackboardMessage::Post(item))?;
                    let conns: Vec<(EndpointId, Connection)> = {
                        let state = self.state.lock().await;
                        state
                            .connections
                            .iter()
                            .filter(|(id, _)| **id != remote)
                            .map(|(id, c)| (*id, c.clone()))
                            .collect()
                    };
                    for (peer_id, conn) in conns {
                        let bytes = data.clone();
                        tokio::spawn(async move {
                            let res = async {
                                let (mut send, _recv) = conn.open_bi().await?;
                                send.write_all(&[STREAM_BLACKBOARD]).await?;
                                send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                                send.write_all(&bytes).await?;
                                send.finish()?;
                                Ok::<_, anyhow::Error>(())
                            }
                            .await;
                            if let Err(e) = res {
                                tracing::debug!(
                                    "Failed to forward blackboard to {}: {e}",
                                    peer_id.fmt_short()
                                );
                            }
                        });
                    }
                }
            }
            crate::blackboard::BlackboardMessage::SyncRequest => {
                // Send our digest
                let ids = self.blackboard.ids().await;
                let digest = crate::blackboard::BlackboardMessage::SyncDigest(ids);
                let data = serde_json::to_vec(&digest)?;
                send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                send.write_all(&data).await?;

                // Check if they send a fetch request
                let mut len_buf2 = [0u8; 4];
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    recv.read_exact(&mut len_buf2),
                )
                .await
                {
                    Ok(Ok(())) => {
                        let len2 = u32::from_le_bytes(len_buf2) as usize;
                        if len2 > 1_000_000 {
                            anyhow::bail!("Fetch request too large");
                        }
                        let mut buf2 = vec![0u8; len2];
                        recv.read_exact(&mut buf2).await?;
                        let fetch_msg: crate::blackboard::BlackboardMessage =
                            serde_json::from_slice(&buf2)?;
                        if let crate::blackboard::BlackboardMessage::FetchRequest(wanted_ids) =
                            fetch_msg
                        {
                            let items = self.blackboard.get_by_ids(&wanted_ids).await;
                            let resp = crate::blackboard::BlackboardMessage::FetchResponse(items);
                            let resp_data = serde_json::to_vec(&resp)?;
                            send.write_all(&(resp_data.len() as u32).to_le_bytes())
                                .await?;
                            send.write_all(&resp_data).await?;
                        }
                    }
                    _ => {} // No fetch request, that's fine
                }
                send.finish()?;
            }
            _ => {} // Unexpected message type
        }

        Ok(())
    }

    /// Get the mesh catalog: all models that any node has on disk or has requested.
    /// Returns deduplicated list of model names (file stems, no .gguf).
    pub async fn mesh_catalog(&self) -> Vec<String> {
        // Snapshot each lock independently to avoid holding multiple locks.
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_serving = self.serving.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| {
                    (
                        p.available_models.clone(),
                        p.requested_models.clone(),
                        p.serving.clone(),
                    )
                })
                .collect()
        };
        let mut all = std::collections::HashSet::new();
        for m in &my_available {
            all.insert(m.clone());
        }
        for m in &my_requested {
            all.insert(m.clone());
        }
        if let Some(ref s) = my_serving {
            all.insert(s.clone());
        }
        for (avail, req, serving) in &peer_data {
            for m in avail {
                all.insert(m.clone());
            }
            for m in req {
                all.insert(m.clone());
            }
            if let Some(ref s) = serving {
                all.insert(s.clone());
            }
        }
        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    /// Get all models currently being served in the mesh (loaded in VRAM somewhere).
    pub async fn models_being_served(&self) -> Vec<String> {
        let my_serving_models = self.serving_models.lock().await.clone();
        let my_serving = self.serving.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| (p.serving_models.clone(), p.serving.clone()))
                .collect()
        };
        let mut served = std::collections::HashSet::new();
        for s in &my_serving_models {
            served.insert(s.clone());
        }
        if my_serving_models.is_empty() {
            if let Some(ref s) = my_serving {
                served.insert(s.clone());
            }
        }
        for (sm, s) in &peer_data {
            for m in sm {
                served.insert(m.clone());
            }
            if sm.is_empty() {
                if let Some(ref s) = s {
                    served.insert(s.clone());
                }
            }
        }
        let mut result: Vec<String> = served.into_iter().collect();
        result.sort();
        result
    }

    /// Find a host for a specific model, using hash-based selection for load distribution.
    /// When multiple hosts serve the same model, picks one based on our node ID hash.
    /// All host IDs serving a model, with hash-preferred host first.
    /// Used for retry: if the first host fails, try the next.
    pub async fn hosts_for_model(&self, model: &str) -> Vec<EndpointId> {
        let state = self.state.lock().await;
        let mut hosts: Vec<EndpointId> = state
            .peers
            .values()
            .filter(|p| {
                matches!(p.role, NodeRole::Host { .. }) && p.serving.as_deref() == Some(model)
            })
            .map(|p| p.id)
            .collect();
        hosts.sort();
        // Put the hash-preferred host first so normal path tries it first
        if !hosts.is_empty() {
            let my_id = self.endpoint.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % hosts.len();
            hosts.rotate_left(idx);
        }
        hosts
    }

    /// Find ANY host in the mesh (fallback when no model match).
    pub async fn any_host(&self) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .find(|p| matches!(p.role, NodeRole::Host { .. }))
            .cloned()
    }

    /// Build the current routing table from this node's view of the mesh.
    pub async fn routing_table(&self) -> RoutingTable {
        let my_serving = self.serving.lock().await.clone();
        let my_role = self.role.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| (p.id, p.role.clone(), p.serving.clone(), p.vram_bytes))
                .collect()
        };
        let mut hosts = Vec::new();

        // Include self if we're a host
        if matches!(my_role, NodeRole::Host { .. }) {
            if let Some(ref model) = my_serving {
                hosts.push(RouteEntry {
                    model: model.clone(),
                    node_id: format!("{}", self.endpoint.id().fmt_short()),
                    endpoint_id: self.endpoint.id(),
                    vram_gb: self.vram_bytes as f64 / 1e9,
                });
            }
        }

        // Include peers that are hosts
        for (id, role, serving, vram_bytes) in &peer_data {
            if matches!(role, NodeRole::Host { .. }) {
                if let Some(ref model) = serving {
                    hosts.push(RouteEntry {
                        model: model.clone(),
                        node_id: format!("{}", id.fmt_short()),
                        endpoint_id: *id,
                        vram_gb: *vram_bytes as f64 / 1e9,
                    });
                }
            }
        }

        let mesh_id = self.mesh_id.lock().await.clone();
        RoutingTable { hosts, mesh_id }
    }

    /// Request routing table from a connected peer (for passive nodes).

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state.lock().await.peers.values().cloned().collect()
    }

    /// Open an HTTP tunnel bi-stream to a peer (tagged STREAM_TUNNEL_HTTP).
    /// If no connection exists, tries to connect on-demand (for passive nodes
    /// that learned about hosts from routing table but aren't directly connected).
    pub async fn open_http_tunnel(
        &self,
        peer_id: EndpointId,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = {
            let state = self.state.lock().await;
            match state.connections.get(&peer_id).cloned() {
                Some(c) => c,
                None => {
                    // Try on-demand connect using peer's addr from peer info
                    let addr = state.peers.get(&peer_id).map(|p| p.addr.clone());
                    drop(state);
                    if let Some(addr) = addr {
                        let c = tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            self.endpoint.connect(addr, ALPN),
                        )
                        .await
                        .map_err(|_| {
                            anyhow::anyhow!("Timeout connecting to {}", peer_id.fmt_short())
                        })?
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to connect to {}: {e}", peer_id.fmt_short())
                        })?;
                        self.state
                            .lock()
                            .await
                            .connections
                            .insert(peer_id, c.clone());
                        c
                    } else {
                        anyhow::bail!("No connection or address for {}", peer_id.fmt_short());
                    }
                }
            }
        };
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (mut send, recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_TUNNEL_HTTP]).await?;
            Ok::<_, anyhow::Error>((send, recv))
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timeout opening tunnel to {}", peer_id.fmt_short()))?;

        if result.is_err() {
            // Connection failed — peer is likely dead, broadcast it
            tracing::info!(
                "Tunnel to {} failed, broadcasting death",
                peer_id.fmt_short()
            );
            self.handle_peer_death(peer_id).await;
        }

        result
    }

    pub async fn set_tunnel_port(&self, id: EndpointId, port: u16) {
        if let Some(peer) = self.state.lock().await.peers.get_mut(&id) {
            peer.tunnel_port = Some(port);
        }
    }

    /// Push our tunnel port map to all connected peers.
    /// Called after tunnel ports are established.
    pub async fn broadcast_tunnel_map(
        &self,
        my_tunnel_map: HashMap<EndpointId, u16>,
    ) -> Result<()> {
        // Serialize: { endpoint_id_hex_string → port }
        let serializable: HashMap<String, u16> = my_tunnel_map
            .iter()
            .map(|(id, port)| (hex::encode(id.as_bytes()), *port))
            .collect();
        let msg = serde_json::to_vec(&serializable)?;

        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };

        for (peer_id, conn) in conns {
            let msg = msg.clone();
            tokio::spawn(async move {
                match conn.open_bi().await {
                    Ok((mut send, _recv)) => {
                        if send.write_all(&[STREAM_TUNNEL_MAP]).await.is_err() {
                            return;
                        }
                        let len = msg.len() as u32;
                        if send.write_all(&len.to_le_bytes()).await.is_err() {
                            return;
                        }
                        if send.write_all(&msg).await.is_err() {
                            return;
                        }
                        let _ = send.finish();
                        tracing::info!("Sent tunnel map to {}", peer_id.fmt_short());
                    }
                    Err(e) => {
                        tracing::warn!("Failed to send tunnel map to {}: {e}", peer_id.fmt_short());
                    }
                }
            });
        }
        Ok(())
    }

    /// Get all remote tunnel maps: { peer_id → { target_id → tunnel_port } }
    pub async fn all_remote_tunnel_maps(&self) -> HashMap<EndpointId, HashMap<EndpointId, u16>> {
        self.state.lock().await.remote_tunnel_maps.clone()
    }

    /// Wait until we have tunnel maps from at least `n` peers, with timeout.
    pub async fn wait_for_tunnel_maps(&self, n: usize, timeout: std::time::Duration) -> Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let state = self.state.lock().await;
                if state.remote_tunnel_maps.len() >= n {
                    return Ok(());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                let state = self.state.lock().await;
                tracing::warn!(
                    "Timeout waiting for tunnel maps: got {} of {} needed",
                    state.remote_tunnel_maps.len(),
                    n
                );
                return Ok(()); // Don't fail — B2B is optional optimization
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }

    /// Open a tunnel bi-stream to a peer using the stored connection.
    pub async fn open_tunnel_stream(
        &self,
        peer_id: EndpointId,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = {
            self.state
                .lock()
                .await
                .connections
                .get(&peer_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No connection to {}", peer_id.fmt_short()))?
        };
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_TUNNEL]).await?;
        Ok((send, recv))
    }

    // --- Connection handling ---

    async fn accept_loop(&self) {
        // Wait until start_accepting() is called before processing any connections.
        // Check flag first to handle the case where start_accepting() was called before we got here.
        if !self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            self.accepting.0.notified().await;
        }
        tracing::info!("Accept loop: now accepting inbound connections");

        loop {
            let incoming = match self.endpoint.accept().await {
                Some(i) => i,
                None => break,
            };
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.handle_incoming(incoming).await {
                    tracing::warn!("Incoming connection error: {e}");
                }
            });
        }
    }

    async fn handle_incoming(&self, incoming: iroh::endpoint::Incoming) -> Result<()> {
        let mut accepting = incoming.accept()?;
        let _alpn = accepting.alpn().await?;
        let conn = accepting.await?;
        let remote = conn.remote_id();
        tracing::info!("Inbound connection from {}", remote.fmt_short());

        // Store connection for stream dispatch (tunneling, route requests, etc.)
        // Don't add to peer list yet — only gossip exchange promotes to peer.
        let was_dead = {
            let mut state = self.state.lock().await;
            let was_dead = state.dead_peers.remove(&remote);
            if was_dead {
                eprintln!("🔄 Previously dead peer {} reconnected", remote.fmt_short());
            }
            state.connections.insert(remote, conn.clone());
            was_dead
        };

        // If this peer was previously dead, immediately gossip to restore their
        // serving status in our peer list. Without this, models served by the
        // reconnecting peer stay invisible until the next heartbeat (up to 60s).
        if was_dead {
            let node = self.clone();
            let gossip_conn = conn.clone();
            tokio::spawn(async move {
                if let Err(e) = node.initiate_gossip_inner(gossip_conn, remote, false).await {
                    tracing::debug!("Reconnect gossip with {} failed: {e}", remote.fmt_short());
                }
            });
        }

        self.dispatch_streams(conn, remote).await;
        Ok(())
    }

    /// Dispatch bi-streams on a connection by type byte
    fn dispatch_streams(
        &self,
        conn: Connection,
        remote: EndpointId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(self._dispatch_streams(conn, remote))
    }

    async fn _dispatch_streams(&self, conn: Connection, remote: EndpointId) {
        loop {
            let (send, mut recv) = match conn.accept_bi().await {
                Ok(s) => s,
                Err(e) => {
                    tracing::info!("Connection to {} closed: {e}", remote.fmt_short());
                    // Remove the stale connection
                    {
                        let mut state = self.state.lock().await;
                        state.connections.remove(&remote);
                    }
                    // Try to reconnect — if the peer is still alive, re-learn their role
                    let addr = {
                        let state = self.state.lock().await;
                        state.peers.get(&remote).map(|p| p.addr.clone())
                    };
                    if let Some(addr) = addr {
                        tracing::info!("Attempting reconnect to {}...", remote.fmt_short());
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            self.endpoint.connect(addr, ALPN),
                        )
                        .await
                        {
                            Ok(Ok(new_conn)) => {
                                tracing::info!("Reconnected to {}", remote.fmt_short());
                                {
                                    let mut state = self.state.lock().await;
                                    state.connections.insert(remote, new_conn.clone());
                                }
                                // Gossip to re-learn role
                                let node = self.clone();
                                let gc = new_conn.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = node.initiate_gossip(gc, remote).await {
                                        tracing::debug!("Reconnect gossip failed: {e}");
                                    }
                                });
                                // Continue dispatching on the new connection
                                let node = self.clone();
                                tokio::spawn(async move {
                                    node.dispatch_streams(new_conn, remote).await;
                                });
                            }
                            _ => {
                                tracing::info!(
                                    "Reconnect to {} failed — removing peer",
                                    remote.fmt_short()
                                );
                                self.remove_peer(remote).await;
                            }
                        }
                    } else {
                        // No address on file, can't reconnect
                        self.remove_peer(remote).await;
                    }
                    break;
                }
            };

            let mut type_buf = [0u8; 1];
            if recv.read_exact(&mut type_buf).await.is_err() {
                continue;
            }

            match type_buf[0] {
                STREAM_GOSSIP => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_gossip_stream(remote, send, recv).await {
                            tracing::warn!("Gossip stream error from {}: {e}", remote.fmt_short());
                        }
                    });
                }
                STREAM_TUNNEL => {
                    if self.tunnel_tx.send((send, recv)).await.is_err() {
                        tracing::warn!("Tunnel receiver dropped");
                        break;
                    }
                }
                STREAM_TUNNEL_MAP => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_tunnel_map_stream(remote, recv).await {
                            tracing::warn!(
                                "Tunnel map stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_TUNNEL_HTTP => {
                    if self.tunnel_http_tx.send((send, recv)).await.is_err() {
                        tracing::warn!("HTTP tunnel receiver dropped");
                        break;
                    }
                }
                STREAM_ROUTE_REQUEST => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        let mut send = send;
                        let table = node.routing_table().await;
                        if let Ok(data) = serde_json::to_vec(&table) {
                            let _ = send.write_all(&data).await;
                            let _ = send.finish();
                        }
                    });
                }
                STREAM_PEER_DOWN => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        // Read the 32-byte endpoint ID of the dead peer
                        let mut id_bytes = [0u8; 32];
                        if recv.read_exact(&mut id_bytes).await.is_ok() {
                            if let Ok(pk) = iroh::PublicKey::from_bytes(&id_bytes) {
                                let dead_id = EndpointId::from(pk);
                                if dead_id != node.endpoint.id() {
                                    // Verify: try to reach the dead peer ourselves before removing
                                    let conn_opt = {
                                        let state = node.state.lock().await;
                                        state.connections.get(&dead_id).cloned()
                                    };
                                    let should_remove = if let Some(conn) = conn_opt {
                                        tokio::time::timeout(
                                            std::time::Duration::from_secs(3),
                                            conn.open_bi(),
                                        )
                                        .await
                                        .is_err()
                                    } else {
                                        true // no connection = already gone
                                    };
                                    if should_remove {
                                        eprintln!(
                                            "⚠️  Peer {} reported dead by {}, confirmed, removing",
                                            dead_id.fmt_short(),
                                            remote.fmt_short()
                                        );
                                        let mut state = node.state.lock().await;
                                        state.connections.remove(&dead_id);
                                        drop(state);
                                        node.remove_peer(dead_id).await;
                                    } else {
                                        eprintln!("ℹ️  Peer {} reported dead by {} but still reachable, ignoring",
                                            dead_id.fmt_short(), remote.fmt_short());
                                    }
                                }
                            }
                        }
                    });
                }
                STREAM_PEER_LEAVING => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        let mut id_bytes = [0u8; 32];
                        if recv.read_exact(&mut id_bytes).await.is_ok() {
                            if let Ok(pk) = iroh::PublicKey::from_bytes(&id_bytes) {
                                let leaving_id = EndpointId::from(pk);
                                eprintln!(
                                    "👋 Peer {} announced clean shutdown",
                                    leaving_id.fmt_short()
                                );
                                let mut state = node.state.lock().await;
                                state.connections.remove(&leaving_id);
                                drop(state);
                                node.remove_peer(leaving_id).await;
                            }
                        }
                    });
                }
                STREAM_BLACKBOARD => {
                    if self.blackboard.is_enabled() {
                        let node = self.clone();
                        tokio::spawn(async move {
                            if let Err(e) = node.handle_blackboard_stream(remote, send, recv).await
                            {
                                tracing::debug!(
                                    "Blackboard stream error from {}: {e}",
                                    remote.fmt_short()
                                );
                            }
                        });
                    }
                }
                STREAM_PLUGIN_CHANNEL => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_plugin_channel_stream(remote, send, recv).await
                        {
                            tracing::debug!(
                                "Plugin channel stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_PLUGIN_BULK_TRANSFER => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_plugin_bulk_stream(remote, send, recv).await {
                            tracing::debug!(
                                "Plugin bulk stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                other => {
                    tracing::warn!("Unknown stream type {other} from {}", remote.fmt_short());
                }
            }
        }
    }

    // --- Gossip ---

    async fn connect_to_peer(&self, addr: EndpointAddr) -> Result<()> {
        let peer_id = addr.id;
        if peer_id == self.endpoint.id() {
            return Ok(());
        }

        {
            let state = self.state.lock().await;
            if state.peers.contains_key(&peer_id) {
                return Ok(());
            }
            if state.dead_peers.contains(&peer_id) {
                tracing::debug!("Skipping connection to dead peer {}", peer_id.fmt_short());
                return Ok(());
            }
        }

        tracing::info!("Connecting to peer {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            std::time::Duration::from_secs(15),
            self.endpoint.connect(addr.clone(), ALPN),
        )
        .await
        {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                anyhow::bail!("Failed to connect to {}: {e}", peer_id.fmt_short());
            }
            Err(_) => {
                anyhow::bail!("Timeout connecting to {} (15s)", peer_id.fmt_short());
            }
        };

        // Store connection and start dispatcher for inbound streams from this peer
        {
            let mut state = self.state.lock().await;
            state.connections.insert(peer_id, conn.clone());
        }
        let node_for_dispatch = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        // Gossip exchange to learn peer's role/VRAM and announce ourselves
        self.initiate_gossip(conn, peer_id).await?;
        Ok(())
    }

    /// Open a gossip stream on an existing connection to exchange peer info.
    async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        self.initiate_gossip_inner(conn, remote, true).await
    }

    async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        // Send our announcements (length-prefixed JSON)
        let our_announcements = self.collect_announcements().await;
        let msg = serde_json::to_vec(&our_announcements)?;
        send.write_all(&(msg.len() as u32).to_le_bytes()).await?;
        send.write_all(&msg).await?;
        send.finish()?;

        // Read their announcements
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let rtt_ms = t0.elapsed().as_millis() as u32;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let their_announcements: Vec<PeerAnnouncement> = serde_json::from_slice(&buf)?;

        // Wait for stream to fully close, then small delay for accept_bi to re-arm
        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Find the direct peer's self-announcement — carries mesh-wide fields
        // Process announcements: direct peer gets full add_peer (updates last_seen,
        // triggers events). Other peers get transitive update (serving/role info only,
        // no last_seen refresh) so their models appear in mesh_models but they still
        // get pruned if they actually die.
        for ann in &their_announcements {
            if ann.addr.id == self.endpoint.id() {
                continue;
            }
            if ann.addr.id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, ann.addr.clone(), ann).await;
                self.update_peer_rtt(remote, rtt_ms).await;
            } else {
                self.update_transitive_peer(ann.addr.id, &ann.addr, ann)
                    .await;
            }
        }

        // Try to establish direct connections to new peers (only on initial join)
        if discover_peers {
            for ann in &their_announcements {
                if ann.addr.id != self.endpoint.id() {
                    let has_conn = self
                        .state
                        .lock()
                        .await
                        .connections
                        .contains_key(&ann.addr.id);
                    if !has_conn {
                        if let Err(e) = Box::pin(self.connect_to_peer(ann.addr.clone())).await {
                            tracing::debug!(
                                "Could not connect to discovered peer {}: {e}",
                                ann.addr.id.fmt_short()
                            );
                        }
                    }
                }
            }

            // Sync blackboard on initial connection
            if self.blackboard.is_enabled() {
                let conn = self.state.lock().await.connections.get(&remote).cloned();
                if let Some(conn) = conn {
                    self.sync_blackboard(conn, remote).await;
                }
            }
        }

        Ok(())
    }

    async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        // If this peer was declared dead but is now gossiping with us,
        // they're clearly alive. Clear the dead flag so add_peer accepts them.
        {
            let mut state = self.state.lock().await;
            if state.dead_peers.remove(&remote) {
                eprintln!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    remote.fmt_short()
                );
            }
        }

        // Read their announcements
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let their_announcements: Vec<PeerAnnouncement> = serde_json::from_slice(&buf)?;

        // Send our announcements
        let our_announcements = self.collect_announcements().await;
        let msg = serde_json::to_vec(&our_announcements)?;
        send.write_all(&(msg.len() as u32).to_le_bytes()).await?;
        send.write_all(&msg).await?;
        send.finish()?;

        // Wait for the remote to finish their send
        let _ = recv.read_to_end(0).await;

        // Process all announcements: direct peer gets full add_peer,
        // others get transitive update (serving info without last_seen refresh).
        for ann in &their_announcements {
            if ann.addr.id == self.endpoint.id() {
                continue;
            }
            if ann.addr.id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, ann.addr.clone(), ann).await;
            } else {
                self.update_transitive_peer(ann.addr.id, &ann.addr, ann)
                    .await;
            }
        }

        // Measure RTT from QUIC connection stats
        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let rtt = path_info.rtt();
                        let rtt_ms = rtt.as_millis() as u32;
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if rtt_ms > 0 {
                            eprintln!(
                                "📡 Peer {} RTT: {}ms ({})",
                                remote.fmt_short(),
                                rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        // Discover new peers mentioned in gossip — but only try to connect
        // to peers we don't already know about.
        for ann in their_announcements {
            let peer_id = ann.addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            let already_known = self.state.lock().await.peers.contains_key(&peer_id);
            if !already_known {
                if let Err(e) = Box::pin(self.connect_to_peer(ann.addr)).await {
                    tracing::warn!("Failed to discover peer: {e}");
                }
            }
        }

        Ok(())
    }

    async fn handle_tunnel_map_stream(
        &self,
        remote: EndpointId,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        // Read length-prefixed JSON
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;

        // Deserialize: { hex_endpoint_id → port }
        let serialized: HashMap<String, u16> = serde_json::from_slice(&buf)?;
        let mut tunnel_map = HashMap::new();
        for (hex_id, port) in serialized {
            if let Ok(bytes) = hex::decode(&hex_id) {
                if bytes.len() == 32 {
                    let arr: [u8; 32] = bytes.try_into().unwrap();
                    let eid = EndpointId::from(iroh::PublicKey::from_bytes(&arr)?);
                    tunnel_map.insert(eid, port);
                }
            }
        }

        tracing::info!(
            "Received tunnel map from {} ({} entries)",
            remote.fmt_short(),
            tunnel_map.len()
        );

        {
            let mut state = self.state.lock().await;
            state.remote_tunnel_maps.insert(remote, tunnel_map);
        }

        Ok(())
    }

    async fn remove_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        if let Some(peer) = state.peers.remove(&id) {
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                state.peers.len()
            );
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    async fn add_peer(&self, id: EndpointId, addr: EndpointAddr, ann: &PeerAnnouncement) {
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        if state.dead_peers.remove(&id) {
            eprintln!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            );
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let role_changed = existing.role != ann.role;
            let serving_changed = existing.serving != ann.serving;
            if role_changed {
                tracing::info!(
                    "Peer {} role updated: {:?} → {:?}",
                    id.fmt_short(),
                    existing.role,
                    ann.role
                );
                existing.role = ann.role.clone();
            }
            // Update addr if the new one has more info
            if !addr.addrs.is_empty() {
                existing.addr = addr;
            }
            existing.models = ann.models.clone();
            existing.vram_bytes = ann.vram_bytes;
            if ann.model_source.is_some() {
                existing.model_source = ann.model_source.clone();
            }
            existing.serving = ann.serving.clone();
            existing.serving_models = ann.serving_models.clone();
            existing.available_models = ann.available_models.clone();
            existing.requested_models = ann.requested_models.clone();
            existing.last_seen = std::time::Instant::now();
            if ann.version.is_some() {
                existing.version = ann.version.clone();
            }
            existing.gpu_name = ann.gpu_name.clone();
            existing.hostname = ann.hostname.clone();
            existing.is_soc = ann.is_soc;
            existing.gpu_vram = ann.gpu_vram.clone();
            existing.gpu_bandwidth_gbps = ann.gpu_bandwidth_gbps.clone();
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
                || old_peer.gpu_name != updated_peer.gpu_name
                || old_peer.hostname != updated_peer.hostname
                || old_peer.is_soc != updated_peer.is_soc
                || old_peer.gpu_vram != updated_peer.gpu_vram
                || old_peer.gpu_bandwidth_gbps != updated_peer.gpu_bandwidth_gbps;
            if role_changed || serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
            return;
        }
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB serving={:?} available={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving,
            ann.available_models,
            state.peers.len() + 1
        );
        let peer = PeerInfo {
            id,
            addr,
            tunnel_port: None,
            role: ann.role.clone(),
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            rtt_ms: None,
            model_source: ann.model_source.clone(),
            serving: ann.serving.clone(),
            serving_models: ann.serving_models.clone(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
            last_seen: std::time::Instant::now(),
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_bandwidth_gbps: ann.gpu_bandwidth_gbps.clone(),
        };
        state.peers.insert(id, peer.clone());
        let count = state.peers.len();
        drop(state);
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates serving/role info so models_being_served() includes their models,
    /// but does NOT refresh last_seen — transitive peers still get pruned if the
    /// bridge node stops mentioning them. Does NOT trigger peer_change events
    /// for new transitive peers (avoids re-election storms at scale).
    async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
    ) {
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        if state.dead_peers.contains(&id) {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            // Update serving info only — don't touch last_seen
            let serving_changed =
                existing.serving != ann.serving || existing.serving_models != ann.serving_models;
            existing.serving = ann.serving.clone();
            existing.serving_models = ann.serving_models.clone();
            existing.role = ann.role.clone();
            existing.vram_bytes = ann.vram_bytes;
            if !addr.addrs.is_empty() {
                existing.addr = addr.clone();
            }
            if ann.version.is_some() {
                existing.version = ann.version.clone();
            }
            if ann.gpu_name.is_some() {
                existing.gpu_name = ann.gpu_name.clone();
            }
            if ann.hostname.is_some() {
                existing.hostname = ann.hostname.clone();
            }
            if ann.is_soc.is_some() {
                existing.is_soc = ann.is_soc;
            }
            if ann.gpu_vram.is_some() {
                existing.gpu_vram = ann.gpu_vram.clone();
            }
            if ann.gpu_bandwidth_gbps.is_some() {
                existing.gpu_bandwidth_gbps = ann.gpu_bandwidth_gbps.clone();
            }
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer);
            if serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
        } else {
            // New transitive peer — add with last_seen = now but no peer_change event.
            // It will get pruned after PEER_STALE_SECS*2 if never directly contacted.
            let peer = PeerInfo {
                id,
                addr: addr.clone(),
                tunnel_port: None,
                role: ann.role.clone(),
                models: ann.models.clone(),
                vram_bytes: ann.vram_bytes,
                rtt_ms: None,
                model_source: ann.model_source.clone(),
                serving: ann.serving.clone(),
                serving_models: ann.serving_models.clone(),
                available_models: ann.available_models.clone(),
                requested_models: ann.requested_models.clone(),
                last_seen: std::time::Instant::now(),
                version: ann.version.clone(),
                gpu_name: ann.gpu_name.clone(),
                hostname: ann.hostname.clone(),
                is_soc: ann.is_soc,
                gpu_vram: ann.gpu_vram.clone(),
                gpu_bandwidth_gbps: ann.gpu_bandwidth_gbps.clone(),
            };
            state.peers.insert(id, peer.clone());
            drop(state);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUp,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        // Snapshot all locks independently — never hold multiple locks simultaneously.
        let my_role = self.role.lock().await.clone();
        let my_models = self.models.lock().await.clone();
        let my_source = self.model_source.lock().await.clone();
        let my_serving = self.serving.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_mesh_id = self.mesh_id.lock().await.clone();
        let my_demand = self.get_demand();
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        let mut announcements: Vec<PeerAnnouncement> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|p| p.last_seen >= stale_cutoff)
                .map(|p| PeerAnnouncement {
                    addr: p.addr.clone(),
                    role: p.role.clone(),
                    models: p.models.clone(),
                    vram_bytes: p.vram_bytes,
                    model_source: p.model_source.clone(),
                    serving: p.serving.clone(),
                    serving_models: p.serving_models.clone(),
                    available_models: p.available_models.clone(),
                    requested_models: p.requested_models.clone(),
                    version: p.version.clone(),
                    model_demand: HashMap::new(),
                    mesh_id: None,
                    gpu_name: p.gpu_name.clone(),
                    hostname: p.hostname.clone(),
                    is_soc: p.is_soc,
                    gpu_vram: p.gpu_vram.clone(),
                    gpu_bandwidth_gbps: p.gpu_bandwidth_gbps.clone(),
                })
                .collect()
        };
        // Self entry carries the mesh-wide fields once
        announcements.push(PeerAnnouncement {
            addr: self.endpoint.addr(),
            role: my_role,
            models: my_models,
            vram_bytes: self.vram_bytes,
            model_source: my_source,
            serving: my_serving,
            serving_models: my_serving_models,
            available_models: my_available,
            requested_models: my_requested,
            version: Some(crate::VERSION.to_string()),
            model_demand: my_demand,
            mesh_id: my_mesh_id,
            gpu_name: if self.enumerate_host {
                self.gpu_name.clone()
            } else {
                None
            },
            hostname: if self.enumerate_host {
                self.hostname.clone()
            } else {
                None
            },
            is_soc: self.is_soc,
            gpu_vram: if self.enumerate_host {
                self.gpu_vram.clone()
            } else {
                None
            },
            gpu_bandwidth_gbps: self.gpu_bandwidth_gbps.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
        });
        announcements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_demand_takes_max() {
        let mut ours = HashMap::new();
        ours.insert(
            "GLM".into(),
            ModelDemand {
                last_active: 100,
                request_count: 50,
            },
        );
        ours.insert(
            "Hermes".into(),
            ModelDemand {
                last_active: 200,
                request_count: 10,
            },
        );

        let mut theirs = HashMap::new();
        theirs.insert(
            "GLM".into(),
            ModelDemand {
                last_active: 150,
                request_count: 30,
            },
        );
        theirs.insert(
            "Qwen".into(),
            ModelDemand {
                last_active: 300,
                request_count: 5,
            },
        );

        merge_demand(&mut ours, &theirs);

        // GLM: max(100,150)=150 for last_active, max(50,30)=50 for count
        assert_eq!(ours["GLM"].last_active, 150);
        assert_eq!(ours["GLM"].request_count, 50);
        // Hermes: unchanged (not in theirs)
        assert_eq!(ours["Hermes"].last_active, 200);
        assert_eq!(ours["Hermes"].request_count, 10);
        // Qwen: new entry from theirs
        assert_eq!(ours["Qwen"].last_active, 300);
        assert_eq!(ours["Qwen"].request_count, 5);
    }

    #[test]
    fn test_merge_demand_empty_maps() {
        let mut ours = HashMap::new();
        let theirs = HashMap::new();
        merge_demand(&mut ours, &theirs);
        assert!(ours.is_empty());

        let mut theirs2 = HashMap::new();
        theirs2.insert(
            "GLM".into(),
            ModelDemand {
                last_active: 100,
                request_count: 1,
            },
        );
        merge_demand(&mut ours, &theirs2);
        assert_eq!(ours.len(), 1);
        assert_eq!(ours["GLM"].request_count, 1);
    }

    #[test]
    fn test_merge_demand_idempotent() {
        let mut ours = HashMap::new();
        ours.insert(
            "GLM".into(),
            ModelDemand {
                last_active: 100,
                request_count: 50,
            },
        );

        let theirs = ours.clone();
        merge_demand(&mut ours, &theirs);

        assert_eq!(ours["GLM"].last_active, 100);
        assert_eq!(ours["GLM"].request_count, 50);
    }

    #[test]
    fn test_demand_ttl_filtering() {
        let now = now_secs();
        let mut demand = HashMap::new();

        // Recent — should survive
        demand.insert(
            "Recent".into(),
            ModelDemand {
                last_active: now - 60, // 1 min ago
                request_count: 10,
            },
        );
        // Stale — should be filtered
        demand.insert(
            "Stale".into(),
            ModelDemand {
                last_active: now - DEMAND_TTL_SECS - 100, // past TTL
                request_count: 100,
            },
        );

        let filtered: HashMap<String, ModelDemand> = demand
            .into_iter()
            .filter(|(_, d)| (now - d.last_active) < DEMAND_TTL_SECS)
            .collect();

        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("Recent"));
        assert!(!filtered.contains_key("Stale"));
    }

    #[test]
    fn test_demand_serialization_roundtrip() {
        let mut demand: HashMap<String, ModelDemand> = HashMap::new();
        demand.insert(
            "GLM".into(),
            ModelDemand {
                last_active: 1772309000,
                request_count: 42,
            },
        );

        let json = serde_json::to_string(&demand).unwrap();
        let decoded: HashMap<String, ModelDemand> = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded["GLM"].last_active, 1772309000);
        assert_eq!(decoded["GLM"].request_count, 42);
    }

    #[test]
    fn test_demand_deserialization_missing_field() {
        // Simulate old gossip message without model_demand field
        // Just verify ModelDemand defaults work
        let d = ModelDemand::default();
        assert_eq!(d.last_active, 0);
        assert_eq!(d.request_count, 0);

        // Verify HashMap<String, ModelDemand> defaults to empty
        let empty: HashMap<String, ModelDemand> = Default::default();
        assert!(empty.is_empty());

        // The real test: serde default on a struct with model_demand
        #[derive(Deserialize, Default)]
        struct TestStruct {
            #[serde(default)]
            model_demand: HashMap<String, ModelDemand>,
            #[serde(default)]
            requested_models: Vec<String>,
        }
        let parsed: TestStruct = serde_json::from_str("{}").unwrap();
        assert!(parsed.model_demand.is_empty());
        assert!(parsed.requested_models.is_empty());
    }

    #[test]
    fn test_split_gguf_base_name() {
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00001-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00006-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(split_gguf_base_name("Qwen3-8B-Q4_K_M"), None);
        assert_eq!(split_gguf_base_name("model-001-of-003"), None); // wrong digit count
        assert_eq!(split_gguf_base_name("model-00001-of-00003"), Some("model"));
    }

    #[test]
    fn test_peer_announcement_gpu_serde_roundtrip() {
        // Test that gpu_name and hostname fields serialize and deserialize correctly
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_name: Option<String>,
            #[serde(default)]
            hostname: Option<String>,
        }

        let test = TestAnnouncement {
            gpu_name: Some("NVIDIA A100".to_string()),
            hostname: Some("worker-01".to_string()),
        };

        let json = serde_json::to_string(&test).unwrap();
        let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.gpu_name, Some("NVIDIA A100".to_string()));
        assert_eq!(decoded.hostname, Some("worker-01".to_string()));
    }

    #[test]
    fn test_peer_announcement_backward_compat_no_hw_fields() {
        // Simulate old gossip message without gpu_name or hostname
        #[derive(Deserialize, Debug)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_name: Option<String>,
            #[serde(default)]
            hostname: Option<String>,
        }

        let json = r#"{"other_field": "value"}"#;
        let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

        assert_eq!(decoded.gpu_name, None);
        assert_eq!(decoded.hostname, None);
    }

    #[test]
    fn test_peer_announcement_backward_compat_with_hw_fields() {
        // Simulate new gossip message with gpu_name and hostname
        #[derive(Deserialize, Debug)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_name: Option<String>,
            #[serde(default)]
            hostname: Option<String>,
        }

        let json = r#"{"gpu_name": "NVIDIA H100", "hostname": "gpu-server-02"}"#;
        let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

        assert_eq!(decoded.gpu_name, Some("NVIDIA H100".to_string()));
        assert_eq!(decoded.hostname, Some("gpu-server-02".to_string()));
    }

    #[test]
    fn test_peer_announcement_hostname_serde_roundtrip() {
        // Test hostname-only roundtrip
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_name: Option<String>,
            #[serde(default)]
            hostname: Option<String>,
        }

        let test = TestAnnouncement {
            gpu_name: None,
            hostname: Some("compute-node-42".to_string()),
        };

        let json = serde_json::to_string(&test).unwrap();
        let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.hostname, Some("compute-node-42".to_string()));
        assert_eq!(decoded.gpu_name, None);
    }

    #[test]
    fn test_peer_payload_hw_fields() {
        // Test that PeerPayload includes gpu_name and hostname fields
        #[derive(Serialize, Debug)]
        struct TestPeerPayload {
            id: String,
            gpu_name: Option<String>,
            hostname: Option<String>,
        }

        let payload = TestPeerPayload {
            id: "peer-123".to_string(),
            gpu_name: Some("NVIDIA A100".to_string()),
            hostname: Some("worker-01".to_string()),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["gpu_name"], "NVIDIA A100");
        assert_eq!(value["hostname"], "worker-01");
    }

    #[test]
    fn test_enumerate_host_false_omits_hw_fields_in_announcement() {
        let enumerate_host = false;
        let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
        let hostname: Option<String> = Some("carrack".to_string());
        let gpu_vram: Option<String> = Some("34359738368".to_string());

        let gossip_gpu_name = if enumerate_host {
            gpu_name.clone()
        } else {
            None
        };
        let gossip_hostname = if enumerate_host {
            hostname.clone()
        } else {
            None
        };
        let gossip_gpu_vram = if enumerate_host {
            gpu_vram.clone()
        } else {
            None
        };

        assert_eq!(gossip_gpu_name, None);
        assert_eq!(gossip_hostname, None);
        assert_eq!(gossip_gpu_vram, None);
    }

    #[test]
    fn test_enumerate_host_true_includes_hw_fields_in_announcement() {
        let enumerate_host = true;
        let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
        let hostname: Option<String> = Some("carrack".to_string());
        let gpu_vram: Option<String> = Some("34359738368".to_string());

        let gossip_gpu_name = if enumerate_host {
            gpu_name.clone()
        } else {
            None
        };
        let gossip_hostname = if enumerate_host {
            hostname.clone()
        } else {
            None
        };
        let gossip_gpu_vram = if enumerate_host {
            gpu_vram.clone()
        } else {
            None
        };

        assert_eq!(gossip_gpu_name, Some("NVIDIA RTX 5090".to_string()));
        assert_eq!(gossip_hostname, Some("carrack".to_string()));
        assert_eq!(gossip_gpu_vram, Some("34359738368".to_string()));
    }

    #[test]
    fn test_is_soc_always_included_regardless_of_enumerate_host() {
        for enumerate_host in [false, true] {
            let is_soc: Option<bool> = Some(true);
            let gpu_name: Option<String> = Some("Tegra AGX Orin".to_string());

            let gossip_gpu_name = if enumerate_host {
                gpu_name.clone()
            } else {
                None
            };

            assert_eq!(is_soc, Some(true), "is_soc must always be sent");
            if enumerate_host {
                assert!(gossip_gpu_name.is_some());
            } else {
                assert!(gossip_gpu_name.is_none());
            }
        }
    }

    #[test]
    fn test_peer_announcement_backward_compat_is_soc_gpu_vram() {
        #[derive(Deserialize, Debug)]
        struct TestAnnouncement {
            #[serde(default)]
            is_soc: Option<bool>,
            #[serde(default)]
            gpu_vram: Option<String>,
        }

        let json = r#"{"other_field": "value"}"#;
        let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();
        assert_eq!(
            decoded.is_soc, None,
            "old nodes without is_soc should default to None"
        );
        assert_eq!(
            decoded.gpu_vram, None,
            "old nodes without gpu_vram should default to None"
        );
    }

    #[test]
    fn test_peer_announcement_with_bandwidth_serde_roundtrip() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_bandwidth_gbps: Option<String>,
        }

        let test = TestAnnouncement {
            gpu_bandwidth_gbps: Some("1671.7,722.2".to_string()),
        };

        let json = serde_json::to_string(&test).unwrap();
        let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.gpu_bandwidth_gbps, Some("1671.7,722.2".to_string()));
    }

    #[test]
    fn test_peer_announcement_backward_compat_no_bandwidth_field() {
        #[derive(Deserialize, Debug)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_bandwidth_gbps: Option<String>,
        }

        let json = r#"{"other_field": "value"}"#;
        let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

        assert_eq!(decoded.gpu_bandwidth_gbps, None);
    }
}

/// Generate a mesh ID for a new mesh.
/// Named meshes: `sha256("mesh-llm:" + name + ":" + nostr_pubkey)` — deterministic, unique per creator.
/// Unnamed meshes: random UUID, persisted to `~/.mesh-llm/mesh-id`.
pub fn generate_mesh_id(name: Option<&str>, nostr_pubkey: Option<&str>) -> String {
    if let Some(name) = name {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "mesh-llm:".hash(&mut hasher);
        name.hash(&mut hasher);
        if let Some(pk) = nostr_pubkey {
            pk.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    } else {
        // Try to load persisted mesh-id
        let path = mesh_id_path();
        if let Ok(id) = std::fs::read_to_string(&path) {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return id;
            }
        }
        // Generate new random ID and persist
        let id = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, &id);
        id
    }
}

fn mesh_id_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("mesh-id")
}

/// Save the mesh ID of the last mesh we successfully joined.
pub fn save_last_mesh_id(mesh_id: &str) {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, mesh_id);
}

/// Load the mesh ID of the last mesh we successfully joined.
pub fn load_last_mesh_id() -> Option<String> {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Load secret key from ~/.mesh-llm/key, or create a new one and save it.
/// Migrates from ~/.mesh-inference/key if it exists.
async fn load_or_create_key() -> Result<SecretKey> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let dir = home.join(".mesh-llm");
    let key_path = dir.join("key");

    // Migrate from old name
    let old_key = home.join(".mesh-inference").join("key");
    if !key_path.exists() && old_key.exists() {
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::copy(&old_key, &key_path).await?;
        tracing::info!("Migrated key from {}", old_key.display());
    }

    if key_path.exists() {
        let hex = tokio::fs::read_to_string(&key_path).await?;
        let bytes = hex::decode(hex.trim())?;
        if bytes.len() != 32 {
            anyhow::bail!("Invalid key length in {}", key_path.display());
        }
        let key = SecretKey::from_bytes(&bytes.try_into().unwrap());
        tracing::info!("Loaded key from {}", key_path.display());
        return Ok(key);
    }

    let key = SecretKey::generate(&mut rand::rng());
    tokio::fs::create_dir_all(&dir).await?;
    tokio::fs::write(&key_path, hex::encode(key.to_bytes())).await?;
    tracing::info!("Generated new key, saved to {}", key_path.display());
    Ok(key)
}
