//! Mesh membership via iroh QUIC connections.
//!
//! Single ALPN, single connection per peer. Bi-streams multiplexed by
//! first byte: 0x01 = gossip, 0x02 = tunnel (RPC), 0x03 = tunnel map, 0x04 = tunnel (HTTP).

use anyhow::Result;
use base64::Engine;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use iroh::endpoint::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, watch};

pub const ALPN: &[u8] = b"mesh-llm/0";
const STREAM_GOSSIP: u8 = 0x01;
const STREAM_TUNNEL: u8 = 0x02;
const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
const STREAM_ROUTE_REQUEST: u8 = 0x05;
const STREAM_PEER_DOWN: u8 = 0x06;
const STREAM_PEER_LEAVING: u8 = 0x07;

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
#[derive(Serialize, Deserialize)]
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
    /// Model currently loaded in VRAM (None = not assigned yet)
    #[serde(default)]
    serving: Option<String>,
    /// All GGUF filenames on disk in ~/.models/ (for mesh catalog)
    #[serde(default)]
    available_models: Vec<String>,
    /// Models this node wants the mesh to serve (from --model flags)
    #[serde(default)]
    requested_models: Vec<String>,
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
    /// Model currently loaded in VRAM
    pub serving: Option<String>,
    /// All GGUFs on disk
    pub available_models: Vec<String>,
    /// Models this node has requested the mesh to serve
    pub requested_models: Vec<String>,
}

/// Scan ~/.models/ for GGUF files and return their stem names.
pub fn scan_local_models() -> Vec<String> {
    let models_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".models");
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // Skip draft models (tiny) and partial downloads
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    if size > 500_000_000 { // > 500MB, skip draft models
                        names.push(stem.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    names
}

/// Detect available VRAM. On Apple Silicon, uses ~75% of system RAM
/// (the rest is reserved for OS/apps on unified memory).
pub fn detect_vram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // sysctl hw.memsize returns total physical RAM
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok();
        if let Some(out) = output {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    // ~75% usable for Metal on unified memory
                    return (bytes as f64 * 0.75) as u64;
                }
            }
        }
    }
    0
}

/// Lightweight routing table for passive nodes (clients + standby GPU).
/// Contains just enough info to route requests to the right host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub hosts: Vec<RouteEntry>,
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
    use std::net::{SocketAddr, SocketAddrV4, Ipv4Addr};

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
        req[0] = 0x00; req[1] = 0x01; // Binding Request
        // length = 0
        req[4] = 0x21; req[5] = 0x12; req[6] = 0xA4; req[7] = 0x42; // Magic Cookie
        rand::fill(&mut req[8..20]);

        let dest: SocketAddr = match tokio::net::lookup_host(server).await {
            Ok(mut addrs) => match addrs.next() {
                Some(a) => a,
                None => continue,
            },
            Err(_) => continue,
        };

        if sock.send_to(&req, dest).await.is_err() { continue; }

        let mut buf = [0u8; 256];
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            sock.recv_from(&mut buf),
        ).await {
            Ok(Ok((len, _))) if len >= 20 => {
                // Parse STUN response for XOR-MAPPED-ADDRESS (0x0020)
                // or MAPPED-ADDRESS (0x0001)
                let magic = &req[4..8];
                let _txn = &req[8..20];
                let mut i = 20;
                while i + 4 <= len {
                    let attr_type = u16::from_be_bytes([buf[i], buf[i + 1]]);
                    let attr_len = u16::from_be_bytes([buf[i + 2], buf[i + 3]]) as usize;
                    if i + 4 + attr_len > len { break; }
                    let val = &buf[i + 4..i + 4 + attr_len];

                    if attr_type == 0x0020 && attr_len >= 8 && val[1] == 0x01 {
                        // XOR-MAPPED-ADDRESS, IPv4 — extract IP only
                        let ip = Ipv4Addr::new(
                            val[4] ^ magic[0], val[5] ^ magic[1],
                            val[6] ^ magic[2], val[7] ^ magic[3],
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
    available_models: Arc<Mutex<Vec<String>>>,
    requested_models: Arc<Mutex<Vec<String>>>,
    vram_bytes: u64,
    peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    tunnel_tx: tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    tunnel_http_tx: tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
}

struct MeshState {
    peers: HashMap<EndpointId, PeerInfo>,
    connections: HashMap<EndpointId, Connection>,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    dead_peers: std::collections::HashSet<EndpointId>,
}

/// Channels returned by Node::start for inbound tunnel streams.
pub struct TunnelChannels {
    pub rpc: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    pub http: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
}

impl Node {
    pub async fn start(role: NodeRole, relay_urls: &[String], bind_port: Option<u16>, max_vram_gb: Option<f64>) -> Result<(Self, TunnelChannels)> {
        // Clients use an ephemeral key so they get a unique identity even
        // when running on the same machine as a GPU node.
        let secret_key = if matches!(role, NodeRole::Client) {
            let key = SecretKey::generate(&mut rand::rng());
            tracing::info!("Client mode: using ephemeral key");
            key
        } else {
            load_or_create_key().await?
        };
        // Configure QUIC transport for heavy RPC traffic:
        // - Allow many concurrent bi-streams (model loading opens hundreds)
        // - Long idle timeout to survive pauses during tensor transfers
        use iroh::endpoint::QuicTransportConfig;
        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .max_idle_timeout(Some(std::time::Duration::from_secs(30).try_into()?))
            .keep_alive_interval(std::time::Duration::from_secs(5))
            .build();
        let mut builder = Endpoint::builder()
            .secret_key(secret_key)
            .alpns(vec![ALPN.to_vec()])
            .transport_config(transport_config);

        if !relay_urls.is_empty() {
            use iroh::{RelayConfig, RelayMap};
            let configs: Vec<RelayConfig> = relay_urls.iter().map(|url| {
                RelayConfig { url: url.parse().expect("invalid relay URL"), quic: None }
            }).collect();
            let relay_map = RelayMap::from_iter(configs);
            tracing::info!("Using custom relay URLs: {:?}", relay_urls);
            builder = builder.relay_mode(iroh::endpoint::RelayMode::Custom(relay_map));
        }
        if let Some(port) = bind_port {
            tracing::info!("Binding QUIC to UDP port {port}");
            builder = builder.bind_addr(std::net::SocketAddr::from(([0, 0, 0, 0], port)))?;
        }
        let endpoint = builder.bind().await?;
        // Wait briefly for relay connection so the invite token includes the relay URL.
        // On sinkholed networks this times out and we proceed without relay (direct UDP only).
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            endpoint.online(),
        ).await {
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
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);

        let mut vram = detect_vram_bytes();
        if let Some(max_gb) = max_vram_gb {
            let max_bytes = (max_gb * 1e9) as u64;
            if max_bytes < vram {
                tracing::info!("Detected VRAM: {:.1} GB, capped to {:.1} GB (--max-vram)", vram as f64 / 1e9, max_gb);
                vram = max_bytes;
            } else {
                tracing::info!("Detected VRAM: {:.1} GB (--max-vram {:.1} has no effect)", vram as f64 / 1e9, max_gb);
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
            })),
            role: Arc::new(Mutex::new(role)),
            models: Arc::new(Mutex::new(Vec::new())),
            model_source: Arc::new(Mutex::new(None)),
            serving: Arc::new(Mutex::new(None)),
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            vram_bytes: vram,
            peer_change_tx,
            peer_change_rx,
            tunnel_tx,
            tunnel_http_tx,
        };

        let node2 = node.clone();
        tokio::spawn(async move { node2.accept_loop().await; });

        Ok((node, TunnelChannels { rpc: tunnel_rx, http: tunnel_http_rx }))
    }

    pub fn invite_token(&self) -> String {
        let mut addr = self.endpoint.addr();
        // Inject STUN-discovered public address if relay STUN didn't provide one.
        if let Some(pub_addr) = self.public_addr {
            use iroh::TransportAddr;
            let has_public = addr.addrs.iter().any(|a| match a {
                TransportAddr::Ip(sock) => {
                    match sock.ip() {
                        std::net::IpAddr::V4(v4) => !v4.is_private() && !v4.is_loopback(),
                        _ => false,
                    }
                }
                _ => false,
            });
            if !has_public {
                addr.addrs.insert(TransportAddr::Ip(pub_addr));
            }
        }
        let json = serde_json::to_vec(&addr).expect("serializable");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json)
    }

    pub async fn join(&self, invite_token: &str) -> Result<()> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
        let addr: EndpointAddr = serde_json::from_slice(&json)?;
        // Clear dead status — explicit join should always attempt connection
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.connect_to_peer(addr).await
    }

    /// Connect to a peer without gossip exchange — for passive nodes (clients/standby).
    /// Establishes QUIC connection and stores it, but doesn't add to peer list.
    /// The passive node can then use route requests and HTTP tunnels.
    pub async fn join_passive(&self, invite_token: &str) -> Result<()> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
        let addr: EndpointAddr = serde_json::from_slice(&json)?;
        let peer_id = addr.id;
        if peer_id == self.endpoint.id() { return Ok(()); }

        // Already have a connection? Done.
        {
            let state = self.state.lock().await;
            if state.connections.contains_key(&peer_id) { return Ok(()); }
        }

        tracing::info!("Passive connect to {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            std::time::Duration::from_secs(15),
            self.endpoint.connect(addr.clone(), ALPN),
        ).await {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => anyhow::bail!("Failed to connect: {e}"),
            Err(_) => anyhow::bail!("Timeout connecting (15s)"),
        };

        {
            let mut state = self.state.lock().await;
            state.connections.insert(peer_id, conn.clone());
        }

        // Start stream dispatcher (for responses to our requests)
        let node = self.clone();
        tokio::spawn(async move {
            node.dispatch_streams(conn, peer_id).await;
        });

        // Get initial routing table
        if let Ok(table) = self.request_routing_table(peer_id).await {
            // Store route entries as lightweight peer info so host_for_model works
            for entry in &table.hosts {
                self.add_route_entry(entry).await;
            }
        }

        Ok(())
    }

    /// Add a route entry as a lightweight peer (from routing table, not gossip).
    async fn add_route_entry(&self, entry: &RouteEntry) {
        let mut state = self.state.lock().await;
        if entry.endpoint_id == self.endpoint.id() { return; }
        // Only add/update if we don't already have full peer info from gossip
        if let Some(existing) = state.peers.get_mut(&entry.endpoint_id) {
            // Update serving info from route table
            existing.serving = Some(entry.model.clone());
            return;
        }
        // Add a minimal peer entry for routing purposes
        state.peers.insert(entry.endpoint_id, PeerInfo {
            id: entry.endpoint_id,
            addr: EndpointAddr { id: entry.endpoint_id, addrs: Default::default() },
            tunnel_port: None,
            role: NodeRole::Host { http_port: 0 },
            models: vec![entry.model.clone()],
            vram_bytes: (entry.vram_gb * 1e9) as u64,
            rtt_ms: None,
            model_source: None,
            serving: Some(entry.model.clone()),
            available_models: vec![],
            requested_models: vec![],
        });
    }

    #[allow(dead_code)]
    pub fn endpoint(&self) -> &Endpoint { &self.endpoint }
    pub fn id(&self) -> EndpointId { self.endpoint.id() }

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

    /// Re-gossip our state to all connected peers.
    /// Call after changing serving/role/models so peers learn the update.
    pub async fn regossip(&self) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state.connections.iter().map(|(id, c)| (*id, c.clone())).collect()
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

    pub async fn serving(&self) -> Option<String> {
        self.serving.lock().await.clone()
    }

    pub async fn set_available_models(&self, models: Vec<String>) {
        *self.available_models.lock().await = models;
    }

    pub async fn available_models(&self) -> Vec<String> {
        self.available_models.lock().await.clone()
    }

    pub async fn set_requested_models(&self, models: Vec<String>) {
        *self.requested_models.lock().await = models;
    }

    pub async fn requested_models(&self) -> Vec<String> {
        self.requested_models.lock().await.clone()
    }

    /// Start a background task that periodically checks peer health.
    /// Probes each peer by attempting a gossip exchange. If the probe fails
    /// (connection dead, peer unresponsive), removes the peer immediately
    /// rather than waiting for QUIC idle timeout.
    /// Start a slow heartbeat (60s) to catch silent failures.
    /// Unlike the old aggressive health check (15s full gossip), this just
    /// verifies connections are alive. Death detection mostly happens on use
    /// (tunnel open fails → broadcast_peer_down).
    pub fn start_heartbeat(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut fail_counts: std::collections::HashMap<EndpointId, u32> = std::collections::HashMap::new();

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;

                let peers_and_conns: Vec<(EndpointId, Option<Connection>)> = {
                    let state = node.state.lock().await;
                    state.peers.keys().map(|id| {
                        let conn = state.connections.get(id).cloned();
                        (*id, conn)
                    }).collect()
                };

                for (peer_id, conn) in peers_and_conns {
                    let alive = if let Some(conn) = conn {
                        // Gossip as heartbeat — syncs state but won't re-discover dead peers
                        tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            node.initiate_gossip_inner(conn, peer_id, false),
                        ).await.map(|r| r.is_ok()).unwrap_or(false)
                    } else {
                        false
                    };

                    if alive {
                        if fail_counts.contains_key(&peer_id) {
                            eprintln!("💚 Heartbeat: {} recovered (was {}/2)", peer_id.fmt_short(), fail_counts.get(&peer_id).unwrap_or(&0));
                            // Clear dead_peers if peer came back
                            node.state.lock().await.dead_peers.remove(&peer_id);
                        }
                        fail_counts.remove(&peer_id);
                    } else {
                        let count = fail_counts.entry(peer_id).or_default();
                        *count += 1;
                        // Mark as suspect immediately to prevent gossip re-adding
                        node.state.lock().await.dead_peers.insert(peer_id);
                        if *count >= 2 {
                            eprintln!("💔 Heartbeat: {} unreachable ({} failures), removing + broadcasting death", peer_id.fmt_short(), count);
                            fail_counts.remove(&peer_id);
                            node.handle_peer_death(peer_id).await;
                        } else {
                            eprintln!("💛 Heartbeat: {} unreachable ({}/2), will retry", peer_id.fmt_short(), count);
                        }
                    }
                }

            }
        });
    }

    /// Handle a peer death: remove from state, broadcast to all other peers.
    pub async fn handle_peer_death(&self, dead_id: EndpointId) {
        eprintln!("⚠️  Peer {} died — removing and broadcasting", dead_id.fmt_short());
        {
            let mut state = self.state.lock().await;
            state.connections.remove(&dead_id);
            state.dead_peers.insert(dead_id);
        }
        self.remove_peer(dead_id).await;
        self.broadcast_peer_down(dead_id).await;
    }

    /// Broadcast that a peer is down to all connected peers.
    async fn broadcast_peer_down(&self, dead_id: EndpointId) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state.connections.iter()
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
                }.await;
                if let Err(e) = res {
                    tracing::debug!("Failed to broadcast peer_down to {}: {e}", peer_id.fmt_short());
                }
            });
        }
    }

    /// Announce clean shutdown to all peers.
    pub async fn broadcast_leaving(&self) {
        let my_id_bytes = self.endpoint.id().as_bytes().to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state.connections.iter()
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
                }.await;
                if let Err(e) = res {
                    tracing::debug!("Failed to send leaving to {}: {e}", peer_id.fmt_short());
                }
            });
        }
        // Give broadcasts a moment to flush
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    /// Get model source from any peer in the mesh (for auto-download on join).
    pub async fn peer_model_source(&self) -> Option<String> {
        let state = self.state.lock().await;
        for p in state.peers.values() {
            if let Some(ref src) = p.model_source {
                return Some(src.clone());
            }
        }
        None
    }

    /// Get the mesh catalog: all models that any node has on disk or has requested.
    /// Returns deduplicated list of model names (file stems, no .gguf).
    pub async fn mesh_catalog(&self) -> Vec<String> {
        let state = self.state.lock().await;
        let my_available = self.available_models.lock().await;
        let my_requested = self.requested_models.lock().await;
        let mut all = std::collections::HashSet::new();
        for m in my_available.iter() {
            all.insert(m.clone());
        }
        for m in my_requested.iter() {
            all.insert(m.clone());
        }
        for p in state.peers.values() {
            for m in &p.available_models {
                all.insert(m.clone());
            }
            for m in &p.requested_models {
                all.insert(m.clone());
            }
        }
        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    /// Get all models currently being served in the mesh (loaded in VRAM somewhere).
    pub async fn models_being_served(&self) -> Vec<String> {
        let state = self.state.lock().await;
        let my_serving = self.serving.lock().await;
        let mut served = std::collections::HashSet::new();
        if let Some(ref s) = *my_serving {
            served.insert(s.clone());
        }
        for p in state.peers.values() {
            if let Some(ref s) = p.serving {
                served.insert(s.clone());
            }
        }
        let mut result: Vec<String> = served.into_iter().collect();
        result.sort();
        result
    }

    /// Get peers serving a specific model (including self if applicable).
    /// Returns (my_serving, peers_serving) — my_serving is true if this node serves it.
    pub async fn peers_serving_model(&self, model: &str) -> (bool, Vec<PeerInfo>) {
        let state = self.state.lock().await;
        let my_serving = self.serving.lock().await;
        let i_serve = my_serving.as_deref() == Some(model);
        let peers: Vec<PeerInfo> = state.peers.values()
            .filter(|p| p.serving.as_deref() == Some(model))
            .cloned()
            .collect();
        (i_serve, peers)
    }

    /// Find a host for a specific model, using hash-based selection for load distribution.
    /// When multiple hosts serve the same model, picks one based on our node ID hash.
    pub async fn host_for_model(&self, model: &str) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        let mut hosts: Vec<&PeerInfo> = state.peers.values()
            .filter(|p| matches!(p.role, NodeRole::Host { .. }) && p.serving.as_deref() == Some(model))
            .collect();
        if hosts.is_empty() { return None; }
        // Sort for deterministic ordering, then hash-select
        hosts.sort_by_key(|p| p.id);
        let my_id = self.endpoint.id();
        let my_id_bytes = my_id.as_bytes();
        let hash = my_id_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % hosts.len();
        Some(hosts[idx].clone())
    }

    /// Find ANY host in the mesh (fallback when no model match).
    pub async fn any_host(&self) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        state.peers.values()
            .find(|p| matches!(p.role, NodeRole::Host { .. }))
            .cloned()
    }

    /// Build the current routing table from this node's view of the mesh.
    pub async fn routing_table(&self) -> RoutingTable {
        let state = self.state.lock().await;
        let my_serving = self.serving.lock().await;
        let my_role = self.role.lock().await.clone();
        let mut hosts = Vec::new();

        // Include self if we're a host
        if matches!(my_role, NodeRole::Host { .. }) {
            if let Some(ref model) = *my_serving {
                hosts.push(RouteEntry {
                    model: model.clone(),
                    node_id: format!("{}", self.endpoint.id().fmt_short()),
                    endpoint_id: self.endpoint.id(),
                    vram_gb: self.vram_bytes as f64 / 1e9,
                });
            }
        }

        // Include peers that are hosts
        for p in state.peers.values() {
            if matches!(p.role, NodeRole::Host { .. }) {
                if let Some(ref model) = p.serving {
                    hosts.push(RouteEntry {
                        model: model.clone(),
                        node_id: format!("{}", p.id.fmt_short()),
                        endpoint_id: p.id,
                        vram_gb: p.vram_bytes as f64 / 1e9,
                    });
                }
            }
        }

        RoutingTable { hosts }
    }

    /// Request routing table from a connected peer (for passive nodes).
    pub async fn request_routing_table(&self, peer_id: EndpointId) -> Result<RoutingTable> {
        let conn = {
            let state = self.state.lock().await;
            state.connections.get(&peer_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("No connection to peer"))?
        };
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_ROUTE_REQUEST]).await?;
        send.finish()?;

        let data = recv.read_to_end(64 * 1024).await?;
        let table: RoutingTable = serde_json::from_slice(&data)?;
        Ok(table)
    }

    /// Refresh routing table from any connected peer (for passive nodes).
    /// Updates local peer info with current host/model mappings.
    pub async fn refresh_routing_table(&self) {
        let connected: Vec<EndpointId> = {
            self.state.lock().await.connections.keys().cloned().collect()
        };
        for peer_id in connected {
            match self.request_routing_table(peer_id).await {
                Ok(table) => {
                    for entry in &table.hosts {
                        self.add_route_entry(entry).await;
                    }
                    // Remove peers no longer in routing table
                    let active_ids: std::collections::HashSet<EndpointId> =
                        table.hosts.iter().map(|e| e.endpoint_id).collect();
                    let mut state = self.state.lock().await;
                    state.peers.retain(|id, _| active_ids.contains(id) || *id == peer_id);
                    return; // Got table from one peer, that's enough
                }
                Err(e) => {
                    tracing::debug!("Route table refresh from {} failed: {e}", peer_id.fmt_short());
                }
            }
        }
    }

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state.lock().await.peers.values().cloned().collect()
    }

    /// Wait for a peer with Host role to appear. Returns its PeerInfo.
    pub async fn wait_for_host(&self) -> Result<PeerInfo> {
        loop {
            let peers = self.peers().await;
            for p in &peers {
                if matches!(p.role, NodeRole::Host { .. }) {
                    return Ok(p.clone());
                }
            }
            // Poll every 500ms
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    /// Open an HTTP tunnel bi-stream to a peer (tagged STREAM_TUNNEL_HTTP).
    /// If no connection exists, tries to connect on-demand (for passive nodes
    /// that learned about hosts from routing table but aren't directly connected).
    pub async fn open_http_tunnel(&self, peer_id: EndpointId) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
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
                        ).await
                            .map_err(|_| anyhow::anyhow!("Timeout connecting to {}", peer_id.fmt_short()))?
                            .map_err(|e| anyhow::anyhow!("Failed to connect to {}: {e}", peer_id.fmt_short()))?;
                        self.state.lock().await.connections.insert(peer_id, c.clone());
                        c
                    } else {
                        anyhow::bail!("No connection or address for {}", peer_id.fmt_short());
                    }
                }
            }
        };
        let result = async {
            let (mut send, recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_TUNNEL_HTTP]).await?;
            Ok::<_, anyhow::Error>((send, recv))
        }.await;

        if result.is_err() {
            // Connection failed — peer is likely dead, broadcast it
            tracing::info!("Tunnel to {} failed, broadcasting death", peer_id.fmt_short());
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
    pub async fn broadcast_tunnel_map(&self, my_tunnel_map: HashMap<EndpointId, u16>) -> Result<()> {
        // Serialize: { endpoint_id_hex_string → port }
        let serializable: HashMap<String, u16> = my_tunnel_map
            .iter()
            .map(|(id, port)| (hex::encode(id.as_bytes()), *port))
            .collect();
        let msg = serde_json::to_vec(&serializable)?;

        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state.connections.iter().map(|(id, c)| (*id, c.clone())).collect()
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
    pub async fn open_tunnel_stream(&self, peer_id: EndpointId) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = {
            self.state.lock().await.connections.get(&peer_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("No connection to {}", peer_id.fmt_short()))?
        };
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_TUNNEL]).await?;
        Ok((send, recv))
    }

    // --- Connection handling ---

    async fn accept_loop(&self) {
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
        {
            let mut state = self.state.lock().await;
            if state.dead_peers.remove(&remote) {
                eprintln!("🔄 Previously dead peer {} reconnected", remote.fmt_short());
            }
            state.connections.insert(remote, conn.clone());
        }

        // Don't auto-gossip: passive nodes (clients/standby) just connect and
        // send streams. Active peers identify themselves by sending STREAM_GOSSIP.
        // Gossip exchange in handle_gossip_stream calls add_peer().

        self.dispatch_streams(conn, remote).await;
        Ok(())
    }

    /// Dispatch bi-streams on a connection by type byte
    fn dispatch_streams(&self, conn: Connection, remote: EndpointId) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
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
                        ).await {
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
                                tracing::info!("Reconnect to {} failed — removing peer", remote.fmt_short());
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
                            tracing::warn!("Tunnel map stream error from {}: {e}", remote.fmt_short());
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
                                    let should_remove = {
                                        let state = node.state.lock().await;
                                        if let Some(conn) = state.connections.get(&dead_id) {
                                            tokio::time::timeout(
                                                std::time::Duration::from_secs(3),
                                                conn.open_bi(),
                                            ).await.is_err()
                                        } else {
                                            true // no connection = already gone
                                        }
                                    };
                                    if should_remove {
                                        eprintln!("⚠️  Peer {} reported dead by {}, confirmed, removing",
                                            dead_id.fmt_short(), remote.fmt_short());
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
                                eprintln!("👋 Peer {} announced clean shutdown", leaving_id.fmt_short());
                                let mut state = node.state.lock().await;
                                state.connections.remove(&leaving_id);
                                drop(state);
                                node.remove_peer(leaving_id).await;
                            }
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
        if peer_id == self.endpoint.id() { return Ok(()); }

        {
            let state = self.state.lock().await;
            if state.peers.contains_key(&peer_id) { return Ok(()); }
            if state.dead_peers.contains(&peer_id) {
                tracing::debug!("Skipping connection to dead peer {}", peer_id.fmt_short());
                return Ok(());
            }
        }

        tracing::info!("Connecting to peer {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            std::time::Duration::from_secs(15),
            self.endpoint.connect(addr.clone(), ALPN),
        ).await {
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
            node_for_dispatch.dispatch_streams(conn_for_dispatch, peer_id).await;
        });

        // Gossip exchange to learn peer's role/VRAM and announce ourselves
        self.initiate_gossip(conn, peer_id).await?;
        Ok(())
    }

    /// Open a gossip stream on an existing connection to exchange peer info.
    async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        self.initiate_gossip_inner(conn, remote, true).await
    }

    async fn initiate_gossip_inner(&self, conn: Connection, remote: EndpointId, discover_peers: bool) -> Result<()> {
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        // Send our peer announcements (length-prefixed JSON)
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

        // Register peer — find their own announcement for role + models + vram
        let peer_ann = their_announcements.iter().find(|a| a.addr.id == remote);
        if let Some(ann) = peer_ann {
            self.add_peer(remote, ann.addr.clone(), ann).await;
            // Store RTT
            let mut state = self.state.lock().await;
            if let Some(peer) = state.peers.get_mut(&remote) {
                peer.rtt_ms = Some(rtt_ms);
                tracing::info!("Peer {} RTT: {}ms", remote.fmt_short(), rtt_ms);
            }
        }

        // Discover new peers (only on initial join, not heartbeat)
        if discover_peers {
            for ann in their_announcements {
                if ann.addr.id != self.endpoint.id() {
                    if let Err(e) = Box::pin(self.connect_to_peer(ann.addr)).await {
                        tracing::warn!("Failed to discover peer: {e}");
                    }
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

        // Register peer with role + models + vram
        for ann in &their_announcements {
            if ann.addr.id == remote {
                self.add_peer(remote, ann.addr.clone(), ann).await;
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
                        if rtt_ms > 0 {
                            let mut state = self.state.lock().await;
                            if let Some(peer) = state.peers.get_mut(&remote) {
                                peer.rtt_ms = Some(rtt_ms);
                                tracing::info!("Peer {} RTT: {}ms (QUIC)", remote.fmt_short(), rtt_ms);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Discover new peers mentioned in gossip — but only try to connect
        // to peers we don't already know about. Skip peers we've recently removed
        // (they'll be rediscovered via the rejoin loop if they come back).
        for ann in their_announcements {
            let peer_id = ann.addr.id;
            if peer_id == self.endpoint.id() { continue; }
            // Only discover if we don't already have this peer
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
        if state.peers.remove(&id).is_some() {
            tracing::info!("Peer removed: {} (total: {})", id.fmt_short(), state.peers.len());
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
        }
    }

    async fn add_peer(&self, id: EndpointId, addr: EndpointAddr, ann: &PeerAnnouncement) {
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() { return; }
        if let Some(existing) = state.peers.get_mut(&id) {
            let role_changed = existing.role != ann.role;
            let serving_changed = existing.serving != ann.serving;
            if role_changed {
                tracing::info!("Peer {} role updated: {:?} → {:?}", id.fmt_short(), existing.role, ann.role);
                existing.role = ann.role.clone();
            }
            existing.models = ann.models.clone();
            existing.vram_bytes = ann.vram_bytes;
            if ann.model_source.is_some() {
                existing.model_source = ann.model_source.clone();
            }
            existing.serving = ann.serving.clone();
            existing.available_models = ann.available_models.clone();
            existing.requested_models = ann.requested_models.clone();
            if role_changed || serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
            }
            return;
        }
        tracing::info!("Peer added: {} role={:?} vram={:.1}GB serving={:?} available={:?} (total: {})",
            id.fmt_short(), ann.role, ann.vram_bytes as f64 / 1e9, ann.serving, ann.available_models, state.peers.len() + 1);
        state.peers.insert(id, PeerInfo {
            id, addr, tunnel_port: None,
            role: ann.role.clone(),
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            rtt_ms: None,
            model_source: ann.model_source.clone(),
            serving: ann.serving.clone(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
        });
        let count = state.peers.len();
        drop(state);
        let _ = self.peer_change_tx.send(count);
    }

    async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        let state = self.state.lock().await;
        let my_role = self.role.lock().await.clone();
        let my_models = self.models.lock().await.clone();
        let my_source = self.model_source.lock().await.clone();
        let my_serving = self.serving.lock().await.clone();
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let mut announcements: Vec<PeerAnnouncement> = state.peers.values()
            .map(|p| PeerAnnouncement {
                addr: p.addr.clone(),
                role: p.role.clone(),
                models: p.models.clone(),
                vram_bytes: p.vram_bytes,
                model_source: p.model_source.clone(),
                serving: p.serving.clone(),
                available_models: p.available_models.clone(),
                requested_models: p.requested_models.clone(),
            })
            .collect();
        announcements.push(PeerAnnouncement {
            addr: self.endpoint.addr(),
            role: my_role,
            models: my_models,
            vram_bytes: self.vram_bytes,
            model_source: my_source,
            serving: my_serving,
            available_models: my_available,
            requested_models: my_requested,
        });
        announcements
    }
}

/// Load secret key from ~/.mesh-llm/key, or create a new one and save it.
/// Migrates from ~/.mesh-inference/key if it exists.
async fn load_or_create_key() -> Result<SecretKey> {
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
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
