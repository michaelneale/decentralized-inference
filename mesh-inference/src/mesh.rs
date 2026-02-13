//! Mesh membership via iroh QUIC connections.
//!
//! Single ALPN, single connection per peer. Bi-streams multiplexed by
//! first byte: 0x01 = gossip, 0x02 = tunnel.

use anyhow::Result;
use base64::Engine;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use iroh::endpoint::Connection;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, watch};

pub const ALPN: &[u8] = b"mesh-inference/0";
const STREAM_GOSSIP: u8 = 0x01;
const STREAM_TUNNEL: u8 = 0x02;
const STREAM_TUNNEL_MAP: u8 = 0x03;

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub tunnel_port: Option<u16>,
}

#[derive(Clone)]
pub struct Node {
    endpoint: Endpoint,
    state: Arc<Mutex<MeshState>>,
    peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    tunnel_tx: tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
}

struct MeshState {
    peers: HashMap<EndpointId, PeerInfo>,
    connections: HashMap<EndpointId, Connection>,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
}

impl Node {
    pub async fn start() -> Result<(Self, tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>)> {
        let secret_key = load_or_create_key().await?;
        // Configure QUIC transport for heavy RPC traffic:
        // - Allow many concurrent bi-streams (model loading opens hundreds)
        // - Long idle timeout to survive pauses during tensor transfers
        use iroh::endpoint::QuicTransportConfig;
        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .max_idle_timeout(Some(std::time::Duration::from_secs(300).try_into()?))
            .build();
        let endpoint = Endpoint::builder()
            .secret_key(secret_key)
            .alpns(vec![ALPN.to_vec()])
            .transport_config(transport_config)
            .bind()
            .await?;
        endpoint.online().await;

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);

        let node = Node {
            endpoint,
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                remote_tunnel_maps: HashMap::new(),
            })),
            peer_change_tx,
            peer_change_rx,
            tunnel_tx,
        };

        let node2 = node.clone();
        tokio::spawn(async move { node2.accept_loop().await; });

        Ok((node, tunnel_rx))
    }

    pub fn invite_token(&self) -> String {
        let addr = self.endpoint.addr();
        let json = serde_json::to_vec(&addr).expect("serializable");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json)
    }

    pub async fn join(&self, invite_token: &str) -> Result<()> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
        let addr: EndpointAddr = serde_json::from_slice(&json)?;
        self.connect_to_peer(addr).await
    }

    #[allow(dead_code)]
    pub fn endpoint(&self) -> &Endpoint { &self.endpoint }
    pub fn id(&self) -> EndpointId { self.endpoint.id() }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state.lock().await.peers.values().cloned().collect()
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

    /// Get a peer's tunnel port map (what tunnel ports they have to other peers).
    /// Returns None if we haven't received their map yet.
    pub async fn get_remote_tunnel_map(&self, peer_id: &EndpointId) -> Option<HashMap<EndpointId, u16>> {
        self.state.lock().await.remote_tunnel_maps.get(peer_id).cloned()
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

        // Store connection and start stream dispatcher
        {
            let mut state = self.state.lock().await;
            state.connections.insert(remote, conn.clone());
        }

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
        }

        tracing::info!("Connecting to peer {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.endpoint.connect(addr.clone(), ALPN),
        ).await {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                tracing::warn!("Failed to connect to {}: {e}", peer_id.fmt_short());
                return Ok(());
            }
            Err(_) => {
                tracing::warn!("Timeout connecting to {} (10s)", peer_id.fmt_short());
                return Ok(());
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

        // Open gossip stream
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        // Send our peer list (length-prefixed)
        let our_addrs = self.collect_peer_addrs().await;
        let msg = serde_json::to_vec(&our_addrs)?;
        send.write_all(&(msg.len() as u32).to_le_bytes()).await?;
        send.write_all(&msg).await?;
        send.finish()?;

        // Read their peer list (length-prefixed)
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let their_addrs: Vec<EndpointAddr> = serde_json::from_slice(&buf)?;

        // Wait for stream to fully close, then small delay for accept_bi to re-arm
        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Register peer
        self.add_peer(peer_id, addr).await;

        // Discover new peers (don't block on failures)
        for peer_addr in their_addrs {
            if let Err(e) = Box::pin(self.connect_to_peer(peer_addr)).await {
                tracing::warn!("Failed to discover peer: {e}");
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

        // Read their peer list
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let their_addrs: Vec<EndpointAddr> = serde_json::from_slice(&buf)?;

        // Send our peer list (length-prefixed)
        let our_addrs = self.collect_peer_addrs().await;
        let msg = serde_json::to_vec(&our_addrs)?;
        send.write_all(&(msg.len() as u32).to_le_bytes()).await?;
        send.write_all(&msg).await?;
        send.finish()?;

        // Wait for the remote to finish their send
        let _ = recv.read_to_end(0).await;

        // Register peer
        for addr in &their_addrs {
            if addr.id == remote {
                self.add_peer(remote, addr.clone()).await;
            }
        }

        // Discover new peers (don't block on failures)
        for addr in their_addrs {
            if addr.id != self.endpoint.id() {
                if let Err(e) = Box::pin(self.connect_to_peer(addr)).await {
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

    async fn add_peer(&self, id: EndpointId, addr: EndpointAddr) {
        let mut state = self.state.lock().await;
        if state.peers.contains_key(&id) || id == self.endpoint.id() { return; }
        state.peers.insert(id, PeerInfo { id, addr, tunnel_port: None });
        let count = state.peers.len();
        drop(state);
        tracing::info!("Peer added: {} (total: {count})", id.fmt_short());
        let _ = self.peer_change_tx.send(count);
    }

    async fn collect_peer_addrs(&self) -> Vec<EndpointAddr> {
        let state = self.state.lock().await;
        let mut addrs: Vec<EndpointAddr> = state.peers.values().map(|p| p.addr.clone()).collect();
        addrs.push(self.endpoint.addr());
        addrs
    }
}

/// Load secret key from ~/.mesh-inference/key, or create a new one and save it.
async fn load_or_create_key() -> Result<SecretKey> {
    let dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
        .join(".mesh-inference");
    let key_path = dir.join("key");

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
