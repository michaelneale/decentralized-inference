//! TCP ↔ QUIC tunnel management.
//!
//! For each peer in the mesh, we:
//! 1. Listen on a local TCP port (the "tunnel port")
//! 2. When llama.cpp connects to that port, open a QUIC bi-stream (on the
//!    persistent connection) and relay bidirectionally
//!
//! On the receiving side:
//! 1. Accept inbound bi-streams tagged as STREAM_TYPE_TUNNEL
//! 2. Connect to the local rpc-server via TCP
//! 3. Bidirectionally relay

use crate::mesh::Node;
use crate::rewrite::{self, PortRewriteMap};
use anyhow::Result;
use iroh::EndpointId;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// Global byte counter for tunnel traffic
static BYTES_TRANSFERRED: AtomicU64 = AtomicU64::new(0);

/// Get total bytes transferred through all tunnels
pub fn bytes_transferred() -> u64 {
    BYTES_TRANSFERRED.load(Ordering::Relaxed)
}

/// Manages all tunnels for a node
#[derive(Clone)]
pub struct Manager {
    node: Node,
    rpc_port: u16,
    /// EndpointId → local tunnel port
    tunnel_ports: Arc<Mutex<HashMap<EndpointId, u16>>>,
    /// Port rewrite map for B2B: orchestrator tunnel port → local tunnel port
    port_rewrite_map: PortRewriteMap,
}

impl Manager {
    /// Start the tunnel manager.
    pub async fn start(
        node: Node,
        rpc_port: u16,
        mut tunnel_stream_rx: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    ) -> Result<Self> {
        let port_rewrite_map = rewrite::new_rewrite_map();
        let mgr = Manager {
            node: node.clone(),
            rpc_port,
            tunnel_ports: Arc::new(Mutex::new(HashMap::new())),
            port_rewrite_map,
        };

        // Watch for peer changes and create outbound tunnels
        let mgr2 = mgr.clone();
        tokio::spawn(async move {
            mgr2.watch_peers().await;
        });

        // Handle inbound tunnel streams (with REGISTER_PEER rewriting)
        let rpc_port = mgr.rpc_port;
        let rewrite_map = mgr.port_rewrite_map.clone();
        tokio::spawn(async move {
            while let Some((send, recv)) = tunnel_stream_rx.recv().await {
                let rewrite_map = rewrite_map.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_stream(send, recv, rpc_port, rewrite_map).await {
                        tracing::warn!("Inbound tunnel stream error: {e}");
                    }
                });
            }
        });

        Ok(mgr)
    }

    /// Wait until we have at least `n` peers with active tunnels
    pub async fn wait_for_peers(&self, n: usize) -> Result<()> {
        let mut rx = self.node.peer_change_rx.clone();
        loop {
            let count = *rx.borrow();
            if count >= n {
                return Ok(());
            }
            rx.changed().await?;
        }
    }

    /// Get the current list of tunnel ports
    pub async fn peer_ports(&self) -> Vec<u16> {
        let ports = self.tunnel_ports.lock().await;
        ports.values().copied().collect()
    }

    /// Get the full mapping of EndpointId → local tunnel port
    pub async fn peer_ports_map(&self) -> HashMap<EndpointId, u16> {
        self.tunnel_ports.lock().await.clone()
    }

    /// Update the B2B port rewrite map from all received remote tunnel maps.
    ///
    /// For each remote peer's tunnel map, maps their tunnel ports to our local
    /// tunnel ports for the same target EndpointIds. This enables REGISTER_PEER
    /// rewriting: when the orchestrator tells us "peer X is at 127.0.0.1:PORT",
    /// we replace PORT (an orchestrator tunnel port) with our own tunnel port
    /// to the same EndpointId.
    pub async fn update_rewrite_map(
        &self,
        remote_maps: &HashMap<EndpointId, HashMap<EndpointId, u16>>,
    ) {
        let my_tunnels = self.tunnel_ports.lock().await;
        let mut rewrite = self.port_rewrite_map.write().await;
        rewrite.clear();

        for (remote_peer, their_map) in remote_maps {
            for (target_id, &their_port) in their_map {
                if let Some(&my_port) = my_tunnels.get(target_id) {
                    rewrite.insert(their_port, my_port);
                    tracing::info!(
                        "B2B rewrite: peer {}'s port {} → my port {} (target {})",
                        remote_peer.fmt_short(),
                        their_port,
                        my_port,
                        target_id.fmt_short()
                    );
                }
            }
        }

        tracing::info!("B2B port rewrite map: {} entries", rewrite.len());
    }

    /// Allocate a free port by binding to :0
    async fn alloc_listener(&self) -> Result<(u16, TcpListener)> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        Ok((port, listener))
    }

    /// Watch for peer changes and create a tunnel for each new peer
    async fn watch_peers(&self) {
        let mut rx = self.node.peer_change_rx.clone();
        loop {
            if rx.changed().await.is_err() {
                break;
            }

            let peers = self.node.peers().await;
            let mut ports = self.tunnel_ports.lock().await;

            for peer in &peers {
                if ports.contains_key(&peer.id) {
                    continue;
                }

                let (port, listener) = match self.alloc_listener().await {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!("Failed to allocate tunnel port: {e}");
                        continue;
                    }
                };
                ports.insert(peer.id, port);

                self.node.set_tunnel_port(peer.id, port).await;

                tracing::info!(
                    "Tunnel 127.0.0.1:{port} → peer {}",
                    peer.id.fmt_short()
                );

                let node = self.node.clone();
                let peer_id = peer.id;
                tokio::spawn(async move {
                    if let Err(e) = run_outbound_tunnel(node, peer_id, listener).await {
                        tracing::error!("Outbound tunnel to {} on :{port} failed: {e}", peer_id.fmt_short());
                    }
                });
            }
        }
    }
}

/// Run a local TCP listener that tunnels to a remote peer via QUIC bi-streams.
async fn run_outbound_tunnel(
    node: Node,
    peer_id: EndpointId,
    listener: TcpListener,
) -> Result<()> {
    loop {
        let (tcp_stream, _addr) = listener.accept().await?;
        tcp_stream.set_nodelay(true)?;

        let node = node.clone();
        tokio::spawn(async move {
            if let Err(e) = relay_outbound(node, peer_id, tcp_stream).await {
                tracing::warn!("Outbound relay to {} ended: {e}", peer_id.fmt_short());
            }
        });
    }
}

/// Relay a single outbound TCP connection through a QUIC bi-stream.
async fn relay_outbound(node: Node, peer_id: EndpointId, tcp_stream: TcpStream) -> Result<()> {
    tracing::info!("Opening tunnel stream to {}", peer_id.fmt_short());
    let (quic_send, quic_recv) = node.open_tunnel_stream(peer_id).await?;
    tracing::info!("Tunnel stream opened to {}", peer_id.fmt_short());

    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

/// Handle an inbound tunnel bi-stream: connect to local rpc-server and relay.
/// The QUIC→TCP direction uses relay_with_rewrite to intercept REGISTER_PEER.
/// The TCP→QUIC direction (responses) is plain byte relay.
async fn handle_inbound_stream(
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
    rpc_port: u16,
    port_rewrite_map: PortRewriteMap,
) -> Result<()> {
    tracing::info!("Inbound tunnel stream → rpc-server :{rpc_port}");
    let tcp_stream = TcpStream::connect(format!("127.0.0.1:{rpc_port}")).await?;
    tcp_stream.set_nodelay(true)?;
    tracing::info!("Connected to rpc-server, starting relay");

    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);

    // QUIC→TCP: use rewrite relay (intercepts REGISTER_PEER)
    let mut t1 = tokio::spawn(async move {
        rewrite::relay_with_rewrite(quic_recv, tcp_write, port_rewrite_map).await
    });
    // TCP→QUIC: plain byte relay (responses from rpc-server)
    let mut t2 = tokio::spawn(async move {
        relay_tcp_to_quic(tcp_read, quic_send).await
    });
    tokio::select! {
        _ = &mut t1 => { t2.abort(); }
        _ = &mut t2 => { t1.abort(); }
    }
    Ok(())
}

/// Bidirectional relay. When either direction finishes, abort the other.
async fn relay_bidirectional(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let mut t1 = tokio::spawn(async move {
        relay_tcp_to_quic(tcp_read, quic_send).await
    });
    let mut t2 = tokio::spawn(async move {
        relay_quic_to_tcp(quic_recv, tcp_write).await
    });
    // When either direction finishes, abort the other so we don't leak
    // tasks waiting on a half-open connection (rpc-server keeps TCP open).
    tokio::select! {
        _ = &mut t1 => { t2.abort(); }
        _ = &mut t2 => { t1.abort(); }
    }
    Ok(())
}

async fn relay_tcp_to_quic(
    mut tcp_read: tokio::io::ReadHalf<TcpStream>,
    mut quic_send: iroh::endpoint::SendStream,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = tcp_read.read(&mut buf).await?;
        if n == 0 {
            tracing::info!("TCP→QUIC: TCP EOF after {total} bytes");
            break;
        }
        quic_send.write_all(&buf[..n]).await?;
        total += n as u64;
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
        tracing::debug!("TCP→QUIC: wrote {n} bytes (total: {total})");
    }
    quic_send.finish()?;
    Ok(())
}

async fn relay_quic_to_tcp(
    mut quic_recv: iroh::endpoint::RecvStream,
    mut tcp_write: tokio::io::WriteHalf<TcpStream>,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    tracing::debug!("QUIC→TCP: starting relay, about to first read");
    loop {
        match quic_recv.read(&mut buf).await {
            Ok(Some(n)) => {
                tcp_write.write_all(&buf[..n]).await?;
                total += n as u64;
                BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
                tracing::debug!("QUIC→TCP: wrote {n} bytes (total: {total})");
            }
            Ok(None) => {
                tracing::info!("QUIC→TCP: stream end after {total} bytes");
                break;
            }
            Err(e) => {
                tracing::warn!("QUIC→TCP: error after {total} bytes: {e}");
                return Err(e.into());
            }
        }
    }
    Ok(())
}
