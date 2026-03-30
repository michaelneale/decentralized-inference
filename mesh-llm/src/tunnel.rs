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
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// Global byte counter for tunnel traffic
static BYTES_TRANSFERRED: AtomicU64 = AtomicU64::new(0);

fn quic_response_first_byte_timeout() -> Duration {
    std::env::var("MESH_LLM_TUNNEL_FIRST_BYTE_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(60))
}

/// Get total bytes transferred through all tunnels
pub fn bytes_transferred() -> u64 {
    BYTES_TRANSFERRED.load(Ordering::Relaxed)
}

/// Manages all tunnels for a node
#[derive(Clone)]
pub struct Manager {
    node: Node,
    rpc_port: Arc<AtomicU16>,
    http_port: Arc<AtomicU16>,
    /// EndpointId → local tunnel port
    tunnel_ports: Arc<Mutex<HashMap<EndpointId, u16>>>,
    /// Port rewrite map for B2B: orchestrator tunnel port → local tunnel port
    port_rewrite_map: PortRewriteMap,
}

impl Manager {
    /// Start the tunnel manager.
    /// `rpc_port` is the local rpc-server port (for inbound RPC tunnel streams).
    /// HTTP port for inbound tunnels is set dynamically via `set_http_port()`.
    pub async fn start(
        node: Node,
        rpc_port: u16,
        mut tunnel_stream_rx: tokio::sync::mpsc::Receiver<(
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
        mut tunnel_http_rx: tokio::sync::mpsc::Receiver<(
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
    ) -> Result<Self> {
        let port_rewrite_map = rewrite::new_rewrite_map();
        let mgr = Manager {
            node: node.clone(),
            rpc_port: Arc::new(AtomicU16::new(rpc_port)),
            http_port: Arc::new(AtomicU16::new(0)),
            tunnel_ports: Arc::new(Mutex::new(HashMap::new())),
            port_rewrite_map,
        };

        // Watch for peer changes and create outbound tunnels
        let mgr2 = mgr.clone();
        tokio::spawn(async move {
            mgr2.watch_peers().await;
        });

        // Handle inbound RPC tunnel streams (with REGISTER_PEER rewriting)
        let rpc_port_ref = mgr.rpc_port.clone();
        let rewrite_map = mgr.port_rewrite_map.clone();
        tokio::spawn(async move {
            while let Some((send, recv)) = tunnel_stream_rx.recv().await {
                let port = rpc_port_ref.load(Ordering::Relaxed);
                if port == 0 {
                    tracing::warn!("Inbound RPC tunnel but no rpc-server running, dropping");
                    continue;
                }
                let rewrite_map = rewrite_map.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_stream(send, recv, port, rewrite_map).await {
                        tracing::warn!("Inbound RPC tunnel stream error: {e}");
                    }
                });
            }
        });

        // Handle inbound HTTP tunnel streams (plain byte relay to llama-server)
        let http_port_ref = mgr.http_port.clone();
        let http_node = mgr.node.clone();
        tokio::spawn(async move {
            while let Some((send, recv)) = tunnel_http_rx.recv().await {
                let port = http_port_ref.load(Ordering::Relaxed);
                if port == 0 {
                    tracing::warn!("Inbound HTTP tunnel but no llama-server running, dropping");
                    tokio::spawn(async move {
                        if let Err(e) = send_http_unavailable(send, "LLM host not ready").await {
                            tracing::warn!("Failed to send HTTP tunnel unavailable response: {e}");
                        }
                        drop(recv);
                    });
                    continue;
                }
                let node = http_node.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_http_stream(node, send, recv, port).await {
                        tracing::warn!("Inbound HTTP tunnel stream error: {e}");
                    }
                });
            }
        });

        Ok(mgr)
    }

    /// Update the local llama-server HTTP port (for inbound HTTP tunnel streams).
    /// Set to 0 to disable (no llama-server running).
    pub fn set_http_port(&self, port: u16) {
        self.http_port.store(port, Ordering::Relaxed);
        tracing::info!("Tunnel manager: http_port updated to {port}");
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

                tracing::info!("Tunnel 127.0.0.1:{port} → peer {}", peer.id.fmt_short());

                let node = self.node.clone();
                let peer_id = peer.id;
                tokio::spawn(async move {
                    if let Err(e) = run_outbound_tunnel(node, peer_id, listener).await {
                        tracing::error!(
                            "Outbound tunnel to {} on :{port} failed: {e}",
                            peer_id.fmt_short()
                        );
                    }
                });
            }
        }
    }
}

/// Run a local TCP listener that tunnels to a remote peer via QUIC bi-streams.
async fn run_outbound_tunnel(node: Node, peer_id: EndpointId, listener: TcpListener) -> Result<()> {
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
    let mut t2 = tokio::spawn(async move { relay_tcp_to_quic(tcp_read, quic_send).await });
    tokio::select! {
        _ = &mut t1 => { t2.abort(); }
        _ = &mut t2 => { t1.abort(); }
    }
    Ok(())
}

/// Handle an inbound HTTP tunnel bi-stream: connect to local llama-server and relay.
/// Plain byte relay — no protocol awareness needed (HTTP/SSE just flows through).
async fn handle_inbound_http_stream(
    node: Node,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
    http_port: u16,
) -> Result<()> {
    tracing::info!("Inbound HTTP tunnel stream → llama-server :{http_port}");
    let tcp_stream = match TcpStream::connect(format!("127.0.0.1:{http_port}")).await {
        Ok(stream) => stream,
        Err(e) => {
            tracing::warn!("Inbound HTTP tunnel can't reach llama-server :{http_port}: {e}");
            send_http_unavailable(quic_send, "LLM host unavailable").await?;
            drop(quic_recv);
            return Ok(());
        }
    };
    tcp_stream.set_nodelay(true)?;
    let _inflight = node.begin_inflight_request();

    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

async fn send_http_unavailable(mut quic_send: iroh::endpoint::SendStream, msg: &str) -> Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    quic_send.write_all(resp.as_bytes()).await?;
    quic_send.finish()?;
    Ok(())
}

pub async fn relay_tcp_via_quic(
    tcp_stream: TcpStream,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

/// Relay only the QUIC response side back to a TCP client.
///
/// Used by the HTTP proxy after it has already buffered and forwarded exactly
/// one request upstream. Any further bytes from the client connection are
/// intentionally ignored so pipelined follow-up requests are not replayed to
/// the selected upstream without a fresh routing decision.
pub async fn relay_quic_response_to_tcp(
    mut tcp_stream: TcpStream,
    mut quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let result = relay_response_with_first_byte_timeout(
        &mut quic_recv,
        &mut tcp_stream,
        quic_response_first_byte_timeout(),
    )
    .await;
    let _ = tcp_stream.shutdown().await;
    result
}

/// Bidirectional relay. When either direction finishes, abort the other.
pub async fn relay_bidirectional(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let mut t1 = tokio::spawn(async move { relay_tcp_to_quic(tcp_read, quic_send).await });
    let mut t2 = tokio::spawn(async move { relay_quic_to_tcp(quic_recv, tcp_write).await });
    // If the request side finishes first, keep draining the response side.
    // This matters for HTTP where the client may finish sending the request
    // before the server has produced the response.
    tokio::select! {
        r1 = &mut t1 => {
            r1??;
            tracing::debug!("relay_bidirectional: request side finished, waiting for response side");
            t2.await??;
        }
        r2 = &mut t2 => {
            r2??;
            t1.abort();
            let _ = t1.await;
        }
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
    relay_response_with_first_byte_timeout(
        &mut quic_recv,
        &mut tcp_write,
        quic_response_first_byte_timeout(),
    )
    .await
}

async fn relay_response_with_first_byte_timeout<R, W>(
    mut reader: R,
    mut writer: W,
    first_byte_timeout: Duration,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    tracing::debug!("QUIC→TCP: starting relay, about to first read");

    // First-byte timeout: allow enough time for remote prefill on real prompts.
    // After first byte arrives, no timeout (streaming responses can take minutes).
    let first_read = tokio::time::timeout(first_byte_timeout, reader.read(&mut buf)).await;
    match first_read {
        Err(_) => {
            anyhow::bail!(
                "QUIC→TCP: no response within {:.3}s — host likely dead or still prefill-bound",
                first_byte_timeout.as_secs_f64()
            );
        }
        Ok(Ok(0)) => {
            tracing::info!("QUIC→TCP: stream end immediately (0 bytes)");
            return Ok(());
        }
        Ok(Ok(n)) => {
            writer.write_all(&buf[..n]).await?;
            total += n as u64;
            BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
            tracing::debug!("QUIC→TCP: first read {n} bytes");
        }
        Ok(Err(e)) => {
            tracing::warn!("QUIC→TCP: error on first read: {e}");
            return Err(e.into());
        }
    }

    // After first byte, relay without timeout
    loop {
        match reader.read(&mut buf).await {
            Ok(0) => {
                tracing::info!("QUIC→TCP: stream end after {total} bytes");
                break;
            }
            Ok(n) => {
                writer.write_all(&buf[..n]).await?;
                total += n as u64;
                BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
                tracing::debug!("QUIC→TCP: wrote {n} bytes (total: {total})");
            }
            Err(e) => {
                tracing::warn!("QUIC→TCP: error after {total} bytes: {e}");
                return Err(e.into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn relay_response_times_out_before_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(75)).await;
            let _ = upstream_write.write_all(b"late response").await;
        });

        let err = relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("no response within"));
        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert!(forwarded.is_empty());
    }

    #[tokio::test]
    async fn relay_response_allows_slow_but_healthy_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            upstream_write
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .await
                .unwrap();
        });

        relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(200),
        )
        .await
        .unwrap();

        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert_eq!(
            forwarded,
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        );
    }

    #[tokio::test]
    async fn relay_response_allows_slow_follow_up_chunks_after_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            upstream_write
                .write_all(b"HTTP/1.1 200 OK\r\n")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(75)).await;
            upstream_write
                .write_all(b"Content-Length: 5\r\n\r\nhello")
                .await
                .unwrap();
        });

        relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(20),
        )
        .await
        .unwrap();

        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert_eq!(
            forwarded,
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        );
    }
}
