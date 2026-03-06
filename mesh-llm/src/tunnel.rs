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
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// Global byte counter for tunnel traffic
static BYTES_TRANSFERRED: AtomicU64 = AtomicU64::new(0);

/// Get total bytes transferred through all tunnels
pub fn bytes_transferred() -> u64 {
    BYTES_TRANSFERRED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HttpRelayUsage {
    pub status_code: Option<u16>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

const MAX_USAGE_CAPTURE_BYTES: usize = 512 * 1024;
const MAX_HTTP_REQUEST_REWRITE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Default)]
struct UsageCapture {
    tail: Vec<u8>,
    status_code: Option<u16>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

impl UsageCapture {
    fn ingest(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > MAX_USAGE_CAPTURE_BYTES {
            let drop = self.tail.len() - MAX_USAGE_CAPTURE_BYTES;
            self.tail.drain(0..drop);
        }
        if self.status_code.is_none() {
            self.status_code = parse_http_status_code(&self.tail);
        }
        self.prompt_tokens =
            extract_last_usage_value(&self.tail, &[b"\"prompt_tokens\"", b"\"input_tokens\""])
                .or(self.prompt_tokens);
        self.completion_tokens = extract_last_usage_value(
            &self.tail,
            &[b"\"completion_tokens\"", b"\"output_tokens\""],
        )
        .or(self.completion_tokens);
    }

    fn snapshot(&self) -> HttpRelayUsage {
        HttpRelayUsage {
            status_code: self.status_code,
            prompt_tokens: self.prompt_tokens.unwrap_or(0),
            completion_tokens: self.completion_tokens.unwrap_or(0),
        }
    }
}

fn find_last_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    for i in (0..=haystack.len() - needle.len()).rev() {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

fn parse_http_status_code(bytes: &[u8]) -> Option<u16> {
    let line_end = bytes.windows(2).position(|w| w == b"\r\n")?;
    let line = std::str::from_utf8(&bytes[..line_end]).ok()?;
    let mut parts = line.split_whitespace();
    let _ = parts.next()?;
    let code = parts.next()?.parse::<u16>().ok()?;
    Some(code)
}

fn extract_json_u64_after_key(bytes: &[u8], key: &[u8]) -> Option<u64> {
    let idx = find_last_subslice(bytes, key)?;
    let mut i = idx + key.len();
    while i < bytes.len() {
        let b = bytes[i];
        if b == b':' {
            i += 1;
            break;
        }
        if !b.is_ascii_whitespace() {
            return None;
        }
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    std::str::from_utf8(&bytes[start..i])
        .ok()?
        .parse::<u64>()
        .ok()
}

fn extract_last_usage_value(bytes: &[u8], keys: &[&[u8]]) -> Option<u64> {
    for key in keys {
        if let Some(value) = extract_json_u64_after_key(bytes, key) {
            return Some(value);
        }
    }
    None
}

fn header_end_offset(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|idx| idx + 4)
}

fn parse_content_length(headers: &str) -> Option<usize> {
    for line in headers.lines() {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                return value.trim().parse::<usize>().ok();
            }
        }
    }
    None
}

fn has_chunked_encoding(headers: &str) -> bool {
    headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .any(|(name, value)| {
            name.trim().eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
        })
}

fn rewrite_chat_completions_request(request: &[u8]) -> Option<Vec<u8>> {
    let header_end = header_end_offset(request)?;
    let headers = std::str::from_utf8(&request[..header_end]).ok()?;
    let mut lines = headers.split("\r\n");
    let request_line = lines.next()?.trim();
    if !(request_line.starts_with("POST /v1/chat/completions")
        || request_line.starts_with("POST /api/chat"))
    {
        return None;
    }
    let original_body = &request[header_end..];
    let mut json = serde_json::from_slice::<Value>(original_body).ok()?;
    let root = json.as_object_mut()?;
    let stream_options = root
        .entry("stream_options".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !stream_options.is_object() {
        *stream_options = Value::Object(serde_json::Map::new());
    }
    stream_options
        .as_object_mut()
        .expect("stream_options should be object")
        .insert("include_usage".to_string(), Value::Bool(true));
    let rewritten_body = serde_json::to_vec(&json).ok()?;

    let mut rebuilt = String::new();
    rebuilt.push_str(request_line);
    rebuilt.push_str("\r\n");
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((name, _)) = trimmed.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                continue;
            }
            if name.trim().eq_ignore_ascii_case("transfer-encoding") {
                continue;
            }
        }
        rebuilt.push_str(trimmed);
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str(&format!("Content-Length: {}\r\n", rewritten_body.len()));
    rebuilt.push_str("\r\n");

    let mut out = rebuilt.into_bytes();
    out.extend_from_slice(&rewritten_body);
    Some(out)
}

async fn relay_http_request_to_quic_with_optional_rewrite(
    mut tcp_read: tokio::io::ReadHalf<TcpStream>,
    mut quic_send: iroh::endpoint::SendStream,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut pending = Vec::<u8>::new();
    let mut first_request_handled = false;

    loop {
        let n = tcp_read.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        if !first_request_handled {
            pending.extend_from_slice(&buf[..n]);
            if pending.len() > MAX_HTTP_REQUEST_REWRITE_BYTES {
                quic_send.write_all(&pending).await?;
                BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                pending.clear();
                first_request_handled = true;
                continue;
            }
            let Some(header_end) = header_end_offset(&pending) else {
                continue;
            };
            let headers = match std::str::from_utf8(&pending[..header_end]) {
                Ok(h) => h,
                Err(_) => {
                    quic_send.write_all(&pending).await?;
                    BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                    pending.clear();
                    first_request_handled = true;
                    continue;
                }
            };
            if has_chunked_encoding(headers) {
                quic_send.write_all(&pending).await?;
                BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                pending.clear();
                first_request_handled = true;
                continue;
            }
            let content_length = parse_content_length(headers).unwrap_or(0);
            let total_len = header_end + content_length;
            if pending.len() < total_len {
                continue;
            }
            let request_bytes = &pending[..total_len];
            let rewritten = rewrite_chat_completions_request(request_bytes)
                .unwrap_or_else(|| request_bytes.to_vec());
            quic_send.write_all(&rewritten).await?;
            BYTES_TRANSFERRED.fetch_add(rewritten.len() as u64, Ordering::Relaxed);
            if pending.len() > total_len {
                let tail = &pending[total_len..];
                quic_send.write_all(tail).await?;
                BYTES_TRANSFERRED.fetch_add(tail.len() as u64, Ordering::Relaxed);
            }
            pending.clear();
            first_request_handled = true;
            continue;
        }

        quic_send.write_all(&buf[..n]).await?;
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
    }

    if !pending.is_empty() {
        quic_send.write_all(&pending).await?;
        BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
    }

    quic_send.finish()?;
    Ok(())
}

async fn relay_http_request_to_tcp_with_optional_rewrite(
    mut src: tokio::io::ReadHalf<TcpStream>,
    dst: &mut tokio::io::WriteHalf<TcpStream>,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut pending = Vec::<u8>::new();
    let mut first_request_handled = false;

    loop {
        let n = src.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        if !first_request_handled {
            pending.extend_from_slice(&buf[..n]);
            if pending.len() > MAX_HTTP_REQUEST_REWRITE_BYTES {
                dst.write_all(&pending).await?;
                BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                pending.clear();
                first_request_handled = true;
                continue;
            }
            let Some(header_end) = header_end_offset(&pending) else {
                continue;
            };
            let headers = match std::str::from_utf8(&pending[..header_end]) {
                Ok(h) => h,
                Err(_) => {
                    dst.write_all(&pending).await?;
                    BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                    pending.clear();
                    first_request_handled = true;
                    continue;
                }
            };
            if has_chunked_encoding(headers) {
                dst.write_all(&pending).await?;
                BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
                pending.clear();
                first_request_handled = true;
                continue;
            }
            let content_length = parse_content_length(headers).unwrap_or(0);
            let total_len = header_end + content_length;
            if pending.len() < total_len {
                continue;
            }
            let request_bytes = &pending[..total_len];
            let rewritten = rewrite_chat_completions_request(request_bytes)
                .unwrap_or_else(|| request_bytes.to_vec());
            dst.write_all(&rewritten).await?;
            BYTES_TRANSFERRED.fetch_add(rewritten.len() as u64, Ordering::Relaxed);
            if pending.len() > total_len {
                let tail = &pending[total_len..];
                dst.write_all(tail).await?;
                BYTES_TRANSFERRED.fetch_add(tail.len() as u64, Ordering::Relaxed);
            }
            pending.clear();
            first_request_handled = true;
            continue;
        }

        dst.write_all(&buf[..n]).await?;
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
    }

    if !pending.is_empty() {
        dst.write_all(&pending).await?;
        BYTES_TRANSFERRED.fetch_add(pending.len() as u64, Ordering::Relaxed);
    }
    Ok(())
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

    /// Update the local rpc-server port (for inbound tunnel streams).
    #[allow(dead_code)]
    pub fn set_rpc_port(&self, port: u16) {
        self.rpc_port.store(port, Ordering::Relaxed);
        tracing::info!("Tunnel manager: rpc_port updated to {port}");
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
    let tcp_stream = TcpStream::connect(format!("127.0.0.1:{http_port}")).await?;
    tcp_stream.set_nodelay(true)?;
    let _inflight = node.begin_inflight_request();

    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

pub async fn relay_tcp_streams_with_usage(a: TcpStream, b: TcpStream) -> Result<HttpRelayUsage> {
    let usage = Arc::new(std::sync::Mutex::new(UsageCapture::default()));
    let usage2 = usage.clone();
    let (a_read, mut a_write) = tokio::io::split(a);
    let (b_read, mut b_write) = tokio::io::split(b);
    let mut t1 = tokio::spawn(async move {
        relay_http_request_to_tcp_with_optional_rewrite(a_read, &mut b_write).await
    });
    let mut t2 =
        tokio::spawn(
            async move { relay_tcp_to_tcp_with_usage(b_read, &mut a_write, usage2).await },
        );
    tokio::select! {
        _ = &mut t1 => { t2.abort(); }
        _ = &mut t2 => { t1.abort(); }
    }
    let snapshot = usage.lock().unwrap().snapshot();
    Ok(snapshot)
}

pub async fn relay_tcp_via_quic_with_usage(
    tcp_stream: TcpStream,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<HttpRelayUsage> {
    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional_with_usage(tcp_read, tcp_write, quic_send, quic_recv).await
}

/// Bidirectional relay. When either direction finishes, abort the other.
pub async fn relay_bidirectional(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let _ = relay_bidirectional_with_usage(tcp_read, tcp_write, quic_send, quic_recv).await?;
    Ok(())
}

async fn relay_bidirectional_with_usage(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<HttpRelayUsage> {
    let usage = Arc::new(std::sync::Mutex::new(UsageCapture::default()));
    let usage2 = usage.clone();
    let mut t1 = tokio::spawn(async move {
        relay_http_request_to_quic_with_optional_rewrite(tcp_read, quic_send).await
    });
    let mut t2 =
        tokio::spawn(
            async move { relay_quic_to_tcp_with_usage(quic_recv, tcp_write, usage2).await },
        );
    // When either direction finishes, abort the other so we don't leak
    // tasks waiting on a half-open connection (rpc-server keeps TCP open).
    tokio::select! {
        _ = &mut t1 => { t2.abort(); }
        _ = &mut t2 => { t1.abort(); }
    }
    let snapshot = usage.lock().unwrap().snapshot();
    Ok(snapshot)
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

async fn relay_quic_to_tcp_with_usage(
    mut quic_recv: iroh::endpoint::RecvStream,
    mut tcp_write: tokio::io::WriteHalf<TcpStream>,
    usage: Arc<std::sync::Mutex<UsageCapture>>,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    tracing::debug!("QUIC→TCP: starting relay, about to first read");

    // First-byte timeout: if remote doesn't respond within 10s, it's dead.
    // After first byte arrives, no timeout (streaming responses can take minutes).
    let first_read =
        tokio::time::timeout(std::time::Duration::from_secs(10), quic_recv.read(&mut buf)).await;
    match first_read {
        Err(_) => {
            anyhow::bail!("QUIC→TCP: no response within 10s — host likely dead");
        }
        Ok(Ok(Some(n))) => {
            tcp_write.write_all(&buf[..n]).await?;
            usage.lock().unwrap().ingest(&buf[..n]);
            total += n as u64;
            BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
            tracing::debug!("QUIC→TCP: first read {n} bytes");
        }
        Ok(Ok(None)) => {
            tracing::info!("QUIC→TCP: stream end immediately (0 bytes)");
            return Ok(());
        }
        Ok(Err(e)) => {
            tracing::warn!("QUIC→TCP: error on first read: {e}");
            return Err(e.into());
        }
    }

    // After first byte, relay without timeout
    loop {
        match quic_recv.read(&mut buf).await {
            Ok(Some(n)) => {
                tcp_write.write_all(&buf[..n]).await?;
                usage.lock().unwrap().ingest(&buf[..n]);
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

async fn relay_tcp_to_tcp_with_usage(
    mut src: tokio::io::ReadHalf<TcpStream>,
    dst: &mut tokio::io::WriteHalf<TcpStream>,
    usage: Arc<std::sync::Mutex<UsageCapture>>,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = src.read(&mut buf).await?;
        if n == 0 {
            return Ok(());
        }
        dst.write_all(&buf[..n]).await?;
        usage.lock().unwrap().ingest(&buf[..n]);
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_status_code() {
        let response = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{}";
        assert_eq!(parse_http_status_code(response), Some(200));
    }

    #[test]
    fn parses_usage_fields() {
        let body =
            br#"{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":128,"total_tokens":170}}"#;
        let mut usage = UsageCapture::default();
        usage.ingest(body);
        let snapshot = usage.snapshot();
        assert_eq!(snapshot.prompt_tokens, 42);
        assert_eq!(snapshot.completion_tokens, 128);
    }

    #[test]
    fn parses_stream_usage_alias_fields() {
        let body = br#"data: {"usage":{"input_tokens":11,"output_tokens":19}}"#;
        let mut usage = UsageCapture::default();
        usage.ingest(body);
        let snapshot = usage.snapshot();
        assert_eq!(snapshot.prompt_tokens, 11);
        assert_eq!(snapshot.completion_tokens, 19);
    }

    #[test]
    fn rewrites_chat_completion_request_to_include_usage() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 33\r\n\r\n{\"stream\":true,\"model\":\"qwen3\"}";
        let rewritten = rewrite_chat_completions_request(req).expect("request should rewrite");
        let as_text = String::from_utf8(rewritten).expect("valid utf8");
        assert!(as_text.contains("\"stream_options\":{\"include_usage\":true}"));
    }
}
