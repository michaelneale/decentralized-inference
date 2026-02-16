//! REGISTER_PEER endpoint rewriting.
//!
//! The B2B fork's orchestrator sends RPC_CMD_REGISTER_PEER to tell each
//! worker about its peers. The endpoint string in that message is a
//! `host:port` that was valid on the orchestrator's machine but meaningless
//! on the worker's machine.
//!
//! This module intercepts that command in the QUIC→TCP relay path (inbound
//! tunnel to local rpc-server) and rewrites the endpoint to the local tunnel
//! port on this machine that routes to the correct peer.
//!
//! Wire format:
//!   Client→Server: | cmd (1 byte) | payload_size (8 bytes LE) | payload |
//!
//! REGISTER_PEER payload (132 bytes):
//!   | peer_id (4 bytes LE) | endpoint (128 bytes, null-terminated string) |
//!
//! We parse the port from the endpoint string, look it up in the
//! `remote_port → local_port` map, and rewrite the endpoint field.
//!
//! All other commands pass through as raw bytes.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

const RPC_CMD_REGISTER_PEER: u8 = 18; // enum position after SET_TENSOR_GGUF (17)
const REGISTER_PEER_PAYLOAD_SIZE: usize = 4 + 128; // peer_id + endpoint

/// Shared map from orchestrator's tunnel port → worker's local tunnel port.
/// Built by combining the orchestrator's tunnel map (received via gossip)
/// with the worker's own tunnel map.
pub type PortRewriteMap = Arc<RwLock<HashMap<u16, u16>>>;

/// Create a new empty rewrite map.
pub fn new_rewrite_map() -> PortRewriteMap {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Relay bytes from QUIC recv to TCP write, rewriting REGISTER_PEER commands.
///
/// RPC framing: | cmd (1 byte) | payload_size (8 bytes LE) | payload |
///
/// For REGISTER_PEER, we rewrite the endpoint field.
/// For everything else, we stream bytes through verbatim.
pub async fn relay_with_rewrite(
    mut quic_recv: iroh::endpoint::RecvStream,
    mut tcp_write: tokio::io::WriteHalf<tokio::net::TcpStream>,
    port_map: PortRewriteMap,
) -> Result<()> {

    loop {
        // Read command byte
        let mut cmd_buf = [0u8; 1];
        if quic_recv.read_exact(&mut cmd_buf).await.is_err() {
            break; // stream closed
        }
        let cmd = cmd_buf[0];

        // Read payload size (8 bytes LE)
        let mut size_buf = [0u8; 8];
        quic_recv.read_exact(&mut size_buf).await?;
        let payload_size = u64::from_le_bytes(size_buf);

        if cmd == RPC_CMD_REGISTER_PEER && payload_size as usize == REGISTER_PEER_PAYLOAD_SIZE {
            // Read the full payload
            let mut payload = vec![0u8; payload_size as usize];
            quic_recv.read_exact(&mut payload).await?;

            // Extract peer_id (first 4 bytes LE)
            let peer_id = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);

            // Extract endpoint string (bytes 4..132) — copy to avoid borrow conflict
            let endpoint_bytes = &payload[4..132];
            let endpoint_str = std::str::from_utf8(
                &endpoint_bytes[..endpoint_bytes.iter().position(|&b| b == 0).unwrap_or(128)]
            ).unwrap_or("").to_string();

            // Parse port from endpoint string like "127.0.0.1:49502"
            if let Some(port_str) = endpoint_str.rsplit(':').next() {
                if let Ok(remote_port) = port_str.parse::<u16>() {
                    let map = port_map.read().await;
                    if let Some(&local_port) = map.get(&remote_port) {
                        // Rewrite endpoint field
                        let new_endpoint = format!("127.0.0.1:{local_port}");
                        let mut new_endpoint_bytes = [0u8; 128];
                        let copy_len = new_endpoint.len().min(127);
                        new_endpoint_bytes[..copy_len]
                            .copy_from_slice(&new_endpoint.as_bytes()[..copy_len]);
                        payload[4..132].copy_from_slice(&new_endpoint_bytes);

                        tracing::info!(
                            "Rewrote REGISTER_PEER: peer_id={peer_id} \
                             {endpoint_str} → 127.0.0.1:{local_port}"
                        );
                    } else {
                        tracing::warn!(
                            "REGISTER_PEER: no rewrite mapping for port {remote_port} \
                             (peer_id={peer_id}, endpoint={endpoint_str}), passing through"
                        );
                    }
                }
            }

            // Forward (possibly rewritten) command
            tcp_write.write_all(&[cmd]).await?;
            tcp_write.write_all(&size_buf).await?;
            tcp_write.write_all(&payload).await?;
        } else {
            // Not REGISTER_PEER — forward verbatim, streaming the payload
            tcp_write.write_all(&[cmd]).await?;
            tcp_write.write_all(&size_buf).await?;

            // Stream payload in chunks
            let mut remaining = payload_size;
            let mut buf = vec![0u8; 64 * 1024];
            while remaining > 0 {
                let to_read = (remaining as usize).min(buf.len());
                let n = quic_recv.read(&mut buf[..to_read]).await?
                    .ok_or_else(|| anyhow::anyhow!("stream closed mid-payload"))?;
                tcp_write.write_all(&buf[..n]).await?;
                remaining -= n as u64;
            }
        }

        tcp_write.flush().await?;
    }

    Ok(())
}
