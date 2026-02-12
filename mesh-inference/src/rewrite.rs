//! REGISTER_PEER endpoint rewriting.
//!
//! The B2B fork's orchestrator sends RPC_CMD_REGISTER_PEER (command byte 17)
//! to tell each worker about its peers. The endpoint string in that message
//! is a `host:port` that was valid on the orchestrator's machine but meaningless
//! on the worker's machine.
//!
//! This module intercepts that one command in the QUIC→TCP relay path and
//! rewrites the endpoint to the local tunnel port on this machine that
//! corresponds to the given peer_id.
//!
//! Wire format:
//!   Client→Server: | cmd (1 byte) | payload_size (8 bytes LE) | payload |
//!
//! REGISTER_PEER payload (132 bytes):
//!   | peer_id (4 bytes LE) | endpoint (128 bytes, null-terminated string) |
//!
//! We rewrite the 128-byte endpoint field to "127.0.0.1:<tunnel_port>".
//!
//! All other commands pass through as raw bytes.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

const RPC_CMD_REGISTER_PEER: u8 = 17;
const REGISTER_PEER_PAYLOAD_SIZE: usize = 4 + 128; // peer_id + endpoint

/// Relay bytes from QUIC recv to TCP write, rewriting REGISTER_PEER commands.
///
/// We parse just enough of the RPC framing to identify commands:
///   - Read 1 byte: command
///   - Read 8 bytes: payload_size
///   - Read payload_size bytes: payload
///
/// For REGISTER_PEER, we rewrite the endpoint field in the payload.
/// For everything else, we forward all bytes verbatim, streaming large
/// payloads (e.g. SET_TENSOR) without buffering.
pub async fn relay_with_rewrite(
    mut quic_recv: iroh::endpoint::RecvStream,
    mut tcp_write: tokio::io::WriteHalf<tokio::net::TcpStream>,
    peer_id_map: Arc<Mutex<HashMap<u32, u16>>>,
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

            // Look up the local tunnel port for this peer_id
            let map = peer_id_map.lock().await;
            if let Some(&tunnel_port) = map.get(&peer_id) {
                // Rewrite endpoint field (bytes 4..132)
                let new_endpoint = format!("127.0.0.1:{tunnel_port}");
                let mut endpoint_bytes = [0u8; 128];
                let copy_len = new_endpoint.len().min(127);
                endpoint_bytes[..copy_len].copy_from_slice(&new_endpoint.as_bytes()[..copy_len]);
                payload[4..132].copy_from_slice(&endpoint_bytes);

                tracing::info!(
                    "Rewrote REGISTER_PEER: peer_id={peer_id} → 127.0.0.1:{tunnel_port}"
                );
            } else {
                tracing::warn!(
                    "REGISTER_PEER for unknown peer_id={peer_id}, passing through unmodified"
                );
            }

            // Forward rewritten command
            tcp_write.write_all(&[cmd]).await?;
            tcp_write.write_all(&size_buf).await?;
            tcp_write.write_all(&payload).await?;
        } else {
            // Not REGISTER_PEER — forward verbatim, streaming the payload
            tcp_write.write_all(&[cmd]).await?;
            tcp_write.write_all(&size_buf).await?;

            // Stream payload without buffering the whole thing
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
