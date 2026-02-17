//! Automatic host election and dynamic mesh management.
//!
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! llama-server always uses --rpc (even solo — host's own rpc-server is in the list).
//! mesh-llm owns :8080 and proxies to llama-server (local or remote).

use crate::{launch, mesh, tunnel};
use mesh::NodeRole;
use std::path::Path;

use tokio::sync::watch;

/// Determine if this node should be host.
/// Deterministic from gossip state — every node computes the same answer.
/// Highest VRAM wins. Tie-break: highest node ID.
pub fn should_be_host(my_id: iroh::EndpointId, my_vram: u64, peers: &[mesh::PeerInfo]) -> bool {
    for peer in peers {
        // Only consider active nodes (workers), not clients
        if matches!(peer.role, NodeRole::Client) {
            continue;
        }
        if peer.vram_bytes > my_vram {
            return false;
        }
        if peer.vram_bytes == my_vram && peer.id > my_id {
            return false;
        }
    }
    true
}

/// The current state of llama-server as managed by the election loop.
/// The API proxy reads this to know where to forward requests.
#[derive(Clone, Debug)]
pub enum InferenceTarget {
    /// No llama-server running anywhere (election in progress, mesh empty, etc.)
    None,
    /// We are host — llama-server is on this local port.
    Local(u16),
    /// Another node is host — proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
}

/// Background election loop. On every mesh change:
/// 1. Kill llama-server (if we're running it)
/// 2. Re-elect (deterministic — highest VRAM wins)
/// 3. Winner starts llama-server with --rpc pointing at all nodes
///
/// Publishes the current InferenceTarget via the watch channel so the
/// API proxy knows where to forward requests.
pub async fn election_loop(
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    rpc_port: u16,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    draft: Option<std::path::PathBuf>,
    draft_max: u16,
    target_tx: watch::Sender<InferenceTarget>,
    mut on_change: impl FnMut(bool, bool) + Send,
) {
    let mut peer_rx = node.peer_change_rx.clone();

    // Initial settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    loop {
        // Step 1: Kill llama-server if we're running it
        let was_host = matches!(node.role().await, NodeRole::Host { .. });
        if was_host {
            launch::kill_llama_server().await;
            tunnel_mgr.set_http_port(0);
            node.set_role(NodeRole::Worker).await;
            target_tx.send_replace(InferenceTarget::None);
            on_change(false, false);
        }

        // Step 2: Elect
        let peers = node.peers().await;
        let i_am_host = should_be_host(node.id(), node.vram_bytes(), &peers);

        if i_am_host {
            // Check if total mesh VRAM is enough for the model.
            // Model needs roughly file_size * 1.1 for weights + KV cache overhead.
            let model_bytes = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
            let min_vram = (model_bytes as f64 * 1.1) as u64;
            let my_vram = node.vram_bytes();
            let peer_vram: u64 = peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .map(|p| p.vram_bytes)
                .sum();
            let total_vram = my_vram + peer_vram;

            if total_vram < min_vram {
                eprintln!("⏳ Waiting for more peers — need {:.1}GB VRAM for model, have {:.1}GB",
                    min_vram as f64 / 1e9, total_vram as f64 / 1e9);
                target_tx.send_replace(InferenceTarget::None);
                on_change(false, false);
                // Wait for next peer change (someone joining with more VRAM)
                if peer_rx.changed().await.is_err() { break; }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                continue;
            }

            eprintln!("🗳 Elected as host ({:.1}GB VRAM available for {:.1}GB model)",
                total_vram as f64 / 1e9, model_bytes as f64 / 1e9);
            on_change(true, false);

            let llama_port = match start_llama(
                &node, &tunnel_mgr, rpc_port, &bin_dir, &model, draft.as_deref(), draft_max,
            ).await {
                Some(port) => port,
                None => {
                    on_change(true, false);
                    // Wait for next mesh change and retry
                    let _ = peer_rx.changed().await;
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            node.set_role(NodeRole::Host { http_port: llama_port }).await;
            tunnel_mgr.set_http_port(llama_port);
            target_tx.send_replace(InferenceTarget::Local(llama_port));
            on_change(true, true);
            eprintln!("✅ llama-server ready on internal port {llama_port}");
        } else {
            // We're a worker. Find who the host is.
            node.set_role(NodeRole::Worker).await;

            // The host might not have announced yet — look for highest VRAM peer
            let host_peer = peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.vram_bytes, p.id));

            if let Some(host) = host_peer {
                if should_be_host(host.id, host.vram_bytes, &peers) {
                    target_tx.send_replace(InferenceTarget::Remote(host.id));
                    eprintln!("📡 Worker — host is {}", host.id.fmt_short());
                } else {
                    target_tx.send_replace(InferenceTarget::None);
                }
            } else {
                target_tx.send_replace(InferenceTarget::None);
            }
            on_change(false, false);
        }

        // Wait for next peer change
        if peer_rx.changed().await.is_err() {
            break;
        }

        // Debounce — wait for mesh to settle
        eprintln!("⚡ Mesh changed — re-electing...");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Start llama-server with --rpc pointing at all nodes (self + workers).
/// Returns the ephemeral port llama-server is listening on, or None on failure.
async fn start_llama(
    node: &mesh::Node,
    tunnel_mgr: &tunnel::Manager,
    my_rpc_port: u16,
    bin_dir: &Path,
    model: &Path,
    draft: Option<&Path>,
    draft_max: u16,
) -> Option<u16> {
    let peers = node.peers().await;
    let worker_ids: Vec<_> = peers.iter()
        .filter(|p| matches!(p.role, NodeRole::Worker))
        .map(|p| p.id)
        .collect();

    // Wait for tunnels to workers
    if !worker_ids.is_empty() {
        eprintln!("  Waiting for tunnels to {} worker(s)...", worker_ids.len());
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tunnel_mgr.wait_for_peers(worker_ids.len()),
        ).await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        // B2B tunnel map exchange
        let my_map = tunnel_mgr.peer_ports_map().await;
        let _ = node.broadcast_tunnel_map(my_map).await;
        let _ = node.wait_for_tunnel_maps(worker_ids.len(), std::time::Duration::from_secs(10)).await;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;
    }

    // Build --rpc list: self first, then remote workers
    let all_ports = tunnel_mgr.peer_ports_map().await;
    let mut rpc_ports = vec![my_rpc_port]; // always include self
    for id in &worker_ids {
        if let Some(&port) = all_ports.get(id) {
            rpc_ports.push(port);
        }
    }

    // Calculate tensor split from VRAM
    let my_vram = node.vram_bytes() as f64;
    let mut all_vrams = vec![my_vram];
    for id in &worker_ids {
        if let Some(peer) = peers.iter().find(|p| p.id == *id) {
            all_vrams.push(if peer.vram_bytes > 0 { peer.vram_bytes as f64 } else { my_vram });
        }
    }
    let total: f64 = all_vrams.iter().sum();
    let split = if total > 0.0 && rpc_ports.len() > 1 {
        let s: Vec<String> = all_vrams.iter().map(|v| format!("{:.2}", v / total)).collect();
        let split_str = s.join(",");
        eprintln!("  Tensor split: {split_str} ({} node(s), {:.0}GB total)", rpc_ports.len(), total / 1e9);
        Some(split_str)
    } else {
        eprintln!("  Solo mode ({:.0}GB)", my_vram / 1e9);
        None
    };

    // Launch on ephemeral port
    let llama_port = match find_free_port().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Failed to find free port: {e}");
            return None;
        }
    };

    match launch::start_llama_server(
        bin_dir, model, llama_port, &rpc_ports, split.as_deref(), draft, draft_max,
    ).await {
        Ok(()) => Some(llama_port),
        Err(e) => {
            eprintln!("  Failed to start llama-server: {e}");
            None
        }
    }
}

async fn find_free_port() -> anyhow::Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}
