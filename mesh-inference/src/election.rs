//! Automatic host election and dynamic mesh management.
//!
//! Shared between CLI and console modes. Every Active node (with rpc-server)
//! participates in election. Highest VRAM wins host. Worker count changes
//! trigger llama-server restart with updated tensor-split.

use crate::{launch, mesh, tunnel};
use mesh::NodeRole;
use std::path::Path;

/// Determine if this node should be host.
/// Rules:
/// 1. If another peer is already Host → I should not be host (stability)
/// 2. If I'm already Host → I stay host (stability)
/// 3. If no host exists → highest VRAM wins, tie-break by node ID
pub async fn should_be_host(node: &mesh::Node) -> bool {
    let my_id = node.id();
    let my_vram = node.vram_bytes();
    let my_role = node.role().await;
    let peers = node.peers().await;

    // If another peer is already Host, I shouldn't take over
    let other_host = peers.iter().find(|p| matches!(p.role, NodeRole::Host { .. }));
    if other_host.is_some() {
        return false;
    }

    // If I'm already Host, stay host (stability)
    if matches!(my_role, NodeRole::Host { .. }) {
        return true;
    }

    // No host exists anywhere. Elect by VRAM (highest wins), tie-break by ID.
    // Only consider Active peers (Workers), not Clients.
    for peer in &peers {
        if !matches!(peer.role, NodeRole::Worker | NodeRole::Host { .. }) {
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

/// Count Worker peers in the mesh.
pub async fn count_workers(node: &mesh::Node) -> usize {
    node.peers().await.iter()
        .filter(|p| matches!(p.role, NodeRole::Worker))
        .count()
}

/// Calculate tensor split and launch llama-server for the current mesh state.
/// Returns true if llama-server started successfully.
pub async fn launch_llama_for_current_mesh(
    node: &mesh::Node,
    tunnel_mgr: &tunnel::Manager,
    bin_dir: &Path,
    model: &Path,
    llama_port: u16,
) -> bool {
    let existing_peers = node.peers().await;
    let has_workers = existing_peers.iter().any(|p| matches!(p.role, NodeRole::Worker));
    if has_workers {
        eprintln!("Found worker(s) in mesh, waiting for tunnels...");
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tunnel_mgr.wait_for_peers(1),
        ).await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    } else {
        eprintln!("No workers in mesh — starting with local GPU only");
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    let worker_peers = node.peers().await;
    let worker_ids: Vec<_> = worker_peers.iter()
        .filter(|p| matches!(p.role, NodeRole::Worker))
        .map(|p| p.id)
        .collect();
    let all_ports = tunnel_mgr.peer_ports_map().await;
    let tunnel_ports: Vec<u16> = worker_ids.iter()
        .filter_map(|id| all_ports.get(id).copied())
        .collect();
    let n_peers = tunnel_ports.len();

    if n_peers > 0 {
        eprintln!("Got {n_peers} GPU worker(s), starting llama-server");
        let my_map = tunnel_mgr.peer_ports_map().await;
        let _ = node.broadcast_tunnel_map(my_map).await;
        let _ = node.wait_for_tunnel_maps(n_peers, std::time::Duration::from_secs(15)).await;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;
    } else {
        eprintln!("No workers found — starting llama-server with local GPU only");
    }

    let effective_split = if n_peers > 0 {
        let my_vram = node.vram_bytes() as f64;
        let worker_vrams: Vec<f64> = worker_ids.iter()
            .filter_map(|id| worker_peers.iter().find(|p| p.id == *id))
            .map(|p| if p.vram_bytes > 0 { p.vram_bytes as f64 } else { my_vram })
            .collect();
        let total: f64 = my_vram + worker_vrams.iter().sum::<f64>();
        if total > 0.0 {
            let mut parts = vec![format!("{:.2}", my_vram / total)];
            for wv in &worker_vrams {
                parts.push(format!("{:.2}", wv / total));
            }
            let split = parts.join(",");
            eprintln!("Auto tensor-split: {split} (local {:.1}GB + {} worker(s))",
                my_vram / 1e9, worker_vrams.len());
            Some(split)
        } else {
            None
        }
    } else {
        None
    };

    match launch::start_llama_server(
        bin_dir, model, llama_port, &tunnel_ports, effective_split.as_deref(),
    ).await {
        Ok(()) => {
            eprintln!("llama-server ready: http://localhost:{llama_port}");
            true
        }
        Err(e) => {
            eprintln!("Failed to start llama-server: {e}");
            false
        }
    }
}

/// Background election loop. Watches peer changes and manages host promotion/demotion.
/// Calls `on_host_change(is_host, llama_ready)` on every state change.
pub async fn election_loop<F>(
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    llama_port: u16,
    mut on_change: F,
) where
    F: FnMut(bool, bool) + Send,
{
    let mut peer_rx = node.peer_change_rx.clone();
    let mut currently_host = false;
    let mut last_worker_count: usize = 0;

    // Initial election after a brief settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    loop {
        let want_host = should_be_host(&node).await;
        let worker_count = count_workers(&node).await;

        if want_host && !currently_host {
            // Promote to host
            eprintln!("🗳 Elected as host — starting llama-server");
            node.set_role(NodeRole::Host { http_port: llama_port }).await;
            on_change(true, false);

            let ok = launch_llama_for_current_mesh(
                &node, &tunnel_mgr, &bin_dir, &model, llama_port,
            ).await;
            currently_host = true;
            last_worker_count = count_workers(&node).await;
            on_change(true, ok);

        } else if !want_host && currently_host {
            // Another node is now host — demote
            eprintln!("🗳 Another node is now host — demoting to worker");
            launch::kill_llama_server();
            node.set_role(NodeRole::Worker).await;
            currently_host = false;
            on_change(false, false);

        } else if currently_host && worker_count != last_worker_count {
            // Worker count changed while we're host — restart with new split
            eprintln!("⚡ Worker count changed: {} → {}. Restarting llama-server...",
                last_worker_count, worker_count);
            launch::kill_llama_server();
            on_change(true, false);
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;

            let ok = launch_llama_for_current_mesh(
                &node, &tunnel_mgr, &bin_dir, &model, llama_port,
            ).await;
            last_worker_count = count_workers(&node).await;
            on_change(true, ok);
        }

        // Wait for next peer change
        if peer_rx.changed().await.is_err() {
            break;
        }

        // Debounce — wait for mesh to settle
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}
