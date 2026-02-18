//! Automatic host election and dynamic mesh management.
//!
//! Per-model election: nodes serving the same model form a group.
//! The highest-VRAM node in each group becomes its host and runs llama-server.
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! mesh-llm owns :api_port and proxies to the right host by model name.

use crate::{launch, mesh, tunnel};
use mesh::NodeRole;
use std::collections::HashMap;
use std::path::Path;

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf
fn total_model_bytes(model: &Path) -> u64 {
    let name = model.to_string_lossy();
    // Check for split pattern: *-00001-of-NNNNN.gguf
    if let Some(pos) = name.find("-00001-of-") {
        let of_pos = pos + 10;
        if let Some(ext_pos) = name[of_pos..].find(".gguf") {
            if let Ok(n_split) = name[of_pos..of_pos + ext_pos].parse::<u32>() {
                let prefix = &name[..pos + 1];
                let suffix = &name[of_pos + ext_pos..];
                let mut total: u64 = 0;
                for i in 1..=n_split {
                    let split_name = format!("{}{:05}-of-{:05}{}", prefix, i, n_split, suffix);
                    total += std::fs::metadata(&split_name).map(|m| m.len()).unwrap_or(0);
                }
                return total;
            }
        }
    }
    std::fs::metadata(model).map(|m| m.len()).unwrap_or(0)
}

use tokio::sync::watch;

/// Determine if this node should be host for its model group.
/// Only considers peers serving the same model.
/// Deterministic: highest VRAM wins, tie-break by node ID.
pub fn should_be_host_for_model(my_id: iroh::EndpointId, my_vram: u64, model_peers: &[mesh::PeerInfo]) -> bool {
    for peer in model_peers {
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

/// Per-model routing table. The API proxy uses this to route by model name.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name → InferenceTarget
    pub targets: HashMap<String, InferenceTarget>,
}

impl ModelTargets {
    /// Get target for a specific model.
    pub fn get(&self, model: &str) -> InferenceTarget {
        self.targets.get(model).cloned().unwrap_or(InferenceTarget::None)
    }

    /// List all available model names.
    pub fn available_models(&self) -> Vec<String> {
        self.targets.keys()
            .filter(|k| !matches!(self.targets[k.as_str()], InferenceTarget::None))
            .cloned()
            .collect()
    }
}

/// Background election loop for a single model.
/// This node serves `model` — it only cares about peers also serving `model`.
///
/// On every mesh change:
/// 1. Kill llama-server (if we're running it)
/// 2. Re-elect within the model group
/// 3. Winner starts llama-server with --rpc pointing at group nodes
///
/// Publishes the current ModelTargets via the watch channel so the
/// API proxy knows where to forward requests.
pub async fn election_loop(
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    rpc_port: u16,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    model_name: String,
    draft: Option<std::path::PathBuf>,
    draft_max: u16,
    force_split: bool,
    target_tx: watch::Sender<ModelTargets>,
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
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_change(false, false);
        }

        // Step 2: Elect within our model group
        let peers = node.peers().await;
        let model_peers: Vec<mesh::PeerInfo> = peers.iter()
            .filter(|p| p.serving.as_deref() == Some(&model_name))
            .cloned()
            .collect();

        let i_am_host = should_be_host_for_model(node.id(), node.vram_bytes(), &model_peers);

        if i_am_host {
            // Check if total model-group VRAM is enough
            let model_bytes = total_model_bytes(&model);
            let min_vram = (model_bytes as f64 * 1.1) as u64;
            let my_vram = node.vram_bytes();
            let peer_vram: u64 = model_peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .map(|p| p.vram_bytes)
                .sum();
            let total_vram = my_vram + peer_vram;

            if total_vram < min_vram {
                eprintln!("⏳ [{}] Waiting for more peers — need {:.1}GB VRAM, have {:.1}GB",
                    model_name, min_vram as f64 / 1e9, total_vram as f64 / 1e9);
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                on_change(false, false);
                if peer_rx.changed().await.is_err() { break; }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                continue;
            }

            eprintln!("🗳 [{}] Elected as host ({:.1}GB VRAM for {:.1}GB model, {} node(s))",
                model_name, total_vram as f64 / 1e9, model_bytes as f64 / 1e9, model_peers.len() + 1);
            on_change(true, false);

            let llama_port = match start_llama(
                &node, &tunnel_mgr, rpc_port, &bin_dir, &model, &model_name,
                &model_peers, draft.as_deref(), draft_max, force_split,
            ).await {
                Some(port) => port,
                None => {
                    on_change(true, false);
                    let _ = peer_rx.changed().await;
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            node.set_role(NodeRole::Host { http_port: llama_port }).await;
            tunnel_mgr.set_http_port(llama_port);
            // Re-gossip so peers learn we're the host for this model
            node.regossip().await;
            update_targets(&node, &model_name, InferenceTarget::Local(llama_port), &target_tx).await;
            on_change(true, true);
            eprintln!("✅ [{}] llama-server ready on internal port {llama_port}", model_name);
        } else {
            // We're a worker for this model. Find who the host is.
            node.set_role(NodeRole::Worker).await;

            // Look for the highest VRAM peer in our model group
            let host_peer = model_peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.vram_bytes, p.id));

            if let Some(host) = host_peer {
                if should_be_host_for_model(host.id, host.vram_bytes, &model_peers) {
                    update_targets(&node, &model_name, InferenceTarget::Remote(host.id), &target_tx).await;
                    eprintln!("📡 [{}] Worker — host is {}", model_name, host.id.fmt_short());
                } else {
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                }
            } else {
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            }
            on_change(false, false);
        }

        // Wait for next peer change
        if peer_rx.changed().await.is_err() {
            break;
        }

        eprintln!("⚡ Mesh changed — re-electing...");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Update the model targets map — sets our model's target and includes
/// targets for other models we know about from peers.
async fn update_targets(
    node: &mesh::Node,
    my_model: &str,
    my_target: InferenceTarget,
    target_tx: &watch::Sender<ModelTargets>,
) {
    let peers = node.peers().await;
    let mut targets = HashMap::new();

    // Our model
    targets.insert(my_model.to_string(), my_target);

    // Other models being served — find their hosts
    let mut other_models: HashMap<String, Vec<&mesh::PeerInfo>> = HashMap::new();
    for p in &peers {
        if let Some(ref serving) = p.serving {
            if serving != my_model {
                other_models.entry(serving.clone()).or_default().push(p);
            }
        }
    }
    for (model, model_peers) in &other_models {
        // Find the host for this model (highest VRAM peer serving it with Host role, or predict who will be)
        if let Some(host) = model_peers.iter().find(|p| matches!(p.role, NodeRole::Host { .. })) {
            targets.insert(model.clone(), InferenceTarget::Remote(host.id));
        } else {
            // No host announced yet — predict based on VRAM
            if let Some(likely_host) = model_peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.vram_bytes, p.id))
            {
                targets.insert(model.clone(), InferenceTarget::Remote(likely_host.id));
            }
        }
    }

    target_tx.send_replace(ModelTargets { targets });
}

/// Start llama-server with --rpc pointing at model-group nodes (self + workers).
/// Returns the ephemeral port llama-server is listening on, or None on failure.
async fn start_llama(
    node: &mesh::Node,
    tunnel_mgr: &tunnel::Manager,
    my_rpc_port: u16,
    bin_dir: &Path,
    model: &Path,
    model_name: &str,
    model_peers: &[mesh::PeerInfo],
    draft: Option<&Path>,
    draft_max: u16,
    force_split: bool,
) -> Option<u16> {
    let my_vram = node.vram_bytes();
    let model_bytes = total_model_bytes(model);
    let min_vram = (model_bytes as f64 * 1.1) as u64;

    // Decide whether to split: only if model doesn't fit on host alone, or --split forced
    let need_split = force_split || my_vram < min_vram;

    const MAX_RTT_MS: u32 = 80;

    // Only use workers from our model group
    let worker_ids: Vec<_> = if need_split {
        model_peers.iter()
            .filter(|p| matches!(p.role, NodeRole::Worker) || p.serving.as_deref() == Some(model_name))
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .filter(|p| {
                match p.rtt_ms {
                    Some(rtt) if rtt > MAX_RTT_MS => {
                        eprintln!("  ⚠ Skipping {} — RTT {}ms exceeds {}ms limit",
                            p.id.fmt_short(), rtt, MAX_RTT_MS);
                        false
                    }
                    _ => true,
                }
            })
            .map(|p| p.id)
            .collect()
    } else {
        let worker_count = model_peers.iter()
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .count();
        if worker_count > 0 {
            eprintln!("  Model fits on host ({:.1}GB VRAM for {:.1}GB model) — loading solo",
                my_vram as f64 / 1e9, model_bytes as f64 / 1e9);
            eprintln!("  Use --split to force distributed mode");
        }
        vec![]
    };

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
    let my_vram_f = my_vram as f64;
    let mut all_vrams = vec![my_vram_f];
    for id in &worker_ids {
        if let Some(peer) = model_peers.iter().find(|p| p.id == *id) {
            all_vrams.push(if peer.vram_bytes > 0 { peer.vram_bytes as f64 } else { my_vram_f });
        }
    }
    let total: f64 = all_vrams.iter().sum();
    let split = if total > 0.0 && rpc_ports.len() > 1 {
        let s: Vec<String> = all_vrams.iter().map(|v| format!("{:.2}", v / total)).collect();
        let split_str = s.join(",");
        eprintln!("  Tensor split: {split_str} ({} node(s), {:.0}GB total)", rpc_ports.len(), total / 1e9);
        Some(split_str)
    } else {
        eprintln!("  Solo mode ({:.0}GB)", my_vram_f / 1e9);
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
