//! Automatic host election and dynamic mesh management.
//!
//! Per-model election: nodes serving the same model form a group.
//! The highest-VRAM node in each group becomes its host and runs llama-server.
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! mesh-llm owns :api_port and proxies to the right host by model name.

use crate::{download, launch, mesh, moe, tunnel};
use mesh::NodeRole;
use std::collections::HashMap;
use std::path::Path;

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf
pub fn total_model_bytes(model: &Path) -> u64 {
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
    /// MoE mode — this node runs its own llama-server with its expert shard.
    /// All MoE nodes are independent; the proxy picks one per session.
    MoeLocal(u16),
    /// MoE mode — another node is running its shard; proxy via QUIC.
    MoeRemote(iroh::EndpointId),
}

/// MoE deployment state shared between election and proxy.
/// The proxy uses this to route sessions to MoE nodes.
#[derive(Clone, Debug, Default)]
pub struct MoeState {
    /// All MoE node targets (local + remote), in stable order.
    pub nodes: Vec<InferenceTarget>,
}

/// Per-model routing table. The API proxy uses this to route by model name.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name → InferenceTarget
    pub targets: HashMap<String, InferenceTarget>,
    /// MoE state — if set, this model uses MoE expert sharding.
    /// The proxy uses this for session-sticky routing across MoE nodes.
    pub moe: Option<MoeState>,
}

impl ModelTargets {
    /// Get target for a specific model.
    pub fn get(&self, model: &str) -> InferenceTarget {
        self.targets.get(model).cloned().unwrap_or(InferenceTarget::None)
    }

    /// Get MoE target for a session (hash-based routing).
    /// Returns None if not in MoE mode.
    pub fn get_moe_target(&self, session_hint: &str) -> Option<InferenceTarget> {
        let moe = self.moe.as_ref()?;
        if moe.nodes.is_empty() { return None; }
        // Simple hash routing: hash the session hint, pick a node
        let hash = session_hint.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % moe.nodes.len();
        Some(moe.nodes[idx].clone())
    }

    /// List all available model names.
    #[allow(dead_code)]
    pub fn available_models(&self) -> Vec<String> {
        self.targets.keys()
            .filter(|k| !matches!(self.targets[k.as_str()], InferenceTarget::None))
            .cloned()
            .collect()
    }
}

/// Look up MoE config for a model from the catalog.
fn lookup_moe_config(model_name: &str) -> Option<download::MoeConfig> {
    let q = model_name.to_lowercase();
    download::MODEL_CATALOG.iter()
        .find(|m| m.name.to_lowercase() == q || m.file.to_lowercase().contains(&q))
        .and_then(|m| m.moe.clone())
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

    // Track the set of model-group worker IDs to detect when we actually need to restart
    let mut last_worker_set: Vec<iroh::EndpointId> = vec![];
    let mut currently_host = false;

    // Initial settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let model_bytes = total_model_bytes(&model);
    let my_vram = node.vram_bytes();
    let model_fits_locally = my_vram >= (model_bytes as f64 * 1.1) as u64;

    // Check if this is a MoE model with pre-computed expert routing
    let moe_config = lookup_moe_config(&model_name);
    if moe_config.is_some() {
        eprintln!("🧩 [{}] MoE model detected ({} experts, top-{})",
            model_name,
            moe_config.as_ref().unwrap().n_expert,
            moe_config.as_ref().unwrap().n_expert_used);
    }

    // MoE mode: each node runs its own llama-server with its expert shard.
    // Only enter MoE split mode if the model doesn't fit locally or --split is forced.
    // Otherwise, just run the full model — every node is independent.
    if let Some(ref moe_cfg) = moe_config {
        let need_moe_split = force_split || !model_fits_locally;
        if need_moe_split {
            moe_election_loop(
                node, bin_dir, model, model_name, moe_cfg.clone(),
                target_tx, &mut on_change,
            ).await;
            return;
        } else {
            eprintln!("🧩 [{}] MoE model fits locally ({:.1}GB VRAM for {:.1}GB model) — no split needed",
                model_name, my_vram as f64 / 1e9, model_bytes as f64 / 1e9);
            // Fall through to normal election loop — each node runs full model independently
        }
    }

    loop {
        // Collect our model group (peers also serving this model)
        let peers = node.peers().await;
        let model_peers: Vec<mesh::PeerInfo> = peers.iter()
            .filter(|p| p.serving.as_deref() == Some(&model_name))
            .cloned()
            .collect();

        // Splitting decision: only split when forced OR when the model
        // genuinely doesn't fit on this node alone. If it fits, every
        // node serving this model runs its own independent llama-server
        // (no election needed — everyone is a host).
        let need_split = force_split || !model_fits_locally;

        let i_am_host = if need_split {
            // Distributed mode: elect one host from the model group
            should_be_host_for_model(node.id(), my_vram, &model_peers)
        } else {
            // Solo mode: this node always runs its own llama-server
            true
        };

        // Compute the worker set (only relevant in split mode)
        let mut new_worker_set: Vec<iroh::EndpointId> = if need_split {
            model_peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .map(|p| p.id)
                .collect()
        } else {
            vec![] // solo mode — no workers
        };
        new_worker_set.sort();

        // If we're already host and nothing changed, skip restart
        if currently_host && i_am_host && new_worker_set == last_worker_set {
            // Just update the target map (in case other models' hosts changed)
            if let NodeRole::Host { http_port } = node.role().await {
                update_targets(&node, &model_name, InferenceTarget::Local(http_port), &target_tx).await;
            }
            // Wait for next change
            if peer_rx.changed().await.is_err() { break; }
            eprintln!("⚡ Mesh changed — re-checking... (still host, no restart needed)");
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            continue;
        }

        // Something changed — kill llama-server if we were running it
        if currently_host {
            launch::kill_llama_server().await;
            tunnel_mgr.set_http_port(0);
            node.set_role(NodeRole::Worker).await;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_change(false, false);
            currently_host = false;
        }

        if i_am_host {
            if need_split {
                // Distributed mode: check total group VRAM
                let peer_vram: u64 = model_peers.iter()
                    .filter(|p| !matches!(p.role, NodeRole::Client))
                    .map(|p| p.vram_bytes)
                    .sum();
                let total_vram = my_vram + peer_vram;
                let min_vram = (model_bytes as f64 * 1.1) as u64;

                if total_vram < min_vram {
                    eprintln!("⏳ [{}] Waiting for more peers — need {:.1}GB VRAM, have {:.1}GB",
                        model_name, min_vram as f64 / 1e9, total_vram as f64 / 1e9);
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_change(false, false);
                    last_worker_set = new_worker_set;
                    if peer_rx.changed().await.is_err() { break; }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }

                eprintln!("🗳 [{}] Elected as host ({:.1}GB VRAM for {:.1}GB model, {} node(s), split)",
                    model_name, total_vram as f64 / 1e9, model_bytes as f64 / 1e9, model_peers.len() + 1);
            } else {
                eprintln!("🗳 [{}] Running as host ({:.1}GB VRAM for {:.1}GB model, solo)",
                    model_name, my_vram as f64 / 1e9, model_bytes as f64 / 1e9);
            }
            on_change(true, false);

            // In solo mode, pass empty model_peers so start_llama won't use any workers
            let peers_for_launch = if need_split { &model_peers[..] } else { &[] };
            let llama_port = match start_llama(
                &node, &tunnel_mgr, rpc_port, &bin_dir, &model, &model_name,
                peers_for_launch, draft.as_deref(), draft_max, force_split,
            ).await {
                Some(port) => port,
                None => {
                    on_change(true, false);
                    last_worker_set = new_worker_set;
                    let _ = peer_rx.changed().await;
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            node.set_role(NodeRole::Host { http_port: llama_port }).await;
            tunnel_mgr.set_http_port(llama_port);
            currently_host = true;
            last_worker_set = new_worker_set;
            // Re-gossip so peers learn we're the host for this model
            node.regossip().await;
            update_targets(&node, &model_name, InferenceTarget::Local(llama_port), &target_tx).await;
            on_change(true, true);
            eprintln!("✅ [{}] llama-server ready on internal port {llama_port}", model_name);
        } else {
            // We're a worker in split mode. Find who the host is.
            node.set_role(NodeRole::Worker).await;
            currently_host = false;
            last_worker_set = new_worker_set;

            let host_peer = model_peers.iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.vram_bytes, p.id));

            if let Some(host) = host_peer {
                if should_be_host_for_model(host.id, host.vram_bytes, &model_peers) {
                    update_targets(&node, &model_name, InferenceTarget::Remote(host.id), &target_tx).await;
                    eprintln!("📡 [{}] Worker — host is {} (split mode)", model_name, host.id.fmt_short());
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

/// MoE election loop: every node runs its own llama-server with its expert shard.
///
/// Unlike tensor-split mode (one host + RPC workers), MoE mode means:
/// - Every node is independent — no host/worker distinction for this model
/// - Each node runs moe-split locally to produce its shard (cached)
/// - Each node starts its own llama-server with its shard GGUF
/// - The proxy routes sessions to nodes via hash-based affinity
async fn moe_election_loop(
    node: mesh::Node,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    model_name: String,
    moe_cfg: download::MoeConfig,
    target_tx: watch::Sender<ModelTargets>,
    on_change: &mut impl FnMut(bool, bool),
) {
    let mut peer_rx = node.peer_change_rx.clone();
    let mut currently_running = false;
    let mut last_n_nodes: usize = 0;

    loop {
        // Count how many nodes (including us) are serving this model
        let peers = node.peers().await;
        let model_peers: Vec<mesh::PeerInfo> = peers.iter()
            .filter(|p| p.serving.as_deref() == Some(&model_name))
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .cloned()
            .collect();
        let n_nodes = model_peers.len() + 1; // +1 for us

        // Determine our shard index: sort all node IDs, find our position
        let my_id = node.id();
        let mut all_ids: Vec<iroh::EndpointId> = model_peers.iter().map(|p| p.id).collect();
        all_ids.push(my_id);
        all_ids.sort();
        let my_shard_index = all_ids.iter().position(|id| *id == my_id).unwrap_or(0);

        // If nothing changed, skip
        if currently_running && n_nodes == last_n_nodes {
            if peer_rx.changed().await.is_err() { break; }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            continue;
        }

        // Something changed — kill existing llama-server
        if currently_running {
            launch::kill_llama_server().await;
            currently_running = false;
            on_change(false, false);
        }

        last_n_nodes = n_nodes;

        if n_nodes == 1 {
            // Solo: just load the full model, no splitting needed
            eprintln!("🧩 [{}] MoE solo mode — loading full model", model_name);
            on_change(true, false);

            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  Failed to find free port: {e}");
                    if peer_rx.changed().await.is_err() { break; }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            let model_bytes = total_model_bytes(&model);
            match launch::start_llama_server(
                &bin_dir, &model, llama_port, &[], None, None, 0, model_bytes,
            ).await {
                Ok(()) => {
                    node.set_role(NodeRole::Host { http_port: llama_port }).await;
                    currently_running = true;
                    update_targets(&node, &model_name, InferenceTarget::Local(llama_port), &target_tx).await;
                    on_change(true, true);
                    eprintln!("✅ [{}] MoE solo — llama-server ready on port {llama_port}", model_name);
                }
                Err(e) => {
                    eprintln!("  Failed to start llama-server: {e}");
                }
            }
        } else {
            // Multi-node MoE: split and load our shard
            eprintln!("🧩 [{}] MoE split mode — {} nodes, I am shard {}/{}",
                model_name, n_nodes, my_shard_index, n_nodes);
            on_change(true, false);

            // Compute assignments and get our shard
            let assignments = moe::compute_assignments(
                moe_cfg.ranking,
                n_nodes,
                moe_cfg.min_experts_per_node,
            );
            let my_assignment = &assignments[my_shard_index];
            eprintln!("  My experts: {} ({} shared + {} unique)",
                my_assignment.experts.len(), my_assignment.n_shared, my_assignment.n_unique);

            // Ensure our split GGUF exists (cached)
            let shard_path = moe::split_path(&model, n_nodes, my_shard_index);
            if !shard_path.exists() {
                eprintln!("  Splitting GGUF → {} ...", shard_path.display());
                match moe::run_split(&bin_dir, &model, my_assignment, &shard_path) {
                    Ok(()) => {
                        let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                        eprintln!("  Split complete: {:.1} GB", size as f64 / 1e9);
                    }
                    Err(e) => {
                        eprintln!("  ❌ moe-split failed: {e}");
                        if peer_rx.changed().await.is_err() { break; }
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                }
            } else {
                let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                eprintln!("  Using cached shard: {} ({:.1} GB)", shard_path.display(), size as f64 / 1e9);
            }

            // Start llama-server with our shard
            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  Failed to find free port: {e}");
                    if peer_rx.changed().await.is_err() { break; }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            let shard_bytes = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
            match launch::start_llama_server(
                &bin_dir, &shard_path, llama_port, &[], None, None, 0, shard_bytes,
            ).await {
                Ok(()) => {
                    node.set_role(NodeRole::Host { http_port: llama_port }).await;
                    currently_running = true;
                    node.regossip().await;

                    // Build MoE target map: our local port + remote peers
                    let mut moe_state = MoeState::default();
                    for &id in &all_ids {
                        if id == my_id {
                            moe_state.nodes.push(InferenceTarget::MoeLocal(llama_port));
                        } else {
                            moe_state.nodes.push(InferenceTarget::MoeRemote(id));
                        }
                    }

                    // Publish as MoeLocal so proxy knows this is MoE mode
                    let mut targets = ModelTargets::default();
                    targets.targets.insert(model_name.clone(), InferenceTarget::MoeLocal(llama_port));
                    targets.moe = Some(moe_state);
                    target_tx.send_replace(targets);

                    on_change(true, true);
                    eprintln!("✅ [{}] MoE shard {} ready on port {llama_port} ({} experts)",
                        model_name, my_shard_index, my_assignment.experts.len());
                }
                Err(e) => {
                    eprintln!("  ❌ Failed to start llama-server: {e}");
                }
            }
        }

        // Wait for next peer change
        if peer_rx.changed().await.is_err() { break; }
        eprintln!("⚡ [{}] Mesh changed — re-checking MoE deployment...", model_name);
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

    target_tx.send_replace(ModelTargets { targets, moe: None });
}

/// Start llama-server with --rpc pointing at model-group nodes (self + workers).
/// Returns the ephemeral port llama-server is listening on, or None on failure.
async fn start_llama(
    node: &mesh::Node,
    tunnel_mgr: &tunnel::Manager,
    _my_rpc_port: u16,
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

    // Only use workers from our model group, preferring lowest-latency peers.
    // Take just enough to cover the VRAM shortfall, sorted by RTT.
    let worker_ids: Vec<_> = if need_split {
        let mut candidates: Vec<_> = model_peers.iter()
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
            .collect();

        // Sort by RTT ascending (unknown RTT sorts last)
        candidates.sort_by_key(|p| p.rtt_ms.unwrap_or(u32::MAX));

        // Take just enough peers to cover the VRAM gap.
        // When --split is forced, always include at least one worker.
        let mut accumulated_vram = my_vram;
        let mut selected = Vec::new();
        for p in &candidates {
            if accumulated_vram >= min_vram && !(force_split && selected.is_empty()) {
                break; // we have enough VRAM already (but force at least 1 if --split)
            }
            accumulated_vram += p.vram_bytes;
            let rtt_str = p.rtt_ms.map(|r| format!("{}ms", r)).unwrap_or("?ms".to_string());
            eprintln!("  ✓ Adding {} — {:.1}GB VRAM, RTT {rtt_str}",
                p.id.fmt_short(), p.vram_bytes as f64 / 1e9);
            selected.push(p.id);
        }
        if accumulated_vram < min_vram {
            eprintln!("  ⚠ Total VRAM {:.1}GB still short of {:.1}GB — using all {} candidates",
                accumulated_vram as f64 / 1e9, min_vram as f64 / 1e9, candidates.len());
            // Fall back to all candidates if we can't cover it
            selected = candidates.iter().map(|p| p.id).collect();
        }
        selected
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

    // Build --rpc list: only remote workers.
    // The host's own GPU is used directly via Metal — no need to route
    // through the local rpc-server (which would add unnecessary TCP round trips).
    let all_ports = tunnel_mgr.peer_ports_map().await;
    let mut rpc_ports: Vec<u16> = Vec::new();
    for id in &worker_ids {
        if let Some(&port) = all_ports.get(id) {
            rpc_ports.push(port);
        }
    }

    // Calculate tensor split from VRAM.
    // Device order: RPC workers first (matching --rpc order), then Metal (host) last.
    let my_vram_f = my_vram as f64;
    let mut all_vrams: Vec<f64> = Vec::new();
    for id in &worker_ids {
        if let Some(peer) = model_peers.iter().find(|p| p.id == *id) {
            all_vrams.push(if peer.vram_bytes > 0 { peer.vram_bytes as f64 } else { my_vram_f });
        }
    }
    all_vrams.push(my_vram_f); // Metal is last device
    let total: f64 = all_vrams.iter().sum();
    let split = if total > 0.0 && !rpc_ports.is_empty() {
        let s: Vec<String> = all_vrams.iter().map(|v| format!("{:.2}", v / total)).collect();
        let split_str = s.join(",");
        eprintln!("  Tensor split: {split_str} ({} node(s), {:.0}GB total)", rpc_ports.len() + 1, total / 1e9);
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
        bin_dir, model, llama_port, &rpc_ports, split.as_deref(), draft, draft_max, model_bytes,
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
