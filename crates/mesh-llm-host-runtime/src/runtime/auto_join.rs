use super::operational_logging::{DiscoveryOperationalEvent, record_discovery_operational_event};
use super::{
    RunAutoJoinOutcome, RunAutoModelSelection, RunAutoModelSelectionContext, StartupModelPlan,
    attach_local_release_attestation, configure_swarm_capture,
    find_remote_catalog_model_exact_blocking, lan_rediscovery, load_resolved_plugins,
    node_display_name, nostr_rediscovery, owner_runtime_config, parse_size_str, plugin_host_mode,
    record_first_joined_mesh_ts, relay_policy_for_runtime_options, resolve_model,
    run_auto_start_new_mesh, run_passive, runtime_model_capacity_for_ref,
    should_start_lan_rediscovery, start_new_mesh, start_relay_health_monitor_for_discovery_mode,
};
use super::{RuntimeOptions, RuntimeSurface};
use crate::MeshRequirements;
use crate::mesh::{self, NodeRole};
use crate::models;
use crate::network::{discovery as mesh_discovery, lan_bootstrap::effective_quic_bind_ip, nostr};
use crate::plugin;
use anyhow::{Context, Result};
use mesh_llm_events::{ConsoleSessionMode, OutputEvent, RuntimeStatus, emit_event};
use std::path::{Path, PathBuf};

pub(super) async fn maybe_discover_join_candidates(
    options: &mut RuntimeOptions,
    has_startup_models: bool,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
) -> Result<()> {
    let discover_active = options.auto || options.discover.is_some();
    if !discover_active || !options.join.is_empty() {
        return Ok(());
    }

    if let Some(name) = options.discover.as_ref().filter(|name| !name.is_empty())
        && options.mesh_name.is_none()
    {
        options.mesh_name = Some(name.clone());
    }

    let my_vram_gb = mesh::detect_vram_bytes_capped(options.max_vram) as f64 / 1e9;
    let target_name = options.mesh_name.clone();

    match options.mesh_discovery_mode {
        mesh_discovery::MeshDiscoveryMode::Nostr => {
            discover_nostr_join_candidates(
                options,
                has_startup_models,
                auto_join_candidates,
                my_vram_gb,
                target_name.clone(),
            )
            .await?;
        }
        mesh_discovery::MeshDiscoveryMode::Mdns => {
            let _ = emit_event(OutputEvent::DiscoveryStarting {
                source: mesh_discovery::discovery_source_label(
                    options.mesh_discovery_mode,
                    "auto-discovery",
                ),
            });
            let filter = nostr::MeshFilter {
                name: target_name.clone(),
                region: options.region.clone(),
                ..Default::default()
            };
            let candidates = mesh_discovery::discover_lan_join_candidates(
                &filter,
                options.join.first().map(String::as_str),
                std::time::Duration::from_secs(5),
            )
            .await
            .inspect_err(|_| {
                record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
            })?;

            if candidates.is_empty() {
                record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "No joinable LAN meshes found — mDNS requires a supplied invite token"
                        .to_string(),
                    detail: Some("Pass --join <token> or start a new LAN mesh.".to_string()),
                });
                let models = default_models_for_vram_blocking(my_vram_gb).await?;
                if options.client {
                    let _ = emit_event(OutputEvent::Info {
                        message:
                            "No joinable LAN mesh yet — starting client API; pass --join with a LAN invite token to connect"
                                .to_string(),
                        context: None,
                    });
                } else {
                    start_new_mesh(options, &models, my_vram_gb, has_startup_models);
                }
            } else {
                for (token, mesh) in candidates {
                    let _ = emit_event(OutputEvent::MeshFound {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                        peers: mesh.listing.node_count,
                        region: mesh.listing.region.clone(),
                    });
                    auto_join_candidates.push((token, mesh.listing.name));
                }
            }
        }
    }

    Ok(())
}

pub(super) async fn discover_nostr_join_candidates(
    options: &mut RuntimeOptions,
    has_startup_models: bool,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
    my_vram_gb: f64,
    target_name: Option<String>,
) -> Result<()> {
    options.nostr_discovery = true;
    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: mesh_discovery::discovery_source_label(
            options.mesh_discovery_mode,
            "auto-discovery",
        ),
    });

    let relays = nostr_relays(&options.nostr_relay);
    let meshes = discover_nostr_meshes(&relays).await?;
    log_nostr_auto_candidates(&meshes, target_name.as_ref());
    handle_auto_decision(
        options,
        smart_auto_blocking(meshes.clone(), my_vram_gb, target_name).await?,
        auto_join_candidates,
        my_vram_gb,
        has_startup_models,
    )
    .await
}

pub(super) async fn discover_nostr_meshes(relays: &[String]) -> Result<Vec<nostr::DiscoveredMesh>> {
    let filter = nostr::MeshFilter::default();
    match nostr::discover(relays, &filter, None).await {
        Ok(meshes) => Ok(meshes),
        Err(err) => {
            record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Nostr auto-discovery failed".to_string(),
                detail: Some(err.to_string()),
            });
            Err(err)
        }
    }
}

pub(super) fn log_nostr_auto_candidates(
    meshes: &[nostr::DiscoveredMesh],
    target_name: Option<&String>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let last_mesh_id = mesh::load_last_mesh_id();
    let listed: Vec<&nostr::DiscoveredMesh> = if target_name.is_some() {
        meshes.iter().collect()
    } else {
        meshes
            .iter()
            .filter(|m| nostr::is_auto_eligible(m))
            .collect()
    };
    for mesh in &listed {
        let score = nostr::score_mesh(mesh, now, last_mesh_id.as_deref());
        let _ = emit_event(OutputEvent::MeshFound {
            mesh: mesh
                .listing
                .name
                .as_deref()
                .unwrap_or("unnamed")
                .to_string(),
            peers: mesh.listing.node_count,
            region: mesh.listing.region.clone(),
        });
        tracing::debug!(
            "Nostr auto-discovery candidate: {} score={} nodes={} vram_gb={:.0} clients={}",
            mesh.listing.name.as_deref().unwrap_or("unnamed"),
            score,
            mesh.listing.node_count,
            mesh.listing.total_vram_bytes as f64 / 1e9,
            mesh.listing.client_count
        );
    }
}

pub(super) fn initial_console_session_mode(
    explicit_surface: Option<RuntimeSurface>,
) -> ConsoleSessionMode {
    initial_console_session_mode_for_surface(
        explicit_surface,
        mesh_llm_events::current_console_session_mode(),
    )
}

pub fn console_session_mode_for_runtime_surface(
    explicit_surface: Option<RuntimeSurface>,
) -> ConsoleSessionMode {
    initial_console_session_mode(explicit_surface)
}

pub(super) fn initial_console_session_mode_for_surface(
    explicit_surface: Option<RuntimeSurface>,
    current_mode: ConsoleSessionMode,
) -> ConsoleSessionMode {
    match explicit_surface {
        Some(RuntimeSurface::Share | RuntimeSurface::Serve | RuntimeSurface::Client) => {
            current_mode
        }
        _ => ConsoleSessionMode::None,
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
///
/// Parse a catalog size string like "18.3GB" or "491MB" into bytes.
pub(super) async fn smart_auto_blocking(
    meshes: Vec<nostr::DiscoveredMesh>,
    my_vram_gb: f64,
    target_name: Option<String>,
) -> Result<nostr::AutoDecision> {
    tokio::task::spawn_blocking(move || {
        nostr::smart_auto(&meshes, my_vram_gb, target_name.as_deref())
    })
    .await
    .context("join smart auto task")
}

pub(super) async fn handle_auto_decision(
    options: &mut RuntimeOptions,
    decision: nostr::AutoDecision,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
    my_vram_gb: f64,
    has_startup_models: bool,
) -> Result<()> {
    match decision {
        nostr::AutoDecision::Join { candidates } => {
            if options.client {
                // Clients skip health probe — joining itself is the test.
                // Queue all candidates so we can fall back if the top one is unreachable.
                let (_, mesh) = &candidates[0];
                if options.mesh_name.is_none()
                    && let Some(ref name) = mesh.listing.name
                {
                    options.mesh_name = Some(name.clone());
                }
                for (token, _) in &candidates {
                    options.join.push(token.clone());
                }
            } else {
                // GPU nodes try each candidate directly. The real join path can use relays,
                // so a separate local probe would reject reachable meshes behind firewalls.
                let mut joined = false;
                for (token, mesh) in &candidates {
                    let _ = emit_event(OutputEvent::MeshFound {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                        peers: mesh.listing.node_count,
                        region: mesh.listing.region.clone(),
                    });
                    auto_join_candidates.push((token.clone(), mesh.listing.name.clone()));
                    joined = true;
                }
                if !joined {
                    record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
                    let _ = emit_event(OutputEvent::DiscoveryFailed {
                        message: "No meshes found — starting new".to_string(),
                        detail: None,
                    });
                    let models = default_models_for_vram_blocking(my_vram_gb).await?;
                    start_new_mesh(options, &models, my_vram_gb, has_startup_models);
                }
            }
        }
        nostr::AutoDecision::StartNew { models } => {
            if options.client {
                // Client mode should still expose its local proxy and management API while
                // it waits for a mesh to appear.
                let _ = emit_event(OutputEvent::Info {
                    message: "No meshes found yet — starting client API while discovery continues"
                        .to_string(),
                    context: None,
                });
            } else {
                start_new_mesh(options, &models, my_vram_gb, has_startup_models);
            }
        }
    }
    Ok(())
}

pub(super) async fn default_models_for_vram_blocking(my_vram_gb: f64) -> Result<Vec<String>> {
    tokio::task::spawn_blocking(move || nostr::default_models_for_vram(my_vram_gb))
        .await
        .context("join default model selection task")
}

#[expect(
    dead_code,
    reason = "used by the retained legacy advertised-model selection lane and its focused tests"
)]
pub(super) async fn auto_model_pack_blocking(my_vram_gb: f64) -> Result<Vec<String>> {
    tokio::task::spawn_blocking(move || nostr::auto_model_pack(my_vram_gb))
        .await
        .context("join auto model pack task")
}

/// Pick which model this node should serve, based on demand signals.
///
/// Priority:
/// 1. Unserved models with active demand that we have on disk (hottest first)
/// 2. Underserved models with demand that we have on disk
/// 3. Unserved models with demand that we can download from catalog
/// 4. Standby if everything is covered
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the retained legacy automatic assignment lane and its focused tests"
    )
)]
pub(super) async fn pick_model_assignment(
    node: &mesh::Node,
    local_models: &[String],
) -> Option<String> {
    let peers = node.peers().await;

    // Get active demand — the unified "what does the mesh want?"
    let demand = node.active_demand().await;

    if demand.is_empty() {
        // No API requests yet — log what the mesh is serving for visibility
        let served: Vec<String> = peers.iter().flat_map(|p| p.routable_models()).collect();
        if !served.is_empty() {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "No demand yet — mesh is serving {:?}, staying standby until needed",
                    served
                ),
                context: None,
            });
        } else {
            let _ = emit_event(OutputEvent::Info {
                message: "No demand signals — no models requested".to_string(),
                context: None,
            });
        }
        return None;
    }

    let _ = emit_event(OutputEvent::Info {
        message: format!("Active demand: {:?}", demand.keys().collect::<Vec<_>>()),
        context: None,
    });

    // Count how many nodes are serving each model
    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    /// Check if a model fits in our VRAM. Returns false and logs if it doesn't.
    fn model_fits(model: &str, my_vram: u64) -> bool {
        let capacity = runtime_model_capacity_for_ref(model, my_vram);
        if !capacity.fits {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Skipping {} — needs {:.1}GB, we have {:.1}GB",
                    model,
                    capacity.required_bytes as f64 / 1e9,
                    my_vram as f64 / 1e9
                ),
                context: None,
            });
            return false;
        }
        true
    }

    // Sort demand entries by request_count descending (hottest first)
    let mut demand_sorted: Vec<(String, mesh::ModelDemand)> = demand.into_iter().collect();
    demand_sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1.request_count));

    // Priority 1: Unserved models on disk, ordered by demand
    let mut candidates: Vec<String> = Vec::new();
    for (m, _d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) == 0
            && local_models.contains(m)
            && model_fits(m, my_vram)
        {
            candidates.push(m.clone());
        }
    }

    if !candidates.is_empty() {
        // If multiple, pick deterministically so concurrent joiners spread out
        if candidates.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % candidates.len();
            let pick = &candidates[idx];
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Assigned to serve {} (unserved, on disk, {} candidates, by demand)",
                    pick,
                    candidates.len()
                ),
                context: None,
            });
            return Some(pick.clone());
        }
        let pick = &candidates[0];
        let _ = emit_event(OutputEvent::Info {
            message: format!("Assigned to serve {} (unserved, on disk, by demand)", pick),
            context: None,
        });
        return Some(pick.clone());
    }

    // Priority 2: Underserved models on disk (fewer servers than others)
    let max_count = serving_count.values().copied().max().unwrap_or(0);
    let mut underserved: Vec<(String, usize, u64)> = Vec::new(); // (model, servers, demand)
    for (m, d) in &demand_sorted {
        let count = serving_count.get(m).copied().unwrap_or(0);
        if count < max_count && local_models.contains(m) && model_fits(m, my_vram) {
            underserved.push((m.clone(), count, d.request_count));
        }
    }
    if !underserved.is_empty() {
        // Pick the least-served, breaking ties by highest demand
        underserved.sort_by_key(|(_, count, demand)| (*count, std::cmp::Reverse(*demand)));
        let (pick, count, _) = &underserved[0];
        let max_model = serving_count
            .iter()
            .max_by_key(|&(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Assigned to serve {} ({} servers vs {} has {}) — rebalancing",
                pick, count, max_model, max_count
            ),
            context: None,
        });
        return Some(pick.clone());
    }

    // Priority 3: Unserved models we can download from catalog
    let mut downloadable: Vec<(String, u64)> = Vec::new(); // (model, demand)
    for (m, d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) > 0 {
            continue;
        }
        if let Some(cat) = find_remote_catalog_model_exact_blocking(m.clone()).await {
            let Some(size_label) = cat.size.as_deref() else {
                continue;
            };
            let Some(needed) = catalog_model_required_bytes(size_label) else {
                let _ = emit_event(OutputEvent::Info {
                    message: format!(
                        "Skipping {} — catalog size {:?} is missing or invalid",
                        m, size_label
                    ),
                    context: None,
                });
                continue;
            };
            if needed <= my_vram {
                downloadable.push((m.clone(), d.request_count));
            } else {
                let _ = emit_event(OutputEvent::Info {
                    message: format!(
                        "Skipping {} — needs {:.1}GB, we have {:.1}GB",
                        m,
                        needed as f64 / 1e9,
                        my_vram as f64 / 1e9
                    ),
                    context: None,
                });
            }
        }
    }
    if !downloadable.is_empty() {
        // Pick hottest downloadable, with node-ID hash for tie-breaking
        if downloadable.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % downloadable.len();
            let (pick, _) = &downloadable[idx];
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Assigned to serve {} (unserved, will download, by demand)",
                    pick
                ),
                context: None,
            });
            return Some(pick.clone());
        }
        let (pick, _) = &downloadable[0];
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Assigned to serve {} (unserved, will download, by demand)",
                pick
            ),
            context: None,
        });
        return Some(pick.clone());
    }

    // Everything with demand is covered
    let all_covered = demand_sorted
        .iter()
        .all(|(m, _)| serving_count.get(m).copied().unwrap_or(0) > 0);
    if all_covered {
        let _ = emit_event(OutputEvent::Info {
            message: "All demanded models are covered — staying on standby".to_string(),
            context: None,
        });
    }

    None
}

/// Pick a model assignment only when this node's mesh role can serve models.
pub(super) async fn pick_model_assignment_for_role(
    node: &mesh::Node,
    local_models: &[String],
) -> Option<String> {
    if matches!(node.role().await, NodeRole::Client) {
        None
    } else {
        pick_model_assignment(node, local_models).await
    }
}

pub(super) fn catalog_model_required_bytes(size_label: &str) -> Option<u64> {
    parse_size_str(size_label)
        .filter(|size| *size > 0)
        .map(|size| (size as f64 * 1.1) as u64)
}

pub(super) async fn pick_run_auto_model_assignment(
    is_client: bool,
    node: &mesh::Node,
    local_models: &[String],
) -> Option<String> {
    if is_client {
        None
    } else {
        pick_model_assignment_for_role(node, local_models).await
    }
}

/// Check if a standby node should promote to serve a model.
/// Uses demand signals — promotes for unserved models with active demand,
/// or for demand-based rebalancing when one model is much hotter than others.
///
/// Rebalancing uses `last_active` to gate on recency (only models active within
/// the last 60 minutes are considered), then `request_count / servers` for
/// relative hotness among those recent models.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by passive-publication compatibility code and focused promotion tests"
    )
)]
pub(super) async fn check_unserved_model(
    node: &mesh::Node,
    local_models: &[String],
) -> Option<String> {
    let peers = node.peers().await;
    let demand = node.active_demand().await;

    if demand.is_empty() {
        return None;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    // Only consider models with recent activity (last 60 minutes).
    // This prevents stale cumulative request_count from triggering promotions
    // for models that were popular hours ago but idle now.
    const RECENT_SECS: u64 = 3600;

    // Priority 1: promote for models with active demand and ZERO servers
    // Sort by demand (hottest first)
    let mut unserved: Vec<(String, u64)> = Vec::new();
    for (m, d) in &demand {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            if !runtime_model_capacity_for_ref(m, my_vram).fits {
                continue;
            }
            unserved.push((m.clone(), d.request_count));
        }
    }
    if !unserved.is_empty() {
        unserved.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        return Some(unserved[0].0.clone());
    }

    // Priority 2: demand-based rebalancing.
    // Only consider models with recent activity, then use request_count / servers
    // for relative hotness. Promote if one model is significantly hotter than others.
    let mut ratios: Vec<(String, f64)> = Vec::new();
    for (m, d) in &demand {
        if now.saturating_sub(d.last_active) > RECENT_SECS {
            continue;
        }
        let servers = serving_count.get(m).copied().unwrap_or(0) as f64;
        if servers > 0.0 && d.request_count > 0 && local_models.contains(m) {
            if !runtime_model_capacity_for_ref(m, my_vram).fits {
                continue;
            }
            ratios.push((m.clone(), d.request_count as f64 / servers));
        }
    }

    if !ratios.is_empty() {
        ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (hottest_model, hottest_ratio) = &ratios[0];
        let coldest_ratio = if ratios.len() >= 2 {
            ratios[ratios.len() - 1].1
        } else {
            0.0
        };
        let should_promote = if ratios.len() >= 2 {
            *hottest_ratio >= coldest_ratio * 3.0 && *hottest_ratio >= 10.0
        } else {
            *hottest_ratio >= 10.0
        };

        if should_promote {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Promoting to serve {} — demand {:.0} req/server (coldest: {:.0})",
                    hottest_model, hottest_ratio, coldest_ratio
                ),
                context: None,
            });
            return Some(hottest_model.clone());
        }
    }

    None
}

#[expect(
    dead_code,
    reason = "retained runtime-side MCP join compatibility entrypoint"
)]
pub(super) async fn join_mesh_for_mcp(options: &RuntimeOptions, node: &mesh::Node) -> Result<()> {
    if !options.join.is_empty() {
        return join_mcp_with_tokens(&options.join, node).await;
    }

    if options.auto || options.discover.is_some() {
        if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Mdns {
            return join_mcp_via_lan_discovery(options, node).await;
        }

        return join_mcp_via_nostr_discovery(options, node).await;
    }

    Ok(())
}

#[expect(
    dead_code,
    reason = "the shipped command lane owns MCP joins while the runtime compatibility path remains available"
)]
pub(super) async fn join_mcp_with_tokens(tokens: &[String], node: &mesh::Node) -> Result<()> {
    for token in tokens {
        match node.join_with_retry(token).await {
            Ok(()) => {
                if node.mesh_id().await.is_some() {
                    record_first_joined_mesh_ts(node).await;
                }
                let _ = emit_event(OutputEvent::Info {
                    message: "Connected to bootstrap peer; awaiting mesh admission".to_string(),
                    context: None,
                });
                return Ok(());
            }
            Err(err) => tracing::warn!("Failed to join via token: {err}"),
        }
    }
    anyhow::bail!("Failed to join any peer for MCP mode");
}

#[expect(
    dead_code,
    reason = "the shipped command lane owns MCP joins while the runtime compatibility path remains available"
)]
pub(super) async fn join_mcp_via_lan_discovery(
    options: &RuntimeOptions,
    node: &mesh::Node,
) -> Result<()> {
    let filter = nostr::MeshFilter {
        region: options.region.clone(),
        name: options
            .discover
            .as_deref()
            .filter(|s| !s.is_empty())
            .or(options.mesh_name.as_deref())
            .map(str::to_owned),
        ..Default::default()
    };
    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: mesh_discovery::discovery_source_label(options.mesh_discovery_mode, "discovery"),
    });
    let candidates = mesh_discovery::discover_lan_join_candidates(
        &filter,
        options.join.first().map(String::as_str),
        std::time::Duration::from_secs(5),
    )
    .await?;
    if candidates.is_empty() {
        let _ = emit_event(OutputEvent::DiscoveryFailed {
            message: "No joinable LAN mesh found for MCP mode".to_string(),
            detail: Some("Pass --join or start a LAN mesh first.".to_string()),
        });
        anyhow::bail!(
            "No joinable LAN mesh found for MCP mode. Pass --join or start a LAN mesh first."
        );
    }

    let mut last_err = None;
    for (token, mesh) in candidates {
        let label = mesh
            .listing
            .name
            .as_deref()
            .unwrap_or("unnamed")
            .to_string();
        let _ = emit_event(OutputEvent::MeshFound {
            mesh: label.clone(),
            peers: mesh.listing.node_count,
            region: mesh.listing.region.clone(),
        });
        match node.join_with_retry(&token).await {
            Ok(()) => {
                if node.mesh_id().await.is_some() {
                    record_first_joined_mesh_ts(node).await;
                }
                let _ = emit_event(OutputEvent::DiscoveryJoined { mesh: label });
                return Ok(());
            }
            Err(err) => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: format!("Failed to join LAN mesh {label}"),
                    detail: Some(err.to_string()),
                });
                last_err = Some(err);
            }
        }
    }

    if let Some(err) = last_err {
        return Err(err);
    }
    Ok(())
}

#[expect(
    dead_code,
    reason = "the shipped command lane owns MCP joins while the runtime compatibility path remains available"
)]
pub(super) async fn join_mcp_via_nostr_discovery(
    options: &RuntimeOptions,
    node: &mesh::Node,
) -> Result<()> {
    let relays = nostr_relays(&options.nostr_relay);
    let filter = nostr::MeshFilter {
        region: options.region.clone(),
        ..Default::default()
    };
    let target_name = options
        .discover
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(options.mesh_name.as_deref())
        .map(str::to_owned);
    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: "Nostr discovery".to_string(),
    });
    let meshes = match nostr::discover(&relays, &filter, None).await {
        Ok(meshes) => meshes,
        Err(err) => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Nostr discovery failed".to_string(),
                detail: Some(err.to_string()),
            });
            return Err(err);
        }
    };

    match smart_auto_blocking(meshes, 0.0, target_name).await? {
        nostr::AutoDecision::Join { candidates } => {
            let mut last_err: Option<anyhow::Error> = None;
            for (token, mesh) in &candidates {
                let label = mesh
                    .listing
                    .name
                    .as_deref()
                    .unwrap_or("unnamed")
                    .to_string();
                let _ = emit_event(OutputEvent::MeshFound {
                    mesh: label.clone(),
                    peers: mesh.listing.node_count,
                    region: mesh.listing.region.clone(),
                });
                match node.join_with_retry(token).await {
                    Ok(()) => {
                        if node.mesh_id().await.is_some() {
                            record_first_joined_mesh_ts(node).await;
                        }
                        let _ = emit_event(OutputEvent::DiscoveryJoined { mesh: label });
                        last_err = None;
                        break;
                    }
                    Err(err) => {
                        let _ = emit_event(OutputEvent::DiscoveryFailed {
                            message: format!("Failed to join mesh {label}"),
                            detail: Some(err.to_string()),
                        });
                        tracing::warn!("Failed to join mesh candidate: {err}");
                        last_err = Some(err);
                    }
                }
            }
            if let Some(err) = last_err {
                return Err(err);
            }
            Ok(())
        }
        nostr::AutoDecision::StartNew { .. } => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "No mesh found for MCP mode".to_string(),
                detail: Some("Pass --join or start a mesh first.".to_string()),
            });
            anyhow::bail!("No mesh found for MCP mode. Pass --join or start a mesh first.");
        }
    }
}

#[expect(
    dead_code,
    reason = "retained as the runtime-side MCP compatibility entrypoint while command dispatch lives in mesh-llm-commands"
)]
pub(crate) async fn run_plugin_mcp(options: &RuntimeOptions) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(options)?;
    let config = plugin::load_config(options.config.as_deref())?;
    let owner_config = owner_runtime_config(options, &config)?;
    let swarm_capture = configure_swarm_capture(options)?;
    let relay_auths: std::collections::HashMap<String, String> =
        options.relay_auth.iter().cloned().collect();
    let (node, _channels) = mesh::Node::start(
        NodeRole::Client,
        mesh::RelayConfig {
            urls: &options.relay,
            auths: &relay_auths,
            policy: relay_policy_for_runtime_options(options),
        },
        mesh::QuicBindSelection {
            ip: effective_quic_bind_ip(options),
            port: options.bind_port,
        },
        Some(0.0),
        !options.no_enumerate_host,
        Some(owner_config),
        options.config.as_deref(),
        MeshRequirements::unrestricted(),
    )
    .await?;
    node.set_swarm_capture_recorder(swarm_capture);
    attach_local_release_attestation(&node).await?;
    node.start_accepting();
    node.set_display_name(node_display_name(options, &node))
        .await;
    node.start_heartbeat();
    node.start_rtt_refresh();
    // iroh owns NAT traversal and relay-to-direct migration; modern nodes do
    // not start the legacy application-level reverse-dial sender.
    start_relay_health_monitor_for_discovery_mode(&node, options.mesh_discovery_mode);
    join_mesh_for_mcp(options, &node).await?;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(options), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    if plugin_manager.list().await.is_empty() {
        tracing::warn!("No plugins are enabled for MCP exposure");
    }

    plugin::mcp::run_mcp_server(plugin_manager).await
}

pub use super::discovery::nostr_relays;

pub(super) async fn attempt_run_auto_join(
    node: &mesh::Node,
    join_attempts: &[(String, Option<String>)],
    prefer_fast_probe: bool,
) -> RunAutoJoinOutcome {
    record_discovery_operational_event(DiscoveryOperationalEvent::JoinStarted);
    let mut outcome = RunAutoJoinOutcome {
        joined: false,
        last_join_error: None,
        successful_join: None,
    };

    if prefer_fast_probe {
        match attempt_fast_auto_join(node, join_attempts).await {
            Some(Ok(successful_join)) => {
                return build_successful_run_auto_join(node, successful_join).await;
            }
            Some(Err(err)) => outcome.last_join_error = Some(format!("{err:#}")),
            None => {}
        }
    }

    for (token, mesh_name) in join_attempts {
        match node.join_with_retry(token).await {
            Ok(()) => {
                if node.mesh_id().await.is_some() {
                    record_first_joined_mesh_ts(node).await;
                }
                let _ = emit_event(OutputEvent::Info {
                    message: "Connected to bootstrap peer; awaiting mesh admission".to_string(),
                    context: None,
                });
                let _ = emit_event(OutputEvent::DiscoveryJoined {
                    mesh: successful_join_mesh_label(mesh_name.as_deref()),
                });
                record_discovery_operational_event(DiscoveryOperationalEvent::JoinSucceeded);
                outcome.joined = true;
                outcome.successful_join = Some((token.clone(), mesh_name.clone()));
                break;
            }
            Err(err) => {
                tracing::warn!("Failed to join via token: {err}");
                outcome.last_join_error = Some(format!("{err:#}"));
            }
        }
    }

    if !outcome.joined {
        record_discovery_operational_event(DiscoveryOperationalEvent::JoinFailed);
    }

    outcome
}

pub(super) async fn attempt_fast_auto_join(
    node: &mesh::Node,
    join_attempts: &[(String, Option<String>)],
) -> Option<Result<(String, Option<String>)>> {
    match node.join_first_responsive_candidate(join_attempts).await {
        Ok(Some(successful_join)) => Some(Ok(successful_join)),
        Ok(None) => None,
        Err(err) => {
            tracing::warn!("Fast auto-join probe failed: {err:#}");
            Some(Err(err))
        }
    }
}

pub(super) async fn build_successful_run_auto_join(
    node: &mesh::Node,
    successful_join: (String, Option<String>),
) -> RunAutoJoinOutcome {
    if node.mesh_id().await.is_some() {
        record_first_joined_mesh_ts(node).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: "Connected to bootstrap peer; awaiting mesh admission".to_string(),
        context: None,
    });
    let _ = emit_event(OutputEvent::DiscoveryJoined {
        mesh: successful_join_mesh_label(successful_join.1.as_deref()),
    });
    record_discovery_operational_event(DiscoveryOperationalEvent::JoinSucceeded);
    RunAutoJoinOutcome {
        joined: true,
        last_join_error: None,
        successful_join: Some(successful_join),
    }
}

pub(super) fn successful_join_mesh_label(mesh_name: Option<&str>) -> String {
    mesh_name.unwrap_or("unnamed").to_string()
}

pub(super) fn update_cli_with_successful_run_auto_join(
    options: &mut RuntimeOptions,
    successful_join: Option<(String, Option<String>)>,
) {
    if !options.join.is_empty() {
        return;
    }

    options.join.clear();
    if let Some((token, mesh_name)) = successful_join {
        options.join.push(token);
        if options.mesh_name.is_none()
            && let Some(name) = mesh_name
        {
            options.mesh_name = Some(name);
        }
    }
}

pub(super) async fn run_auto_join_existing_mesh(
    options: &mut RuntimeOptions,
    node: &mesh::Node,
    auto_join_candidates: &[(String, Option<String>)],
) {
    let join_attempts: Vec<(String, Option<String>)> = if !options.join.is_empty() {
        options
            .join
            .iter()
            .cloned()
            .map(|token| (token, None))
            .collect()
    } else {
        auto_join_candidates.to_vec()
    };
    let prefer_fast_probe = should_prefer_fast_auto_join(options, auto_join_candidates);
    let outcome = attempt_run_auto_join(node, &join_attempts, prefer_fast_probe).await;
    update_cli_with_successful_run_auto_join(options, outcome.successful_join);

    if !outcome.joined {
        let reason = outcome.last_join_error.as_deref().unwrap_or("unknown");
        let _ = emit_event(OutputEvent::Warning {
            message: format!("Failed to join any peer — running standalone ({reason})"),
            context: None,
        });
    }

    spawn_run_auto_post_join_tasks(options, node).await;
}

pub(super) fn should_prefer_fast_auto_join(
    options: &RuntimeOptions,
    auto_join_candidates: &[(String, Option<String>)],
) -> bool {
    options.client || (options.join.is_empty() && !auto_join_candidates.is_empty())
}

pub(super) async fn spawn_run_auto_post_join_tasks(options: &RuntimeOptions, node: &mesh::Node) {
    let save_node = node.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        if let Some(id) = save_node.mesh_id().await {
            record_first_joined_mesh_ts(&save_node).await;
            if let Err(error) = mesh::save_last_mesh_id(&id) {
                tracing::warn!(error = %error, "failed to save last mesh ID");
            }
            tracing::info!("Mesh ID: {id}");
        }
    });

    let mesh_id = node
        .mesh_id()
        .await
        .unwrap_or_else(|| "pending".to_string());
    let _ = emit_event(OutputEvent::InviteToken {
        token: node.invite_token().await,
        mesh_id,
        mesh_name: options.mesh_name.clone(),
    });

    let rejoin_node = node.clone();
    let rejoin_tokens: Vec<String> = options.join.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            for t in &rejoin_tokens {
                if let Err(e) = rejoin_node.join(t).await {
                    tracing::debug!("Rejoin failed: {e}");
                }
            }
        }
    });

    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
    {
        let rediscover_node = node.clone();
        let rediscover_relays = nostr_relays(&options.nostr_relay);
        let rediscover_relay_urls = options.relay.clone();
        let rediscover_mesh_name = options.mesh_name.clone();
        tokio::spawn(Box::pin(nostr_rediscovery(
            rediscover_node,
            rediscover_relays,
            rediscover_relay_urls,
            rediscover_mesh_name,
        )));
    } else if should_start_lan_rediscovery(options.mesh_discovery_mode, &options.join) {
        let rediscover_node = node.clone();
        let rediscover_join_tokens = options.join.clone();
        let rediscover_mesh_name = options.mesh_name.clone();
        let rediscover_region = options.region.clone();
        tokio::spawn(Box::pin(lan_rediscovery(
            rediscover_node,
            rediscover_join_tokens,
            rediscover_mesh_name,
            rediscover_region,
        )));
    }
}

#[expect(
    dead_code,
    reason = "the daemon startup path supersedes legacy automatic model selection while focused tests retain coverage"
)]
pub(super) async fn select_run_auto_model_path(
    ctx: &mut RunAutoModelSelectionContext<'_>,
) -> Result<RunAutoModelSelection> {
    let primary_startup_model = ctx.startup_models.first().cloned();
    if let Some(primary) = primary_startup_model.as_ref() {
        return Ok(RunAutoModelSelection::Model(primary.resolved_path.clone()));
    }

    let _ = emit_event(OutputEvent::WaitingForPeers {
        detail: Some("No --model specified, checking local models against mesh...".to_string()),
    });
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let assignment =
        pick_run_auto_model_assignment(ctx.is_client, ctx.node, ctx.local_models).await;
    let assignment = if assignment.is_none()
        && (ctx.options.auto || ctx.options.discover.is_some())
        && !ctx.is_client
    {
        let pack = auto_model_pack_blocking(ctx.node.vram_bytes() as f64 / 1e9).await?;
        if !pack.is_empty() {
            Some(pack[0].clone())
        } else {
            assignment
        }
    } else {
        assignment
    };

    let Some(model_name) = assignment else {
        let passive_api_listener = match ctx.bootstrap_listener_tx.take() {
            Some(tx) => {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                if tx.send(resp_tx).await.is_ok() {
                    Some(
                        resp_rx
                            .await
                            .context("bootstrap API listener handoff was cancelled")?,
                    )
                } else {
                    None
                }
            }
            _ => None,
        };
        if ctx.is_client {
            let _ = emit_event(OutputEvent::PassiveMode {
                role: "client".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: None,
                models_on_disk: None,
                detail: Some("Running as client — proxying requests to mesh".to_string()),
            });
        } else {
            let _ = emit_event(OutputEvent::PassiveMode {
                role: "standby".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: Some(ctx.node.vram_bytes() as f64 / 1e9),
                models_on_disk: Some(ctx.local_models.to_vec()),
                detail: Some(
                    "No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed."
                        .to_string(),
                ),
            });
        }
        return match run_passive(
            ctx.options,
            ctx.node.clone(),
            ctx.is_client,
            ctx.plugin_manager.clone(),
            passive_api_listener,
            ctx.embedded_control_rx.take(),
        )
        .await?
        {
            Some(model_name) => Ok(RunAutoModelSelection::Model(models::find_model_path(
                &model_name,
            ))),
            None => Ok(RunAutoModelSelection::Shutdown),
        };
    };

    let _ = emit_event(OutputEvent::HostElected {
        model: model_name.clone(),
        host: ctx.node.id().fmt_short().to_string(),
        role: Some("host".to_string()),
        capacity_gb: Some(ctx.node.vram_bytes() as f64 / 1e9),
    });
    let model_path = models::find_model_path(&model_name);
    if model_path.exists() {
        return Ok(RunAutoModelSelection::Model(model_path));
    }
    if let Some(cat) = find_remote_catalog_model_exact_blocking(model_name.clone()).await {
        let _ = emit_event(OutputEvent::Info {
            message: format!("Downloading {model_name} for mesh..."),
            context: None,
        });
        let model_ref = models::remote_catalog_model_ref(&cat);
        return Ok(RunAutoModelSelection::Model(
            resolve_model(&PathBuf::from(model_ref)).await?,
        ));
    }
    Ok(RunAutoModelSelection::Model(model_path))
}

pub(super) async fn run_auto_join_mesh_phase(
    options: &mut RuntimeOptions,
    node: &mesh::Node,
    auto_join_candidates: &[(String, Option<String>)],
) -> Result<()> {
    if !options.join.is_empty() || !auto_join_candidates.is_empty() {
        record_discovery_operational_event(DiscoveryOperationalEvent::DecisionJoin);
        run_auto_join_existing_mesh(options, node, auto_join_candidates).await;
    } else {
        record_discovery_operational_event(DiscoveryOperationalEvent::DecisionStartNew);
        run_auto_start_new_mesh(options, node).await?;
    }
    Ok(())
}

pub(super) fn run_auto_model_identity(
    primary_startup_model: Option<&StartupModelPlan>,
    model: &Path,
) -> (String, String) {
    let model_name = primary_startup_model
        .map(|startup_model| startup_model.declared_ref.clone())
        .unwrap_or_else(|| models::model_ref_for_path(model));
    let model_source = primary_startup_model
        .map(|startup_model| startup_model.declared_ref.clone())
        .unwrap_or_else(|| model_name.clone());
    (model_name, model_source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_join_mesh_label_preserves_named_and_unnamed_meshes() {
        assert_eq!(successful_join_mesh_label(Some("mesh-llm")), "mesh-llm");
        assert_eq!(successful_join_mesh_label(None), "unnamed");
    }
}
