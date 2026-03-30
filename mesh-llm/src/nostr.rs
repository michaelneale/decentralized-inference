//! Publish and discover mesh-llm meshes via Nostr relays.
//!
//! A running mesh publishes a replaceable event (kind 31990, d-tag "mesh-llm")
//! containing its invite token, served models, VRAM, node count, etc.
//! Other nodes can discover available meshes and auto-join.

use anyhow::Result;
use nostr_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// NIP-89 "Application Handler" kind — used for service advertisements.
pub const MESH_SERVICE_KIND: u16 = 31990;

/// Default public relays.
pub const DEFAULT_RELAYS: &[&str] = &[
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
];

/// What we publish about a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshListing {
    /// Base64 invite token (others use this to --join)
    pub invite_token: String,
    /// Models currently loaded and serving inference
    pub serving: Vec<String>,
    /// Models the mesh wants but nobody is serving yet (need more GPUs)
    #[serde(default)]
    pub wanted: Vec<String>,
    /// Models on disk across the mesh (could be loaded if a GPU becomes free)
    #[serde(default)]
    pub on_disk: Vec<String>,
    /// Total VRAM across all GPU nodes (bytes)
    pub total_vram_bytes: u64,
    /// Number of GPU nodes in the mesh
    pub node_count: usize,
    /// Number of connected clients (API-only nodes)
    #[serde(default)]
    pub client_count: usize,
    /// Maximum clients this mesh accepts (0 = unlimited)
    #[serde(default)]
    pub max_clients: usize,
    /// Optional human-readable name for the mesh
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional geographic region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Stable mesh identity — all nodes in the same mesh share this ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_id: Option<String>,
}

/// Discovered mesh from Nostr.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscoveredMesh {
    pub listing: MeshListing,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

impl std::fmt::Display for DiscoveredMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vram_gb = self.listing.total_vram_bytes as f64 / 1e9;
        let models = if self.listing.serving.is_empty() {
            "(no models loaded)".to_string()
        } else {
            self.listing.serving.join(", ")
        };
        write!(
            f,
            "{}  {} node(s), {:.0}GB VRAM  serving: {}",
            self.listing.name.as_deref().unwrap_or("(unnamed)"),
            self.listing.node_count,
            vram_gb,
            models,
        )?;
        if let Some(ref region) = self.listing.region {
            write!(f, "  region: {}", region)?;
        }
        if !self.listing.wanted.is_empty() {
            write!(f, "  wanted: {}", self.listing.wanted.join(", "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Keys — stored in ~/.mesh-llm/nostr.nsec
// ---------------------------------------------------------------------------

fn nostr_key_path() -> Result<std::path::PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".mesh-llm").join("nostr.nsec"))
}

/// Load or generate a Nostr keypair for publishing.
pub fn load_or_create_keys() -> Result<Keys> {
    let path = nostr_key_path()?;
    if path.exists() {
        let nsec = std::fs::read_to_string(&path)?;
        let sk = SecretKey::from_bech32(nsec.trim())?;
        Ok(Keys::new(sk))
    } else {
        let keys = Keys::generate();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let nsec = keys.secret_key().to_bech32()?;
        std::fs::write(&path, &nsec)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&path, perms)?;
        }
        tracing::info!("Generated new Nostr key, saved to {}", path.display());
        Ok(keys)
    }
}

/// Delete the Nostr key (forces a new identity on next publish).
pub fn rotate_keys() -> Result<()> {
    let path = nostr_key_path()?;
    if path.exists() {
        std::fs::remove_file(&path)?;
        eprintln!(
            "🔑 Deleted {}. A new key will be generated on next --publish.",
            path.display()
        );
    } else {
        eprintln!("No key to rotate (none exists yet).");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Publisher — background task that keeps the listing fresh
// ---------------------------------------------------------------------------

pub struct Publisher {
    client: Client,
    keys: Keys,
}

impl Publisher {
    pub async fn new(keys: Keys, relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::new(keys.clone());
        for relay in relays {
            client.add_relay(relay).await?;
        }
        client.connect().await;
        Ok(Self { client, keys })
    }

    pub fn npub(&self) -> String {
        self.keys.public_key().to_bech32().unwrap_or_default()
    }

    /// Publish (or replace) the mesh listing. Uses a replaceable event
    /// (kind 31990 + d-tag) so each publisher has exactly one listing.
    pub async fn publish(&self, listing: &MeshListing, ttl_secs: u64) -> Result<()> {
        let expiration = Timestamp::now().as_secs() + ttl_secs;
        let content = serde_json::to_string(listing)?;

        let tags = vec![
            Tag::custom(TagKind::Custom("d".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(TagKind::Custom("k".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(
                TagKind::Custom("expiration".into()),
                vec![expiration.to_string()],
            ),
        ];

        let builder = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), content).tags(tags);
        self.client.send_event_builder(builder).await?;
        Ok(())
    }

    /// Delete our listing (e.g. on shutdown).
    pub async fn unpublish(&self) -> Result<()> {
        // Fetch our own events
        let filter = Filter::new()
            .kind(Kind::Custom(MESH_SERVICE_KIND))
            .author(self.keys.public_key())
            .limit(10);
        let events = self
            .client
            .fetch_events(filter, Duration::from_secs(5))
            .await?;
        for event in events.iter() {
            let request = EventDeletionRequest::new().id(event.id);
            let _ = self
                .client
                .send_event_builder(EventBuilder::delete(request))
                .await;
        }
        Ok(())
    }
}

/// Background publish loop. Republishes every `interval` seconds using
/// fresh data from the mesh node.
///
/// If `max_clients` is set, delists when that many clients are connected
/// and re-publishes when clients drop below the cap.
pub async fn publish_loop(
    node: crate::mesh::Node,
    keys: Keys,
    relays: Vec<String>,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
    interval_secs: u64,
) {
    let publisher = match Publisher::new(keys.clone(), &relays).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create Nostr publisher: {e}");
            return;
        }
    };

    let npub = publisher.npub();
    if let Some(cap) = max_clients {
        eprintln!("   Will delist when {} clients connected", cap);
    }

    // Wait for llama-server to be ready before first publish (up to 60s).
    for _ in 0..120 {
        if node.is_llama_ready().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    eprintln!(
        "📡 Publishing mesh to Nostr (npub: {}...{})",
        &npub[..12],
        &npub[npub.len() - 8..]
    );

    let mut delisted = false;

    loop {
        let invite_token = node.invite_token();
        let peers = node.peers().await;

        // Count clients
        let client_count = peers
            .iter()
            .filter(|p| matches!(p.role, crate::mesh::NodeRole::Client))
            .count();

        // Check max-clients cap
        if let Some(cap) = max_clients {
            if client_count >= cap && !delisted {
                if let Err(e) = publisher.unpublish().await {
                    tracing::warn!("Failed to unpublish from Nostr: {e}");
                }
                eprintln!(
                    "📡 Delisted from Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = true;
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                continue;
            } else if client_count < cap && delisted {
                eprintln!(
                    "📡 Re-publishing to Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = false;
            }
        }

        if delisted {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            continue;
        }

        // ── Solo convergence: if we have no GPU peers, look for a mesh to join ──
        // First try split-heal (same mesh_id, different publisher, more nodes).
        // Then try merging with any other mesh (different mesh, unnamed only).
        // Only merge into meshes strictly larger than us to avoid two solo nodes
        // endlessly unpublishing and trying to join each other.
        let gpu_peers = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count();
        let my_node_count = gpu_peers + 1; // peers + self
        if gpu_peers == 0 {
            let filter = MeshFilter::default();
            if let Ok(listings) = discover(&relays, &filter).await {
                let my_npub = publisher.npub();
                let my_mesh_id = node.mesh_id().await;

                // 1. Split-heal: same mesh_id from a different publisher with more nodes than us
                let split_target = my_mesh_id.as_ref().and_then(|mid| {
                    listings.iter().find(|m| {
                        m.listing.mesh_id.as_deref() == Some(mid.as_str())
                            && m.publisher_npub != my_npub
                            && m.listing.node_count > my_node_count
                    })
                });

                // 2. Merge: any unnamed mesh from a different publisher that is
                //    strictly larger than us. Two solo nodes (both node_count=1)
                //    must NOT try to merge — that creates a storm where both
                //    unpublish and race to join each other.
                let merge_target = if split_target.is_none() && name.is_none() {
                    listings.iter().find(|m| {
                        m.publisher_npub != my_npub
                            && m.listing.name.is_none()
                            && m.listing.node_count > my_node_count
                    })
                } else {
                    None
                };

                if let Some(target) = split_target.or(merge_target) {
                    eprintln!(
                        "📡 Found larger mesh '{}' ({} nodes vs our {}) — rejoining",
                        target.listing.name.as_deref().unwrap_or("unnamed"),
                        target.listing.node_count,
                        my_node_count
                    );
                    if let Err(e) = publisher.unpublish().await {
                        tracing::warn!("Failed to unpublish solo listing: {e}");
                    }
                    if let Err(e) = node.join(&target.listing.invite_token).await {
                        tracing::warn!("Merge/rejoin failed: {e}");
                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                    eprintln!("📡 Merged into mesh — resuming publish as member");
                    // Cooldown: give the mesh time to stabilize before checking again
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    continue;
                }
            }
        }

        // "Actually serving" = a Host node has llama-server running for this model.
        let my_role = node.role().await;
        let my_serving = node.serving().await;
        let mut actually_serving: Vec<String> = Vec::new();
        if matches!(my_role, crate::mesh::NodeRole::Host { .. }) {
            if let Some(ref s) = my_serving {
                if !actually_serving.contains(s) {
                    actually_serving.push(s.clone());
                }
            }
        }
        for p in &peers {
            if matches!(p.role, crate::mesh::NodeRole::Host { .. }) {
                if let Some(ref s) = p.serving {
                    if !actually_serving.contains(s) {
                        actually_serving.push(s.clone());
                    }
                }
            }
        }

        let served_set: std::collections::HashSet<&str> =
            actually_serving.iter().map(|s| s.as_str()).collect();

        // Wanted = models with active demand but not currently served by a host
        let active_demand = node.active_demand().await;
        let mut wanted: Vec<String> = Vec::new();
        for m in active_demand.keys() {
            if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                wanted.push(m.clone());
            }
        }

        // Available = all GGUFs on disk across mesh, minus what's already warm
        let mut available: Vec<String> = Vec::new();
        let my_available = node.available_models().await;
        for m in &my_available {
            if !served_set.contains(m.as_str()) && !available.contains(m) {
                available.push(m.clone());
            }
        }
        for p in &peers {
            for m in &p.available_models {
                if !served_set.contains(m.as_str()) && !available.contains(m) {
                    available.push(m.clone());
                }
            }
        }

        let total_vram: u64 = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .map(|p| p.vram_bytes)
            .sum::<u64>()
            + node.vram_bytes();

        let node_count = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count()
            + 1; // +1 for self

        let mesh_id = node.mesh_id().await;

        let listing = MeshListing {
            invite_token,
            serving: actually_serving,
            wanted: wanted,
            on_disk: available,
            total_vram_bytes: total_vram,
            node_count,
            client_count,
            max_clients: max_clients.unwrap_or(0),
            name: name.clone(),
            region: region.clone(),
            mesh_id,
        };

        let ttl = interval_secs * 2;
        match publisher.publish(&listing, ttl).await {
            Ok(()) => tracing::debug!(
                "Published mesh listing ({} models, {} nodes, {} clients)",
                listing.serving.len(),
                listing.node_count,
                client_count
            ),
            Err(e) => tracing::warn!("Failed to publish to Nostr: {e}"),
        }

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

// ---------------------------------------------------------------------------
// Publish watchdog — take over publishing if the original publisher dies
// ---------------------------------------------------------------------------

/// Watch for our mesh's Nostr listing to disappear, then start publishing.
/// Multiple nodes may start publishing simultaneously — that's fine, each
/// publishes with their own key and invite token, giving discoverers
/// multiple entry points to the same mesh.
///
/// Only runs on active (non-client) nodes that joined via `--auto`.
pub async fn publish_watchdog(
    node: crate::mesh::Node,
    relays: Vec<String>,
    mesh_name: Option<String>,
    region: Option<String>,
    check_interval_secs: u64,
) {
    // Short initial wait with jitter (10-30s) — start watching quickly
    let jitter = (rand::random::<u64>() % 20) + 10;
    tokio::time::sleep(Duration::from_secs(jitter)).await;

    loop {
        // Check if any listing for our mesh exists on Nostr
        let filter = MeshFilter::default();
        match discover(&relays, &filter).await {
            Ok(meshes) => {
                let our_peers = node.peers().await;
                let served = node.models_being_served().await;

                // Our mesh is "listed" if any Nostr listing shares at least one
                // model with what we're currently serving.
                let mesh_listed = if served.is_empty() {
                    false
                } else {
                    meshes
                        .iter()
                        .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                };

                if !mesh_listed && (!our_peers.is_empty() || !served.is_empty()) {
                    // Brief backoff with jitter to avoid stampede (3-10s)
                    let backoff = (rand::random::<u64>() % 7) + 3;
                    eprintln!("📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over...");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;

                    // Re-check — maybe another watchdog already took over
                    if let Ok(recheck) = discover(&relays, &filter).await {
                        let still_missing = if served.is_empty() {
                            true
                        } else {
                            !recheck
                                .iter()
                                .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                        };
                        if !still_missing {
                            eprintln!("📡 Someone else took over publishing — standing down");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    }

                    eprintln!("📡 Taking over Nostr publishing for the mesh");
                    let keys = match load_or_create_keys() {
                        Ok(k) => k,
                        Err(e) => {
                            tracing::warn!("Failed to load Nostr keys for publish takeover: {e}");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    };
                    // Start publish loop (blocks forever)
                    publish_loop(node, keys, relays, mesh_name, region, None, 60).await;
                    return;
                }
            }
            Err(e) => {
                tracing::debug!("Publish watchdog: Nostr check failed: {e}");
            }
        }

        // Check frequently so we catch gaps fast
        let next_check = (rand::random::<u64>() % 15) + 20; // 20-35s
        tokio::time::sleep(Duration::from_secs(next_check)).await;
    }
}

// ---------------------------------------------------------------------------
// Discovery — find meshes on Nostr
// ---------------------------------------------------------------------------

/// Criteria for filtering discovered meshes.
#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    /// Match meshes serving (or wanting) this model name (substring match)
    pub model: Option<String>,
    /// Minimum total VRAM in GB
    pub min_vram_gb: Option<f64>,
    /// Geographic region
    pub region: Option<String>,
}

impl MeshFilter {
    pub fn matches(&self, mesh: &DiscoveredMesh) -> bool {
        if let Some(ref model) = self.model {
            let model_lower = model.to_lowercase();
            let has_model = mesh
                .listing
                .serving
                .iter()
                .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .wanted
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .on_disk
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower));
            if !has_model {
                return false;
            }
        }
        if let Some(min_gb) = self.min_vram_gb {
            let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
            if vram_gb < min_gb {
                return false;
            }
        }
        if let Some(ref region) = self.region {
            match &mesh.listing.region {
                Some(r) if r.eq_ignore_ascii_case(region) => {}
                _ => return false,
            }
        }
        true
    }
}

/// Discover meshes from Nostr relays.
pub async fn discover(relays: &[String], filter: &MeshFilter) -> Result<Vec<DiscoveredMesh>> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Anonymous client for read-only discovery
    let keys = Keys::generate();
    let client = Client::new(keys);
    let mut added = 0;
    for relay in relays {
        match client.add_relay(relay).await {
            Ok(_) => added += 1,
            Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
        }
    }
    if added == 0 {
        anyhow::bail!(
            "Could not connect to any Nostr relay (tried {})",
            relays.len()
        );
    }
    client.connect().await;

    let nostr_filter = Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(
            SingleLetterTag::lowercase(Alphabet::K),
            "mesh-llm".to_string(),
        )
        .limit(100);

    let events = match client
        .fetch_events(nostr_filter, Duration::from_secs(5))
        .await
    {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Nostr fetch failed: {e}");
            return Ok(Vec::new()); // No results rather than hard error
        }
    };

    let now = Timestamp::now().as_secs();

    // Dedupe by publisher (keep latest per pubkey, using replaceable event semantics)
    let mut latest: std::collections::HashMap<String, &Event> = std::collections::HashMap::new();
    for event in events.iter() {
        let pubkey = event.pubkey.to_hex();
        if let Some(existing) = latest.get(&pubkey) {
            if event.created_at.as_secs() > existing.created_at.as_secs() {
                latest.insert(pubkey, event);
            }
        } else {
            latest.insert(pubkey, event);
        }
    }

    let mut meshes = Vec::new();
    for (_, event) in &latest {
        // Check expiration
        let expires_at = event
            .tags
            .iter()
            .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("expiration"))
            .and_then(|t| t.as_slice().get(1))
            .and_then(|s| s.parse::<u64>().ok());

        if let Some(exp) = expires_at {
            if exp < now {
                continue; // expired
            }
        }

        let listing: MeshListing = match serde_json::from_str(&event.content) {
            Ok(l) => l,
            Err(_) => continue,
        };

        let publisher_npub = event.pubkey.to_bech32().unwrap_or_default();
        let discovered = DiscoveredMesh {
            listing,
            publisher_npub,
            published_at: event.created_at.as_secs(),
            expires_at,
        };

        if filter.matches(&discovered) {
            meshes.push(discovered);
        }
    }

    // Sort by node count (bigger meshes first), then VRAM
    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok(meshes)
}

// ---------------------------------------------------------------------------
// Smart auto-join: score meshes, detect staleness, prefer geo match
// ---------------------------------------------------------------------------

/// Score a mesh for auto-join. Higher = better.
/// Considers region match, capacity, and model availability.
/// Freshness is mostly irrelevant since Nostr listings expire at 120s (TTL=2×60s),
/// so anything we see from discover() is already reasonably fresh.
pub fn score_mesh(mesh: &DiscoveredMesh, _now_secs: u64, last_mesh_id: Option<&str>) -> i64 {
    let mut score: i64 = 100; // base score — if we can see it, it's alive

    // Target mesh name: very strong bonus if user asked for a specific mesh
    // Default "mesh-llm" name: moderate bonus for the community mesh
    // Other named meshes: penalty (they're someone's private group)
    if let Some(ref name) = mesh.listing.name {
        if name.eq_ignore_ascii_case("mesh-llm") {
            score += 300; // prefer the default community mesh
        } else {
            score -= 200; // named mesh — probably someone's private group
        }
    }

    // Sticky preference: strong bonus for the mesh we were last on
    if let (Some(last_id), Some(mesh_id)) = (last_mesh_id, &mesh.listing.mesh_id) {
        if last_id == mesh_id {
            score += 500; // strong preference, not infinite — dead/degraded mesh loses on other factors
        }
    }

    // Capacity: prefer meshes that aren't full
    if mesh.listing.max_clients > 0 {
        if mesh.listing.client_count >= mesh.listing.max_clients {
            score -= 1000; // full — don't join
        } else {
            let headroom = mesh.listing.max_clients - mesh.listing.client_count;
            score += (headroom as i64).min(20); // some capacity bonus
        }
    }

    // Size: prefer meshes with more nodes (more resilient)
    score += (mesh.listing.node_count as i64).min(10) * 5;

    // Models: prefer meshes with more warm models
    score += (mesh.listing.serving.len() as i64) * 10;

    // Wanted models: mesh needs help — bonus if we'd be useful
    score += (mesh.listing.wanted.len() as i64) * 15;

    score
}

/// Decision from smart auto-join.
#[derive(Debug)]
pub enum AutoDecision {
    /// Ranked list of meshes to try joining (best first)
    Join {
        candidates: Vec<(String, DiscoveredMesh)>,
    },
    /// No suitable mesh found — start a new one with these models
    StartNew { models: Vec<String> },
}

/// Pick meshes to join, ranked by score, or decide to start a new one.
///
/// - Scores all discovered meshes (freshness, region, capacity)
/// - Filters out stale/full meshes
/// - Returns all viable candidates ranked by score so the caller
///   can probe each in order and fall back to the next on failure
pub fn smart_auto(
    meshes: &[DiscoveredMesh],
    my_vram_gb: f64,
    target_name: Option<&str>,
) -> AutoDecision {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = crate::mesh::load_last_mesh_id();

    // If target name is set, only consider meshes with that exact name
    let candidates: Vec<&DiscoveredMesh> = if let Some(target) = target_name {
        meshes
            .iter()
            .filter(|m| {
                m.listing
                    .name
                    .as_ref()
                    .map(|n| n.eq_ignore_ascii_case(target))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        meshes.iter().collect()
    };

    // Score and rank
    let mut scored: Vec<(&DiscoveredMesh, i64)> = candidates
        .iter()
        .map(|m| (*m, score_mesh(m, now, last_mesh_id.as_deref())))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    // Collect viable candidates.
    // If the user specified --mesh-name, take all candidates (they already
    // filtered by name above — the user explicitly asked for this mesh).
    // Otherwise, require positive score to filter out stale/private meshes.
    let viable: Vec<(String, DiscoveredMesh)> = scored
        .iter()
        .filter(|(_, score)| target_name.is_some() || *score > 0)
        .map(|(m, _)| (m.listing.invite_token.clone(), (*m).clone()))
        .collect();

    if !viable.is_empty() {
        return AutoDecision::Join { candidates: viable };
    }

    // No suitable mesh — recommend models for a new one based on VRAM
    let models = default_models_for_vram(my_vram_gb);
    AutoDecision::StartNew { models }
}

/// Model tiers by VRAM requirement (approximate loaded size × 1.1 headroom).
/// Model tiers for auto-selection, ordered largest-first.
/// min_vram = file_size * 1.1 rounded up. Prefer Qwen3 over 2.5 at same tier.
/// Parse a size string like "2.5GB" to GB as f64.
fn parse_size_gb(s: &str) -> f64 {
    s.trim_end_matches("GB").parse::<f64>().unwrap_or(0.0)
}

/// Build model tiers from the catalog, sorted largest first.
/// Each entry is (model_name, min_vram_gb) where min_vram = file_size * 1.1.
/// Excludes draft models (< 1GB).
fn model_tiers() -> Vec<(&'static str, f64)> {
    let mut tiers: Vec<_> = crate::models::CURATED_MODELS
        .iter()
        .filter(|m| parse_size_gb(m.size) >= 1.0) // skip drafts
        .map(|m| (m.name, parse_size_gb(m.size) * 1.1))
        .collect();
    tiers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tiers
}

/// Pick models to SERVE for `--auto` based on VRAM and what's on disk.
/// Returns models this node should actually load into llama-servers.
///
/// Strategy: one strong generalist + one code specialist when VRAM allows.
/// MoE models preferred (faster tok/s per VRAM GB). Prefers on-disk models
/// to avoid download wait. Leaves ~15% VRAM headroom for KV cache.
///
/// Packs by VRAM tier:
///   <13GB:   Qwen3-8B (5G)
///   13-22GB: Qwen3-8B (5G) + Coder-7B (4.4G)
///   22-28GB: GLM-4.7-Flash (18G) — MoE, fast, good all-rounder
///   28-52GB: Qwen3-30B-A3B (17G) + Coder-7B (4.4G)
///   55-58GB: Qwen2.5-32B (20G) + Qwen3-30B-A3B (17G) + Coder-7B (4.4G)
///   58-85GB: Qwen2.5-72B (47G)
///   85-165GB: Qwen2.5-72B (47G) + Coder-32B (20G)
///   165GB+:  MiniMax-M2.5 (138G)
pub fn auto_model_pack(vram_gb: f64) -> Vec<String> {
    let local_models = crate::models::scan_local_models();
    let tiers = model_tiers();

    // Helper: check if a model is on disk
    let on_disk = |name: &str| local_models.contains(&name.to_string());
    // Helper: model size from tiers
    let size_of = |name: &str| -> f64 {
        tiers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or(f64::MAX)
    };
    let fits = |name: &str, budget: f64| -> bool { size_of(name) <= budget };

    let usable = vram_gb * 0.85; // 15% headroom for KV cache

    // Opinionated packs — each is (generalist, optional specialist(s))
    // The order within a tier prefers: on-disk first, then opinionated default.
    struct Pack {
        min_vram: f64,
        models: &'static [&'static str],
    }
    let packs: &[Pack] = &[
        // Sizes: MiniMax=138G, 72B=47G, Coder-32B=20G, 32B=20G, 30B-A3B=17.3G,
        //        GLM-Flash=18G, 14B=9G, Qwen3-8B=5G, Coder-7B=4.4G
        // With 1.1× tier multiplier and 0.85× usable VRAM.
        Pack {
            min_vram: 165.0,
            models: &["MiniMax-M2.5-Q4_K_M"],
        },
        Pack {
            min_vram: 85.0,
            models: &[
                "Qwen2.5-72B-Instruct-Q4_K_M",
                "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
            ],
        },
        Pack {
            min_vram: 58.0,
            models: &["Qwen2.5-72B-Instruct-Q4_K_M"],
        },
        Pack {
            min_vram: 55.0,
            models: &[
                "Qwen2.5-32B-Instruct-Q4_K_M",
                "Qwen3-30B-A3B-Q4_K_M",
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
            ],
        },
        Pack {
            min_vram: 28.0,
            models: &["Qwen3-30B-A3B-Q4_K_M", "Qwen2.5-Coder-7B-Instruct-Q4_K_M"],
        },
        Pack {
            min_vram: 22.0,
            models: &["GLM-4.7-Flash-Q4_K_M"],
        },
        Pack {
            min_vram: 13.0,
            models: &["Qwen3-8B-Q4_K_M", "Qwen2.5-Coder-7B-Instruct-Q4_K_M"],
        },
        Pack {
            min_vram: 0.0,
            models: &["Qwen3-8B-Q4_K_M"],
        },
    ];

    // Find the best pack that fits
    for pack in packs {
        if vram_gb < pack.min_vram {
            continue;
        }
        // Check all models in the pack actually fit within usable VRAM
        let total: f64 = pack.models.iter().map(|m| size_of(m)).sum();
        if total <= usable {
            return pack.models.iter().map(|m| m.to_string()).collect();
        }
    }

    // Fallback: find the largest single model that fits, prefer on-disk
    let on_disk_fit = tiers
        .iter()
        .find(|(name, min_vram)| *min_vram <= usable && on_disk(name));
    let any_fit = tiers.iter().find(|(_, min_vram)| *min_vram <= usable);

    let primary = on_disk_fit
        .or(any_fit)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "Qwen2.5-3B-Instruct-Q4_K_M".into());

    // Try to add a code specialist if there's room
    let remaining = usable - size_of(&primary);
    let coders = [
        "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
        "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
        "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
    ];
    for coder in coders {
        if coder != primary && fits(coder, remaining) {
            return vec![primary, coder.to_string()];
        }
    }

    vec![primary]
}

/// Models to advertise as "wanted" for demand seeding.
/// These tell other nodes what the mesh could use, covering every VRAM tier.
/// NOT served by this node — just demand hints for the mesh.
pub fn demand_seed_models() -> Vec<String> {
    vec![
        "GLM-4.7-Flash-Q4_K_M".into(),
        "Qwen3-30B-A3B-Q4_K_M".into(),
        "Qwen3-8B-Q4_K_M".into(),
        "Qwen2.5-3B-Instruct-Q4_K_M".into(),
        "Qwen3-0.6B-Q4_K_M".into(),
    ]
}

/// Legacy wrapper — returns serving models + demand seeds combined.
/// Used by `smart_auto` for the StartNew decision.
pub fn default_models_for_vram(vram_gb: f64) -> Vec<String> {
    let mut models = auto_model_pack(vram_gb);
    for m in demand_seed_models() {
        if !models.contains(&m) {
            models.push(m);
        }
    }
    models
}

#[cfg(test)]
mod auto_pack_tests {
    use super::*;

    #[test]
    fn pack_8gb_single_model() {
        let pack = auto_model_pack(8.0);
        assert_eq!(pack.len(), 1);
        assert_eq!(pack[0], "Qwen3-8B-Q4_K_M");
    }

    #[test]
    fn pack_16gb_dual_model() {
        let pack = auto_model_pack(16.0);
        assert_eq!(pack.len(), 2);
        assert_eq!(pack[0], "Qwen3-8B-Q4_K_M");
        assert_eq!(pack[1], "Qwen2.5-Coder-7B-Instruct-Q4_K_M");
    }

    #[test]
    fn pack_24gb_glm_flash() {
        let pack = auto_model_pack(24.0);
        assert_eq!(pack.len(), 1);
        assert_eq!(pack[0], "GLM-4.7-Flash-Q4_K_M");
    }

    #[test]
    fn pack_32gb_generalist_plus_coder() {
        let pack = auto_model_pack(32.0);
        assert_eq!(pack.len(), 2);
        assert_eq!(pack[0], "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(pack[1], "Qwen2.5-Coder-7B-Instruct-Q4_K_M");
    }

    #[test]
    fn pack_52gb_generalist_plus_coder() {
        // 52GB isn't enough for triple pack (needs 55+), gets dual instead
        let pack = auto_model_pack(52.0);
        assert_eq!(pack.len(), 2);
        assert_eq!(pack[0], "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(pack[1], "Qwen2.5-Coder-7B-Instruct-Q4_K_M");
    }

    #[test]
    fn pack_55gb_triple() {
        let pack = auto_model_pack(55.0);
        assert_eq!(pack.len(), 3);
        assert!(pack.contains(&"Qwen2.5-32B-Instruct-Q4_K_M".to_string()));
        assert!(pack.contains(&"Qwen3-30B-A3B-Q4_K_M".to_string()));
        assert!(pack.contains(&"Qwen2.5-Coder-7B-Instruct-Q4_K_M".to_string()));
    }

    #[test]
    fn pack_72gb_frontier() {
        let pack = auto_model_pack(72.0);
        assert_eq!(pack.len(), 1);
        assert_eq!(pack[0], "Qwen2.5-72B-Instruct-Q4_K_M");
    }

    #[test]
    fn pack_96gb_frontier_plus_coder() {
        let pack = auto_model_pack(96.0);
        assert_eq!(pack.len(), 2);
        assert_eq!(pack[0], "Qwen2.5-72B-Instruct-Q4_K_M");
        assert_eq!(pack[1], "Qwen2.5-Coder-32B-Instruct-Q4_K_M");
    }

    #[test]
    fn pack_206gb_minimax() {
        let pack = auto_model_pack(206.0);
        assert_eq!(pack.len(), 1);
        assert_eq!(pack[0], "MiniMax-M2.5-Q4_K_M");
    }

    #[test]
    fn demand_seeds_are_separate() {
        let seeds = demand_seed_models();
        assert!(seeds.len() >= 4);
        // Seeds should cover small to large
        assert!(seeds.contains(&"Qwen3-0.6B-Q4_K_M".to_string()));
        assert!(seeds.contains(&"GLM-4.7-Flash-Q4_K_M".to_string()));
    }

    #[test]
    fn default_models_includes_both() {
        let all = default_models_for_vram(30.0);
        let pack = auto_model_pack(30.0);
        let seeds = demand_seed_models();
        // Pack models come first
        for m in &pack {
            assert!(all.contains(m));
        }
        // Seeds are also present
        for m in &seeds {
            assert!(all.contains(m));
        }
        // No duplicates
        let mut deduped = all.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(all.len(), deduped.len());
    }
}
