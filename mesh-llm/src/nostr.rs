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
#[derive(Debug, Clone)]
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
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
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
        eprintln!("🔑 Deleted {}. A new key will be generated on next --publish.", path.display());
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
            Tag::custom(TagKind::Custom("expiration".into()), vec![expiration.to_string()]),
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
        let events = self.client.fetch_events(filter, Duration::from_secs(5)).await?;
        for event in events.iter() {
            let request = EventDeletionRequest::new().id(event.id);
            let _ = self.client.send_event_builder(EventBuilder::delete(request)).await;
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
    eprintln!("📡 Publishing mesh to Nostr (npub: {}...{})", &npub[..12], &npub[npub.len()-8..]);
    if let Some(cap) = max_clients {
        eprintln!("   Will delist when {} clients connected", cap);
    }

    let mut delisted = false;

    loop {
        let invite_token = node.invite_token();
        let peers = node.peers().await;

        // Count clients
        let client_count = peers.iter()
            .filter(|p| matches!(p.role, crate::mesh::NodeRole::Client))
            .count();

        // Check max-clients cap
        if let Some(cap) = max_clients {
            if client_count >= cap && !delisted {
                if let Err(e) = publisher.unpublish().await {
                    tracing::warn!("Failed to unpublish from Nostr: {e}");
                }
                eprintln!("📡 Delisted from Nostr ({} clients, cap is {})", client_count, cap);
                delisted = true;
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                continue;
            } else if client_count < cap && delisted {
                eprintln!("📡 Re-publishing to Nostr ({} clients, cap is {})", client_count, cap);
                delisted = false;
            }
        }

        if delisted {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            continue;
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

        let served_set: std::collections::HashSet<&str> = actually_serving.iter().map(|s| s.as_str()).collect();

        // Wanted = requested but not actually being served by a host
        let mut wanted: Vec<String> = Vec::new();
        let my_requested = node.requested_models().await;
        for m in &my_requested {
            if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                wanted.push(m.clone());
            }
        }
        for p in &peers {
            for m in &p.requested_models {
                if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                    wanted.push(m.clone());
                }
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

        let total_vram: u64 = peers.iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .map(|p| p.vram_bytes)
            .sum::<u64>()
            + node.vram_bytes();

        let node_count = peers.iter()
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
            Ok(()) => tracing::debug!("Published mesh listing ({} models, {} nodes, {} clients)", listing.serving.len(), listing.node_count, client_count),
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
    // Random jitter (30-90s) so multiple watchdogs don't all fire at once
    let jitter = (rand::random::<u64>() % 60) + 30;
    tokio::time::sleep(Duration::from_secs(check_interval_secs + jitter)).await;

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
                    meshes.iter().any(|m| {
                        served.iter().any(|s| m.listing.serving.contains(s))
                    })
                };

                if !mesh_listed && !our_peers.is_empty() {
                    // Double-check after a short wait — maybe another watchdog
                    // already took over and we just haven't seen it yet
                    let backoff = (rand::random::<u64>() % 30) + 10;
                    eprintln!("📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over...");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;

                    // Re-check
                    if let Ok(recheck) = discover(&relays, &filter).await {
                        let still_missing = if served.is_empty() {
                            true
                        } else {
                            !recheck.iter().any(|m| {
                                served.iter().any(|s| m.listing.serving.contains(s))
                            })
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

        tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
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
            let has_model = mesh.listing.serving.iter().any(|m| m.to_lowercase().contains(&model_lower))
                || mesh.listing.wanted.iter().any(|m| m.to_lowercase().contains(&model_lower))
                || mesh.listing.on_disk.iter().any(|m| m.to_lowercase().contains(&model_lower));
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
    for relay in relays {
        client.add_relay(relay).await?;
    }
    client.connect().await;

    let nostr_filter = Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(SingleLetterTag::lowercase(Alphabet::K), "mesh-llm".to_string())
        .limit(100);

    let events = client.fetch_events(nostr_filter, Duration::from_secs(10)).await?;

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
        let expires_at = event.tags.iter()
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
        b.listing.node_count.cmp(&a.listing.node_count)
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
pub fn score_mesh(mesh: &DiscoveredMesh, my_region: Option<&str>, _now_secs: u64, last_mesh_id: Option<&str>) -> i64 {
    let mut score: i64 = 100; // base score — if we can see it, it's alive

    // Sticky preference: strong bonus for the mesh we were last on
    if let (Some(last_id), Some(mesh_id)) = (last_mesh_id, &mesh.listing.mesh_id) {
        if last_id == mesh_id {
            score += 500; // strong preference, not infinite — dead/degraded mesh loses on other factors
        }
    }

    // Region match: strong preference for same region
    if let (Some(my_r), Some(mesh_r)) = (my_region, &mesh.listing.region) {
        if my_r.eq_ignore_ascii_case(mesh_r) {
            score += 200; // same region — low latency
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

    // VRAM: prefer meshes with more total VRAM
    let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
    score += (vram_gb as i64).min(50);

    // Models: prefer meshes with more warm models
    score += (mesh.listing.serving.len() as i64) * 10;

    // Wanted models: mesh needs help — bonus if we'd be useful
    score += (mesh.listing.wanted.len() as i64) * 15;

    score
}

/// Decision from smart auto-join.
#[derive(Debug)]
pub enum AutoDecision {
    /// Join this mesh's invite token
    Join { token: String, mesh: DiscoveredMesh },
    /// No suitable mesh found — start a new one with these models
    StartNew { models: Vec<String> },
}

/// Pick the best mesh to join, or decide to start a new one.
///
/// - Scores all discovered meshes (freshness, region, capacity)
/// - Filters out stale/full meshes
/// - If no good mesh found, recommends starting a new one with
///   models appropriate for the node's VRAM
pub fn smart_auto(
    meshes: &[DiscoveredMesh],
    my_region: Option<&str>,
    my_vram_gb: f64,
) -> AutoDecision {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = crate::mesh::load_last_mesh_id();

    // Score and rank
    let mut scored: Vec<(&DiscoveredMesh, i64)> = meshes.iter()
        .map(|m| (m, score_mesh(m, my_region, now, last_mesh_id.as_deref())))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    // Best mesh must have a positive score (not stale, not full)
    if let Some((best, score)) = scored.first() {
        if *score > 0 {
            return AutoDecision::Join {
                token: best.listing.invite_token.clone(),
                mesh: (*best).clone(),
            };
        }
    }

    // No suitable mesh — recommend models for a new one based on VRAM
    let models = default_models_for_vram(my_vram_gb);
    AutoDecision::StartNew { models }
}

/// Auto-detect region by briefly connecting to iroh's default relay.
/// iroh picks the closest relay, so the relay URL tells us our region.
pub async fn detect_region_auto() -> Option<String> {
    use iroh::Endpoint;
    // Endpoint::builder().bind() generates its own key
    let ep = Endpoint::builder()
        .bind().await.ok()?;
    // Wait briefly for relay assignment
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    let addr = ep.addr();
    for transport_addr in &addr.addrs {
        if let iroh::TransportAddr::Relay(url) = transport_addr {
            let region = region_from_relay_url(url.as_str());
            ep.close().await;
            return region;
        }
    }
    ep.close().await;
    None
}

/// Extract region code from a relay URL.
pub fn region_from_relay_url(url: &str) -> Option<String> {
    // Extract hostname prefix before first dot
    let host = url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let prefix = host.split('.').next()?;
    // prefix is like "aps1-1" — take first part before the dash-number
    let region_code = prefix.split('-').next()?;
    match region_code {
        "aps1" | "aps2" => Some("AU".into()),  // Asia Pacific South
        "apn1" | "apn2" => Some("JP".into()),  // Asia Pacific North
        "usw1" | "usw2" => Some("US".into()),  // US West
        "use1" | "use2" => Some("US".into()),  // US East
        "euw1" | "euw2" => Some("EU".into()),  // Europe West
        "euc1" | "euc2" => Some("EU".into()),  // Europe Central
        _ => None,
    }
}

/// Model tiers by VRAM requirement (approximate loaded size × 1.1 headroom).
/// Model tiers for auto-selection, ordered largest-first.
/// min_vram = file_size * 1.1 rounded up. Prefer Qwen3 over 2.5 at same tier.
const MODEL_TIERS: &[(&str, f64)] = &[
    ("Qwen2.5-72B-Instruct-Q4_K_M", 53.0),
    ("Llama-3.3-70B-Instruct-Q4_K_M", 48.0),
    ("Llama-4-Scout-Q4_K_M", 26.0),
    ("Qwen3-32B-Q4_K_M", 24.0),
    ("GLM-4-32B-0414-Q4_K_M", 24.0),
    ("Qwen2.5-Coder-32B-Instruct-Q4_K_M", 24.0),
    ("GLM-4.7-Flash-Q4_K_M", 21.0),
    ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", 21.0),
    ("Gemma-3-27B-it-Q4_K_M", 20.0),
    ("Devstral-Small-2505-Q4_K_M", 17.0),
    ("Mistral-Small-3.1-24B-Instruct-Q4_K_M", 17.0),
    ("Qwen3-14B-Q4_K_M", 11.0),
    ("Qwen2.5-Coder-14B-Instruct-Q4_K_M", 11.0),
    ("DeepSeek-R1-Distill-Qwen-14B-Q4_K_M", 11.0),
    ("Gemma-3-12B-it-Q4_K_M", 9.0),
    ("Qwen3-8B-Q4_K_M", 6.0),
    ("Qwen2.5-Coder-7B-Instruct-Q4_K_M", 6.0),
    ("Qwen3-4B-Q4_K_M", 3.0),
    ("Qwen2.5-3B-Instruct-Q4_K_M", 3.0),
];

/// Pick models for a new mesh based on VRAM and what's on disk.
/// Returns 1-3 model names. The first one is what this node will serve
/// (largest that fits, preferring on-disk). The rest are "wanted" models
/// so other joiners know what the mesh could use.
pub fn default_models_for_vram(vram_gb: f64) -> Vec<String> {
    let local_models = crate::mesh::scan_local_models();

    // Find the largest model that fits AND is already on disk
    let on_disk_fit = MODEL_TIERS.iter()
        .find(|(name, min_vram)| *min_vram <= vram_gb && local_models.contains(&name.to_string()));

    // Find the largest model that fits (may need download)
    let any_fit = MODEL_TIERS.iter()
        .find(|(_, min_vram)| *min_vram <= vram_gb);

    // Primary: prefer on-disk, fall back to downloadable
    let primary = on_disk_fit.or(any_fit)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "Qwen2.5-3B-Instruct-Q4_K_M".into());

    let mut models = vec![primary.clone()];

    // Add a small model as secondary (for other nodes to serve)
    if !models.contains(&"Qwen2.5-3B-Instruct-Q4_K_M".into()) {
        models.push("Qwen2.5-3B-Instruct-Q4_K_M".into());
    }

    // Add a coder model as tertiary if we have room in the mesh
    let coder = "Qwen2.5-Coder-7B-Instruct-Q4_K_M".to_string();
    if !models.contains(&coder) && models.len() < 3 {
        models.push(coder);
    }

    models
}
