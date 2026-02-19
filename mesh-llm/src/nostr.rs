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
    /// Models currently warm (loaded in VRAM)
    pub models: Vec<String>,
    /// Models the mesh wants served but nobody has loaded yet
    #[serde(default)]
    pub wanted_models: Vec<String>,
    /// Total VRAM across all GPU nodes (bytes)
    pub total_vram_bytes: u64,
    /// Number of GPU nodes in the mesh
    pub node_count: usize,
    /// Optional human-readable name for the mesh
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional geographic region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
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
        let models = if self.listing.models.is_empty() {
            "(no models loaded)".to_string()
        } else {
            self.listing.models.join(", ")
        };
        write!(
            f,
            "{}  {} node(s), {:.0}GB VRAM  models: {}",
            self.listing.name.as_deref().unwrap_or("(unnamed)"),
            self.listing.node_count,
            vram_gb,
            models,
        )?;
        if let Some(ref region) = self.listing.region {
            write!(f, "  region: {}", region)?;
        }
        if !self.listing.wanted_models.is_empty() {
            write!(f, "  wanted: {}", self.listing.wanted_models.join(", "))?;
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
pub async fn publish_loop(
    node: crate::mesh::Node,
    keys: Keys,
    relays: Vec<String>,
    name: Option<String>,
    region: Option<String>,
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

    loop {
        // Gather current mesh state
        let invite_token = node.invite_token();
        let peers = node.peers().await;
        let models = node.models_being_served().await;

        // Wanted = requested but not yet served
        let served_set: std::collections::HashSet<&str> = models.iter().map(|s| s.as_str()).collect();
        let mut wanted: Vec<String> = Vec::new();
        for p in &peers {
            for m in &p.requested_models {
                if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                    wanted.push(m.clone());
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

        let listing = MeshListing {
            invite_token,
            models,
            wanted_models: wanted,
            total_vram_bytes: total_vram,
            node_count,
            name: name.clone(),
            region: region.clone(),
        };

        // TTL = 2× interval, so listing expires if we stop publishing
        let ttl = interval_secs * 2;
        match publisher.publish(&listing, ttl).await {
            Ok(()) => tracing::debug!("Published mesh listing ({} models, {} nodes)", listing.models.len(), listing.node_count),
            Err(e) => tracing::warn!("Failed to publish to Nostr: {e}"),
        }

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
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
            let has_model = mesh.listing.models.iter().any(|m| m.to_lowercase().contains(&model_lower))
                || mesh.listing.wanted_models.iter().any(|m| m.to_lowercase().contains(&model_lower));
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
