mod console;
mod download;
mod election;
mod launch;
mod mesh;
mod nostr;
mod rewrite;
mod tunnel;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mesh::NodeRole;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mesh-llm", about = "P2P mesh for distributed llama.cpp inference over QUIC")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Join an existing mesh via an invite token.
    /// Can be specified multiple times — only one needs to be reachable.
    #[arg(long, short, global = true)]
    join: Vec<String>,

    /// Discover a mesh from Nostr and join it automatically.
    /// Optionally specify a model name to filter by.
    #[arg(long, default_missing_value = "", num_args = 0..=1)]
    discover: Option<String>,

    /// Auto-join: discover a mesh via Nostr and join it.
    /// Equivalent to: mesh-llm --join $(mesh-llm discover --auto)
    #[arg(long)]
    auto: bool,

    /// GGUF model to serve. Can be a path, catalog name, or HuggingFace URL.
    /// Specify multiple times to seed the mesh with multiple models.
    /// When joining without --model, the mesh assigns one automatically.
    #[arg(long)]
    model: Vec<PathBuf>,

    /// Local HTTP port for the API (default: 9337).
    /// The elected host runs llama-server here; workers proxy to the host.
    #[arg(long, default_value = "9337")]
    port: u16,

    /// Path to directory containing rpc-server and llama-server binaries.
    /// Defaults to the same directory as the mesh-llm binary itself.
    #[arg(long)]
    bin_dir: Option<PathBuf>,

    /// Device for rpc-server (e.g. MTL0, CPU). Default: auto-detect.
    #[arg(long)]
    device: Option<String>,

    /// Tensor split ratios for llama-server (e.g. "0.8,0.2").
    /// Without this, split is auto-calculated from VRAM.
    #[arg(long)]
    tensor_split: Option<String>,

    /// Run as a lite client — no GPU, no rpc-server, no model needed.
    #[arg(long)]
    client: bool,

    /// Path to a draft model for speculative decoding (e.g. a small quant of the same model).
    /// Only used on the host — the draft model runs locally, not distributed.
    /// If omitted, auto-detected from catalog when the main model has a known draft pairing.
    #[arg(long)]
    draft: Option<PathBuf>,

    /// Max draft tokens for speculative decoding (default: 8).
    #[arg(long, default_value = "8")]
    draft_max: u16,

    /// Disable automatic draft model detection from catalog.
    #[arg(long)]
    no_draft: bool,

    /// Force tensor split across all GPU nodes even if the model fits on the host.
    /// Without this, the host loads solo when it has enough VRAM.
    #[arg(long)]
    split: bool,

    /// Limit VRAM advertised to the mesh (in GB). Other nodes will see this
    /// instead of your actual VRAM, capping how much work gets split to you.
    #[arg(long)]
    max_vram: Option<f64>,

    /// Override iroh relay URLs (e.g. --relay https://staging-use1-1.relay.iroh.network./).
    /// Can be specified multiple times. Without this, iroh uses its built-in defaults.
    #[arg(long, global = true)]
    relay: Vec<String>,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, global = true)]
    bind_port: Option<u16>,

    /// Start the web console on this port (default: 3131 if flag is present).
    #[arg(long, default_missing_value = "3131", num_args = 0..=1)]
    console: Option<u16>,

    /// Publish this mesh to Nostr for discovery by others.
    /// Republishes every 60s so the listing stays fresh.
    #[arg(long)]
    publish: bool,

    /// Human-readable name for this mesh (shown in discovery).
    #[arg(long)]
    mesh_name: Option<String>,

    /// Geographic region tag (e.g. "US", "EU", "AU"). Shown in discovery.
    #[arg(long)]
    region: Option<String>,

    /// Stop advertising on Nostr when this many clients are connected.
    /// Re-publishes when clients drop below the cap. No cap by default.
    #[arg(long)]
    max_clients: Option<usize>,

    /// Nostr relay URLs for publishing/discovery (default: damus, nos.lol, nostr.band).
    #[arg(long)]
    nostr_relay: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Download a model from the catalog
    Download {
        /// Model name (e.g. "Qwen2.5-32B-Instruct-Q4_K_M" or just "32b")
        name: Option<String>,
        /// Also download the recommended draft model for speculative decoding
        #[arg(long)]
        draft: bool,
    },
    /// Drop a model from the mesh — stops all nodes serving it
    Drop {
        /// Model name to drop
        name: String,
        /// API port of the running mesh-llm instance (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Discover meshes published to Nostr and optionally auto-join one
    Discover {
        /// Filter by model name (substring match)
        #[arg(long)]
        model: Option<String>,
        /// Filter by minimum VRAM (GB)
        #[arg(long)]
        min_vram: Option<f64>,
        /// Filter by region
        #[arg(long)]
        region: Option<String>,
        /// Print the invite token of the best match (for piping to --join)
        #[arg(long)]
        auto: bool,
        /// Nostr relay URLs (default: damus, nos.lol, nostr.band)
        #[arg(long)]
        relay: Vec<String>,
    },
    /// Rotate the Nostr identity key (forces new keypair on next --publish)
    RotateKey,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?)
                .add_directive("nostr_relay_pool=off".parse()?)
                .add_directive("nostr_sdk=warn".parse()?),
        )
        .with_writer(std::io::stderr)
        .init();

    let mut cli = Cli::parse();

    // Subcommand dispatch
    if let Some(cmd) = &cli.command {
        match cmd {
            Command::Download { name, draft } => {
                match name {
                    Some(query) => {
                        let model = download::find_model(query)
                            .ok_or_else(|| anyhow::anyhow!("No model matching '{}' in catalog. Run `mesh-llm download` to list.", query))?;
                        download::download_model(model).await?;
                        if *draft {
                            if let Some(draft_name) = model.draft {
                                let draft_model = download::find_model(draft_name)
                                    .ok_or_else(|| anyhow::anyhow!("Draft model '{}' not found in catalog", draft_name))?;
                                download::download_model(draft_model).await?;
                            } else {
                                eprintln!("⚠ No draft model available for {}", model.name);
                            }
                        }
                    }
                    None => download::list_models(),
                }
                return Ok(());
            }
            Command::Drop { name, port } => {
                return run_drop(name, *port).await;
            }
            Command::Discover { model, min_vram, region, auto, relay } => {
                return run_discover(model.clone(), *min_vram, region.clone(), *auto, relay.clone()).await;
            }
            Command::RotateKey => {
                return nostr::rotate_keys().map_err(Into::into);
            }
        }
    }

    // --- Auto-discover ---
    if cli.auto && cli.join.is_empty() {
        eprintln!("🔍 Discovering meshes via Nostr...");

        // Auto-detect region from iroh relay if not specified
        let my_region = if let Some(ref r) = cli.region {
            Some(r.clone())
        } else {
            eprintln!("  Detecting region from relay...");
            let detected = nostr::detect_region_auto().await;
            if let Some(ref r) = detected {
                eprintln!("  Region: {r} (auto-detected)");
            }
            detected
        };

        let relays = nostr_relays(&cli.nostr_relay);
        // Don't hard-filter by region — let scoring prefer same-region
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: None,
        };
        let meshes = nostr::discover(&relays, &filter).await?;
        let my_vram_gb = mesh::detect_vram_bytes_capped(cli.max_vram) as f64 / 1e9;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let last_mesh_id = mesh::load_last_mesh_id();
        eprintln!("  Found {} mesh(es)", meshes.len());
        for m in &meshes {
            let score = nostr::score_mesh(m, my_region.as_deref(), now, last_mesh_id.as_deref());
            eprintln!("  · {} (score: {}, {} nodes, {:.0}GB, {} clients{})",
                m.listing.name.as_deref().unwrap_or("unnamed"),
                score,
                m.listing.node_count,
                m.listing.total_vram_bytes as f64 / 1e9,
                m.listing.client_count,
                m.listing.region.as_ref().map(|r| format!(", {r}")).unwrap_or_default());
        }

        match nostr::smart_auto(&meshes, my_region.as_deref(), my_vram_gb) {
            nostr::AutoDecision::Join { token, mesh } => {
                // Health probe: try connecting before committing
                eprintln!("  Probing mesh health...");
                match probe_mesh_health(&token).await {
                    Ok(()) => {
                        eprintln!("✅ Joining: {} ({} nodes, {} models{})",
                            mesh.listing.name.as_deref().unwrap_or("unnamed"),
                            mesh.listing.node_count,
                            mesh.listing.serving.len(),
                            mesh.listing.region.as_ref().map(|r| format!(", region: {r}")).unwrap_or_default());
                        cli.join.push(token);
                    }
                    Err(e) => {
                        eprintln!("⚠️  Best mesh unreachable: {e}");
                        if cli.client {
                            anyhow::bail!("Mesh found but unreachable. Try again later.");
                        }
                        let models = nostr::default_models_for_vram(my_vram_gb);
                        start_new_mesh(&mut cli, &models, my_vram_gb, &my_region);
                    }
                }
            }
            nostr::AutoDecision::StartNew { models } => {
                if cli.client {
                    anyhow::bail!("No meshes found to join. Run without --client to start a new mesh.");
                }
                start_new_mesh(&mut cli, &models, my_vram_gb, &my_region);
            }
        }
    }

    // --- Validation ---
    if cli.client && cli.join.is_empty() {
        anyhow::bail!("--client requires --join to connect to a mesh");
    }
    if cli.client && !cli.model.is_empty() {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    if cli.model.is_empty() && cli.join.is_empty() && !cli.client && !cli.auto {
        anyhow::bail!("--model is required (or use --join, --auto, or --client)");
    }

    // --- Client mode (passive, never serves) ---
    if cli.client {
        let (node, _channels) = mesh::Node::start(NodeRole::Client, &cli.relay, cli.bind_port, None).await?;
        // No heartbeat for passive nodes — they don't track peers

        let mut joined = false;
        for t in &cli.join {
            match node.join_passive(t).await {
                Ok(()) => { eprintln!("Joined mesh (passive)"); joined = true; break; }
                Err(e) => tracing::warn!("Failed to join via token: {e}"),
            }
        }
        if !joined {
            anyhow::bail!("Failed to join any peer in the mesh");
        }

        // Periodic route table refresh + reconnect (instead of gossip heartbeat)
        let refresh_node = node.clone();
        let refresh_tokens: Vec<String> = cli.join.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                // Reconnect if needed
                for t in &refresh_tokens {
                    let _ = refresh_node.join_passive(t).await;
                }
                // Refresh routing table from any connected peer
                refresh_node.refresh_routing_table().await;
            }
        });

        // Client never promotes — discard the return
        run_passive(&cli, node, true).await?;
        return Ok(());
    }

    // --- Resolve models from CLI ---
    let mut resolved_models: Vec<PathBuf> = Vec::new();
    for m in &cli.model {
        resolved_models.push(resolve_model(m).await?);
    }

    // Collect requested model names (what the user explicitly asked for)
    let requested_model_names: Vec<String> = resolved_models.iter()
        .filter_map(|m| m.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .collect();

    let bin_dir = match &cli.bin_dir {
        Some(d) => d.clone(),
        None => detect_bin_dir()?,
    };

    run_auto(cli, resolved_models, requested_model_names, bin_dir).await
}

/// Resolve a model path: local file, catalog name, or HuggingFace URL.
async fn resolve_model(input: &std::path::Path) -> Result<PathBuf> {
    let s = input.to_string_lossy();

    // Already a local file
    if input.exists() {
        return Ok(input.to_path_buf());
    }

    // Check ~/.models/ for just a filename
    if !s.contains('/') {
        let in_models = download::models_dir().join(input);
        if in_models.exists() {
            return Ok(in_models);
        }
        // Try catalog match
        if let Some(entry) = download::find_model(&s) {
            return download::download_model(entry).await;
        }
        anyhow::bail!(
            "Model not found: {}\nNot a local file, not in ~/.models/, not in catalog.\n\
             Use a path, a catalog name (run `mesh-llm download` to list), or a HuggingFace URL.",
            s
        );
    }

    // HuggingFace URL
    if s.starts_with("https://huggingface.co/") || s.starts_with("http://huggingface.co/") {
        let filename = s.rsplit('/').next()
            .ok_or_else(|| anyhow::anyhow!("Can't extract filename from URL: {}", s))?;
        let dest = download::models_dir().join(filename);
        if dest.exists() {
            let size = tokio::fs::metadata(&dest).await?.len();
            if size > 1_000_000 {
                eprintln!("✅ {} already exists ({:.1}GB)", filename, size as f64 / 1e9);
                return Ok(dest);
            }
        }
        eprintln!("📥 Downloading {}...", filename);
        download::download_url(&s, &dest).await?;
        return Ok(dest);
    }

    // HF shorthand: org/repo/file.gguf
    if s.contains('/') && s.ends_with(".gguf") {
        let url = if s.contains("/resolve/") {
            format!("https://huggingface.co/{}", s)
        } else {
            let parts: Vec<&str> = s.splitn(3, '/').collect();
            if parts.len() == 3 {
                format!("https://huggingface.co/{}/{}/resolve/main/{}", parts[0], parts[1], parts[2])
            } else {
                anyhow::bail!("Can't parse HF shorthand: {}. Use org/repo/file.gguf", s);
            }
        };
        let filename = s.rsplit('/').next().unwrap();
        let dest = download::models_dir().join(filename);
        if dest.exists() {
            let size = tokio::fs::metadata(&dest).await?.len();
            if size > 1_000_000 {
                eprintln!("✅ {} already exists ({:.1}GB)", filename, size as f64 / 1e9);
                return Ok(dest);
            }
        }
        eprintln!("📥 Downloading {}...", filename);
        download::download_url(&url, &dest).await?;
        return Ok(dest);
    }

    anyhow::bail!("Model not found: {}", s);
}

/// Look up the model filename in the catalog and check if its draft model exists on disk.
/// If not on disk, downloads it (drafts are <1GB).
pub async fn ensure_draft(model: &std::path::Path) -> Option<PathBuf> {
    let filename = model.file_name()?.to_str()?;
    let catalog_entry = download::MODEL_CATALOG.iter().find(|m| m.file == filename)?;
    let draft_name = catalog_entry.draft?;
    let draft_entry = download::MODEL_CATALOG.iter().find(|m| m.name == draft_name)?;
    let draft_path = download::models_dir().join(draft_entry.file);
    if draft_path.exists() {
        return Some(draft_path);
    }
    // Draft not on disk — download it (small, <1GB)
    eprintln!("📥 Downloading draft model {} ({})...", draft_entry.name, draft_entry.size);
    match download::download_model(draft_entry).await {
        Ok(_path) => {
            eprintln!("✅ Draft model ready: {}", draft_entry.name);
            Some(draft_path)
        }
        Err(e) => {
            eprintln!("⚠ Failed to download draft model: {e} — continuing without speculative decoding");
            None
        }
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
/// 3. The most underserved model (fewest nodes serving it relative to its size)
/// 4. Fall back to the first requested model in the mesh
async fn pick_model_assignment(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;

    // Collect what the mesh wants served (from our own + all peers' requested_models)
    let mut mesh_requested: std::collections::HashSet<String> = std::collections::HashSet::new();
    for m in &node.requested_models().await {
        mesh_requested.insert(m.clone());
    }
    for p in &peers {
        for m in &p.requested_models {
            mesh_requested.insert(m.clone());
        }
    }

    if mesh_requested.is_empty() {
        // Nobody has requested anything — shouldn't happen if seeder ran
        return None;
    }

    // Count how many nodes are serving each model
    let mut serving_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &peers {
        if let Some(ref s) = p.serving {
            *serving_count.entry(s.clone()).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    // Find all unserved models we could solo
    let mut candidates: Vec<String> = Vec::new();
    for m in &mesh_requested {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            let model_path = download::models_dir().join(format!("{}.gguf", m));
            let model_bytes = std::fs::metadata(&model_path).map(|md| md.len()).unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                eprintln!("📋 Skipping {} — needs {:.1}GB, we have {:.1}GB",
                    m, needed as f64 / 1e9, my_vram as f64 / 1e9);
                continue;
            }
            candidates.push(m.clone());
        }
    }

    if !candidates.is_empty() {
        // Pick deterministically based on node ID so concurrent joiners spread out.
        // Sort candidates, then hash our node ID to pick an index.
        candidates.sort();
        let my_id = node.id();
        let id_bytes = my_id.as_bytes();
        let hash = id_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % candidates.len();
        let pick = &candidates[idx];
        eprintln!("📋 Assigned to serve {} (needed by mesh, already on disk, {} candidates)", pick, candidates.len());
        return Some(pick.clone());
    }

    // Also check: are there models with fewer servers than others?
    // If model A has 3 servers and model B has 1, we should add to B not go standby.
    let mut underserved: Vec<(String, usize)> = Vec::new();
    let max_count = serving_count.values().copied().max().unwrap_or(0);
    for m in &mesh_requested {
        let count = serving_count.get(m).copied().unwrap_or(0);
        if count < max_count && local_models.contains(m) {
            let model_path = download::models_dir().join(format!("{}.gguf", m));
            let model_bytes = std::fs::metadata(&model_path).map(|md| md.len()).unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            underserved.push((m.clone(), count));
        }
    }
    if !underserved.is_empty() {
        // Pick the least-served model
        underserved.sort_by_key(|(_, count)| *count);
        let (pick, count) = &underserved[0];
        let max_model = serving_count.iter().max_by_key(|(_, &v)| v).map(|(k, _)| k.as_str()).unwrap_or("?");
        eprintln!("📋 Assigned to serve {} ({} servers vs {} has {}) — rebalancing",
            pick, count, max_model, max_count);
        return Some(pick.clone());
    }

    // Everything is balanced — stay standby
    let all_covered = mesh_requested.iter()
        .all(|m| serving_count.get(m).copied().unwrap_or(0) > 0);
    if all_covered {
        eprintln!("📋 All mesh models are balanced — staying on standby");
        return None;
    }

    None
}

/// Check if any mesh-requested model has zero servers and we have it on disk.
/// Unlike pick_model_assignment(), this only returns a model when one is truly
/// unserved — it won't promote just to add redundancy.
async fn check_unserved_model(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;

    let mut mesh_requested: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &peers {
        for m in &p.requested_models {
            mesh_requested.insert(m.clone());
        }
    }

    if mesh_requested.is_empty() { return None; }

    let mut serving_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &peers {
        if let Some(ref s) = p.serving {
            *serving_count.entry(s.clone()).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    // Priority 1: promote for models with ZERO servers
    for m in &mesh_requested {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            let model_path = download::models_dir().join(format!("{}.gguf", m));
            let model_bytes = std::fs::metadata(&model_path).map(|md| md.len()).unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            return Some(m.clone());
        }
    }

    // Priority 2: demand-based rebalancing
    // Aggregate request rates across all peers for each model
    let mut total_demand: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for p in &peers {
        for (model, &rate) in &p.request_rates {
            *total_demand.entry(model.clone()).or_default() += rate;
        }
    }
    // Add our own rates
    let my_rates = node.snapshot_request_rates();
    for (model, rate) in &my_rates {
        *total_demand.entry(model.clone()).or_default() += rate;
    }

    // Find the model with the worst demand/server ratio.
    // Promote if: a model we can serve has significantly higher demand per server
    // than others, OR is hot enough on its own (≥10 req/min per server).
    if !total_demand.is_empty() {
        let mut ratios: Vec<(String, f64)> = Vec::new();
        for m in &mesh_requested {
            let demand = *total_demand.get(m).unwrap_or(&0) as f64;
            let servers = serving_count.get(m).copied().unwrap_or(0) as f64;
            if servers > 0.0 && local_models.contains(m) {
                let model_path = download::models_dir().join(format!("{}.gguf", m));
                let model_bytes = std::fs::metadata(&model_path).map(|md| md.len()).unwrap_or(0);
                let needed = (model_bytes as f64 * 1.1) as u64;
                if model_bytes > 0 && needed > my_vram {
                    continue;
                }
                ratios.push((m.clone(), demand / servers));
            }
        }

        if !ratios.is_empty() {
            ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let (hottest_model, hottest_ratio) = &ratios[0];

            // Two cases for promotion:
            // 1. Multiple models with demand: hottest is ≥3x coldest (and ≥10 req/min)
            // 2. Single hot model: ≥10 req/min per server with no other models getting traffic
            let coldest_ratio = if ratios.len() >= 2 { ratios[ratios.len() - 1].1 } else { 0.0 };
            let should_promote = if ratios.len() >= 2 {
                *hottest_ratio >= coldest_ratio * 3.0 && *hottest_ratio >= 10.0
            } else {
                // Only one model has demand — promote if it's clearly hot
                // and there's at least one model with 0 demand we could serve instead
                // (otherwise adding capacity to the only active model is always right)
                *hottest_ratio >= 10.0
            };

            if should_promote {
                eprintln!("📋 Promoting to serve {} — demand {:.0} req/min/server (coldest: {:.0})",
                    hottest_model, hottest_ratio, coldest_ratio);
                return Some(hottest_model.clone());
            }
        }
    }

    None
}

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
async fn run_auto(mut cli: Cli, resolved_models: Vec<PathBuf>, requested_model_names: Vec<String>, bin_dir: PathBuf) -> Result<()> {
    let api_port = cli.port;
    let console_port = cli.console;

    // Scan local models on disk
    let local_models = mesh::scan_local_models();
    eprintln!("Local models on disk: {:?}", local_models);

    // Start mesh node
    let (node, channels) = mesh::Node::start(NodeRole::Worker, &cli.relay, cli.bind_port, cli.max_vram).await?;
    let token = node.invite_token();

    // Advertise what we have on disk and what we want the mesh to serve
    node.set_available_models(local_models.clone()).await;
    node.set_requested_models(requested_model_names.clone()).await;

    // Start periodic health check to detect dead peers
    node.start_heartbeat();

    // Join mesh if --join was given
    if !cli.join.is_empty() {
        let mut joined = false;
        for t in &cli.join {
            match node.join(t).await {
                Ok(()) => {
                    eprintln!("Joined mesh");
                    joined = true;
                    break;
                }
                Err(e) => tracing::warn!("Failed to join via token: {e}"),
            }
        }
        if !joined {
            eprintln!("Failed to join any peer — running standalone");
        }

        // Save mesh_id for sticky preference after gossip propagates it
        {
            let save_node = node.clone();
            tokio::spawn(async move {
                // Wait for gossip to propagate mesh_id
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if let Some(id) = save_node.mesh_id().await {
                    mesh::save_last_mesh_id(&id);
                    eprintln!("📌 Mesh ID: {id}");
                }
            });
        }

        eprintln!("This node's token (for others to join): {token}");

        // Periodic rejoin: re-connect to bootstrap tokens every 60s.
        // No-op if already connected (connect_to_peer returns early).
        // Recovers from dropped connections without manual intervention.
        let rejoin_node = node.clone();
        let rejoin_tokens: Vec<String> = cli.join.clone();
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
    } else {
        // Originator — generate mesh_id
        let nostr_pubkey = if cli.publish {
            nostr::load_or_create_keys().ok().map(|k| k.public_key().to_hex())
        } else {
            None
        };
        let mesh_id = mesh::generate_mesh_id(cli.mesh_name.as_deref(), nostr_pubkey.as_deref());
        node.set_mesh_id_force(mesh_id.clone()).await;
        mesh::save_last_mesh_id(&mesh_id);
        eprintln!("📌 Mesh ID: {mesh_id}");
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for peers to join...");
    }

    // Start bootstrap proxy if joining an existing mesh.
    // This gives instant API access via tunnel while our GPU loads.
    let mut bootstrap_listener_tx = if !cli.join.is_empty() {
        let (stop_tx, stop_rx) = tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
        let boot_node = node.clone();
        let boot_port = api_port;
        tokio::spawn(async move {
            bootstrap_proxy(boot_node, boot_port, stop_rx).await;
        });
        Some(stop_tx)
    } else {
        None
    };

    // Decide which model THIS node will serve
    let model = if !resolved_models.is_empty() {
        // We have explicit --model(s). If only one, serve it.
        // If multiple, we seeded the mesh catalog but only serve one.
        if resolved_models.len() == 1 {
            resolved_models[0].clone()
        } else {
            // Multiple models seeded — pick based on allocation
            // Give gossip time to propagate so we see what peers are serving
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let assignment = pick_model_assignment(&node, &local_models).await;
            match assignment {
                Some(name) => {
                    // Find the resolved path for this model
                    resolved_models.iter()
                        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(&name))
                        .cloned()
                        .unwrap_or_else(|| {
                            // Not in our resolved list but we have it on disk
                            download::models_dir().join(format!("{}.gguf", name))
                        })
                }
                None => {
                    // All models covered — go standby
                    // Stop bootstrap proxy first (run_passive binds its own listener)
                    drop(bootstrap_listener_tx.take());
                    // Small delay for port to be released
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    eprintln!("💤 All models served — running as standby GPU node");
                    eprintln!("   VRAM: {:.1}GB, models on disk: {:?}", node.vram_bytes() as f64 / 1e9, &local_models);
                    match run_passive(&cli, node.clone(), false).await? {
                        Some(model_name) => {
                            resolved_models.iter()
                                .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(&model_name))
                                .cloned()
                                .unwrap_or_else(|| download::models_dir().join(format!("{}.gguf", model_name)))
                        }
                        None => return Ok(()), // clean shutdown
                    }
                }
            }
        }
    } else {
        // No --model: try to find a model on disk that the mesh needs
        eprintln!("No --model specified, checking local models against mesh...");

        // Give gossip a moment to propagate
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let assignment = pick_model_assignment(&node, &local_models).await;
        if let Some(model_name) = assignment {
            eprintln!("Mesh assigned model: {model_name}");
            let model_path = download::models_dir().join(format!("{}.gguf", model_name));
            if model_path.exists() {
                model_path
            } else {
                // Check other common extensions
                let alt = download::models_dir().join(&model_name);
                if alt.exists() { alt } else { model_path }
            }
        } else {
            // Nothing on disk matches — go passive, act as proxy
            // Stop bootstrap proxy first (run_passive binds its own listener)
            drop(bootstrap_listener_tx.take());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            // If a model becomes unserved while we're standby, we'll promote
            eprintln!("💤 No matching model on disk — running as standby GPU node");
            eprintln!("   VRAM: {:.1}GB, models on disk: {:?}", node.vram_bytes() as f64 / 1e9, local_models);
            eprintln!("   Proxying requests to other nodes. Will activate when needed.");
            match run_passive(&cli, node.clone(), false).await? {
                Some(model_name) => {
                    // Promoted! Resolve the model path and continue to serving
                    let model_path = download::models_dir().join(format!("{}.gguf", model_name));
                    if model_path.exists() {
                        model_path
                    } else {
                        let alt = download::models_dir().join(&model_name);
                        if alt.exists() { alt } else { model_path }
                    }
                }
                None => return Ok(()), // clean shutdown
            }
        }
    };

    let model_name = model.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Set model source for gossip (so other joiners can discover it too)
    let model_source = if !cli.model.is_empty() {
        cli.model[0].to_string_lossy().to_string()
    } else {
        model_name.clone()
    };
    node.set_model_source(model_source).await;
    node.set_serving(Some(model_name.clone())).await;
    node.set_models(vec![model_name.clone()]).await;
    // Re-gossip so peers learn what we're serving
    node.regossip().await;

    // Ensure draft model is available (downloads if needed, <1GB)
    if cli.draft.is_none() && !cli.no_draft {
        if let Some(draft_path) = ensure_draft(&model).await {
            eprintln!("Auto-detected draft model: {}", draft_path.display());
            cli.draft = Some(draft_path);
        }
    }

    // Start rpc-server
    let rpc_port = launch::start_rpc_server(
        &bin_dir, cli.device.as_deref(), Some(&model),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port} serving {model_name}");

    let tunnel_mgr = tunnel::Manager::start(
        node.clone(), rpc_port, channels.rpc, channels.http,
    ).await?;

    // Election publishes per-model targets
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());

    // Drop channel: API proxy sends model names to drop, main loop handles shutdown
    let (drop_tx, mut drop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Take over listener from bootstrap proxy (if running), or bind a new one
    let existing_listener = if let Some(tx) = bootstrap_listener_tx {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(resp_tx).await;
        // Wait for bootstrap to hand back the TcpListener
        resp_rx.await.ok()
    } else {
        None
    };

    // API proxy: model-aware routing
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    tokio::spawn(async move {
        api_proxy(proxy_node, api_port, proxy_rx, drop_tx, existing_listener).await;
    });

    // Console (optional)
    let model_name_for_console = model_name.clone();
    let console_state = if let Some(cport) = console_port {
        let model_size_bytes = election::total_model_bytes(&model);
        let cs = console::ConsoleState::new(node.clone(), model_name_for_console.clone(), api_port, model_size_bytes);
        if let Some(draft) = &cli.draft {
            let dn = draft.file_stem().unwrap_or_default().to_string_lossy().to_string();
            cs.set_draft_name(dn).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        let mn = model_name_for_console.clone();
        tokio::spawn(async move {
            // Console still takes old-style InferenceTarget for now — adapt
            let (adapted_tx, adapted_rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
            tokio::spawn(async move {
                let mut rx = console_rx;
                loop {
                    let targets = rx.borrow().clone();
                    let target = targets.get(&mn);
                    adapted_tx.send_replace(target);
                    if rx.changed().await.is_err() { break; }
                }
            });
            console::start(cport, cs2, adapted_rx).await;
        });
        Some(cs)
    } else {
        None
    };

    // Election loop
    eprintln!("Entering auto-election for model: {model_name}");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    let force_split = cli.split;
    let model_name_for_cb = model_name.clone();
    let model_name_for_election = model_name.clone();
    tokio::spawn(async move {
        election::election_loop(
            node2, tunnel_mgr2, rpc_port, bin_dir2, model2, model_name_for_election,
            draft2, draft_max, force_split, target_tx,
            move |is_host, llama_ready| {
                if is_host && llama_ready {
                    let url = format!("http://localhost:{api_port}");
                    eprintln!("  API: {url}");
                    update_pi_models_json(&model_name_for_cb, api_port);
                    eprintln!();
                    eprintln!("  pi:    pi --provider mesh --model {model_name_for_cb}");
                    eprintln!("  goose: GOOSE_PROVIDER=openai OPENAI_HOST={url} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name_for_cb} goose session");
                } else if is_host {
                    eprintln!("⏳ Starting llama-server...");
                } else {
                    eprintln!("  API: http://localhost:{api_port} (proxied to host)");
                }
                if let Some(ref cs) = console_state {
                    let cs = cs.clone();
                    tokio::spawn(async move {
                        cs.update(is_host, llama_ready).await;
                    });
                }
            },
        ).await;
    });

    // Nostr publish loop (if --publish) or watchdog (if --auto, to take over if publisher dies)
    let nostr_publisher = if cli.publish {
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_node = node.clone();
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        Some(tokio::spawn(async move {
            nostr::publish_loop(pub_node, nostr_keys, relays, pub_name, pub_region, pub_max_clients, 60).await;
        }))
    } else if cli.auto {
        // Watchdog: if we joined via --auto, watch for the publisher to die and take over
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        Some(tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120).await;
        }))
    } else {
        None
    };

    // Wait for ctrl-c or a drop command for our model
    let drop_model_name = model_name.clone();
    let drop_node = node.clone();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\nShutting down...");
        }
        dropped = async {
            while let Some(name) = drop_rx.recv().await {
                if name == drop_model_name {
                    return name;
                }
                eprintln!("⚠ Drop request for '{}' — not our model ({}), ignoring", name, drop_model_name);
            }
            drop_model_name.clone() // channel closed
        } => {
            eprintln!("\n🗑 Model '{}' dropped from mesh — shutting down", dropped);
            drop_node.set_serving(None).await;
        }
    }

    // Announce clean departure to peers
    node.broadcast_leaving().await;

    // Clean up Nostr listing on shutdown
    if cli.publish {
        if let Ok(keys) = nostr::load_or_create_keys() {
            let relays = nostr_relays(&cli.nostr_relay);
            if let Ok(publisher) = nostr::Publisher::new(keys, &relays).await {
                let _ = publisher.unpublish().await;
                eprintln!("Removed Nostr listing");
            }
        }
    }
    if let Some(handle) = nostr_publisher {
        handle.abort();
    }

    launch::kill_llama_server().await;
    Ok(())
}

/// Run in passive mode: proxy requests to active hosts, no local llama-server.
/// Used by both --client (pure consumer) and idle GPU nodes (standby, no matching model).
/// If `create_node` is true, creates a new Node (--client path). Otherwise reuses existing.
/// Run as passive node (client or standby GPU).
/// Returns Ok(Some(model_name)) if a standby GPU should promote to serve a model.
/// Returns Ok(None) on clean shutdown.
async fn run_passive(cli: &Cli, node: mesh::Node, is_client: bool) -> Result<Option<String>> {
    let local_port = cli.port;

    // Nostr publishing (if --publish, for idle GPU nodes advertising capacity)
    if cli.publish && !is_client {
        let pub_node = node.clone();
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        tokio::spawn(async move {
            nostr::publish_loop(pub_node, nostr_keys, relays, pub_name, pub_region, pub_max_clients, 60).await;
        });
    } else if cli.auto && !is_client {
        // Watchdog: take over publishing if the original publisher dies
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120).await;
        });
    }

    // Wait briefly for gossip to propagate
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let served = node.models_being_served().await;
    if !served.is_empty() {
        eprintln!("Models available in mesh: {:?}", served);
    }

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{local_port}")).await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    let mode = if is_client { "client" } else { "standby" };
    eprintln!("Passive {mode} ready: http://localhost:{local_port}");
    eprintln!("  Requests routed to active hosts via QUIC tunnel");

    // Console (optional)
    if let Some(cport) = cli.console {
        let label = if is_client { "(client)".to_string() } else { "(standby)".to_string() };
        let cs = console::ConsoleState::new(node.clone(), label, local_port, 0);
        if is_client { cs.set_client(true).await; }
        cs.update(false, !is_client).await;
        let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
        tokio::spawn(async move {
            console::start(cport, cs, rx).await;
        });
    }

    // Reactive rebalancing: watch for topology changes and promote if needed.
    // Only for standby GPU nodes (not clients — they never serve).
    let (promote_tx, mut promote_rx) = tokio::sync::mpsc::channel::<String>(1);
    if !is_client {
        let watch_node = node.clone();
        let mut peer_rx = node.peer_change_rx.clone();
        let local_models = mesh::scan_local_models();
        tokio::spawn(async move {
            // Wait for initial mesh settle
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            // Periodic demand check interval (aligned with gossip cycle)
            let mut demand_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            demand_interval.tick().await; // consume first immediate tick
            loop {
                // Wait for EITHER a topology change OR periodic demand check
                tokio::select! {
                    res = peer_rx.changed() => {
                        if res.is_err() { break; }
                        // Debounce — multiple changes can fire in quick succession
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        // Drain any queued changes
                        while peer_rx.has_changed().unwrap_or(false) {
                            let _ = peer_rx.borrow_and_update();
                        }
                    }
                    _ = demand_interval.tick() => {
                        // Periodic check for demand-based rebalancing
                    }
                }
                // Check if there's an unserved or demand-imbalanced model we can handle
                if let Some(model_name) = check_unserved_model(&watch_node, &local_models).await {
                    eprintln!("🚀 Promoting from standby — serving {model_name}");
                    let _ = promote_tx.send(model_name).await;
                    break;
                }
            }
        });
    }

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Connection from {addr}");
                let node = node.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 32768];
                    let (n, model_name) = match peek_request(&tcp_stream, &mut buf).await {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                    // Handle /v1/models locally
                    if is_models_list_request(&buf[..n]) {
                        let served = node.models_being_served().await;
                        let _ = send_models_list(tcp_stream, &served).await;
                        return;
                    }

                    // Record request for demand tracking
                    if let Some(ref name) = model_name {
                        node.record_request(name);
                    }

                    // Route to host by model name (hash-based selection)
                    let target_host = if let Some(ref name) = model_name {
                        node.host_for_model(name).await.map(|p| p.id)
                    } else {
                        None
                    };
                    // Fall back to any host in the mesh
                    let target_host = match target_host {
                        Some(id) => id,
                        None => match node.any_host().await {
                            Some(p) => p.id,
                            None => {
                                let _ = send_503(tcp_stream).await;
                                return;
                            }
                        }
                    };

                    match node.open_http_tunnel(target_host).await {
                        Ok((quic_send, quic_recv)) => {
                            if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                                tracing::debug!("HTTP tunnel relay ended: {e}");
                            }
                        }
                        Err(e) => tracing::warn!("Failed to open HTTP tunnel to host: {e}"),
                    }
                });
            }
            Some(model_name) = promote_rx.recv() => {
                eprintln!("⬆️  Standby promoting to serve: {model_name}");
                return Ok(Some(model_name));
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nShutting down...");
                node.broadcast_leaving().await;
                return Ok(None);
            }
        }
    }

    Ok(None)
}

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
async fn api_proxy(node: mesh::Node, port: u16, target_rx: tokio::sync::watch::Receiver<election::ModelTargets>, drop_tx: tokio::sync::mpsc::UnboundedSender<String>, existing_listener: Option<tokio::net::TcpListener>) {
    let listener = match existing_listener {
        Some(l) => l,
        None => match tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Failed to bind API proxy to port {port}: {e}");
                return;
            }
        },
    };

    loop {
        let (tcp_stream, _addr) = match listener.accept().await {
            Ok(r) => r,
            Err(_) => break,
        };
        let _ = tcp_stream.set_nodelay(true);

        let targets = target_rx.borrow().clone();
        let node = node.clone();

        let drop_tx = drop_tx.clone();
        tokio::spawn(async move {
            // Read the HTTP request to extract the model name
            let mut buf = vec![0u8; 32768];
            match peek_request(&tcp_stream, &mut buf).await {
                Ok((n, model_name)) => {
                    // Handle /v1/models ourselves if it's a GET
                    if is_models_list_request(&buf[..n]) {
                        let models: Vec<String> = targets.targets.keys().cloned().collect();
                        let _ = send_models_list(tcp_stream, &models).await;
                        return;
                    }

                    // Handle /mesh/drop control endpoint
                    if is_drop_request(&buf[..n]) {
                        if let Some(ref name) = model_name {
                            let _ = drop_tx.send(name.clone());
                            let _ = send_json_ok(tcp_stream, &serde_json::json!({"dropped": name})).await;
                        } else {
                            let _ = send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    // Record request for demand tracking
                    if let Some(ref name) = model_name {
                        node.record_request(name);
                    }

                    let target = if let Some(ref name) = model_name {
                        let t = targets.get(name);
                        if matches!(t, election::InferenceTarget::None) {
                            tracing::debug!("Model '{}' not found, trying first available", name);
                            first_available_target(&targets)
                        } else {
                            t
                        }
                    } else {
                        first_available_target(&targets)
                    };

                    route_request(node, tcp_stream, target).await;
                }
                Err(_) => return,
            };
        });
    }
}

/// Bootstrap proxy: runs during GPU startup, tunnels all requests to mesh hosts.
/// Returns the TcpListener when signaled to stop (so api_proxy can take it over).
async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    mut stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
) {
    let listener = match tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Bootstrap proxy: failed to bind to port {port}: {e}");
            return;
        }
    };
    eprintln!("⚡ API ready (bootstrap): http://localhost:{port}");
    eprintln!("  Requests tunneled to mesh while GPU loads...");

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (tcp_stream, _addr) = match accept {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let _ = tcp_stream.set_nodelay(true);
                let node = node.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 32768];
                    let (n, model_name) = match peek_request(&tcp_stream, &mut buf).await {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                    // Handle /v1/models locally
                    if is_models_list_request(&buf[..n]) {
                        let served = node.models_being_served().await;
                        let _ = send_models_list(tcp_stream, &served).await;
                        return;
                    }

                    // Record request for demand tracking
                    if let Some(ref name) = model_name {
                        node.record_request(name);
                    }

                    // Route to host by model name
                    let target_host = if let Some(ref name) = model_name {
                        node.host_for_model(name).await.map(|p| p.id)
                    } else {
                        None
                    };
                    let target_host = match target_host {
                        Some(id) => id,
                        None => match node.any_host().await {
                            Some(p) => p.id,
                            None => {
                                let _ = send_503(tcp_stream).await;
                                return;
                            }
                        }
                    };

                    match node.open_http_tunnel(target_host).await {
                        Ok((quic_send, quic_recv)) => {
                            if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                                tracing::debug!("Bootstrap tunnel relay ended: {e}");
                            }
                        }
                        Err(e) => tracing::warn!("Bootstrap: failed to tunnel to host: {e}"),
                    }
                });
            }
            resp_tx = stop_rx.recv() => {
                // Hand over listener to api_proxy
                if let Some(tx) = resp_tx {
                    eprintln!("⚡ Bootstrap proxy handing off to full API proxy");
                    let _ = tx.send(listener);
                }
                return;
            }
        }
    }
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for target in targets.targets.values() {
        if !matches!(target, election::InferenceTarget::None) {
            return target.clone();
        }
    }
    election::InferenceTarget::None
}

fn is_drop_request(buf: &[u8]) -> bool {
    let s = String::from_utf8_lossy(buf);
    s.starts_with("POST ") && s.contains("/mesh/drop")
}

fn is_models_list_request(buf: &[u8]) -> bool {
    let s = String::from_utf8_lossy(buf);
    s.starts_with("GET ") && (s.contains("/v1/models") || s.contains("/models"))
        && !s.contains("/v1/models/")
}

/// Extract model name from a JSON POST body in an HTTP request.
fn extract_model_from_http(buf: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(buf).ok()?;
    // Find the JSON body (after \r\n\r\n)
    let body_start = s.find("\r\n\r\n")? + 4;
    let body = &s[body_start..];
    // Simple extraction — look for "model": "..."
    let model_key = "\"model\"";
    let pos = body.find(model_key)?;
    let after_key = &body[pos + model_key.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    // Extract quoted string
    let after_quote = after_ws.strip_prefix('"')?;
    let end = after_quote.find('"')?;
    Some(after_quote[..end].to_string())
}

/// Peek at the request without consuming it. Returns bytes read and optional model name.
async fn peek_request(stream: &tokio::net::TcpStream, buf: &mut [u8]) -> Result<(usize, Option<String>)> {
    let n = stream.peek(buf).await?;
    if n == 0 {
        anyhow::bail!("Empty request");
    }
    let model = extract_model_from_http(&buf[..n]);
    Ok((n, model))
}

async fn route_request(node: mesh::Node, tcp_stream: tokio::net::TcpStream, target: election::InferenceTarget) {
    match target {
        election::InferenceTarget::Local(llama_port) => {
            match tokio::net::TcpStream::connect(format!("127.0.0.1:{llama_port}")).await {
                Ok(upstream) => {
                    let _ = upstream.set_nodelay(true);
                    if let Err(e) = tunnel::relay_tcp_streams(tcp_stream, upstream).await {
                        tracing::debug!("API proxy (local) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't reach local llama-server on {llama_port}: {e}");
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::Remote(host_id) => {
            match node.open_http_tunnel(host_id).await {
                Ok((quic_send, quic_recv)) => {
                    if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                        tracing::debug!("API proxy (remote) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't tunnel to host {}: {e}", host_id.fmt_short());
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::None => {
            let _ = send_503(tcp_stream).await;
        }
    }
}

async fn send_models_list(mut stream: tokio::net::TcpStream, models: &[String]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let data: Vec<serde_json::Value> = models.iter().map(|m| {
        serde_json::json!({
            "id": m,
            "object": "model",
            "owned_by": "mesh-llm",
        })
    }).collect();
    let body = serde_json::json!({ "object": "list", "data": data }).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn send_json_ok(mut stream: tokio::net::TcpStream, data: &serde_json::Value) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let body = data.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn send_400(mut stream: tokio::net::TcpStream, msg: &str) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let body = format!("{{\"error\":\"{msg}\"}}");
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn send_503(mut stream: tokio::net::TcpStream) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let body = r#"{"error":"No inference server available — election in progress"}"#;
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn detect_bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe()
        .context("Failed to determine own binary path")?;
    let dir = exe.parent()
        .context("Binary has no parent directory")?;

    if dir.join("rpc-server").exists() && dir.join("llama-server").exists() {
        return Ok(dir.to_path_buf());
    }
    let dev = dir.join("../llama.cpp/build/bin");
    if dev.join("rpc-server").exists() && dev.join("llama-server").exists() {
        return Ok(dev.canonicalize()?);
    }
    let cargo = dir.join("../../../llama.cpp/build/bin");
    if cargo.join("rpc-server").exists() && cargo.join("llama-server").exists() {
        return Ok(cargo.canonicalize()?);
    }

    Ok(dir.to_path_buf())
}

/// Update ~/.pi/agent/models.json to include a "mesh" provider.
fn update_pi_models_json(model_id: &str, port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    let models_path = home.join(".pi/agent/models.json");

    let mut root: serde_json::Value = if models_path.exists() {
        match std::fs::read_to_string(&models_path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let providers = root.as_object_mut()
        .and_then(|r| {
            r.entry("providers").or_insert_with(|| serde_json::json!({}));
            r.get_mut("providers")?.as_object_mut()
        });
    let Some(providers) = providers else { return };

    let mesh = serde_json::json!({
        "baseUrl": format!("http://localhost:{port}/v1"),
        "api": "openai-completions",
        "apiKey": "mesh",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "compat": {
                "supportsUsageInStreaming": false,
                "maxTokensField": "max_tokens",
                "supportsDeveloperRole": false
            }
        }]
    });

    providers.insert("mesh".to_string(), mesh);

    if let Some(parent) = models_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&root) {
        if let Err(e) = std::fs::write(&models_path, json) {
            tracing::warn!("Failed to update {}: {e}", models_path.display());
        }
    }
}

/// Resolve Nostr relay URLs from CLI or defaults.
/// Health probe: try QUIC connect to the mesh's bootstrap node.
/// Returns Ok if reachable within 10s, Err if not.
async fn probe_mesh_health(invite_token: &str) -> Result<()> {
    use base64::Engine;
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
    let addr: iroh::EndpointAddr = serde_json::from_slice(&json)?;

    let key = iroh::SecretKey::generate(&mut rand::rng());
    let ep = iroh::Endpoint::builder()
        .secret_key(key)
        .bind().await?;

    match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        ep.connect(addr, mesh::ALPN),
    ).await {
        Ok(Ok(_conn)) => {
            ep.close().await;
            Ok(())
        }
        Ok(Err(e)) => {
            ep.close().await;
            anyhow::bail!("Connection failed: {e}")
        }
        Err(_) => {
            ep.close().await;
            anyhow::bail!("Connection timed out (10s)")
        }
    }
}

/// Helper for StartNew path — configure CLI to start a new mesh.
fn start_new_mesh(cli: &mut Cli, models: &[String], my_vram_gb: f64, my_region: &Option<String>) {
    eprintln!("🆕 Starting a new mesh");
    eprintln!("   Primary model: {}", models[0]);
    if models.len() > 1 {
        eprintln!("   Also declaring: {:?}", &models[1..]);
    }
    eprintln!("   VRAM: {:.0}GB", my_vram_gb);
    if cli.model.is_empty() {
        for m in models {
            cli.model.push(m.into());
        }
    }
    if !cli.publish {
        cli.publish = true;
        eprintln!("   Auto-enabling --publish for discovery");
    }
    if cli.region.is_none() {
        if let Some(r) = my_region {
            cli.region = Some(r.clone());
            eprintln!("   Region: {r} (auto-detected)");
        }
    }
}

fn nostr_relays(cli_relays: &[String]) -> Vec<String> {
    if cli_relays.is_empty() {
        nostr::DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect()
    } else {
        cli_relays.to_vec()
    }
}

/// Discover meshes on Nostr and optionally join one.
async fn run_discover(
    model: Option<String>,
    min_vram: Option<f64>,
    region: Option<String>,
    auto_join: bool,
    relays: Vec<String>,
) -> Result<()> {
    let relays = if relays.is_empty() {
        nostr::DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect()
    } else {
        relays
    };

    let filter = nostr::MeshFilter {
        model,
        min_vram_gb: min_vram,
        region,
    };

    eprintln!("🔍 Searching Nostr relays for mesh-llm meshes...");
    let meshes = nostr::discover(&relays, &filter).await?;

    if meshes.is_empty() {
        eprintln!("No meshes found.");
        if filter.model.is_some() || filter.min_vram_gb.is_some() || filter.region.is_some() {
            eprintln!("Try broader filters or check relays.");
        }
        return Ok(());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = mesh::load_last_mesh_id();
    eprintln!("Found {} mesh(es):\n", meshes.len());
    for (i, mesh) in meshes.iter().enumerate() {
        let score = nostr::score_mesh(mesh, filter.region.as_deref(), now, last_mesh_id.as_deref());
        let age = now.saturating_sub(mesh.published_at);
        let freshness = if age < 120 { "fresh" } else if age < 300 { "ok" } else { "stale" };
        let capacity = if mesh.listing.max_clients > 0 {
            format!("{}/{} clients", mesh.listing.client_count, mesh.listing.max_clients)
        } else {
            format!("{} clients", mesh.listing.client_count)
        };
        eprintln!("  [{}] {} (score: {}, {}, {})", i + 1, mesh, score, freshness, capacity);
        let token = &mesh.listing.invite_token;
        let display_token = if token.len() > 40 {
            format!("{}...{}", &token[..20], &token[token.len()-12..])
        } else {
            token.clone()
        };
        if !mesh.listing.on_disk.is_empty() {
            eprintln!("      on disk: {}", mesh.listing.on_disk.join(", "));
        }
        eprintln!("      token: {}", display_token);
        eprintln!();
    }

    if auto_join {
        let best = &meshes[0];
        eprintln!("Auto-joining best match: {}", best);
        eprintln!("\nRun:");
        eprintln!("  mesh-llm --join {}", best.listing.invite_token);
        // Print the full token so it can be piped
        println!("{}", best.listing.invite_token);
    } else {
        eprintln!("To join a mesh:");
        eprintln!("  mesh-llm --join <token>");
        eprintln!("\nOr use `mesh-llm discover --join` to auto-join the best match.");
    }

    Ok(())
}

/// Drop a model from the mesh by sending a control request to the running instance.
async fn run_drop(model_name: &str, port: u16) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let body = serde_json::json!({ "model": model_name }).to_string();
    let request = format!(
        "POST /mesh/drop HTTP/1.1\r\nHost: localhost:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")).await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    stream.write_all(request.as_bytes()).await?;

    let mut response = vec![0u8; 4096];
    let n = stream.read(&mut response).await?;
    let resp = String::from_utf8_lossy(&response[..n]);

    if resp.contains("200 OK") {
        eprintln!("✅ Dropped model: {model_name}");
    } else {
        eprintln!("❌ Failed to drop model: {}", resp.lines().last().unwrap_or("unknown error"));
    }

    Ok(())
}
