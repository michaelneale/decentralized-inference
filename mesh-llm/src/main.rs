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
        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: None,
        };
        let meshes = nostr::discover(&relays, &filter).await?;
        if let Some(best) = meshes.first() {
            eprintln!("Found mesh: {} ({} nodes, {} models)",
                best.listing.name.as_deref().unwrap_or("unnamed"),
                best.listing.node_count,
                best.listing.models.len());
            cli.join.push(best.listing.invite_token.clone());
        } else {
            anyhow::bail!("No meshes found via Nostr. Try --join <token> instead.");
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

        return run_passive(&cli, node, true).await;
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
pub fn auto_detect_draft(model: &std::path::Path) -> Option<PathBuf> {
    let filename = model.file_name()?.to_str()?;
    let catalog_entry = download::MODEL_CATALOG.iter().find(|m| m.file == filename)?;
    let draft_name = catalog_entry.draft?;
    let draft_entry = download::MODEL_CATALOG.iter().find(|m| m.name == draft_name)?;
    let draft_path = download::models_dir().join(draft_entry.file);
    if draft_path.exists() {
        Some(draft_path)
    } else {
        None
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

    // Collect what the mesh wants served (from requested_models across all peers)
    let mut mesh_requested: std::collections::HashSet<String> = std::collections::HashSet::new();
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

    // Priority 1: models the mesh wants that nobody is serving, and we have on disk
    for m in &mesh_requested {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            eprintln!("📋 Assigned to serve {} (needed by mesh, already on disk)", m);
            return Some(m.clone());
        }
    }

    // Priority 2: least-served model that we have on disk
    let mut candidates: Vec<(&String, usize)> = mesh_requested.iter()
        .filter(|m| local_models.contains(m))
        .map(|m| (m, serving_count.get(m).copied().unwrap_or(0)))
        .collect();
    candidates.sort_by_key(|(_m, count)| *count);
    if let Some((m, _)) = candidates.first() {
        eprintln!("📋 Assigned to serve {} (least served model on disk)", m);
        return Some((*m).clone());
    }

    // Priority 3: any model we have on disk that the mesh wants
    for m in &mesh_requested {
        if local_models.contains(m) {
            eprintln!("📋 Assigned to serve {} (on disk)", m);
            return Some(m.clone());
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
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for peers to join...");
    }

    // Decide which model THIS node will serve
    let model = if !resolved_models.is_empty() {
        // We have explicit --model(s). If only one, serve it.
        // If multiple, we seeded the mesh catalog but only serve one.
        if resolved_models.len() == 1 {
            resolved_models[0].clone()
        } else {
            // Multiple models seeded — pick based on allocation
            // First, give gossip a moment to propagate
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
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
                None => resolved_models[0].clone(), // fallback to first
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
            eprintln!("💤 No matching model on disk — running as standby GPU node");
            eprintln!("   VRAM: {:.1}GB, models on disk: {:?}", node.vram_bytes() as f64 / 1e9, local_models);
            eprintln!("   Proxying requests to other nodes. Will activate when needed.");
            return run_passive(&cli, node, false).await;
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

    // Auto-detect draft model
    if cli.draft.is_none() && !cli.no_draft {
        if let Some(draft_path) = auto_detect_draft(&model) {
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

    // API proxy: model-aware routing
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    tokio::spawn(async move {
        api_proxy(proxy_node, api_port, proxy_rx, drop_tx).await;
    });

    // Console (optional)
    let model_name_for_console = model_name.clone();
    let console_state = if let Some(cport) = console_port {
        let cs = console::ConsoleState::new(node.clone(), model_name_for_console.clone(), api_port);
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

    // Nostr publish loop (if --publish)
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
async fn run_passive(cli: &Cli, node: mesh::Node, is_client: bool) -> Result<()> {
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
        let cs = console::ConsoleState::new(node.clone(), label, local_port);
        if is_client { cs.set_client(true).await; }
        cs.update(false, !is_client).await;
        let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
        tokio::spawn(async move {
            console::start(cport, cs, rx).await;
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
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nShutting down...");
                node.broadcast_leaving().await;
                break;
            }
        }
    }

    Ok(())
}

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
async fn api_proxy(node: mesh::Node, port: u16, target_rx: tokio::sync::watch::Receiver<election::ModelTargets>, drop_tx: tokio::sync::mpsc::UnboundedSender<String>) {
    let listener = match tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind API proxy to port {port}: {e}");
            return;
        }
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

    eprintln!("Found {} mesh(es):\n", meshes.len());
    for (i, mesh) in meshes.iter().enumerate() {
        eprintln!("  [{}] {}", i + 1, mesh);
        let token = &mesh.listing.invite_token;
        let display_token = if token.len() > 40 {
            format!("{}...{}", &token[..20], &token[token.len()-12..])
        } else {
            token.clone()
        };
        if !mesh.listing.available_models.is_empty() {
            eprintln!("      on disk: {}", mesh.listing.available_models.join(", "));
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
