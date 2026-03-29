mod api;
mod download;
mod election;
mod hardware;
mod launch;
mod mesh;
mod moe;
mod nostr;
mod pipeline;
mod plugin;
mod plugin_mcp;
mod plugins;
mod proxy;
mod rewrite;
mod router;
mod tunnel;

pub use plugins::blackboard;
pub use plugins::blackboard::mcp as blackboard_mcp;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mesh::NodeRole;
use std::path::PathBuf;

pub const VERSION: &str = "0.49.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Llama,
    Mlx,
}

#[derive(Parser, Debug)]
#[command(name = "mesh-llm", version = VERSION,
    about = "Pool GPUs over the internet for LLM inference",
    after_help = "Run with --help-advanced for all options.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Show all options (including advanced/niche ones).
    #[arg(long, hide = true)]
    help_advanced: bool,

    /// Join a mesh via invite token (can repeat).
    #[arg(long, short)]
    join: Vec<String>,

    /// Discover a mesh via Nostr and join it.
    #[arg(long, default_missing_value = "", num_args = 0..=1)]
    discover: Option<String>,

    /// Auto-join the best mesh found via Nostr.
    #[arg(long)]
    auto: bool,

    /// Model to serve (path, catalog name, or HuggingFace URL).
    #[arg(long)]
    model: Vec<PathBuf>,

    /// Inference backend to use.
    #[arg(long, value_enum, default_value_t = Backend::Llama)]
    backend: Backend,

    /// API port (default: 9337).
    #[arg(long, default_value = "9337")]
    port: u16,

    /// Run as a client — no GPU, no model needed.
    #[arg(long)]
    client: bool,

    /// Web console port (default: 3131).
    #[arg(long, default_value = "3131")]
    console: u16,

    /// Publish this mesh for discovery by others.
    #[arg(long)]
    publish: bool,

    /// Name for this mesh (shown in discovery).
    #[arg(long)]
    mesh_name: Option<String>,

    /// Region tag, e.g. "US", "EU", "AU" (shown in discovery).
    #[arg(long)]
    region: Option<String>,

    /// Enable blackboard on public meshes (on by default for private meshes).
    #[arg(long)]
    blackboard: bool,

    /// Your display name on the blackboard.
    #[arg(long)]
    name: Option<String>,

    /// Internal plugin service mode.
    #[arg(long, hide = true)]
    plugin: Option<String>,

    /// Override the native MLX worker executable used by --backend mlx.
    #[arg(long, hide = true)]
    mlx_native_bin: Option<PathBuf>,

    // ── Advanced options (hidden from default --help) ─────────────
    /// Draft model for speculative decoding.
    #[arg(long, hide = true)]
    draft: Option<PathBuf>,

    /// Max draft tokens (default: 8).
    #[arg(long, default_value = "8", hide = true)]
    draft_max: u16,

    /// Disable automatic draft model detection.
    #[arg(long, hide = true)]
    no_draft: bool,

    /// Force tensor split even if the model fits on one node.
    #[arg(long, hide = true)]
    split: bool,

    /// Override context size (tokens). Default: auto-scaled to available VRAM.
    #[arg(long, hide = true)]
    ctx_size: Option<u32>,

    /// Limit VRAM advertised to the mesh (GB).
    #[arg(long, hide = true)]
    max_vram: Option<f64>,

    /// Enumerate host hardware (GPU name, hostname) at startup.
    #[arg(long, hide = true)]
    enumerate_host: bool,

    /// Path to rpc-server and llama-server binaries.
    #[arg(long, hide = true)]
    bin_dir: Option<PathBuf>,

    /// Device for rpc-server (e.g. MTL0, CPU).
    #[arg(long, hide = true)]
    device: Option<String>,

    /// Tensor split ratios for llama-server (e.g. "0.8,0.2").
    #[arg(long, hide = true)]
    tensor_split: Option<String>,

    /// Override iroh relay URLs.
    #[arg(long, hide = true)]
    relay: Vec<String>,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, hide = true)]
    bind_port: Option<u16>,

    /// Bind to 0.0.0.0 (for containers/Fly.io).
    #[arg(long, hide = true)]
    listen_all: bool,

    /// Stop advertising when N clients connected.
    #[arg(long, hide = true)]
    max_clients: Option<usize>,

    /// Custom Nostr relay URLs.
    #[arg(long, hide = true)]
    nostr_relay: Vec<String>,

    /// Ignored (backward compat).
    #[arg(long, hide = true)]
    no_console: bool,

    /// Optional path to the mesh-llm config file.
    #[arg(long, hide = true)]
    config: Option<PathBuf>,

    /// Internal: set when this node joined via Nostr discovery (not --join).
    #[arg(skip)]
    nostr_discovery: bool,
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
    /// Drop a model from the mesh.
    #[command(hide = true)]
    Drop {
        /// Model name to drop
        name: String,
        /// API port of the running mesh-llm instance (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Discover meshes on Nostr and optionally auto-join one.
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
    /// Rotate the Nostr identity key.
    #[command(hide = true)]
    RotateKey,
    /// Launch Goose with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "goose")]
    Goose {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch Claude Code with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "claude")]
    Claude {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Stop all running mesh-llm, llama-server, and rpc-server processes.
    Stop,
    /// Blackboard — post, search, and read messages shared across the mesh.
    ///
    /// Post a message:   mesh-llm blackboard "your message here"
    /// Show feed:        mesh-llm blackboard
    /// Search:           mesh-llm blackboard --search "query"
    /// From a peer:      mesh-llm blackboard --from tyler
    /// MCP server:       mesh-llm --client --join <token> blackboard --mcp
    /// Install skill:    mesh-llm blackboard install-skill
    ///
    /// Conventions: prefix messages with QUESTION:, STATUS:, FINDING:, TIP: etc.
    /// Search picks these up naturally via multi-term OR matching.
    #[command(name = "blackboard")]
    Blackboard {
        /// Message to post (if provided).
        text: Option<String>,
        /// Search the blackboard.
        #[arg(long)]
        search: Option<String>,
        /// Filter by author name.
        #[arg(long)]
        from: Option<String>,
        /// Only show items from the last N hours (default: 24).
        #[arg(long)]
        since: Option<f64>,
        /// Max items to show (default: 20).
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Console/API port of the running mesh-llm instance.
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Run as an MCP server over stdio (for agent integration).
        #[arg(long)]
        mcp: bool,
    },
    /// Plugin management.
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
}

#[derive(Subcommand, Debug)]
enum PluginCommand {
    /// Compatibility shim for the old install workflow.
    Install {
        /// Plugin name.
        name: String,
    },
    /// List auto-registered and configured plugins.
    List,
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

    // --help-advanced: print full help with all hidden options and commands visible
    if std::env::args().any(|a| a == "--help-advanced") {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        // Unhide all arguments
        let args: Vec<clap::Id> = cmd.get_arguments().map(|a| a.get_id().clone()).collect();
        for id in args {
            cmd = cmd.mut_arg(id, |a| a.hide(false));
        }
        // Unhide all subcommands
        let sub_names: Vec<String> = cmd
            .get_subcommands()
            .map(|s| s.get_name().to_string())
            .collect();
        for name in sub_names {
            cmd = cmd.mut_subcommand(name, |s| s.hide(false));
        }
        cmd.print_help().ok();
        eprintln!();
        std::process::exit(0);
    }

    let mut cli = Cli::parse();

    if let Some(name) = cli.plugin.clone() {
        return plugin::run_plugin_process(name).await;
    }

    // Clean up orphan processes from previous runs (skip for client — never runs llama-server)
    if !cli.client {
        match cli.backend {
            Backend::Llama => {
                launch::kill_llama_server().await;
                launch::kill_orphan_rpc_servers().await;
            }
            Backend::Mlx => {}
        }
    }

    // Background version check (non-blocking)
    tokio::spawn(async {
        check_for_update().await;
    });

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
                                let draft_model =
                                    download::find_model(draft_name).ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "Draft model '{}' not found in catalog",
                                            draft_name
                                        )
                                    })?;
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
            Command::Stop => {
                return run_stop();
            }
            Command::Discover {
                model,
                min_vram,
                region,
                auto,
                relay,
            } => {
                return run_discover(
                    model.clone(),
                    *min_vram,
                    region.clone(),
                    *auto,
                    relay.clone(),
                )
                .await;
            }
            Command::RotateKey => {
                return nostr::rotate_keys().map_err(Into::into);
            }
            Command::Goose { model, port } => {
                return run_goose(model.clone(), *port).await;
            }
            Command::Claude { model, port } => {
                return run_claude(model.clone(), *port).await;
            }
            Command::Blackboard {
                text,
                search,
                from,
                since,
                limit,
                port,
                mcp,
            } => {
                if *mcp {
                    return run_plugin_mcp(&cli).await;
                }
                if text.as_deref() == Some("install-skill") {
                    return install_skill();
                }
                return run_blackboard(
                    text.clone(),
                    search.clone(),
                    from.clone(),
                    *since,
                    *limit,
                    *port,
                )
                .await;
            }
            Command::Plugin { command } => {
                return run_plugin_command(command, &cli).await;
            }
        }
    }

    // Auto-enable publishing when mesh is named
    if cli.mesh_name.is_some() && !cli.publish {
        cli.publish = true;
    }

    // --- Auto-discover ---
    if cli.auto && cli.join.is_empty() {
        cli.nostr_discovery = true;
        eprintln!("🔍 Discovering meshes via Nostr...");

        let relays = nostr_relays(&cli.nostr_relay);
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
        let target_name = cli.mesh_name.as_deref();
        for m in &meshes {
            let score = nostr::score_mesh(m, now, last_mesh_id.as_deref());
            eprintln!(
                "  · {} (score: {}, {} nodes, {:.0}GB, {} clients{})",
                m.listing.name.as_deref().unwrap_or("unnamed"),
                score,
                m.listing.node_count,
                m.listing.total_vram_bytes as f64 / 1e9,
                m.listing.client_count,
                m.listing
                    .region
                    .as_ref()
                    .map(|r| format!(", {r}"))
                    .unwrap_or_default()
            );
        }

        match nostr::smart_auto(&meshes, my_vram_gb, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                if cli.client {
                    // Clients skip health probe — joining itself is the test.
                    // Use the best candidate.
                    let (token, mesh) = &candidates[0];
                    if cli.mesh_name.is_none() {
                        if let Some(ref name) = mesh.listing.name {
                            cli.mesh_name = Some(name.clone());
                        }
                    }
                    eprintln!(
                        "✅ Joining: {} ({} nodes, {} models{})",
                        mesh.listing.name.as_deref().unwrap_or("unnamed"),
                        mesh.listing.node_count,
                        mesh.listing.serving.len(),
                        mesh.listing
                            .region
                            .as_ref()
                            .map(|r| format!(", region: {r}"))
                            .unwrap_or_default()
                    );
                    cli.join.push(token.clone());
                } else {
                    // GPU nodes: try to join each candidate directly.
                    // No ephemeral probe — it fails when the target has a firewall
                    // even though the real join (via relay) would succeed.
                    let mut joined = false;
                    for (i, (token, mesh)) in candidates.iter().enumerate() {
                        eprintln!(
                            "  Trying mesh {}{}...",
                            mesh.listing.name.as_deref().unwrap_or("unnamed"),
                            if candidates.len() > 1 {
                                format!(" ({}/{})", i + 1, candidates.len())
                            } else {
                                String::new()
                            }
                        );
                        if cli.mesh_name.is_none() {
                            if let Some(ref name) = mesh.listing.name {
                                cli.mesh_name = Some(name.clone());
                            }
                        }
                        eprintln!(
                            "✅ Joining: {} ({} nodes, {} models{})",
                            mesh.listing.name.as_deref().unwrap_or("unnamed"),
                            mesh.listing.node_count,
                            mesh.listing.serving.len(),
                            mesh.listing
                                .region
                                .as_ref()
                                .map(|r| format!(", region: {r}"))
                                .unwrap_or_default()
                        );
                        cli.join.push(token.clone());
                        joined = true;
                        break;
                    }
                    if !joined {
                        eprintln!("⚠️  No meshes found — starting new");
                        let models = nostr::default_models_for_vram(my_vram_gb);
                        start_new_mesh(&mut cli, &models, my_vram_gb);
                    }
                }
            }
            nostr::AutoDecision::StartNew { models } => {
                if cli.client {
                    // Retry discovery — meshes may appear later
                    eprintln!("⏳ No meshes found yet — retrying in 15s...");
                    let mut found = false;
                    for attempt in 1..=20 {
                        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                        eprintln!("🔍 Retry {attempt}/20...");
                        if let Ok(retry_meshes) = nostr::discover(&relays, &filter).await {
                            if let nostr::AutoDecision::Join { candidates } =
                                nostr::smart_auto(&retry_meshes, my_vram_gb, target_name)
                            {
                                let (token, mesh) = &candidates[0];
                                if cli.mesh_name.is_none() {
                                    if let Some(ref name) = mesh.listing.name {
                                        cli.mesh_name = Some(name.clone());
                                    }
                                }
                                eprintln!(
                                    "✅ Joining: {} ({} nodes, {} models)",
                                    mesh.listing.name.as_deref().unwrap_or("unnamed"),
                                    mesh.listing.node_count,
                                    mesh.listing.serving.len()
                                );
                                cli.join.push(token.clone());
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        anyhow::bail!("No meshes found after 5 minutes of retrying.");
                    }
                } else {
                    start_new_mesh(&mut cli, &models, my_vram_gb);
                }
            }
        }
    }

    // --- Validation ---
    if cli.client && !cli.model.is_empty() {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    // No args at all = idle mode with console for browsing/joining
    if cli.model.is_empty() && cli.join.is_empty() && !cli.client && !cli.auto {
        let bin_dir = if cli.backend == Backend::Llama {
            match &cli.bin_dir {
                Some(d) => d.clone(),
                None => detect_bin_dir()?,
            }
        } else {
            PathBuf::new()
        };
        return run_idle(cli, bin_dir).await;
    }

    // --- Resolve models from CLI ---
    // All --model entries get resolved/downloaded. First is primary (gets rpc/tunnel).
    // Additional models run as solo llama-servers (must fit in VRAM independently).
    let mut resolved_models: Vec<PathBuf> = Vec::new();
    for m in &cli.model {
        resolved_models.push(resolve_model(m).await?);
    }

    // Build requested model names from all resolved models
    // Strip split GGUF suffix so "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
    let requested_model_names: Vec<String> = resolved_models
        .iter()
        .filter_map(|m| model_path_name(m).map(|s| router::strip_split_suffix_owned(&s)))
        .collect();

    if cli.backend == Backend::Mlx {
        return run_mlx(cli, resolved_models, requested_model_names).await;
    }

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

    // Check all model directories (including goose) for just a filename
    if !s.contains('/') {
        for dir in mesh::model_dirs() {
            let candidate = dir.join(input);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        // Try catalog match
        if let Some(entry) = download::find_model(&s) {
            return download::download_model(entry).await;
        }
        anyhow::bail!(
            "Model not found: {}\nNot a local file, not in ~/.models/ or goose models, not in catalog.\n\
             Use a path, a catalog name (run `mesh-llm download` to list), or a HuggingFace URL.",
            s
        );
    }

    // HuggingFace URL
    if s.starts_with("https://huggingface.co/") || s.starts_with("http://huggingface.co/") {
        let filename = s
            .rsplit('/')
            .next()
            .ok_or_else(|| anyhow::anyhow!("Can't extract filename from URL: {}", s))?;
        let dest = download::models_dir().join(filename);
        if dest.exists() {
            let size = tokio::fs::metadata(&dest).await?.len();
            if size > 1_000_000 {
                eprintln!(
                    "✅ {} already exists ({:.1}GB)",
                    filename,
                    size as f64 / 1e9
                );
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
                format!(
                    "https://huggingface.co/{}/{}/resolve/main/{}",
                    parts[0], parts[1], parts[2]
                )
            } else {
                anyhow::bail!("Can't parse HF shorthand: {}. Use org/repo/file.gguf", s);
            }
        };
        let filename = s.rsplit('/').next().unwrap();
        let dest = download::models_dir().join(filename);
        if dest.exists() {
            let size = tokio::fs::metadata(&dest).await?.len();
            if size > 1_000_000 {
                eprintln!(
                    "✅ {} already exists ({:.1}GB)",
                    filename,
                    size as f64 / 1e9
                );
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
    let catalog_entry = download::MODEL_CATALOG
        .iter()
        .find(|m| m.file == filename)?;
    let draft_name = catalog_entry.draft?;
    let draft_entry = download::MODEL_CATALOG
        .iter()
        .find(|m| m.name == draft_name)?;
    let draft_stem = draft_entry
        .file
        .strip_suffix(".gguf")
        .unwrap_or(draft_entry.file);
    let draft_path = mesh::find_model_path(draft_stem);
    if draft_path.exists() {
        return Some(draft_path);
    }
    // Draft not on disk — download it (small, <1GB)
    eprintln!(
        "📥 Downloading draft model {} ({})...",
        draft_entry.name, draft_entry.size
    );
    match download::download_model(draft_entry).await {
        Ok(_path) => {
            eprintln!("✅ Draft model ready: {}", draft_entry.name);
            Some(draft_path)
        }
        Err(e) => {
            eprintln!(
                "⚠ Failed to download draft model: {e} — continuing without speculative decoding"
            );
            None
        }
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
/// Parse a catalog size string like "18.3GB" or "491MB" into bytes.
fn parse_size_str(s: &str) -> u64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        (gb.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(mb) = s.strip_suffix("MB") {
        (mb.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else {
        0
    }
}

/// Pick which model this node should serve, based on demand signals.
///
/// Priority:
/// 1. Unserved models with active demand that we have on disk (hottest first)
/// 2. Underserved models with demand that we have on disk
/// 3. Unserved models with demand that we can download from catalog
/// 4. Standby if everything is covered
async fn pick_model_assignment(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;

    // Get active demand — the unified "what does the mesh want?"
    let demand = node.active_demand().await;

    if demand.is_empty() {
        // No API requests yet — log what the mesh is serving for visibility
        let served: Vec<&str> = peers.iter().filter_map(|p| p.serving.as_deref()).collect();
        if !served.is_empty() {
            eprintln!(
                "📋 No demand yet — mesh is serving {:?}, staying standby until needed",
                served
            );
        } else {
            eprintln!("📋 No demand signals — no models requested");
        }
        return None;
    }

    eprintln!("📋 Active demand: {:?}", demand.keys().collect::<Vec<_>>());

    // Count how many nodes are serving each model
    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        if let Some(ref s) = p.serving {
            *serving_count.entry(s.clone()).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    /// Check if a model fits in our VRAM. Returns false and logs if it doesn't.
    fn model_fits(model: &str, my_vram: u64) -> bool {
        let model_path = mesh::find_model_path(model);
        let model_bytes = std::fs::metadata(&model_path)
            .map(|md| md.len())
            .unwrap_or(0);
        let needed = (model_bytes as f64 * 1.1) as u64;
        if model_bytes > 0 && needed > my_vram {
            eprintln!(
                "📋 Skipping {} — needs {:.1}GB, we have {:.1}GB",
                model,
                needed as f64 / 1e9,
                my_vram as f64 / 1e9
            );
            return false;
        }
        true
    }

    // Sort demand entries by request_count descending (hottest first)
    let mut demand_sorted: Vec<(String, mesh::ModelDemand)> = demand.into_iter().collect();
    demand_sorted.sort_by(|a, b| b.1.request_count.cmp(&a.1.request_count));

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
            eprintln!(
                "📋 Assigned to serve {} (unserved, on disk, {} candidates, by demand)",
                pick,
                candidates.len()
            );
            return Some(pick.clone());
        }
        let pick = &candidates[0];
        eprintln!(
            "📋 Assigned to serve {} (unserved, on disk, by demand)",
            pick
        );
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
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        eprintln!(
            "📋 Assigned to serve {} ({} servers vs {} has {}) — rebalancing",
            pick, count, max_model, max_count
        );
        return Some(pick.clone());
    }

    // Priority 3: Unserved models we can download from catalog
    let mut downloadable: Vec<(String, u64)> = Vec::new(); // (model, demand)
    for (m, d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) > 0 {
            continue;
        }
        if let Some(cat) = download::find_model(m) {
            let size_bytes = parse_size_str(cat.size);
            let needed = (size_bytes as f64 * 1.1) as u64;
            if needed <= my_vram {
                downloadable.push((m.clone(), d.request_count));
            } else {
                eprintln!(
                    "📋 Skipping {} — needs {:.1}GB, we have {:.1}GB",
                    m,
                    needed as f64 / 1e9,
                    my_vram as f64 / 1e9
                );
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
            eprintln!(
                "📋 Assigned to serve {} (unserved, will download, by demand)",
                pick
            );
            return Some(pick.clone());
        }
        let (pick, _) = &downloadable[0];
        eprintln!(
            "📋 Assigned to serve {} (unserved, will download, by demand)",
            pick
        );
        return Some(pick.clone());
    }

    // Everything with demand is covered
    let all_covered = demand_sorted
        .iter()
        .all(|(m, _)| serving_count.get(m).copied().unwrap_or(0) > 0);
    if all_covered {
        eprintln!("📋 All demanded models are covered — staying on standby");
    }

    None
}

/// Check if a standby node should promote to serve a model.
/// Uses demand signals — promotes for unserved models with active demand,
/// or for demand-based rebalancing when one model is much hotter than others.
///
/// Rebalancing uses `last_active` to gate on recency (only models active within
/// the last 60 minutes are considered), then `request_count / servers` for
/// relative hotness among those recent models.
async fn check_unserved_model(node: &mesh::Node, local_models: &[String]) -> Option<String> {
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
        if let Some(ref s) = p.serving {
            *serving_count.entry(s.clone()).or_default() += 1;
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
            let model_path = mesh::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
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
            let model_path = mesh::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
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
            eprintln!(
                "📋 Promoting to serve {} — demand {:.0} req/server (coldest: {:.0})",
                hottest_model, hottest_ratio, coldest_ratio
            );
            return Some(hottest_model.clone());
        }
    }

    None
}

fn load_resolved_plugins(cli: &Cli) -> Result<plugin::ResolvedPlugins> {
    let config = plugin::load_config(cli.config.as_deref())?;
    plugin::resolve_plugins(&config, plugin_host_mode(cli))
}

fn detect_mlx_native_command(cli: &Cli) -> Result<String> {
    if let Some(path) = &cli.mlx_native_bin {
        return Ok(path.display().to_string());
    }

    let current_exe =
        std::env::current_exe().context("Cannot determine mesh-llm executable path")?;
    let exe_dir = current_exe
        .parent()
        .context("mesh-llm executable has no parent directory")?;
    let exe_suffix = std::env::consts::EXE_SUFFIX;
    for candidate in [exe_dir.join(format!("mesh-llm-mlx{exe_suffix}"))] {
        if candidate.exists() {
            return Ok(candidate.display().to_string());
        }
    }

    for command in ["mesh-llm-mlx"] {
        if command_on_path(command) {
            return Ok(command.to_string());
        }
    }

    anyhow::bail!(
        "Couldn't find the native MLX worker for --backend mlx. Build `mesh-llm-mlx`, put it next to `mesh-llm`, put it on PATH, or pass --mlx-native-bin."
    );
}

fn command_on_path(command: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    let exe_suffix = std::env::consts::EXE_SUFFIX;
    std::env::split_paths(&path_var).any(|dir| {
        let candidate = dir.join(format!("{command}{exe_suffix}"));
        if candidate.exists() {
            return true;
        }
        if exe_suffix.is_empty() {
            dir.join(command).exists()
        } else {
            false
        }
    })
}

fn plugin_host_mode(cli: &Cli) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if cli.publish || cli.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

fn blackboard_display_name(cli: &Cli, node: &mesh::Node) -> String {
    cli.name
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("USERNAME").ok())
        .unwrap_or_else(|| node.id().fmt_short().to_string())
}

async fn join_mesh_for_mcp(cli: &Cli, node: &mesh::Node) -> Result<()> {
    if !cli.join.is_empty() {
        for token in &cli.join {
            match node.join(token).await {
                Ok(()) => {
                    eprintln!("Joined mesh");
                    return Ok(());
                }
                Err(err) => tracing::warn!("Failed to join via token: {err}"),
            }
        }
        anyhow::bail!("Failed to join any peer for MCP mode");
    }

    if cli.auto || cli.discover.is_some() {
        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: cli.region.clone(),
        };
        let target_name = cli.discover.as_deref().or(cli.mesh_name.as_deref());
        let meshes = nostr::discover(&relays, &filter).await?;
        match nostr::smart_auto(&meshes, 0.0, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                let (token, mesh) = &candidates[0];
                eprintln!(
                    "✅ Joining: {} ({} nodes, {} models{})",
                    mesh.listing.name.as_deref().unwrap_or("unnamed"),
                    mesh.listing.node_count,
                    mesh.listing.serving.len(),
                    mesh.listing
                        .region
                        .as_ref()
                        .map(|r| format!(", region: {r}"))
                        .unwrap_or_default()
                );
                node.join(token).await?;
            }
            nostr::AutoDecision::StartNew { .. } => {
                anyhow::bail!("No mesh found for MCP mode. Pass --join or start a mesh first.");
            }
        }
    }

    Ok(())
}

async fn run_plugin_mcp(cli: &Cli) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(cli)?;
    let (node, _channels) = mesh::Node::start(
        NodeRole::Client,
        &cli.relay,
        cli.bind_port,
        Some(0.0),
        cli.enumerate_host,
    )
    .await?;
    node.start_accepting();
    node.set_blackboard_name(blackboard_display_name(cli, &node))
        .await;
    node.start_heartbeat();
    join_mesh_for_mcp(cli, &node).await?;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    if plugin_manager.list().await.is_empty() {
        tracing::warn!("No plugins are enabled for MCP exposure");
    }

    plugin_mcp::run_mcp_server(plugin_manager).await
}

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
async fn run_auto(
    mut cli: Cli,
    resolved_models: Vec<PathBuf>,
    requested_model_names: Vec<String>,
    bin_dir: PathBuf,
) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(&cli)?;
    let api_port = cli.port;
    let console_port = Some(cli.console);
    let is_client = cli.client;

    // Scan local models on disk
    let local_models = if is_client {
        vec![]
    } else {
        mesh::scan_local_models()
    };
    tracing::info!("Local models on disk: {:?}", local_models);

    // Start mesh node — clients use ephemeral key (unique identity per run)
    let role = if is_client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    // Clients report 0 VRAM so they're never assigned a model to serve
    let max_vram = if is_client { Some(0.0) } else { cli.max_vram };
    let (node, channels) = mesh::Node::start(
        role,
        &cli.relay,
        cli.bind_port,
        max_vram,
        cli.enumerate_host,
    )
    .await?;
    node.start_accepting();
    let token = node.invite_token();
    node.set_blackboard_name(blackboard_display_name(&cli, &node))
        .await;
    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    // Advertise what we have on disk and what we want the mesh to serve
    node.set_available_models(local_models.clone()).await;
    node.set_requested_models(requested_model_names.clone())
        .await;

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
                    tracing::info!("Mesh ID: {id}");
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

        // Nostr re-discovery: if we joined via --auto (Nostr discovery) and lose
        // all peers, re-discover and join a new mesh. This handles the case where
        // the original mesh publisher restarts with a new identity.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    } else {
        // Originator — generate mesh_id
        let nostr_pubkey = if cli.publish {
            nostr::load_or_create_keys()
                .ok()
                .map(|k| k.public_key().to_hex())
        } else {
            None
        };
        let mesh_id = mesh::generate_mesh_id(cli.mesh_name.as_deref(), nostr_pubkey.as_deref());
        node.set_mesh_id_force(mesh_id.clone()).await;
        mesh::save_last_mesh_id(&mesh_id);
        tracing::info!("Mesh ID: {mesh_id}");
        eprintln!("Invite: {token}");
        eprintln!("Waiting for peers...");

        // Originator also re-discovers: if we started solo and a matching mesh
        // already exists on Nostr, we should join it instead of staying alone.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    }

    // Start bootstrap proxy if joining an existing mesh.
    // This gives instant API access via tunnel while our GPU loads.
    let mut bootstrap_listener_tx = if !cli.join.is_empty() {
        let (stop_tx, stop_rx) =
            tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
        let boot_node = node.clone();
        let boot_port = api_port;
        tokio::spawn(async move {
            bootstrap_proxy(boot_node, boot_port, stop_rx, cli.listen_all).await;
        });
        Some(stop_tx)
    } else {
        None
    };

    // Decide which model THIS node will serve
    let model = if !resolved_models.is_empty() {
        // First --model is what we serve (already resolved/downloaded)
        resolved_models[0].clone()
    } else {
        // No --model: try to find a model on disk that the mesh needs
        eprintln!("No --model specified, checking local models against mesh...");

        // Give gossip a moment to propagate
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let assignment = pick_model_assignment(&node, &local_models).await;
        // If no demand-based assignment but we have VRAM, use auto pack's primary model
        let assignment = if assignment.is_none() && cli.auto && !is_client {
            let pack = nostr::auto_model_pack(node.vram_bytes() as f64 / 1e9);
            if !pack.is_empty() {
                eprintln!(
                    "📋 No unserved demand — serving {} for {:.0}GB VRAM",
                    pack[0],
                    node.vram_bytes() as f64 / 1e9
                );
                Some(pack[0].clone())
            } else {
                assignment
            }
        } else {
            assignment
        };
        if let Some(model_name) = assignment {
            eprintln!("Mesh assigned model: {model_name}");
            let model_path = mesh::find_model_path(&model_name);
            if model_path.exists() {
                model_path
            } else if let Some(cat) = download::find_model(&model_name) {
                // Model not on disk but in catalog — download it
                eprintln!("📥 Downloading {} for mesh...", model_name);
                let dest = download::models_dir().join(cat.file);
                download::download_model(cat).await?;
                dest
            } else {
                // Not on disk and not in catalog — try common paths
                let alt = download::models_dir().join(&model_name);
                if alt.exists() {
                    alt
                } else {
                    model_path
                }
            }
        } else {
            // Nothing on disk matches — go passive, act as proxy
            // Stop bootstrap proxy first (run_passive binds its own listener)
            drop(bootstrap_listener_tx.take());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            // If a model becomes unserved while we're standby, we'll promote
            if is_client {
                eprintln!("📡 Running as client — proxying requests to mesh");
            } else {
                eprintln!("💤 No matching model on disk — running as standby GPU node");
                eprintln!(
                    "   VRAM: {:.1}GB, models on disk: {:?}",
                    node.vram_bytes() as f64 / 1e9,
                    local_models
                );
                eprintln!("   Proxying requests to other nodes. Will activate when needed.");
            }
            match run_passive(&cli, node.clone(), is_client, plugin_manager.clone()).await? {
                Some(model_name) => {
                    // Promoted! Resolve the model path and continue to serving
                    let model_path = mesh::find_model_path(&model_name);
                    if model_path.exists() {
                        model_path
                    } else {
                        let alt = download::models_dir().join(&model_name);
                        if alt.exists() {
                            alt
                        } else {
                            model_path
                        }
                    }
                }
                None => return Ok(()), // clean shutdown
            }
        }
    };

    let model_name = {
        let stem = model_path_name(&model).unwrap_or_default();
        // Strip split GGUF suffix: "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
        router::strip_split_suffix_owned(&stem)
    };

    // Set model source for gossip (so other joiners can discover it too)
    let model_source = if !cli.model.is_empty() {
        cli.model[0].to_string_lossy().to_string()
    } else {
        model_name.clone()
    };
    node.set_model_source(model_source).await;
    // Set all serving models (primary + extras)
    let all_serving = build_serving_list(&resolved_models, &model_name);
    node.set_serving_models(all_serving.clone()).await;
    node.set_models(all_serving).await;
    // Re-gossip so peers learn what we're serving
    node.regossip().await;

    // Ensure draft model is available (downloads if needed, <1GB)
    if cli.draft.is_none() && !cli.no_draft {
        if let Some(draft_path) = ensure_draft(&model).await {
            eprintln!("Auto-detected draft model: {}", draft_path.display());
            cli.draft = Some(draft_path);
        }
    }

    // Clean up stale processes from previous runs
    launch::kill_orphan_rpc_servers().await;

    // Start rpc-server
    let rpc_port = launch::start_rpc_server(&bin_dir, cli.device.as_deref(), Some(&model)).await?;
    tracing::info!("rpc-server on 127.0.0.1:{rpc_port} serving {model_name}");

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), rpc_port, channels.rpc, channels.http).await?;

    // Election publishes per-model targets
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

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
        api_proxy(
            proxy_node,
            api_port,
            proxy_rx,
            drop_tx,
            existing_listener,
            cli.listen_all,
        )
        .await;
    });

    // Console (optional)
    let model_name_for_console = model_name.clone();
    let console_state = if let Some(cport) = console_port {
        let model_size_bytes = election::total_model_bytes(&model);
        let cs = api::MeshApi::new(
            node.clone(),
            model_name_for_console.clone(),
            api_port,
            model_size_bytes,
            plugin_manager.clone(),
        );
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if let Some(draft) = &cli.draft {
            let dn = draft
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            cs.set_draft_name(dn).await;
        }
        if let Some(ref name) = cli.mesh_name {
            cs.set_mesh_name(name.clone()).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        let mn = model_name_for_console.clone();
        tokio::spawn(async move {
            // Console still takes old-style InferenceTarget for now — adapt
            let (adapted_tx, adapted_rx) =
                tokio::sync::watch::channel(election::InferenceTarget::None);
            tokio::spawn(async move {
                let mut rx = console_rx;
                loop {
                    let targets = rx.borrow().clone();
                    let target = targets.get(&mn);
                    adapted_tx.send_replace(target);
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            });
            api::start(cport, cs2, adapted_rx, cli.listen_all).await;
        });
        Some(cs)
    } else {
        None
    };

    // Election loop
    tracing::info!("Entering auto-election for model: {model_name}");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    let force_split = cli.split;
    let cb_console_port = console_port;
    let model_name_for_cb = model_name.clone();
    let model_name_for_election = model_name.clone();
    let node_for_cb = node.clone();
    let primary_target_tx = target_tx.clone();
    tokio::spawn(async move {
        election::election_loop(
            node2, tunnel_mgr2, rpc_port, bin_dir2, model2, model_name_for_election,
            draft2, draft_max, force_split, cli.ctx_size, primary_target_tx,
            move |is_host, llama_ready| {
                if llama_ready {
                    let n = node_for_cb.clone();
                    tokio::spawn(async move { n.set_llama_ready(true).await; });
                }
                if is_host && llama_ready {
                    let url = format!("http://localhost:{api_port}");
                    eprintln!("  API:     {url}");
                    if let Some(cp) = cb_console_port {
                        eprintln!("  Console: http://localhost:{cp}");
                    }
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

    // Additional model election loops (multi-model per node)
    // Each additional model gets its own solo election loop — no rpc, no draft, no split.
    // They share the same target_tx so the proxy sees all models.
    if resolved_models.len() > 1 {
        eprintln!(
            "🔀 Multi-model mode: {} additional model(s)",
            resolved_models.len() - 1
        );
        // Announce all models to mesh
        let all_names: Vec<String> = resolved_models
            .iter()
            .map(|m| model_path_name(m).unwrap_or_default())
            .collect();
        node.set_models(all_names).await;
        node.regossip().await;

        for extra_model in resolved_models.iter().skip(1) {
            let extra_name = {
                let stem = model_path_name(extra_model).unwrap_or_default();
                router::strip_split_suffix_owned(&stem)
            };
            let extra_node = node.clone();
            let extra_tunnel = tunnel_mgr.clone();
            let extra_bin = bin_dir.clone();
            let extra_path = extra_model.clone();
            let extra_target_tx = target_tx.clone();
            let extra_model_name = extra_name.clone();
            let api_port_extra = api_port;
            eprintln!("  + {extra_name}");
            tokio::spawn(async move {
                election::election_loop(
                    extra_node, extra_tunnel, 0, extra_bin, extra_path, extra_model_name.clone(),
                    None, 8, false, cli.ctx_size, extra_target_tx,
                    move |is_host, llama_ready| {
                        if is_host && llama_ready {
                            eprintln!("✅ [{extra_model_name}] ready (multi-model)");
                            eprintln!("  API: http://localhost:{api_port_extra} (model={extra_model_name})");
                        }
                    },
                ).await;
            });
        }
    }

    // Nostr publish loop (if --publish) or watchdog (if --auto, to take over if publisher dies)
    let nostr_publisher = if cli.publish {
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_node = node.clone();
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        Some(tokio::spawn(async move {
            nostr::publish_loop(
                pub_node,
                nostr_keys,
                relays,
                pub_name,
                pub_region,
                pub_max_clients,
                60,
            )
            .await;
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
    launch::kill_orphan_rpc_servers().await;
    Ok(())
}

async fn run_mlx(
    cli: Cli,
    resolved_models: Vec<PathBuf>,
    _requested_model_names: Vec<String>,
) -> Result<()> {
    anyhow::ensure!(
        cfg!(target_os = "macos"),
        "--backend mlx is currently supported on macOS only"
    );
    anyhow::ensure!(
        !cli.client,
        "--backend mlx currently runs as a host node; use the default client/passive path on remote consumers"
    );
    anyhow::ensure!(
        !cli.split && cli.tensor_split.is_none(),
        "--backend mlx does not support split mode yet"
    );
    anyhow::ensure!(
        cli.draft.is_none(),
        "--backend mlx does not support draft/speculative mode yet"
    );
    anyhow::ensure!(
        resolved_models.len() == 1,
        "--backend mlx currently supports exactly one --model"
    );

    let api_port = cli.port;
    let model = resolved_models[0].clone();
    let model_name = model_path_name(&model).unwrap_or_else(|| "local".to_string());
    let mlx_port = launch::find_free_port().await?;
    let mlx_command = detect_mlx_native_command(&cli)?;
    let mut mlx_death = launch::start_mlx_native_server(&mlx_command, &model, mlx_port).await?;
    let resolved_plugins = load_resolved_plugins(&cli)?;

    let mut local_models = mesh::scan_local_models();
    let (node, channels) = mesh::Node::start(
        NodeRole::Worker,
        &cli.relay,
        cli.bind_port,
        cli.max_vram,
        cli.enumerate_host,
    )
    .await?;
    node.start_accepting();
    let token = node.invite_token();
    node.set_blackboard_name(blackboard_display_name(&cli, &node))
        .await;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);
    if !local_models.contains(&model_name) {
        local_models.push(model_name.clone());
        local_models.sort();
    }

    node.set_available_models(local_models).await;
    node.set_requested_models(vec![model_name.clone()]).await;
    node.set_model_source(
        cli.model
            .first()
            .map(|m| m.to_string_lossy().to_string())
            .unwrap_or_else(|| model.display().to_string()),
    )
    .await;
    node.set_serving_models(vec![model_name.clone()]).await;
    node.set_models(vec![model_name.clone()]).await;
    node.set_role(NodeRole::Host {
        http_port: mlx_port,
    })
    .await;
    node.set_llama_ready(true).await;
    node.regossip().await;
    node.start_heartbeat();

    let tunnel_mgr = tunnel::Manager::start(node.clone(), 0, channels.rpc, channels.http).await?;
    tunnel_mgr.set_http_port(mlx_port);

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

        {
            let save_node = node.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if let Some(id) = save_node.mesh_id().await {
                    mesh::save_last_mesh_id(&id);
                    tracing::info!("Mesh ID: {id}");
                }
            });
        }

        eprintln!("This node's token (for others to join): {token}");

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

        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    } else {
        let nostr_pubkey = if cli.publish {
            nostr::load_or_create_keys()
                .ok()
                .map(|k| k.public_key().to_hex())
        } else {
            None
        };
        let mesh_id = mesh::generate_mesh_id(cli.mesh_name.as_deref(), nostr_pubkey.as_deref());
        node.set_mesh_id_force(mesh_id.clone()).await;
        mesh::save_last_mesh_id(&mesh_id);
        tracing::info!("Mesh ID: {mesh_id}");
        eprintln!("Invite: {token}");
        eprintln!("Waiting for peers...");

        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    }

    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        model_name.clone(),
        vec![election::InferenceTarget::Local(mlx_port)],
    );
    let (target_tx, target_rx) = tokio::sync::watch::channel(targets.clone());
    target_tx.send_replace(targets);
    let (console_target_tx, console_target_rx) =
        tokio::sync::watch::channel(election::InferenceTarget::Local(mlx_port));
    console_target_tx.send_replace(election::InferenceTarget::Local(mlx_port));
    let (drop_tx, mut drop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let proxy_node = node.clone();
    tokio::spawn(async move {
        api_proxy(
            proxy_node,
            api_port,
            target_rx,
            drop_tx,
            None,
            cli.listen_all,
        )
        .await;
    });

    if cli.console > 0 {
        let cs = api::MeshApi::new(
            node.clone(),
            model_name.clone(),
            api_port,
            election::total_model_bytes(&model),
            plugin_manager.clone(),
        );
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if let Some(ref name) = cli.mesh_name {
            cs.set_mesh_name(name.clone()).await;
        }
        cs.set_llama_port(Some(mlx_port)).await;
        cs.update(true, true).await;
        tokio::spawn(async move {
            api::start(cli.console, cs, console_target_rx, cli.listen_all).await;
        });
    }

    eprintln!("✅ [{}] MLX backend ready", model_name);
    eprintln!("  API: http://localhost:{api_port}");
    if cli.console > 0 {
        eprintln!("  Console: http://localhost:{}", cli.console);
    }
    eprintln!("  MLX upstream: http://127.0.0.1:{mlx_port}");
    eprintln!("  MLX worker: {mlx_command}");

    let nostr_publisher = if cli.publish {
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_node = node.clone();
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        Some(tokio::spawn(async move {
            nostr::publish_loop(
                pub_node,
                nostr_keys,
                relays,
                pub_name,
                pub_region,
                pub_max_clients,
                60,
            )
            .await;
        }))
    } else if cli.auto {
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
            drop_model_name.clone()
        } => {
            eprintln!("\n🗑 Model '{}' dropped from mesh — shutting down", dropped);
            drop_node.set_serving(None).await;
        }
        _ = &mut mlx_death => {
            eprintln!("\n⚠️ MLX worker exited unexpectedly — shutting down");
            drop_node.set_serving(None).await;
        }
    }

    node.broadcast_leaving().await;

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

    launch::kill_mlx_native_server().await;

    Ok(())
}

/// Idle mode: no args → show instructions and read-only console.
/// Use --auto or --join to actually connect to a mesh.
async fn run_idle(cli: Cli, _bin_dir: PathBuf) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(&cli)?;
    let my_vram_gb = mesh::detect_vram_bytes_capped(cli.max_vram) as f64 / 1e9;
    let local_models = mesh::scan_local_models();
    eprintln!(
        "mesh-llm v{VERSION} — {:.0}GB VRAM, {} models on disk",
        my_vram_gb,
        local_models.len()
    );
    eprintln!();
    eprintln!("  Console: http://localhost:{}", cli.console);
    eprintln!();
    eprintln!("  Start a mesh:");
    eprintln!("    mesh-llm --model Qwen2.5-32B                 serve a model");
    eprintln!("    mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name \"my-mesh\"");
    eprintln!();
    eprintln!("  Join a mesh:");
    eprintln!("    mesh-llm --auto              discover and join automatically");
    eprintln!("    mesh-llm --join <token>      join by invite token");
    eprintln!("    mesh-llm --client --auto     join as API-only client");
    eprintln!();

    // Start a dormant node just for the console
    let (node, _channels) = mesh::Node::start(
        NodeRole::Worker,
        &cli.relay,
        cli.bind_port,
        cli.max_vram,
        cli.enumerate_host,
    )
    .await?;
    node.set_available_models(local_models).await;
    node.set_blackboard_name(blackboard_display_name(&cli, &node))
        .await;
    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    let cs = api::MeshApi::new(node.clone(), "(idle)".into(), cli.port, 0, plugin_manager);
    cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
    cs.update(false, false).await;
    let cs2 = cs.clone();
    let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
    tokio::spawn(async move {
        api::start(cli.console, cs2, rx, cli.listen_all).await;
    });

    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");
    Ok(())
}

/// Used by both --client (pure consumer) and idle GPU nodes (standby, no matching model).
/// If `create_node` is true, creates a new Node (--client path). Otherwise reuses existing.
/// Run as passive node (client or standby GPU).
/// Returns Ok(Some(model_name)) if a standby GPU should promote to serve a model.
/// Returns Ok(None) on clean shutdown.
async fn run_passive(
    cli: &Cli,
    node: mesh::Node,
    is_client: bool,
    plugin_manager: plugin::PluginManager,
) -> Result<Option<String>> {
    let local_port = cli.port;
    node.set_blackboard_name(blackboard_display_name(cli, &node))
        .await;

    // Nostr publishing (if --publish, for idle GPU nodes advertising capacity)
    if cli.publish && !is_client {
        let pub_node = node.clone();
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        tokio::spawn(async move {
            nostr::publish_loop(
                pub_node,
                nostr_keys,
                relays,
                pub_name,
                pub_region,
                pub_max_clients,
                60,
            )
            .await;
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

    let addr = if cli.listen_all {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let listener = tokio::net::TcpListener::bind(format!("{addr}:{local_port}"))
        .await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    let mode = if is_client { "client" } else { "standby" };
    eprintln!("Passive {mode} ready:");
    eprintln!("  API:     http://localhost:{local_port}");
    eprintln!("  Console: http://localhost:{}", cli.console);

    // Console
    {
        let cport = cli.console;
        let label = if is_client {
            "(client)".to_string()
        } else {
            "(standby)".to_string()
        };
        let cs = api::MeshApi::new(node.clone(), label, local_port, 0, plugin_manager);
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if is_client {
            cs.set_client(true).await;
        }
        // Both clients and standby nodes can proxy requests through the mesh
        cs.update(false, true).await;
        let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
        let la = cli.listen_all;
        tokio::spawn(async move {
            api::start(cport, cs, rx, la).await;
        });
    }

    // Heartbeat (started in run_auto) handles periodic gossip via random-K.
    // No extra gossip loop needed here.

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
                tokio::spawn(proxy::handle_mesh_request(node, tcp_stream, true));
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
}

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    drop_tx: tokio::sync::mpsc::UnboundedSender<String>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
) {
    let listener = match existing_listener {
        Some(l) => l,
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {e}");
                    return;
                }
            }
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
            match proxy::peek_request(&tcp_stream, &mut buf).await {
                Ok((n, model_name)) => {
                    if proxy::is_models_list_request(&buf[..n]) {
                        let models: Vec<String> = targets.targets.keys().cloned().collect();
                        let _ = proxy::send_models_list(tcp_stream, &models).await;
                        return;
                    }

                    if proxy::is_drop_request(&buf[..n]) {
                        if let Some(ref name) = model_name {
                            let _ = drop_tx.send(name.clone());
                            let _ = proxy::send_json_ok(
                                tcp_stream,
                                &serde_json::json!({"dropped": name}),
                            )
                            .await;
                        } else {
                            let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    // Smart routing: if no model specified (or model="auto"), classify and pick
                    let (effective_model, classification) =
                        if model_name.is_none() || model_name.as_deref() == Some("auto") {
                            if let Some(body_json) = proxy::extract_body_json(&buf[..n]) {
                                let cl = router::classify(&body_json);
                                let available: Vec<(&str, f64)> = targets
                                    .targets
                                    .keys()
                                    .map(|name| (name.as_str(), 0.0))
                                    .collect();
                                let picked = router::pick_model_classified(&cl, &available);
                                if let Some(name) = picked {
                                    tracing::info!(
                                        "router: {:?}/{:?} tools={} → {name}",
                                        cl.category,
                                        cl.complexity,
                                        cl.needs_tools
                                    );
                                    (Some(name.to_string()), Some(cl))
                                } else {
                                    (None, Some(cl))
                                }
                            } else {
                                (None, None)
                            }
                        } else {
                            (model_name.clone(), None)
                        };

                    if let Some(ref name) = effective_model {
                        node.record_request(name);
                    }

                    // Pipeline routing: for complex agentic tasks, pre-plan with a small model
                    let use_pipeline = classification
                        .as_ref()
                        .map(|cl| pipeline::should_pipeline(cl))
                        .unwrap_or(false);

                    if use_pipeline {
                        if let Some(ref strong_name) = effective_model {
                            // Find a planner: any local model that isn't the strong model
                            let planner = targets
                                .targets
                                .iter()
                                .find(|(name, target_vec)| {
                                    *name != strong_name
                                        && target_vec.iter().any(|t| {
                                            matches!(t, election::InferenceTarget::Local(_))
                                        })
                                })
                                .and_then(|(name, target_vec)| {
                                    target_vec.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => {
                                            Some((name.clone(), *p))
                                        }
                                        _ => None,
                                    })
                                });

                            let strong_local_port =
                                targets.targets.get(strong_name.as_str()).and_then(|tv| {
                                    tv.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => Some(*p),
                                        _ => None,
                                    })
                                });

                            if let (Some((planner_name, planner_port)), Some(strong_port)) =
                                (planner, strong_local_port)
                            {
                                tracing::info!(
                                    "pipeline: {planner_name} (plan) → {strong_name} (execute)"
                                );
                                proxy::pipeline_proxy_local(
                                    tcp_stream,
                                    &buf,
                                    n,
                                    planner_port,
                                    &planner_name,
                                    strong_port,
                                    &node,
                                )
                                .await;
                                return;
                            }
                        }
                        // Fall through to normal routing if pipeline setup fails
                    }

                    // MoE routing: use session hint for sticky routing across shards
                    let target = if targets.moe.is_some() {
                        let session_hint = proxy::extract_session_hint(&buf[..n])
                            .unwrap_or_else(|| format!("{_addr}"));
                        targets
                            .get_moe_target(&session_hint)
                            .unwrap_or(first_available_target(&targets))
                    } else if let Some(ref name) = effective_model {
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

                    proxy::route_to_target(node, tcp_stream, target).await;
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
    listen_all: bool,
) {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
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
                tokio::spawn(proxy::handle_mesh_request(node, tcp_stream, true));
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
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}

fn detect_bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let dir = exe.parent().context("Binary has no parent directory")?;

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

    let providers = root.as_object_mut().and_then(|r| {
        r.entry("providers")
            .or_insert_with(|| serde_json::json!({}));
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
/// Re-discover meshes via Nostr when all peers are lost.
/// Only runs for --auto nodes that originally discovered via Nostr.
/// Checks every 30s; if 0 peers for 90s straight, re-discovers and joins.
async fn nostr_rediscovery(
    node: mesh::Node,
    nostr_relays: Vec<String>,
    _relay_urls: Vec<String>,
    mesh_name: Option<String>,
) {
    const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
    const GRACE_PERIOD: std::time::Duration = std::time::Duration::from_secs(90);

    // Don't start checking immediately — give the initial connection time to establish
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    let mut alone_since: Option<std::time::Instant> = None;

    loop {
        tokio::time::sleep(CHECK_INTERVAL).await;

        let peers = node.peers().await;
        if !peers.is_empty() {
            // We have peers — reset the timer
            if alone_since.is_some() {
                tracing::debug!("Nostr rediscovery: peers recovered, resetting timer");
                alone_since = None;
            }
            continue;
        }

        // Zero peers
        let now = std::time::Instant::now();
        let start = *alone_since.get_or_insert(now);

        if now.duration_since(start) < GRACE_PERIOD {
            tracing::debug!(
                "Nostr rediscovery: 0 peers for {}s (grace: {}s)",
                now.duration_since(start).as_secs(),
                GRACE_PERIOD.as_secs()
            );
            continue;
        }

        // Grace period expired — re-discover
        eprintln!("🔍 No peers — re-discovering meshes via Nostr...");

        let filter = nostr::MeshFilter::default();
        let meshes = match nostr::discover(&nostr_relays, &filter).await {
            Ok(m) => m,
            Err(e) => {
                eprintln!("⚠️  Nostr re-discovery failed: {e}");
                // Reset timer so we don't spam
                alone_since = Some(std::time::Instant::now());
                continue;
            }
        };

        // Filter by mesh name if set
        let filtered: Vec<_> = if let Some(ref name) = mesh_name {
            meshes
                .iter()
                .filter(|m| {
                    m.listing
                        .name
                        .as_ref()
                        .map(|n| n.eq_ignore_ascii_case(name))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            meshes.iter().collect()
        };

        if filtered.is_empty() {
            let name_hint = mesh_name.as_deref().unwrap_or("any");
            eprintln!("⚠️  No meshes found on Nostr matching \"{name_hint}\" — will retry");
            alone_since = Some(std::time::Instant::now());
            continue;
        }

        // Try to join the best mesh
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_mesh_id = mesh::load_last_mesh_id();

        let mut candidates: Vec<_> = filtered
            .iter()
            .map(|m| (*m, nostr::score_mesh(m, now_ts, last_mesh_id.as_deref())))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        // Skip our own mesh (we'd be joining ourselves)
        let our_mesh_id = node.mesh_id().await;

        let mut rejoined = false;
        for (mesh, _score) in &candidates {
            if let (Some(ref ours), Some(ref theirs)) = (&our_mesh_id, &mesh.listing.mesh_id) {
                if ours == theirs {
                    continue;
                }
            }
            let token = &mesh.listing.invite_token;
            eprintln!(
                "✅ Re-joining: {} ({} nodes)",
                mesh.listing.name.as_deref().unwrap_or("unnamed"),
                mesh.listing.node_count
            );
            // Join directly — no probe. The probe uses a separate ephemeral endpoint
            // which can fail due to firewalls even when the real node.join() would
            // succeed (our persistent endpoint may already have a relay path).
            match node.join(token).await {
                Ok(()) => {
                    eprintln!("📡 Re-joined mesh via Nostr re-discovery");
                    rejoined = true;
                }
                Err(e) => {
                    eprintln!("⚠️  Re-join failed: {e}");
                }
            }
            if rejoined {
                break;
            }
        }

        if rejoined {
            // Reset — if we lose peers again, the cycle restarts
            alone_since = None;
        } else {
            eprintln!("⚠️  Could not re-join any mesh — will retry");
            alone_since = Some(std::time::Instant::now());
        }
    }
}

/// Helper for StartNew path — configure CLI to start a new mesh.
fn start_new_mesh(cli: &mut Cli, _models: &[String], my_vram_gb: f64) {
    // Pick the best single model for this VRAM tier.
    // Multi-model requires explicit --model A --model B.
    let pack = nostr::auto_model_pack(my_vram_gb);
    let primary = pack.first().cloned().unwrap_or_default();
    eprintln!("🆕 Starting a new mesh");
    eprintln!("   Serving: {primary}");
    eprintln!("   VRAM: {:.0}GB", my_vram_gb);
    if cli.model.is_empty() {
        cli.model.push(primary.into());
    }
    if !cli.publish {
        cli.publish = true;
        eprintln!("   Auto-enabling --publish for discovery");
    }
}

fn nostr_relays(cli_relays: &[String]) -> Vec<String> {
    if cli_relays.is_empty() {
        nostr::DEFAULT_RELAYS
            .iter()
            .map(|s| s.to_string())
            .collect()
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
    let relays = nostr_relays(&relays);

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
        let score = nostr::score_mesh(mesh, now, last_mesh_id.as_deref());
        let age = now.saturating_sub(mesh.published_at);
        let freshness = if age < 120 {
            "fresh"
        } else if age < 300 {
            "ok"
        } else {
            "stale"
        };
        let capacity = if mesh.listing.max_clients > 0 {
            format!(
                "{}/{} clients",
                mesh.listing.client_count, mesh.listing.max_clients
            )
        } else {
            format!("{} clients", mesh.listing.client_count)
        };
        eprintln!(
            "  [{}] {} (score: {}, {}, {})",
            i + 1,
            mesh,
            score,
            freshness,
            capacity
        );
        let token = &mesh.listing.invite_token;
        let display_token = if token.len() > 40 {
            format!("{}...{}", &token[..20], &token[token.len() - 12..])
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
fn run_stop() -> Result<()> {
    use std::process::Command as Cmd;
    let mut killed = 0u32;
    for name in &["mesh-llm", "llama-server", "rpc-server", "mesh-llm-mlx"] {
        // pkill sends SIGTERM; ignore errors (process might not exist)
        let status = Cmd::new("pkill").arg("-f").arg(name).status();
        match status {
            Ok(s) if s.success() => {
                eprintln!("🧹 Stopped {name}");
                killed += 1;
            }
            _ => {}
        }
    }
    if killed == 0 {
        eprintln!("Nothing running.");
    }
    Ok(())
}

async fn run_drop(model_name: &str, port: u16) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let body = serde_json::json!({ "model": model_name }).to_string();
    let request = format!(
        "POST /mesh/drop HTTP/1.1\r\nHost: localhost:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    stream.write_all(request.as_bytes()).await?;

    let mut response = vec![0u8; 4096];
    let n = stream.read(&mut response).await?;
    let resp = String::from_utf8_lossy(&response[..n]);

    if resp.contains("200 OK") {
        eprintln!("✅ Dropped model: {model_name}");
    } else {
        eprintln!(
            "❌ Failed to drop model: {}",
            resp.lines().last().unwrap_or("unknown error")
        );
    }

    Ok(())
}

/// Ensure mesh-llm is running on `port`, then return (available_models, chosen_model, spawned_child).
///
/// Launcher behavior: if nothing is listening yet, auto-start `mesh-llm --client --auto`
/// (client node — tunnels to mesh peers without publishing to Nostr).
/// Returns the child process handle if we spawned one, so callers can clean up on exit.
async fn check_mesh(
    client: &reqwest::Client,
    port: u16,
    model: &Option<String>,
) -> Result<(Vec<String>, String, Option<std::process::Child>)> {
    let url = format!("http://127.0.0.1:{port}/v1/models");

    // If no local mesh API is up, start a full auto-join node in the background.
    let mut child: Option<std::process::Child> = None;
    if client.get(&url).send().await.is_err() {
        eprintln!("🔍 No mesh-llm on port {port} — starting background auto-join node...");
        let exe = std::env::current_exe().unwrap_or_else(|_| "mesh-llm".into());
        child = Some(
            std::process::Command::new(&exe)
                .args(["--client", "--auto", "--port", &port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to start mesh-llm node")?,
        );
    }

    // Wait for API/models readiness.
    let mut models: Vec<String> = Vec::new();
    for i in 0..40 {
        if let Ok(resp) = client.get(&url).send().await {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                models = body["data"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect();
                if !models.is_empty() {
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        if i % 5 == 4 {
            eprintln!(
                "   Waiting for mesh/models... ({:.0}s)",
                (i + 1) as f64 * 3.0
            );
        }
    }

    if models.is_empty() {
        // Clean up the child we spawned before bailing
        if let Some(mut c) = child {
            let _ = c.kill();
        }
        anyhow::bail!(
            "mesh-llm on port {port} has no models yet (or could not be reached).\n\
             Ensure at least one serving peer is available on the mesh."
        );
    }

    let chosen = if let Some(ref m) = model {
        if !models.iter().any(|n| n == m) {
            if let Some(mut c) = child {
                let _ = c.kill();
                let _ = c.wait();
            }
            anyhow::bail!(
                "Model '{}' not available. Available: {}",
                m,
                models.join(", ")
            );
        }
        m.clone()
    } else {
        // Pick the strongest tool-capable model for agentic work.
        let available: Vec<(&str, f64)> = models.iter().map(|n| (n.as_str(), 0.0)).collect();
        let agentic = router::Classification {
            category: router::Category::Code,
            complexity: router::Complexity::Deep,
            needs_tools: true,
        };
        router::pick_model_classified(&agentic, &available)
            .map(|s| s.to_string())
            .unwrap_or_else(|| models[0].clone())
    };
    eprintln!("   Models: {}", models.join(", "));
    eprintln!("   Using: {chosen}");
    Ok((models, chosen, child))
}

async fn run_goose(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut _mesh_child) = check_mesh(&client, port, &model).await?;

    // Write custom provider JSON
    let goose_config_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("goose")
        .join("custom_providers");
    std::fs::create_dir_all(&goose_config_dir)?;

    let provider_models: Vec<serde_json::Value> = models
        .iter()
        .map(|name| serde_json::json!({"name": name, "context_limit": 65536}))
        .collect();

    let provider = serde_json::json!({
        "name": "mesh",
        "engine": "openai",
        "display_name": "mesh-llm",
        "description": "Distributed LLM inference via mesh-llm",
        "api_key_env": "",
        "base_url": format!("http://localhost:{port}"),
        "models": provider_models,
        "timeout_seconds": 600,
        "supports_streaming": true,
        "requires_auth": false
    });

    let provider_path = goose_config_dir.join("mesh.json");
    std::fs::write(&provider_path, serde_json::to_string_pretty(&provider)?)?;
    eprintln!("✅ Wrote {}", provider_path.display());

    // Launch Goose
    let goose_app = std::path::Path::new("/Applications/Goose.app");
    if goose_app.exists() {
        eprintln!("🪿 Launching Goose.app...");
        std::process::Command::new("open")
            .arg("-a")
            .arg(goose_app)
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .spawn()?;
        // Goose.app is a GUI — can't wait for it. Mesh stays running.
        if _mesh_child.is_some() {
            eprintln!(
                "ℹ️  mesh-llm node running in background (kill manually or use `mesh-llm stop`)"
            );
        }
    } else {
        eprintln!("🪿 Launching goose session...");
        let status = std::process::Command::new("goose")
            .arg("session")
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("goose exited with {s}"),
            Err(_) => {
                eprintln!("goose not found. Install: https://github.com/block/goose");
                eprintln!("Or run manually:");
                eprintln!("  GOOSE_PROVIDER=mesh GOOSE_MODEL={chosen} goose session");
            }
        }
        // CLI goose exited — clean up mesh if we started it
        if let Some(ref mut c) = _mesh_child {
            eprintln!("🧹 Stopping mesh-llm node we started...");
            let _ = c.kill();
            let _ = c.wait();
        }
    }
    Ok(())
}

async fn run_claude(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (_models, chosen, mut _mesh_child) = check_mesh(&client, port, &model).await?;

    // Configure and launch Claude Code
    // llama-server natively serves the Anthropic /v1/messages API, and
    // mesh-llm's TCP tunnel passes it through transparently. No proxy needed.
    let base_url = format!("http://127.0.0.1:{port}");
    // Settings optimized for local LLMs.
    // CLAUDE_CODE_ATTRIBUTION_HEADER=0 is critical — without it, Claude Code
    // prepends a changing attribution header that invalidates the KV cache on
    // every request, making inference ~90% slower. See:
    // https://unsloth.ai/docs/basics/claude-code
    let settings = serde_json::json!({
        "env": {
            "ANTHROPIC_BASE_URL": &base_url,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": &chosen,
            "CLAUDE_CODE_SUBAGENT_MODEL": &chosen,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "DISABLE_PROMPT_CACHING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1"
        },
        "attribution": {
            "commit": "",
            "pr": ""
        },
        "prefersReducedMotion": true,
        "terminalProgressBarEnabled": false
    });
    let settings_json = serde_json::to_string(&settings)?;

    eprintln!("🚀 Launching Claude Code with {chosen} → {base_url}\n");
    let status = std::process::Command::new("claude")
        .args(["--model", &chosen, "--settings", &settings_json])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("claude exited with {s}"),
        Err(_) => {
            eprintln!("claude not found. Install: https://docs.anthropic.com/en/docs/claude-code");
            eprintln!("Or run manually:");
            eprintln!("  ANTHROPIC_BASE_URL={base_url} ANTHROPIC_API_KEY= claude --model {chosen}");
        }
    }
    // Claude exited — clean up mesh if we started it
    if let Some(ref mut c) = _mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

async fn run_blackboard(
    text: Option<String>,
    search: Option<String>,
    from: Option<String>,
    since_hours: Option<f64>,
    limit: usize,
    port: u16,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let base = format!("http://127.0.0.1:{port}");

    // Quick connectivity check
    let status_resp = client.get(format!("{base}/api/status")).send().await;
    if status_resp.is_err() {
        eprintln!("No mesh-llm node running on port {port}.");
        eprintln!();
        eprintln!("Blackboard requires a running mesh node:");
        eprintln!("  Private mesh:  mesh-llm --client  (share the join token printed out)");
        eprintln!("  Join a mesh:   mesh-llm --client --join <token>");
        eprintln!("  Public mesh:   mesh-llm --client --auto");
        eprintln!();
        eprintln!("See https://github.com/michaelneale/mesh-llm for setup guide.");
        std::process::exit(1);
    }

    // Check if blackboard is enabled on this node
    let feed_check = client
        .get(format!("{base}/api/blackboard/feed?limit=1"))
        .send()
        .await;
    if let Ok(resp) = feed_check {
        if resp.status().as_u16() == 404 {
            eprintln!("Mesh is running but blackboard is disabled on that node.");
            eprintln!("Re-enable it in the mesh config if you want to use the blackboard plugin.");
            std::process::exit(1);
        }
    }

    // Default: 24h for feed/search, override with --since
    let default_hours = 24.0;
    let since_secs = {
        let hours = since_hours.unwrap_or(default_hours);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub((hours * 3600.0) as u64)
    };

    // Post a message
    if let Some(msg) = text {
        // PII check
        let issues = blackboard::pii_check(&msg);
        if !issues.is_empty() {
            eprintln!("⚠️  PII/secret issues detected:");
            for issue in &issues {
                eprintln!("   • {issue}");
            }
            eprintln!("Scrubbing and posting...");
        }
        let clean = blackboard::pii_scrub(&msg);

        let body = serde_json::json!({ "text": clean });
        let resp = client
            .post(format!("{base}/api/blackboard/post"))
            .json(&body)
            .send()
            .await
            .context("Cannot reach mesh-llm — is it running?")?;
        if resp.status().is_success() {
            let item: blackboard::BlackboardItem = resp.json().await?;
            eprintln!("📝 Posted (id: {:x})", item.id);
        } else {
            let err = resp.text().await.unwrap_or_default();
            eprintln!("Error: {err}");
        }
        return Ok(());
    }

    // Search
    if let Some(q) = search {
        let resp = client
            .get(format!("{base}/api/blackboard/search"))
            .query(&[
                ("q", q.as_str()),
                ("limit", &limit.to_string()),
                ("since", &since_secs.to_string()),
            ])
            .send()
            .await
            .context("Cannot reach mesh-llm — is it running?")?;
        let items: Vec<blackboard::BlackboardItem> = resp.json().await?;
        if items.is_empty() {
            eprintln!("No results.");
        } else {
            print_blackboard_items(&items);
        }
        return Ok(());
    }

    // Feed (optionally filtered by peer)
    let mut params = vec![
        ("limit", limit.to_string()),
        ("since", since_secs.to_string()),
    ];
    if let Some(ref f) = from {
        params.push(("from", f.clone()));
    }
    let resp = client
        .get(format!("{base}/api/blackboard/feed"))
        .query(&params)
        .send()
        .await
        .context("Cannot reach mesh-llm — is it running?")?;
    let items: Vec<blackboard::BlackboardItem> = resp.json().await?;
    if items.is_empty() {
        eprintln!("Blackboard is empty.");
    } else {
        print_blackboard_items(&items);
    }
    Ok(())
}

fn print_blackboard_items(items: &[blackboard::BlackboardItem]) {
    for item in items {
        let time = chrono_format(item.timestamp);
        println!("{:x} │ {} │ {}", item.id, time, item.from);
        // Indent the text
        for line in item.text.lines() {
            println!("  {line}");
        }
        println!();
    }
}

async fn run_plugin_command(command: &PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        PluginCommand::Install { name } if name == plugin::BLACKBOARD_PLUGIN_ID => {
            eprintln!("Blackboard is auto-registered by mesh-llm. Nothing to install.");
            eprintln!("Disable it with [[plugin]] name = \"blackboard\" enabled = false in the config if needed.");
        }
        PluginCommand::Install { name } => {
            let config = plugin::config_path(cli.config.as_deref())?;
            anyhow::bail!(
                "Plugins are configured as executables in {}. No install step exists for '{}'.",
                config.display(),
                name
            );
        }
        PluginCommand::List => {
            let resolved = load_resolved_plugins(cli)?;
            for spec in resolved.externals {
                println!(
                    "{}\tkind=external\tcommand={}\targs={}",
                    spec.name,
                    spec.command,
                    spec.args.join(" ")
                );
            }
        }
    }
    Ok(())
}

fn chrono_format(ts: u64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let ago = now.saturating_sub(ts);
    if ago < 60 {
        format!("{ago}s ago")
    } else if ago < 3600 {
        format!("{}m ago", ago / 60)
    } else if ago < 86400 {
        format!("{}h ago", ago / 3600)
    } else {
        format!("{}d ago", ago / 86400)
    }
}

fn install_skill() -> Result<()> {
    let skill_content = include_str!("../skills/blackboard/SKILL.md");
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let skill_dir = home.join(".agents").join("skills").join("blackboard");
    std::fs::create_dir_all(&skill_dir)?;
    let skill_path = skill_dir.join("SKILL.md");
    std::fs::write(&skill_path, skill_content)?;
    eprintln!("✅ Installed blackboard skill to {}", skill_path.display());
    eprintln!("   Works with pi, Goose, and other agents that read ~/.agents/skills/");
    eprintln!(
        "   Make sure mesh-llm is running and the blackboard plugin is not disabled in config."
    );
    Ok(())
}

async fn check_for_update() {
    if let Some(latest) = latest_release_version().await {
        if version_newer(&latest, VERSION) {
            eprintln!("💡 Update available: v{VERSION} → v{latest}  https://github.com/michaelneale/mesh-llm/releases");
            eprintln!("   curl -fsSL https://github.com/michaelneale/mesh-llm/releases/latest/download/mesh-bundle.tar.gz | tar xz && mv mesh-bundle/* ~/.local/bin/");
        }
    }
}

pub(crate) async fn latest_release_version() -> Option<String> {
    let url = "https://api.github.com/repos/michaelneale/mesh-llm/releases/latest";
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    let tag = body["tag_name"].as_str()?;
    let latest = tag.trim_start_matches('v').trim();
    if latest.is_empty() {
        None
    } else {
        Some(latest.to_string())
    }
}

pub(crate) fn version_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    parse(a) > parse(b)
}

/// Build the list of models this node is serving for gossip announcement.
/// `resolved_models` comes from explicit `--model` args (may be empty for `--auto`).
/// `model_name` is the actual model we're about to serve (always set).
/// The primary model must always appear in the result.
fn build_serving_list(resolved_models: &[PathBuf], model_name: &str) -> Vec<String> {
    let clean_name = router::strip_split_suffix_owned(model_name);
    let mut all: Vec<String> = resolved_models
        .iter()
        .map(|m| {
            let stem = model_path_name(m).unwrap_or_default();
            // Strip split GGUF suffix: "Model-00001-of-00004" → "Model"
            router::strip_split_suffix_owned(&stem)
        })
        .collect();
    if !all.contains(&clean_name) {
        all.insert(0, clean_name);
    }
    all
}

fn model_path_name(path: &std::path::Path) -> Option<String> {
    if let Some(model_name) = huggingface_snapshot_model_name(path) {
        return Some(model_name);
    }
    if path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.ends_with(".gguf"))
        .unwrap_or(false)
    {
        path.file_stem().map(|s| s.to_string_lossy().to_string())
    } else {
        path.file_name().map(|s| s.to_string_lossy().to_string())
    }
}

fn huggingface_snapshot_model_name(path: &std::path::Path) -> Option<String> {
    let snapshots_dir = path.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }

    let repo_dir = snapshots_dir.parent()?.file_name()?.to_str()?;
    let repo_id = repo_dir.strip_prefix("models--")?.replace("--", "/");
    repo_id.rsplit('/').next().map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_build_serving_list_auto_no_resolved() {
        // --auto: resolved_models is empty, model picked dynamically
        let resolved: Vec<PathBuf> = vec![];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
    }

    #[test]
    fn test_build_serving_list_explicit_single_model() {
        // --model Qwen3-30B: resolved_models has the model
        let resolved = vec![PathBuf::from("/home/.models/Qwen3-30B-A3B-Q4_K_M.gguf")];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
        // No duplicate
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_explicit_multi_model() {
        // --model A --model B: both resolved
        let resolved = vec![
            PathBuf::from("/home/.models/Qwen3-30B-A3B-Q4_K_M.gguf"),
            PathBuf::from("/home/.models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
        ];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(
            result,
            vec!["Qwen3-30B-A3B-Q4_K_M", "Qwen2.5-Coder-7B-Instruct-Q4_K_M"]
        );
    }

    #[test]
    fn test_build_serving_list_split_gguf() {
        // Split GGUF: file is "MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"
        // Serving list should strip the split suffix
        let resolved = vec![PathBuf::from(
            "/home/.models/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_split_gguf_model_name_also_has_suffix() {
        // If model_name also has the suffix (from dynamic pick), strip it too
        let resolved = vec![PathBuf::from(
            "/home/.models/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M-00001-of-00004");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_mlx_directory_keeps_full_name() {
        let resolved = vec![PathBuf::from(
            "/home/.models/mlx/Qwen2.5-0.5B-Instruct-4bit",
        )];
        let result = build_serving_list(&resolved, "Qwen2.5-0.5B-Instruct-4bit");
        assert_eq!(result, vec!["Qwen2.5-0.5B-Instruct-4bit"]);
    }

    #[test]
    fn test_model_path_name_huggingface_snapshot_uses_repo_name() {
        let path = PathBuf::from(
            "/home/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/863c846a9ac6fad4e49e1743d52984dff262e953",
        );
        assert_eq!(
            model_path_name(&path).as_deref(),
            Some("Llama-3.2-1B-Instruct-bf16")
        );
    }

    #[test]
    fn test_version_newer() {
        assert!(version_newer("0.33.1", "0.33.0"));
        assert!(!version_newer("0.33.0", "0.33.0"));
        assert!(!version_newer("0.32.0", "0.33.0"));
    }
}
