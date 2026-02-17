mod console;
mod download;
mod election;
mod launch;
mod mesh;
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

    /// Path to GGUF model file. Starts rpc-server and enters auto-election.
    #[arg(long)]
    model: Option<PathBuf>,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?),
        )
        .init();

    let mut cli = Cli::parse();

    // Subcommand dispatch
    if let Some(Command::Download { name, draft }) = &cli.command {
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

    // --- Validation ---
    if cli.client && cli.join.is_empty() {
        anyhow::bail!("--client requires --join to connect to a mesh");
    }
    if cli.client && cli.model.is_some() {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    if cli.model.is_none() && cli.join.is_empty() && !cli.client {
        anyhow::bail!("--model is required (or use --join to auto-discover, or --client for lite mode)");
    }

    // --- Lite client mode ---
    if cli.client {
        return run_client(cli).await;
    }

    // --- Resolve model (may be deferred if joining without --model) ---
    let model = match &cli.model {
        Some(m) => Some(resolve_model(m).await?),
        None => None, // will learn from mesh after joining
    };

    let bin_dir = match &cli.bin_dir {
        Some(d) => d.clone(),
        None => detect_bin_dir()?,
    };

    // Auto-detect draft model from catalog if not explicitly set
    if cli.draft.is_none() && !cli.no_draft {
        if let Some(ref m) = model {
            if let Some(draft_path) = auto_detect_draft(m) {
                eprintln!("Auto-detected draft model: {}", draft_path.display());
                cli.draft = Some(draft_path);
            }
        }
    }

    run_auto(cli, model, bin_dir).await
}

/// Resolve a model path: local file, catalog name, or HuggingFace URL.
///
/// - If it's an existing file path, use it directly.
/// - If it matches a catalog entry name, download if needed.
/// - If it looks like a HF URL (https://huggingface.co/...), download to ~/.models/.
/// - If it looks like a HF repo shorthand (org/repo/file.gguf), construct URL and download.
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
        // Try as bare filename in ~/.models/
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

    // HF shorthand: org/repo/file.gguf or org/repo/resolve/main/file.gguf
    if s.contains('/') && s.ends_with(".gguf") {
        let url = if s.contains("/resolve/") {
            format!("https://huggingface.co/{}", s)
        } else {
            // org/repo/file.gguf -> https://huggingface.co/org/repo/resolve/main/file.gguf
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

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
/// mesh-llm owns :port and proxies to llama-server (local or remote).
async fn run_auto(mut cli: Cli, model: Option<PathBuf>, bin_dir: PathBuf) -> Result<()> {
    let api_port = cli.port;
    let console_port = cli.console;

    // Start mesh node first (needed for join-and-discover flow)
    let (node, channels) = mesh::Node::start(NodeRole::Worker, &cli.relay, cli.bind_port, cli.max_vram).await?;
    let token = node.invite_token();

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
    } else {
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for peers to join...");
    }

    // Resolve model — either from CLI or discovered from mesh peers
    let model = match model {
        Some(m) => m,
        None => {
            // Learn model from mesh gossip
            eprintln!("No --model specified, discovering from mesh...");
            let source = tokio::time::timeout(
                std::time::Duration::from_secs(15),
                async {
                    loop {
                        if let Some(src) = node.peer_model_source().await {
                            return src;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }
                }
            ).await.map_err(|_| anyhow::anyhow!("Timed out waiting for model info from mesh peers. Use --model to specify explicitly."))?;

            eprintln!("Mesh model: {source}");
            resolve_model(std::path::Path::new(&source)).await?
        }
    };

    // Set model source for gossip (so other joiners can discover it too)
    let model_source = if let Some(ref m) = cli.model {
        // Use the original CLI input — could be catalog name, HF URL, etc.
        m.to_string_lossy().to_string()
    } else {
        // We discovered it — propagate the filename stem as catalog-resolvable name
        model.file_stem().unwrap_or_default().to_string_lossy().to_string()
    };
    node.set_model_source(model_source).await;

    // Auto-detect draft if we didn't have --model originally (deferred detection)
    if cli.draft.is_none() && !cli.no_draft {
        if let Some(draft_path) = auto_detect_draft(&model) {
            eprintln!("Auto-detected draft model: {}", draft_path.display());
            cli.draft = Some(draft_path);
        }
    }

    // Start rpc-server — every node contributes GPU
    let rpc_port = launch::start_rpc_server(
        &bin_dir, cli.device.as_deref(), Some(&model),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    // Pass None for http_port — we'll handle HTTP proxying ourselves
    let tunnel_mgr = tunnel::Manager::start(
        node.clone(), rpc_port, channels.rpc, channels.http,
    ).await?;

    // Advertise local models
    let model_name = model.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    node.set_models(vec![model_name]).await;

    // Election publishes where llama-server is via this channel
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::InferenceTarget::None);

    // API proxy: mesh-llm owns :api_port, forwards to wherever llama-server is
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    tokio::spawn(async move {
        api_proxy(proxy_node, api_port, proxy_rx).await;
    });

    // Console (optional)
    let model_name_str = model.file_stem()
        .unwrap_or_default().to_string_lossy().to_string();
    let console_state = if let Some(cport) = console_port {
        let cs = console::ConsoleState::new(node.clone(), model_name_str.clone(), api_port);
        if let Some(draft) = &cli.draft {
            let dn = draft.file_stem().unwrap_or_default().to_string_lossy().to_string();
            cs.set_draft_name(dn).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        tokio::spawn(async move {
            console::start(cport, cs2, console_rx).await;
        });
        Some(cs)
    } else {
        None
    };

    // Election loop
    eprintln!("Entering auto-election (highest VRAM becomes host)...");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    let force_split = cli.split;
    let model_name_for_cb = model_name_str.clone();
    tokio::spawn(async move {
        election::election_loop(
            node2, tunnel_mgr2, rpc_port, bin_dir2, model2, draft2, draft_max, force_split, target_tx,
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
                // Update console if running
                if let Some(ref cs) = console_state {
                    let cs = cs.clone();
                    tokio::spawn(async move {
                        cs.update(is_host, llama_ready).await;
                    });
                }
            },
        ).await;
    });

    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");
    launch::kill_llama_server().await;
    Ok(())
}

/// Run as a lite client: join mesh, find host, expose local HTTP proxy.
async fn run_client(cli: Cli) -> Result<()> {
    let local_port = cli.port;

    let (node, _channels) = mesh::Node::start(NodeRole::Client, &cli.relay, cli.bind_port, None).await?;
    let token = node.invite_token();

    let mut joined = false;
    for t in &cli.join {
        match node.join(t).await {
            Ok(()) => { eprintln!("Joined mesh"); joined = true; break; }
            Err(e) => tracing::warn!("Failed to join via token: {e}"),
        }
    }
    if !joined {
        anyhow::bail!("Failed to join any peer in the mesh");
    }
    eprintln!("This node's token (for others to join): {token}");

    eprintln!("Looking for a host node with llama-server...");
    let host = node.wait_for_host().await?;
    let host_id = host.id;
    if let NodeRole::Host { http_port } = &host.role {
        eprintln!("Found host {} (llama-server on port {})", host_id.fmt_short(), http_port);
    } else {
        eprintln!("Found host {}", host_id.fmt_short());
    }

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{local_port}")).await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    eprintln!("Lite client ready: http://localhost:{local_port}");
    eprintln!("  → tunneling to host {}", host_id.fmt_short());
    eprintln!();
    eprintln!("Launch with pi:");
    eprintln!("  OPENAI_API_KEY=mesh OPENAI_BASE_URL=http://localhost:{local_port}/v1 pi --provider openai");
    eprintln!();
    eprintln!("Launch with goose:");
    eprintln!("  GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{local_port} OPENAI_API_KEY=mesh goose session");

    // Console (optional)
    if let Some(cport) = cli.console {
        let model_name = host.models.first().cloned().unwrap_or_default();
        let cs = console::ConsoleState::new(node.clone(), model_name, local_port);
        cs.set_client(true).await;
        cs.update(false, true).await; // not host, but ready (tunneling to host)
        tokio::spawn(async move {
            let (_tx, rx) = tokio::sync::watch::channel(
                election::InferenceTarget::Remote(host_id)
            );
            console::start(cport, cs, rx).await;
        });
    }

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Client connection from {addr}");
                let node = node.clone();
                tokio::spawn(async move {
                    match node.open_http_tunnel(host_id).await {
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
                break;
            }
        }
    }

    Ok(())
}

/// Unified API proxy. mesh-llm owns :port and forwards every request
/// to wherever llama-server is running, based on the election target.
///   - Local(port): llama-server on this machine → TCP proxy to localhost:port
///   - Remote(peer): llama-server on another node → QUIC HTTP tunnel
///   - None: no server yet → 503
async fn api_proxy(node: mesh::Node, port: u16, target_rx: tokio::sync::watch::Receiver<election::InferenceTarget>) {
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

        let target = target_rx.borrow().clone();
        let node = node.clone();

        tokio::spawn(async move {
            match target {
                election::InferenceTarget::Local(llama_port) => {
                    // Proxy to local llama-server
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
                    // Proxy via QUIC tunnel to remote host
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
        });
    }
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

    // Same directory (bundle layout)
    if dir.join("rpc-server").exists() && dir.join("llama-server").exists() {
        return Ok(dir.to_path_buf());
    }
    // Dev layout: ../llama.cpp/build/bin/
    let dev = dir.join("../llama.cpp/build/bin");
    if dev.join("rpc-server").exists() && dev.join("llama-server").exists() {
        return Ok(dev.canonicalize()?);
    }
    // Cargo target/release/ layout
    let cargo = dir.join("../../../llama.cpp/build/bin");
    if cargo.join("rpc-server").exists() && cargo.join("llama-server").exists() {
        return Ok(cargo.canonicalize()?);
    }

    Ok(dir.to_path_buf())
}

/// Update ~/.pi/agent/models.json to include a "mesh" provider pointing at localhost.
/// Creates the file if it doesn't exist, or updates the mesh provider's model list.
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

    // Build the mesh provider entry
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

    // Write back
    if let Some(parent) = models_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&root) {
        if let Err(e) = std::fs::write(&models_path, json) {
            tracing::warn!("Failed to update {}: {e}", models_path.display());
        }
    }
}
