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
#[command(name = "mesh-inference", about = "P2P mesh for distributed llama.cpp inference over QUIC")]
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
    /// Defaults to the same directory as the mesh-inference binary itself.
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
    #[arg(long)]
    draft: Option<PathBuf>,

    /// Max draft tokens for speculative decoding (default: 8).
    #[arg(long, default_value = "8")]
    draft_max: u16,

    /// Override iroh relay URLs (e.g. --relay https://staging-use1-1.relay.iroh.network./).
    /// Can be specified multiple times. Without this, iroh uses its built-in defaults.
    #[arg(long, global = true)]
    relay: Vec<String>,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, global = true)]
    bind_port: Option<u16>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Launch the web console for interactive mesh management
    Console {
        /// Port for the web console (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
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

    let cli = Cli::parse();

    // Subcommand dispatch
    if let Some(Command::Console { port }) = &cli.command {
        return console::run(*port, cli.join.clone()).await;
    }
    if let Some(Command::Download { name, draft }) = &cli.command {
        match name {
            Some(query) => {
                let model = download::find_model(query)
                    .ok_or_else(|| anyhow::anyhow!("No model matching '{}' in catalog. Run `mesh-inference download` to list.", query))?;
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

    // --- Lite client mode ---
    if cli.client {
        return run_client(cli).await;
    }

    // --- Need a model for worker or host ---
    let model = match &cli.model {
        Some(m) => m.clone(),
        None => anyhow::bail!("--model is required (or use --client for lite mode)"),
    };

    let bin_dir = match &cli.bin_dir {
        Some(d) => d.clone(),
        None => detect_bin_dir()?,
    };

    run_auto(cli, model, bin_dir).await
}

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
/// mesh-inference owns :port and proxies to llama-server (local or remote).
async fn run_auto(cli: Cli, model: PathBuf, bin_dir: PathBuf) -> Result<()> {
    let api_port = cli.port;

    // Start rpc-server — every node contributes GPU
    let rpc_port = launch::start_rpc_server(
        &bin_dir, cli.device.as_deref(), Some(&model),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    // Start mesh node as Worker (election may promote to Host)
    let (node, channels) = mesh::Node::start(NodeRole::Worker, &cli.relay, cli.bind_port).await?;
    let token = node.invite_token();

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

    // Join mesh or print token
    if cli.join.is_empty() {
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for peers to join...");
    } else {
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
    }

    // Election publishes where llama-server is via this channel
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::InferenceTarget::None);

    // API proxy: mesh-inference owns :api_port, forwards to wherever llama-server is
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    tokio::spawn(async move {
        api_proxy(proxy_node, api_port, proxy_rx).await;
    });

    // Election loop
    eprintln!("Entering auto-election (highest VRAM becomes host)...");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    tokio::spawn(async move {
        election::election_loop(
            node2, tunnel_mgr2, rpc_port, bin_dir2, model2, draft2, draft_max, target_tx,
            move |is_host, llama_ready| {
                if is_host && llama_ready {
                    eprintln!("  API: http://localhost:{api_port}");
                } else if is_host {
                    eprintln!("⏳ Starting llama-server...");
                } else {
                    eprintln!("  API: http://localhost:{api_port} (proxied to host)");
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

    let (node, _channels) = mesh::Node::start(NodeRole::Client, &cli.relay, cli.bind_port).await?;
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
    eprintln!("Use it like any OpenAI-compatible API:");
    eprintln!("  curl http://localhost:{local_port}/v1/models");

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

/// Unified API proxy. mesh-inference owns :port and forwards every request
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
