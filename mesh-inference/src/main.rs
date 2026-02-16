mod console;
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

    /// Force this node to be the host on a specific HTTP port.
    /// Without this flag, the mesh auto-elects a host based on VRAM.
    #[arg(long)]
    serve: Option<u16>,

    /// Path to GGUF model file. Starts rpc-server and enters auto-election.
    /// Both workers and the elected host need this.
    #[arg(long)]
    model: Option<PathBuf>,

    /// HTTP port for llama-server when elected as host (default: 8090).
    /// Only used in auto-election mode (--model without --serve).
    #[arg(long, default_value = "8090")]
    port: u16,

    /// Minimum number of peers to wait for before starting llama-server.
    /// Only used with --serve (manual host mode).
    #[arg(long, default_value = "1")]
    min_peers: usize,

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

    // --- Validation ---
    if cli.serve.is_some() && cli.model.is_none() {
        anyhow::bail!("--model is required when using --serve");
    }
    if cli.client && cli.serve.is_some() {
        anyhow::bail!("--client and --serve are mutually exclusive");
    }
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

    if cli.serve.is_some() {
        // Legacy manual host mode: --serve PORT
        run_manual_host(cli, model, bin_dir).await
    } else {
        // Auto-election mode: --model (with optional --join)
        run_auto(cli, model, bin_dir).await
    }
}

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
async fn run_auto(cli: Cli, model: PathBuf, bin_dir: PathBuf) -> Result<()> {
    let llama_port = cli.port;

    // Start rpc-server — every node contributes GPU
    let rpc_port = launch::start_rpc_server(
        &bin_dir, cli.device.as_deref(), Some(&model),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    // Start mesh node as Worker (election may promote to Host)
    let (node, channels) = mesh::Node::start(NodeRole::Worker, &cli.relay, cli.bind_port).await?;
    let token = node.invite_token();

    let tunnel_mgr = tunnel::Manager::start(
        node.clone(), rpc_port, channels.rpc, Some(llama_port), channels.http,
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

    // Enter election loop — blocks until ctrl-c
    eprintln!("Entering auto-election (highest VRAM becomes host)...");

    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    tokio::spawn(async move {
        election::election_loop(
            node2, tunnel_mgr2, bin_dir2, model2, llama_port,
            |is_host, llama_ready| {
                if is_host && llama_ready {
                    eprintln!("✅ This node is HOST — llama-server ready on port {llama_port}");
                } else if is_host {
                    eprintln!("⏳ This node is HOST — llama-server starting...");
                } else {
                    eprintln!("📡 This node is WORKER — contributing GPU to the mesh");
                }
            },
        ).await;
    });

    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");
    launch::kill_llama_server();
    Ok(())
}

/// Legacy manual host mode: --serve PORT forces this node to be host.
async fn run_manual_host(cli: Cli, model: PathBuf, bin_dir: PathBuf) -> Result<()> {
    let http_port = cli.serve.unwrap();

    let rpc_port = launch::start_rpc_server(
        &bin_dir, cli.device.as_deref(), Some(&model),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    let role = NodeRole::Host { http_port };
    let (node, channels) = mesh::Node::start(role.clone(), &cli.relay, cli.bind_port).await?;
    let token = node.invite_token();

    let tunnel_mgr = tunnel::Manager::start(
        node.clone(), rpc_port, channels.rpc, Some(http_port), channels.http,
    ).await?;

    if cli.join.is_empty() {
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for inbound connections...");
    } else {
        let mut joined = false;
        for t in &cli.join {
            match node.join(t).await {
                Ok(()) => { eprintln!("Joined mesh"); joined = true; break; }
                Err(e) => tracing::warn!("Failed to join via token: {e}"),
            }
        }
        if !joined {
            eprintln!("Failed to join any peer, running standalone");
        }
        eprintln!("This node's token (for others to join): {token}");
    }

    eprintln!("Waiting for {} peer(s) with active tunnels...", cli.min_peers);
    tunnel_mgr.wait_for_peers(cli.min_peers).await?;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let tunnel_ports = tunnel_mgr.peer_ports().await;
    eprintln!(
        "Got {} peer(s), starting llama-server with {} remote RPC endpoint(s)",
        tunnel_ports.len(), tunnel_ports.len()
    );

    let my_tunnel_map = tunnel_mgr.peer_ports_map().await;
    eprintln!("Broadcasting tunnel map ({} entries) for B2B rewriting", my_tunnel_map.len());
    node.broadcast_tunnel_map(my_tunnel_map).await?;
    node.wait_for_tunnel_maps(tunnel_ports.len(), std::time::Duration::from_secs(15)).await?;
    let remote_maps = node.all_remote_tunnel_maps().await;
    tunnel_mgr.update_rewrite_map(&remote_maps).await;

    launch::start_llama_server(
        &bin_dir, &model, http_port, &tunnel_ports, cli.tensor_split.as_deref(),
    ).await?;
    eprintln!("llama-server ready: http://localhost:{http_port}");

    eprintln!("Running. Ctrl-C to stop.");
    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");
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
