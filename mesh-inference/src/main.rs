mod launch;
mod mesh;
mod rewrite;
mod tunnel;

use anyhow::{Context, Result};
use clap::Parser;
use mesh::NodeRole;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mesh-inference", about = "P2P mesh for distributed llama.cpp inference over QUIC")]
struct Cli {
    /// Join an existing mesh via an invite token (base64-encoded endpoint address).
    /// Can be specified multiple times — only one needs to be reachable.
    #[arg(long, short)]
    join: Vec<String>,

    /// Start llama-server and expose an OpenAI-compatible HTTP API on this port.
    /// Requires --model. Waits for at least --min-peers peers before starting.
    #[arg(long)]
    serve: Option<u16>,

    /// Path to GGUF model file. Required with --serve.
    /// If provided without --serve, the local rpc-server uses it for zero-transfer loading.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Minimum number of peers to wait for before starting llama-server (default: 1)
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
    /// Controls how model layers are distributed across GPU backends.
    /// Order: remote RPC backends first, then local GPU (if available).
    #[arg(long)]
    tensor_split: Option<String>,

    /// Run as a lite client — no GPU, no rpc-server, no model needed.
    /// Joins the mesh and exposes a local HTTP endpoint that tunnels to
    /// a host node's llama-server. Any OpenAI-compatible client can use it.
    #[arg(long)]
    client: bool,

    /// Local port for the lite client's HTTP endpoint (default: 8080).
    /// Only used with --client.
    #[arg(long, default_value = "8080")]
    port: u16,
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

    if cli.serve.is_some() && cli.model.is_none() {
        anyhow::bail!("--model is required when using --serve");
    }
    if cli.client && cli.serve.is_some() {
        anyhow::bail!("--client and --serve are mutually exclusive");
    }
    if cli.client && cli.join.is_empty() {
        anyhow::bail!("--client requires --join to connect to a mesh");
    }

    // Determine role
    let role = if cli.client {
        NodeRole::Client
    } else if let Some(http_port) = cli.serve {
        NodeRole::Host { http_port }
    } else {
        NodeRole::Worker
    };

    // Client mode: lightweight — no rpc-server, no model, just mesh + HTTP tunnel
    if cli.client {
        return run_client(cli, role).await;
    }

    // Resolve bin_dir
    let bin_dir = match cli.bin_dir {
        Some(d) => d,
        None => std::env::current_exe()
            .context("Failed to determine own binary path")?
            .parent()
            .context("Binary has no parent directory")?
            .to_path_buf(),
    };

    // 1. Start local rpc-server (every worker/host node offers compute)
    let rpc_port = launch::start_rpc_server(
        &bin_dir,
        cli.device.as_deref(),
        cli.model.as_deref(),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    // 2. Start the mesh node
    let (node, channels) = mesh::Node::start(role.clone()).await?;
    let token = node.invite_token();

    // 3. Start tunnel manager (HTTP port set later for host after llama-server starts)
    let http_port = if let NodeRole::Host { http_port } = &role { Some(*http_port) } else { None };
    let tunnel_mgr = tunnel::Manager::start(
        node.clone(), rpc_port, channels.rpc, http_port, channels.http,
    ).await?;

    // 4. Join the mesh via provided tokens
    if cli.join.is_empty() {
        eprintln!("Invite token: {token}");
        eprintln!("Waiting for inbound connections...");
    } else {
        let mut joined = false;
        for t in &cli.join {
            match node.join(t).await {
                Ok(()) => {
                    eprintln!("Joined mesh");
                    joined = true;
                    break;
                }
                Err(e) => {
                    tracing::warn!("Failed to join via token: {e}");
                }
            }
        }
        if !joined {
            eprintln!("Failed to join any peer, running standalone");
        }
        eprintln!("This node's token (for others to join): {token}");
    }

    // 5. If --serve, wait for peers then launch llama-server
    if let Some(http_port) = cli.serve {
        let model = cli.model.unwrap();
        eprintln!("Waiting for {} peer(s) with active tunnels...", cli.min_peers);
        tunnel_mgr.wait_for_peers(cli.min_peers).await?;

        // Small delay for tunnel port allocation to complete after peer discovery
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let tunnel_ports = tunnel_mgr.peer_ports().await;
        eprintln!(
            "Got {} peer(s), starting llama-server with {} remote RPC endpoint(s)",
            tunnel_ports.len(),
            tunnel_ports.len()
        );

        // 5a. Broadcast our tunnel map so workers can build B2B rewrite maps
        let my_tunnel_map = tunnel_mgr.peer_ports_map().await;
        eprintln!("Broadcasting tunnel map ({} entries) for B2B rewriting", my_tunnel_map.len());
        node.broadcast_tunnel_map(my_tunnel_map).await?;

        // 5b. Wait for workers to send their tunnel maps, then build our rewrite map
        node.wait_for_tunnel_maps(tunnel_ports.len(), std::time::Duration::from_secs(15)).await?;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;

        // 5c. Launch llama-server
        launch::start_llama_server(&bin_dir, &model, http_port, &tunnel_ports, cli.tensor_split.as_deref()).await?;
        eprintln!("llama-server ready: http://localhost:{http_port}");
    } else {
        // Worker node: broadcast tunnel map and update rewrite maps in background
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let my_tunnel_map = tunnel_mgr.peer_ports_map().await;
        if !my_tunnel_map.is_empty() {
            eprintln!("Broadcasting tunnel map ({} entries) for B2B", my_tunnel_map.len());
            node.broadcast_tunnel_map(my_tunnel_map).await?;
        }

        let node_bg = node.clone();
        let tunnel_mgr_bg = tunnel_mgr.clone();
        tokio::spawn(async move {
            let mut last_count = 0;
            for _ in 0..30 {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let remote_maps = node_bg.all_remote_tunnel_maps().await;
                let count = remote_maps.len();
                if count > last_count {
                    tunnel_mgr_bg.update_rewrite_map(&remote_maps).await;
                    last_count = count;
                }
            }
        });
    }

    // Run until interrupted
    eprintln!("Running. Ctrl-C to stop.");
    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");

    Ok(())
}

/// Run as a lite client: join mesh, find host, expose local HTTP proxy.
async fn run_client(cli: Cli, role: NodeRole) -> Result<()> {
    let local_port = cli.port;

    // 1. Start mesh node (no rpc-server, no model)
    let (node, _channels) = mesh::Node::start(role).await?;
    let token = node.invite_token();

    // 2. Join mesh
    let mut joined = false;
    for t in &cli.join {
        match node.join(t).await {
            Ok(()) => {
                eprintln!("Joined mesh");
                joined = true;
                break;
            }
            Err(e) => {
                tracing::warn!("Failed to join via token: {e}");
            }
        }
    }
    if !joined {
        anyhow::bail!("Failed to join any peer in the mesh");
    }
    eprintln!("This node's token (for others to join): {token}");

    // 3. Wait for a host node to appear
    eprintln!("Looking for a host node with llama-server...");
    let host = node.wait_for_host().await?;
    let host_id = host.id;
    if let NodeRole::Host { http_port } = &host.role {
        eprintln!("Found host {} (llama-server on port {})", host_id.fmt_short(), http_port);
    } else {
        eprintln!("Found host {}", host_id.fmt_short());
    }

    // 4. Start local HTTP listener
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{local_port}")).await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    eprintln!("Lite client ready: http://localhost:{local_port}");
    eprintln!("  → tunneling to host {}", host_id.fmt_short());
    eprintln!();
    eprintln!("Use it like any OpenAI-compatible API:");
    eprintln!("  curl http://localhost:{local_port}/v1/models");
    eprintln!("  curl http://localhost:{local_port}/v1/chat/completions \\");
    eprintln!("    -H 'Content-Type: application/json' \\");
    eprintln!("    -d '{{\"model\":\"test\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'");

    // 5. Accept local connections and tunnel each to the host
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
                        Err(e) => {
                            tracing::warn!("Failed to open HTTP tunnel to host: {e}");
                        }
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
