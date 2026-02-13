mod launch;
mod mesh;
mod rewrite;
mod tunnel;

use anyhow::{Context, Result};
use clap::Parser;
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

    // Resolve bin_dir: explicit flag, or same directory as our own binary
    let bin_dir = match cli.bin_dir {
        Some(d) => d,
        None => std::env::current_exe()
            .context("Failed to determine own binary path")?
            .parent()
            .context("Binary has no parent directory")?
            .to_path_buf(),
    };

    // 1. Start local rpc-server (every node offers compute)
    let rpc_port = launch::start_rpc_server(
        &bin_dir,
        cli.device.as_deref(),
        cli.model.as_deref(),
    ).await?;
    eprintln!("rpc-server on 127.0.0.1:{rpc_port}");

    // 2. Start the mesh node (returns a channel for inbound tunnel bi-streams)
    let (node, tunnel_stream_rx) = mesh::Node::start().await?;
    let token = node.invite_token();

    // 3. Start tunnel manager
    let tunnel_mgr = tunnel::Manager::start(node.clone(), rpc_port, tunnel_stream_rx).await?;

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
        // Print token so a third node can join via this one
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
        // Only use remote tunnel ports as RPC backends.
        // The orchestrator uses its own GPU directly (local Metal / CUDA) — much faster
        // than going through a local rpc-server since it avoids the RPC round-trip overhead.
        // Remote workers contribute their GPUs via the QUIC tunnel.
        eprintln!(
            "Got {} peer(s), starting llama-server with {} remote RPC endpoint(s)",
            tunnel_ports.len(),
            tunnel_ports.len()
        );

        launch::start_llama_server(&bin_dir, &model, http_port, &tunnel_ports).await?;
        eprintln!("llama-server ready: http://localhost:{http_port}");
    }

    // Run until interrupted
    eprintln!("Running. Ctrl-C to stop.");
    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");

    Ok(())
}
