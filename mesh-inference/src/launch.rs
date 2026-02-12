//! Process management for llama.cpp binaries.
//!
//! Starts rpc-server and optionally llama-server as child processes,
//! wired up to the mesh tunnel ports.

use anyhow::{Context, Result};
use std::path::Path;
use tokio::process::Command;
use tokio::net::TcpListener;

/// Start a local rpc-server and return the port it's listening on.
/// Picks an available port automatically.
pub async fn start_rpc_server(bin_dir: &Path, device: Option<&str>) -> Result<u16> {
    let rpc_server = bin_dir.join("rpc-server");
    anyhow::ensure!(
        rpc_server.exists(),
        "rpc-server not found at {}. Build llama.cpp with -DGGML_RPC=ON first.",
        rpc_server.display()
    );

    // Find a free port
    let port = find_free_port().await?;

    let device = device.map(|s| s.to_string()).unwrap_or_else(detect_device);

    tracing::info!("Starting rpc-server on :{port} (device: {device})");

    let rpc_log = format!("/tmp/mesh-inference-rpc-{port}.log");
    let rpc_log_file = std::fs::File::create(&rpc_log)
        .with_context(|| format!("Failed to create rpc-server log file {rpc_log}"))?;
    let rpc_log_file2 = rpc_log_file.try_clone()?;

    let mut child = Command::new(&rpc_server)
        .args(["-d", &device, "-p", &port.to_string()])
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        .spawn()
        .with_context(|| format!("Failed to start rpc-server at {}", rpc_server.display()))?;

    // Wait for it to be listening
    for _ in 0..30 {
        if is_port_open(port).await {
            // Detach — let it run in the background
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(port);
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    anyhow::bail!("rpc-server failed to start on port {port} within 15s");
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
pub async fn start_llama_server(
    bin_dir: &Path,
    model: &Path,
    http_port: u16,
    tunnel_ports: &[u16],
) -> Result<()> {
    let llama_server = bin_dir.join("llama-server");
    anyhow::ensure!(
        llama_server.exists(),
        "llama-server not found at {}. Build llama.cpp first.",
        llama_server.display()
    );

    anyhow::ensure!(
        model.exists(),
        "Model not found at {}",
        model.display()
    );

    // Build --rpc argument: all tunnel ports as localhost endpoints
    let rpc_endpoints: Vec<String> = tunnel_ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect();
    let rpc_arg = rpc_endpoints.join(",");

    tracing::info!(
        "Starting llama-server on :{http_port} with model {} and {} RPC endpoints",
        model.display(),
        tunnel_ports.len()
    );

    let log_file = std::fs::File::create("/tmp/mesh-inference-llama-server.log")
        .context("Failed to create llama-server log file")?;
    let log_file2 = log_file.try_clone()?;

    let mut child = Command::new(&llama_server)
        .args([
            "-m", &model.to_string_lossy(),
            "--rpc", &rpc_arg,
            "-ngl", "99",
            "--host", "0.0.0.0",
            "--port", &http_port.to_string(),
        ])
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start llama-server at {}", llama_server.display()))?;

    // Wait for health check
    let url = format!("http://localhost:{http_port}/health");
    for i in 0..120 {
        if i > 0 && i % 10 == 0 {
            tracing::info!("Still waiting for llama-server to load model... ({i}s)");
        }
        if reqwest_health_check(&url).await {
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("llama-server failed to become healthy within 120s");
}

/// Find an available TCP port
async fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Check if a port is accepting connections
async fn is_port_open(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

/// Detect the best available compute device
fn detect_device() -> String {
    // On macOS with Metal, use MTL0. Otherwise CPU.
    if cfg!(target_os = "macos") {
        "MTL0".to_string()
    } else {
        // Could detect CUDA here in the future
        "CPU".to_string()
    }
}

/// Simple HTTP health check (avoid adding reqwest as a dep — just use TCP + raw HTTP)
async fn reqwest_health_check(url: &str) -> bool {
    // Parse host:port from URL
    let url = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = url.split_once('/').unwrap_or((url, ""));
    let path = format!("/{path}");

    let Ok(mut stream) = tokio::net::TcpStream::connect(host_port).await else {
        return false;
    };

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }

    let mut response = vec![0u8; 1024];
    let Ok(n) = stream.read(&mut response).await else {
        return false;
    };

    let response = String::from_utf8_lossy(&response[..n]);
    response.contains("200 OK")
}

use tokio::io::{AsyncReadExt, AsyncWriteExt};
