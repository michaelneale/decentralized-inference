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
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(bin_dir: &Path, device: Option<&str>, gguf_path: Option<&Path>) -> Result<u16> {
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

    let mut args = vec!["-d".to_string(), device.clone(), "-p".to_string(), port.to_string()];
    if let Some(path) = gguf_path {
        args.push("--gguf".to_string());
        args.push(path.to_string_lossy().to_string());
        tracing::info!("rpc-server will load weights from local GGUF: {}", path.display());
    }

    let mut child = Command::new(&rpc_server)
        .args(&args)
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

/// Kill all running llama-server processes.
pub async fn kill_llama_server() {
    let _ = std::process::Command::new("pkill").args(["-f", "llama-server"]).status();
    // Wait for the process to actually exit and release the port
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        // Check if any llama-server is still running
        let output = std::process::Command::new("pgrep").args(["-f", "llama-server"]).output();
        if let Ok(o) = output {
            if o.stdout.is_empty() { return; }
        } else {
            return;
        }
    }
    // Force kill if still alive after 5s
    let _ = std::process::Command::new("pkill").args(["-9", "-f", "llama-server"]).status();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
pub async fn start_llama_server(
    bin_dir: &Path,
    model: &Path,
    http_port: u16,
    tunnel_ports: &[u16],
    tensor_split: Option<&str>,
    draft: Option<&Path>,
    draft_max: u16,
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
        "Starting llama-server on :{http_port} with model {} and --rpc {}",
        model.display(),
        rpc_arg
    );

    let log_file = std::fs::File::create("/tmp/mesh-inference-llama-server.log")
        .context("Failed to create llama-server log file")?;
    let log_file2 = log_file.try_clone()?;

    // llama-server always uses --rpc, even solo.
    // The host's own rpc-server is always in the list.
    let mut args = vec![
        "-m".to_string(), model.to_string_lossy().to_string(),
    ];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    args.extend_from_slice(&[
        "-ngl".to_string(), "99".to_string(),
        "-fit".to_string(), "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(), "0.0.0.0".to_string(),
        "--port".to_string(), http_port.to_string(),
    ]);
    if let Some(ts) = tensor_split {
        args.push("--tensor-split".to_string());
        args.push(ts.to_string());
    }
    if let Some(draft_path) = draft {
        if draft_path.exists() {
            args.push("-md".to_string());
            args.push(draft_path.to_string_lossy().to_string());
            args.push("-ngld".to_string());
            args.push("99".to_string());
            args.push("--device-draft".to_string());
            args.push("MTL0".to_string());
            args.push("--draft-max".to_string());
            args.push(draft_max.to_string());
            tracing::info!("Speculative decoding: draft={}, draft-max={}", draft_path.display(), draft_max);
        } else {
            tracing::warn!("Draft model not found at {}, skipping speculative decoding", draft_path.display());
        }
    }
    let mut child = Command::new(&llama_server)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start llama-server at {}", llama_server.display()))?;

    // Wait for health check
    let url = format!("http://localhost:{http_port}/health");
    for i in 0..600 {
        if i > 0 && i % 10 == 0 {
            let bytes = crate::tunnel::bytes_transferred();
            let kb = bytes as f64 / 1024.0;
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let transferred = if gb >= 1.0 {
                format!("{gb:.1} GB")
            } else if mb >= 1.0 {
                format!("{mb:.1} MB")
            } else {
                format!("{kb:.0} KB")
            };
            tracing::info!("Still waiting for llama-server to load model... ({i}s, {transferred} transferred)");
        }
        if reqwest_health_check(&url).await {
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("llama-server failed to become healthy within 600s");
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
