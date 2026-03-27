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

    let rpc_log = format!("/tmp/mesh-llm-rpc-{port}.log");
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

/// Kill orphan rpc-server processes from previous mesh-llm runs.
/// Only kills rpc-servers with PPID 1 (parent died, adopted by init).
/// Safe to call while a live mesh-llm has its own rpc-server child.
pub async fn kill_orphan_rpc_servers() {
    if let Ok(output) = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,comm"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut killed = 0;
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3
                && parts[2].contains("rpc-server")
                && parts[1] == "1"
            {
                if let Ok(pid) = parts[0].parse::<u32>() {
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .status();
                    killed += 1;
                }
            }
        }
        if killed > 0 {
            eprintln!("🧹 Cleaned up {killed} orphan rpc-server process(es)");
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }
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
/// `model_bytes` is the total GGUF file size, used to select KV cache quantization:
///   - < 5GB: FP16 (default) — small models, KV cache is tiny
///   - 5-50GB: Q8_0 — no measurable quality loss, saves ~50% KV memory
///   - > 50GB: Q4_0 — slight long-context degradation, but these models need every byte
/// Start llama-server. Returns a oneshot receiver that fires when the process exits.
pub async fn start_llama_server(
    bin_dir: &Path,
    model: &Path,
    http_port: u16,
    tunnel_ports: &[u16],
    tensor_split: Option<&str>,
    draft: Option<&Path>,
    draft_max: u16,
    model_bytes: u64,
    my_vram: u64,
    mmproj: Option<&Path>,
    ctx_size_override: Option<u32>,
    total_group_vram: Option<u64>,
) -> Result<tokio::sync::oneshot::Receiver<()>> {
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

    let log_file = std::fs::File::create("/tmp/mesh-llm-llama-server.log")
        .context("Failed to create llama-server log file")?;
    let log_file2 = log_file.try_clone()?;

    // llama-server uses --rpc only for remote workers.
    // Context size: scale to available VRAM on the host node.
    // In split mode (pipeline parallel), each node holds a range of layers
    // and the KV cache for those layers is allocated on the same device.
    // So both weights and KV are distributed. The host only needs VRAM for
    // its share of weights + its share of KV. We estimate the host's weight
    // share proportionally and let llama-server pick the largest -c that fits.
    const GB: u64 = 1_000_000_000;
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        // Split mode: host holds its share of the weights
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        // Local mode: host holds all weights
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size: u32 = if let Some(override_ctx) = ctx_size_override {
        override_ctx
    } else if vram_after_model >= 30 * GB {
        65536  // 30GB+ free: full 64K context
    } else if vram_after_model >= 12 * GB {
        32768  // 12-30GB free: 32K
    } else if vram_after_model >= 6 * GB {
        16384  // 6-12GB free: 16K
    } else if vram_after_model >= 3 * GB {
        8192   // 3-6GB free: 8K
    } else {
        4096   // <3GB free: minimal
    };
    tracing::info!("Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB VRAM, {:.1}GB free{})",
        model_bytes as f64 / GB as f64, host_model_bytes as f64 / GB as f64,
        my_vram as f64 / GB as f64, vram_after_model as f64 / GB as f64,
        if total_group_vram.is_some() { " [split]" } else { "" });

    let mut args = vec![
        "-m".to_string(), model.to_string_lossy().to_string(),
    ];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    args.extend_from_slice(&[
        "-ngl".to_string(), "99".to_string(),
        "-fa".to_string(), "on".to_string(),  // Flash attention: reduces KV memory, enables longer contexts
        "-fit".to_string(), "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(), "0.0.0.0".to_string(),
        "--port".to_string(), http_port.to_string(),
        "-c".to_string(), ctx_size.to_string(),
        // Use deepseek format: thinking goes into reasoning_content field.
        // Goose/OpenAI clients parse this correctly. "none" leaks raw <think>
        // tags into content which is worse.
        "--reasoning-format".to_string(), "deepseek".to_string(),
        // Disable thinking by default. Thinking models (Qwen3, MiniMax) burn
        // 15-80s on hidden reasoning for no quality gain on most tasks, and
        // Qwen3.5-9B is completely broken (reasoning consumes all max_tokens).
        // API users can opt-in per-request with:
        //   "chat_template_kwargs": {"enable_thinking": true}
        "--reasoning-budget".to_string(), "0".to_string(),
    ]);
    // KV cache quantization based on model size:
    //   < 5GB: leave default (FP16) — small models, KV cache is negligible
    //   5-50GB: Q8_0 — essentially lossless, halves KV memory
    //   > 50GB: Q4_0 — slight long-context quality trade, but critical memory savings
    if model_bytes >= 50 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(), "q4_0".to_string(),
            "--cache-type-v".to_string(), "q4_0".to_string(),
        ]);
        tracing::info!("KV cache: Q4_0 (model > 50GB)");
    } else if model_bytes >= 5 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(), "q8_0".to_string(),
            "--cache-type-v".to_string(), "q8_0".to_string(),
        ]);
        tracing::info!("KV cache: Q8_0 (model 5-50GB)");
    }
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
    if let Some(proj) = mmproj {
        if proj.exists() {
            args.push("--mmproj".to_string());
            args.push(proj.to_string_lossy().to_string());
            // Vision images can produce large token batches — need ubatch >= 2048
            args.push("--ubatch-size".to_string());
            args.push("2048".to_string());
            tracing::info!("Vision: mmproj={}", proj.display());
        } else {
            tracing::warn!("mmproj not found at {}, skipping vision", proj.display());
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
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                eprintln!("⚠️  llama-server process exited unexpectedly");
                let _ = death_tx.send(());
            });
            return Ok(death_rx);
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
    if cfg!(target_os = "macos") {
        return "MTL0".to_string();
    }

    // Linux: check for NVIDIA CUDA
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let gpu_count = String::from_utf8_lossy(&output.stdout)
                .lines().filter(|l| !l.trim().is_empty()).count();
            if gpu_count > 0 {
                return "CUDA0".to_string();
            }
        }
    }

    // Linux: check for NVIDIA Tegra/Jetson (tegrastats — Jetson AGX/NX devices support CUDA)
    // nvidia-smi is absent on Tegra; tegrastats is the canonical hardware stats tool.
    if let Ok(mut child) = std::process::Command::new("tegrastats")
        .args(["--interval", "1"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        let _ = child.kill();
        let _ = child.wait();
        return "CUDA0".to_string();
    }

    // Linux: check for AMD ROCm/HIP
    if std::path::Path::new("/opt/rocm").exists() {
        return "HIP0".to_string();
    }

    // Linux: check for Vulkan
    if let Ok(output) = std::process::Command::new("vulkaninfo")
        .args(["--summary"])
        .output()
    {
        if output.status.success() {
            return "Vulkan0".to_string();
        }
    }

    "CPU".to_string()
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
