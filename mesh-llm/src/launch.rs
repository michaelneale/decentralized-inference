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
pub async fn start_llama_server(
    bin_dir: &Path,
    model: &Path,
    http_port: u16,
    tunnel_ports: &[u16],
    tensor_split: Option<&str>,
    draft: Option<&Path>,
    draft_max: u16,
    model_bytes: u64,
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

    let log_file = std::fs::File::create("/tmp/mesh-llm-llama-server.log")
        .context("Failed to create llama-server log file")?;
    let log_file2 = log_file.try_clone()?;

    // llama-server uses --rpc only for remote workers.
    // The host's own GPU is used directly via Metal (no local rpc-server in the list).
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
    // KV cache quantization based on model size:
    //   < 5GB: leave default (FP16) — small models, KV cache is negligible
    //   5-50GB: Q8_0 — essentially lossless, halves KV memory
    //   > 50GB: Q4_0 — slight long-context quality trade, but critical memory savings
    const GB: u64 = 1_000_000_000;
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

/// Start llama-server in router mode with --models-dir for multi-model on-demand serving.
/// Used by offline mode. No --rpc, no tensor split — solo GPU only.
pub async fn start_llama_server_router(
    bin_dir: &Path,
    models_dir: &Path,
    http_port: u16,
    models_max: u16,
    extra_model_paths: &[(String, std::path::PathBuf)],
) -> Result<()> {
    let llama_server = bin_dir.join("llama-server");
    anyhow::ensure!(
        llama_server.exists(),
        "llama-server not found at {}. Build llama.cpp first.",
        llama_server.display()
    );

    tracing::info!(
        "Starting llama-server (router mode) on :{http_port} with models-dir {}",
        models_dir.display()
    );

    // Create symlinks for ollama models in a temp dir alongside models_dir models.
    // We use a staging dir that contains symlinks to everything: models_dir GGUFs + ollama blobs.
    let staging_dir = std::env::temp_dir().join("mesh-llm-offline-models");
    let _ = std::fs::remove_dir_all(&staging_dir);
    std::fs::create_dir_all(&staging_dir)
        .context("Failed to create offline model staging dir")?;

    // Symlink all .gguf files from models_dir (skip partials and drafts)
    if models_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    if size > 500_000_000 {
                        let link = staging_dir.join(entry.file_name());
                        let _ = std::os::unix::fs::symlink(&path, &link);
                    }
                }
            }
        }
    }

    // Symlink ollama models with friendly .gguf names
    for (name, blob_path) in extra_model_paths {
        // Turn "ollama/glm-4.7-flash" into "ollama-glm-4.7-flash.gguf"
        let safe_name = name.replace('/', "-").replace(':', "-");
        let link = staging_dir.join(format!("{safe_name}.gguf"));
        let _ = std::os::unix::fs::symlink(blob_path, &link);
    }

    let log_file = std::fs::File::create("/tmp/mesh-llm-llama-server.log")
        .context("Failed to create llama-server log file")?;
    let log_file2 = log_file.try_clone()?;

    let args = vec![
        "--models-dir".to_string(), staging_dir.to_string_lossy().to_string(),
        "--models-max".to_string(), models_max.to_string(),
        "--models-autoload".to_string(),
        "-ngl".to_string(), "99".to_string(),
        "--host".to_string(), "127.0.0.1".to_string(),
        "--port".to_string(), http_port.to_string(),
    ];

    tracing::info!("llama-server args: {:?}", args);

    let mut child = Command::new(&llama_server)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start llama-server at {}", llama_server.display()))?;

    // Wait for health check — router mode starts fast since models load on-demand
    let url = format!("http://localhost:{http_port}/health");
    for i in 0..60 {
        if i > 0 && i % 5 == 0 {
            tracing::info!("Waiting for llama-server router to start... ({i}s)");
        }
        if reqwest_health_check(&url).await {
            // Detach
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("llama-server (router) failed to start within 60s");
}

/// An image model bundle: diffusion model + companion files (VAE, CLIP, T5).
/// Stored as a directory in ~/.models/ (e.g. ~/.models/flux1-dev/).
#[derive(Clone, Debug)]
pub struct ImageModelBundle {
    /// Human-readable name (directory name, e.g. "flux1-dev")
    pub name: String,
    /// Path to the diffusion model GGUF
    pub diffusion_model: std::path::PathBuf,
    /// Path to VAE (ae.safetensors or similar)
    pub vae: Option<std::path::PathBuf>,
    /// Path to CLIP-L text encoder
    pub clip_l: Option<std::path::PathBuf>,
    /// Path to T5XXL text encoder
    pub t5xxl: Option<std::path::PathBuf>,
    /// Total size of all files in the bundle
    pub total_bytes: u64,
    /// Size of just the diffusion model (approximates GPU VRAM needed)
    pub diffusion_bytes: u64,
}

/// Scan ~/.models/ for image model bundles.
/// A bundle is a directory containing a *.gguf diffusion model + companion files.
pub fn scan_image_model_bundles() -> Vec<ImageModelBundle> {
    let models_dir = crate::download::models_dir();
    let mut bundles = Vec::new();

    if !models_dir.exists() {
        return bundles;
    }

    let Ok(entries) = std::fs::read_dir(&models_dir) else {
        return bundles;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        // Look for a diffusion model GGUF inside the directory
        let Ok(dir_entries) = std::fs::read_dir(&path) else { continue };

        let mut diffusion_model: Option<std::path::PathBuf> = None;
        let mut vae: Option<std::path::PathBuf> = None;
        let mut clip_l: Option<std::path::PathBuf> = None;
        let mut t5xxl: Option<std::path::PathBuf> = None;
        let mut total_bytes: u64 = 0;

        for de in dir_entries.flatten() {
            let fp = de.path();
            let fname = fp.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            let size = std::fs::metadata(&fp).map(|m| m.len()).unwrap_or(0);
            total_bytes += size;

            if fname.ends_with(".gguf") && size > 500_000_000 {
                // Largest GGUF is the diffusion model
                if diffusion_model.as_ref().map_or(true, |prev| {
                    std::fs::metadata(prev).map(|m| m.len()).unwrap_or(0) < size
                }) {
                    diffusion_model = Some(fp.clone());
                }
            } else if fname.contains("ae") && fname.ends_with(".safetensors")
                || fname.contains("vae") && fname.ends_with(".safetensors")
            {
                vae = Some(fp.clone());
            } else if fname.contains("clip_l") && fname.ends_with(".safetensors") {
                clip_l = Some(fp.clone());
            } else if fname.contains("t5xxl") && fname.ends_with(".safetensors") {
                t5xxl = Some(fp.clone());
            }
        }

        // Must have at least a diffusion model to be a valid bundle
        if let Some(ref dm) = diffusion_model {
            let diffusion_bytes = std::fs::metadata(dm).map(|m| m.len()).unwrap_or(0);
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            bundles.push(ImageModelBundle {
                name,
                diffusion_model: dm.clone(),
                vae,
                clip_l,
                t5xxl,
                total_bytes,
                diffusion_bytes,
            });
        }
    }

    bundles.sort_by(|a, b| a.name.cmp(&b.name));
    bundles
}

/// Start sd-server for an image model bundle.
/// Returns the port it's listening on.
pub async fn start_sd_server(
    bin_dir: &Path,
    bundle: &ImageModelBundle,
    http_port: u16,
) -> Result<u16> {
    // Look for sd-server in bin_dir, then fall back to sd_build_dir
    let sd_server = find_sd_server(bin_dir)?;

    tracing::info!(
        "Starting sd-server on :{http_port} with model {}",
        bundle.name
    );

    let log_file = std::fs::File::create(format!("/tmp/mesh-llm-sd-server-{http_port}.log"))
        .context("Failed to create sd-server log file")?;
    let log_file2 = log_file.try_clone()?;

    let mut args = vec![
        "--diffusion-model".to_string(),
        bundle.diffusion_model.to_string_lossy().to_string(),
        "--listen-port".to_string(),
        http_port.to_string(),
        "-l".to_string(),
        "127.0.0.1".to_string(),
    ];

    if let Some(ref vae) = bundle.vae {
        args.push("--vae".to_string());
        args.push(vae.to_string_lossy().to_string());
    }
    if let Some(ref clip_l) = bundle.clip_l {
        args.push("--clip_l".to_string());
        args.push(clip_l.to_string_lossy().to_string());
    }
    if let Some(ref t5xxl) = bundle.t5xxl {
        args.push("--t5xxl".to_string());
        args.push(t5xxl.to_string_lossy().to_string());
    }

    // Keep text encoders on CPU to save VRAM for the diffusion model
    args.push("--clip-on-cpu".to_string());
    // Flash attention for speed
    args.push("--diffusion-fa".to_string());

    tracing::info!("sd-server args: {:?}", args);

    let mut child = Command::new(&sd_server)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start sd-server at {}", sd_server.display()))?;

    // sd-server health check: GET / returns "Stable Diffusion Server is running" with 200
    let url = format!("http://localhost:{http_port}/");
    for i in 0..120 {
        if i > 0 && i % 10 == 0 {
            tracing::info!("Waiting for sd-server to load model... ({i}s)");
        }
        if reqwest_health_check(&url).await {
            // Detach
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            tracing::info!("sd-server ready on :{http_port}");
            return Ok(http_port);
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("sd-server failed to start within 120s");
}

/// Kill all running sd-server processes.
pub async fn kill_sd_server() {
    let _ = std::process::Command::new("pkill").args(["-f", "sd-server"]).status();
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let output = std::process::Command::new("pgrep").args(["-f", "sd-server"]).output();
        if let Ok(o) = output {
            if o.stdout.is_empty() { return; }
        } else {
            return;
        }
    }
    let _ = std::process::Command::new("pkill").args(["-9", "-f", "sd-server"]).status();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Find sd-server binary. Checks bin_dir first, then ../stable-diffusion.cpp/build/bin/
pub fn find_sd_server(bin_dir: &Path) -> Result<std::path::PathBuf> {
    // Check alongside llama-server
    let candidate = bin_dir.join("sd-server");
    if candidate.exists() {
        return Ok(candidate);
    }
    // Check stable-diffusion.cpp build dir relative to bin_dir
    // bin_dir is typically llama.cpp/build/bin, so go up to project root
    if let Some(project_root) = bin_dir.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
        let candidate = project_root.join("stable-diffusion.cpp/build/bin/sd-server");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "sd-server not found in {} or nearby. Build stable-diffusion.cpp with `just build-sd`.",
        bin_dir.display()
    );
}

/// Find an available TCP port
pub async fn find_free_port() -> Result<u16> {
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
