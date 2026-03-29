//! Process management for backend server binaries.
//!
//! Starts rpc-server and backend inference processes wired up to the mesh
//! tunnel ports. Concrete backends plug into this module via BackendOps.

use crate::backend;
use anyhow::{Context, Result};
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::process::Command;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }

    fn preferred_devices(self) -> &'static [&'static str] {
        match self {
            BinaryFlavor::Cpu => &["CPU"],
            BinaryFlavor::Cuda => &["CUDA0", "CPU"],
            BinaryFlavor::Rocm => &["HIP0", "CPU"],
            BinaryFlavor::Vulkan => &["Vulkan0", "CPU"],
            BinaryFlavor::Metal => &["MTL0", "CPU"],
        }
    }

    fn primary_device(self) -> &'static str {
        self.preferred_devices()[0]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedBinary {
    pub(crate) path: PathBuf,
    pub(crate) flavor: Option<BinaryFlavor>,
}

fn flavored_bin_name(name: &str, flavor: BinaryFlavor) -> String {
    format!("{name}-{}", flavor.suffix())
}

fn infer_binary_flavor(name: &str, path: &Path) -> Option<BinaryFlavor> {
    let file_name = path.file_name()?.to_string_lossy();
    for flavor in BinaryFlavor::ALL {
        if file_name == flavored_bin_name(name, flavor) {
            return Some(flavor);
        }
    }
    None
}

pub(crate) fn resolve_binary_path(
    bin_dir: &Path,
    name: &str,
    requested_flavor: Option<BinaryFlavor>,
) -> Result<ResolvedBinary> {
    if let Some(flavor) = requested_flavor {
        let flavored = bin_dir.join(flavored_bin_name(name, flavor));
        if flavored.exists() {
            return Ok(ResolvedBinary {
                path: flavored,
                flavor: Some(flavor),
            });
        }

        let generic = bin_dir.join(name);
        if generic.exists() {
            return Ok(ResolvedBinary {
                path: generic,
                flavor: Some(flavor),
            });
        }

        anyhow::bail!(
            "{} not found in {} for requested flavor '{}'",
            flavored.display(),
            bin_dir.display(),
            flavor.suffix()
        );
    }

    let generic = bin_dir.join(name);
    if generic.exists() {
        let flavor = infer_binary_flavor(name, &generic);
        return Ok(ResolvedBinary {
            path: generic,
            flavor,
        });
    }

    let matches: Vec<ResolvedBinary> = BinaryFlavor::ALL
        .into_iter()
        .map(|flavor| ResolvedBinary {
            path: bin_dir.join(flavored_bin_name(name, flavor)),
            flavor: Some(flavor),
        })
        .filter(|candidate| candidate.path.exists())
        .collect();

    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => anyhow::bail!(
            "{} not found in {}",
            bin_dir.join(name).display(),
            bin_dir.display()
        ),
        _ => {
            let options = matches
                .iter()
                .filter_map(|candidate| candidate.flavor.map(|flavor| flavor.suffix()))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "multiple {} flavors found in {} ({options}). Pass --llama-flavor to choose one.",
                name,
                bin_dir.display()
            );
        }
    }
}

pub(crate) fn temp_log_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[derive(Clone, Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
}

impl InferenceServerHandle {
    pub(crate) fn new(pid: u32, expected_exit: Arc<AtomicBool>) -> Self {
        Self { pid, expected_exit }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process(self.pid).await;
    }
}

#[derive(Debug)]
pub struct InferenceServerProcess {
    pub handle: InferenceServerHandle,
    pub death_rx: tokio::sync::oneshot::Receiver<()>,
}

pub struct ModelLaunchSpec<'a> {
    pub backend: backend::BackendKind,
    pub model: &'a Path,
    pub served_model_name: &'a str,
    pub http_port: u16,
    pub tunnel_ports: &'a [u16],
    pub tensor_split: Option<&'a str>,
    pub draft: Option<&'a Path>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub my_vram: u64,
    pub mmproj: Option<&'a Path>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram: Option<u64>,
}

fn log_tail(path: &Path, max_lines: usize) -> String {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return String::new();
    };

    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn parse_available_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_devices = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed == "available devices:" {
            in_devices = true;
            continue;
        }
        if !in_devices || trimmed.is_empty() {
            continue;
        }
        let Some((name, _rest)) = trimmed.split_once(':') else {
            continue;
        };
        if !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }
        devices.push(name.to_string());
    }

    devices
}

fn probe_available_devices(binary: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new(binary)
        .args(["-d", "__mesh_llm_probe_invalid__", "-p", "0"])
        .output()
    else {
        return Vec::new();
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !combined.is_empty() && !output.stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_available_devices(&combined)
}

fn preferred_device(available: &[String], flavor: Option<BinaryFlavor>) -> Option<String> {
    let candidates: &[&str] = if let Some(flavor) = flavor {
        flavor.preferred_devices()
    } else {
        &["MTL0", "CUDA0", "HIP0", "Vulkan0", "CPU"]
    };

    for candidate in candidates {
        if available.iter().any(|device| device == candidate) {
            return Some((*candidate).to_string());
        }
    }
    available.first().cloned()
}

pub(crate) fn resolve_device_for_binary(
    binary: &Path,
    flavor: Option<BinaryFlavor>,
    requested: Option<&str>,
) -> Result<String> {
    let available = probe_available_devices(binary);

    if let Some(device) = requested {
        if !available.is_empty() && !available.iter().any(|candidate| candidate == device) {
            anyhow::bail!(
                "requested device {device} is not supported by {}. Available devices: {}",
                binary.display(),
                available.join(", ")
            );
        }
        return Ok(device.to_string());
    }

    if let Some(selected) = preferred_device(&available, flavor) {
        return Ok(selected);
    }

    if let Some(flavor) = flavor {
        return Ok(flavor.primary_device().to_string());
    }

    Ok(detect_device())
}

fn command_has_output(command: &str, args: &[&str]) -> bool {
    let Ok(output) = std::process::Command::new(command).args(args).output() else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| !line.trim().is_empty())
}

/// Start a local rpc-server and return the port it's listening on.
/// Picks an available port automatically.
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    device: Option<&str>,
    gguf_path: Option<&Path>,
) -> Result<u16> {
    let rpc_server = resolve_binary_path(bin_dir, "rpc-server", binary_flavor)?;

    // Find a free port
    let port = find_free_port().await?;

    let device = resolve_device_for_binary(&rpc_server.path, rpc_server.flavor, device)?;
    let startup_timeout = if device.starts_with("Vulkan") {
        std::time::Duration::from_secs(90)
    } else {
        std::time::Duration::from_secs(15)
    };
    let startup_polls = (startup_timeout.as_millis() / 500) as usize;

    tracing::info!("Starting rpc-server on :{port} (device: {device})");

    let rpc_log = temp_log_path(&format!("mesh-llm-rpc-{port}.log"));
    let rpc_log_file = std::fs::File::create(&rpc_log)
        .with_context(|| format!("Failed to create rpc-server log file {}", rpc_log.display()))?;
    let rpc_log_file2 = rpc_log_file.try_clone()?;

    let mut args = vec![
        "-d".to_string(),
        device.clone(),
        "-p".to_string(),
        port.to_string(),
    ];
    if let Some(path) = gguf_path {
        args.push("--gguf".to_string());
        args.push(path.to_string_lossy().to_string());
        tracing::info!(
            "rpc-server will load weights from local GGUF: {}",
            path.display()
        );
    }

    let mut child = Command::new(&rpc_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start rpc-server at {}",
                rpc_server.path.display()
            )
        })?;

    // Wait for it to be listening
    for _ in 0..startup_polls {
        if is_port_open(port).await {
            // Detach — let it run in the background
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(port);
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll rpc-server status for {}",
                rpc_server.path.display()
            )
        })? {
            let tail = log_tail(&rpc_log, 40);
            let tail_msg = if tail.is_empty() {
                format!("See {}", rpc_log.display())
            } else {
                format!("See {}:\n{}", rpc_log.display(), tail)
            };
            anyhow::bail!(
                "rpc-server exited before listening on port {port} (device: {device}, status: {status}). {tail_msg}"
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let tail = log_tail(&rpc_log, 40);
    let tail_msg = if tail.is_empty() {
        format!("See {}", rpc_log.display())
    } else {
        format!("See {}:\n{}", rpc_log.display(), tail)
    };
    anyhow::bail!(
        "rpc-server failed to start on port {port} within {}s (device: {device}). {tail_msg}",
        startup_timeout.as_secs()
    );
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
            if parts.len() >= 3 && parts[2].contains("rpc-server") && parts[1] == "1" {
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

async fn terminate_process(pid: u32) {
    let pid_str = pid.to_string();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid_str])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let alive = std::process::Command::new("kill")
            .args(["-0", &pid_str])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !alive {
            return;
        }
    }
    let _ = std::process::Command::new("kill")
        .args(["-9", &pid_str])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
}

/// Start a backend server for a model. This dispatches to the concrete backend
/// implementation so other runtimes can plug into the same control plane.
pub async fn start_model_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    spec: ModelLaunchSpec<'_>,
) -> Result<InferenceServerProcess> {
    let ops = backend::backend_ops(spec.backend);
    tracing::debug!(
        "dispatching backend '{}' via {} (health: {})",
        ops.as_str(),
        ops.process_label(),
        ops.health_path()
    );
    ops.start_server(bin_dir, binary_flavor, spec).await
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
    if command_has_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        return "CUDA0".to_string();
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
    if command_has_output("rocm-smi", &["--showproductname"]) || command_has_output("rocminfo", &[])
    {
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
pub(crate) async fn reqwest_health_check(url: &str) -> bool {
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
#[cfg(test)]
mod tests {
    use super::{parse_available_devices, preferred_device, BinaryFlavor};
    use std::path::Path;

    #[test]
    fn parse_available_devices_ignores_non_device_lines() {
        let output = r#"
error: unknown device: HIP0
available devices:
No devices found
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 13737 MiB free)
  CPU: AMD Ryzen 7 7800X3D 8-Core Processor (192857 MiB, 192857 MiB free)
"#;

        assert_eq!(
            parse_available_devices(output),
            vec!["Vulkan0".to_string(), "CPU".to_string()]
        );
    }

    #[test]
    fn preferred_device_picks_vulkan_when_that_is_all_binary_supports() {
        let available = vec!["Vulkan0".to_string(), "CPU".to_string()];
        assert_eq!(
            preferred_device(&available, Some(BinaryFlavor::Vulkan)),
            Some("Vulkan0".to_string())
        );
    }

    #[test]
    fn infer_binary_flavor_from_filename() {
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan")),
            Some(BinaryFlavor::Vulkan)
        );
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server")),
            None
        );
    }
}
