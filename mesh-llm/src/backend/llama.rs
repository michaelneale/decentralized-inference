use super::{BackendControlFuture, BackendLaunchFuture, BackendOps};
use crate::launch::{
    reqwest_health_check, resolve_binary_path, resolve_device_for_binary, temp_log_path,
    BinaryFlavor, InferenceServerHandle, InferenceServerProcess, ModelLaunchSpec,
};
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::process::Command;

pub(super) struct LlamaBackend;

impl BackendOps for LlamaBackend {
    fn as_str(&self) -> &'static str {
        "llama"
    }

    fn process_label(&self) -> &'static str {
        "llama-server"
    }

    fn start_server<'a>(
        &self,
        bin_dir: &'a Path,
        binary_flavor: Option<BinaryFlavor>,
        spec: ModelLaunchSpec<'a>,
    ) -> BackendLaunchFuture<'a> {
        Box::pin(start_llama_server(bin_dir, binary_flavor, spec))
    }

    fn kill_server_processes<'a>(&'a self) -> BackendControlFuture<'a> {
        Box::pin(kill_llama_server())
    }
}

pub(super) static LLAMA_BACKEND: LlamaBackend = LlamaBackend;

/// Kill all running llama-server processes.
async fn kill_llama_server() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "llama-server"])
        .status();
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let output = std::process::Command::new("pgrep")
            .args(["-f", "llama-server"])
            .output();
        if let Ok(o) = output {
            if o.stdout.is_empty() {
                return;
            }
        } else {
            return;
        }
    }
    let _ = std::process::Command::new("pkill")
        .args(["-9", "-f", "llama-server"])
        .status();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
/// `model_bytes` is the total GGUF file size, used to select KV cache quantization:
///   - < 5GB: FP16 (default) — small models, KV cache is tiny
///   - 5-50GB: Q8_0 — no measurable quality loss, saves ~50% KV memory
///   - > 50GB: Q4_0 — slight long-context degradation, but these models need every byte
async fn start_llama_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    spec: ModelLaunchSpec<'_>,
) -> Result<InferenceServerProcess> {
    let model = spec.model;
    let http_port = spec.http_port;
    let tunnel_ports = spec.tunnel_ports;
    let tensor_split = spec.tensor_split;
    let draft = spec.draft;
    let draft_max = spec.draft_max;
    let model_bytes = spec.model_bytes;
    let my_vram = spec.my_vram;
    let mmproj = spec.mmproj;
    let ctx_size_override = spec.ctx_size_override;
    let total_group_vram = spec.total_group_vram;
    let llama_server = resolve_binary_path(bin_dir, "llama-server", binary_flavor)?;

    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());

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

    let llama_log = temp_log_path("mesh-llm-llama-server.log");
    let log_file = std::fs::File::create(&llama_log).with_context(|| {
        format!(
            "Failed to create llama-server log file {}",
            llama_log.display()
        )
    })?;
    let log_file2 = log_file.try_clone()?;

    const GB: u64 = 1_000_000_000;
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size: u32 = if let Some(override_ctx) = ctx_size_override {
        override_ctx
    } else if vram_after_model >= 30 * GB {
        65536
    } else if vram_after_model >= 12 * GB {
        32768
    } else if vram_after_model >= 6 * GB {
        16384
    } else if vram_after_model >= 3 * GB {
        8192
    } else {
        4096
    };
    tracing::info!(
        "Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB VRAM, {:.1}GB free{})",
        model_bytes as f64 / GB as f64,
        host_model_bytes as f64 / GB as f64,
        my_vram as f64 / GB as f64,
        vram_after_model as f64 / GB as f64,
        if total_group_vram.is_some() {
            " [split]"
        } else {
            ""
        }
    );

    let mut args = vec!["-m".to_string(), model.to_string_lossy().to_string()];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    args.extend_from_slice(&[
        "-ngl".to_string(),
        "99".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-fit".to_string(),
        "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        http_port.to_string(),
        "-c".to_string(),
        ctx_size.to_string(),
        "--reasoning-format".to_string(),
        "deepseek".to_string(),
        "--reasoning-budget".to_string(),
        "0".to_string(),
    ]);
    if model_bytes >= 50 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            "q4_0".to_string(),
            "--cache-type-v".to_string(),
            "q4_0".to_string(),
        ]);
        tracing::info!("KV cache: Q4_0 (model > 50GB)");
    } else if model_bytes >= 5 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            "q8_0".to_string(),
            "--cache-type-v".to_string(),
            "q8_0".to_string(),
        ]);
        tracing::info!("KV cache: Q8_0 (model 5-50GB)");
    }
    if let Some(ts) = tensor_split {
        args.push("--tensor-split".to_string());
        args.push(ts.to_string());
    }
    let local_device = resolve_device_for_binary(&llama_server.path, llama_server.flavor, None)?;
    if let Some(draft_path) = draft {
        if draft_path.exists() {
            if local_device != "CPU" {
                args.push("-md".to_string());
                args.push(draft_path.to_string_lossy().to_string());
                args.push("-ngld".to_string());
                args.push("99".to_string());
                args.push("--device-draft".to_string());
                args.push(local_device.clone());
                args.push("--draft-max".to_string());
                args.push(draft_max.to_string());
                tracing::info!(
                    "Speculative decoding: draft={}, draft-max={}, device={}",
                    draft_path.display(),
                    draft_max,
                    local_device
                );
            } else {
                tracing::warn!(
                    "Draft model present at {} but no GPU backend detected, skipping speculative decoding",
                    draft_path.display()
                );
            }
        } else {
            tracing::warn!(
                "Draft model not found at {}, skipping speculative decoding",
                draft_path.display()
            );
        }
    }
    if let Some(proj) = mmproj {
        if proj.exists() {
            args.push("--mmproj".to_string());
            args.push(proj.to_string_lossy().to_string());
            args.push("--ubatch-size".to_string());
            args.push("2048".to_string());
            tracing::info!("Vision: mmproj={}", proj.display());
        } else {
            tracing::warn!("mmproj not found at {}, skipping vision", proj.display());
        }
    }
    let mut child = Command::new(&llama_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start llama-server at {}",
                llama_server.path.display()
            )
        })?;

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
            tracing::info!(
                "Still waiting for llama-server to load model... ({i}s, {transferred} transferred)"
            );
        }
        if reqwest_health_check(&url).await {
            let pid = child
                .id()
                .context("llama-server started but did not expose a PID")?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle::new(pid, expected_exit.clone());
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                if !expected_exit.load(Ordering::Relaxed) {
                    eprintln!("⚠️  llama-server process exited unexpectedly");
                }
                let _ = death_tx.send(());
            });
            return Ok(InferenceServerProcess { handle, death_rx });
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("llama-server failed to become healthy within 600s");
}
