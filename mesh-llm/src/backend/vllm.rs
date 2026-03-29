use super::{BackendControlFuture, BackendLaunchFuture, BackendOps};
use crate::launch::{
    reqwest_health_check, InferenceServerHandle, InferenceServerProcess, ModelLaunchSpec,
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::process::Command;

pub(super) struct VllmBackend;

impl BackendOps for VllmBackend {
    fn as_str(&self) -> &'static str {
        "vllm"
    }

    fn process_label(&self) -> &'static str {
        "vllm"
    }

    fn start_server<'a>(
        &self,
        _bin_dir: &'a Path,
        _binary_flavor: Option<crate::launch::BinaryFlavor>,
        spec: ModelLaunchSpec<'a>,
    ) -> BackendLaunchFuture<'a> {
        Box::pin(start_vllm_server(spec))
    }

    fn kill_server_processes<'a>(&'a self) -> BackendControlFuture<'a> {
        Box::pin(kill_vllm_server())
    }
}

pub(super) static VLLM_BACKEND: VllmBackend = VllmBackend;

async fn kill_vllm_server() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "vllm serve"])
        .status();
    let _ = std::process::Command::new("pkill")
        .args(["-f", "vllm.entrypoints.openai.api_server"])
        .status();
}

fn search_path(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn resolve_vllm_executable() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MESH_LLM_VLLM_BIN") {
        let path = PathBuf::from(path);
        anyhow::ensure!(
            path.exists(),
            "MESH_LLM_VLLM_BIN points to missing path {}",
            path.display()
        );
        return Ok(path);
    }

    if let Some(home) = dirs::home_dir() {
        let default = home.join(".venv-vllm-metal").join("bin").join("vllm");
        if default.exists() {
            return Ok(default);
        }
    }

    if let Some(path) = search_path("vllm") {
        return Ok(path);
    }

    anyhow::bail!("vllm executable not found. Install vllm/vllm-metal or set MESH_LLM_VLLM_BIN");
}

async fn start_vllm_server(spec: ModelLaunchSpec<'_>) -> Result<InferenceServerProcess> {
    anyhow::ensure!(
        spec.tunnel_ports.is_empty(),
        "vllm backend does not support rpc split workers yet"
    );
    anyhow::ensure!(
        spec.tensor_split.is_none(),
        "vllm backend does not support tensor split yet"
    );
    anyhow::ensure!(
        spec.draft.is_none(),
        "vllm backend does not support speculative draft models yet"
    );
    anyhow::ensure!(
        spec.mmproj.is_none(),
        "vllm backend does not support llama.cpp mmproj launch args"
    );
    anyhow::ensure!(
        spec.model.exists(),
        "Model not found at {}",
        spec.model.display()
    );

    let vllm = resolve_vllm_executable()?;
    let log_path = std::env::temp_dir().join("mesh-llm-vllm.log");
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("Failed to create vllm log file {}", log_path.display()))?;
    let log_file2 = log_file.try_clone()?;

    let mut args = vec![
        "serve".to_string(),
        spec.model.to_string_lossy().to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        spec.http_port.to_string(),
        "--served-model-name".to_string(),
        spec.served_model_name.to_string(),
    ];
    if let Some(ctx_size) = spec.ctx_size_override {
        args.push("--max-model-len".to_string());
        args.push(ctx_size.to_string());
    }

    tracing::info!(
        "Starting vllm on :{} with model {}",
        spec.http_port,
        spec.model.display()
    );

    let mut child = Command::new(&vllm)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start vllm at {}", vllm.display()))?;

    let url = format!("http://localhost:{}/health", spec.http_port);
    for _ in 0..600 {
        if reqwest_health_check(&url).await {
            let pid = child
                .id()
                .context("vllm started but did not expose a PID")?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle::new(pid, expected_exit.clone());
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                if !expected_exit.load(Ordering::Relaxed) {
                    eprintln!("⚠️  vllm process exited unexpectedly");
                }
                let _ = death_tx.send(());
            });
            return Ok(InferenceServerProcess { handle, death_rx });
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!(
        "vllm failed to become healthy within 600s. See {}",
        log_path.display()
    );
}
