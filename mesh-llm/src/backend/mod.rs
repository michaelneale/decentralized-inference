mod llama;
mod vllm;

use crate::launch::{BinaryFlavor, InferenceServerProcess, ModelLaunchSpec};
use anyhow::Result;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Llama,
    Vllm,
}

impl BackendKind {
    pub const ALL: [BackendKind; 2] = [BackendKind::Llama, BackendKind::Vllm];

    pub fn as_str(self) -> &'static str {
        backend_ops(self).as_str()
    }
}

pub type BackendLaunchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<InferenceServerProcess>> + Send + 'a>>;
pub type BackendControlFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

pub trait BackendOps: Send + Sync {
    fn as_str(&self) -> &'static str;
    fn process_label(&self) -> &'static str;
    fn health_path(&self) -> &'static str {
        "/health"
    }
    fn start_server<'a>(
        &self,
        bin_dir: &'a Path,
        binary_flavor: Option<BinaryFlavor>,
        spec: ModelLaunchSpec<'a>,
    ) -> BackendLaunchFuture<'a>;
    fn kill_server_processes<'a>(&'a self) -> BackendControlFuture<'a>;
}

pub fn backend_ops(kind: BackendKind) -> &'static dyn BackendOps {
    match kind {
        BackendKind::Llama => &llama::LLAMA_BACKEND,
        BackendKind::Vllm => &vllm::VLLM_BACKEND,
    }
}

pub fn requires_rpc_server(kind: BackendKind) -> bool {
    matches!(kind, BackendKind::Llama)
}

pub fn supports_rpc_split(kind: BackendKind) -> bool {
    matches!(kind, BackendKind::Llama)
}

pub async fn kill_all_server_processes() {
    for kind in BackendKind::ALL {
        backend_ops(kind).kill_server_processes().await;
    }
}

fn looks_like_hf_model_dir(model_path: &Path) -> bool {
    model_path.is_dir() && model_path.join("config.json").exists()
}

pub fn detect_backend(model_path: &Path) -> BackendKind {
    if looks_like_hf_model_dir(model_path) {
        return BackendKind::Vllm;
    }

    BackendKind::Llama
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn backend_registry_returns_llama_ops() {
        let ops = backend_ops(BackendKind::Llama);
        assert_eq!(ops.as_str(), "llama");
        assert_eq!(ops.process_label(), "llama-server");
        assert_eq!(ops.health_path(), "/health");
    }

    #[test]
    fn backend_registry_returns_vllm_ops() {
        let ops = backend_ops(BackendKind::Vllm);
        assert_eq!(ops.as_str(), "vllm");
        assert_eq!(ops.process_label(), "vllm");
        assert_eq!(ops.health_path(), "/health");
    }

    #[test]
    fn detect_backend_uses_vllm_for_hf_model_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mesh-llm-vllm-backend-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();

        assert_eq!(detect_backend(&dir), BackendKind::Vllm);

        let _ = std::fs::remove_dir_all(dir);
    }
}
