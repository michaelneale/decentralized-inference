use crate::launch::{BinaryFlavor, InferenceServerProcess, ModelLaunchSpec};
use anyhow::Result;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Llama,
}

pub type BackendLaunchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<InferenceServerProcess>> + Send + 'a>>;

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
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        backend_ops(self).as_str()
    }
}

struct LlamaBackend;

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
        Box::pin(crate::launch::start_llama_server(
            bin_dir,
            binary_flavor,
            spec,
        ))
    }
}

static LLAMA_BACKEND: LlamaBackend = LlamaBackend;

pub fn backend_ops(kind: BackendKind) -> &'static dyn BackendOps {
    match kind {
        BackendKind::Llama => &LLAMA_BACKEND,
    }
}

pub fn detect_backend(_model_path: &Path) -> BackendKind {
    // Stage 1 keeps llama.cpp as the only concrete backend implementation,
    // but runtime launch/control now flows through BackendOps so MLX or
    // other backends can plug in without reshaping the control plane.
    BackendKind::Llama
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_registry_returns_llama_ops() {
        let ops = backend_ops(BackendKind::Llama);
        assert_eq!(ops.as_str(), "llama");
        assert_eq!(ops.process_label(), "llama-server");
        assert_eq!(ops.health_path(), "/health");
    }
}
