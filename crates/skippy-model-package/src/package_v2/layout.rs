//! Artifact layout planning for repacked v2 packages.
//!
//! Decides which payload-bearing artifact each source tensor binds to, using
//! the native role classifier. The metadata carrier is planning metadata, not
//! storage: no tensor binds to it. Every logical tensor binds exactly once to
//! an artifact with a real stored extent.
use std::collections::BTreeMap;

use anyhow::Result;
use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;

/// A payload artifact the writer must emit, with the source tensor names that
/// bind to it. `layer` is set only for per-layer artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedArtifact {
    pub(crate) id: String,
    /// Relative package path; the verifier derives layer ordinals from
    /// `layers/layer-N.gguf` naming.
    pub(crate) path: String,
    pub(crate) kind: PlannedArtifactKind,
    /// Canonical tensor names bound to this artifact. Repeated copies emitted
    /// by the native slice writer stay unreferenced.
    pub(crate) tensor_names: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlannedArtifactKind {
    /// Metadata/Tokenizer/Unknown-role payload tensors. Emitted only when the
    /// set is nonempty; a plain slice carries them with real payload.
    Common,
    Embeddings,
    /// FinalNorm + Output role tensors.
    Output,
    Layer {
        ordinal: u32,
    },
}

/// Assign every native tensor to exactly one payload artifact.
///
/// `tensors` must be the complete native inventory of the source. Fails if any
/// tensor cannot be bound, so the writer never produces a package with an
/// unowned tensor.
pub(crate) fn plan_artifacts(tensors: &[TensorInfo]) -> Result<Vec<PlannedArtifact>> {
    let mut common: Vec<String> = Vec::new();
    let mut embeddings: Vec<String> = Vec::new();
    let mut output: Vec<String> = Vec::new();
    let mut layers: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for tensor in tensors {
        match tensor.role {
            TensorRole::Layer => {
                let ordinal = tensor.layer_index.ok_or_else(|| {
                    anyhow::anyhow!("layer tensor {:?} has no layer index", tensor.name)
                })?;
                layers.entry(ordinal).or_default().push(tensor.name.clone());
            }
            TensorRole::Embedding => embeddings.push(tensor.name.clone()),
            TensorRole::FinalNorm | TensorRole::Output => output.push(tensor.name.clone()),
            TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown => {
                common.push(tensor.name.clone())
            }
        }
    }
    common.sort();
    embeddings.sort();
    output.sort();

    let mut planned = Vec::new();
    if !common.is_empty() {
        planned.push(PlannedArtifact {
            id: "common".to_string(),
            path: "shared/common.gguf".to_string(),
            kind: PlannedArtifactKind::Common,
            tensor_names: common,
        });
    }
    if !embeddings.is_empty() {
        planned.push(PlannedArtifact {
            id: "embeddings".to_string(),
            path: "shared/embeddings.gguf".to_string(),
            kind: PlannedArtifactKind::Embeddings,
            tensor_names: embeddings,
        });
    }
    if !output.is_empty() {
        planned.push(PlannedArtifact {
            id: "output".to_string(),
            path: "shared/output.gguf".to_string(),
            kind: PlannedArtifactKind::Output,
            tensor_names: output,
        });
    }
    for (ordinal, mut names) in layers {
        names.sort();
        planned.push(PlannedArtifact {
            id: format!("layer-{ordinal:05}"),
            path: format!("layers/layer-{ordinal:05}.gguf"),
            kind: PlannedArtifactKind::Layer { ordinal },
            tensor_names: names,
        });
    }
    Ok(planned)
}

#[cfg(test)]
mod tests;
