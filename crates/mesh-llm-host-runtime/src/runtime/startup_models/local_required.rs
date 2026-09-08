use std::path::Path;

use anyhow::{Context, Result};

use super::{StartupModelPlan, StartupModelSpec};
use crate::models;

pub(in crate::runtime) fn validate_local_required_source(
    path: &Path,
    model_id: &str,
) -> Result<()> {
    anyhow::ensure!(
        path.is_absolute(),
        "skippy.source_policy = \"local-required\" for {model_id} requires an absolute local GGUF path: {}",
        path.display()
    );
    let metadata = std::fs::symlink_metadata(path).with_context(|| {
        format!(
            "skippy.source_policy = \"local-required\" for {model_id} requires a local GGUF file: {}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "skippy.source_policy = \"local-required\" for {model_id} requires a non-symlink local GGUF file: {}",
        path.display()
    );
    Ok(())
}

pub(super) fn resolve_local_required_startup_model(
    spec: &StartupModelSpec,
) -> Result<StartupModelPlan> {
    let declared_ref = spec
        .declared_ref
        .clone()
        .unwrap_or_else(|| models::model_ref_for_path(&spec.model_ref));
    validate_local_required_source(&spec.model_ref, &declared_ref)?;
    let resolved_path = spec.model_ref.canonicalize().with_context(|| {
        format!(
            "canonicalize local-required GGUF for {declared_ref}: {}",
            spec.model_ref.display()
        )
    })?;
    let mmproj_path = spec
        .mmproj_ref
        .as_ref()
        .map(|path| {
            let projector_id = format!("{declared_ref} multimodal projector");
            validate_local_required_source(path, &projector_id)?;
            path.canonicalize().with_context(|| {
                format!(
                    "canonicalize local-required multimodal projector: {}",
                    path.display()
                )
            })
        })
        .transpose()?;
    Ok(StartupModelPlan {
        declared_ref,
        config_model_id: spec.config_model_id.clone(),
        resolved_path,
        preindexed_split_package: None,
        mmproj_path,
        ctx_size: spec.ctx_size,
        gpu_id: spec.gpu_id.clone(),
        pinned_gpu: None,
        parallel: spec.parallel,
        cache_type_k: spec.cache_type_k.clone(),
        cache_type_v: spec.cache_type_v.clone(),
        n_batch: spec.n_batch,
        n_ubatch: spec.n_ubatch,
        flash_attention: spec.flash_attention,
        local_source_required: true,
        profile: spec.profile.clone(),
    })
}
