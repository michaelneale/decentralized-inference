use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result};
use skippy_protocol::LoadMode;
use tokio::sync::Mutex;

use crate::inference::skippy::materialization::{
    inspect_stage_package, is_layer_package_ref, resolve_stage_load_package,
};

use super::{
    SourceModelKind, StageInventoryRequest, StageLoadRequest, StagePackagePrefetcher,
    StagePreparationState, StagePreparationStatus, StagePrepareRequest,
    preparation_status_from_load,
};

#[derive(Clone, Debug)]
pub(super) struct InventorySource {
    pub(super) path: PathBuf,
    pub(super) bytes: Option<u64>,
    pub(super) layer_count: u32,
    pub(super) kind: SourceModelKind,
    pub(super) sha256: Option<String>,
}

pub(super) fn resolve_inventory_source(request: &StageInventoryRequest) -> Option<InventorySource> {
    let local_source_required = crate::inference::skippy::effective_local_source_required(
        &request.model_id,
        request.runtime_profile.as_deref(),
        request.local_source_required,
    );
    if local_source_required
        && !crate::inference::skippy::is_content_addressed_gguf_ref(&request.package_ref)
    {
        return None;
    }
    if crate::inference::skippy::is_content_addressed_gguf_ref(&request.package_ref) {
        let expected_sha256 = request.expected_source_model_sha256.as_deref()?;
        let identity = crate::inference::skippy::verify_registered_content_source(
            &request.model_id,
            &request.package_ref,
            &request.manifest_sha256,
            expected_sha256,
        )
        .map_err(|error| {
            tracing::debug!(
                package_ref = request.package_ref,
                error = %error,
                "content-addressed GGUF inventory verification failed"
            );
            error
        })
        .ok()?;
        let kind = if is_split_gguf_path(&identity.source_model_path) {
            SourceModelKind::SplitGguf
        } else {
            SourceModelKind::PlainGguf
        };
        return Some(InventorySource {
            path: identity.source_model_path,
            bytes: Some(identity.source_model_bytes),
            layer_count: identity.layer_count,
            kind,
            sha256: Some(identity.source_model_sha256),
        });
    }
    if is_layer_package_ref(&request.package_ref) {
        let identity = crate::inference::skippy::identity_from_layer_package(&request.package_ref)
            .map_err(|error| {
                tracing::debug!(
                    package_ref = request.package_ref,
                    error = %error,
                    "package inventory verification failed"
                );
                error
            })
            .ok()?;
        if crate::inference::skippy::is_package_v2_identity(&identity) {
            let kind = if is_split_gguf_path(&identity.source_model_path) {
                SourceModelKind::SplitGguf
            } else {
                SourceModelKind::PlainGguf
            };
            return Some(InventorySource {
                path: identity.source_model_path,
                bytes: Some(identity.source_model_bytes),
                layer_count: identity.layer_count,
                kind,
                sha256: Some(identity.source_model_sha256),
            });
        }
        let info = inspect_stage_package(&request.package_ref).ok()?;
        return Some(InventorySource {
            path: info.package_dir,
            bytes: info.source_model_bytes,
            layer_count: info.layer_count,
            kind: SourceModelKind::LayerPackage,
            sha256: Some(info.source_model_sha256),
        });
    }

    for candidate in inventory_source_candidates(request) {
        if let Some(source) = resolve_direct_gguf_inventory_source(&candidate) {
            return Some(source);
        }
    }
    None
}

fn resolve_direct_gguf_inventory_source(candidate: &Path) -> Option<InventorySource> {
    let source_paths = match crate::inference::skippy::direct_gguf_source_paths(candidate) {
        Ok(paths) => paths,
        Err(error) => {
            tracing::debug!(
                path = %candidate.display(),
                "direct GGUF inventory source is unavailable: {error:#}"
            );
            return None;
        }
    };
    let source_path = source_paths.first()?.clone();
    let layer_count = crate::models::gguf::scan_gguf_compact_meta(&source_path)
        .and_then(|meta| meta.executable_layer_count())
        .or_else(|| crate::inference::skippy::infer_layer_count(&source_path).ok())?;
    let bytes = source_paths
        .iter()
        .filter_map(|path| path.metadata().ok().map(|metadata| metadata.len()))
        .sum();
    let kind = if is_split_gguf_path(&source_path) {
        SourceModelKind::SplitGguf
    } else {
        SourceModelKind::PlainGguf
    };
    Some(InventorySource {
        path: source_path,
        bytes: Some(bytes),
        layer_count,
        kind,
        sha256: None,
    })
}

pub(super) fn inventory_source_candidates(request: &StageInventoryRequest) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = request.package_ref.strip_prefix("gguf://")
        && !path.is_empty()
    {
        candidates.push(PathBuf::from(path));
    }
    if !request.model_id.is_empty() {
        candidates.push(crate::models::find_model_path(&request.model_id));
    }
    candidates
}

fn is_split_gguf_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(model_ref::split_gguf_shard_info)
        .is_some()
}

pub(super) async fn run_stage_prepare_task(
    preparations: Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: String,
    request: StagePrepareRequest,
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
    cancelled: Arc<AtomicBool>,
) {
    let load = request.load.clone();
    if !update_preparation(
        &preparations,
        &key,
        preparation_status_from_load(&load, StagePreparationState::Resolving, None),
    )
    .await
        || cancelled.load(Ordering::Acquire)
    {
        return;
    }
    let peer_prefetch_error =
        prefetch_stage_package_if_needed(&preparations, &key, &request, package_prefetcher).await;
    if cancelled.load(Ordering::Acquire) {
        return;
    }
    if peer_prefetch_error.is_none()
        && load.load_mode != LoadMode::LayerPackage
        && !is_layer_package_ref(&load.package_ref)
        && !update_preparation(
            &preparations,
            &key,
            preparation_status_from_load(&load, StagePreparationState::Downloading, None),
        )
        .await
    {
        return;
    }
    let result = prepare_stage_source(&load).await;
    if cancelled.load(Ordering::Acquire) {
        return;
    }
    let state = match result {
        Ok(PrepareSourceResult { bytes_total }) => {
            let mut status =
                preparation_status_from_load(&load, StagePreparationState::Available, None);
            status.bytes_done = bytes_total;
            status.bytes_total = bytes_total;
            status
        }
        Err(error) => {
            let mut status =
                preparation_status_from_load(&load, StagePreparationState::Failed, None);
            status.error = Some(format_stage_prepare_error(
                &error,
                peer_prefetch_error.as_deref(),
            ));
            status
        }
    };
    update_preparation(&preparations, &key, state).await;
}

async fn prefetch_stage_package_if_needed(
    preparations: &Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: &str,
    request: &StagePrepareRequest,
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
) -> Option<String> {
    let load = &request.load;
    if load.load_mode != LoadMode::LayerPackage && !is_layer_package_ref(&load.package_ref) {
        return None;
    }
    let prefetcher = package_prefetcher?;
    let _ = update_preparation(
        preparations,
        key,
        preparation_status_from_load(load, StagePreparationState::Downloading, None),
    )
    .await;
    match prefetcher.prefetch_stage_package(request).await {
        Ok(()) => None,
        Err(error) => {
            let error_message = format!("{error:#}");
            tracing::debug!(
                topology_id = %load.topology_id,
                run_id = %load.run_id,
                stage_id = %load.stage_id,
                "peer artifact prefetch failed, falling back to local/HF resolver: {error_message}"
            );
            Some(error_message)
        }
    }
}

fn format_stage_prepare_error(error: &anyhow::Error, peer_prefetch_error: Option<&str>) -> String {
    let message = format!("{error:#}");
    match peer_prefetch_error {
        Some(prefetch_error) => {
            format!("{message}; peer artifact prefetch failed: {prefetch_error}")
        }
        None => message,
    }
}

#[derive(Debug)]
pub(super) struct PrepareSourceResult {
    bytes_total: Option<u64>,
}

pub(super) async fn prepare_stage_source(load: &StageLoadRequest) -> Result<PrepareSourceResult> {
    let mut effective_load = load.clone();
    let (effective_load, verified) = tokio::task::spawn_blocking(move || {
        let verified = crate::inference::skippy::apply_verified_local_source(&mut effective_load)?;
        anyhow::Ok((effective_load, verified))
    })
    .await
    .context("join verify local-required stage source task")??;
    if verified {
        return Ok(PrepareSourceResult {
            bytes_total: effective_load.source_model_bytes,
        });
    }
    if load.load_mode == LoadMode::LayerPackage || is_layer_package_ref(&load.package_ref) {
        let load = load.clone();
        let package = tokio::task::spawn_blocking(move || resolve_stage_load_package(&load))
            .await??
            .ok_or_else(|| anyhow::anyhow!("layer package load did not resolve a package"))?;
        return Ok(PrepareSourceResult {
            bytes_total: package.source_model_bytes,
        });
    }

    for candidate in [
        load.model_path.as_deref(),
        Some(load.model_id.as_str()),
        load.package_ref.strip_prefix("gguf://"),
    ]
    .into_iter()
    .flatten()
    .filter(|candidate| !candidate.is_empty())
    {
        match crate::models::resolve_model_spec_with_progress(Path::new(candidate), true).await {
            Ok(path) => {
                let bytes_total = crate::inference::election::total_model_bytes(&path);
                return Ok(PrepareSourceResult {
                    bytes_total: Some(bytes_total),
                });
            }
            Err(last_error) => {
                tracing::debug!(
                    stage_id = %load.stage_id,
                    candidate,
                    error = %last_error,
                    "stage source prepare candidate failed"
                );
            }
        }
    }
    anyhow::bail!("stage source model is not available")
}

async fn update_preparation(
    preparations: &Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: &str,
    status: StagePreparationStatus,
) -> bool {
    let mut preparations = preparations.lock().await;
    if preparations.get(key).is_some_and(|existing| {
        matches!(existing.state, StagePreparationState::Cancelled)
            && existing.shutdown_generation >= status.shutdown_generation
    }) {
        return false;
    }
    preparations.insert(key.to_string(), status);
    true
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;

    use super::format_stage_prepare_error;

    #[test]
    fn stage_prepare_error_preserves_source_chain() {
        let error = anyhow!("No locks available (os error 77)")
            .context("download layer package file: shared/embeddings.gguf");

        let message = format_stage_prepare_error(&error, None);

        assert!(message.contains("download layer package file: shared/embeddings.gguf"));
        assert!(message.contains("No locks available (os error 77)"));
    }

    #[test]
    fn stage_prepare_error_includes_prefetch_source_chain() {
        let error = anyhow!("No locks available (os error 77)")
            .context("download layer package file: shared/embeddings.gguf");

        let message = format_stage_prepare_error(&error, Some("peer refused package"));

        assert!(message.contains("No locks available (os error 77)"));
        assert!(message.contains("peer artifact prefetch failed: peer refused package"));
    }
}
