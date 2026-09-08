use std::{
    cmp::Reverse,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context, Result};

use super::{manifest_artifact_bytes, safe_manifest_file_path, verify_cached_hf_package_files};

pub(super) fn cached_package_snapshots(
    cache_dir: &Path,
    repo_folder: &str,
) -> Result<Vec<PathBuf>> {
    let snapshots_dir = cache_dir.join(repo_folder).join("snapshots");
    let Ok(entries) = fs::read_dir(&snapshots_dir) else {
        return Ok(Vec::new());
    };
    let mut snapshots = Vec::new();
    for entry in entries {
        let entry =
            entry.with_context(|| format!("read snapshot entry {}", snapshots_dir.display()))?;
        let path = entry.path();
        if path.join("model-package.json").is_file() {
            snapshots.push(path);
        }
    }
    snapshots.sort_by_key(|path| cached_snapshot_sort_key(path));
    Ok(snapshots)
}

pub(super) fn resolve_cached_hf_package_snapshot(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Option<String>> {
    if !should_prefer_cached_snapshot_for_request(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )? {
        tracing::debug!(
            package_dir = %package_dir.display(),
            "cached HF layer package snapshot is metadata-only or incomplete; looking for a better snapshot"
        );
        return Ok(None);
    }
    verify_cached_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}

fn cached_snapshot_sort_key(path: &Path) -> (Reverse<SystemTime>, PathBuf) {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    (Reverse(modified), path.to_path_buf())
}

fn should_prefer_cached_snapshot_for_request(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    _include_embeddings: bool,
    _include_output: bool,
) -> Result<bool> {
    let manifest_contents =
        fs::read(package_dir.join("model-package.json")).context("read cached package manifest")?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&manifest_contents)
        .context("parse cached package manifest")?
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if schema_version == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION)) {
        return Ok(layer_start == layer_end);
    }
    // Metadata-only probes (layer_start == layer_end == 0) need at least one
    // layer artifact on disk so the snapshot hash that gets baked into the
    // canonical package_ref is not a skeleton.  Real stage loads only need
    // their assigned layer range.
    if layer_start == 0 && layer_end == 0 {
        cached_snapshot_has_any_layer_artifact(package_dir)
    } else {
        cached_snapshot_has_requested_layers(package_dir, layer_start, layer_end)
    }
}

/// Returns `true` when at least one declared layer artifact is present on disk.
/// Used for metadata-only probes (`layer_start == layer_end == 0`) to
/// distinguish a real snapshot from a skeleton (manifest + shared/ only).
fn cached_snapshot_has_any_layer_artifact(package_dir: &Path) -> Result<bool> {
    let manifest_contents =
        fs::read(package_dir.join("model-package.json")).context("read cached package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse cached package manifest")?;
    let Some(layers) = manifest.get("layers").and_then(|layers| layers.as_array()) else {
        return Ok(false);
    };
    for layer in layers {
        let Some(path) = layer.get("path").and_then(|path| path.as_str()) else {
            continue;
        };
        let path = safe_manifest_file_path(path)?;
        let Ok(metadata) = fs::metadata(package_dir.join(path)) else {
            continue;
        };
        if metadata.is_file() {
            if let Some(expected_bytes) = manifest_artifact_bytes(layer) {
                if metadata.len() == expected_bytes {
                    return Ok(true);
                }
            } else {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Returns `true` when every layer in `[layer_start, layer_end)` is present on
/// disk with the expected size.  Used for real stage loads where only the
/// assigned range needs to be available locally.
fn cached_snapshot_has_requested_layers(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
) -> Result<bool> {
    let manifest_contents =
        fs::read(package_dir.join("model-package.json")).context("read cached package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse cached package manifest")?;
    let Some(layers) = manifest.get("layers").and_then(|layers| layers.as_array()) else {
        return Ok(false);
    };
    for (i, layer) in layers.iter().enumerate() {
        let idx = layer
            .get("layer_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(i as u64) as u32;
        if idx < layer_start || idx >= layer_end {
            continue;
        }
        let Some(path) = layer.get("path").and_then(|path| path.as_str()) else {
            return Ok(false);
        };
        let path = safe_manifest_file_path(path)?;
        let Ok(metadata) = fs::metadata(package_dir.join(path)) else {
            return Ok(false);
        };
        if !metadata.is_file() {
            return Ok(false);
        }
        if let Some(expected_bytes) = manifest_artifact_bytes(layer)
            && metadata.len() != expected_bytes
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// Write a multi-layer snapshot with `layer_count` layers.  Layers in
    /// `present_layers` get a real file on disk; the rest are declared in
    /// the manifest but missing from the filesystem.
    fn write_multi_layer_snapshot(dir: &Path, layer_count: u32, present_layers: &[u32]) {
        fs::create_dir_all(dir.join("shared")).unwrap();
        fs::create_dir_all(dir.join("layers")).unwrap();
        fs::write(dir.join("shared/metadata.gguf"), b"metadata").unwrap();

        let layer_content = b"layer";
        let layer_sha = sha256_hex(layer_content);
        let layers: Vec<_> = (0..layer_count)
            .map(|i| {
                let path = format!("layers/layer-{i:03}.gguf");
                if present_layers.contains(&i) {
                    fs::write(dir.join(&path), layer_content).unwrap();
                }
                serde_json::json!({
                    "layer_index": i,
                    "path": path,
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": layer_content.len(),
                    "sha256": layer_sha,
                })
            })
            .collect();

        let manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "model-a",
            "source_model": {
                "path": "model-a.gguf",
                "sha256": "aaaa",
                "files": []
            },
            "format": "layer-package",
            "layer_count": layer_count,
            "activation_width": 4096,
            "shared": {
                "metadata": {
                    "path": "shared/metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                }
            },
            "layers": layers,
            "skippy_abi_version": "0.1.0",
        });
        fs::write(
            dir.join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    // --- cached_snapshot_has_any_layer_artifact ---

    #[test]
    fn any_layer_artifact_rejects_skeleton_with_no_layers() {
        let dir = tempfile::tempdir().unwrap();
        write_multi_layer_snapshot(dir.path(), 4, &[]);
        assert!(!cached_snapshot_has_any_layer_artifact(dir.path()).unwrap());
    }

    #[test]
    fn any_layer_artifact_accepts_single_layer_present() {
        let dir = tempfile::tempdir().unwrap();
        write_multi_layer_snapshot(dir.path(), 4, &[2]);
        assert!(cached_snapshot_has_any_layer_artifact(dir.path()).unwrap());
    }

    #[test]
    fn any_layer_artifact_accepts_all_layers_present() {
        let dir = tempfile::tempdir().unwrap();
        write_multi_layer_snapshot(dir.path(), 4, &[0, 1, 2, 3]);
        assert!(cached_snapshot_has_any_layer_artifact(dir.path()).unwrap());
    }

    // --- cached_snapshot_has_requested_layers ---

    #[test]
    fn requested_layers_accepts_when_range_present() {
        let dir = tempfile::tempdir().unwrap();
        // 8 layers, only 4..8 present on disk
        write_multi_layer_snapshot(dir.path(), 8, &[4, 5, 6, 7]);
        assert!(cached_snapshot_has_requested_layers(dir.path(), 4, 8).unwrap());
    }

    #[test]
    fn requested_layers_rejects_when_range_partially_missing() {
        let dir = tempfile::tempdir().unwrap();
        // 8 layers, only 4,5,7 present — layer 6 missing
        write_multi_layer_snapshot(dir.path(), 8, &[4, 5, 7]);
        assert!(!cached_snapshot_has_requested_layers(dir.path(), 4, 8).unwrap());
    }

    #[test]
    fn requested_layers_accepts_when_only_requested_subset_present() {
        let dir = tempfile::tempdir().unwrap();
        // 8 layers, only 2,3 present — outside range missing is fine
        write_multi_layer_snapshot(dir.path(), 8, &[2, 3]);
        assert!(cached_snapshot_has_requested_layers(dir.path(), 2, 4).unwrap());
    }

    #[test]
    fn requested_layers_rejects_completely_empty() {
        let dir = tempfile::tempdir().unwrap();
        write_multi_layer_snapshot(dir.path(), 8, &[]);
        assert!(!cached_snapshot_has_requested_layers(dir.path(), 0, 4).unwrap());
    }

    // --- should_prefer_cached_snapshot_for_request (dispatch) ---

    #[test]
    fn metadata_probe_uses_any_layer_check() {
        let dir = tempfile::tempdir().unwrap();
        // 8 layers, only layer 5 present — metadata probe should accept
        write_multi_layer_snapshot(dir.path(), 8, &[5]);
        assert!(should_prefer_cached_snapshot_for_request(dir.path(), 0, 0, false, false).unwrap());
    }

    #[test]
    fn metadata_probe_rejects_skeleton() {
        let dir = tempfile::tempdir().unwrap();
        write_multi_layer_snapshot(dir.path(), 8, &[]);
        assert!(
            !should_prefer_cached_snapshot_for_request(dir.path(), 0, 0, false, false).unwrap()
        );
    }

    #[test]
    fn stage_load_uses_requested_range_check() {
        let dir = tempfile::tempdir().unwrap();
        // 8 layers, only 4..8 present
        write_multi_layer_snapshot(dir.path(), 8, &[4, 5, 6, 7]);
        // Requesting 4..8 = passes
        assert!(should_prefer_cached_snapshot_for_request(dir.path(), 4, 8, false, false).unwrap());
        // Requesting 0..4 = fails (layers 0-3 missing)
        assert!(
            !should_prefer_cached_snapshot_for_request(dir.path(), 0, 4, false, false).unwrap()
        );
    }
}
