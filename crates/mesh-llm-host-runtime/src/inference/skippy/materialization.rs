use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use skippy_protocol::LoadMode;
use skippy_runtime::package::PackageGenerationInfo;
use skippy_runtime::package::{self, LayerPackageInfo};

use super::StageLoadRequest;

mod cache_management;
mod package_download;

pub use cache_management::{
    materialized_stages_for_sources, prune_unpinned_materialized_stages,
    remove_materialized_stages_for_sources,
};
pub use package_download::{
    StagePackageRef, is_layer_package_ref, resolve_hf_package_to_local,
    resolve_package_v2_full_model_to_local, resolve_package_v2_stage_to_local,
};

pub fn configure_materialized_stage_cache() {
    if std::env::var_os("SKIPPY_MATERIALIZED_DIR").is_none() {
        // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
        // runtime work can access the process environment. The current call
        // graph does not enforce that startup boundary; retain the audit TODO.
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("SKIPPY_MATERIALIZED_DIR", materialized_stage_cache_dir()) };
    }
}

pub fn materialized_stage_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-stages")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePackageInfo {
    pub package_ref: String,
    pub package_dir: PathBuf,
    pub manifest_sha256: String,
    pub model_id: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub layer_count: u32,
    pub activation_width: u32,
    pub generation: Option<PackageGenerationInfo>,
    pub projector_path: Option<String>,
    pub layers: Vec<StagePackageLayerInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePackageLayerInfo {
    pub layer_index: u32,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub artifact_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedStagePackage {
    pub local_ref: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub model_part_paths: Vec<PathBuf>,
    pub projector_path: Option<PathBuf>,
}

pub fn ensure_package_manifest_sha(package_ref: &str, expected_sha256: &str) -> Result<()> {
    if expected_sha256.trim().is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        expected_sha256.len() == 64 && expected_sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
        "package manifest sha256 must be a hex SHA-256 digest"
    );
    let manifest_path = Path::new(package_ref).join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let actual_sha = hex::encode(Sha256::digest(&manifest_contents));
    anyhow::ensure!(
        actual_sha.eq_ignore_ascii_case(expected_sha256),
        "package manifest sha256 mismatch"
    );
    Ok(())
}

pub fn inspect_stage_package(package_ref: &str) -> Result<StagePackageInfo> {
    // Resolve hf:// to local for inspection, downloading the manifest and any
    // shared package metadata that resolver path needs.
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let info = package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect skippy layer package {package_ref}"))?;
    stage_package_info(package_ref, info)
}

/// Resolve an `hf://` package ref in a stage load request to a local directory.
/// Returns the resolved local path if the package ref needed resolution, or `None`
/// if it was already local / not a layer package.
pub fn resolve_stage_load_package(load: &StageLoadRequest) -> Result<Option<ResolvedStagePackage>> {
    if load.load_mode == LoadMode::RuntimeSlice && is_layer_package_ref(&load.package_ref) {
        let descriptor = skippy_package_format::stage_admission::StageAdmissionDescriptor {
            package_id: load.admission.package_id.clone(),
            resident_tensor_ids: load.admission.resident_tensor_ids.clone(),
            sidecars: load
                .admission
                .sidecars
                .iter()
                .map(|sidecar| skippy_package_format::Sidecar {
                    kind: match sidecar.kind {
                        skippy_protocol::StageAdmissionSidecarKind::Mmproj => {
                            skippy_package_format::SidecarKind::Mmproj
                        }
                    },
                    artifact_id: sidecar.artifact_id.clone(),
                    name: sidecar.name.clone(),
                })
                .collect(),
        };
        let (local_ref, model_part_paths, projector_path) =
            resolve_package_v2_stage_to_local(&load.package_ref, &descriptor)?;
        ensure_package_manifest_sha(&local_ref, &load.manifest_sha256)?;
        let manifest_path = Path::new(&local_ref).join("model-package.json");
        let manifest: skippy_package_format::PackageManifest =
            serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                format!("read package-v2 manifest {}", manifest_path.display())
            })?)
            .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
        let source_model_bytes = manifest
            .source_model
            .files
            .iter()
            .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
            .context("package-v2 source byte count overflow")?;
        let source_model_path = model_part_paths
            .first()
            .context("package-v2 admission selected no model artifacts")?
            .to_string_lossy()
            .into_owned();
        return Ok(Some(ResolvedStagePackage {
            local_ref,
            source_model_path,
            source_model_sha256: manifest.source_model.sha256,
            source_model_bytes: Some(source_model_bytes),
            model_part_paths,
            projector_path,
        }));
    }
    anyhow::ensure!(
        load.load_mode != LoadMode::LayerPackage,
        "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
    );
    Ok(None)
}

fn stage_package_info(package_ref: &str, info: LayerPackageInfo) -> Result<StagePackageInfo> {
    Ok(StagePackageInfo {
        package_ref: package_ref.to_string(),
        package_dir: info.package_dir,
        manifest_sha256: info.manifest_sha256,
        model_id: info.model_id,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        activation_width: 0,
        generation: info.generation,
        projector_path: info
            .projectors
            .first()
            .map(|projector| projector.path.to_string_lossy().to_string()),
        layers: info
            .layers
            .into_iter()
            .map(|layer| StagePackageLayerInfo {
                layer_index: layer.layer_index,
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{FlashAttentionType, LoadMode};

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn write_local_package_v2_fixture(root: &Path) -> (String, String) {
        use skippy_package_format::{
            Artifact, ArtifactCatalog, PackageManifest, SourceFile, SourceModel, Tensor,
            TensorCatalog, TensorIntegrity, TensorStorage,
        };
        let primary = b"primary";
        let resident = b"resident";
        let unused = b"unused";
        fs::create_dir_all(root.join("artifacts")).unwrap();
        fs::write(root.join("artifacts/primary.gguf"), primary).unwrap();
        fs::write(root.join("artifacts/resident.gguf"), resident).unwrap();
        fs::write(root.join("artifacts/unused.gguf"), unused).unwrap();
        let mut manifest = PackageManifest {
            schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "model-a".to_string(),
            source_model: SourceModel {
                sha256: sha256_hex(primary),
                metadata_artifact_id: "primary".to_string(),
                repo: None,
                revision: None,
                primary_file: Some("model.gguf".to_string()),
                canonical_ref: None,
                distribution_id: None,
                files: vec![SourceFile {
                    path: "model.gguf".to_string(),
                    byte_size: primary.len() as u64,
                    sha256: sha256_hex(primary),
                }],
            },
            format: "gguf".to_string(),
            layer_count: 1,
            model_metadata: std::collections::BTreeMap::from([(
                "general.architecture".to_string(),
                serde_json::Value::String("llama".to_string()),
            )]),
            artifact_catalog: ArtifactCatalog {
                entries: vec![
                    Artifact {
                        id: "primary".to_string(),
                        path: "artifacts/primary.gguf".to_string(),
                        byte_size: primary.len() as u64,
                        sha256: sha256_hex(primary),
                    },
                    Artifact {
                        id: "resident".to_string(),
                        path: "artifacts/resident.gguf".to_string(),
                        byte_size: resident.len() as u64,
                        sha256: sha256_hex(resident),
                    },
                    Artifact {
                        id: "unused".to_string(),
                        path: "artifacts/unused.gguf".to_string(),
                        byte_size: unused.len() as u64,
                        sha256: sha256_hex(unused),
                    },
                ],
            },
            tensor_catalog: TensorCatalog {
                entries: vec![Tensor {
                    id: "resident-tensor".to_string(),
                    name: "blk.0.weight".to_string(),
                    ggml_type: 0,
                    dimensions: vec![1],
                    layer_ordinal: Some(0),
                    storage: TensorStorage::Owned {
                        artifact_id: "resident".to_string(),
                        data_offset: 0,
                        stored_length: resident.len() as u64,
                        alignment: 1,
                        integrity: TensorIntegrity::ArtifactSha256,
                    },
                }],
            },
            sidecars: Vec::new(),
            generation: None,
            native_abi_version: "test".to_string(),
            generator_version: "test".to_string(),
            created_at_unix_secs: 1,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();
        let package_id = manifest.package_id.clone();
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        let manifest_sha = sha256_hex(&bytes);
        fs::write(root.join("model-package.json"), bytes).unwrap();
        (manifest_sha, package_id)
    }

    fn stage_load_request_for_package(
        package_dir: &Path,
        manifest_sha256: String,
    ) -> StageLoadRequest {
        StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            runtime_profile: Some(String::new()),
            backend: "skippy".to_string(),
            package_ref: package_dir.to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            admission: crate::inference::skippy::test_stage_admission(0, 1),
            participant_set_hash: "participants".to_string(),
            topology_hash: "topology".to_string(),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
            topology_stages: Vec::new(),
            model_path: Some(package_dir.to_string_lossy().to_string()),
            source_model_bytes: None,
            source_model_sha256: None,
            local_source_required: false,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            ctx_size: 8192,
            lane_count: 1,
            continuous_batching: true,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            runtime_settings: Default::default(),
            native_mtp_enabled: true,
            shutdown_generation: 1,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        }
    }

    fn write_cached_package_snapshot(snapshot: &Path, layer_sha: String) {
        fs::create_dir_all(snapshot.join("shared")).unwrap();
        fs::create_dir_all(snapshot.join("layers")).unwrap();
        fs::write(snapshot.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(snapshot.join("layers/layer-000.gguf"), b"layer").unwrap();
        fs::write(
            snapshot.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "model_id": "model-a",
                "source_model": {
                    "path": "model-a.gguf",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "files": [
                        {
                            "path": "model-a.gguf",
                            "size_bytes": 123,
                            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        }
                    ]
                },
                "format": "layer-package",
                "layer_count": 1,
                "shared": {
                    "metadata": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "embeddings": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "output": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    }
                },
                "layers": [
                    {
                        "layer_index": 0,
                        "path": "layers/layer-000.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 5,
                        "sha256": layer_sha
                    }
                ],
                "skippy_abi_version": "0.1.0",
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn resolve_stage_load_package_rejects_legacy_package_mode() {
        let dir = tempfile::tempdir().unwrap();
        let load = stage_load_request_for_package(dir.path(), "0".repeat(64));
        let error = resolve_stage_load_package(&load).unwrap_err().to_string();
        assert!(error.contains("package-v2 graph admission"), "{error}");
    }

    #[test]
    fn runtime_slice_package_v2_resolves_only_admitted_model_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let (manifest_sha, package_id) = write_local_package_v2_fixture(dir.path());
        let mut load = stage_load_request_for_package(dir.path(), manifest_sha);
        load.load_mode = LoadMode::RuntimeSlice;
        load.admission.package_id = package_id;
        load.admission.resident_tensor_ids = vec!["resident-tensor".to_string()];

        let resolved = resolve_stage_load_package(&load).unwrap().unwrap();
        assert_eq!(
            resolved.model_part_paths,
            vec![
                dir.path().join("artifacts/primary.gguf"),
                dir.path().join("artifacts/resident.gguf")
            ]
        );
        assert!(
            !resolved
                .model_part_paths
                .iter()
                .any(|path| path.ends_with("unused.gguf"))
        );
    }

    #[test]
    fn resolved_stage_load_package_does_not_accept_v1_manifest() {
        let dir = tempfile::tempdir().unwrap();
        write_cached_package_snapshot(dir.path(), sha256_hex(b"layer"));
        let manifest_bytes = fs::read(dir.path().join("model-package.json")).unwrap();
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let load = StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            runtime_profile: Some(String::new()),
            backend: "skippy".to_string(),
            package_ref: dir.path().to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            admission: crate::inference::skippy::test_stage_admission(0, 1),
            participant_set_hash: "participants".to_string(),
            topology_hash: "topology".to_string(),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
            topology_stages: Vec::new(),
            model_path: None,
            source_model_bytes: None,
            source_model_sha256: None,
            local_source_required: false,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            ctx_size: 512,
            lane_count: 1,
            continuous_batching: true,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
            runtime_settings: Default::default(),
            native_mtp_enabled: true,
            shutdown_generation: 0,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        };

        let error = resolve_stage_load_package(&load)
            .expect_err("v1 package selection must not remain reachable from serving");
        assert!(error.to_string().contains("package-v2 graph admission"));
    }
}
