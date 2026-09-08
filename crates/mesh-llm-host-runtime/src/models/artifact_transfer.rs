use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) const PACKAGE_MANIFEST_FILE: &str = "model-package.json";
pub(crate) const MAX_PACKAGE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArtifactTransferMode {
    Disabled,
    TrustedOnly,
    Open,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PackageArtifactRequest {
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) relative_path: PathBuf,
    pub(crate) expected_size: Option<u64>,
    pub(crate) expected_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServableArtifact {
    pub(crate) path: PathBuf,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HfPackageRef {
    repo: String,
    revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ManifestArtifact {
    relative_path: PathBuf,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StageArtifactSelection {
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) include_embeddings: bool,
    pub(crate) include_output: bool,
    pub(crate) include_projectors: bool,
}

pub(crate) fn artifact_transfer_mode() -> ArtifactTransferMode {
    std::env::var("MESH_LLM_ARTIFACT_TRANSFER")
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" | "open" | "any" | "public" => ArtifactTransferMode::Open,
            "trusted" | "trust" | "owner" | "owners" | "same-owner" | "allowlist" => {
                ArtifactTransferMode::TrustedOnly
            }
            _ => ArtifactTransferMode::Disabled,
        })
        .unwrap_or(ArtifactTransferMode::Disabled)
}

pub(crate) fn artifact_transfer_enabled() -> bool {
    artifact_transfer_mode() != ArtifactTransferMode::Disabled
}

pub(crate) fn artifact_transfer_advertised(local_owner: &crate::crypto::OwnershipSummary) -> bool {
    match artifact_transfer_mode() {
        ArtifactTransferMode::Disabled => false,
        ArtifactTransferMode::Open => true,
        ArtifactTransferMode::TrustedOnly => verified_owner_id(local_owner).is_some(),
    }
}

pub(crate) fn artifact_transfer_allowed_between(
    local_owner: &crate::crypto::OwnershipSummary,
    peer_owner: &crate::crypto::OwnershipSummary,
    trust_store: &crate::crypto::TrustStore,
) -> bool {
    match artifact_transfer_mode() {
        ArtifactTransferMode::Disabled => false,
        ArtifactTransferMode::Open => true,
        ArtifactTransferMode::TrustedOnly => {
            let Some(local_owner_id) = verified_owner_id(local_owner) else {
                return false;
            };
            let Some(peer_owner_id) = verified_owner_id(peer_owner) else {
                return false;
            };
            local_owner_id == peer_owner_id
                || trust_store
                    .trusted_owners
                    .iter()
                    .any(|entry| entry.owner_id == peer_owner_id)
        }
    }
}

fn verified_owner_id(summary: &crate::crypto::OwnershipSummary) -> Option<&str> {
    if summary.status == crate::crypto::OwnershipStatus::Verified && summary.verified {
        summary.owner_id.as_deref()
    } else {
        None
    }
}

pub(crate) fn safe_relative_artifact_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.trim().is_empty(), "artifact path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("artifact path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "artifact path must be a safe relative path"
    );
    Ok(path.to_path_buf())
}

pub(crate) fn package_cache_dir_for_ref(package_ref: &str) -> Result<PathBuf> {
    let parsed = parse_hf_package_ref(package_ref)?;
    let revision_path = safe_relative_artifact_path(&parsed.revision)
        .context("unsupported package revision for peer transfer")?;
    Ok(hf_repo_cache_root(&parsed.repo)
        .join("snapshots")
        .join(revision_path))
}

pub(crate) fn manifest_artifact_request(
    package_ref: &str,
    manifest_sha256: &str,
) -> Result<PackageArtifactRequest> {
    validate_sha256(manifest_sha256).context("invalid package manifest sha256")?;
    Ok(PackageArtifactRequest {
        package_ref: package_ref.to_string(),
        manifest_sha256: manifest_sha256.to_ascii_lowercase(),
        relative_path: PathBuf::from(PACKAGE_MANIFEST_FILE),
        expected_size: None,
        expected_sha256: Some(manifest_sha256.to_ascii_lowercase()),
    })
}

pub(crate) fn required_stage_package_artifacts(
    package_dir: &Path,
    package_ref: &str,
    manifest_sha256: &str,
    selection: StageArtifactSelection,
) -> Result<Vec<PackageArtifactRequest>> {
    validate_sha256(manifest_sha256).context("invalid package manifest sha256")?;
    let manifest_contents = read_bounded_package_manifest(package_dir)?;
    let actual_manifest_sha = sha256_bytes(&manifest_contents);
    anyhow::ensure!(
        actual_manifest_sha.eq_ignore_ascii_case(manifest_sha256),
        "package manifest sha256 mismatch"
    );
    let manifest: Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    push_manifest_artifact(
        &mut out,
        &mut seen,
        package_ref,
        manifest_sha256,
        manifest
            .pointer("/shared/metadata")
            .context("manifest missing shared metadata")?,
    )?;
    if selection.include_embeddings
        && let Some(embeddings) = manifest.pointer("/shared/embeddings")
    {
        push_manifest_artifact(
            &mut out,
            &mut seen,
            package_ref,
            manifest_sha256,
            embeddings,
        )?;
    }
    if selection.include_output
        && let Some(output) = manifest.pointer("/shared/output")
    {
        push_manifest_artifact(&mut out, &mut seen, package_ref, manifest_sha256, output)?;
    }
    if let Some(layers) = manifest.get("layers").and_then(Value::as_array) {
        for (index, layer) in layers.iter().enumerate() {
            let layer_index = layer
                .get("layer_index")
                .and_then(Value::as_u64)
                .unwrap_or(index as u64) as u32;
            if layer_index >= selection.layer_start && layer_index < selection.layer_end {
                push_manifest_artifact(&mut out, &mut seen, package_ref, manifest_sha256, layer)?;
            }
        }
    }
    if selection.include_projectors
        && let Some(projectors) = manifest.get("projectors").and_then(Value::as_array)
    {
        for projector in projectors {
            push_manifest_artifact(&mut out, &mut seen, package_ref, manifest_sha256, projector)?;
        }
    }
    Ok(out)
}

pub(crate) fn package_manifest_schema_version(package_dir: &Path) -> Result<u64> {
    let manifest_contents = read_bounded_package_manifest(package_dir)?;
    let manifest =
        serde_json::from_slice::<Value>(&manifest_contents).context("parse package manifest")?;
    Ok(manifest
        .get("schema_version")
        .and_then(Value::as_u64)
        .unwrap_or(1))
}

pub(crate) fn required_admitted_stage_package_artifacts(
    package_dir: &Path,
    package_ref: &str,
    manifest_sha256: &str,
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> Result<Vec<PackageArtifactRequest>> {
    validate_sha256(manifest_sha256).context("invalid package manifest sha256")?;
    let manifest_contents = read_bounded_package_manifest(package_dir)?;
    let actual_manifest_sha = sha256_bytes(&manifest_contents);
    anyhow::ensure!(
        actual_manifest_sha.eq_ignore_ascii_case(manifest_sha256),
        "package manifest sha256 mismatch"
    );
    let manifest: skippy_package_format::PackageManifest =
        serde_json::from_slice(&manifest_contents).context("parse package-v2 manifest")?;
    let descriptor = skippy_package_format::stage_admission::StageAdmissionDescriptor {
        package_id: admission.package_id.clone(),
        resident_tensor_ids: admission.resident_tensor_ids.clone(),
        sidecars: admission
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
    let resolved = manifest
        .resolve_stage_admission(&descriptor)
        .context("resolve exact package-v2 stage admission")?;
    resolved
        .required_artifacts
        .into_iter()
        .map(|artifact| {
            Ok(PackageArtifactRequest {
                package_ref: package_ref.to_string(),
                manifest_sha256: manifest_sha256.to_ascii_lowercase(),
                relative_path: safe_relative_artifact_path(&artifact.path)?,
                expected_size: Some(artifact.byte_size),
                expected_sha256: Some(artifact.sha256.to_ascii_lowercase()),
            })
        })
        .collect()
}

pub(crate) fn required_stage_package_bytes(
    package_dir: &Path,
    package_ref: &str,
    manifest_sha256: &str,
    selection: StageArtifactSelection,
) -> Result<u64> {
    let manifest_bytes = u64::try_from(read_bounded_package_manifest(package_dir)?.len())
        .context("package manifest length exceeds u64")?;
    required_stage_package_artifacts(package_dir, package_ref, manifest_sha256, selection)?
        .into_iter()
        .try_fold(manifest_bytes, |total, artifact| {
            let artifact_bytes = artifact.expected_size.with_context(|| {
                format!(
                    "package manifest is missing the size of {}",
                    artifact.relative_path.display()
                )
            })?;
            Ok(total.saturating_add(artifact_bytes))
        })
}

pub(crate) fn local_artifact_path(package_dir: &Path, request: &PackageArtifactRequest) -> PathBuf {
    package_dir.join(&request.relative_path)
}

pub(crate) fn ensure_local_artifact_install_parent(
    package_ref: &str,
    destination: &Path,
) -> Result<()> {
    let package_ref = parse_hf_package_ref(package_ref)?;
    let parent = destination
        .parent()
        .context("artifact destination has no parent directory")?;
    let repo_root = hf_repo_cache_root(&package_ref.repo);
    ensure_path_inside_repo_root(&repo_root, parent)
        .context("artifact destination escapes the managed HF cache repo")
}

pub(crate) fn local_artifact_satisfies(
    package_dir: &Path,
    request: &PackageArtifactRequest,
    verify_sha: bool,
) -> Result<bool> {
    let path = local_artifact_path(package_dir, request);
    let Ok(metadata) = fs::metadata(&path) else {
        return Ok(false);
    };
    if !metadata.is_file() {
        return Ok(false);
    }
    if let Some(expected_size) = request.expected_size
        && metadata.len() != expected_size
    {
        return Ok(false);
    }
    if verify_sha && let Some(expected_sha) = request.expected_sha256.as_deref() {
        return Ok(file_sha256_hex(&path)?.eq_ignore_ascii_case(expected_sha));
    }
    Ok(true)
}

pub(crate) fn servable_artifact_from_request(
    request: &skippy_protocol::proto::stage::StageArtifactTransferRequest,
) -> Result<ServableArtifact> {
    let package_ref = parse_hf_package_ref(&request.package_ref)?;
    validate_sha256(&request.manifest_sha256).context("invalid manifest sha256")?;
    if let Some(expected_sha) = request.expected_sha256.as_deref() {
        validate_sha256(expected_sha).context("invalid expected artifact sha256")?;
    }
    let relative_path = safe_relative_artifact_path(&request.relative_path)?;
    let package_dir = package_cache_dir_for_ref(&request.package_ref)?;
    let repo_root = hf_repo_cache_root(&package_ref.repo);
    let path = package_dir.join(&relative_path);
    ensure_path_inside_repo_root(&repo_root, &path)?;

    if relative_path.as_path() == Path::new(PACKAGE_MANIFEST_FILE) {
        let metadata = fs::metadata(&path).context("artifact is not cached")?;
        anyhow::ensure!(metadata.is_file(), "artifact is not a file");
        anyhow::ensure!(
            metadata.len() <= MAX_PACKAGE_MANIFEST_BYTES,
            "package manifest exceeds transfer limit"
        );
        if let Some(expected_size) = request.expected_size {
            anyhow::ensure!(metadata.len() == expected_size, "artifact size mismatch");
        }
        let sha256 = file_sha256_hex(&path)?;
        anyhow::ensure!(
            sha256.eq_ignore_ascii_case(&request.manifest_sha256),
            "manifest sha256 mismatch"
        );
        if let Some(expected_sha) = request.expected_sha256.as_deref() {
            anyhow::ensure!(
                sha256.eq_ignore_ascii_case(expected_sha),
                "artifact sha256 mismatch"
            );
        }
        return Ok(ServableArtifact {
            path,
            size: metadata.len(),
            sha256,
        });
    }

    let manifest_path = package_dir.join(PACKAGE_MANIFEST_FILE);
    ensure_path_inside_repo_root(&repo_root, &manifest_path)?;
    let manifest_contents = read_bounded_package_manifest(&package_dir)?;
    let actual_manifest_sha = sha256_bytes(&manifest_contents);
    anyhow::ensure!(
        actual_manifest_sha.eq_ignore_ascii_case(&request.manifest_sha256),
        "package manifest sha256 mismatch"
    );
    let manifest: Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;
    let declared = declared_manifest_artifacts(&manifest)?
        .into_iter()
        .find(|artifact| artifact.relative_path == relative_path)
        .context("artifact path is not declared by package manifest")?;
    if let Some(expected_size) = request.expected_size {
        anyhow::ensure!(
            expected_size == declared.artifact_bytes,
            "artifact size does not match manifest"
        );
    }
    if let Some(expected_sha) = request.expected_sha256.as_deref() {
        anyhow::ensure!(
            declared.sha256.eq_ignore_ascii_case(expected_sha),
            "artifact sha256 does not match manifest"
        );
    }
    let metadata = fs::metadata(&path).context("artifact is not cached")?;
    anyhow::ensure!(metadata.is_file(), "artifact is not a file");
    anyhow::ensure!(
        metadata.len() == declared.artifact_bytes,
        "cached artifact size mismatch"
    );
    let actual_sha = file_sha256_hex(&path)?;
    anyhow::ensure!(
        actual_sha.eq_ignore_ascii_case(&declared.sha256),
        "cached artifact sha256 mismatch"
    );
    Ok(ServableArtifact {
        path,
        size: declared.artifact_bytes,
        sha256: declared.sha256,
    })
}

pub(crate) fn file_sha256_hex(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = fs::File::open(path).context("open artifact for sha256")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).context("read artifact for sha256")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn parse_hf_package_ref(package_ref: &str) -> Result<HfPackageRef> {
    let rest = package_ref
        .strip_prefix("hf://")
        .context("artifact transfer only supports hf:// package refs")?;
    let (repo, revision) = rest
        .split_once('@')
        .context("artifact transfer requires an explicit immutable hf://namespace/repo@revision")?;
    anyhow::ensure!(
        repo.split('/').count() == 2 && !repo.contains(':') && !repo.contains('@'),
        "HF package repo id must look like namespace/repo"
    );
    let revision = revision.trim();
    anyhow::ensure!(!revision.is_empty(), "HF package revision is empty");
    anyhow::ensure!(
        is_immutable_revision_hint(revision),
        "artifact transfer requires an immutable HF package revision"
    );
    Ok(HfPackageRef {
        repo: repo.to_string(),
        revision: revision.to_string(),
    })
}

fn is_immutable_revision_hint(revision: &str) -> bool {
    !matches!(
        revision.trim().to_ascii_lowercase().as_str(),
        "main" | "master" | "latest" | "dev" | "develop" | "development"
    )
}

fn hf_repo_cache_root(repo: &str) -> PathBuf {
    crate::models::huggingface_hub_cache_dir().join(
        crate::models::local::huggingface_repo_folder_name(repo, hf_hub::RepoTypeModel),
    )
}

fn read_bounded_package_manifest(package_dir: &Path) -> Result<Vec<u8>> {
    let manifest_path = package_dir.join(PACKAGE_MANIFEST_FILE);
    let metadata = fs::metadata(&manifest_path).context("stat package manifest")?;
    anyhow::ensure!(metadata.is_file(), "package manifest is not a file");
    anyhow::ensure!(
        metadata.len() <= MAX_PACKAGE_MANIFEST_BYTES,
        "package manifest exceeds transfer limit"
    );
    fs::read(&manifest_path).context("read package manifest")
}

fn push_manifest_artifact(
    out: &mut Vec<PackageArtifactRequest>,
    seen: &mut HashSet<PathBuf>,
    package_ref: &str,
    manifest_sha256: &str,
    value: &Value,
) -> Result<()> {
    let artifact = manifest_artifact(value)?;
    if seen.insert(artifact.relative_path.clone()) {
        out.push(PackageArtifactRequest {
            package_ref: package_ref.to_string(),
            manifest_sha256: manifest_sha256.to_ascii_lowercase(),
            relative_path: artifact.relative_path,
            expected_size: Some(artifact.artifact_bytes),
            expected_sha256: Some(artifact.sha256),
        });
    }
    Ok(())
}

fn declared_manifest_artifacts(manifest: &Value) -> Result<Vec<ManifestArtifact>> {
    let mut artifacts = Vec::new();
    let mut seen = HashSet::new();
    if manifest.get("schema_version").and_then(Value::as_u64)
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION))
    {
        let entries = manifest
            .pointer("/artifact_catalog/entries")
            .and_then(Value::as_array)
            .context("package-v2 manifest is missing artifact_catalog.entries")?;
        for entry in entries {
            let relative_path = entry
                .get("path")
                .and_then(Value::as_str)
                .context("package-v2 artifact is missing path")
                .and_then(safe_relative_artifact_path)?;
            let artifact_bytes = entry
                .get("byte_size")
                .and_then(Value::as_u64)
                .context("package-v2 artifact is missing byte_size")?;
            anyhow::ensure!(
                artifact_bytes > 0,
                "package artifact bytes must be positive"
            );
            let sha256 = entry
                .get("sha256")
                .and_then(Value::as_str)
                .context("package-v2 artifact is missing sha256")?
                .to_ascii_lowercase();
            validate_sha256(&sha256)?;
            anyhow::ensure!(
                seen.insert(relative_path.clone()),
                "package-v2 artifact catalog contains duplicate path {}",
                relative_path.display()
            );
            artifacts.push(ManifestArtifact {
                relative_path,
                artifact_bytes,
                sha256,
            });
        }
        return Ok(artifacts);
    }
    if let Some(metadata) = manifest.pointer("/shared/metadata") {
        push_declared_artifact(&mut artifacts, &mut seen, metadata)?;
    }
    if let Some(embeddings) = manifest.pointer("/shared/embeddings") {
        push_declared_artifact(&mut artifacts, &mut seen, embeddings)?;
    }
    if let Some(output) = manifest.pointer("/shared/output") {
        push_declared_artifact(&mut artifacts, &mut seen, output)?;
    }
    if let Some(layers) = manifest.get("layers").and_then(Value::as_array) {
        for layer in layers {
            push_declared_artifact(&mut artifacts, &mut seen, layer)?;
        }
    }
    if let Some(projectors) = manifest.get("projectors").and_then(Value::as_array) {
        for projector in projectors {
            push_declared_artifact(&mut artifacts, &mut seen, projector)?;
        }
    }
    Ok(artifacts)
}

fn push_declared_artifact(
    out: &mut Vec<ManifestArtifact>,
    seen: &mut HashSet<PathBuf>,
    value: &Value,
) -> Result<()> {
    let artifact = manifest_artifact(value)?;
    if seen.insert(artifact.relative_path.clone()) {
        out.push(artifact);
    }
    Ok(())
}

fn manifest_artifact(value: &Value) -> Result<ManifestArtifact> {
    let relative_path = value
        .get("path")
        .and_then(Value::as_str)
        .context("package artifact is missing path")
        .and_then(safe_relative_artifact_path)?;
    let artifact_bytes = value
        .get("artifact_bytes")
        .and_then(Value::as_u64)
        .context("package artifact is missing artifact_bytes")?;
    anyhow::ensure!(
        artifact_bytes > 0,
        "package artifact bytes must be positive"
    );
    let sha256 = value
        .get("sha256")
        .and_then(Value::as_str)
        .context("package artifact is missing sha256")?
        .to_ascii_lowercase();
    validate_sha256(&sha256)?;
    Ok(ManifestArtifact {
        relative_path,
        artifact_bytes,
        sha256,
    })
}

fn validate_sha256(value: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()),
        "value is not a SHA-256 digest"
    );
    Ok(())
}

fn ensure_path_inside_repo_root(repo_root: &Path, path: &Path) -> Result<()> {
    let canonical_root =
        fs::canonicalize(repo_root).context("package repo cache is not available")?;
    let canonical_path = fs::canonicalize(path).context("artifact is not cached")?;
    anyhow::ensure!(
        canonical_path.starts_with(canonical_root),
        "artifact path escapes the managed HF cache repo"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use sha2::{Digest, Sha256};
    use std::{ffi::OsString, fs, path::Path};

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::set_var(key, value) };
        } else {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::remove_var(key) };
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    fn verified_owner(owner_id: &str) -> crate::crypto::OwnershipSummary {
        crate::crypto::OwnershipSummary {
            owner_id: Some(owner_id.to_string()),
            status: crate::crypto::OwnershipStatus::Verified,
            verified: true,
            ..crate::crypto::OwnershipSummary::default()
        }
    }

    fn write_package_fixture(root: &Path) -> (PathBuf, String, String) {
        let package_dir = root
            .join("models--meshllm--demo-layers")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(package_dir.join("shared")).unwrap();
        fs::create_dir_all(package_dir.join("layers")).unwrap();
        fs::create_dir_all(package_dir.join("projectors")).unwrap();
        fs::write(package_dir.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(package_dir.join("shared/output.gguf"), b"output").unwrap();
        fs::write(package_dir.join("layers/layer-000.gguf"), b"layer000").unwrap();
        fs::write(package_dir.join("layers/layer-001.gguf"), b"layer001").unwrap();
        fs::write(package_dir.join("projectors/mmproj.gguf"), b"projector").unwrap();
        let manifest = serde_json::json!({
            "format": "layer-package",
            "format_version": 1,
            "model_id": "meshllm/demo",
            "layer_count": 2,
            "activation_width": 8,
            "source_model": {
                "path": "hf://meshllm/demo",
                "sha256": sha256_hex(b"source"),
                "files": [{ "path": "model.gguf", "sha256": sha256_hex(b"source"), "size_bytes": 42 }]
            },
            "shared": {
                "metadata": { "path": "shared/metadata.gguf", "sha256": sha256_hex(b"metadata"), "artifact_bytes": 8, "tensor_count": 1, "tensor_bytes": 8 },
                "output": { "path": "shared/output.gguf", "sha256": sha256_hex(b"output"), "artifact_bytes": 6, "tensor_count": 1, "tensor_bytes": 6 }
            },
            "layers": [
                { "layer_index": 0, "path": "layers/layer-000.gguf", "sha256": sha256_hex(b"layer000"), "artifact_bytes": 8, "tensor_count": 1, "tensor_bytes": 8 },
                { "layer_index": 1, "path": "layers/layer-001.gguf", "sha256": sha256_hex(b"layer001"), "artifact_bytes": 8, "tensor_count": 1, "tensor_bytes": 8 }
            ],
            "projectors": [
                { "kind": "mmproj", "path": "projectors/mmproj.gguf", "sha256": sha256_hex(b"projector"), "artifact_bytes": 9 }
            ]
        });
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        let manifest_sha = sha256_hex(&manifest_bytes);
        fs::write(package_dir.join(PACKAGE_MANIFEST_FILE), manifest_bytes).unwrap();
        (
            package_dir,
            "hf://meshllm/demo-layers@abc123".to_string(),
            manifest_sha,
        )
    }

    fn write_package_v2_manifest(root: &Path) -> (PathBuf, String, String, String) {
        use skippy_package_format::{
            Artifact, ArtifactCatalog, PackageManifest, SourceFile, SourceModel, Tensor,
            TensorCatalog, TensorIntegrity, TensorStorage,
        };
        let digest = "aa".repeat(32);
        let mut manifest = PackageManifest {
            schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "meshllm/demo".to_string(),
            source_model: SourceModel {
                sha256: digest.clone(),
                metadata_artifact_id: "primary".to_string(),
                repo: Some("meshllm/demo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("model-00001-of-00002.gguf".to_string()),
                canonical_ref: None,
                distribution_id: None,
                files: vec![SourceFile {
                    path: "model-00001-of-00002.gguf".to_string(),
                    byte_size: 256,
                    sha256: digest.clone(),
                }],
            },
            format: "gguf".to_string(),
            layer_count: 2,
            model_metadata: std::collections::BTreeMap::from([(
                "general.architecture".to_string(),
                serde_json::Value::String("llama".to_string()),
            )]),
            artifact_catalog: ArtifactCatalog {
                entries: vec![
                    Artifact {
                        id: "primary".to_string(),
                        path: "artifacts/source-00000.gguf".to_string(),
                        byte_size: 256,
                        sha256: digest.clone(),
                    },
                    Artifact {
                        id: "resident".to_string(),
                        path: "artifacts/source-00001.gguf".to_string(),
                        byte_size: 128,
                        sha256: digest.clone(),
                    },
                    Artifact {
                        id: "unowned".to_string(),
                        path: "artifacts/unowned.gguf".to_string(),
                        byte_size: 64,
                        sha256: digest.clone(),
                    },
                ],
            },
            tensor_catalog: TensorCatalog {
                entries: vec![Tensor {
                    id: "tensor-resident".to_string(),
                    name: "blk.1.weight".to_string(),
                    ggml_type: 0,
                    dimensions: vec![4, 4],
                    layer_ordinal: Some(1),
                    storage: TensorStorage::Owned {
                        artifact_id: "resident".to_string(),
                        data_offset: 0,
                        stored_length: 64,
                        alignment: 32,
                        integrity: TensorIntegrity::ArtifactSha256,
                    },
                }],
            },
            sidecars: Vec::new(),
            generation: None,
            native_abi_version: "0.1.49".to_string(),
            generator_version: "test".to_string(),
            created_at_unix_secs: 1,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();
        let package_id = manifest.package_id.clone();
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        let manifest_sha = sha256_hex(&bytes);
        fs::create_dir_all(root).unwrap();
        fs::write(root.join(PACKAGE_MANIFEST_FILE), bytes).unwrap();
        (
            root.to_path_buf(),
            "hf://meshllm/demo@abc123".to_string(),
            manifest_sha,
            package_id,
        )
    }

    #[test]
    fn safe_relative_artifact_path_rejects_absolute_parent_and_empty_paths() {
        for path in [
            "",
            "/tmp/model.gguf",
            "../model.gguf",
            "layers/../model.gguf",
        ] {
            assert!(
                safe_relative_artifact_path(path).is_err(),
                "{path} must be rejected"
            );
        }
        assert_eq!(
            safe_relative_artifact_path("layers/layer-000.gguf").unwrap(),
            PathBuf::from("layers/layer-000.gguf")
        );
    }

    #[test]
    #[serial]
    fn artifact_transfer_policy_defaults_to_disabled_and_supports_opt_in_modes() {
        let prev = std::env::var_os("MESH_LLM_ARTIFACT_TRANSFER");

        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("MESH_LLM_ARTIFACT_TRANSFER") };
        assert_eq!(artifact_transfer_mode(), ArtifactTransferMode::Disabled);
        assert!(!artifact_transfer_enabled());

        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("MESH_LLM_ARTIFACT_TRANSFER", "off") };
        assert_eq!(artifact_transfer_mode(), ArtifactTransferMode::Disabled);
        assert!(!artifact_transfer_enabled());

        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("MESH_LLM_ARTIFACT_TRANSFER", "trusted") };
        assert_eq!(artifact_transfer_mode(), ArtifactTransferMode::TrustedOnly);
        assert!(artifact_transfer_enabled());

        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("MESH_LLM_ARTIFACT_TRANSFER", "1") };
        assert_eq!(artifact_transfer_mode(), ArtifactTransferMode::Open);
        assert!(artifact_transfer_enabled());

        restore_env("MESH_LLM_ARTIFACT_TRANSFER", prev);
    }

    #[test]
    #[serial]
    fn artifact_transfer_default_policy_does_not_advertise_or_serve_public_mesh() {
        let prev = std::env::var_os("MESH_LLM_ARTIFACT_TRANSFER");
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("MESH_LLM_ARTIFACT_TRANSFER") };

        let unsigned = crate::crypto::OwnershipSummary::default();
        let trust_store = crate::crypto::TrustStore::default();

        assert!(!artifact_transfer_advertised(&unsigned));
        assert!(!artifact_transfer_allowed_between(
            &unsigned,
            &unsigned,
            &trust_store
        ));

        restore_env("MESH_LLM_ARTIFACT_TRANSFER", prev);
    }

    #[test]
    #[serial]
    fn artifact_transfer_trusted_policy_requires_owned_or_allowlisted_peer() {
        let prev = std::env::var_os("MESH_LLM_ARTIFACT_TRANSFER");
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("MESH_LLM_ARTIFACT_TRANSFER", "trusted") };

        let local = verified_owner("owner-a");
        let same_owner_peer = verified_owner("owner-a");
        let trusted_peer = verified_owner("owner-b");
        let untrusted_peer = verified_owner("owner-c");
        let mut trust_store = crate::crypto::TrustStore::default();
        trust_store.add_trusted_owner("owner-b".to_string(), None);

        assert!(artifact_transfer_advertised(&local));
        assert!(artifact_transfer_allowed_between(
            &local,
            &same_owner_peer,
            &trust_store
        ));
        assert!(artifact_transfer_allowed_between(
            &local,
            &trusted_peer,
            &trust_store
        ));
        assert!(!artifact_transfer_allowed_between(
            &local,
            &untrusted_peer,
            &trust_store
        ));

        restore_env("MESH_LLM_ARTIFACT_TRANSFER", prev);
    }

    #[test]
    #[serial]
    fn artifact_transfer_open_policy_is_explicit_public_mesh_opt_in() {
        let prev = std::env::var_os("MESH_LLM_ARTIFACT_TRANSFER");
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("MESH_LLM_ARTIFACT_TRANSFER", "open") };

        let unsigned = crate::crypto::OwnershipSummary::default();
        let trust_store = crate::crypto::TrustStore::default();

        assert!(artifact_transfer_advertised(&unsigned));
        assert!(artifact_transfer_allowed_between(
            &unsigned,
            &unsigned,
            &trust_store
        ));

        restore_env("MESH_LLM_ARTIFACT_TRANSFER", prev);
    }

    #[test]
    #[serial]
    fn required_stage_package_artifacts_include_stage_shared_and_projectors() {
        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());

        let artifacts = required_stage_package_artifacts(
            &package_dir,
            &package_ref,
            &manifest_sha,
            StageArtifactSelection {
                layer_start: 1,
                layer_end: 2,
                include_embeddings: false,
                include_output: true,
                include_projectors: true,
            },
        )
        .unwrap();
        let paths = artifacts
            .iter()
            .map(|artifact| artifact.relative_path.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "shared/metadata.gguf",
                "shared/output.gguf",
                "layers/layer-001.gguf",
                "projectors/mmproj.gguf",
            ]
        );

        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    fn package_v2_artifact_selection_uses_exact_admitted_tensor_ids() {
        let temp = tempfile::tempdir().unwrap();
        let (package_dir, package_ref, manifest_sha, package_id) =
            write_package_v2_manifest(temp.path());
        let admission = skippy_protocol::StageAdmissionDescriptor {
            version: skippy_protocol::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id,
            plan_id: format!("skippy-plan:v1:{}", "bb".repeat(32)),
            layer_start: 1,
            layer_end: 2,
            resident_tensor_ids: vec!["tensor-resident".to_string()],
            sidecars: Vec::new(),
            profiles: Vec::new(),
        };

        let artifacts = required_admitted_stage_package_artifacts(
            &package_dir,
            &package_ref,
            &manifest_sha,
            &admission,
        )
        .unwrap();
        let paths = artifacts
            .iter()
            .map(|artifact| artifact.relative_path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec!["artifacts/source-00000.gguf", "artifacts/source-00001.gguf"]
        );
        assert!(!paths.iter().any(|path| path.contains("unowned")));
    }

    #[test]
    #[serial]
    fn required_stage_package_bytes_include_manifest_and_selected_shared_artifacts() {
        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());
        let manifest_bytes = fs::metadata(package_dir.join(PACKAGE_MANIFEST_FILE))
            .unwrap()
            .len();

        let selected_bytes = required_stage_package_bytes(
            &package_dir,
            &package_ref,
            &manifest_sha,
            StageArtifactSelection {
                layer_start: 1,
                layer_end: 2,
                include_embeddings: false,
                include_output: true,
                include_projectors: true,
            },
        )
        .unwrap();

        assert_eq!(selected_bytes, manifest_bytes + 8 + 6 + 8 + 9);
        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    #[serial]
    fn required_stage_package_artifacts_rejects_oversize_manifest() {
        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());
        let manifest = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(package_dir.join(PACKAGE_MANIFEST_FILE))
            .unwrap();
        manifest.set_len(MAX_PACKAGE_MANIFEST_BYTES + 1).unwrap();

        assert!(
            required_stage_package_artifacts(
                &package_dir,
                &package_ref,
                &manifest_sha,
                StageArtifactSelection {
                    layer_start: 0,
                    layer_end: 1,
                    include_embeddings: true,
                    include_output: false,
                    include_projectors: false,
                },
            )
            .is_err()
        );

        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    #[serial]
    fn servable_artifact_requires_manifest_declared_path_and_matching_sha() {
        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (_package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());

        let request = skippy_protocol::proto::stage::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: vec![1; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref,
            manifest_sha256: manifest_sha,
            relative_path: "layers/layer-000.gguf".to_string(),
            offset: 0,
            expected_size: Some(8),
            expected_sha256: Some(sha256_hex(b"layer000")),
        };
        let artifact = servable_artifact_from_request(&request).unwrap();
        assert_eq!(artifact.size, 8);
        assert_eq!(artifact.sha256, sha256_hex(b"layer000"));

        let mut undeclared = request.clone();
        undeclared.relative_path = "layers/not-declared.gguf".to_string();
        assert!(servable_artifact_from_request(&undeclared).is_err());

        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    #[serial]
    fn servable_artifact_rejects_same_size_corrupt_cached_bytes() {
        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());
        fs::write(package_dir.join("layers/layer-000.gguf"), b"corrupt!").unwrap();

        let request = skippy_protocol::proto::stage::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: vec![1; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref,
            manifest_sha256: manifest_sha,
            relative_path: "layers/layer-000.gguf".to_string(),
            offset: 0,
            expected_size: Some(8),
            expected_sha256: Some(sha256_hex(b"layer000")),
        };

        let error = servable_artifact_from_request(&request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cached artifact sha256 mismatch"),
            "unexpected error: {error}"
        );

        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    fn peer_artifact_transfer_requires_explicit_non_mutable_revision() {
        for package_ref in [
            "hf://meshllm/demo-layers",
            "hf://meshllm/demo-layers:abc123",
            "hf://meshllm/demo-layers@main",
            "hf://meshllm/demo-layers@master",
            "hf://meshllm/demo-layers@latest",
        ] {
            assert!(
                package_cache_dir_for_ref(package_ref).is_err(),
                "{package_ref} must not be eligible for peer transfer"
            );
        }

        assert!(package_cache_dir_for_ref("hf://meshllm/demo-layers@abc123").is_ok());
    }

    #[test]
    #[cfg(unix)]
    #[serial]
    fn servable_artifact_rejects_symlink_escape_from_hf_repo_root() {
        use std::os::unix::fs as unix_fs;

        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, manifest_sha) = write_package_fixture(temp.path());
        fs::write(temp.path().join("outside.gguf"), b"outside!").unwrap();
        fs::remove_file(package_dir.join("layers/layer-000.gguf")).unwrap();
        unix_fs::symlink(
            temp.path().join("outside.gguf"),
            package_dir.join("layers/layer-000.gguf"),
        )
        .unwrap();

        let request = skippy_protocol::proto::stage::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: vec![1; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref,
            manifest_sha256: manifest_sha,
            relative_path: "layers/layer-000.gguf".to_string(),
            offset: 0,
            expected_size: Some(8),
            expected_sha256: Some(sha256_hex(b"outside!")),
        };
        assert!(servable_artifact_from_request(&request).is_err());

        restore_env("HF_HUB_CACHE", prev);
    }

    #[test]
    #[cfg(unix)]
    #[serial]
    fn local_artifact_install_parent_rejects_symlink_escape_from_hf_repo_root() {
        use std::os::unix::fs as unix_fs;

        let prev = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HUB_CACHE", temp.path()) };
        let (package_dir, package_ref, _manifest_sha) = write_package_fixture(temp.path());
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::remove_dir_all(package_dir.join("layers")).unwrap();
        unix_fs::symlink(&outside, package_dir.join("layers")).unwrap();

        assert!(
            ensure_local_artifact_install_parent(
                &package_ref,
                &package_dir.join("layers/layer-000.gguf")
            )
            .is_err()
        );

        restore_env("HF_HUB_CACHE", prev);
    }
}
