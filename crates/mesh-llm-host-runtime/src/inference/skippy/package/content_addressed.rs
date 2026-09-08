use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{SkippyPackageIdentity, SkippyPackageSourceFile, hex_lower, synthetic_gguf_package};

#[derive(Serialize)]
struct ContentAddressedGgufManifest<'a> {
    schema_version: u32,
    package_kind: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [ContentAddressedGgufManifestFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

#[derive(Serialize)]
struct ContentAddressedGgufManifestFile {
    ordinal: u32,
    bytes: u64,
    sha256: String,
}

/// Build a path-independent identity for a GGUF that must already exist on
/// every split-serving participant.
///
/// The returned package and manifest identities contain no model alias,
/// filename, or absolute path. The path remains node-local and is registered
/// only for resolving a later inventory or load request on this process.
pub fn synthetic_content_addressed_gguf_package(
    model_id: &str,
    model_path: &Path,
) -> Result<SkippyPackageIdentity> {
    synthetic_gguf_package(model_id, model_path)
}

pub(super) fn validate_source_set(model_path: &Path) -> Result<()> {
    anyhow::ensure!(
        model_path.is_absolute(),
        "content-addressed GGUF source path must be absolute: {}",
        model_path.display()
    );
    anyhow::ensure!(
        model_path.to_str().is_some(),
        "content-addressed GGUF source path must be valid UTF-8: {}",
        model_path.display()
    );
    let validate_file = |path: &Path| -> Result<()> {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("stat content-addressed GGUF source {}", path.display()))?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "content-addressed GGUF source must be a non-symlink file: {}",
            path.display()
        );
        anyhow::ensure!(
            path.to_str().is_some(),
            "content-addressed GGUF source path must be valid UTF-8: {}",
            path.display()
        );
        Ok(())
    };
    validate_file(model_path)?;
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!(
            "content-addressed GGUF source has no UTF-8 filename: {}",
            model_path.display()
        );
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        return Ok(());
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        model_path.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .with_context(|| format!("parse split GGUF shard total in {file_name}"))?;
    let parent = model_path
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", model_path.display()))?;
    for index in 1..=total {
        validate_file(&parent.join(format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total)))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn manifest_sha256(
    source_model_sha256: &str,
    source_model_bytes: u64,
    source_files: &[SkippyPackageSourceFile],
    architecture: &str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
) -> Result<String> {
    let files = source_files
        .iter()
        .enumerate()
        .map(|(index, file)| ContentAddressedGgufManifestFile {
            ordinal: u32::try_from(index).unwrap_or(u32::MAX),
            bytes: file.bytes,
            sha256: file.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = ContentAddressedGgufManifest {
        schema_version: 2,
        package_kind: "content-addressed-direct-gguf",
        source_model_sha256,
        source_model_bytes,
        source_files: &files,
        architecture,
        context_length,
        layer_count,
        activation_width,
        tensor_count,
    };
    let bytes =
        serde_json::to_vec(&manifest).context("serialize content-addressed GGUF manifest")?;
    Ok(hex_lower(&Sha256::digest(bytes)))
}

pub(super) fn ensure_fingerprint_unchanged(
    source_files: &[SkippyPackageSourceFile],
    expected: Option<&[super::super::local_source::VerifiedFileFingerprint]>,
) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let paths = source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        super::super::local_source::verified_path_fingerprint(&paths).as_deref() == Some(expected),
        "content-addressed GGUF source changed while its identity was being computed"
    );
    Ok(())
}

pub(super) fn aggregate_source_sha256(source_files: &[SkippyPackageSourceFile]) -> String {
    if source_files.len() == 1 {
        return source_files[0].sha256.clone();
    }
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-llm-split-gguf-v1\0");
    hasher.update((source_files.len() as u64).to_le_bytes());
    for (index, file) in source_files.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        hasher.update(file.bytes.to_le_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    hex_lower(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn push_test_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as i64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_test_metadata_gguf(path: &Path, context_length: u32) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&4i64.to_le_bytes());
        push_test_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        push_test_gguf_string(&mut bytes, "llama");
        for (key, value) in [
            ("llama.block_count", 2),
            ("llama.embedding_length", 128),
            ("llama.context_length", context_length),
        ] {
            push_test_gguf_string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn identity_ignores_alias_and_absolute_path() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_path = first_dir.path().join("first-name.gguf");
        let second_path = second_dir.path().join("other-name.gguf");
        write_test_metadata_gguf(&first_path, 4096);
        std::fs::copy(&first_path, &second_path).unwrap();

        let first = synthetic_content_addressed_gguf_package("alias-a", &first_path).unwrap();
        let second = synthetic_content_addressed_gguf_package("alias-b", &second_path).unwrap();

        assert_eq!(first.package_ref, second.package_ref);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.source_model_sha256, second.source_model_sha256);
        assert_ne!(first.source_model_path, second.source_model_path);
        assert!(
            super::super::super::local_source::is_content_addressed_gguf_ref(&first.package_ref)
        );
    }

    #[test]
    fn identity_changes_with_source_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.gguf");
        let second_path = dir.path().join("second.gguf");
        write_test_metadata_gguf(&first_path, 4096);
        write_test_metadata_gguf(&second_path, 8192);

        let first = synthetic_content_addressed_gguf_package("same-alias", &first_path).unwrap();
        let second = synthetic_content_addressed_gguf_package("same-alias", &second_path).unwrap();

        assert_ne!(first.package_ref, second.package_ref);
        assert_ne!(first.manifest_sha256, second.manifest_sha256);
        assert_ne!(first.source_model_sha256, second.source_model_sha256);
    }

    #[test]
    fn split_digest_is_path_independent_and_order_sensitive() {
        let first = vec![
            SkippyPackageSourceFile {
                path: PathBuf::from("/node-a/model-00001-of-00002.gguf"),
                bytes: 10,
                sha256: "a".repeat(64),
            },
            SkippyPackageSourceFile {
                path: PathBuf::from("/node-a/model-00002-of-00002.gguf"),
                bytes: 20,
                sha256: "b".repeat(64),
            },
        ];
        let relocated = vec![
            SkippyPackageSourceFile {
                path: PathBuf::from("/other/first.gguf"),
                ..first[0].clone()
            },
            SkippyPackageSourceFile {
                path: PathBuf::from("/other/second.gguf"),
                ..first[1].clone()
            },
        ];
        let reversed = vec![first[1].clone(), first[0].clone()];

        assert_eq!(
            aggregate_source_sha256(&first),
            aggregate_source_sha256(&relocated)
        );
        assert_ne!(
            aggregate_source_sha256(&first),
            aggregate_source_sha256(&reversed)
        );
    }

    #[test]
    fn split_identity_matches_across_paths_and_filename_prefixes() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_primary = first_dir.path().join("alpha-00001-of-00002.gguf");
        let first_secondary = first_dir.path().join("alpha-00002-of-00002.gguf");
        let second_primary = second_dir.path().join("beta-00001-of-00002.gguf");
        let second_secondary = second_dir.path().join("beta-00002-of-00002.gguf");
        write_test_metadata_gguf(&first_primary, 4096);
        write_test_metadata_gguf(&first_secondary, 4096);
        std::fs::copy(&first_primary, &second_primary).unwrap();
        std::fs::copy(&first_secondary, &second_secondary).unwrap();

        let first =
            synthetic_content_addressed_gguf_package("logical/model", &first_primary).unwrap();
        let second =
            synthetic_content_addressed_gguf_package("logical/model", &second_primary).unwrap();

        assert_eq!(first.source_files.len(), 2);
        assert_eq!(second.source_files.len(), 2);
        assert_eq!(first.package_ref, second.package_ref);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.source_model_sha256, second.source_model_sha256);
        assert_ne!(first.source_model_path, second.source_model_path);
    }

    #[test]
    fn identity_requires_an_absolute_path() {
        let error = synthetic_content_addressed_gguf_package(
            "logical/model",
            Path::new("relative-model.gguf"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("must be absolute"));
    }

    #[cfg(unix)]
    #[test]
    fn split_rejects_symlinked_secondary_shard() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("model-00001-of-00002.gguf");
        let secondary = dir.path().join("model-00002-of-00002.gguf");
        let target = dir.path().join("secondary-target.gguf");
        write_test_metadata_gguf(&primary, 4096);
        write_test_metadata_gguf(&target, 4096);
        symlink(&target, &secondary).unwrap();

        let error = synthetic_content_addressed_gguf_package("logical/model", &primary)
            .unwrap_err()
            .to_string();

        assert!(error.contains("non-symlink file"));
        assert!(error.contains("00002-of-00002"));
    }

    // APFS rejects invalid UTF-8 path bytes at creation time; Linux permits
    // them and therefore exercises the canonical-parent edge directly.
    #[cfg(target_os = "linux")]
    #[test]
    fn identity_rejects_non_utf8_canonical_parent() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target_dir = dir.path().join(OsString::from_vec(vec![b'm', 0xff]));
        std::fs::create_dir(&target_dir).unwrap();
        let target_model = target_dir.join("model.gguf");
        write_test_metadata_gguf(&target_model, 4096);
        let utf8_parent = dir.path().join("models");
        symlink(&target_dir, &utf8_parent).unwrap();

        let error = synthetic_content_addressed_gguf_package(
            "logical/model",
            &utf8_parent.join("model.gguf"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("canonical content-addressed GGUF path"));
        assert!(error.contains("valid UTF-8"));
    }
}
