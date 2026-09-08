use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{PackagePart, PackageStageRequest, file_fingerprint};

const RECORD_SCHEMA_VERSION: u32 = 1;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub(super) struct MaterializedCacheIdentity {
    schema_version: u32,
    request: MaterializedRequestIdentity,
    manifest_sha256: String,
    selected_parts: Vec<MaterializedPartIdentity>,
}

impl MaterializedCacheIdentity {
    pub(super) fn new(
        request: &PackageStageRequest,
        manifest_sha256: &str,
        selected_parts: &[PackagePart],
    ) -> Self {
        Self {
            schema_version: RECORD_SCHEMA_VERSION,
            request: MaterializedRequestIdentity {
                model_id: request.model_id.clone(),
                topology_id: request.topology_id.clone(),
                stage_id: request.stage_id.clone(),
                layer_start: request.layer_start,
                layer_end: request.layer_end,
                include_embeddings: request.include_embeddings,
                include_output: request.include_output,
            },
            manifest_sha256: manifest_sha256.to_string(),
            selected_parts: selected_parts
                .iter()
                .map(MaterializedPartIdentity::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct MaterializedRequestIdentity {
    model_id: String,
    topology_id: String,
    stage_id: String,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct MaterializedPartIdentity {
    role: String,
    layer_index: Option<u32>,
    path: String,
    sha256: String,
    artifact_bytes: u64,
}

impl From<&PackagePart> for MaterializedPartIdentity {
    fn from(part: &PackagePart) -> Self {
        Self {
            role: part.role.clone(),
            layer_index: part.layer_index,
            path: part.path.to_string_lossy().to_string(),
            sha256: part.sha256.clone(),
            artifact_bytes: part.artifact_bytes,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct MaterializedCacheRecord {
    schema_version: u32,
    identity: MaterializedCacheIdentity,
    output_len: u64,
    output_modified_unix_nanos: Option<u128>,
}

pub(super) struct MaterializedOutputLock {
    file: fs::File,
}

impl Drop for MaterializedOutputLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub(super) fn lock_output(output: &Path) -> Result<MaterializedOutputLock> {
    let lock_path = sibling_path(output, "lock");
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialized cache lock dir {}", parent.display()))?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open materialized cache lock {}", lock_path.display()))?;

    file.lock()
        .with_context(|| format!("lock materialized cache {}", lock_path.display()))?;

    Ok(MaterializedOutputLock { file })
}

pub(super) fn record_matches_output(
    output: &Path,
    identity: &MaterializedCacheIdentity,
) -> Result<bool> {
    let metadata = match fs::metadata(output) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("read {}", output.display()));
        }
    };
    if !metadata.is_file() || metadata.len() == 0 {
        return Ok(false);
    }

    let record_path = record_path(output);
    let bytes = match fs::read(&record_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("read {}", record_path.display()));
        }
    };
    let Ok(record) = serde_json::from_slice::<MaterializedCacheRecord>(&bytes) else {
        return Ok(false);
    };

    Ok(record.schema_version == RECORD_SCHEMA_VERSION
        && record.identity == *identity
        && record.output_len == metadata.len()
        && record.output_modified_unix_nanos == file_fingerprint(&metadata))
}

pub(super) fn write_record(output: &Path, identity: &MaterializedCacheIdentity) -> Result<()> {
    let metadata =
        fs::metadata(output).with_context(|| format!("read materialized {}", output.display()))?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() > 0,
        "materialized output is empty or not a file: {}",
        output.display()
    );
    let record = MaterializedCacheRecord {
        schema_version: RECORD_SCHEMA_VERSION,
        identity: identity.clone(),
        output_len: metadata.len(),
        output_modified_unix_nanos: file_fingerprint(&metadata),
    };
    let path = record_path(output);
    let tmp = temporary_sibling_path(&path, "json");
    if let Some(parent) = tmp.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialized cache dir {}", parent.display()))?;
    }
    fs::write(&tmp, serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("write materialized cache record {}", tmp.display()))?;
    publish_output(&tmp, &path)
}

pub(super) fn temporary_output_path(output: &Path, manifest_sha256: &str) -> PathBuf {
    temporary_sibling_path(output, manifest_sha256.get(..12).unwrap_or("manifest"))
}

pub(super) fn cleanup_temporary_output(path: &Path) {
    let _ = fs::remove_file(path);
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }
}

pub(super) fn publish_output(tmp: &Path, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialized cache dir {}", parent.display()))?;
    }
    match fs::rename(tmp, output) {
        Ok(()) => Ok(()),
        Err(error) => publish_after_rename_error(error, tmp, output),
    }
}

#[cfg(windows)]
fn publish_after_rename_error(error: std::io::Error, tmp: &Path, output: &Path) -> Result<()> {
    if !output.exists() {
        return Err(error).with_context(|| {
            format!(
                "publish materialized output {} -> {}",
                tmp.display(),
                output.display()
            )
        });
    }
    fs::remove_file(output)
        .with_context(|| format!("remove stale materialized output {}", output.display()))?;
    fs::rename(tmp, output).with_context(|| {
        format!(
            "publish materialized output {} -> {}",
            tmp.display(),
            output.display()
        )
    })
}

#[cfg(not(windows))]
fn publish_after_rename_error(error: std::io::Error, tmp: &Path, output: &Path) -> Result<()> {
    Err(error).with_context(|| {
        format!(
            "publish materialized output {} -> {}",
            tmp.display(),
            output.display()
        )
    })
}

pub(super) fn record_path(output: &Path) -> PathBuf {
    sibling_path(output, "cache.json")
}

fn temporary_sibling_path(output: &Path, suffix: &str) -> PathBuf {
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "materialized".into());
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    parent.join(".staging").join(format!(
        "{}.tmp-{}-{}-{}",
        name,
        std::process::id(),
        counter,
        sanitize_suffix(suffix)
    ))
}

fn sibling_path(output: &Path, suffix: &str) -> PathBuf {
    let name = output
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "materialized".into());
    output.with_file_name(format!("{}.{}", name, suffix))
}

fn sanitize_suffix(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn output_lock_excludes_other_processes_until_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model.gguf");
        let guard = lock_output(&output).unwrap();
        assert_child_lock_state(&output, "locked");
        drop(guard);
        assert_child_lock_state(&output, "unlocked");
        let guard = lock_output(&output).unwrap();
        assert_child_lock_state(&output, "locked");
        drop(guard);
    }

    fn assert_child_lock_state(output: &Path, expected: &str) {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "package::materialized_cache::tests::probe_output_lock_in_child",
                "--ignored",
                "--nocapture",
            ])
            .env("SKIPPY_TEST_OUTPUT_LOCK_PATH", output)
            .env("SKIPPY_TEST_OUTPUT_LOCK_STATE", expected)
            .output()
            .unwrap();
        assert!(
            result.status.success()
                && String::from_utf8_lossy(&result.stdout).contains("1 passed;"),
            "lock probe failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    #[test]
    #[ignore = "subprocess helper for output_lock_excludes_other_processes_until_dropped"]
    fn probe_output_lock_in_child() {
        let output = PathBuf::from(std::env::var_os("SKIPPY_TEST_OUTPUT_LOCK_PATH").unwrap());
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(sibling_path(&output, "lock"))
            .unwrap();
        match std::env::var("SKIPPY_TEST_OUTPUT_LOCK_STATE")
            .unwrap()
            .as_str()
        {
            "locked" => assert!(matches!(file.try_lock(), Err(fs::TryLockError::WouldBlock))),
            "unlocked" => file.try_lock().unwrap(),
            state => panic!("unexpected lock state: {state}"),
        }
    }
}
