use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_runtime::package;

fn materialized_stage_cache_dir() -> PathBuf {
    super::materialized_stage_cache_dir()
}

#[derive(Debug, Serialize, Deserialize)]
struct PinFile {
    artifact_path: PathBuf,
    package_ref: String,
    topology_id: String,
    run_id: String,
    stage_id: String,
}

pub fn prune_unpinned_materialized_stages() -> Result<usize> {
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(0);
    }
    let pins = active_pin_artifacts(&root)?;
    let mut removed = 0usize;
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        if pins.iter().any(|pin| pin == &path) {
            continue;
        }
        if remove_materialized_stage_artifact(&path)? {
            removed += 1;
        }
    }
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !index.artifact_path.exists() && !pins.iter().any(|pin| pin == &index.artifact_path) {
            let _ = fs::remove_file(path);
        }
    }
    Ok(removed)
}

pub fn remove_materialized_stages_for_sources(sources: &[PathBuf]) -> Result<usize> {
    let candidates = materialized_stage_removal_candidates(sources)?;
    let mut removed = 0usize;
    for candidate in candidates {
        if remove_materialized_stage_artifact(&candidate.artifact_path)? {
            removed += 1;
        }
        let _ = fs::remove_file(candidate.source_index_path);
    }
    Ok(removed)
}

pub fn materialized_stages_for_sources(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    Ok(materialized_stage_removal_candidates(sources)?
        .into_iter()
        .filter(|candidate| candidate.artifact_path.exists())
        .map(|candidate| candidate.artifact_path)
        .collect())
}

fn materialized_stage_removal_candidates(
    sources: &[PathBuf],
) -> Result<Vec<MaterializedStageRemovalCandidate>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let source_strings = sources
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let pins = active_pin_artifacts(&root)?;
    let mut candidates = Vec::new();
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !source_strings
            .iter()
            .any(|source| source == &index.source_model_path)
        {
            continue;
        }
        if pins.iter().any(|pin| pin == &index.artifact_path) {
            continue;
        }
        candidates.push(MaterializedStageRemovalCandidate {
            artifact_path: index.artifact_path,
            source_index_path: path,
        });
    }
    candidates.sort_by(|left, right| left.artifact_path.cmp(&right.artifact_path));
    Ok(candidates)
}

#[derive(Debug)]
struct MaterializedStageRemovalCandidate {
    artifact_path: PathBuf,
    source_index_path: PathBuf,
}

fn remove_materialized_stage_artifact(path: &Path) -> Result<bool> {
    let removed = match fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error).with_context(|| format!("remove {}", path.display())),
    };
    let record_path = package::materialized_layer_package_cache_record_path(path);
    match fs::remove_file(&record_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("remove {}", record_path.display()));
        }
    }
    Ok(removed)
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceIndex {
    artifact_path: PathBuf,
    source_model_path: String,
}

fn active_pin_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let pin_dir = root.join("pins");
    if !pin_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(&pin_dir).with_context(|| format!("read {}", pin_dir.display()))? {
        let path = entry?.path();
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(pin) = serde_json::from_slice::<PinFile>(&bytes) else {
            continue;
        };
        artifacts.push(pin.artifact_path);
    }
    Ok(artifacts)
}

fn cache_key(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(24);
    for byte in &digest[..12] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    use serial_test::serial;

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

    struct EnvRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            restore_env(self.key, self.previous.take());
        }
    }

    #[test]
    #[serial]
    fn materialized_stage_preview_matches_source_removal_candidates() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        let _xdg_restore = EnvRestore {
            key: "XDG_CACHE_HOME",
            previous: prev_xdg,
        };

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path()) };

        let root = materialized_stage_cache_dir();
        fs::create_dir_all(&root).unwrap();
        let source = temp
            .path()
            .join("source-package")
            .join("model-package.json");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, b"{}").unwrap();
        let fixture_id = cache_key(&temp.path().to_string_lossy());
        let artifact = root.join(format!("stage-{fixture_id}.gguf"));
        fs::write(&artifact, b"stage").unwrap();
        let cache_record_path = package::materialized_layer_package_cache_record_path(&artifact);
        fs::write(&cache_record_path, b"{}").unwrap();
        let index = SourceIndex {
            artifact_path: artifact.clone(),
            source_model_path: source.to_string_lossy().to_string(),
        };
        let index_path = root.join(format!("source-{fixture_id}.json"));
        fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
        let unreadable_index_path = root.join(format!("source-unreadable-{fixture_id}.json"));
        fs::create_dir(&unreadable_index_path).unwrap();

        let preview = materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(preview, vec![artifact.clone()]);

        let removed =
            remove_materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(removed, 1);
        assert!(!artifact.exists());
        assert!(!cache_record_path.exists());
        assert!(!index_path.exists());
        fs::remove_dir(unreadable_index_path).unwrap();
    }
}
