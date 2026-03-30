use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub mesh_name: String,
    pub display_name: String,
    pub source_repo: Option<String>,
    pub source_revision: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProvenanceSidecar {
    version: u32,
    identity: ProvenanceSidecarIdentity,
    source: ProvenanceSidecarSource,
}

#[derive(Debug, Deserialize)]
struct ProvenanceSidecarIdentity {
    canonical_id: String,
    display_name: String,
}

#[derive(Debug, Deserialize)]
struct ProvenanceSidecarSource {
    repo: Option<String>,
    revision: Option<String>,
}

pub fn resolved_model_name(path: &Path) -> String {
    resolve_model_identity(path).mesh_name
}

pub fn resolve_model_identity(path: &Path) -> ModelIdentity {
    if let Some(identity) = resolve_sidecar_identity(path) {
        return identity;
    }

    if path.is_dir() {
        return resolve_directory_identity(path);
    }

    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let mesh_name = crate::router::strip_split_suffix_owned(&stem);
    ModelIdentity {
        display_name: mesh_name.clone(),
        mesh_name,
        source_repo: None,
        source_revision: None,
    }
}

fn resolve_sidecar_identity(path: &Path) -> Option<ModelIdentity> {
    let sidecar_path = model_sidecar_path(path);
    let text = std::fs::read_to_string(sidecar_path).ok()?;
    let sidecar: ProvenanceSidecar = serde_json::from_str(&text).ok()?;
    if sidecar.version != 1 {
        return None;
    }
    let mesh_name = match (&sidecar.source.repo, &sidecar.source.revision) {
        (Some(repo), Some(revision)) => format!("{repo}@{revision}"),
        (Some(repo), None) => repo.clone(),
        (None, _) if !sidecar.identity.canonical_id.is_empty() => sidecar.identity.canonical_id,
        (None, _) => sidecar.identity.display_name.clone(),
    };
    Some(ModelIdentity {
        mesh_name,
        display_name: sidecar.identity.display_name,
        source_repo: sidecar.source.repo,
        source_revision: sidecar.source.revision,
    })
}

fn model_sidecar_path(path: &Path) -> std::path::PathBuf {
    let filename = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    path.with_file_name(format!("{filename}.manifest.json"))
}

fn resolve_directory_identity(path: &Path) -> ModelIdentity {
    let path_identity = model_identity_from_hf_cache_path(path);
    let source_repo = path_identity
        .as_ref()
        .map(|(repo, _)| repo.clone())
        .or_else(|| model_source_repo_from_config(path));
    let source_revision = path_identity.and_then(|(_, revision)| revision);
    let display_name = source_repo
        .as_deref()
        .and_then(display_name_from_repo)
        .unwrap_or_else(|| {
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        });
    let mesh_name = match (&source_repo, &source_revision) {
        (Some(repo), Some(revision)) => format!("{repo}@{revision}"),
        (Some(repo), None) => repo.clone(),
        (None, _) => display_name.clone(),
    };

    ModelIdentity {
        mesh_name,
        display_name,
        source_repo,
        source_revision,
    }
}

fn model_source_repo_from_config(path: &Path) -> Option<String> {
    let config_path = path.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let raw_name = config.get("_name_or_path")?.as_str()?.trim();
    normalize_hf_repo(raw_name)
}

fn normalize_hf_repo(raw: &str) -> Option<String> {
    let value = raw.trim().trim_end_matches('/');
    if value.is_empty() || value == "." {
        return None;
    }
    if value.starts_with('/') || value.starts_with('.') {
        return None;
    }

    let parts: Vec<_> = value.split('/').filter(|part| !part.is_empty()).collect();
    if parts.len() == 2 {
        return Some(format!("{}/{}", parts[0], parts[1]));
    }

    None
}

fn display_name_from_repo(repo: &str) -> Option<String> {
    repo.rsplit('/')
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

fn model_identity_from_hf_cache_path(path: &Path) -> Option<(String, Option<String>)> {
    let mut current = Some(path);
    while let Some(dir) = current {
        let name = dir.file_name()?.to_str()?;
        if name == "snapshots" {
            let revision = path
                .strip_prefix(dir)
                .ok()?
                .components()
                .next()
                .and_then(|c| c.as_os_str().to_str())
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            let model_dir = dir.parent()?;
            let model_dir_name = model_dir.file_name()?.to_str()?;
            if let Some(repo) = model_dir_name.strip_prefix("models--") {
                return Some((repo.replace("--", "/"), revision));
            }
        }
        current = dir.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{resolve_model_identity, resolved_model_name};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mesh-llm-{prefix}-{unique}"))
    }

    #[test]
    fn resolved_model_name_strips_split_suffix_for_gguf() {
        let path = Path::new("/tmp/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf");
        assert_eq!(resolved_model_name(path), "MiniMax-M2.5-Q4_K_M");
    }

    #[test]
    fn resolved_model_name_uses_full_repo_name_for_hf_dirs() {
        let dir = temp_test_dir("model-name-config");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"_name_or_path":"Qwen/Qwen3-0.6B"}"#,
        )
        .unwrap();

        assert_eq!(resolved_model_name(&dir), "Qwen/Qwen3-0.6B");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolved_model_name_falls_back_to_directory_name_for_hf_dirs() {
        let root = temp_test_dir("model-name-dir");
        let dir = root.join("Qwen3-0.6B-bf16");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();

        assert_eq!(resolved_model_name(&dir), "Qwen3-0.6B-bf16");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_model_identity_uses_snapshot_revision_when_path_matches_hf_cache() {
        let root = temp_test_dir("hf-snapshot");
        let dir = root
            .join("models--Qwen--Qwen3-0.6B")
            .join("snapshots")
            .join("0123456789abcdef")
            .join("export");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"_name_or_path":"Qwen/Qwen3-0.6B"}"#,
        )
        .unwrap();

        let identity = resolve_model_identity(&dir);
        assert_eq!(identity.mesh_name, "Qwen/Qwen3-0.6B@0123456789abcdef");
        assert_eq!(identity.display_name, "Qwen3-0.6B");
        assert_eq!(identity.source_repo.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(
            identity.source_revision.as_deref(),
            Some("0123456789abcdef")
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_model_identity_prefers_sidecar_provenance_for_files() {
        let dir = temp_test_dir("model-sidecar-file");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("Qwen3-8B-Q4_K_M.gguf");
        std::fs::write(&model, b"gguf").unwrap();
        std::fs::write(
            dir.join("Qwen3-8B-Q4_K_M.gguf.manifest.json"),
            br#"{
                "version": 1,
                "identity": {
                    "canonical_id": "huggingface:Qwen/Qwen3-8B-GGUF@abc123/Qwen3-8B-Q4_K_M.gguf",
                    "display_name": "Qwen3-8B-Q4_K_M.gguf"
                },
                "source": {
                    "repo": "Qwen/Qwen3-8B-GGUF",
                    "revision": "abc123"
                }
            }"#,
        )
        .unwrap();

        let identity = resolve_model_identity(&model);
        assert_eq!(identity.mesh_name, "Qwen/Qwen3-8B-GGUF@abc123");
        assert_eq!(identity.display_name, "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(identity.source_repo.as_deref(), Some("Qwen/Qwen3-8B-GGUF"));
        assert_eq!(identity.source_revision.as_deref(), Some("abc123"));

        let _ = std::fs::remove_dir_all(dir);
    }
}
