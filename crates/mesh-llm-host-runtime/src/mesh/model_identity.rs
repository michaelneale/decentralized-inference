use super::*;

pub(crate) fn infer_remote_served_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source.and_then(identity_from_model_source);
    let primary_index = serving_models
        .iter()
        .position(|model_name| model_name == primary_model_name);
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if Some(idx) == primary_index {
                let mut identity = primary
                    .clone()
                    .unwrap_or_else(|| unknown_identity(model_name));
                identity.model_name = model_name.clone();
                identity.is_primary = true;
                if identity.local_file_name.is_none() {
                    identity.local_file_name = Some(format!("{model_name}.gguf"));
                }
                identity
            } else {
                unknown_identity(model_name)
            };
            ServedModelDescriptor {
                identity,
                capabilities_known: false,
                capabilities: crate::models::ModelCapabilities::default(),
                topology: None,
                metadata: None,
            }
        })
        .collect()
}

pub(crate) fn unknown_identity(model_name: &str) -> ServedModelIdentity {
    ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::Unknown,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: Some(format!("{model_name}.gguf")),
        identity_hash: None,
        weights_digest: None,
    }
}

pub(crate) fn identity_from_model_source(source: &str) -> Option<ServedModelIdentity> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(model_ref) = model_ref::ModelRef::parse(trimmed) {
        let display_id = model_ref.display_id();
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(display_id.clone()),
            repository: Some(model_ref.repo),
            revision: model_ref.revision,
            artifact: model_ref.selector,
            local_file_name: None,
            identity_hash: Some(identity_hash_for(&display_id)),
            weights_digest: None,
        });
    }

    if trimmed.starts_with('/') || trimmed.starts_with("./") || trimmed.starts_with("../") {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    if let Some((repo_id, revision, file)) = parse_hf_resolve_url_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if let Some((repo_id, revision, file)) = parse_hf_ref_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::DirectUrl,
            canonical_ref: Some(trimmed.to_string()),
            repository: None,
            revision: None,
            artifact: None,
            local_file_name: trimmed.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(trimmed)),
            weights_digest: None,
        });
    }

    if trimmed.ends_with(".gguf")
        || (trimmed.contains('/') && !trimmed.ends_with('/') && trimmed.split('/').count() != 2)
    {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    Some(ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::Catalog,
        canonical_ref: Some(trimmed.to_string()),
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: None,
        identity_hash: Some(identity_hash_for(&format!("catalog:{trimmed}"))),
        weights_digest: None,
    })
}

pub(crate) fn local_gguf_identity_from_source(source: &str) -> ServedModelIdentity {
    let local_file_name = std::path::Path::new(source)
        .file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string);
    ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name,
        identity_hash: None,
        weights_digest: None,
    }
}

pub(crate) fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    if input.starts_with('/') || input.starts_with("./") || input.starts_with("../") {
        return None;
    }
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    if parts[0].is_empty() || repo_tail.is_empty() || parts[2].is_empty() {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

pub(crate) fn parse_hf_resolve_url_parts(url: &str) -> Option<(String, Option<String>, String)> {
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = path.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    if repo.is_empty() || revision.is_empty() || file.is_empty() {
        return None;
    }
    Some((
        repo.to_string(),
        Some(revision.to_string()),
        file.to_string(),
    ))
}

pub(crate) fn format_hf_canonical_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

pub(crate) fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_supports_top_level_hugging_face_repository() {
        // Given: a resolve URL for a top-level Hugging Face repository.
        let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";

        // When: the URL identity is parsed.
        let parts = parse_hf_resolve_url_parts(url);

        // Then: the repository, revision, and artifact are preserved.
        assert_eq!(
            parts,
            Some((
                "gpt2".to_string(),
                Some("main".to_string()),
                "model.safetensors".to_string(),
            ))
        );
    }
}
