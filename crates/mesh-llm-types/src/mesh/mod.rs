use crate::models::{ModelCapabilities, ModelTopology};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    pub last_active: u64,
    pub request_count: u64,
}

pub const DEMAND_TTL_SECS: u64 = 86400;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Catalog,
    HuggingFace,
    LocalGguf,
    DirectUrl,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ServedModelIdentity {
    pub model_name: String,
    pub is_primary: bool,
    pub source_kind: ModelSourceKind,
    pub canonical_ref: Option<String>,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub artifact: Option<String>,
    pub local_file_name: Option<String>,
    /// Hash of a reference STRING identifying the model source (repo,
    /// revision, file) -- `None` for a bare local path, where there is no
    /// such string. NOT a hash of the served weight bytes; see
    /// [`Self::weights_digest`] for that fact.
    pub identity_hash: Option<String>,
    /// SHA-256 of the served GGUF's file BYTES, lowercase hex -- computed at
    /// load time from the file actually opened for serving. `None` when the
    /// file could not be read, or when no load-time hash has been computed
    /// for this model yet -- an absent fact, never a fabricated digest.
    /// Distinct from `identity_hash`: a name/reference-string hash and a
    /// content hash are different facts about the model, and this field
    /// never replaces the other.
    pub weights_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServedModelDescriptor {
    pub identity: ServedModelIdentity,
    #[serde(default, skip_serializing_if = "is_false")]
    pub capabilities_known: bool,
    pub capabilities: ModelCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<ModelTopology>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ServedModelMetadata>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServedModelMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_count_b: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_expert_count: Option<u32>,
}

impl ServedModelMetadata {
    pub fn is_empty(&self) -> bool {
        self.architecture.is_none()
            && self.parameter_size.is_none()
            && self.parameter_count_b.is_none()
            && self.quant.is_none()
            && self.native_context_length.is_none()
            && self.tokenizer.is_none()
            && self.layer_count.is_none()
            && self.embedding_size.is_none()
            && self.head_count.is_none()
            && self.kv_head_count.is_none()
            && self.expert_count.is_none()
            && self.active_expert_count.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelRuntimeDescriptor {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    pub ready: bool,
}

impl ModelRuntimeDescriptor {
    pub fn advertised_context_length(&self) -> Option<u32> {
        self.ready.then_some(self.context_length).flatten()
    }
}

pub fn merge_demand(
    ours: &mut HashMap<String, ModelDemand>,
    theirs: &HashMap<String, ModelDemand>,
) {
    for (model, their_demand) in theirs {
        let entry = ours.entry(model.clone()).or_default();
        entry.last_active = entry.last_active.max(their_demand.last_active);
        entry.request_count = entry.request_count.max(their_demand.request_count);
    }
}

pub fn infer_served_model_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
    primary_model_path: Option<&std::path::Path>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source
        .and_then(identity_from_model_source)
        .or_else(|| {
            primary_model_path.and_then(|path| identity_from_local_path(primary_model_name, path))
        });
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if idx == 0 || model_name == primary_model_name {
                let mut id = primary.clone().unwrap_or_default();
                id.model_name = model_name.clone();
                id.is_primary = true;
                if id.local_file_name.is_none() {
                    id.local_file_name = Some(format!("{model_name}.gguf"));
                }
                id
            } else {
                ServedModelIdentity {
                    model_name: model_name.clone(),
                    is_primary: false,
                    source_kind: ModelSourceKind::Unknown,
                    local_file_name: Some(format!("{model_name}.gguf")),
                    ..Default::default()
                }
            };
            ServedModelDescriptor {
                identity,
                capabilities_known: false,
                capabilities: ModelCapabilities::default(),
                topology: None,
                metadata: None,
            }
        })
        .collect()
}

pub fn infer_available_model_descriptors(
    _available_models: &[String],
) -> Vec<ServedModelDescriptor> {
    Vec::new()
}

pub fn infer_local_served_model_descriptor(
    _model_name: &str,
    _is_primary: bool,
) -> Option<ServedModelDescriptor> {
    None
}

fn identity_from_local_path(
    model_name: &str,
    path: &std::path::Path,
) -> Option<ServedModelIdentity> {
    let local_file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .or_else(|| Some(format!("{model_name}.gguf")));
    Some(ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        local_file_name,
        ..Default::default()
    })
}

fn identity_from_model_source(source: &str) -> Option<ServedModelIdentity> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((repo_id, revision, selector)) = parse_model_ref_source(trimmed) {
        let canonical_ref = format_model_ref(&repo_id, revision.as_deref(), selector.as_deref());
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: selector,
            local_file_name: None,
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if is_explicit_local_path(trimmed) {
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

fn parse_model_ref_source(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    if input.starts_with("http://")
        || input.starts_with("https://")
        || is_explicit_local_path(input)
    {
        return None;
    }
    let (org, tail) = input.split_once('/')?;
    if org.is_empty() || tail.is_empty() || tail.contains('/') || org.contains(':') {
        return None;
    }
    let (repo, revision, selector) = parse_repo_tail_selector_and_revision(tail)?;
    Some((format!("{org}/{repo}"), revision, selector))
}

fn parse_repo_tail_selector_and_revision(
    tail: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let at_pos = tail.find('@');
    let colon_pos = tail.find(':');
    match (at_pos, colon_pos) {
        (Some(at), Some(colon)) if at < colon => nonempty_model_ref_parts(
            &tail[..at],
            Some(&tail[at + 1..colon]),
            Some(&tail[colon + 1..]),
        ),
        (Some(at), Some(colon)) if colon < at => nonempty_model_ref_parts(
            &tail[..colon],
            Some(&tail[at + 1..]),
            Some(&tail[colon + 1..at]),
        ),
        (Some(at), None) => nonempty_model_ref_parts(&tail[..at], Some(&tail[at + 1..]), None),
        (None, Some(colon)) => {
            nonempty_model_ref_parts(&tail[..colon], None, Some(&tail[colon + 1..]))
        }
        (None, None) => nonempty_model_ref_parts(tail, None, None),
        _ => None,
    }
}

fn nonempty_model_ref_parts(
    repo: &str,
    revision: Option<&str>,
    selector: Option<&str>,
) -> Option<(String, Option<String>, Option<String>)> {
    if repo.is_empty() || revision.is_some_and(str::is_empty) || selector.is_some_and(str::is_empty)
    {
        return None;
    }
    Some((
        repo.to_string(),
        revision.map(str::to_string),
        selector.map(str::to_string),
    ))
}

fn format_model_ref(repo: &str, revision: Option<&str>, selector: Option<&str>) -> String {
    match (revision, selector) {
        (Some(revision), Some(selector)) => format!("{repo}@{revision}:{selector}"),
        (Some(revision), None) => format!("{repo}@{revision}"),
        (None, Some(selector)) => format!("{repo}:{selector}"),
        (None, None) => repo.to_string(),
    }
}

fn is_explicit_local_path(source: &str) -> bool {
    source.starts_with('/') || source.starts_with("./") || source.starts_with("../")
}

fn local_gguf_identity_from_source(source: &str) -> ServedModelIdentity {
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

fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    if is_explicit_local_path(input) {
        return None;
    }
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, rev)) => (repo, Some(rev.to_string())),
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

fn parse_hf_resolve_url_parts(url: &str) -> Option<(String, Option<String>, String)> {
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = path.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    let canonical = format!("{repo}@{revision}/{file}");
    parse_hf_ref_parts(&canonical)
}

fn format_hf_canonical_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(rev) => format!("{repo}@{rev}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}
