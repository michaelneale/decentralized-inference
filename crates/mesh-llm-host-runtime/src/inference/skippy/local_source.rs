use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use anyhow::{Context, Result};

use super::{
    SkippyPackageIdentity, StageLoadRequest, package::synthetic_content_addressed_gguf_package,
};

pub(super) const CONTENT_ADDRESSED_GGUF_PREFIX: &str = "local-gguf://sha256/";

static CONTENT_ADDRESSED_SOURCES: OnceLock<Mutex<HashMap<String, BTreeSet<PathBuf>>>> =
    OnceLock::new();
static VERIFIED_CONTENT_IDENTITIES: OnceLock<
    Mutex<HashMap<String, HashMap<PathBuf, VerifiedContentIdentity>>>,
> = OnceLock::new();
static LOCAL_SOURCE_POLICIES: OnceLock<Mutex<HashMap<String, HashMap<String, bool>>>> =
    OnceLock::new();

#[derive(Clone)]
struct VerifiedContentIdentity {
    identity: SkippyPackageIdentity,
    fingerprint: Vec<VerifiedFileFingerprint>,
}

#[derive(Clone, Eq, PartialEq)]
pub(super) struct VerifiedFileFingerprint {
    path: PathBuf,
    bytes: u64,
    mtime_nanos: u128,
    ctime_nanos: i128,
    device: u64,
    inode: u64,
}

fn source_registry() -> &'static Mutex<HashMap<String, BTreeSet<PathBuf>>> {
    CONTENT_ADDRESSED_SOURCES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn verified_identity_registry()
-> &'static Mutex<HashMap<String, HashMap<PathBuf, VerifiedContentIdentity>>> {
    VERIFIED_CONTENT_IDENTITIES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn local_source_policy_registry() -> &'static Mutex<HashMap<String, HashMap<String, bool>>> {
    LOCAL_SOURCE_POLICIES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the current local source policy for a logical model profile.
///
/// Each exact profile remains independent. Only local lifecycle code calls
/// this function, so replacing a prior value lets a failed start or config
/// reload move from strict back to fallback without an immortal process-wide
/// policy bit. Profile-unaware inbound requests still fail closed whenever any
/// currently recorded profile for the model is strict.
pub(crate) fn register_local_source_policy(
    model_id: &str,
    runtime_profile: &str,
    local_source_required: bool,
) {
    if model_id.is_empty() {
        return;
    }
    if let Ok(mut models) = local_source_policy_registry().lock() {
        let profiles = models.entry(model_id.to_string()).or_default();
        profiles.insert(runtime_profile.to_string(), local_source_required);
    }
}

/// Forget a policy profile after its last local runtime has stopped.
///
/// `source_policy` contributes to the derived runtime profile, so retaining a
/// stopped strict profile would make later profile-unaware fallback requests
/// fail closed forever. Callers must keep the entry registered while another
/// runtime with the same model/profile pair is still active.
pub(crate) fn unregister_local_source_policy(model_id: &str, runtime_profile: &str) {
    let Ok(mut models) = local_source_policy_registry().lock() else {
        return;
    };
    let remove_model = if let Some(profiles) = models.get_mut(model_id) {
        profiles.remove(runtime_profile);
        profiles.is_empty()
    } else {
        false
    };
    if remove_model {
        models.remove(model_id);
    }
}

pub(crate) fn local_source_required_for_model(
    model_id: &str,
    runtime_profile: Option<&str>,
) -> bool {
    let Ok(models) = local_source_policy_registry().lock() else {
        return true;
    };
    let Some(profiles) = models.get(model_id) else {
        return false;
    };
    if let Some(runtime_profile) = runtime_profile
        && let Some(required) = profiles.get(runtime_profile)
    {
        return *required;
    }
    // A legacy or unknown profile cannot safely select between local policies.
    // If any known profile is strict, fail closed until the sender provides an
    // exact profile that has been registered as fallback.
    profiles.values().any(|required| *required)
}

pub(crate) fn effective_local_source_required(
    model_id: &str,
    runtime_profile: Option<&str>,
    requested: bool,
) -> bool {
    requested || local_source_required_for_model(model_id, runtime_profile)
}

pub(super) fn content_addressed_package_ref(sha256: &str) -> Result<String> {
    anyhow::ensure!(
        is_sha256(sha256),
        "content-addressed GGUF identity must be 64 lowercase hex characters"
    );
    Ok(format!("{CONTENT_ADDRESSED_GGUF_PREFIX}{sha256}"))
}

pub(crate) fn is_content_addressed_gguf_ref(package_ref: &str) -> bool {
    package_ref
        .strip_prefix(CONTENT_ADDRESSED_GGUF_PREFIX)
        .is_some_and(is_sha256)
}

pub(super) fn register_content_addressed_identity(
    identity: &SkippyPackageIdentity,
    fingerprint: Option<Vec<VerifiedFileFingerprint>>,
) {
    if !is_content_addressed_gguf_ref(&identity.package_ref) {
        return;
    }
    if let Ok(mut sources) = source_registry().lock() {
        sources
            .entry(identity.package_ref.clone())
            .or_default()
            .insert(identity.source_model_path.clone());
    }
    if let Ok(mut identities) = verified_identity_registry().lock() {
        let entries = identities.entry(identity.package_ref.clone()).or_default();
        match fingerprint {
            Some(fingerprint) => {
                entries.insert(
                    identity.source_model_path.clone(),
                    VerifiedContentIdentity {
                        identity: identity.clone(),
                        fingerprint,
                    },
                );
            }
            None => {
                // A platform or transient filesystem state that cannot produce
                // a complete fingerprint must invalidate any older cache entry.
                entries.remove(&identity.source_model_path);
            }
        }
    }
}

pub(crate) fn into_content_addressed_identity(
    mut identity: SkippyPackageIdentity,
) -> Result<SkippyPackageIdentity> {
    identity.package_ref = content_addressed_package_ref(&identity.source_model_sha256)?;
    let fingerprint = verified_file_fingerprint(&identity);
    anyhow::ensure!(
        fingerprint.is_some(),
        "content-addressed package source contains an unsupported file or symlink"
    );
    register_content_addressed_identity(&identity, fingerprint);
    Ok(identity)
}

#[cfg(test)]
pub(super) fn register_content_addressed_source(package_ref: &str, path: &Path) {
    if !is_content_addressed_gguf_ref(package_ref) {
        return;
    }
    if let Ok(mut sources) = source_registry().lock() {
        sources
            .entry(package_ref.to_string())
            .or_default()
            .insert(path.to_path_buf());
    }
}

pub(super) fn verified_path_fingerprint(paths: &[PathBuf]) -> Option<Vec<VerifiedFileFingerprint>> {
    paths
        .iter()
        .map(|path| {
            let metadata = std::fs::symlink_metadata(path).ok()?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return None;
            }
            let (device, inode) = file_identity(&metadata)?;
            Some(VerifiedFileFingerprint {
                path: path.clone(),
                bytes: metadata.len(),
                mtime_nanos: super::hash_cache::file_mtime_nanos(&metadata)?,
                // Without an inode change timestamp, a same-size rewrite with
                // restored mtime cannot be distinguished. Rehash instead.
                ctime_nanos: super::hash_cache::file_ctime_nanos(&metadata)?,
                device,
                inode,
            })
        })
        .collect()
}

#[cfg(unix)]
fn file_identity(metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;

    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(_metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    None
}

fn verified_file_fingerprint(
    identity: &SkippyPackageIdentity,
) -> Option<Vec<VerifiedFileFingerprint>> {
    let paths = identity
        .source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    verified_path_fingerprint(&paths)
}

fn cached_verified_identity(package_ref: &str, path: &Path) -> Option<SkippyPackageIdentity> {
    let cached = verified_identity_registry()
        .lock()
        .ok()?
        .get(package_ref)?
        .get(path)?
        .clone();
    (verified_file_fingerprint(&cached.identity)? == cached.fingerprint).then_some(cached.identity)
}

pub(super) fn registered_content_addressed_source(package_ref: &str) -> Option<PathBuf> {
    registered_content_addressed_sources(package_ref)
        .into_iter()
        .next()
}

fn registered_content_addressed_sources(package_ref: &str) -> Vec<PathBuf> {
    if !is_content_addressed_gguf_ref(package_ref) {
        return Vec::new();
    }
    source_registry()
        .lock()
        .ok()
        .and_then(|sources| sources.get(package_ref).cloned())
        .unwrap_or_default()
        .into_iter()
        .filter(|path| path.is_file())
        .collect()
}

/// Resolve a content-addressed source from this process's registry and verify
/// its path-free synthetic manifest at the point of use. The first strict
/// verification hashes every byte; later checks may reuse only the identity
/// whose hash-bound strong file fingerprint is still unchanged.
///
/// The registry is only a locator. It is never accepted as proof that a path
/// still contains the content observed during startup or inventory.
pub(crate) fn verify_registered_content_source(
    model_id: &str,
    package_ref: &str,
    expected_manifest_sha256: &str,
    expected_source_sha256: &str,
) -> Result<SkippyPackageIdentity> {
    anyhow::ensure!(
        is_content_addressed_gguf_ref(package_ref),
        "unsupported content-addressed GGUF reference: {package_ref}"
    );
    anyhow::ensure!(
        is_sha256(expected_source_sha256),
        "expected content-addressed GGUF SHA-256 is invalid"
    );
    let candidates = registered_content_addressed_sources(package_ref);
    anyhow::ensure!(
        !candidates.is_empty(),
        "local GGUF content {package_ref} is not registered"
    );
    let mut failure_count = 0_usize;
    for path in candidates {
        let identity_result = cached_verified_identity(package_ref, &path)
            .map(Ok)
            .unwrap_or_else(|| {
                synthetic_content_addressed_gguf_package(model_id, &path)
                    .with_context(|| format!("verify local GGUF content at {}", path.display()))
            });
        let identity = match identity_result {
            Ok(identity) => identity,
            Err(error) => {
                failure_count += 1;
                tracing::debug!(
                    path = %path.display(),
                    package_ref,
                    error = %error,
                    "registered local GGUF failed content verification"
                );
                continue;
            }
        };
        if identity.package_ref == package_ref
            && identity.source_model_sha256 == expected_source_sha256
            && identity.manifest_sha256 == expected_manifest_sha256
        {
            return Ok(identity);
        }
        failure_count += 1;
        tracing::debug!(
            path = %path.display(),
            package_ref,
            "registered local GGUF content identity mismatched"
        );
    }
    anyhow::bail!(
        "no registered local GGUF matches {package_ref} ({failure_count} candidate(s) failed verification)"
    )
}

/// Apply the effective local source policy and resolve a verified worker-local
/// path without consulting catalogs, Hugging Face, or peer artifact transfer.
pub(crate) fn apply_verified_local_source(load: &mut StageLoadRequest) -> Result<bool> {
    let local_source_required = effective_local_source_required(
        &load.model_id,
        load.runtime_profile.as_deref(),
        load.local_source_required || is_content_addressed_gguf_ref(&load.package_ref),
    );
    if !local_source_required {
        return Ok(false);
    }
    load.local_source_required = true;
    anyhow::ensure!(
        load.load_mode == skippy_protocol::LoadMode::RuntimeSlice
            && is_content_addressed_gguf_ref(&load.package_ref),
        "local-required stage source must be a content-addressed RuntimeSlice GGUF"
    );
    let expected_source_sha256 = load
        .source_model_sha256
        .as_deref()
        .context("local-required stage source is missing expected SHA-256")?;
    let identity = verify_registered_content_source(
        &load.model_id,
        &load.package_ref,
        &load.manifest_sha256,
        expected_source_sha256,
    )?;
    load.model_path = Some(
        identity
            .source_model_path
            .to_str()
            .context("verified local GGUF path is not valid UTF-8")?
            .to_string(),
    );
    load.source_model_bytes = Some(identity.source_model_bytes);
    load.source_model_sha256 = Some(identity.source_model_sha256);
    Ok(true)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_addressed_ref_requires_full_lowercase_sha256() {
        let digest = "a".repeat(64);
        assert_eq!(
            content_addressed_package_ref(&digest).unwrap(),
            format!("{CONTENT_ADDRESSED_GGUF_PREFIX}{digest}")
        );
        assert!(content_addressed_package_ref(&"a".repeat(63)).is_err());
        assert!(content_addressed_package_ref(&"A".repeat(64)).is_err());
        assert!(content_addressed_package_ref(&"g".repeat(64)).is_err());
    }

    #[test]
    fn registry_is_only_available_for_existing_content_addressed_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"not-a-real-model").unwrap();
        let package_ref = content_addressed_package_ref(&"b".repeat(64)).unwrap();

        register_content_addressed_source(&package_ref, &path);
        assert_eq!(
            registered_content_addressed_source(&package_ref),
            Some(path.clone())
        );

        std::fs::remove_file(path).unwrap();
        assert_eq!(registered_content_addressed_source(&package_ref), None);
    }

    #[test]
    fn local_required_policy_is_profile_scoped_and_strengthens_legacy_requests() {
        let model_id = format!("strict-model-{}", std::process::id());
        assert!(!effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(effective_local_source_required(
            &model_id,
            Some("strict"),
            true
        ));
        register_local_source_policy(&model_id, "strict", true);
        register_local_source_policy(&model_id, "fallback", false);
        assert!(effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(!effective_local_source_required(
            &model_id,
            Some("fallback"),
            false
        ));
        assert!(effective_local_source_required(
            &model_id,
            Some("unknown"),
            false
        ));
        assert!(effective_local_source_required(&model_id, None, false));

        register_local_source_policy(&model_id, "", false);
        assert!(!effective_local_source_required(&model_id, Some(""), false));
        assert!(effective_local_source_required(&model_id, None, false));

        register_local_source_policy(&model_id, "strict", false);
        assert!(!effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(!effective_local_source_required(&model_id, None, false));
    }

    #[test]
    fn stopped_strict_profile_does_not_poison_later_fallback_loads() {
        let model_id = format!("reloaded-policy-model-{}", std::process::id());
        register_local_source_policy(&model_id, "strict", true);

        unregister_local_source_policy(&model_id, "strict");
        register_local_source_policy(&model_id, "fallback", false);

        assert!(!effective_local_source_required(&model_id, None, false));
        assert!(!effective_local_source_required(
            &model_id,
            Some("fallback"),
            false
        ));
    }
}
