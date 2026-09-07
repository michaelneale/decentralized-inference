//! Native runtime resolution, download, and installation.

use crate::cache::{host_runtime_profile, native_runtime_cache};
use crate::manifest::{
    NativeRuntimeCatalogSources, load_release_manifest_with_sources, normalize_sha256,
};
use crate::types::*;
use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use mesh_llm_native_runtime::{
    CandidateEvaluation, CandidateRejection, InstalledNativeRuntime, NativeRuntimeArtifact,
    NativeRuntimeCache, NativeRuntimeManifest, NativeRuntimeResolver, NativeRuntimeSource,
    RuntimeSelection,
};
use sha2::Digest;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
/// Installs the native runtime selected by `options`: loads the merged
/// runtime catalog (release manifest plus bundle directories), resolves the
/// best candidate for this host, then serves it in place from a bundle, from
/// the cache, or through a verified download.
pub async fn install_native_runtime(
    options: NativeRuntimeInstallOptions,
) -> Result<NativeRuntimeInstallOutcome> {
    let (manifest, sources) = load_release_manifest_with_sources(NativeRuntimeManifestOptions {
        mesh_version: options.mesh_version.clone(),
        manifest_path: options.manifest_path.clone(),
        manifest_url: options.manifest_url.clone(),
        bundle_dirs: options.bundle_dirs.clone(),
        // Offline installs (`allow_download: false`) must not reach out
        // for the default catalog either; bundles and the cache are the
        // only sources then. Explicit manifest URLs are still honoured.
        allow_default_manifest_url: options.allow_download,
    })
    .await?;
    if manifest.artifacts.is_empty() {
        return Err(NativeRuntimeResolutionError::empty_catalog(
            sources,
            options.selection.clone(),
        )
        .into());
    }
    let skippy_abi_version = options
        .skippy_abi_version
        .clone()
        .unwrap_or_else(|| manifest.skippy_abi.clone());
    let cache = native_runtime_cache(options.cache_dir.as_deref())?;
    let resolver = NativeRuntimeResolver::new(
        &options.mesh_version,
        host_runtime_profile(),
        manifest,
        cache.clone(),
    )
    .with_skippy_abi_version(skippy_abi_version)
    .with_bundle_dirs(sources.bundle_dirs.clone());
    let resolution = match resolver.resolve(&options.selection) {
        Ok(resolution) => resolution,
        Err(err) => {
            // The explanation becomes the outermost context of the resolver's
            // own error: one line on top, the resolver's verdict and the
            // deeper causes (an unreadable cache, a manifest that failed to
            // parse) preserved underneath for the chain readers.
            let explanation = match resolver.evaluate(&options.selection) {
                Ok(evaluated) => NativeRuntimeResolutionError::rejected(
                    sources,
                    options.selection.clone(),
                    &evaluated,
                ),
                // `evaluate` fails for the same reason `resolve` did when the
                // candidates could not be enumerated at all; that is not a
                // selection problem and must not read like one.
                Err(_) => NativeRuntimeResolutionError::enumeration_failed(
                    sources,
                    options.selection.clone(),
                ),
            };
            return Err(err.context(explanation));
        }
    };
    let mut outcome = install_resolved_runtime(&cache, resolution, &options).await?;
    outcome.sources = sources;
    Ok(outcome)
}

/// Why native runtime resolution failed, kept structured so `--json`
/// consumers can read the catalogs and the candidates instead of parsing
/// prose. It travels as the outermost context of the install error: its
/// `Display` is a single line, the resolver's own verdict and the deeper
/// causes stay underneath in the error chain, and
/// `anyhow::Error::downcast_ref` recovers the structure.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NativeRuntimeResolutionError {
    /// One-line verdict; also what this error displays as.
    pub summary: String,
    /// What the caller asked for.
    pub selection: RuntimeSelection,
    /// Which catalogs were consulted.
    pub catalogs: NativeRuntimeCatalogSources,
    /// Candidates that were plausible for this host and selection, with why
    /// each one was rejected. Empty when the candidates could not be
    /// enumerated at all.
    pub candidates: Vec<RejectedCandidate>,
    /// Candidates set aside before explaining anything: built for another
    /// platform, or not matching an explicit selection.
    pub set_aside: usize,
    /// `true` when the candidates could not be enumerated (unreadable bundle
    /// manifest, unreadable cache). The cause is in the error chain and the
    /// catalog is not at fault.
    pub enumeration_failed: bool,
}

/// A plausible candidate and the reasons it was rejected.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RejectedCandidate {
    pub id: String,
    pub reasons: Vec<CandidateRejection>,
}

impl NativeRuntimeResolutionError {
    /// The merged catalog listed nothing at all.
    pub(crate) fn empty_catalog(
        catalogs: NativeRuntimeCatalogSources,
        selection: RuntimeSelection,
    ) -> Self {
        Self {
            summary: "no native runtime manifest entries found".to_string(),
            selection,
            catalogs,
            candidates: Vec::new(),
            set_aside: 0,
            enumeration_failed: false,
        }
    }

    /// Every candidate was evaluated and none could be selected.
    pub(crate) fn rejected(
        catalogs: NativeRuntimeCatalogSources,
        selection: RuntimeSelection,
        evaluated: &[CandidateEvaluation],
    ) -> Self {
        let (plausible, set_aside): (Vec<_>, Vec<_>) = evaluated
            .iter()
            .partition(|candidate| candidate_is_plausible(candidate));
        let candidates = plausible
            .into_iter()
            .map(|candidate| RejectedCandidate {
                id: candidate.artifact.id.clone(),
                reasons: candidate.rejection_reasons.clone(),
            })
            .collect::<Vec<_>>();
        let summary = if candidates.is_empty() {
            format!(
                "no native runtime candidate applies to this host and selection ({} set aside)",
                set_aside.len()
            )
        } else {
            format!(
                "no compatible native runtime: {} candidate(s) rejected, {} set aside",
                candidates.len(),
                set_aside.len()
            )
        };
        Self {
            summary,
            selection,
            catalogs,
            candidates,
            set_aside: set_aside.len(),
            enumeration_failed: false,
        }
    }

    /// The candidates could not be enumerated, so nothing was evaluated.
    pub(crate) fn enumeration_failed(
        catalogs: NativeRuntimeCatalogSources,
        selection: RuntimeSelection,
    ) -> Self {
        Self {
            summary: "native runtime candidates could not be enumerated".to_string(),
            selection,
            catalogs,
            candidates: Vec::new(),
            set_aside: 0,
            enumeration_failed: true,
        }
    }

    /// Lines for a human reader: the catalogs consulted, then the plausible
    /// candidates with their rejection reasons, then what was set aside.
    pub fn explanation_lines(&self) -> Vec<String> {
        let mut lines = self
            .catalogs
            .describe()
            .into_iter()
            .map(|line| format!("catalog: {line}"))
            .collect::<Vec<_>>();
        if self.enumeration_failed {
            lines.push(
                "the candidates could not be enumerated; the cause is reported below, the catalog is not at fault"
                    .to_string(),
            );
            return lines;
        }
        for candidate in &self.candidates {
            let reasons = if candidate.reasons.is_empty() {
                "compatible".to_string()
            } else {
                candidate
                    .reasons
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            lines.push(format!("candidate {}: {reasons}", candidate.id));
        }
        if self.candidates.is_empty() && self.set_aside > 0 {
            lines.push(
                "no catalog entry applies to this host and the requested selection".to_string(),
            );
        }
        if self.set_aside > 0 {
            lines.push(format!(
                "{} other candidates were set aside: built for another platform, or not matching the requested selection",
                self.set_aside
            ));
        }
        if let RuntimeSelection::Backend { kind, .. } = &self.selection {
            lines.push(format!(
                "the {kind} runtime was requested explicitly; it was not replaced by another backend"
            ));
        }
        lines
    }
}

impl std::fmt::Display for NativeRuntimeResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary)
    }
}

impl std::error::Error for NativeRuntimeResolutionError {}

/// A candidate is worth explaining when it is built for this host and
/// matches the requested selection; anything else is only counted, so a
/// plain `runtime install` does not list every other platform's artifact.
fn candidate_is_plausible(candidate: &CandidateEvaluation) -> bool {
    !candidate.rejection_reasons.iter().any(|reason| {
        matches!(
            reason,
            CandidateRejection::SelectionMismatch { .. }
                | CandidateRejection::OsMismatch { .. }
                | CandidateRejection::ArchMismatch { .. }
                | CandidateRejection::TargetTripleMismatch { .. }
        )
    })
}

pub(crate) async fn install_resolved_runtime(
    cache: &NativeRuntimeCache,
    resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
    options: &NativeRuntimeInstallOptions,
) -> Result<NativeRuntimeInstallOutcome> {
    match resolution.source.clone() {
        NativeRuntimeSource::Installed { path: _ } => installed_outcome(cache, resolution),
        NativeRuntimeSource::Bundle { path } => {
            if should_install_explicit_bundle_into_cache(&path, options)? {
                let runtime = cache.install_from_dir(&path)?;
                return Ok(NativeRuntimeInstallOutcome {
                    status: NativeRuntimeInstallStatus::Installed,
                    runtime,
                    resolution,
                    sources: crate::manifest::NativeRuntimeCatalogSources::default(),
                });
            }
            in_place_bundle_outcome(&path, resolution)
        }
        NativeRuntimeSource::Download { url } if options.allow_download => {
            let runtime =
                download_and_install_runtime(cache, &resolution.selected, &url, options).await?;
            Ok(NativeRuntimeInstallOutcome {
                status: NativeRuntimeInstallStatus::Installed,
                runtime,
                resolution,
                sources: crate::manifest::NativeRuntimeCatalogSources::default(),
            })
        }
        NativeRuntimeSource::Download { url: _ } => {
            bail!("selected native runtime is downloadable, but downloads are disabled")
        }
        NativeRuntimeSource::Missing => {
            bail!(
                "selected native runtime {} is not installed and no bundle or download URL was available",
                resolution.selected.id
            )
        }
    }
}

pub(crate) fn should_install_explicit_bundle_into_cache(
    bundle_path: &Path,
    options: &NativeRuntimeInstallOptions,
) -> Result<bool> {
    match options.bundle_install_policy {
        NativeRuntimeBundleInstallPolicy::UseInPlace => Ok(false),
        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache => {
            bundle_path_matches_explicit_root(bundle_path, &options.bundle_dirs)
        }
    }
}

pub(crate) fn bundle_path_matches_explicit_root(
    bundle_path: &Path,
    explicit_dirs: &[PathBuf],
) -> Result<bool> {
    if explicit_dirs.is_empty() {
        return Ok(false);
    }
    let bundle_path = bundle_path.canonicalize().with_context(|| {
        format!(
            "canonicalize native runtime bundle {}",
            bundle_path.display()
        )
    })?;
    for explicit_dir in explicit_dirs {
        let Ok(explicit_dir) = explicit_dir.canonicalize() else {
            continue;
        };
        if bundle_path.starts_with(&explicit_dir) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn in_place_bundle_outcome(
    path: &Path,
    resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
) -> Result<NativeRuntimeInstallOutcome> {
    let manifest = NativeRuntimeManifest::read_from_dir(path)?;
    let runtime = InstalledNativeRuntime {
        mesh_version: manifest
            .runtime
            .mesh_version
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        native_runtime_id: manifest.runtime.id.clone(),
        flavor: manifest.runtime.backend.kind.to_string(),
        path: path.to_path_buf(),
        manifest,
    };
    Ok(NativeRuntimeInstallOutcome {
        status: NativeRuntimeInstallStatus::AlreadyInstalled,
        runtime,
        resolution,
        sources: crate::manifest::NativeRuntimeCatalogSources::default(),
    })
}

pub(crate) fn installed_outcome(
    cache: &NativeRuntimeCache,
    resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
) -> Result<NativeRuntimeInstallOutcome> {
    let runtime = cache
        .find_installed(
            resolution.selected.mesh_version_or(CURRENT_MESH_VERSION),
            resolution.selected.native_runtime_id(),
        )?
        .context("selected native runtime was not found in cache")?;
    Ok(NativeRuntimeInstallOutcome {
        status: NativeRuntimeInstallStatus::AlreadyInstalled,
        runtime,
        resolution,
        sources: crate::manifest::NativeRuntimeCatalogSources::default(),
    })
}

async fn download_and_install_runtime(
    cache: &NativeRuntimeCache,
    artifact: &NativeRuntimeArtifact,
    url: &str,
    options: &NativeRuntimeInstallOptions,
) -> Result<InstalledNativeRuntime> {
    let temp = tempfile::Builder::new()
        .prefix("mesh-native-runtime-")
        .tempdir()
        .context("create native runtime download workspace")?;
    let archive = temp
        .path()
        .join(format!("{}.tar.gz", artifact.native_runtime_id()));
    download_runtime_archive(url, &archive, artifact, options).await?;
    let extracted = temp.path().join("extracted");
    fs::create_dir_all(&extracted).with_context(|| {
        format!(
            "create native runtime extraction dir {}",
            extracted.display()
        )
    })?;
    extract_runtime_archive(&archive, &extracted)?;
    let bundle_dir = find_extracted_runtime_dir(&extracted)?;
    cache.install_from_dir(&bundle_dir)
}

async fn download_runtime_archive(
    url: &str,
    path: &Path,
    artifact: &NativeRuntimeArtifact,
    options: &NativeRuntimeInstallOptions,
) -> Result<()> {
    verify_download_policy_before_fetch(artifact, options.verification_policy)?;
    let diagnostic_url = crate::manifest::url_without_query(url);
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("build native runtime download HTTP client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("download native runtime {diagnostic_url}"))?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("native runtime request failed for {diagnostic_url}"))?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("create native runtime archive {}", path.display()))?;
    let mut downloaded = 0_u64;
    let mut hasher = sha2::Sha256::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(reqwest::Error::without_url)
            .with_context(|| format!("read native runtime body from {diagnostic_url}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write native runtime archive {}", path.display()))?;
        downloaded += chunk.len() as u64;
        sha2::Digest::update(&mut hasher, &chunk);
        emit_download_progress(artifact, url, downloaded, total, false, options);
    }
    file.flush()
        .await
        .with_context(|| format!("flush native runtime archive {}", path.display()))?;
    emit_download_progress(artifact, url, downloaded, total, true, options);
    verify_downloaded_archive(artifact, hasher)?;
    Ok(())
}

pub(crate) fn verify_download_policy_before_fetch(
    artifact: &NativeRuntimeArtifact,
    policy: NativeRuntimeVerificationPolicy,
) -> Result<()> {
    if artifact.sha256.is_none() {
        bail!(
            "native runtime {} is missing required sha256 verification metadata",
            artifact.native_runtime_id()
        );
    }
    if policy == NativeRuntimeVerificationPolicy::RequireChecksumAndSignature {
        let signature = artifact.signature.as_deref().unwrap_or_default();
        if signature.trim().is_empty() {
            bail!(
                "native runtime {} is missing required signature metadata",
                artifact.native_runtime_id()
            );
        }
        bail!("native runtime signature verification is not implemented yet");
    }
    Ok(())
}

pub(crate) fn verify_downloaded_archive(
    artifact: &NativeRuntimeArtifact,
    hasher: sha2::Sha256,
) -> Result<()> {
    let expected = artifact
        .sha256
        .as_deref()
        .context("native runtime sha256 missing after download")?;
    let expected = normalize_sha256(expected)?;
    let actual = hex::encode(sha2::Digest::finalize(hasher));
    if actual != expected {
        bail!("native runtime checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

pub(crate) fn emit_download_progress(
    artifact: &NativeRuntimeArtifact,
    url: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    finished: bool,
    options: &NativeRuntimeInstallOptions,
) {
    let Some(progress) = &options.progress else {
        return;
    };
    progress(NativeRuntimeDownloadProgress {
        native_runtime_id: artifact.id.clone(),
        url: crate::manifest::url_without_query(url),
        downloaded_bytes,
        total_bytes,
        finished,
    });
}

pub(crate) fn extract_runtime_archive(archive: &Path, extracted: &Path) -> Result<()> {
    let file = fs::File::open(archive)
        .with_context(|| format!("open native runtime archive {}", archive.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(extracted).with_context(|| {
        format!(
            "extract native runtime archive into {}",
            extracted.display()
        )
    })
}

pub(crate) fn find_extracted_runtime_dir(extracted: &Path) -> Result<PathBuf> {
    let mut matches = Vec::new();
    collect_runtime_manifest_dirs(extracted, &mut matches)?;
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("downloaded native runtime archive did not contain a manifest.json"),
        count => bail!("downloaded native runtime archive contained {count} manifest.json files"),
    }
}

pub(crate) fn collect_runtime_manifest_dirs(dir: &Path, matches: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if path
                .join(mesh_llm_native_runtime::NATIVE_RUNTIME_MANIFEST_FILE)
                .is_file()
            {
                matches.push(path);
            } else {
                collect_runtime_manifest_dirs(&path, matches)?;
            }
        }
    }
    Ok(())
}
