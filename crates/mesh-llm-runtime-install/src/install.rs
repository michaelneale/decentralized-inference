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
        bail!(
            "no native runtime manifest entries found\n{}",
            describe_catalogs(&sources)
        );
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
            let evaluated = resolver.evaluate(&options.selection).unwrap_or_default();
            bail!(
                "{err}\n{}",
                describe_resolution_failure(&sources, &options.selection, &evaluated)
            );
        }
    };
    let mut outcome = install_resolved_runtime(&cache, resolution, &options).await?;
    outcome.sources = sources;
    Ok(outcome)
}

fn describe_catalogs(sources: &NativeRuntimeCatalogSources) -> String {
    sources
        .describe()
        .into_iter()
        .map(|line| format!("catalog: {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Explains a failed selection: which catalogs were consulted, and why each
/// candidate that matched the requested selection was rejected. Candidates
/// that never matched the selection (another backend, another id) are
/// summarised as a count so the explanation stays focused on what the user
/// asked for.
pub(crate) fn describe_resolution_failure(
    sources: &NativeRuntimeCatalogSources,
    selection: &RuntimeSelection,
    evaluated: &[CandidateEvaluation],
) -> String {
    let mut lines = vec![describe_catalogs(sources)];
    let (matching, others): (Vec<_>, Vec<_>) = evaluated.iter().partition(|candidate| {
        !candidate
            .rejection_reasons
            .iter()
            .any(|reason| matches!(reason, CandidateRejection::SelectionMismatch { .. }))
    });
    if matching.is_empty() {
        lines.push(format!(
            "no catalog entry matches the requested selection ({} other candidates were not considered)",
            others.len()
        ));
    } else {
        for candidate in matching {
            let reasons = if candidate.rejection_reasons.is_empty() {
                "compatible".to_string()
            } else {
                candidate
                    .rejection_reasons
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            lines.push(format!("candidate {}: {reasons}", candidate.artifact.id));
        }
        if !others.is_empty() {
            lines.push(format!(
                "{} other candidates did not match the requested selection",
                others.len()
            ));
        }
    }
    if let RuntimeSelection::Backend { kind, .. } = selection {
        lines.push(format!(
            "the {kind} runtime was requested explicitly; it was not replaced by another backend"
        ));
    }
    lines.join("\n")
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
