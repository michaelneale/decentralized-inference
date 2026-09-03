//! Release manifest discovery, download, and verification.

use crate::cache::current_skippy_abi_version;
use crate::discovery::discover_native_runtime_bundle_dirs;
use crate::types::{NATIVE_RUNTIME_MANIFEST_URL_ENV, NativeRuntimeManifestOptions};
use anyhow::{Context, Result, bail};
use mesh_llm_native_runtime::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest,
};
use sha2::Digest;
use std::path::PathBuf;
use std::time::Duration;
pub fn default_release_manifest_url(mesh_version: &str) -> String {
    format!(
        "https://github.com/Mesh-LLM/mesh-llm/releases/download/v{mesh_version}/native-runtimes.json"
    )
}

pub fn default_manifest_url(build_version: &str, release_version: &str) -> String {
    if mesh_llm_build_info::is_sha_build(build_version) {
        "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
            .to_string()
    } else {
        default_release_manifest_url(release_version)
    }
}

pub(crate) fn request_default_manifest_url(mesh_version: &str) -> String {
    if mesh_version == mesh_llm_build_info::RELEASE_VERSION {
        default_manifest_url(
            mesh_llm_build_info::BUILD_VERSION,
            mesh_llm_build_info::RELEASE_VERSION,
        )
    } else {
        default_release_manifest_url(mesh_version)
    }
}

/// Loads the merged runtime catalog for `options` and returns the manifest
/// alone. See `load_release_manifest_with_sources` for the discovery rules.
pub async fn load_release_manifest(
    options: NativeRuntimeManifestOptions,
) -> Result<NativeRuntimeReleaseManifest> {
    Ok(load_release_manifest_with_bundle_dirs(options).await?.0)
}

/// Which catalogs a merged manifest load consulted, so callers can explain a
/// selection (or a rejection) in terms of where the candidates came from.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NativeRuntimeCatalogSources {
    /// Explicit `--manifest` file that was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_path: Option<PathBuf>,
    /// Remote manifest URL that was consulted (explicit, environment, or the
    /// default release URL), with any query string removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_url: Option<String>,
    /// Artifacts contributed by the manifest file or URL.
    #[serde(default)]
    pub manifest_artifacts: usize,
    /// Why the remote manifest could not be used, when bundled artifacts
    /// carried the load instead. `None` when the fetch succeeded or was not
    /// attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_error: Option<String>,
    /// Bundle directories whose artifacts were merged in.
    #[serde(default)]
    pub bundle_dirs: Vec<PathBuf>,
    /// Artifacts contributed by the bundle directories (after deduplication
    /// against the manifest).
    #[serde(default)]
    pub bundle_artifacts: usize,
}

impl NativeRuntimeCatalogSources {
    /// Human-readable lines stating which catalogs were consulted, suitable
    /// for CLI output and for explaining a failed selection.
    pub fn describe(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(path) = &self.manifest_path {
            lines.push(format!(
                "manifest file {} ({} artifacts)",
                path.display(),
                self.manifest_artifacts
            ));
        } else if let Some(url) = &self.manifest_url {
            match &self.remote_error {
                Some(error) => lines.push(format!(
                    "release catalog {url} unavailable, using bundles only: {error}"
                )),
                None => lines.push(format!(
                    "release catalog {url} ({} artifacts)",
                    self.manifest_artifacts
                )),
            }
        } else {
            lines.push("no release catalog consulted (downloads disabled)".to_string());
        }
        if self.bundle_dirs.is_empty() {
            lines.push("no native runtime bundle directories".to_string());
        } else {
            let dirs = self
                .bundle_dirs
                .iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            // Every bundle directory carries exactly one runtime manifest;
            // say how many of them were not already listed by the catalog.
            lines.push(format!(
                "bundle directories ({} runtimes, {} not already in the catalog): {dirs}",
                self.bundle_dirs.len(),
                self.bundle_artifacts
            ));
        }
        lines
    }
}

/// Like `load_release_manifest_with_sources`, returning only the bundle
/// directories that were merged in, for callers that resolve bundles by path.
pub(crate) async fn load_release_manifest_with_bundle_dirs(
    options: NativeRuntimeManifestOptions,
) -> Result<(NativeRuntimeReleaseManifest, Vec<PathBuf>)> {
    let (manifest, sources) = load_release_manifest_with_sources(options).await?;
    Ok((manifest, sources.bundle_dirs))
}

/// Merges every catalog the options allow: an explicit manifest file, else a
/// remote manifest (explicit URL, environment URL, or the default release
/// URL), plus every discovered bundle directory.
///
/// Precedence:
/// 1. An explicit manifest file is required: a read failure is an error.
/// 2. A remote manifest is consulted even when bundle directories exist, so
///    an adjacent bundle cannot hide downloadable runtimes. If the fetch
///    fails and bundles were discovered, the bundles carry the load and the
///    failure is recorded in `NativeRuntimeCatalogSources::remote_error`.
///    Without bundles the fetch failure is an error, as before.
/// 3. Bundle artifacts are appended after the manifest artifacts. The
///    manifest's `mesh_version` and `skippy_abi` win when a manifest was
///    loaded; bundle values only describe the release when no manifest was
///    available at all.
///
/// Candidates with the same identity coming from several sources are
/// deduplicated downstream by the resolver, which also prefers a bundled copy
/// over a download for the same artifact.
pub async fn load_release_manifest_with_sources(
    mut options: NativeRuntimeManifestOptions,
) -> Result<(NativeRuntimeReleaseManifest, NativeRuntimeCatalogSources)> {
    options.bundle_dirs = discover_native_runtime_bundle_dirs(&options.bundle_dirs)?;
    let mut sources = NativeRuntimeCatalogSources {
        bundle_dirs: options.bundle_dirs.clone(),
        ..Default::default()
    };
    let mut artifacts = Vec::new();
    let mut mesh_version = options.mesh_version.clone();
    let mut skippy_abi = current_skippy_abi_version();
    let mut manifest_loaded = false;
    if let Some(path) = options.manifest_path.take() {
        let manifest = NativeRuntimeReleaseManifest::read_from_path(&path)?;
        sources.manifest_path = Some(path);
        mesh_version = manifest.mesh_version.clone();
        skippy_abi = manifest.skippy_abi.clone();
        artifacts.extend(manifest.artifacts);
        manifest_loaded = true;
    } else if let Some(url) = manifest_url(&options) {
        sources.manifest_url = Some(url_without_query(&url));
        match download_release_manifest(&url).await {
            Ok(manifest) => {
                mesh_version = manifest.mesh_version.clone();
                skippy_abi = manifest.skippy_abi.clone();
                artifacts.extend(manifest.artifacts);
                manifest_loaded = true;
            }
            Err(err) if !options.bundle_dirs.is_empty() => {
                sources.remote_error = Some(format!("{err:#}"));
            }
            Err(err) => return Err(err),
        }
    }
    sources.manifest_artifacts = artifacts.len();
    sources.bundle_artifacts = append_bundle_artifacts(
        &mut artifacts,
        &mut mesh_version,
        &mut skippy_abi,
        &options.bundle_dirs,
        manifest_loaded,
    )?;
    Ok((
        NativeRuntimeReleaseManifest {
            mesh_version,
            skippy_abi,
            artifacts,
        },
        sources,
    ))
}

pub(crate) async fn download_release_manifest(url: &str) -> Result<NativeRuntimeReleaseManifest> {
    let diagnostic_url = url_without_query(url);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("build native runtime manifest HTTP client")?;
    let bytes = client
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("download native runtime release manifest {diagnostic_url}"))?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("native runtime release manifest request failed for {diagnostic_url}")
        })?
        .bytes()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("read native runtime release manifest {diagnostic_url}"))?;
    let checksum_url = release_manifest_checksum_url(url);
    let diagnostic_checksum_url = url_without_query(&checksum_url);
    let checksum = client
        .get(&checksum_url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("download native runtime manifest checksum {diagnostic_checksum_url}")
        })?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("native runtime manifest checksum request failed for {diagnostic_checksum_url}")
        })?
        .text()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("read native runtime manifest checksum {diagnostic_checksum_url}")
        })?;
    verify_release_manifest_checksum(&bytes, &checksum)
        .with_context(|| format!("verify native runtime release manifest {diagnostic_url}"))?;
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("decode native runtime release manifest {diagnostic_url}"))?;
    NativeRuntimeReleaseManifest::from_json_str(text)
        .with_context(|| format!("parse native runtime release manifest {diagnostic_url}"))
}

pub(crate) fn verify_release_manifest_checksum(
    manifest_bytes: &[u8],
    checksum_text: &str,
) -> Result<()> {
    let expected = normalize_sha256(checksum_text)?;
    let actual = hex::encode(sha2::Sha256::digest(manifest_bytes));
    if actual != expected {
        bail!("native runtime manifest checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

pub(crate) fn release_manifest_checksum_url(url: &str) -> String {
    match url.split_once('?') {
        Some((base, query)) => format!("{base}.sha256?{query}"),
        None => format!("{url}.sha256"),
    }
}

/// Strips the query string and redacts any userinfo (`user:pass@`) from a
/// URL before it is surfaced in error context or progress events. Mirrors
/// `redact_url_userinfo` in `mesh-llm-host-runtime::logging::policy`; kept
/// local because this crate does not otherwise depend on host-runtime.
pub(crate) fn url_without_query(url: &str) -> String {
    let without_query = url.split_once('?').map_or(url, |(base, _)| base);
    redact_url_userinfo(without_query)
}

fn redact_url_userinfo(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let authority_start = scheme_end + 3;
    let authority_end = url[authority_start..]
        .find(['/', '#'])
        .map_or(url.len(), |offset| authority_start + offset);
    let authority = &url[authority_start..authority_end];
    let Some(user_info_end) = authority.rfind('@') else {
        return url.to_string();
    };
    format!(
        "{}[REDACTED]@{}{}",
        &url[..authority_start],
        &authority[user_info_end + 1..],
        &url[authority_end..]
    )
}

/// Picks the remote catalog URL to consult: the explicit option, then the
/// `MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL` override, then the default release
/// URL when `allow_default_manifest_url` is set. `None` means no remote
/// catalog is consulted.
pub(crate) fn manifest_url(options: &NativeRuntimeManifestOptions) -> Option<String> {
    options
        .manifest_url
        .clone()
        .or_else(|| {
            std::env::var(NATIVE_RUNTIME_MANIFEST_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .or_else(|| {
            // Bundle directories no longer suppress the default catalog: an
            // adjacent CPU bundle must not hide downloadable GPU runtimes
            // (#1612). Offline hosts fall back to the bundles when the fetch
            // fails, see `load_release_manifest_with_sources`.
            options
                .allow_default_manifest_url
                .then(|| request_default_manifest_url(&options.mesh_version))
        })
}

/// Appends the runtime described by each bundle directory to `artifacts`.
///
/// When no release manifest was loaded, the bundles also supply the release
/// identity (`mesh_version` and `skippy_abi`). A runtime that is both bundled
/// and already listed by the manifest is kept once; the resolver still serves
/// that identity from the bundle directory.
pub(crate) fn append_bundle_artifacts(
    artifacts: &mut Vec<NativeRuntimeArtifact>,
    mesh_version: &mut String,
    skippy_abi: &mut String,
    bundle_dirs: &[PathBuf],
    manifest_loaded: bool,
) -> Result<usize> {
    let mut appended = 0;
    for dir in bundle_dirs {
        let manifest = NativeRuntimeManifest::read_from_dir(dir)
            .with_context(|| format!("read bundled native runtime {}", dir.display()))?;
        // A loaded release manifest describes the release; a bundle only
        // stands in for it when no manifest was available at all.
        if !manifest_loaded {
            if let Some(version) = &manifest.runtime.mesh_version {
                *mesh_version = version.clone();
            }
            *skippy_abi = manifest.runtime.skippy_abi.clone();
        }
        // The same runtime can be both bundled and published. Keep one entry
        // so listings stay unambiguous; the resolver still prefers the bundle
        // directory as the source for that identity.
        let duplicate = artifacts
            .iter()
            .any(|existing| same_artifact_identity(existing, &manifest.runtime));
        if !duplicate {
            artifacts.push(manifest.runtime);
            appended += 1;
        }
    }
    Ok(appended)
}

/// Two catalog entries describe the same runtime when their id and release
/// identity (`mesh_version`, `skippy_abi`) match, whichever source listed them.
fn same_artifact_identity(left: &NativeRuntimeArtifact, right: &NativeRuntimeArtifact) -> bool {
    left.id == right.id
        && left.mesh_version == right.mesh_version
        && left.skippy_abi == right.skippy_abi
}

pub(crate) fn normalize_sha256(value: &str) -> Result<String> {
    let trimmed = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    let digest = trimmed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(digest)
    } else {
        bail!("native runtime manifest contains invalid sha256: {value}");
    }
}
