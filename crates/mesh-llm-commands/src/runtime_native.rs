mod formatters;
mod setup_helpers;

use anyhow::Result;
use mesh_llm_native_runtime::{
    CandidateEvaluation, NativeRuntimeArtifact, NativeRuntimePruneMode,
    NativeRuntimeReleaseManifest, NativeRuntimeResolver, RuntimeSelection,
};
use mesh_llm_runtime_install::{
    CURRENT_MESH_VERSION, NativeRuntimeBundleInstallPolicy, NativeRuntimeDownloadProgressCallback,
    NativeRuntimeInstallOptions, NativeRuntimeManifestOptions, current_skippy_abi_version,
    discover_local_native_runtimes, discover_native_runtime_bundle_dirs, host_runtime_profile,
    install_native_runtime, load_release_manifest_with_sources, native_runtime_cache,
};
use mesh_llm_tui::terminal_progress::{
    ratio_complete_u64, render_inline_gauge_with_reserved_width,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use formatters::{AvailableRuntimeRow, NativeRuntimeDoctorReport, runtime_native_formatter};
pub use setup_helpers::{
    SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult,
    SetupNativeRuntimeStatus, install_and_prune_native_runtime_for_setup,
};
use setup_helpers::{
    native_runtime_install_options, prune_native_runtime_cache, resolve_runtime_selection,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeRuntimeDoctorReadiness {
    healthy: bool,
    status: String,
    blockers: Vec<String>,
    recommendations: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeRuntimeConfigSelection<'a> {
    pub mesh_version: Option<&'a str>,
    pub skippy_abi_version: Option<&'a str>,
    pub selection: Option<&'a str>,
}

impl<'a> NativeRuntimeConfigSelection<'a> {
    fn mesh_version_or_current(self) -> &'a str {
        self.mesh_version.unwrap_or(CURRENT_MESH_VERSION)
    }
}

pub async fn run_native_runtime_list(
    available: bool,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    configured: NativeRuntimeConfigSelection<'_>,
    json_output: bool,
) -> Result<()> {
    let mesh_version = configured.mesh_version_or_current();
    let selection = RuntimeSelection::parse(configured.selection)?;
    let cache = native_runtime_cache(cache_dir)?;
    let formatter = runtime_native_formatter(json_output);
    if available {
        let discovered_bundle_dirs = discover_native_runtime_bundle_dirs(bundle_dirs)?;
        print_configured_selector(configured, json_output);
        if !json_output && manifest_path.is_none() {
            eprintln!("🔎 Loading native runtime release manifest");
        }
        let (manifest, sources) =
            load_release_manifest_with_sources(NativeRuntimeManifestOptions {
                mesh_version: mesh_version.to_string(),
                manifest_path: manifest_path.map(Path::to_path_buf),
                bundle_dirs: discovered_bundle_dirs.clone(),
                ..Default::default()
            })
            .await?;
        let profile = host_runtime_profile();
        let cache = native_runtime_cache(cache_dir)?;
        let resolver =
            NativeRuntimeResolver::new(mesh_version, profile.clone(), manifest.clone(), cache)
                .with_bundle_dirs(sources.bundle_dirs.clone())
                .with_skippy_abi_version(listing_skippy_abi_version(
                    configured,
                    mesh_version,
                    &manifest,
                ));
        let evaluated = resolver.evaluate(&selection)?;
        let rows = available_runtime_rows(&manifest, &evaluated);
        return formatter.render_available(&rows, &sources);
    }

    let installed = discover_local_native_runtimes(bundle_dirs, &cache)?;
    formatter.render_installed(&installed, cache.root())
}

fn listing_skippy_abi_version(
    configured: NativeRuntimeConfigSelection<'_>,
    mesh_version: &str,
    manifest: &NativeRuntimeReleaseManifest,
) -> String {
    configured
        .skippy_abi_version
        .map(ToString::to_string)
        .unwrap_or_else(|| {
            if mesh_version == CURRENT_MESH_VERSION {
                current_skippy_abi_version()
            } else {
                manifest.skippy_abi.clone()
            }
        })
}

fn available_runtime_rows(
    manifest: &NativeRuntimeReleaseManifest,
    evaluated: &[CandidateEvaluation],
) -> Vec<AvailableRuntimeRow> {
    manifest
        .artifacts
        .iter()
        .map(|artifact| {
            let evaluation = evaluated.iter().find(|candidate| {
                runtime_artifact_identity(&candidate.artifact, &manifest.mesh_version)
                    == runtime_artifact_identity(artifact, &manifest.mesh_version)
            });
            AvailableRuntimeRow {
                id: artifact.id.clone(),
                mesh_version: artifact.mesh_version.clone(),
                skippy_abi: artifact.skippy_abi.clone(),
                backend: artifact.backend.kind.to_string(),
                os: artifact.platform.os.clone(),
                arch: artifact.platform.arch.clone(),
                supported: evaluation.is_some_and(|candidate| candidate.compatible),
                rejection_reasons: evaluation
                    .map(|candidate| candidate.rejection_reasons.clone())
                    .unwrap_or_default(),
                url: artifact.url.clone(),
            }
        })
        .collect()
}

fn runtime_artifact_identity<'a>(
    artifact: &'a NativeRuntimeArtifact,
    manifest_mesh_version: &'a str,
) -> (&'a str, &'a str, &'a str) {
    (
        artifact.id.as_str(),
        artifact
            .mesh_version
            .as_deref()
            .unwrap_or(manifest_mesh_version),
        artifact.skippy_abi.as_str(),
    )
}

pub async fn run_native_runtime_install(
    requested_runtime: Option<&str>,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    configured: NativeRuntimeConfigSelection<'_>,
    json_output: bool,
) -> Result<()> {
    let resolved_selection = resolve_runtime_selection(requested_runtime, configured)?;
    if !json_output && manifest_path.is_none() {
        eprintln!("🔎 Loading native runtime release manifest");
    }
    if !json_output {
        eprintln!("🔎 Detecting host runtime profile");
    }
    print_configured_selector(
        NativeRuntimeConfigSelection {
            selection: resolved_selection.configured_selection,
            ..configured
        },
        json_output,
    );
    let formatter = runtime_native_formatter(json_output);
    let install_options = cli_native_runtime_install_options(native_runtime_install_options(
        resolved_selection.selection,
        manifest_path,
        bundle_dirs,
        cache_dir,
        configured,
        cli_download_progress(json_output),
    ));
    let outcome = match install_native_runtime(install_options).await {
        Ok(outcome) => outcome,
        Err(error) => {
            formatter.render_install_error(&error)?;
            return Err(error);
        }
    };
    formatter.render_install(&outcome)
}

fn cli_native_runtime_install_options(
    mut options: NativeRuntimeInstallOptions,
) -> NativeRuntimeInstallOptions {
    options.bundle_install_policy =
        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache;
    options
}

fn print_configured_selector(configured: NativeRuntimeConfigSelection<'_>, json_output: bool) {
    if json_output || configured.mesh_version.is_none() {
        return;
    }
    let mesh_version = configured.mesh_version_or_current();
    eprintln!("🔒 Using native runtime selector from config");
    eprintln!("   mesh version: {mesh_version}");
    if let Some(skippy_abi_version) = configured.skippy_abi_version {
        eprintln!("   Skippy ABI: {skippy_abi_version}");
    }
    if let Some(configured_selection) = configured.selection {
        eprintln!("   selection: {configured_selection}");
    }
}

struct DownloadProgress {
    native_runtime_id: Option<String>,
    last_percent: Option<u64>,
    last_tick: Instant,
}

impl DownloadProgress {
    fn new() -> Self {
        Self {
            native_runtime_id: None,
            last_percent: None,
            last_tick: Instant::now(),
        }
    }

    fn tick(
        &mut self,
        native_runtime_id: &str,
        downloaded: u64,
        total: Option<u64>,
        finished: bool,
    ) {
        if self.native_runtime_id.is_none() {
            self.native_runtime_id = Some(native_runtime_id.to_string());
            eprintln!("⬇️  Downloading native runtime {native_runtime_id}");
        }
        if finished {
            self.finish(downloaded);
            return;
        }
        let should_print = match total {
            Some(total) if total > 0 => {
                let percent = downloaded.saturating_mul(100) / total;
                let crossed_step = self
                    .last_percent
                    .map(|last| percent >= last.saturating_add(5))
                    .unwrap_or(true);
                if crossed_step || percent == 100 {
                    self.last_percent = Some(percent);
                    true
                } else {
                    false
                }
            }
            _ => self.last_tick.elapsed() >= Duration::from_secs(1),
        };
        if should_print {
            self.last_tick = Instant::now();
            match total {
                Some(total) if total > 0 => {
                    let gauge = render_inline_gauge_with_reserved_width(
                        ratio_complete_u64(downloaded, total),
                        &format!(
                            "downloaded {} / {} ({}%)",
                            human_bytes(downloaded),
                            human_bytes(total),
                            self.last_percent.unwrap_or(0)
                        ),
                        3,
                    );
                    eprint!("\r\x1b[2K   {gauge}");
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                _ => {
                    eprint!("\r\x1b[2K   downloaded {}", human_bytes(downloaded));
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
            }
        }
    }

    fn finish(&mut self, downloaded: u64) {
        eprintln!("\r\x1b[2K   downloaded {}", human_bytes(downloaded));
    }
}

fn cli_download_progress(json_output: bool) -> Option<NativeRuntimeDownloadProgressCallback> {
    if json_output {
        return None;
    }
    let progress = Arc::new(Mutex::new(DownloadProgress::new()));
    Some(Arc::new(move |event| {
        let Ok(mut progress) = progress.lock() else {
            return;
        };
        progress.tick(
            &event.native_runtime_id,
            event.downloaded_bytes,
            event.total_bytes,
            event.finished,
        );
    }))
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in UNITS.iter().skip(1) {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

pub fn run_native_runtime_remove(
    native_runtime_id: &str,
    mesh_version: Option<&str>,
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let cache = native_runtime_cache(cache_dir)?;
    let removed = cache.remove(version, native_runtime_id)?;
    runtime_native_formatter(json_output).render_remove(native_runtime_id, version, removed)
}

pub fn run_native_runtime_prune(
    active_only: bool,
    mesh_version: Option<&str>,
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let mode = if active_only {
        NativeRuntimePruneMode::ActiveOnly
    } else {
        NativeRuntimePruneMode::KeepActiveAndPrevious
    };
    let plan = prune_native_runtime_cache(version, mode, cache_dir)?;
    runtime_native_formatter(json_output).render_prune(&plan)
}

pub fn run_native_runtime_doctor(
    mesh_version: Option<&str>,
    skippy_abi_version: Option<&str>,
    configured_selection: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let cache = native_runtime_cache(None)?;
    let profile = host_runtime_profile();
    let installed = discover_local_native_runtimes(&[], &cache)?;
    let selected_mesh_version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let runtime_selection = RuntimeSelection::parse(configured_selection)?;
    let selected_version_runtimes = installed
        .iter()
        .filter(|runtime| runtime.mesh_version == selected_mesh_version)
        .collect::<Vec<_>>();
    let installed_artifacts = selected_version_runtimes
        .iter()
        .map(|runtime| runtime.manifest.runtime.clone())
        .collect::<Vec<_>>();
    let selected_candidate = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
        &installed_artifacts,
        &profile,
        selected_mesh_version,
        skippy_abi_version,
        &runtime_selection,
    );
    let selected = selected_candidate.as_ref().and_then(|candidate| {
        selected_version_runtimes.iter().find(|runtime| {
            runtime.native_runtime_id == candidate.artifact.id
                && runtime.manifest.runtime.skippy_abi == candidate.artifact.skippy_abi
        })
    });
    let readiness =
        native_runtime_doctor_readiness(selected.map(|runtime| runtime.native_runtime_id.as_str()));

    let report = NativeRuntimeDoctorReport {
        healthy: readiness.healthy,
        status: readiness.status,
        blockers: readiness.blockers,
        recommendations: readiness.recommendations,
        running_mesh_version: CURRENT_MESH_VERSION.to_string(),
        selected_mesh_version: selected_mesh_version.to_string(),
        configured_skippy_abi: skippy_abi_version.map(ToString::to_string),
        configured_selection: configured_selection.map(ToString::to_string),
        host: profile,
        cache_path: cache.root().to_path_buf(),
        selected_runtime_id: selected.map(|runtime| runtime.native_runtime_id.clone()),
        selected_runtime_flavor: selected.map(|runtime| runtime.flavor.clone()),
        selected_runtime_path: selected.map(|runtime| runtime.path.clone()),
        installed_count: installed.len(),
        selected_version_installed_count: selected_version_runtimes.len(),
    };

    runtime_native_formatter(json_output).render_doctor(&report)?;
    if !report.healthy {
        anyhow::bail!(
            "{}",
            report
                .blockers
                .first()
                .map(String::as_str)
                .unwrap_or("native runtime doctor found a blocking readiness issue")
        );
    }
    Ok(())
}

fn native_runtime_doctor_readiness(
    selected_runtime_id: Option<&str>,
) -> NativeRuntimeDoctorReadiness {
    if selected_runtime_id.is_some() {
        return NativeRuntimeDoctorReadiness {
            healthy: true,
            status: "ok".to_string(),
            blockers: Vec::new(),
            recommendations: Vec::new(),
        };
    }
    NativeRuntimeDoctorReadiness {
        healthy: false,
        status: "unhealthy".to_string(),
        blockers: vec![
            "No compatible native runtime is installed for the selected MeshLLM version and host."
                .to_string(),
        ],
        recommendations: vec![
            "Run `mesh-llm runtime install` to install the recommended native runtime.".to_string(),
            "Run `mesh-llm runtime list --available` to inspect compatible and rejected runtimes."
                .to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_native_runtime::{
        CandidateEvaluation, CandidateRejection, NativeRuntimeArtifact, NativeRuntimeBackend,
        NativeRuntimeCache, NativeRuntimeManifest, NativeRuntimePlatform,
    };

    fn available_runtime_artifact(
        runtime_id: &str,
        mesh_version: Option<&str>,
        skippy_abi: &str,
    ) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: runtime_id.to_string(),
            mesh_version: mesh_version.map(ToString::to_string),
            skippy_abi: skippy_abi.to_string(),
            platform: NativeRuntimePlatform {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                target: None,
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        }
    }

    fn write_test_runtime(path: &Path, runtime_id: &str) {
        std::fs::create_dir_all(path.join("lib")).unwrap();
        std::fs::write(path.join("lib/libllama.so"), b"native runtime").unwrap();
        NativeRuntimeManifest {
            runtime: NativeRuntimeArtifact {
                id: runtime_id.to_string(),
                mesh_version: Some(CURRENT_MESH_VERSION.to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec!["lib/libllama.so".to_string()],
                files: Default::default(),
                tools: Default::default(),
                url: None,
                sha256: None,
                signature: None,
            },
        }
        .write_to_dir(path)
        .unwrap();
    }

    fn test_artifact(
        runtime_id: &str,
        mesh_version: &str,
        skippy_abi: &str,
    ) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: runtime_id.to_string(),
            mesh_version: Some(mesh_version.to_string()),
            skippy_abi: skippy_abi.to_string(),
            platform: NativeRuntimePlatform {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                target: None,
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        }
    }

    #[test]
    fn current_listing_defaults_to_the_build_skippy_abi() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            skippy_abi: "0.1.44".to_string(),
            artifacts: Vec::new(),
        };

        assert_eq!(
            listing_skippy_abi_version(
                NativeRuntimeConfigSelection::default(),
                CURRENT_MESH_VERSION,
                &manifest,
            ),
            current_skippy_abi_version()
        );
    }

    #[test]
    fn explicitly_selected_other_mesh_version_keeps_manifest_abi_default() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.75.0".to_string(),
            skippy_abi: "0.1.44".to_string(),
            artifacts: Vec::new(),
        };

        assert_eq!(
            listing_skippy_abi_version(
                NativeRuntimeConfigSelection {
                    mesh_version: Some("0.75.0"),
                    ..Default::default()
                },
                "0.75.0",
                &manifest,
            ),
            "0.1.44"
        );
    }

    #[test]
    fn available_rows_match_same_id_by_mesh_version_and_skippy_abi() {
        let current_abi = current_skippy_abi_version();
        let stale_abi = "0.1.44";
        let runtime_id = "meshllm-runtime-macos-arm64-cpu";
        let current = test_artifact(runtime_id, CURRENT_MESH_VERSION, &current_abi);
        let stale = test_artifact(runtime_id, CURRENT_MESH_VERSION, stale_abi);
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            skippy_abi: stale_abi.to_string(),
            artifacts: vec![stale.clone(), current.clone()],
        };
        let cache = tempfile::tempdir().unwrap();
        let evaluated = NativeRuntimeResolver::new(
            CURRENT_MESH_VERSION,
            host_runtime_profile(),
            manifest.clone(),
            NativeRuntimeCache::new(cache.path()),
        )
        .with_skippy_abi_version(current_abi.clone())
        .evaluate(&RuntimeSelection::Recommended)
        .unwrap();

        let rows = available_runtime_rows(&manifest, &evaluated);
        assert_eq!(rows.len(), 2);
        let current_row = rows
            .iter()
            .find(|row| row.skippy_abi == current_abi)
            .expect("current ABI row should be listed");
        assert!(current_row.supported);
        assert!(current_row.rejection_reasons.is_empty());

        let stale_row = rows
            .iter()
            .find(|row| row.skippy_abi == stale_abi)
            .expect("stale ABI row should be listed");
        assert!(!stale_row.supported);
        assert!(stale_row.rejection_reasons.iter().any(|reason| matches!(
            reason,
            mesh_llm_native_runtime::CandidateRejection::SkippyAbiMismatch { expected, actual }
                if expected == &current_abi && actual == stale_abi
        )));
    }

    #[test]
    fn doctor_readiness_blocks_missing_selected_runtime() {
        let readiness = native_runtime_doctor_readiness(None);

        assert!(!readiness.healthy);
        assert_eq!(readiness.status, "unhealthy");
        assert!(
            readiness
                .blockers
                .iter()
                .any(|item| item.contains("No compatible native runtime"))
        );
        assert!(
            readiness
                .recommendations
                .iter()
                .any(|item| item.contains("mesh-llm runtime install"))
        );
        assert!(
            readiness
                .recommendations
                .iter()
                .any(|item| item.contains("mesh-llm runtime list --available"))
        );
    }

    #[test]
    fn doctor_readiness_accepts_selected_runtime() {
        let readiness = native_runtime_doctor_readiness(Some("meshllm-native-runtime-test-cpu"));

        assert!(readiness.healthy);
        assert_eq!(readiness.status, "ok");
        assert!(readiness.blockers.is_empty());
    }

    #[test]
    fn available_runtime_rows_match_runtime_identity_not_id_only() {
        let release_artifact = available_runtime_artifact("runtime", None, "0.1.44");
        let bundled_artifact =
            available_runtime_artifact("runtime", Some(CURRENT_MESH_VERSION), "0.1.49");
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            skippy_abi: "0.1.44".to_string(),
            artifacts: vec![release_artifact.clone(), bundled_artifact.clone()],
        };
        let mut evaluated_release_artifact = release_artifact;
        evaluated_release_artifact.mesh_version = Some(CURRENT_MESH_VERSION.to_string());
        let evaluated = vec![
            CandidateEvaluation {
                artifact: evaluated_release_artifact,
                compatible: true,
                rank: 0,
                rejection_reasons: Vec::new(),
            },
            CandidateEvaluation {
                artifact: bundled_artifact,
                compatible: false,
                rank: 0,
                rejection_reasons: vec![CandidateRejection::SkippyAbiMismatch {
                    expected: "0.1.44".to_string(),
                    actual: "0.1.49".to_string(),
                }],
            },
        ];

        let rows = available_runtime_rows(&manifest, &evaluated);

        assert_eq!(rows.len(), 2);
        assert!(rows[0].supported);
        assert!(!rows[1].supported);
        assert_eq!(rows[1].rejection_reasons, evaluated[1].rejection_reasons);
    }

    #[test]
    fn available_runtime_bundle_dirs_expand_a_product_root() {
        let temp = tempfile::tempdir().unwrap();
        let product_root = temp.path().join("mesh-bundle");
        let runtime_dir = product_root.join("native-runtimes/runtime-a");
        write_test_runtime(&runtime_dir, "runtime-a");

        let discovered =
            discover_native_runtime_bundle_dirs(std::slice::from_ref(&product_root)).unwrap();

        assert_eq!(discovered, vec![runtime_dir.canonicalize().unwrap()]);
    }

    #[test]
    fn doctor_discovery_includes_a_bundle_when_the_cache_is_empty() {
        let temp = tempfile::tempdir().unwrap();
        let product_root = temp.path().join("mesh-bundle");
        let runtime_dir = product_root.join("native-runtimes/runtime-a");
        write_test_runtime(&runtime_dir, "runtime-a");
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));

        let discovered =
            discover_local_native_runtimes(std::slice::from_ref(&product_root), &cache).unwrap();

        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].path, runtime_dir.canonicalize().unwrap());
    }

    #[test]
    fn cli_runtime_install_options_install_explicit_bundles_into_cache() {
        let shared_options = native_runtime_install_options(
            RuntimeSelection::Recommended,
            None,
            &[],
            None,
            NativeRuntimeConfigSelection::default(),
            None,
        );
        let cli_options = cli_native_runtime_install_options(shared_options.clone());

        assert_eq!(
            shared_options.bundle_install_policy,
            NativeRuntimeBundleInstallPolicy::UseInPlace
        );
        assert_eq!(
            cli_options.bundle_install_policy,
            NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache
        );
    }
}
