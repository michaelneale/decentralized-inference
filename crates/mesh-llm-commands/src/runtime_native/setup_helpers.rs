use super::NativeRuntimeConfigSelection;
use anyhow::Result;
use mesh_llm_native_runtime::{CachePrunePlan, NativeRuntimePruneMode, RuntimeSelection};
use mesh_llm_runtime_install::{
    NativeRuntimeDownloadProgressCallback, NativeRuntimeInstallOptions,
    NativeRuntimeInstallOutcome, install_native_runtime, native_runtime_cache,
};
use std::future::Future;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct SetupNativeRuntimeOptions<'a> {
    pub skip_runtime: bool,
    pub requested_runtime: Option<&'a str>,
    pub manifest_path: Option<&'a Path>,
    pub bundle_dirs: &'a [PathBuf],
    pub cache_dir: Option<&'a Path>,
    pub configured: NativeRuntimeConfigSelection<'a>,
    pub progress: Option<NativeRuntimeDownloadProgressCallback>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SetupNativeRuntimeStatus {
    Skipped,
    Installed(Box<NativeRuntimeInstallOutcome>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SetupNativeRuntimePruneResult {
    Skipped,
    Pruned(CachePrunePlan),
    Warning(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupNativeRuntimeOutcome {
    pub status: SetupNativeRuntimeStatus,
    pub prune: SetupNativeRuntimePruneResult,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedNativeRuntimeSelection<'a> {
    pub(super) selection: RuntimeSelection,
    pub(super) configured_selection: Option<&'a str>,
}

pub async fn install_and_prune_native_runtime_for_setup(
    options: SetupNativeRuntimeOptions<'_>,
) -> Result<SetupNativeRuntimeOutcome> {
    install_and_prune_native_runtime_for_setup_with(
        options,
        install_native_runtime,
        prune_inactive_native_runtime_cache,
    )
    .await
}

pub(super) fn resolve_runtime_selection<'a>(
    requested_runtime: Option<&'a str>,
    configured: NativeRuntimeConfigSelection<'a>,
) -> Result<ResolvedNativeRuntimeSelection<'a>> {
    let configured_selection = requested_runtime
        .is_none()
        .then_some(configured.selection)
        .flatten();
    let selection = RuntimeSelection::parse(requested_runtime.or(configured_selection))?;
    Ok(ResolvedNativeRuntimeSelection {
        selection,
        configured_selection,
    })
}

pub(super) fn native_runtime_install_options(
    selection: RuntimeSelection,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    configured: NativeRuntimeConfigSelection<'_>,
    progress: Option<NativeRuntimeDownloadProgressCallback>,
) -> NativeRuntimeInstallOptions {
    NativeRuntimeInstallOptions {
        mesh_version: configured.mesh_version_or_current().to_string(),
        skippy_abi_version: configured.skippy_abi_version.map(ToString::to_string),
        selection,
        manifest_path: manifest_path.map(Path::to_path_buf),
        bundle_dirs: bundle_dirs.to_vec(),
        cache_dir: cache_dir.map(Path::to_path_buf),
        progress,
        ..Default::default()
    }
}

pub(super) fn prune_native_runtime_cache(
    mesh_version: &str,
    mode: NativeRuntimePruneMode,
    cache_dir: Option<&Path>,
) -> Result<CachePrunePlan> {
    let cache = native_runtime_cache(cache_dir)?;
    cache.prune(mesh_version, mode)
}

fn prune_inactive_native_runtime_cache(
    mesh_version: &str,
    cache_dir: Option<&Path>,
) -> Result<CachePrunePlan> {
    prune_native_runtime_cache(
        mesh_version,
        NativeRuntimePruneMode::KeepActiveAndPrevious,
        cache_dir,
    )
}

async fn install_and_prune_native_runtime_for_setup_with<InstallFn, InstallFuture, PruneFn>(
    options: SetupNativeRuntimeOptions<'_>,
    install: InstallFn,
    prune: PruneFn,
) -> Result<SetupNativeRuntimeOutcome>
where
    InstallFn: Fn(NativeRuntimeInstallOptions) -> InstallFuture,
    InstallFuture: Future<Output = Result<NativeRuntimeInstallOutcome>>,
    PruneFn: Fn(&str, Option<&Path>) -> Result<CachePrunePlan>,
{
    if options.skip_runtime {
        return Ok(SetupNativeRuntimeOutcome {
            status: SetupNativeRuntimeStatus::Skipped,
            prune: SetupNativeRuntimePruneResult::Skipped,
        });
    }

    let resolved_selection =
        resolve_runtime_selection(options.requested_runtime, options.configured)?;
    let install_options = native_runtime_install_options(
        resolved_selection.selection,
        options.manifest_path,
        options.bundle_dirs,
        options.cache_dir,
        options.configured,
        options.progress,
    );
    let mesh_version = install_options.mesh_version.clone();
    let cache_dir = install_options.cache_dir.clone();
    let outcome = install(install_options).await?;
    let prune = match prune(&mesh_version, cache_dir.as_deref()) {
        Ok(plan) => SetupNativeRuntimePruneResult::Pruned(plan),
        Err(error) => SetupNativeRuntimePruneResult::Warning(error.to_string()),
    };

    Ok(SetupNativeRuntimeOutcome {
        status: SetupNativeRuntimeStatus::Installed(Box::new(outcome)),
        prune,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_native_runtime::{
        CachePrunePlan, InstalledNativeRuntime, NativeRuntimeArtifact, NativeRuntimeBackend,
        NativeRuntimeBackendKind, NativeRuntimePlatform, NativeRuntimeResolution,
        NativeRuntimeSource,
    };
    use mesh_llm_runtime_install::{
        CURRENT_MESH_VERSION, NativeRuntimeBundleInstallPolicy, NativeRuntimeInstallStatus,
    };
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    #[tokio::test]
    async fn setup_runtime_helper_honors_configured_selection() {
        let install_calls = Arc::new(Mutex::new(Vec::new()));
        let install_calls_for_executor = Arc::clone(&install_calls);

        let outcome = install_and_prune_native_runtime_for_setup_with(
            SetupNativeRuntimeOptions {
                skip_runtime: false,
                requested_runtime: None,
                manifest_path: None,
                bundle_dirs: &[],
                cache_dir: None,
                configured: NativeRuntimeConfigSelection {
                    mesh_version: Some("0.68.0"),
                    skippy_abi_version: Some("0.1.25"),
                    selection: Some("cpu"),
                },
                progress: None,
            },
            move |options| {
                install_calls_for_executor
                    .lock()
                    .expect("lock install calls")
                    .push(options.clone());
                async move { Ok(fake_install_outcome("0.68.0")) }
            },
            |_mesh_version, _cache_dir| {
                Ok(CachePrunePlan {
                    remove_dirs: Vec::new(),
                })
            },
        )
        .await
        .expect("setup runtime helper should install");

        let calls = install_calls.lock().expect("lock install calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].mesh_version, "0.68.0");
        assert_eq!(calls[0].skippy_abi_version.as_deref(), Some("0.1.25"));
        assert_eq!(
            calls[0].bundle_install_policy,
            NativeRuntimeBundleInstallPolicy::UseInPlace
        );
        assert_eq!(
            calls[0].selection,
            RuntimeSelection::Backend {
                kind: NativeRuntimeBackendKind::Cpu,
                cuda_toolkit_major: None,
            }
        );
        assert!(matches!(
            outcome.status,
            SetupNativeRuntimeStatus::Installed(_)
        ));
        assert_eq!(
            outcome.prune,
            SetupNativeRuntimePruneResult::Pruned(CachePrunePlan {
                remove_dirs: Vec::new(),
            })
        );
    }

    #[tokio::test]
    async fn setup_runtime_helper_skips_install_and_prune() {
        let install_called = AtomicBool::new(false);
        let prune_called = AtomicBool::new(false);

        let outcome = install_and_prune_native_runtime_for_setup_with(
            SetupNativeRuntimeOptions {
                skip_runtime: true,
                requested_runtime: None,
                manifest_path: None,
                bundle_dirs: &[],
                cache_dir: None,
                configured: NativeRuntimeConfigSelection::default(),
                progress: None,
            },
            |_options| {
                install_called.store(true, Ordering::SeqCst);
                async move { Ok(fake_install_outcome(CURRENT_MESH_VERSION)) }
            },
            |_mesh_version, _cache_dir| {
                prune_called.store(true, Ordering::SeqCst);
                Ok(CachePrunePlan {
                    remove_dirs: Vec::new(),
                })
            },
        )
        .await
        .expect("skip runtime should succeed");

        assert_eq!(outcome.status, SetupNativeRuntimeStatus::Skipped);
        assert_eq!(outcome.prune, SetupNativeRuntimePruneResult::Skipped);
        assert!(!install_called.load(Ordering::SeqCst));
        assert!(!prune_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn setup_runtime_helper_keeps_install_success_when_prune_warns() {
        let outcome = install_and_prune_native_runtime_for_setup_with(
            SetupNativeRuntimeOptions {
                skip_runtime: false,
                requested_runtime: None,
                manifest_path: None,
                bundle_dirs: &[],
                cache_dir: None,
                configured: NativeRuntimeConfigSelection {
                    mesh_version: Some("0.68.0"),
                    skippy_abi_version: None,
                    selection: Some("recommended"),
                },
                progress: None,
            },
            |_options| async move { Ok(fake_install_outcome("0.68.0")) },
            |_mesh_version, _cache_dir| anyhow::bail!("prune failed after install"),
        )
        .await
        .expect("prune warnings should not fail setup install");

        match outcome.status {
            SetupNativeRuntimeStatus::Installed(ref installed) => {
                assert_eq!(installed.status, NativeRuntimeInstallStatus::Installed);
                assert_eq!(installed.runtime.mesh_version, "0.68.0");
            }
            SetupNativeRuntimeStatus::Skipped => panic!("install should have run"),
        }

        assert_eq!(
            outcome.prune,
            SetupNativeRuntimePruneResult::Warning("prune failed after install".to_string())
        );
    }

    fn fake_install_outcome(mesh_version: &str) -> NativeRuntimeInstallOutcome {
        let artifact = NativeRuntimeArtifact {
            id: "meshllm-runtime-linux-x86_64-cpu".to_string(),
            mesh_version: Some(mesh_version.to_string()),
            skippy_abi: "0.1.25".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: None,
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["libmeshllm_runtime.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        };

        NativeRuntimeInstallOutcome {
            status: NativeRuntimeInstallStatus::Installed,
            runtime: InstalledNativeRuntime {
                mesh_version: mesh_version.to_string(),
                native_runtime_id: artifact.id.clone(),
                flavor: "cpu".to_string(),
                path: PathBuf::from("/tmp/meshllm-runtime-linux-x86_64-cpu"),
                manifest: mesh_llm_native_runtime::NativeRuntimeManifest {
                    runtime: artifact.clone(),
                },
            },
            resolution: NativeRuntimeResolution {
                selected: artifact,
                source: NativeRuntimeSource::Installed {
                    path: PathBuf::from("/tmp/meshllm-runtime-linux-x86_64-cpu"),
                },
                evaluated: Vec::new(),
            },
            sources: Default::default(),
        }
    }
}
