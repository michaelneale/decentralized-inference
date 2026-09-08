#[cfg(feature = "dynamic-native-runtime")]
mod dynamic {
    use crate::system::native_runtime_install::{
        NativeRuntimeInstallOptions, NativeRuntimeInstallOutcome,
    };
    use anyhow::{Context, Result};
    use mesh_llm_native_runtime::{
        HostRuntimeProfile, InstalledNativeRuntime, NativeRuntimeArtifact, NativeRuntimeCache,
        NativeRuntimeLoadPlan, NativeRuntimeReleaseManifest, RuntimeSelection,
    };
    use std::{future::Future, path::PathBuf};

    #[derive(Clone, Debug)]
    pub(crate) struct LoadedNativeRuntime {
        pub(crate) native_runtime_id: String,
        pub(crate) libraries: Vec<PathBuf>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) enum NativeRuntimePlanSource {
        CacheHit,
        LocalDiscovery,
        PostInstall,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct NativeRuntimeStartupLoadPlan {
        pub(crate) cache_mesh_version: String,
        pub(crate) native_runtime_id: String,
        pub(crate) root: PathBuf,
        pub(crate) selected_library_path: PathBuf,
        pub(crate) libraries: Vec<PathBuf>,
        pub(crate) source: NativeRuntimePlanSource,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct NativeRuntimeStartupSelection {
        pub(crate) mesh_version: String,
        pub(crate) skippy_abi: Option<String>,
        pub(crate) runtime_selection: RuntimeSelection,
    }

    impl NativeRuntimeStartupSelection {
        pub(crate) fn current() -> Self {
            Self {
                mesh_version: crate::RELEASE_VERSION.to_string(),
                skippy_abi: Some(
                    crate::system::native_runtime_install::current_skippy_abi_version(),
                ),
                runtime_selection: RuntimeSelection::Recommended,
            }
        }

        pub(crate) fn explicit(
            mesh_version: String,
            skippy_abi: Option<String>,
            runtime_selection: RuntimeSelection,
        ) -> Self {
            Self {
                mesh_version,
                skippy_abi,
                runtime_selection,
            }
        }
    }

    pub(crate) fn load_local_native_runtime_for_embedded_serving()
    -> Result<Option<LoadedNativeRuntime>> {
        if skippy_runtime::native_runtime_loaded() {
            return Ok(None);
        }
        let cache = default_native_runtime_cache()?;
        let local_runtimes =
            crate::system::native_runtime_install::discover_local_native_runtimes(&[], &cache)?;
        let Some(plan) = resolve_local_native_runtime_plan(
            &local_runtimes,
            &host_runtime_profile(),
            crate::BUILD_VERSION,
            crate::RELEASE_VERSION,
            Some(&crate::system::native_runtime_install::current_skippy_abi_version()),
            &RuntimeSelection::Recommended,
        )?
        else {
            return Ok(None);
        };
        unsafe { skippy_runtime::load_native_runtime_libraries(&plan.libraries) }
            .map_err(anyhow::Error::from)
            .with_context(|| {
                format!(
                    "load local native runtime {} from {} for embedded serving",
                    plan.native_runtime_id,
                    plan.root.display()
                )
            })?;
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    pub(crate) async fn try_load_installed_native_runtime(
        startup_selection: NativeRuntimeStartupSelection,
    ) -> Result<Option<LoadedNativeRuntime>> {
        try_load_installed_native_runtime_with(
            skippy_runtime::native_runtime_loaded,
            default_native_runtime_cache,
            host_runtime_profile,
            default_install_options,
            default_install_executor,
            startup_selection,
            |libraries| {
                unsafe { skippy_runtime::load_native_runtime_libraries(libraries) }
                    .map_err(anyhow::Error::from)
            },
        )
        .await
    }

    async fn try_load_installed_native_runtime_with<
        NativeRuntimeLoadedFn,
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
        InstallFuture,
        LoadLibrariesFn,
    >(
        native_runtime_loaded: NativeRuntimeLoadedFn,
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
        startup_selection: NativeRuntimeStartupSelection,
        load_libraries: LoadLibrariesFn,
    ) -> Result<Option<LoadedNativeRuntime>>
    where
        NativeRuntimeLoadedFn: Fn() -> bool,
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> InstallFuture,
        InstallFuture: Future<Output = Result<NativeRuntimeInstallOutcome>>,
        LoadLibrariesFn: Fn(&[PathBuf]) -> Result<()>,
    {
        if native_runtime_loaded() {
            return Ok(None);
        }
        let Some(plan) = resolve_startup_native_runtime_plan_with(
            cache,
            profile,
            install_options,
            install_executor,
            startup_selection,
        )
        .await?
        else {
            return Ok(None);
        };
        load_libraries(&plan.libraries).with_context(|| {
            format!(
                "load native runtime {} from {}",
                plan.native_runtime_id,
                plan.root.display()
            )
        })?;
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    async fn resolve_startup_native_runtime_plan_with<
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
        InstallFuture,
    >(
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
        startup_selection: NativeRuntimeStartupSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>>
    where
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> InstallFuture,
        InstallFuture: Future<Output = Result<NativeRuntimeInstallOutcome>>,
    {
        let cache = cache()?;
        let profile = profile();
        let mut options = install_options();
        options.mesh_version = startup_selection.mesh_version.clone();
        options.skippy_abi_version = startup_selection.skippy_abi.clone();
        options.selection = startup_selection.runtime_selection.clone();
        if options.cache_dir.is_none() {
            options.cache_dir = Some(cache.root().to_path_buf());
        }
        let discovered_bundle_dirs =
            crate::system::native_runtime_install::discover_native_runtime_bundle_dirs(
                &options.bundle_dirs,
            )?;
        let discovered_bundle_dirs_empty = discovered_bundle_dirs.is_empty();
        if discovered_bundle_dirs_empty {
            if let Some(plan) = resolve_installed_native_runtime_plan(
                &cache,
                &profile,
                crate::BUILD_VERSION,
                &startup_selection.mesh_version,
                startup_selection.skippy_abi.as_deref(),
                &startup_selection.runtime_selection,
            )? {
                return Ok(Some(plan));
            }
        } else {
            options.bundle_dirs = discovered_bundle_dirs;
        }

        tracing::info!(
            cache_root = %cache.root().display(),
            mesh_version = %options.mesh_version,
            "{}",
            startup_install_message(discovered_bundle_dirs_empty)
        );

        let install_result = install_executor(options.clone()).await;
        match install_result {
            Ok(outcome) => {
                let load_plan = outcome.runtime.load_plan()?;
                Ok(Some(startup_load_plan_from_installed(
                    outcome.runtime.mesh_version.clone(),
                    load_plan,
                    NativeRuntimePlanSource::PostInstall,
                )?))
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    cache_root = %cache.root().display(),
                    mesh_version = %options.mesh_version,
                    manifest_path = ?options.manifest_path,
                    manifest_url = ?options.manifest_url,
                    bundle_dirs = ?options.bundle_dirs,
                    allow_download = options.allow_download,
                    "Failed to install a compatible MeshLLM native runtime during startup; stopping before Skippy FFI load"
                );
                Err(err.context(startup_missing_native_runtime_guidance(&options)))
            }
        }
    }

    fn startup_missing_native_runtime_guidance(options: &NativeRuntimeInstallOptions) -> String {
        let abi = options
            .skippy_abi_version
            .as_deref()
            .unwrap_or("not configured");
        format!(
            "no compatible MeshLLM native runtime is installed or installable for MeshLLM {} / Skippy ABI {abi}; run `mesh-llm runtime install` or inspect available runtimes with `mesh-llm runtime list --available`",
            options.mesh_version
        )
    }

    const fn startup_install_message(discovered_bundle_dirs_empty: bool) -> &'static str {
        if discovered_bundle_dirs_empty {
            "No compatible installed MeshLLM native runtime found; attempting one-shot startup install"
        } else {
            "Discovered native runtime bundles take precedence over installed runtimes; attempting one-shot startup install"
        }
    }

    fn resolve_installed_native_runtime_plan(
        cache: &NativeRuntimeCache,
        profile: &HostRuntimeProfile,
        build_version: &str,
        target_mesh_version: &str,
        target_skippy_abi: Option<&str>,
        selection: &RuntimeSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let scan = cache.installed_lenient()?;
        for skipped in &scan.skipped {
            tracing::warn!(
                path = %skipped.path.display(),
                reason = %skipped.reason,
                "Skipping unusable native runtime cache entry during startup"
            );
        }
        let installed = scan.runtimes;
        if installed.is_empty() {
            return Ok(None);
        }
        let initial_cache_version =
            startup_native_runtime_cache_version(build_version, target_mesh_version);
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: initial_cache_version.to_string(),
            skippy_abi: target_skippy_abi.unwrap_or_default().to_string(),
            artifacts: installed
                .iter()
                .map(|runtime| runtime.manifest.runtime.clone())
                .collect(),
        };
        let Some(candidate) = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
            &manifest.artifacts,
            profile,
            initial_cache_version,
            target_skippy_abi,
            selection,
        ) else {
            return Ok(None);
        };
        load_plan_from_candidate(cache, &manifest, candidate.artifact)
    }

    fn resolve_local_native_runtime_plan(
        runtimes: &[InstalledNativeRuntime],
        profile: &HostRuntimeProfile,
        build_version: &str,
        target_mesh_version: &str,
        target_skippy_abi: Option<&str>,
        selection: &RuntimeSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        if runtimes.is_empty() {
            return Ok(None);
        }
        let cache_mesh_version =
            startup_native_runtime_cache_version(build_version, target_mesh_version);
        let artifacts = runtimes
            .iter()
            .map(|runtime| runtime.manifest.runtime.clone())
            .collect::<Vec<_>>();
        let Some(candidate) = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
            &artifacts,
            profile,
            cache_mesh_version,
            target_skippy_abi,
            selection,
        ) else {
            return Ok(None);
        };
        let selected_mesh_version = candidate
            .artifact
            .mesh_version_or(cache_mesh_version)
            .to_string();
        let Some(runtime) = runtimes.iter().find(|runtime| {
            runtime.mesh_version == selected_mesh_version
                && runtime.native_runtime_id == candidate.artifact.native_runtime_id()
                && runtime.manifest.runtime.skippy_abi == candidate.artifact.skippy_abi
        }) else {
            return Ok(None);
        };
        Ok(Some(startup_load_plan_from_installed(
            selected_mesh_version,
            runtime.load_plan()?,
            NativeRuntimePlanSource::LocalDiscovery,
        )?))
    }

    fn startup_native_runtime_cache_version<'a>(
        _build_version: &'a str,
        release_version: &'a str,
    ) -> &'a str {
        release_version
    }

    fn load_plan_from_candidate(
        cache: &NativeRuntimeCache,
        manifest: &NativeRuntimeReleaseManifest,
        artifact: NativeRuntimeArtifact,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let cache_mesh_version = artifact
            .mesh_version_or(manifest.mesh_version.as_str())
            .to_string();
        let Some(installed) =
            cache.find_installed(&cache_mesh_version, artifact.native_runtime_id())?
        else {
            return Ok(None);
        };
        let load_plan = installed.load_plan()?;
        Ok(Some(startup_load_plan_from_installed(
            cache_mesh_version,
            load_plan,
            NativeRuntimePlanSource::CacheHit,
        )?))
    }

    fn startup_load_plan_from_installed(
        cache_mesh_version: String,
        load_plan: NativeRuntimeLoadPlan,
        source: NativeRuntimePlanSource,
    ) -> Result<NativeRuntimeStartupLoadPlan> {
        let selected_library_path = load_plan
            .libraries
            .first()
            .cloned()
            .context("native runtime load plan did not include a library path")?;
        Ok(NativeRuntimeStartupLoadPlan {
            cache_mesh_version,
            native_runtime_id: load_plan.native_runtime_id,
            root: load_plan.root,
            selected_library_path,
            libraries: load_plan.libraries,
            source,
        })
    }

    fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
        crate::system::native_runtime_install::default_native_runtime_cache()
    }

    fn host_runtime_profile() -> HostRuntimeProfile {
        crate::system::native_runtime_install::host_runtime_profile()
    }

    fn default_install_options() -> NativeRuntimeInstallOptions {
        NativeRuntimeInstallOptions {
            mesh_version: crate::RELEASE_VERSION.to_string(),
            skippy_abi_version: Some(
                crate::system::native_runtime_install::current_skippy_abi_version(),
            ),
            selection: RuntimeSelection::Recommended,
            ..Default::default()
        }
    }

    async fn default_install_executor(
        options: NativeRuntimeInstallOptions,
    ) -> Result<NativeRuntimeInstallOutcome> {
        crate::system::native_runtime_install::install_native_runtime(options).await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use mesh_llm_native_runtime::{
            NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
        };
        use std::{
            fs,
            path::Path,
            sync::{Arc, Mutex},
        };

        fn write_runtime(dir: &Path, version: &str, id: &str) {
            write_runtime_with_manifest_mesh_version(dir, Some(version), id);
        }

        fn write_runtime_without_mesh_version(dir: &Path, id: &str) {
            write_runtime_with_manifest_mesh_version(dir, None, id);
        }

        fn write_runtime_with_manifest_mesh_version(dir: &Path, version: Option<&str>, id: &str) {
            let library_rel_path = test_library_rel_path();
            fs::create_dir_all(dir.join(library_rel_path.parent().unwrap())).unwrap();
            fs::write(dir.join(&library_rel_path), b"native runtime").unwrap();
            let manifest = NativeRuntimeManifest {
                runtime: NativeRuntimeArtifact {
                    id: id.to_string(),
                    mesh_version: version.map(ToString::to_string),
                    skippy_abi: "0.1.25".to_string(),
                    platform: NativeRuntimePlatform {
                        os: std::env::consts::OS.to_string(),
                        arch: std::env::consts::ARCH.to_string(),
                        target: None,
                    },
                    backend: NativeRuntimeBackend::cpu(),
                    rank: 0,
                    libraries: vec![library_rel_path.to_string_lossy().to_string()],
                    files: Default::default(),
                    tools: Default::default(),
                    url: None,
                    sha256: None,
                    signature: None,
                },
            };
            manifest.write_to_dir(dir).unwrap();
        }

        fn test_library_rel_path() -> PathBuf {
            let file = if cfg!(target_os = "windows") {
                "meshllm_ffi.dll"
            } else if cfg!(target_os = "macos") {
                "libmeshllm_ffi.dylib"
            } else {
                "libmeshllm_ffi.so"
            };
            PathBuf::from("lib").join(file)
        }

        fn test_install_options() -> NativeRuntimeInstallOptions {
            NativeRuntimeInstallOptions {
                mesh_version: "0.68.0".to_string(),
                allow_download: false,
                ..Default::default()
            }
        }

        #[test]
        fn sha_build_uses_release_cache_identity_for_installed_runtime_lookup() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let sha_build_version = "0.68.0+gAB131C";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                sha_build_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
            assert_eq!(
                plan.selected_library_path,
                runtime_dir.join(test_library_rel_path())
            );
            assert_eq!(
                plan.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
        }

        #[test]
        fn stale_pre_checksum_cache_entry_does_not_block_startup_plan() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.75.0";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            // Simulate a cache entry written by a pre-0.75 loader: the
            // manifest has no per-file checksums (issue #1162).
            let legacy_dir = cache.runtime_dir("0.74.0", runtime_id);
            let library_rel_path = test_library_rel_path();
            fs::create_dir_all(legacy_dir.join(library_rel_path.parent().unwrap())).unwrap();
            fs::write(legacy_dir.join(&library_rel_path), b"legacy runtime").unwrap();
            fs::write(
                legacy_dir.join("manifest.json"),
                format!(
                    r#"{{
  "runtime": {{
    "id": "{runtime_id}",
    "mesh_version": "0.74.0",
    "skippy_abi": "0.1.25",
    "platform": {{"os": "{os}", "arch": "{arch}"}},
    "backend": {{"kind": "cpu"}},
    "libraries": ["{library}"]
  }}
}}"#,
                    os = std::env::consts::OS,
                    arch = std::env::consts::ARCH,
                    library = library_rel_path.to_string_lossy().replace('\\', "/"),
                ),
            )
            .unwrap();

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                release_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("a stale pre-checksum cache entry must not block the valid runtime");

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
        }

        #[test]
        fn explicit_runtime_version_can_select_other_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let artifact_mesh_version = "0.69.0";
            let runtime_dir = cache.runtime_dir(artifact_mesh_version, runtime_id);
            write_runtime(&runtime_dir, artifact_mesh_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                artifact_mesh_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, artifact_mesh_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
        }

        #[test]
        fn default_startup_plan_rejects_other_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let artifact_mesh_version = "0.69.0";
            let runtime_dir = cache.runtime_dir(artifact_mesh_version, runtime_id);
            write_runtime(&runtime_dir, artifact_mesh_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap();

            assert!(plan.is_none());
        }

        #[test]
        fn startup_plan_rejects_installed_runtime_without_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = cache.runtime_dir("unknown", runtime_id);
            write_runtime_without_mesh_version(&runtime_dir, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap();

            assert!(plan.is_none());
        }

        #[test]
        fn startup_plan_can_represent_post_install_source_without_loading() {
            let temp = tempfile::tempdir().unwrap();
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = temp.path().join(runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);
            let load_plan = NativeRuntimeLoadPlan {
                mesh_version: release_version.to_string(),
                native_runtime_id: runtime_id.to_string(),
                root: runtime_dir.clone(),
                libraries: vec![runtime_dir.join(test_library_rel_path())],
            };

            let plan = startup_load_plan_from_installed(
                release_version.to_string(),
                load_plan,
                NativeRuntimePlanSource::PostInstall,
            )
            .unwrap();

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::PostInstall);
        }

        #[test]
        fn local_discovery_prefers_bundle_over_identical_cached_runtime() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            write_runtime(
                &cache.runtime_dir(release_version, runtime_id),
                release_version,
                runtime_id,
            );
            let bundled_runtime_dir = temp
                .path()
                .join("product")
                .join("native-runtimes")
                .join(runtime_id);
            write_runtime(&bundled_runtime_dir, release_version, runtime_id);

            let local_runtimes =
                crate::system::native_runtime_install::discover_local_native_runtimes(
                    std::slice::from_ref(&bundled_runtime_dir),
                    &cache,
                )
                .unwrap();
            let plan = resolve_local_native_runtime_plan(
                &local_runtimes,
                &HostRuntimeProfile::current_without_gpu_probe(),
                release_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected bundled runtime plan");

            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::LocalDiscovery);
            assert_eq!(plan.root, bundled_runtime_dir.canonicalize().unwrap());
        }

        #[test]
        fn disappeared_cache_entry_is_treated_as_cache_miss() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let manifest = NativeRuntimeReleaseManifest {
                mesh_version: release_version.to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: Vec::new(),
            };
            let artifact = NativeRuntimeArtifact {
                id: runtime_id.to_string(),
                mesh_version: Some(release_version.to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec![test_library_rel_path().to_string_lossy().to_string()],
                files: Default::default(),
                tools: Default::default(),
                url: None,
                sha256: None,
                signature: None,
            };

            let plan = load_plan_from_candidate(&cache, &manifest, artifact).unwrap();

            assert!(plan.is_none());
        }

        #[tokio::test]
        async fn cache_hit_skips_install_and_loads_cached_runtime_once() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(0_usize));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |_| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            *install_calls.lock().unwrap() += 1;
                            anyhow::bail!("install should not run on cache hit")
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    release_version.to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected cached runtime to load");

            assert_eq!(*install_calls.lock().unwrap(), 0);
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn bundled_runtime_precedes_identical_cache_entry_at_startup() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let cached_runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&cached_runtime_dir, release_version, runtime_id);
            let product_root = temp.path().join("mesh-bundle");
            let bundled_runtime_dir = product_root.join("native-runtimes").join(runtime_id);
            write_runtime(&bundled_runtime_dir, release_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(0_usize));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));
            let options_product_root = product_root.clone();
            let options_cache_root = cache.root().to_path_buf();

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                move || NativeRuntimeInstallOptions {
                    mesh_version: release_version.to_string(),
                    skippy_abi_version: Some("0.1.25".to_string()),
                    bundle_dirs: vec![options_product_root.clone()],
                    cache_dir: Some(options_cache_root.clone()),
                    allow_download: false,
                    ..Default::default()
                },
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |options| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            *install_calls.lock().unwrap() += 1;
                            crate::system::native_runtime_install::install_native_runtime(options)
                                .await
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    release_version.to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected bundled runtime to load");

            assert_eq!(*install_calls.lock().unwrap(), 1);
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![
                    bundled_runtime_dir
                        .canonicalize()
                        .unwrap()
                        .join(test_library_rel_path())
                ]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn cache_miss_installs_once_and_loads_post_install_runtime() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let bundle_dir = temp.path().join("bundle");
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let manifest_mesh_version = "0.68.0";
            write_runtime(&bundle_dir, manifest_mesh_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    let bundle_dir = bundle_dir.clone();
                    let cache = cache.clone();
                    move |mut options| {
                        let install_calls = Arc::clone(&install_calls);
                        let bundle_dir = bundle_dir.clone();
                        let cache = cache.clone();
                        async move {
                            install_calls.lock().unwrap().push(options.clone());
                            let source = options.bundle_dirs.pop().unwrap_or(bundle_dir.clone());
                            let runtime = cache.install_from_dir(&source)?;
                            Ok(NativeRuntimeInstallOutcome {
                                status: crate::system::native_runtime_install::NativeRuntimeInstallStatus::Installed,
                                runtime,
                                resolution: mesh_llm_native_runtime::NativeRuntimeResolution {
                                    source: mesh_llm_native_runtime::NativeRuntimeSource::Bundle {
                                        path: source,
                                    },
                                    selected: NativeRuntimeManifest::read_from_dir(&bundle_dir)?
                                        .runtime,
                                    evaluated: Vec::new(),
                                },
                                sources: Default::default(),
                            })
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    "0.68.0".to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected installed runtime to load");

            let recorded_options = install_calls.lock().unwrap();
            assert_eq!(recorded_options.len(), 1);
            assert_eq!(recorded_options[0].mesh_version, "0.68.0");
            assert_eq!(
                recorded_options[0].skippy_abi_version.as_deref(),
                Some("0.1.25")
            );
            assert_eq!(recorded_options[0].cache_dir.as_deref(), Some(cache.root()));
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![
                    cache
                        .runtime_dir(manifest_mesh_version, runtime_id)
                        .join(test_library_rel_path())
                ]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn cache_miss_install_failure_stops_startup_before_ffi_load() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(0_usize));

            let error = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |options| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            install_calls.lock().unwrap().push(options);
                            anyhow::bail!(
                                "no compatible native runtime found for Skippy ABI 0.1.25 on test/test"
                            )
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    "0.68.0".to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |_| {
                        *load_calls.lock().unwrap() += 1;
                        Ok(())
                    }
                },
            )
            .await
            .expect_err("missing native runtime should stop startup");

            let message = error.to_string();
            assert!(message.contains("no compatible MeshLLM native runtime"));
            assert!(message.contains("mesh-llm runtime install"));
            assert!(message.contains("mesh-llm runtime list --available"));
            assert_eq!(install_calls.lock().unwrap().len(), 1);
            assert_eq!(*load_calls.lock().unwrap(), 0);
        }

        #[test]
        fn startup_install_message_distinguishes_empty_and_nonempty_discovery() {
            assert_eq!(
                startup_install_message(true),
                "No compatible installed MeshLLM native runtime found; attempting one-shot startup install"
            );
            assert_eq!(
                startup_install_message(false),
                "Discovered native runtime bundles take precedence over installed runtimes; attempting one-shot startup install"
            );
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use dynamic::*;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub(crate) fn try_load_installed_native_runtime() -> anyhow::Result<Option<()>> {
    Ok(None)
}
