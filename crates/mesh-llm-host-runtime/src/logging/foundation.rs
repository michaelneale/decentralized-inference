//! Logging foundation: app-state root resolution, store/artifact initialization, health status.
//!
//! Wires validated logging config into host initialization without broadly instrumenting producers yet.
//! On startup (when enabled), creates the application-state layout expected by mesh-llm-log-store.
//! Follows fail-open policy: if the root is unwritable, disable logging and continue serving.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Platform-specific default root behavior, kept injectable for deterministic tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootPlatform {
    NonWindows,
    Windows,
}

impl RootPlatform {
    #[cfg(windows)]
    const CURRENT: Self = Self::Windows;

    #[cfg(not(windows))]
    const CURRENT: Self = Self::NonWindows;
}

/// A root could not be resolved without falling back to an unsafe platform location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AppStateRootResolutionError {
    WindowsLocalAppDataUnavailable,
}

impl std::fmt::Display for AppStateRootResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowsLocalAppDataUnavailable => {
                formatter.write_str("logging application-state root is unavailable")
            }
        }
    }
}

/// Resolved application-state layout for the logging subsystem.
pub struct LoggingFoundation {
    /// Whether logging was successfully initialized (true) or failed-open/disabled (false).
    healthy: AtomicBool,
    /// The resolved root directory for all logging application state.
    app_state_root: PathBuf,
    /// Path to the SQLite-backed log store directory (`<root>/store/`).
    store_dir: PathBuf,
    /// Path to the artifact file storage root (`<root>/artifacts/`).
    artifact_dir: PathBuf,
}

impl LoggingFoundation {
    /// Resolve and initialize the logging foundation from config.
    ///
    /// If `enabled` is false, returns a disabled (unhealthy) instance that creates NO files.
    /// If initialization fails (unwritable root), returns an unhealthy instance with a sanitized diagnostic.
    pub fn init(enabled: bool, application_state_root: Option<&PathBuf>) -> Self {
        if enabled {
            Self::init_enabled(application_state_root)
        } else {
            Self::disabled()
        }
    }

    fn init_enabled(application_state_root: Option<&PathBuf>) -> Self {
        let Some(app_state_root) = Self::resolve_root_or_fail_open(application_state_root) else {
            return Self::unavailable();
        };
        Self::initialize_layout(app_state_root)
    }

    fn resolve_root_or_fail_open(application_state_root: Option<&PathBuf>) -> Option<PathBuf> {
        resolve_app_state_root(application_state_root)
            .map_err(|error| {
                tracing::warn!(
                    reason = %error,
                    "Failed to resolve logging application-state root; disabling logging (fail-open)"
                );
            })
            .ok()
    }

    fn initialize_layout(app_state_root: PathBuf) -> Self {
        // Attempt to create the store and artifact directories (idempotent).
        let store_dir = app_state_root.join("store");
        let artifact_dir = app_state_root.join("artifacts");

        if !try_create_dirs(&app_state_root, &store_dir, &artifact_dir) {
            tracing::warn!(
                root = LOGGING_ROOT_LABEL,
                "Failed to create logging application-state directories; disabling logging (fail-open)"
            );
            return Self {
                healthy: AtomicBool::new(false),
                app_state_root,
                store_dir,
                artifact_dir,
            };
        }

        tracing::info!(
            root = LOGGING_ROOT_LABEL,
            "Logging application-state layout initialized"
        );

        Self {
            healthy: AtomicBool::new(true),
            app_state_root,
            store_dir,
            artifact_dir,
        }
    }

    /// Create a disabled foundation (logging.enabled = false). No files are created.
    pub fn disabled() -> Self {
        let dummy_path = PathBuf::from("/disabled");
        Self {
            healthy: AtomicBool::new(false),
            app_state_root: dummy_path.clone(),
            store_dir: dummy_path.join("store"),
            artifact_dir: dummy_path.join("artifacts"),
        }
    }

    /// Create an unhealthy foundation when no safe logging root can be resolved.
    fn unavailable() -> Self {
        let dummy_path = PathBuf::from("unavailable");
        Self {
            healthy: AtomicBool::new(false),
            app_state_root: dummy_path.clone(),
            store_dir: dummy_path.join("store"),
            artifact_dir: dummy_path.join("artifacts"),
        }
    }

    /// Whether logging is initialized and operational.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// The resolved application-state root directory.
    pub fn app_state_root(&self) -> &Path {
        &self.app_state_root
    }

    /// Path to the log store directory (contains `log_store.db`).
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Path to the artifact file storage root.
    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }

    /// Reinitialize after a restart-required config change (e.g., application_state_root changed).
    /// Returns a new LoggingFoundation if successful, None otherwise. Callers replace their handle with the returned value.
    pub fn reinit(new_root: Option<&PathBuf>) -> Option<Self> {
        let new_app_state_root = match resolve_app_state_root(new_root) {
            Ok(root) => root,
            Err(error) => {
                tracing::warn!(
                    reason = %error,
                    "Failed to resolve logging application-state root; keeping existing logging foundation"
                );
                return None;
            }
        };

        let store_dir = new_app_state_root.join("store");
        let artifact_dir = new_app_state_root.join("artifacts");

        if try_create_dirs(&new_app_state_root, &store_dir, &artifact_dir) {
            Some(Self {
                healthy: AtomicBool::new(true),
                app_state_root: new_app_state_root,
                store_dir,
                artifact_dir,
            })
        } else {
            None
        }
    }

    /// Sanitized health summary for diagnostics (no sensitive paths leaked).
    pub fn health_summary(&self) -> String {
        if !self.is_healthy() {
            return "logging disabled or unhealthy".to_string();
        }
        "logging healthy (storage ready)".to_string()
    }

    /// Check whether the store directory exists on disk (for idempotent startup verification).
    #[cfg(test)]
    pub fn store_dir_exists_on_disk(&self) -> bool {
        self.store_dir.exists() && self.store_dir.is_dir()
    }

    /// Check whether the artifact directory exists on disk.
    #[cfg(test)]
    pub fn artifact_dir_exists_on_disk(&self) -> bool {
        self.artifact_dir.exists() && self.artifact_dir.is_dir()
    }
}

/// Resolve the application-state root from config, environment, or platform defaults.
fn resolve_app_state_root(
    config_path: Option<&PathBuf>,
) -> Result<PathBuf, AppStateRootResolutionError> {
    resolve_app_state_root_with(
        config_path.map(PathBuf::as_path),
        std::env::var_os("MESH_LLM_DATA_DIR"),
        RootPlatform::CURRENT,
        local_app_data_dir,
    )
}

/// Resolve a root with explicit inputs so platform defaults and precedence are testable without
/// mutating process environment variables.
fn resolve_app_state_root_with<F>(
    config_path: Option<&Path>,
    data_dir: Option<OsString>,
    platform: RootPlatform,
    local_app_data: F,
) -> Result<PathBuf, AppStateRootResolutionError>
where
    F: FnOnce() -> Option<PathBuf>,
{
    if let Some(path) = config_path {
        return Ok(path.to_path_buf());
    }

    // Follow existing mesh-llm conventions for app data directories:
    // 1. MESH_LLM_DATA_DIR env var (highest priority, used by model-hf crate)
    // 2. platform default
    if let Some(env_path) = data_dir {
        return Ok(PathBuf::from(env_path).join("logging"));
    }

    match platform {
        RootPlatform::Windows => local_app_data()
            .map(|root| root.join("mesh-llm").join("logging"))
            .ok_or(AppStateRootResolutionError::WindowsLocalAppDataUnavailable),
        RootPlatform::NonWindows => Ok(dirs::home_dir()
            .map(|home| home.join(".mesh-llm").join("logging"))
            .unwrap_or_else(|| PathBuf::from("/tmp/mesh-llm/logging"))),
    }
}

#[cfg(windows)]
fn local_app_data_dir() -> Option<PathBuf> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStringExt;

    use windows_sys::Win32::Foundation::S_OK;
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::UI::Shell::{FOLDERID_LocalAppData, SHGetKnownFolderPath};

    let mut path = std::ptr::null_mut();
    let result =
        unsafe { SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, std::ptr::null_mut(), &mut path) };
    if result != S_OK || path.is_null() {
        if !path.is_null() {
            unsafe { CoTaskMemFree(path.cast::<c_void>()) };
        }
        return None;
    }

    let length = unsafe { windows_sys::Win32::Globalization::lstrlenW(path) } as usize;
    let local_app_data = unsafe { std::slice::from_raw_parts(path, length) };
    let root = PathBuf::from(OsString::from_wide(local_app_data));
    unsafe { CoTaskMemFree(path.cast::<c_void>()) };
    Some(root)
}

#[cfg(not(windows))]
fn local_app_data_dir() -> Option<PathBuf> {
    None
}

/// Attempt to create the app_state_root, store_dir, and artifact_dir. Returns true if all succeed.
fn try_create_dirs(root: &Path, store: &Path, artifacts: &Path) -> bool {
    for dir in [root, store, artifacts] {
        match std::fs::create_dir_all(dir) {
            Ok(()) => {}
            Err(e) => {
                tracing::debug!(
                    root = LOGGING_ROOT_LABEL,
                    error_kind = ?e.kind(),
                    "failed to create logging application-state directory"
                );
                return false;
            }
        }
    }

    // Verify the root is actually writable by attempting a quick write test.
    let test_file = root.join(".write_test");
    if std::fs::write(&test_file, "").is_err() {
        tracing::debug!(
            root = LOGGING_ROOT_LABEL,
            "logging application-state root exists but is not writable"
        );
        return false;
    }

    // Clean up the test file.
    let _ = std::fs::remove_file(&test_file);
    true
}

/// Stable diagnostic label for the logging application-state root.
///
/// Filesystem paths can contain user names, mounted-volume names, and other
/// sensitive deployment details. Diagnostics report this label rather than a
/// transformed path so non-home roots are safe too.
const LOGGING_ROOT_LABEL: &str = "logging-application-state";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn temp_root() -> PathBuf {
        static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "mesh-llm-log-foundation-{}-{}",
            std::process::id(),
            NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn init_enabled_creates_layout() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));
        assert!(foundation.is_healthy());
        assert_eq!(foundation.app_state_root(), &root);
        assert_eq!(foundation.store_dir(), &root.join("store"));
        assert_eq!(foundation.artifact_dir(), &root.join("artifacts"));
        assert!(foundation.store_dir_exists_on_disk());
        assert!(foundation.artifact_dir_exists_on_disk());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn init_disabled_creates_no_files() {
        let root = temp_root().join("disabled-test");
        // Root doesn't exist yet — disabled should NOT create it.
        assert!(!root.exists());

        let foundation = LoggingFoundation::init(false, Some(&root));
        assert!(!foundation.is_healthy());
        assert!(
            !root.exists(),
            "disabled logging must not create any directories"
        );

        // Cleanup (noop since nothing was created).
    }

    #[test]
    fn init_unwritable_root_fails_open() {
        let root = temp_root();
        std::fs::write(&root, "not a directory").unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));

        assert!(!foundation.is_healthy(), "a file root must fail open");
        std::fs::remove_file(root).unwrap();
    }

    #[test]
    fn init_idempotent_same_root() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let f1 = LoggingFoundation::init(true, Some(&root));
        assert!(f1.is_healthy());

        // Second initialization against the same root is safe (idempotent).
        let f2 = LoggingFoundation::init(true, Some(&root));
        assert!(f2.is_healthy());
        assert_eq!(f2.app_state_root(), &root);

        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(not(windows))]
    #[test]
    fn resolve_default_without_environment_override_falls_back_to_home() {
        let resolved =
            resolve_app_state_root_with(None, None, RootPlatform::NonWindows, || None).unwrap();

        if let Some(home) = dirs::home_dir() {
            assert!(resolved.starts_with(&home));
            assert!(resolved.ends_with(".mesh-llm/logging"));
        } else {
            // Fallback to /tmp when home is unavailable (rare in tests).
            assert_eq!(resolved, PathBuf::from("/tmp/mesh-llm/logging"));
        }
    }

    #[test]
    fn root_resolution_explicit_path_precedes_environment_and_windows_default() {
        let explicit = PathBuf::from("/configured/logging");
        let resolved = resolve_app_state_root_with(
            Some(&explicit),
            Some(OsString::from("/environment/data")),
            RootPlatform::Windows,
            || panic!("explicit configuration must not resolve LocalAppData"),
        )
        .unwrap();

        assert_eq!(resolved, explicit);
    }

    #[test]
    fn root_resolution_environment_precedes_windows_default() {
        let resolved = resolve_app_state_root_with(
            None,
            Some(OsString::from("/environment/data")),
            RootPlatform::Windows,
            || panic!("environment override must not resolve LocalAppData"),
        )
        .unwrap();

        assert_eq!(resolved, PathBuf::from("/environment/data/logging"));
    }

    #[test]
    fn root_resolution_uses_windows_local_app_data_without_home_fallback() {
        let resolved = resolve_app_state_root_with(None, None, RootPlatform::Windows, || {
            Some(PathBuf::from(r"C:\Users\Alice\AppData\Local"))
        })
        .unwrap();

        assert_eq!(
            resolved,
            PathBuf::from(r"C:\Users\Alice\AppData\Local")
                .join("mesh-llm")
                .join("logging")
        );
    }

    #[test]
    fn root_resolution_fails_when_windows_local_app_data_is_unavailable() {
        let result = resolve_app_state_root_with(None, None, RootPlatform::Windows, || None);

        assert_eq!(
            result,
            Err(AppStateRootResolutionError::WindowsLocalAppDataUnavailable)
        );
    }

    #[cfg(windows)]
    #[test]
    fn local_app_data_dir_resolves_known_folder() {
        assert!(local_app_data_dir().is_some());
    }

    #[test]
    fn health_summary_healthy() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));
        let summary = foundation.health_summary();
        assert!(summary.contains("logging healthy"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn health_summary_disabled() {
        let root = temp_root().join("disabled-summary");
        let foundation = LoggingFoundation::init(false, Some(&root));
        assert_eq!(foundation.health_summary(), "logging disabled or unhealthy");
    }

    #[test]
    fn health_summary_never_reveals_configured_root() {
        for root in [
            PathBuf::from("/Volumes/secret/alice/mesh-logs"),
            PathBuf::from(r"C:\\Users\\Alice\\AppData\\Local\\mesh-logs"),
        ] {
            let foundation = LoggingFoundation {
                healthy: AtomicBool::new(true),
                store_dir: root.join("store"),
                artifact_dir: root.join("artifacts"),
                app_state_root: root,
            };
            let summary = foundation.health_summary();
            assert_eq!(summary, "logging healthy (storage ready)");
            assert!(!summary.contains("secret"));
            assert!(!summary.contains("Alice"));
        }
    }
}
