use anyhow::{Result, bail};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const HUB_CACHE_ENV: &str = "HF_HUB_CACHE";
const HUB_CACHE_ALIAS_ENV: &str = "HUGGINGFACE_HUB_CACHE";
const HF_HOME_ENV: &str = "HF_HOME";
const XET_CACHE_ENV: &str = "HF_XET_CACHE";
const XDG_CACHE_HOME_ENV: &str = "XDG_CACHE_HOME";
const MESH_LLM_DATA_DIR_ENV: &str = "MESH_LLM_DATA_DIR";
const WRITE_PROBE_PREFIX: &str = ".mesh-llm-write-probe";

static WRITE_PROBE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DownloadDirectoryKind {
    HuggingFaceHub,
    HuggingFaceXet,
}

impl fmt::Display for DownloadDirectoryKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HuggingFaceHub => formatter.write_str("Hugging Face Hub cache"),
            Self::HuggingFaceXet => formatter.write_str("Hugging Face Xet cache"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadDirectoryFallback {
    pub kind: DownloadDirectoryKind,
    pub requested: PathBuf,
    pub selected: PathBuf,
    pub error: String,
    pub raw_os_error: Option<i32>,
}

impl fmt::Display for DownloadDirectoryFallback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {} is not writable ({}); using {}. Set {} or {} to choose a different writable location.",
            self.kind,
            self.requested.display(),
            self.error,
            self.selected.display(),
            match self.kind {
                DownloadDirectoryKind::HuggingFaceHub => HUB_CACHE_ENV,
                DownloadDirectoryKind::HuggingFaceXet => XET_CACHE_ENV,
            },
            MESH_LLM_DATA_DIR_ENV,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDownloadDirectories {
    pub hub_cache: PathBuf,
    pub xet_cache: PathBuf,
    pub fallbacks: Vec<DownloadDirectoryFallback>,
}

impl PreparedDownloadDirectories {
    /// Apply the validated directories to the process environment.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no other threads are reading or writing
    /// process environment variables. The shipped binary calls this before it
    /// constructs its Tokio runtime.
    pub unsafe fn apply_to_process_environment(&self) {
        // SAFETY: Upheld by the caller as documented above.
        unsafe {
            std::env::set_var(HUB_CACHE_ENV, &self.hub_cache);
            std::env::set_var(XET_CACHE_ENV, &self.xet_cache);
        }
    }
}

pub fn huggingface_hub_cache_dir() -> PathBuf {
    requested_hub_cache_dir()
}

pub fn huggingface_xet_cache_dir() -> PathBuf {
    requested_xet_cache_dir()
}

pub fn mesh_llm_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .unwrap_or_else(|| std::env::temp_dir().join("mesh-llm-cache"))
        .join("mesh-llm")
}

pub fn prepare_download_directories() -> Result<PreparedDownloadDirectories> {
    prepare_download_directories_in(
        requested_hub_cache_dir(),
        requested_xet_cache_dir(),
        fallback_data_roots(),
    )
}

pub fn download_cache_diagnostic() -> String {
    download_cache_diagnostic_for(&huggingface_hub_cache_dir())
}

pub fn download_cache_diagnostic_for(hub_cache: &Path) -> String {
    format!(
        "Hub cache: {}; Xet cache: {}. Ensure both paths are writable, or set {} to a writable application-data directory",
        hub_cache.display(),
        huggingface_xet_cache_dir().display(),
        MESH_LLM_DATA_DIR_ENV,
    )
}

fn requested_hub_cache_dir() -> PathBuf {
    env_path(HUB_CACHE_ENV)
        .or_else(|| env_path(HUB_CACHE_ALIAS_ENV))
        .or_else(|| env_path(HF_HOME_ENV).map(|path| path.join("hub")))
        .or_else(|| env_path(XDG_CACHE_HOME_ENV).map(|path| path.join("huggingface").join("hub")))
        .or_else(|| dirs::cache_dir().map(|path| path.join("huggingface").join("hub")))
        .unwrap_or_else(|| default_data_root().join("huggingface").join("hub"))
}

fn requested_xet_cache_dir() -> PathBuf {
    env_path(XET_CACHE_ENV)
        .or_else(|| env_path(HF_HOME_ENV).map(|path| path.join("xet")))
        .or_else(|| env_path(XDG_CACHE_HOME_ENV).map(|path| path.join("huggingface").join("xet")))
        .or_else(|| dirs::cache_dir().map(|path| path.join("huggingface").join("xet")))
        .unwrap_or_else(|| default_data_root().join("huggingface").join("xet"))
}

fn env_path(key: &str) -> Option<PathBuf> {
    let value = std::env::var_os(key)?;
    nonempty_path(value)
}

fn nonempty_path(value: OsString) -> Option<PathBuf> {
    if value.to_str().is_some_and(|value| value.trim().is_empty()) {
        return None;
    }
    let path = PathBuf::from(value);
    (!path.as_os_str().is_empty()).then_some(path)
}

fn default_data_root() -> PathBuf {
    env_path(MESH_LLM_DATA_DIR_ENV)
        .or_else(|| dirs::data_local_dir().map(|path| path.join("mesh-llm")))
        .or_else(|| dirs::home_dir().map(|path| path.join(".mesh-llm").join("data")))
        .unwrap_or_else(|| std::env::temp_dir().join("mesh-llm-data"))
}

fn fallback_data_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(path) = env_path(MESH_LLM_DATA_DIR_ENV) {
        roots.push(path);
    }
    if let Some(path) = dirs::data_local_dir() {
        roots.push(path.join("mesh-llm"));
    }
    if let Some(path) = dirs::home_dir() {
        roots.push(path.join(".mesh-llm").join("data"));
    }
    roots.push(std::env::temp_dir().join("mesh-llm-data"));
    deduplicate_paths(roots)
}

fn deduplicate_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

fn prepare_download_directories_in(
    requested_hub: PathBuf,
    requested_xet: PathBuf,
    fallback_roots: Vec<PathBuf>,
) -> Result<PreparedDownloadDirectories> {
    let (hub_cache, hub_fallback) = prepare_directory(
        DownloadDirectoryKind::HuggingFaceHub,
        requested_hub,
        fallback_roots
            .iter()
            .map(|root| root.join("huggingface").join("hub")),
    )?;
    let (xet_cache, xet_fallback) = prepare_directory(
        DownloadDirectoryKind::HuggingFaceXet,
        requested_xet,
        fallback_roots
            .iter()
            .map(|root| root.join("huggingface").join("xet")),
    )?;
    let fallbacks = hub_fallback.into_iter().chain(xet_fallback).collect();
    Ok(PreparedDownloadDirectories {
        hub_cache,
        xet_cache,
        fallbacks,
    })
}

fn prepare_directory(
    kind: DownloadDirectoryKind,
    requested: PathBuf,
    fallbacks: impl IntoIterator<Item = PathBuf>,
) -> Result<(PathBuf, Option<DownloadDirectoryFallback>)> {
    let requested_error = match probe_writable_directory(&requested) {
        Ok(()) => return Ok((requested, None)),
        Err(error) => error,
    };
    let mut attempts = vec![format!("{}: {}", requested.display(), requested_error)];
    for fallback in deduplicate_paths(fallbacks.into_iter().collect()) {
        if fallback == requested {
            continue;
        }
        match probe_writable_directory(&fallback) {
            Ok(()) => {
                return Ok((
                    fallback.clone(),
                    Some(DownloadDirectoryFallback {
                        kind,
                        requested,
                        selected: fallback,
                        error: requested_error.to_string(),
                        raw_os_error: requested_error.raw_os_error(),
                    }),
                ));
            }
            Err(error) => attempts.push(format!("{}: {}", fallback.display(), error)),
        }
    }
    bail!(
        "no writable {kind} directory was available; tried {}",
        attempts.join("; ")
    )
}

fn probe_writable_directory(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    let sequence = WRITE_PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let probe = path.join(format!(
        "{WRITE_PROBE_PREFIX}-{}-{sequence}",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)?;
    file.write_all(b"mesh-llm")?;
    drop(file);
    std::fs::remove_file(&probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_requested_directories_are_preserved() {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("requested-hub");
        let xet = temp.path().join("requested-xet");

        let prepared = prepare_download_directories_in(
            hub.clone(),
            xet.clone(),
            vec![temp.path().join("fallback")],
        )
        .unwrap();

        assert_eq!(prepared.hub_cache, hub);
        assert_eq!(prepared.xet_cache, xet);
        assert!(prepared.fallbacks.is_empty());
    }

    #[test]
    fn non_directory_cache_path_uses_writable_data_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let blocked_parent = temp.path().join("not-a-directory");
        std::fs::write(&blocked_parent, b"file").unwrap();
        let fallback = temp.path().join("data");

        let prepared = prepare_download_directories_in(
            blocked_parent.join("hub"),
            blocked_parent.join("xet"),
            vec![fallback.clone()],
        )
        .unwrap();

        assert_eq!(prepared.hub_cache, fallback.join("huggingface").join("hub"));
        assert_eq!(prepared.xet_cache, fallback.join("huggingface").join("xet"));
        assert_eq!(prepared.fallbacks.len(), 2);
        assert!(
            prepared
                .fallbacks
                .iter()
                .all(|warning| !warning.error.is_empty())
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn operating_system_read_only_directory_uses_writable_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let read_only_root = if cfg!(target_os = "macos") {
            PathBuf::from("/System")
        } else {
            PathBuf::from("/sys")
        };
        let requested_hub = read_only_root.join("mesh-llm-issue-980").join("hub");
        let writable_xet = temp.path().join("xet");

        let prepared = prepare_download_directories_in(
            requested_hub.clone(),
            writable_xet.clone(),
            vec![temp.path().join("data")],
        )
        .unwrap();

        assert_ne!(prepared.hub_cache, requested_hub);
        assert_eq!(prepared.xet_cache, writable_xet);
        assert_eq!(prepared.fallbacks.len(), 1);
        let warning = &prepared.fallbacks[0];
        assert!(matches!(
            warning.raw_os_error,
            Some(30) | Some(13) | Some(1)
        ));
    }

    #[test]
    fn empty_environment_path_is_ignored() {
        assert_eq!(nonempty_path(OsString::new()), None);
        assert_eq!(nonempty_path(OsString::from("   ")), None);
    }

    #[test]
    fn read_only_filesystem_warning_includes_os_error_and_recovery_path() {
        // Raw OS error 30 is EROFS on POSIX but ERROR_READ_FAULT on Windows, so
        // assert the contract (the OS error text is surfaced verbatim) rather
        // than the POSIX wording.
        let os_error = std::io::Error::from_raw_os_error(30).to_string();
        let warning = DownloadDirectoryFallback {
            kind: DownloadDirectoryKind::HuggingFaceXet,
            requested: PathBuf::from("/"),
            selected: PathBuf::from("/writable/data/huggingface/xet"),
            error: os_error.clone(),
            raw_os_error: Some(30),
        };

        let message = warning.to_string();
        assert!(message.contains(&os_error));
        assert!(message.contains("/writable/data/huggingface/xet"));
        assert!(message.contains("HF_XET_CACHE"));
        assert!(message.contains("MESH_LLM_DATA_DIR"));
    }
}
