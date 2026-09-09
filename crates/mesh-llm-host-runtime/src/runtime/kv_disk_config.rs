use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use anyhow::{Context, Result, bail};
use mesh_llm_config::{
    DEFAULT_KV_DISK_MINIMUM_FREE_MIB, KvDiskTierMode, MIN_KV_DISK_MINIMUM_FREE_MIB, MeshConfig,
    parse_iec_size,
};
use serde::Serialize;
use skippy_cache::{L3CacheManager, StoreLimits};

use super::RuntimeOptions;

const MIB: u64 = 1024 * 1024;
const LEGACY_DEFAULT_BUDGET_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const AUTO_MAX_BUDGET_BYTES: u64 = 64 * 1024 * 1024 * 1024;

static NODE_KV_DISK_CACHE: OnceLock<RwLock<NodeKvDiskCache>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum KvDiskConfigSource {
    Default,
    Config,
    Environment,
    Cli,
    LegacyEnvironment,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct KvDiskConfigSources {
    pub(crate) mode: KvDiskConfigSource,
    pub(crate) directory: KvDiskConfigSource,
    pub(crate) budget: KvDiskConfigSource,
    pub(crate) minimum_free: KvDiskConfigSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedKvDiskConfig {
    pub(crate) mode: KvDiskTierMode,
    pub(crate) directory: PathBuf,
    /// Fixed-mode cap. Auto mode resolves this from live filesystem facts
    /// immediately before the node manager opens the root.
    pub(crate) budget_bytes: Option<u64>,
    pub(crate) minimum_free_bytes: u64,
    pub(crate) sources: KvDiskConfigSources,
    pub(crate) warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeKvDiskCache {
    pub(crate) configured: ResolvedKvDiskConfig,
    pub(crate) manager: Option<L3CacheManager>,
    runtime_options: RuntimeOptions,
}

pub(crate) struct KvDiskLiveRollback {
    configured: ResolvedKvDiskConfig,
    limits: Option<StoreLimits>,
}

impl ResolvedKvDiskConfig {
    pub(crate) fn enabled(&self) -> bool {
        self.mode != KvDiskTierMode::Off
    }
}

pub(crate) fn resolve_kv_disk_config(
    config: &MeshConfig,
    options: &RuntimeOptions,
) -> Result<ResolvedKvDiskConfig> {
    resolve_kv_disk_config_with_env(config, options, |name| std::env::var_os(name))
}

/// Resolve the public/legacy configuration once and acquire the sole cache
/// manager for this node process. Store availability is fail-open for
/// inference: an unusable cache is reported as a warning and cold prefill
/// remains available.
pub(crate) fn configure_node_kv_disk_cache(
    config: &MeshConfig,
    options: &RuntimeOptions,
) -> Result<NodeKvDiskCache> {
    if let Some(cache) = NODE_KV_DISK_CACHE.get() {
        let snapshot = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        return Ok(snapshot);
    }
    let mut configured = resolve_kv_disk_config(config, options)?;
    let mut manager = None;
    if configured.enabled() {
        match acquire_manager(&mut configured) {
            Ok(acquired) => manager = acquired,
            Err(error) => configured.warnings.push(format!(
                "disk prompt cache is unavailable; inference will use cold prefill: {error:#}"
            )),
        }
    }
    let snapshot = NodeKvDiskCache {
        configured,
        manager,
        runtime_options: options.clone(),
    };
    let _ = NODE_KV_DISK_CACHE.set(RwLock::new(snapshot.clone()));
    Ok(snapshot)
}

pub(crate) fn node_kv_disk_manager() -> Option<L3CacheManager> {
    NODE_KV_DISK_CACHE.get().and_then(|cache| {
        cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .manager
            .clone()
    })
}

pub(crate) fn node_kv_disk_cache() -> Option<NodeKvDiskCache> {
    NODE_KV_DISK_CACHE.get().map(|cache| {
        cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    })
}

pub(crate) fn apply_live_kv_disk_limits(config: &MeshConfig) -> Result<KvDiskLiveRollback> {
    let cache = NODE_KV_DISK_CACHE
        .get()
        .context("node disk prompt-cache runtime is not initialized")?;
    let mut cache = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut resolved = resolve_kv_disk_config(config, &cache.runtime_options)?;
    let previous = cache.configured.clone();
    let previous_limits = cache.manager.as_ref().map(L3CacheManager::limits);
    preserve_restart_fields(&previous, &mut resolved);
    if let Some(manager) = cache.manager.as_ref() {
        let current = manager.limits();
        let budget = match resolved.mode {
            KvDiskTierMode::Fixed => resolved.budget_bytes.unwrap_or(current.budget_bytes),
            KvDiskTierMode::Auto => auto_budget_bytes(manager.root(), resolved.minimum_free_bytes)?
                .min(current.budget_bytes),
            KvDiskTierMode::Off => current.budget_bytes,
        };
        resolved.budget_bytes = Some(budget);
        manager.update_limits(StoreLimits::new(budget, resolved.minimum_free_bytes))?;
    }
    cache.configured = resolved;
    Ok(KvDiskLiveRollback {
        configured: previous,
        limits: previous_limits,
    })
}

fn preserve_restart_fields(previous: &ResolvedKvDiskConfig, next: &mut ResolvedKvDiskConfig) {
    if next.mode != previous.mode {
        next.mode = previous.mode;
        next.sources.mode = previous.sources.mode;
        next.budget_bytes = previous.budget_bytes;
        next.sources.budget = previous.sources.budget;
    }
    if next.directory != previous.directory {
        next.directory = previous.directory.clone();
        next.sources.directory = previous.sources.directory;
    }
}

pub(crate) fn restore_live_kv_disk_limits(rollback: KvDiskLiveRollback) {
    let Some(cache) = NODE_KV_DISK_CACHE.get() else {
        return;
    };
    let mut cache = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let (Some(manager), Some(limits)) = (cache.manager.as_ref(), rollback.limits) {
        let _ = manager.update_limits(limits);
    }
    cache.configured = rollback.configured;
}

fn acquire_manager(config: &mut ResolvedKvDiskConfig) -> Result<Option<L3CacheManager>> {
    std::fs::create_dir_all(&config.directory).with_context(|| {
        format!(
            "create disk prompt-cache root {}",
            config.directory.display()
        )
    })?;
    let budget_bytes = match config.mode {
        KvDiskTierMode::Off => return Ok(None),
        KvDiskTierMode::Fixed => config
            .budget_bytes
            .context("fixed disk prompt-cache mode has no budget")?,
        KvDiskTierMode::Auto => {
            let budget = auto_budget_bytes(&config.directory, config.minimum_free_bytes)?;
            config.budget_bytes = Some(budget);
            if budget == 0 {
                config.warnings.push(
                    "automatic disk prompt-cache budget is zero after the minimum-free reserve; disk cache remains disabled"
                        .to_string(),
                );
                return Ok(None);
            }
            budget
        }
    };
    Ok(Some(L3CacheManager::acquire(
        &config.directory,
        StoreLimits::new(budget_bytes, config.minimum_free_bytes),
    )?))
}

fn auto_budget_bytes(root: &Path, minimum_free_bytes: u64) -> Result<u64> {
    let available = skippy_cache::fsinfo::available_bytes(root)?;
    let managed = managed_root_bytes(root)?;
    Ok(auto_budget_from_space(
        available,
        managed,
        minimum_free_bytes,
    ))
}

fn auto_budget_from_space(available: u64, managed: u64, minimum_free: u64) -> u64 {
    let capacity_basis = available.saturating_add(managed);
    let twenty_percent = capacity_basis / 5;
    let allocatable = available
        .saturating_sub(minimum_free)
        .saturating_add(managed);
    twenty_percent.min(allocatable).min(AUTO_MAX_BUDGET_BYTES)
}

fn managed_root_bytes(root: &Path) -> Result<u64> {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("read cache directory {}", directory.display()))?
        {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() {
                bail!(
                    "disk prompt-cache root contains a symlink: {}",
                    entry.path().display()
                );
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

fn resolve_kv_disk_config_with_env(
    config: &MeshConfig,
    options: &RuntimeOptions,
    env: impl Fn(&str) -> Option<OsString>,
) -> Result<ResolvedKvDiskConfig> {
    let disk = &config.runtime.kv_cache.disk;
    let mut warnings = Vec::new();

    let mesh_home = match env("MESH_LLM_HOME") {
        Some(value) => absolute_path(value, "MESH_LLM_HOME")?,
        None => dirs::home_dir()
            .context("cannot determine home directory for the disk prompt cache")?
            .join(".mesh-llm"),
    };
    let mut directory = mesh_home.join("kv-cache");
    let mut directory_source = KvDiskConfigSource::Default;
    if let Some(value) = disk.directory.as_ref() {
        directory = value.clone();
        directory_source = KvDiskConfigSource::Config;
    }

    let mut mode = disk.mode.unwrap_or_default();
    let mut mode_source = if disk.mode.is_some() {
        KvDiskConfigSource::Config
    } else {
        KvDiskConfigSource::Default
    };
    let mut budget_bytes = disk.budget_mib.map(|mib| mib.saturating_mul(MIB));
    let mut budget_source = if disk.budget_mib.is_some() {
        KvDiskConfigSource::Config
    } else {
        KvDiskConfigSource::Default
    };
    let mut minimum_free_bytes = disk
        .minimum_free_mib
        .unwrap_or(DEFAULT_KV_DISK_MINIMUM_FREE_MIB)
        .saturating_mul(MIB);
    let mut minimum_free_source = if disk.minimum_free_mib.is_some() {
        KvDiskConfigSource::Config
    } else {
        KvDiskConfigSource::Default
    };

    let public_env_disk = env_utf8(&env, "MESH_LLM_KV_CACHE_DISK")?;
    let public_env_directory = env_path(&env, "MESH_LLM_KV_CACHE_DISK_DIR")?;
    let public_env_minimum = env_utf8(&env, "MESH_LLM_KV_CACHE_MIN_FREE")?;
    let public_mode =
        disk.mode.is_some() || public_env_disk.is_some() || options.kv_cache_disk.is_some();
    let public_directory = disk.directory.is_some()
        || public_env_directory.is_some()
        || options.kv_cache_disk_dir.is_some();
    let public_budget = disk.budget_mib.is_some() || public_mode;

    if let Some(legacy_directory) = env_path(&env, "SKIPPY_L3_DIR")? {
        let mut used_legacy = false;
        if !public_directory {
            directory = require_absolute(legacy_directory, "SKIPPY_L3_DIR")?;
            directory_source = KvDiskConfigSource::LegacyEnvironment;
            used_legacy = true;
        }
        if !public_mode {
            mode = KvDiskTierMode::Fixed;
            mode_source = KvDiskConfigSource::LegacyEnvironment;
            used_legacy = true;
        }
        if !public_budget {
            budget_source = KvDiskConfigSource::LegacyEnvironment;
            budget_bytes = match env_utf8(&env, "SKIPPY_L3_BUDGET_BYTES")? {
                Some(value) => match value.parse::<u64>() {
                    Ok(0) => {
                        warnings.push(
                            "SKIPPY_L3_BUDGET_BYTES=0 no longer means unbounded; using the legacy 32GiB default"
                                .to_string(),
                        );
                        Some(LEGACY_DEFAULT_BUDGET_BYTES)
                    }
                    Ok(value) => Some(value),
                    Err(_) => bail!("SKIPPY_L3_BUDGET_BYTES must be a positive byte count"),
                },
                None => Some(LEGACY_DEFAULT_BUDGET_BYTES),
            };
            used_legacy = true;
        }
        if used_legacy {
            warnings.push(
                "SKIPPY_L3_DIR and SKIPPY_L3_BUDGET_BYTES are deprecated; use runtime.kv_cache.disk or MESH_LLM_KV_CACHE_*"
                    .to_string(),
            );
        }
    } else if env("SKIPPY_L3_BUDGET_BYTES").is_some() {
        warnings.push(
            "ignoring deprecated SKIPPY_L3_BUDGET_BYTES because SKIPPY_L3_DIR is not set"
                .to_string(),
        );
    }

    if let Some(value) = public_env_disk.as_deref() {
        (mode, budget_bytes) = parse_mode_or_size(value, "MESH_LLM_KV_CACHE_DISK")?;
        mode_source = KvDiskConfigSource::Environment;
        budget_source = KvDiskConfigSource::Environment;
    }
    if let Some(value) = public_env_directory {
        directory = require_absolute(value, "MESH_LLM_KV_CACHE_DISK_DIR")?;
        directory_source = KvDiskConfigSource::Environment;
    }
    if let Some(value) = public_env_minimum.as_deref() {
        minimum_free_bytes = parse_minimum_free(value, "MESH_LLM_KV_CACHE_MIN_FREE")?;
        minimum_free_source = KvDiskConfigSource::Environment;
    }

    if let Some(value) = options.kv_cache_disk.as_deref() {
        (mode, budget_bytes) = parse_mode_or_size(value, "--kv-cache-disk")?;
        mode_source = KvDiskConfigSource::Cli;
        budget_source = KvDiskConfigSource::Cli;
    }
    if let Some(value) = options.kv_cache_disk_dir.as_ref() {
        directory = require_absolute(value.clone(), "--kv-cache-disk-dir")?;
        directory_source = KvDiskConfigSource::Cli;
    }
    if let Some(value) = options.kv_cache_min_free.as_deref() {
        minimum_free_bytes = parse_minimum_free(value, "--kv-cache-min-free")?;
        minimum_free_source = KvDiskConfigSource::Cli;
    }

    match mode {
        KvDiskTierMode::Fixed if budget_bytes.is_none() => {
            bail!("fixed disk prompt-cache mode requires a positive budget")
        }
        KvDiskTierMode::Off | KvDiskTierMode::Auto => budget_bytes = None,
        KvDiskTierMode::Fixed => {}
    }

    Ok(ResolvedKvDiskConfig {
        mode,
        directory,
        budget_bytes,
        minimum_free_bytes,
        sources: KvDiskConfigSources {
            mode: mode_source,
            directory: directory_source,
            budget: budget_source,
            minimum_free: minimum_free_source,
        },
        warnings,
    })
}

fn parse_mode_or_size(value: &str, name: &str) -> Result<(KvDiskTierMode, Option<u64>)> {
    match value.trim() {
        "off" => Ok((KvDiskTierMode::Off, None)),
        "auto" => Ok((KvDiskTierMode::Auto, None)),
        size => parse_iec_size(size)
            .map(|bytes| (KvDiskTierMode::Fixed, Some(bytes)))
            .with_context(|| format!("invalid {name} value {value:?}")),
    }
}

fn parse_minimum_free(value: &str, name: &str) -> Result<u64> {
    let bytes = parse_iec_size(value).with_context(|| format!("invalid {name} value {value:?}"))?;
    let minimum = MIN_KV_DISK_MINIMUM_FREE_MIB.saturating_mul(MIB);
    if bytes < minimum {
        bail!("{name} must preserve at least {MIN_KV_DISK_MINIMUM_FREE_MIB}MiB");
    }
    Ok(bytes)
}

fn env_utf8(env: &impl Fn(&str) -> Option<OsString>, name: &str) -> Result<Option<String>> {
    env(name)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("{name} must contain valid UTF-8"))
        })
        .transpose()
}

fn env_path(env: &impl Fn(&str) -> Option<OsString>, name: &str) -> Result<Option<PathBuf>> {
    Ok(env(name).map(PathBuf::from))
}

fn absolute_path(value: OsString, name: &str) -> Result<PathBuf> {
    require_absolute(PathBuf::from(value), name)
}

fn require_absolute(path: PathBuf, name: &str) -> Result<PathBuf> {
    if !is_absolute_on_supported_host(&path) {
        bail!("{name} must be an absolute path: {}", path.display());
    }
    Ok(path)
}

fn is_absolute_on_supported_host(path: &Path) -> bool {
    if path.is_absolute() {
        return true;
    }
    if !cfg!(windows) {
        // `C:\cache` is a relative path on this host; treating it as absolute
        // would create the cache root under the working directory.
        return false;
    }
    let rendered = path.to_string_lossy();
    rendered.as_bytes().get(1) == Some(&b':')
        && rendered
            .as_bytes()
            .get(2)
            .is_some_and(|byte| matches!(byte, b'/' | b'\\'))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn resolve(
        config: &str,
        options: RuntimeOptions,
        env: &[(&str, &str)],
    ) -> Result<ResolvedKvDiskConfig> {
        let config = mesh_llm_config::parse_config_toml(config)?;
        let env = env
            .iter()
            .map(|(key, value)| ((*key).to_string(), OsString::from(value)))
            .collect::<BTreeMap<_, _>>();
        resolve_kv_disk_config_with_env(&config, &options, |name| env.get(name).cloned())
    }

    #[test]
    fn precedence_is_field_by_field_and_reports_each_source() {
        let options = RuntimeOptions {
            kv_cache_disk: Some("48GiB".to_string()),
            kv_cache_min_free: Some("20GiB".to_string()),
            ..RuntimeOptions::default()
        };
        let resolved = resolve(
            "[runtime.kv_cache.disk]\nmode='fixed'\ndirectory='/config/cache'\nbudget_mib=32768\nminimum_free_mib=12288\n",
            options,
            &[
                ("MESH_LLM_KV_CACHE_DISK", "auto"),
                ("MESH_LLM_KV_CACHE_DISK_DIR", "/env/cache"),
            ],
        )
        .unwrap();

        assert_eq!(resolved.mode, KvDiskTierMode::Fixed);
        assert_eq!(resolved.budget_bytes, Some(48 * 1024_u64.pow(3)));
        assert_eq!(resolved.directory, PathBuf::from("/env/cache"));
        assert_eq!(resolved.minimum_free_bytes, 20 * 1024_u64.pow(3));
        assert_eq!(resolved.sources.mode, KvDiskConfigSource::Cli);
        assert_eq!(resolved.sources.budget, KvDiskConfigSource::Cli);
        assert_eq!(resolved.sources.directory, KvDiskConfigSource::Environment);
        assert_eq!(resolved.sources.minimum_free, KvDiskConfigSource::Cli);
    }

    #[test]
    fn legacy_environment_is_fallback_only_and_zero_is_bounded() {
        let resolved = resolve(
            "",
            RuntimeOptions::default(),
            &[
                ("SKIPPY_L3_DIR", "/legacy/cache"),
                ("SKIPPY_L3_BUDGET_BYTES", "0"),
            ],
        )
        .unwrap();
        assert_eq!(resolved.mode, KvDiskTierMode::Fixed);
        assert_eq!(resolved.budget_bytes, Some(LEGACY_DEFAULT_BUDGET_BYTES));
        assert_eq!(resolved.sources.mode, KvDiskConfigSource::LegacyEnvironment);
        assert_eq!(resolved.warnings.len(), 2);

        let public = resolve(
            "[runtime.kv_cache.disk]\nmode='off'\n",
            RuntimeOptions::default(),
            &[("SKIPPY_L3_DIR", "/legacy/cache")],
        )
        .unwrap();
        assert_eq!(public.mode, KvDiskTierMode::Off);
        assert_eq!(public.sources.mode, KvDiskConfigSource::Config);
        assert_eq!(
            public.directory,
            PathBuf::from("/legacy/cache"),
            "legacy directory remains a field-level fallback"
        );

        let minimum_only = resolve(
            "[runtime.kv_cache.disk]\nminimum_free_mib=20480\n",
            RuntimeOptions::default(),
            &[("SKIPPY_L3_DIR", "/legacy/cache")],
        )
        .unwrap();
        assert_eq!(minimum_only.mode, KvDiskTierMode::Fixed);
        assert_eq!(minimum_only.budget_bytes, Some(LEGACY_DEFAULT_BUDGET_BYTES));
        assert_eq!(
            minimum_only.sources.minimum_free,
            KvDiskConfigSource::Config
        );
    }

    #[test]
    fn ambiguous_sizes_and_relative_paths_fail_closed() {
        let options = RuntimeOptions {
            kv_cache_disk: Some("32".to_string()),
            ..RuntimeOptions::default()
        };
        assert!(resolve("", options, &[]).is_err());

        let options = RuntimeOptions {
            kv_cache_disk_dir: Some(PathBuf::from("relative/cache")),
            ..RuntimeOptions::default()
        };
        assert!(resolve("", options, &[]).is_err());

        let options = RuntimeOptions {
            kv_cache_min_free: Some("512MiB".to_string()),
            ..RuntimeOptions::default()
        };
        assert!(resolve("", options, &[]).is_err());
    }

    #[test]
    fn live_apply_preserves_restart_only_mode_directory_and_coupled_budget() {
        let previous = resolve(
            "[runtime.kv_cache.disk]\nmode='fixed'\ndirectory='/old/cache'\nbudget_mib=32768\nminimum_free_mib=16384\n",
            RuntimeOptions::default(),
            &[],
        )
        .unwrap();
        let mut next = resolve(
            "[runtime.kv_cache.disk]\nmode='auto'\ndirectory='/new/cache'\nminimum_free_mib=20480\n",
            RuntimeOptions::default(),
            &[],
        )
        .unwrap();

        preserve_restart_fields(&previous, &mut next);

        assert_eq!(next.mode, KvDiskTierMode::Fixed);
        assert_eq!(next.directory, PathBuf::from("/old/cache"));
        assert_eq!(next.budget_bytes, Some(32 * 1024_u64.pow(3)));
        assert_eq!(next.minimum_free_bytes, 20 * 1024_u64.pow(3));
    }

    #[test]
    fn auto_budget_uses_available_plus_managed_as_stable_capacity_basis() {
        let gib = 1024_u64.pow(3);
        assert_eq!(auto_budget_from_space(100 * gib, 0, 16 * gib), 20 * gib);
        assert_eq!(
            auto_budget_from_space(84 * gib, 16 * gib, 16 * gib),
            20 * gib
        );
        assert_eq!(auto_budget_from_space(8 * gib, 0, 16 * gib), 0);
    }
}
