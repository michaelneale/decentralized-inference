use anyhow::{Result, bail};
use skippy_protocol::{FlashAttentionType, StageKvCacheMode, StageKvCachePayload};

use super::super::KvCachePolicy;
use super::types::{
    BUILTIN_BATCH, BUILTIN_PARALLEL, BUILTIN_SAFETY_MARGIN_GB, BUILTIN_UBATCH,
    ResolvedStageKvCache, ResolvedStageKvCacheTemplate,
};
use crate::plugin::{
    BoolOrAuto, HardwareConfig, IntegerOrString, ModelConfigDefaults, ModelFitConfig, SkippyConfig,
    StringOrStringList,
};

pub(super) fn derive_fit_target_mib(
    allocatable_memory_bytes: Option<u64>,
    safety_margin_gb: f64,
) -> Option<u64> {
    let allocatable_mib = allocatable_memory_bytes?.checked_div(1024 * 1024)?;
    Some(allocatable_mib.saturating_sub(safety_margin_mib(safety_margin_gb)))
}

fn safety_margin_mib(safety_margin_gb: f64) -> u64 {
    (safety_margin_gb * 1024.0).round().max(0.0) as u64
}

/// Bytes the local fit withholds on top of the driver reserve: the
/// `safety_margin_gb` from the global hardware defaults, or the built-in
/// default, rounded the way `derive_fit_target_mib` rounds it. The advertised
/// capacity breakdown reports this same value as the configured reserve, so
/// peers see the margin the fit actually applies.
pub(crate) fn effective_safety_margin_bytes(defaults: Option<&ModelConfigDefaults>) -> u64 {
    let safety_margin_gb = defaults
        .and_then(|defaults| defaults.hardware.as_ref())
        .and_then(|hardware| hardware.safety_margin_gb)
        .unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    // The float-to-integer cast saturates on absurd margins; saturate the
    // byte conversion the same way instead of overflowing.
    safety_margin_mib(safety_margin_gb).saturating_mul(1024 * 1024)
}

pub(super) fn effective_flash_attention(cache_type_v: &str) -> FlashAttentionType {
    if cache_type_v.eq_ignore_ascii_case("f16") {
        FlashAttentionType::Auto
    } else {
        FlashAttentionType::Enabled
    }
}

pub(super) fn resolve_prefill_chunk_policy(value: &str) -> String {
    if value.eq_ignore_ascii_case("auto") {
        "fixed".to_string()
    } else {
        value.to_string()
    }
}

pub(super) fn has_explicit_prefill_controls(config: &SkippyConfig) -> bool {
    config
        .prefill_chunking
        .as_deref()
        .is_some_and(|value| !value.eq_ignore_ascii_case("auto"))
        || config.prefill_chunk_size.unwrap_or(0) > 0
        || config.prefill_chunk_schedule.is_some()
}

pub(super) fn reject_unsupported_model_fit_controls(
    config: Option<&ModelFitConfig>,
    _base_path: &str,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    if config.cache_ram_mib.unwrap_or(0) > 0 {
        bail!("skippy model_fit.cache_ram_mib is not supported by the pinned runtime");
    }
    if config.keep_tokens.unwrap_or(0) > 0 {
        bail!("skippy model_fit.keep_tokens is not supported by the pinned runtime");
    }
    reject_auto_only_bool(config.context_shift.as_ref(), "model_fit.context_shift")?;
    if config.checkpoint_interval.is_some() || config.checkpoint_count.is_some() {
        bail!("skippy checkpoint controls are not supported by the pinned runtime");
    }
    if config.lookup_cache_static.is_some() || config.lookup_cache_dynamic.is_some() {
        bail!("skippy lookup cache controls are not supported by the pinned runtime");
    }
    Ok(())
}

pub(super) fn reject_unsupported_hardware_controls(
    config: Option<&HardwareConfig>,
    base_path: &str,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    if config.placement.is_some() {
        bail!("skippy {base_path}.placement is not supported by the pinned runtime");
    }
    if config.tensor_split.is_some() {
        bail!("skippy {base_path}.tensor_split is not supported by the pinned runtime");
    }
    if config.cpu_moe.is_some() {
        bail!("skippy {base_path}.cpu_moe is not supported by the pinned runtime");
    }
    if config.n_cpu_moe.is_some() {
        bail!("skippy {base_path}.n_cpu_moe is not supported by the pinned runtime");
    }
    Ok(())
}

fn reject_auto_only_bool(value: Option<&BoolOrAuto>, label: &str) -> Result<()> {
    match value {
        None => Ok(()),
        Some(BoolOrAuto::String(mode)) if mode.eq_ignore_ascii_case("auto") => Ok(()),
        Some(BoolOrAuto::Bool(false)) => Ok(()),
        Some(_) => bail!("skippy {label} is not supported by the pinned runtime"),
    }
}

pub(super) fn resolve_bool_or_auto(
    value: Option<&BoolOrAuto>,
    label: &str,
) -> Result<Option<bool>> {
    Ok(match value {
        None => None,
        Some(BoolOrAuto::Bool(value)) => Some(*value),
        Some(BoolOrAuto::String(value)) if value.eq_ignore_ascii_case("auto") => None,
        Some(BoolOrAuto::String(_)) => bail!("skippy {label} must be a boolean or \"auto\""),
    })
}

pub(super) fn parse_kv_offload_string(value: &str) -> Option<bool> {
    if value.eq_ignore_ascii_case("true") {
        Some(true)
    } else if value.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

pub(super) fn resolve_prefix_cache(
    model_fit: Option<&ModelFitConfig>,
    global_model_fit: Option<&ModelFitConfig>,
) -> Result<ResolvedStageKvCache> {
    let prompt_cache = model_fit
        .and_then(|fit| fit.prompt_cache.as_ref())
        .or_else(|| global_model_fit.and_then(|fit| fit.prompt_cache.as_ref()));
    let prefix_cache = model_fit
        .and_then(|fit| fit.prefix_cache.as_ref())
        .or_else(|| global_model_fit.and_then(|fit| fit.prefix_cache.as_ref()));
    if matches!(prompt_cache, Some(BoolOrAuto::Bool(false))) {
        if prefix_cache.is_some_and(|config| config.enabled != Some(false)) {
            bail!("skippy prefix_cache cannot be enabled when prompt_cache = false");
        }
        return Ok(ResolvedStageKvCache::Disabled);
    }
    let Some(prefix_cache) = prefix_cache else {
        return Ok(match prompt_cache {
            Some(BoolOrAuto::Bool(false)) => ResolvedStageKvCache::Disabled,
            _ => ResolvedStageKvCache::FamilyDefault,
        });
    };
    if prefix_cache.enabled == Some(false) {
        return Ok(ResolvedStageKvCache::Disabled);
    }
    Ok(ResolvedStageKvCache::Explicit(
        ResolvedStageKvCacheTemplate {
            mode: StageKvCacheMode::LookupRecord,
            payload: match prefix_cache.payload_mode.as_deref().unwrap_or("auto") {
                "resident-kv" => StageKvCachePayload::ResidentKv,
                "kv-recurrent" => StageKvCachePayload::KvRecurrent,
                "full-state" => StageKvCachePayload::FullState,
                _ => StageKvCachePayload::Auto,
            },
            max_entries: prefix_cache.max_entries.map(|value| value as usize),
            max_bytes: prefix_cache.max_bytes,
            min_tokens: prefix_cache.min_tokens.map(u64::from),
            shared_prefix_stride_tokens: prefix_cache.shared_stride_tokens.map(u64::from),
            shared_prefix_record_limit: prefix_cache
                .shared_record_limit
                .map(|value| value as usize),
        },
    ))
}

pub(super) struct KvMacroDefaults {
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) kv_offload: Option<String>,
}

pub(super) fn kv_macro_defaults(policy: &str, kv_policy: KvCachePolicy) -> KvMacroDefaults {
    match policy {
        "quality" => KvMacroDefaults {
            cache_type_k: Some("f16".to_string()),
            cache_type_v: Some("f16".to_string()),
            kv_offload: Some("auto".to_string()),
        },
        "saver" => KvMacroDefaults {
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q8_0".to_string()),
            kv_offload: Some("true".to_string()),
        },
        "auto" | "balanced" => KvMacroDefaults {
            cache_type_k: Some(kv_policy.cache_type_k().to_string()),
            cache_type_v: Some(kv_policy.cache_type_v().to_string()),
            kv_offload: Some("auto".to_string()),
        },
        _ => KvMacroDefaults {
            cache_type_k: Some(kv_policy.cache_type_k().to_string()),
            cache_type_v: Some(kv_policy.cache_type_v().to_string()),
            kv_offload: Some("auto".to_string()),
        },
    }
}

pub(super) struct ThroughputMacroDefaults {
    pub(super) batch: Option<u32>,
    pub(super) ubatch: Option<u32>,
    pub(super) parallel: Option<usize>,
    pub(super) continuous_batching: Option<String>,
}

pub(super) fn resolve_field_value<T: Copy>(
    per_model_explicit: Option<T>,
    per_model_macro: Option<T>,
    global_explicit: Option<T>,
    global_macro: Option<T>,
    builtin: T,
) -> T {
    per_model_explicit
        .or(per_model_macro)
        .or(global_explicit)
        .or(global_macro)
        .unwrap_or(builtin)
}

pub(super) fn resolve_field_string(
    per_model_explicit: Option<&str>,
    per_model_macro: Option<&str>,
    global_explicit: Option<&str>,
    global_macro: Option<&str>,
    builtin: &str,
) -> String {
    per_model_explicit
        .or(per_model_macro)
        .or(global_explicit)
        .or(global_macro)
        .unwrap_or(builtin)
        .to_string()
}

pub(super) fn throughput_macro_defaults(policy: &str) -> ThroughputMacroDefaults {
    match policy {
        "throughput" => ThroughputMacroDefaults {
            batch: Some(BUILTIN_BATCH * 2),
            ubatch: Some(BUILTIN_UBATCH * 2),
            parallel: Some(2),
            continuous_batching: Some("true".to_string()),
        },
        "saver" => ThroughputMacroDefaults {
            batch: Some(BUILTIN_BATCH / 2),
            ubatch: Some(BUILTIN_UBATCH / 2),
            parallel: Some(1),
            continuous_batching: Some("false".to_string()),
        },
        _ => ThroughputMacroDefaults {
            batch: Some(BUILTIN_BATCH),
            ubatch: Some(BUILTIN_UBATCH),
            parallel: Some(BUILTIN_PARALLEL),
            continuous_batching: Some("auto".to_string()),
        },
    }
}

pub(super) fn parse_gpu_layers(
    model_value: Option<&IntegerOrString>,
    global_value: Option<&IntegerOrString>,
) -> Result<Option<i32>> {
    let Some(value) = model_value.or(global_value) else {
        return Ok(None);
    };

    let gpu_layers = match value {
        IntegerOrString::Integer(value) => i32::try_from(*value).map(Some).map_err(|_| {
            anyhow::anyhow!("hardware.gpu_layers must fit in a 32-bit signed integer")
        }),
        IntegerOrString::String(value) if value.eq_ignore_ascii_case("auto") => Ok(Some(-1)),
        IntegerOrString::String(value) => value
            .parse::<i32>()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("hardware.gpu_layers must be an integer or \"auto\"")),
    }?;

    match gpu_layers {
        Some(value) if value < -1 => bail!("hardware.gpu_layers must be at least -1"),
        _ => Ok(gpu_layers),
    }
}

pub(super) fn bool_or_auto_value(value: &BoolOrAuto) -> String {
    match value {
        BoolOrAuto::Bool(value) => value.to_string(),
        BoolOrAuto::String(value) => value.clone(),
    }
}

pub(super) fn string_list_value(value: &StringOrStringList) -> Vec<String> {
    match value {
        StringOrStringList::String(value) => vec![value.clone()],
        StringOrStringList::List(values) => values.clone(),
    }
}

pub(super) fn pick_value<T: Copy>(
    model_value: Option<T>,
    global_value: Option<T>,
    builtin: T,
) -> T {
    model_value.or(global_value).unwrap_or(builtin)
}

pub(super) fn pick_owned<T>(model_value: Option<T>, global_value: Option<T>) -> Option<T> {
    model_value.or(global_value)
}

pub(super) fn pick_string<'a>(
    model_value: Option<&'a str>,
    global_value: Option<&'a str>,
    builtin: Option<&'a str>,
) -> &'a str {
    model_value.or(global_value).or(builtin).unwrap_or_default()
}

pub(super) fn pick_string_owned(
    model_value: Option<&str>,
    global_value: Option<&str>,
    builtin: Option<&str>,
) -> String {
    pick_string(model_value, global_value, builtin).to_string()
}
