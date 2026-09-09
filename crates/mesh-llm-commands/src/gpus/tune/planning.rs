use super::*;

const BUILTIN_BATCH: u32 = 512;
const BUILTIN_UBATCH: u32 = 512;
const BUILTIN_SAFETY_MARGIN_GB: f64 = 2.0;
const LARGE_MODEL_MIN_BYTES: u64 = 50 * 1024 * 1024 * 1024;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
const KV_CACHE_BUDGET_NUMERATOR: u64 = 85;
const KV_CACHE_BUDGET_DENOMINATOR: u64 = 100;
const FALLBACK_CONTEXT_8K_FREE_BYTES: u64 = 3_000_000_000;
const FALLBACK_CONTEXT_16K_FREE_BYTES: u64 = 6_000_000_000;
const FALLBACK_CONTEXT_32K_FREE_BYTES: u64 = 12_000_000_000;
const FALLBACK_CONTEXT_64K_FREE_BYTES: u64 = 30_000_000_000;

pub(crate) fn derive_fit_target_mib(allocatable_memory_bytes: u64) -> u64 {
    let allocatable_mib = allocatable_memory_bytes / (1024 * 1024);
    let reserve_mib = (BUILTIN_SAFETY_MARGIN_GB * 1024.0).round().max(0.0) as u64;
    allocatable_mib.saturating_sub(reserve_mib)
}

pub(crate) fn recommended_kv_cache_quant(
    model_bytes: u64,
) -> model_artifact::gguf::GgufKvCacheQuant {
    if model_bytes >= LARGE_MODEL_MIN_BYTES {
        model_artifact::gguf::GgufKvCacheQuant::Q4_0
    } else {
        model_artifact::gguf::GgufKvCacheQuant::Q8_0
    }
}

pub(crate) fn tune_kv_cache_type(value: &str) -> Option<TuneKvCacheType> {
    match value {
        value if value.eq_ignore_ascii_case("f16") => Some(TuneKvCacheType::F16),
        value if value.eq_ignore_ascii_case("q8_0") => Some(TuneKvCacheType::Q8_0),
        value if value.eq_ignore_ascii_case("q4_0") => Some(TuneKvCacheType::Q4_0),
        _ => None,
    }
}

pub(crate) fn tune_kv_cache_type_from_quant(
    quant: model_artifact::gguf::GgufKvCacheQuant,
) -> TuneKvCacheType {
    match quant.v {
        model_artifact::gguf::GgufKvCacheType::F16 => TuneKvCacheType::F16,
        model_artifact::gguf::GgufKvCacheType::Q8_0 => TuneKvCacheType::Q8_0,
        model_artifact::gguf::GgufKvCacheType::Q4_0 => TuneKvCacheType::Q4_0,
    }
}

pub(crate) fn effective_flash_attention(cache_type_v: &TuneKvCacheType) -> TuneFlashAttentionValue {
    match cache_type_v {
        TuneKvCacheType::F16 => TuneFlashAttentionValue::Disabled,
        TuneKvCacheType::Q8_0 | TuneKvCacheType::Q4_0 => TuneFlashAttentionValue::Enabled,
    }
}

pub(crate) fn recommended_batch(ctx_size: u32) -> u32 {
    ctx_size.min(BUILTIN_BATCH)
}

pub(crate) fn recommended_ubatch(batch: u32) -> u32 {
    batch.clamp(1, BUILTIN_UBATCH)
}

pub(crate) fn minimum_context_fits(
    resident_model_bytes: u64,
    memory_budget_bytes: u64,
    kv_bytes_per_token: u64,
) -> bool {
    let required_kv = kv_bytes_per_token.saturating_mul(u64::from(MIN_AUTO_CONTEXT_LENGTH));
    resident_model_bytes.saturating_add(required_kv) <= memory_budget_bytes
}

pub(crate) fn resident_model_bytes_for_layers(
    model_bytes: u64,
    layer_count: u32,
    gpu_layers: u32,
) -> u64 {
    if gpu_layers == 0 || layer_count == 0 {
        return 0;
    }
    let numerator = u128::from(model_bytes).saturating_mul(u128::from(gpu_layers));
    let denominator = u128::from(layer_count);
    let rounded = numerator.saturating_add(denominator.saturating_sub(1)) / denominator;
    rounded.min(u128::from(u64::MAX)) as u64
}

pub(crate) fn planned_context_length(
    metadata: &model_artifact::gguf::GgufCompactMeta,
    resident_model_bytes: u64,
    memory_budget_bytes: u64,
    kv_cache_quant: model_artifact::gguf::GgufKvCacheQuant,
) -> u32 {
    let fallback_context = fallback_context_length(memory_budget_bytes, resident_model_bytes);
    let native_context = metadata.context_length;
    if native_context == 0 {
        return fallback_context;
    }
    let Some(kv_bytes_per_token) = kv_cache_quant.kv_cache_bytes_per_token(metadata) else {
        return fallback_context.min(native_context);
    };
    let kv_budget = usable_kv_cache_budget(memory_budget_bytes, resident_model_bytes);
    if kv_bytes_per_token == 0 {
        return native_context;
    }
    let max_affordable_context = kv_budget / kv_bytes_per_token;
    if max_affordable_context == 0 {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    }
    let planned = max_affordable_context
        .min(u64::from(native_context))
        .min(u64::from(u32::MAX)) as u32;
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    if planned < minimum {
        minimum
    } else {
        snap_context_length_down(planned).max(minimum)
    }
}

fn usable_kv_cache_budget(memory_budget_bytes: u64, resident_model_bytes: u64) -> u64 {
    let free_bytes = memory_budget_bytes.saturating_sub(resident_model_bytes);
    let budget = u128::from(free_bytes) * u128::from(KV_CACHE_BUDGET_NUMERATOR)
        / u128::from(KV_CACHE_BUDGET_DENOMINATOR);
    budget.min(u128::from(u64::MAX)) as u64
}

fn fallback_context_length(memory_budget_bytes: u64, resident_model_bytes: u64) -> u32 {
    let free_bytes = memory_budget_bytes.saturating_sub(resident_model_bytes);
    if free_bytes >= FALLBACK_CONTEXT_64K_FREE_BYTES {
        65_536
    } else if free_bytes >= FALLBACK_CONTEXT_32K_FREE_BYTES {
        32_768
    } else if free_bytes >= FALLBACK_CONTEXT_16K_FREE_BYTES {
        16_384
    } else if free_bytes >= FALLBACK_CONTEXT_8K_FREE_BYTES {
        8192
    } else {
        4096
    }
}

fn snap_context_length_down(value: u32) -> u32 {
    const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];
    CONTEXT_STEPS
        .iter()
        .rev()
        .copied()
        .find(|step| *step <= value)
        .unwrap_or(value)
}
