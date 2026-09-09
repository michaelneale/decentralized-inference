use crate::models::gguf::{GgufCompactMeta, GgufKvCacheQuant};

const DEFAULT_CONTEXT_LENGTH: u32 = 4096;
const DEFAULT_PARALLEL_SLOTS: usize = 4;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
/// Auto-planner ceiling on concurrent lanes.
///
/// Matches upstream llama-server: when `--parallel` is left to auto,
/// llama-server picks `n_parallel = 4` and turns on `kv_unified = true`
/// (see `tools/server/server.cpp`,
/// `"n_parallel is set to auto, using n_parallel = 4 and kv_unified = true"`).
///
/// Skippy's stage-runtime patches also set `kv_unified = true` whenever
/// `lane_count > 1` (`third_party/llama.cpp/patches/0034-*.patch`). In
/// unified mode llama allocates exactly `n_ctx` cells total, shared
/// across all `n_seq_max` sequences. The previous ceiling of 16 was
/// inherited from a VRAM-based slot calculation that pretended each
/// lane carved off its own `n_ctx × bytes_per_token` allocation —
/// which is the `kv_unified = false` semantics, not what skippy
/// actually does. On any node with comfortable VRAM that math
/// happily picked 16 lanes even though all 16 raced for the *same*
/// pool of `n_ctx` cells.
///
/// Concrete failure mode that prompted this change: Qwen3-8B on a
/// 32k `n_ctx` got `slots = 16`. Three concurrent agent-shape
/// requests (~14k tokens each — OpenCode system prompt plus tools
/// plus a tool-result follow-up) need ~45k cells in the shared 32k
/// pool; llama's `find_slot` fails on the third request and skippy
/// surfaces it as an HTTP 502 with body `RuntimeError: llama_decode failed`.
///
/// 4 is the same conservative ceiling llama-server uses for the
/// same `kv_unified = true` reason. Operators who know their
/// workload (e.g. all short chat turns, or a single-user MoA host)
/// can still go higher via `parallel_override` /
/// `[models.throughput] parallel = N` in the TOML config.
const MAX_AUTO_PARALLEL_SLOTS: usize = 4;
/// Default ceiling on auto-planned context length (128k).
///
/// Some published GGUFs advertise a native window far larger than is useful on
/// a mesh — e.g. the Nemotron family ships 1,048,576-token artifacts. Left
/// unclamped, the auto-planner would try to drive a 1M context, spend the whole
/// KV budget on depth, and starve the parallel lanes that agentic serving needs
/// (or shrink context per-lane below what an agent can use). 128k is the
/// agent-serving sweet spot: deep enough for real tool loops and replay
/// corpora, shallow enough to keep multiple lanes and usable decode throughput.
///
/// This clamp is a `min`, so native windows at or below 128k keep their full
/// native size. It is a *default* only: an explicit `--ctx-size` /
/// `[models] ctx_size` override bypasses planning entirely and can still request
/// the full native window.
///
/// Deepening the default past 128k (toward 256k) is deliberately **not** done
/// here: it is memory-bandwidth-bound, not capacity-bound, and picking the
/// depth safely needs a populated-KV tok/s calibration we do not have yet. That
/// work is tracked as a follow-up (bandwidth-aware context/lane planning).
const MAX_AUTO_CONTEXT_LENGTH: u32 = 131_072;
const KV_CACHE_BUDGET_NUMERATOR: u64 = 85;
const KV_CACHE_BUDGET_DENOMINATOR: u64 = 100;
const FALLBACK_CONTEXT_8K_FREE_BYTES: u64 = 3_000_000_000;
const FALLBACK_CONTEXT_16K_FREE_BYTES: u64 = 6_000_000_000;
const FALLBACK_CONTEXT_32K_FREE_BYTES: u64 = 12_000_000_000;
const FALLBACK_CONTEXT_64K_FREE_BYTES: u64 = 30_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RuntimeResourcePlanningProfile {
    /// A dedicated-local launch (no shared mesh-serving surface requested).
    DedicatedLocal,
    /// A shared mesh-serving launch (`--auto` / `--publish` / `--discover` /
    /// `--join`).
    ///
    /// Both profiles currently plan the same context (`min(native, 128k)` held
    /// at single-lane depth, followed by the capped lane count over the shared
    /// pool). The distinction is retained for the bandwidth-aware planner
    /// follow-up, where a shared mesh host and a dedicated local host will want
    /// different tok/s floors.
    SharedMesh,
}

impl RuntimeResourcePlanningProfile {
    /// Number of concurrent lanes to *plan context depth* around.
    ///
    /// Both profiles plan context at single-lane (deepest) depth — `min(native,
    /// 128k)` sized to fit one shared `n_ctx` pool in the KV budget — and then
    /// run [`planned_parallel_slots`] lanes over that shared pool. Because
    /// `kv_unified = true` makes those lanes share the one pool, the lane count
    /// is the capped concurrency target and does not shrink with the residual
    /// budget. The profile axis is kept because it is threaded through the
    /// serving surfaces and will regain distinct behavior in the bandwidth-aware
    /// planner follow-up (where a shared mesh host and a dedicated local host
    /// want different tok/s floors).
    fn context_slot_target(self) -> u64 {
        match self {
            Self::DedicatedLocal | Self::SharedMesh => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeResourcePlan {
    pub(super) context_length: u32,
    pub(super) slots: usize,
    /// Structured breakdown of the inputs and intermediate results the planner
    /// used, emitted once per model start so every later memory-planning change
    /// is measurable in production logs. `None` only for plans built outside
    /// [`plan_runtime_resources`].
    pub(super) breakdown: Option<RuntimeResourcePlanBreakdown>,
}

/// The measured-vs-charged picture at plan time. All byte fields are the
/// planner's *charges* (estimates), not runtime measurements — the point of
/// carrying them is to compare against the measured KV/compute buffer sizes
/// llama.cpp reports at context init (forwarded as structured
/// `buffer_mib`/`backend_device` native log events by `skippy-runtime`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeResourcePlanBreakdown {
    pub(super) vram_bytes: u64,
    pub(super) model_bytes: u64,
    /// KV budget after the 85% utilization tax (`usable_kv_cache_budget`).
    pub(super) kv_budget_bytes: u64,
    /// Planned KV allocation: context_length x kv_bytes_per_token.
    pub(super) planned_kv_bytes: u64,
    /// Per-token KV cost used by the plan (layer-fraction scaled).
    pub(super) kv_bytes_per_token: u64,
    pub(super) slots: usize,
    pub(super) context_length: u32,
    /// True when slots came from the flat auto default rather than an
    /// explicit override.
    pub(super) slots_auto: bool,
    /// True when the context length came from planning rather than an
    /// explicit override.
    pub(super) context_auto: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RuntimeResourcePlanInput<'a> {
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
    /// Model weight bytes **local to this node**.  For a split/layer-package
    /// load, pass only this node's share of the model weights.
    pub(super) model_bytes: u64,
    pub(super) vram_bytes: u64,
    pub(super) metadata: Option<&'a GgufCompactMeta>,
    /// The KV cache quant that will be used.  Default is Q8_0 everywhere.
    /// Only differs when the user explicitly passes `--cache-type-k/v`.
    pub(super) kv_cache_quant: GgufKvCacheQuant,
    /// Fraction of the model's layers that reside on this node (0.0–1.0).
    /// `None` means the whole model is local (fraction = 1.0).
    pub(super) local_layer_fraction: Option<f64>,
    pub(super) planning_profile: RuntimeResourcePlanningProfile,
    /// Measured native buffer footprint from a prior context init of the same
    /// shape on this node (compute + KV buffer sizes and the context length
    /// they were measured at). When present, the planner charges these
    /// measured sizes instead of the 85% KV tax (budget-driven sizing); when
    /// absent it falls back to the static tax ladder.
    pub(super) measured_buffers: Option<MeasuredBufferFootprint>,
}

/// Measured native buffer sizes from one context init, the ground truth the
/// budget-driven planner charges in place of the KV-scaled tax.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MeasuredBufferFootprint {
    /// Compute-graph buffer(s) at context init, bytes.
    pub(super) compute_bytes: u64,
    /// KV buffer at `context_length`, bytes.
    pub(super) kv_bytes: u64,
    /// The context length the KV buffer was measured at.
    pub(super) context_length: u32,
    /// Lane count the buffers were measured at. Compute buffers scale
    /// ~linearly with lanes (measured 399/783/1551 MiB at 2/4/8 lanes on the
    /// 5080 for granite), so a plan resolving a different lane count scales
    /// the compute charge by the lane ratio.
    pub(super) lane_count: u32,
}

/// Plan context length and parallel slots.
///
/// Strategy: maximise context up to `min(native, MAX_AUTO_CONTEXT_LENGTH)` (the
/// 128k agent-serving default ceiling) using the provided KV quant (default
/// Q8_0), holding that context and filling as many lanes as the budget affords.
/// No negotiation — the quant is decided upstream (Q8_0 default, or user
/// override via CLI flags), and an explicit `--ctx-size` override bypasses this
/// entirely.
pub(super) fn plan_runtime_resources(input: RuntimeResourcePlanInput<'_>) -> RuntimeResourcePlan {
    let context_auto = input.ctx_size_override.is_none();
    let slots_auto = input.parallel_override.is_none();
    let context_length = input
        .ctx_size_override
        .unwrap_or_else(|| planned_context_length(&input));
    let slots = input
        .parallel_override
        .unwrap_or_else(planned_parallel_slots);

    let kv_bytes_per_token = input
        .metadata
        .and_then(|metadata| {
            input
                .kv_cache_quant
                .kv_cache_bytes_per_token(metadata)
                .map(|bytes| scale_by_layer_fraction(bytes, &input))
        })
        .unwrap_or(0);
    let kv_budget_bytes = usable_kv_cache_budget(input.vram_bytes, input.model_bytes);
    let planned_kv_bytes = kv_bytes_per_token.saturating_mul(u64::from(context_length));

    RuntimeResourcePlan {
        context_length,
        slots,
        breakdown: Some(RuntimeResourcePlanBreakdown {
            vram_bytes: input.vram_bytes,
            model_bytes: input.model_bytes,
            kv_budget_bytes,
            planned_kv_bytes,
            kv_bytes_per_token,
            slots,
            context_length,
            slots_auto,
            context_auto,
        }),
    }
}

fn planned_context_length(input: &RuntimeResourcePlanInput<'_>) -> u32 {
    if let Some(planned) = planned_context_length_budget_driven(input) {
        return planned;
    }
    let fallback_context = fallback_context_length(input);
    let Some(metadata) = input.metadata else {
        return fallback_context;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return fallback_context;
    }
    // Clamp the *native* window (read per-artifact from this GGUF's header, not
    // from a model-name lookup) to the default auto-context ceiling before KV
    // planning. Keeps 1M-token natives from over-committing KV to a single very
    // deep context while leaving smaller native windows untouched.
    let native_context = native_context.min(MAX_AUTO_CONTEXT_LENGTH);
    let Some(kv_bytes_per_token_full) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata)
    else {
        return fallback_context.min(native_context);
    };

    // In a pipeline-parallel split each stage only holds KV state for its
    // own layers.  Scale the per-token cost by the local layer fraction.
    let kv_bytes_per_token = scale_by_layer_fraction(kv_bytes_per_token_full, input);

    let kv_budget = usable_kv_cache_budget(input.vram_bytes, input.model_bytes);
    if kv_bytes_per_token == 0 {
        return native_context;
    }
    let slot_target = context_slot_target(input);
    let Some(kv_bytes_for_target_slots) = kv_bytes_per_token.checked_mul(slot_target) else {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    };
    let max_affordable_context = kv_budget / kv_bytes_for_target_slots;
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

fn context_slot_target(input: &RuntimeResourcePlanInput<'_>) -> u64 {
    input
        .parallel_override
        .map(|slots| slots.max(1) as u64)
        .unwrap_or_else(|| input.planning_profile.context_slot_target())
}

/// Plan the number of concurrent lanes to run at the chosen context depth.
///
/// Under `kv_unified = true` — which skippy's stage runtime sets whenever
/// `lane_count > 1` (`third_party/llama.cpp/patches/0034-*.patch`) and which
/// llama-server's `--parallel auto` also selects — every lane shares a single
/// `n_ctx` cell pool. The attention KV cache is one allocation of
/// `context_length × kv_bytes_per_token`, **not** one allocation per lane, so
/// adding a lane over that shared pool costs no additional KV memory.
/// [`planned_context_length`] already sized that pool to fit the node's KV
/// budget, so the lane count is a concurrency choice bounded only by the
/// [`MAX_AUTO_PARALLEL_SLOTS`] safety ceiling — not a division of residual
/// budget by a per-lane `n_ctx` allocation.
///
/// The previous `usable_kv_cache_budget / (context_length × kv_bytes)` math was
/// the `kv_unified = false` accounting. It matched neither the runtime nor the
/// split topology planner (`skippy-coordinator/src/topology.rs`
/// `candidate_bytes_per_layer`, which deliberately does *not* multiply KV by
/// lanes), and it produced two wrong results:
///   * On a tight node a deep context consumed most of the budget, so the
///     residual implied 1–2 lanes — even though the pool is the *same* size a
///     fat node would hold and the extra lanes are free. That capped concurrency
///     for no benefit: the shared pool (and thus runtime find-slot contention)
///     is identical at any lane count.
///   * It made the lane count depend on the KV quant (q4 vs q8) at a fixed
///     context, even though quant changes bytes-per-cell, not the *cell* count
///     that governs how many lanes safely share the pool.
///
/// Recurrent/SSM layers do keep per-lane state; that per-lane cost is accounted
/// for by the split topology planner and is bounded here by the 4-lane cap. A
/// finer single-node recurrent-aware bound is left to the bandwidth-aware
/// planner follow-up.
fn planned_parallel_slots() -> usize {
    // llama-server's conservative unified-KV auto default, never above our
    // safety cap. Independent of KV allocation size by construction.
    DEFAULT_PARALLEL_SLOTS.min(MAX_AUTO_PARALLEL_SLOTS)
}

fn scale_by_layer_fraction(kv_bytes_per_token: u64, input: &RuntimeResourcePlanInput<'_>) -> u64 {
    let fraction = input.local_layer_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    if fraction < 1.0 && fraction > 0.0 {
        ((kv_bytes_per_token as f64) * fraction).ceil() as u64
    } else {
        kv_bytes_per_token
    }
}

fn usable_kv_cache_budget(vram_bytes: u64, model_bytes: u64) -> u64 {
    let free_bytes = vram_bytes.saturating_sub(model_bytes);
    let budget = u128::from(free_bytes) * u128::from(KV_CACHE_BUDGET_NUMERATOR)
        / u128::from(KV_CACHE_BUDGET_DENOMINATOR);
    budget.min(u128::from(u64::MAX)) as u64
}

/// Measured-vs-charged reconciliation, produced once the native context
/// exists (after model open) and the `sched_reserve` buffer lines have been
/// parsed.
///
/// `charged_compute_reserve_bytes` is the compute-buffer proxy the planner
/// actually held back: the 15% slice of post-weight space implied by
/// [`usable_kv_cache_budget`]'s 85% numerator. `measured_*` are what
/// llama.cpp allocated. The gap between the charged proxy and the measured
/// sizes is the memory Step-2's budget-driven planner can spend, and the
/// residual is what a re-plan with actual free memory would see.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MemoryPlanReconciliation {
    /// Post-weight space the 85% KV tax held back as a compute-buffer proxy.
    pub(super) charged_compute_reserve_bytes: u64,
    /// Measured compute buffer(s) at context init, if any were observed.
    pub(super) measured_compute_bytes: Option<u64>,
    /// Measured KV buffer(s) at context init, if any were observed.
    pub(super) measured_kv_bytes: Option<u64>,
    /// `vram - model - measured_compute - measured_kv` when both
    /// measurements exist; the headroom a re-plan starts from.
    pub(super) residual_free_bytes: Option<u64>,
}

pub(super) fn reconcile_memory_plan_with_measurements(
    breakdown: &RuntimeResourcePlanBreakdown,
    measured: Option<skippy_runtime::MeasuredNativeBuffers>,
) -> MemoryPlanReconciliation {
    let charged_compute_reserve_bytes = breakdown
        .vram_bytes
        .saturating_sub(breakdown.model_bytes)
        .saturating_sub(breakdown.kv_budget_bytes);
    let mib_to_bytes = |mib: Option<f64>| mib.map(|mib| (mib * 1024.0 * 1024.0).round() as u64);
    let measured_compute_bytes = mib_to_bytes(measured.and_then(|m| m.compute_mib));
    let measured_kv_bytes = mib_to_bytes(measured.and_then(|m| m.kv_mib));
    let residual_free_bytes = match (measured_compute_bytes, measured_kv_bytes) {
        (Some(compute), Some(kv)) => Some(
            breakdown
                .vram_bytes
                .saturating_sub(breakdown.model_bytes)
                .saturating_sub(compute)
                .saturating_sub(kv),
        ),
        _ => None,
    };
    MemoryPlanReconciliation {
        charged_compute_reserve_bytes,
        measured_compute_bytes,
        measured_kv_bytes,
        residual_free_bytes,
    }
}

fn fallback_context_length(input: &RuntimeResourcePlanInput<'_>) -> u32 {
    let free_bytes = input.vram_bytes.saturating_sub(input.model_bytes);
    if free_bytes >= FALLBACK_CONTEXT_64K_FREE_BYTES {
        65_536
    } else if free_bytes >= FALLBACK_CONTEXT_32K_FREE_BYTES {
        32_768
    } else if free_bytes >= FALLBACK_CONTEXT_16K_FREE_BYTES {
        16_384
    } else if free_bytes >= FALLBACK_CONTEXT_8K_FREE_BYTES {
        8192
    } else {
        DEFAULT_CONTEXT_LENGTH
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

/// Fraction of usable memory the budget-driven planner targets (Step 2).
///
/// vLLM reserves ~8% of free memory after weights (`gpu_memory_utilization`
/// 0.92), SGLang ~10% (`mem-fraction-static` 0.9). Ours is deliberately a
/// little more conservative: mesh nodes can co-host other stages, and Metal
/// unified-memory nodes share the pool with the OS (where an over-commit
/// page-out stalls decode rather than failing loudly like a CUDA OOM).
const DEFAULT_UTILIZATION_TARGET_NUMERATOR: u64 = 88;
const DEFAULT_UTILIZATION_TARGET_DENOMINATOR: u64 = 100;

/// Budget-driven context planning (Step 2) over the measured footprint from a
/// prior context init.
///
/// Model: `budget = (vram - model) × utilization - measured_compute`, then
/// solve for the deepest context whose *scaled* KV cost fits the budget. KV
/// scales linearly with context (unified pool of `n_ctx` cells), so the
/// measured KV bytes at `measured_ctx` give `kv_bytes_per_token_measured =
/// kv_bytes / measured_ctx`, and the deepest affordable context is
/// `budget / kv_bytes_per_token_measured`.
///
/// Returns `None` when the measurement cannot support the model (zero
/// context, or a KV per-token cost of zero), letting the static tax ladder
/// answer instead. Compute buffers do not scale linearly with context
/// (ubatch/graph shape dominates), so the measured value is charged as-is,
/// unchanged from the probe context.
fn planned_context_length_budget_driven(input: &RuntimeResourcePlanInput<'_>) -> Option<u32> {
    let measured = input.measured_buffers?;
    if measured.context_length == 0 || measured.kv_bytes == 0 || measured.lane_count == 0 {
        return None;
    }
    let Some(metadata) = input.metadata else {
        return None;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return None;
    }
    let native_context = native_context.min(MAX_AUTO_CONTEXT_LENGTH);

    let post_weight = input.vram_bytes.saturating_sub(input.model_bytes);
    let utilised = u128::from(post_weight) * u128::from(DEFAULT_UTILIZATION_TARGET_NUMERATOR)
        / u128::from(DEFAULT_UTILIZATION_TARGET_DENOMINATOR);
    // Scale the measured compute charge by the lane ratio when the plan's
    // resolved lane count differs from the one the footprint was measured at:
    // compute buffers scale ~linearly with lanes (CUDA_Host + device graphs),
    // while the unified KV pool is lane-invariant. Linear-through-origin
    // slightly overestimates when scaling up (~2-3% at 4→8 on measured data),
    // which is the conservative direction; scaling down is symmetric.
    let resolved_lanes = input
        .parallel_override
        .map(|slots| slots.max(1) as u64)
        .unwrap_or_else(|| planned_parallel_slots() as u64);
    let lane_ratio_num = u128::from(resolved_lanes);
    let lane_ratio_den = u128::from(measured.lane_count);
    let compute_charge = u128::from(measured.compute_bytes) * lane_ratio_num / lane_ratio_den;
    let budget = utilised.saturating_sub(compute_charge);
    let kv_per_token = u128::from(measured.kv_bytes) / u128::from(measured.context_length);
    if kv_per_token == 0 {
        return None;
    }
    let max_affordable = (budget / kv_per_token).min(u128::from(u32::MAX)) as u32;
    if max_affordable == 0 {
        return None;
    }
    let planned = max_affordable.min(native_context);
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    Some(if planned < minimum {
        minimum
    } else {
        snap_context_length_down(planned).max(minimum)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gqa_metadata(context_length: u32) -> GgufCompactMeta {
        GgufCompactMeta {
            context_length,
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_overrides_are_preserved() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: Some(16_384),
            parallel_override: Some(7),
            model_bytes: 10_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert_eq!(plan.slots, 7);
    }

    #[test]
    fn auto_context_clamped_to_native() {
        let metadata = gqa_metadata(16_384);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(
            plan.context_length, 16_384,
            "should reach native context, not exceed it"
        );
    }

    #[test]
    fn q8_default_reaches_larger_context_than_f16() {
        // Tight VRAM so f16 can only reach 8K but q8_0 reaches 16K.
        // KV budget = (7.0 - 5.0) * 0.85 = 1.7 GB.
        // f16: 131072 B/tok → 1.7G / 131K ≈ 12K → snaps 8K
        // q8:   69632 B/tok → 1.7G / 69K  ≈ 24K → snaps 16K
        let metadata = gqa_metadata(131_072);
        let f16_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 7_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });
        let q8_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 7_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert!(
            q8_plan.context_length > f16_plan.context_length,
            "q8_0 should afford more context: q8={}K, f16={}K",
            q8_plan.context_length / 1024,
            f16_plan.context_length / 1024
        );
    }

    #[test]
    fn fallback_defaults_without_metadata() {
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 16_000_000_000,
            metadata: None,
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert_eq!(plan.slots, 4);
    }

    #[test]
    fn both_profiles_produce_identical_auto_plans() {
        // The old behavior traded context for concurrency on the shared-mesh
        // profile (shallower context, more lanes). The default is now uniform:
        // hold context at `min(native, 128k)` and run the capped lane count,
        // so both profiles plan the same context and lane count. The
        // profile axis is retained for the bandwidth-aware follow-up.
        let metadata = gqa_metadata(131_072);
        let input = |profile| RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 16_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: profile,
            measured_buffers: None,
        };
        let dedicated_plan =
            plan_runtime_resources(input(RuntimeResourcePlanningProfile::DedicatedLocal));
        let shared_plan = plan_runtime_resources(input(RuntimeResourcePlanningProfile::SharedMesh));

        assert_eq!(
            dedicated_plan, shared_plan,
            "both profiles hold the 128k floor and fill lanes identically"
        );
        assert!(dedicated_plan.context_length <= 131_072);
    }

    #[test]
    fn auto_context_capped_at_128k_for_million_token_native() {
        // Nemotron-class 1M-token native on a fat node. Left unclamped the
        // planner would drive a multi-hundred-K context; the default ceiling
        // holds it at 128k and spends the rest of the budget on lanes.
        let metadata = gqa_metadata(1_048_576);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::SharedMesh,
            measured_buffers: None,
        });

        assert_eq!(
            plan.context_length, 131_072,
            "1M native must clamp to the 128k default ceiling"
        );
        assert_eq!(
            plan.slots, 4,
            "fat node should keep the full 4 lanes at 128k"
        );
    }

    #[test]
    fn tight_budget_holds_128k_floor_and_keeps_full_lanes() {
        // A >128k native on a node whose budget affords 128k at only ~1 lane
        // under the old per-lane allocation math. Under `kv_unified = true` all
        // lanes share the single 128k pool, so the tight node runs the same
        // 4-lane target a fat node would: reducing lanes here buys neither memory
        // (unified pool) nor contention headroom (the pool — hence runtime
        // find-slot contention — is identical at any lane count). Regression for
        // the unified-KV lane-accounting fix; the pre-fix planner returned 1 lane
        // here because a deep context ate the residual budget.
        let metadata = gqa_metadata(262_144);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 18_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::SharedMesh,
            measured_buffers: None,
        });

        assert_eq!(
            plan.context_length, 131_072,
            "context is held at the 128k floor"
        );
        assert_eq!(
            plan.slots, 4,
            "a tight node shares the same n_ctx pool and keeps the full 4-lane \
             target; got {} lanes",
            plan.slots
        );
    }

    #[test]
    fn explicit_parallel_with_auto_context() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(2),
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert_eq!(plan.slots, 2);
    }

    #[test]
    fn auto_slots_capped_at_llama_server_default() {
        // Regression: a small model on a huge-VRAM box used to plan
        // `slots = 16` because the VRAM-derived per-lane math pretended
        // each lane carved off its own `n_ctx × bytes/token` allocation.
        // With `kv_unified = true` (skippy patch 0034) those 16 lanes
        // race for the same `n_ctx` cell pool, and 3 concurrent agent
        // requests at ~14k tokens each blow it up with
        // `find_slot` failures → HTTP 502
        // `RuntimeError: llama_decode failed`.
        //
        // Match llama-server's auto default of 4 (see
        // `.deps/llama.cpp/tools/server/server.cpp`: "n_parallel is
        // set to auto, using n_parallel = 4 and kv_unified = true").
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            // 128GB free — plenty for many "per-lane" slots under the
            // old broken math.
            vram_bytes: 128_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert!(
            plan.slots <= 4,
            "auto-planner should not exceed llama-server's 4-lane unified-KV ceiling; got {}",
            plan.slots
        );
    }

    #[test]
    fn explicit_parallel_can_exceed_auto_ceiling() {
        // Operators who know their workload can still go higher than
        // the auto ceiling via `parallel_override`.
        let metadata = gqa_metadata(131_072);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(8),
            model_bytes: 5_000_000_000,
            vram_bytes: 128_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert_eq!(plan.slots, 8);
    }

    #[test]
    fn split_model_uses_local_layer_fraction() {
        // 480B-class model: 94 layers, 264GB total, host holds 62/94 layers.
        let metadata = GgufCompactMeta {
            context_length: 131_072,
            head_count: 64,
            kv_head_count: 8,
            layer_count: 94,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let total_model_bytes: u64 = 264_000_000_000;
        let local_fraction = 62.0 / 94.0;
        let local_model_bytes = (total_model_bytes as f64 * local_fraction) as u64;

        // Without split awareness: 206 GB VRAM, 264 GB model → negative budget → minimum
        let no_split = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: total_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        // With split awareness: local model ~174 GB, local KV fraction 0.66
        let split = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: local_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: Some(local_fraction),
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });

        assert!(
            split.context_length > no_split.context_length,
            "split-aware should produce larger context: split={}K, no_split={}K",
            split.context_length / 1024,
            no_split.context_length / 1024
        );
        assert!(
            split.context_length >= 65_536,
            "480B split on 206+103 GB with q8_0 should get at least 64K, got {}K",
            split.context_length / 1024
        );
    }

    #[test]
    fn lane_count_is_independent_of_kv_quant() {
        // Under `kv_unified = true` the lanes share one `n_ctx` cell pool. The KV
        // quant changes bytes-per-cell, not the cell count that governs safe
        // concurrency, so at the same context depth q4_0 and q8_0 must plan the
        // *same* number of lanes. (The pre-fix `budget / (context × kv_bytes)`
        // math gave q4 more lanes than q8 purely because q4 cells are smaller —
        // the accounting bug this follow-up removes.)
        let metadata = gqa_metadata(131_072);
        let plan_with = |quant| {
            plan_runtime_resources(RuntimeResourcePlanInput {
                ctx_size_override: None,
                parallel_override: None,
                model_bytes: 5_000_000_000,
                vram_bytes: 80_000_000_000,
                metadata: Some(&metadata),
                kv_cache_quant: quant,
                local_layer_fraction: None,
                planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
                measured_buffers: None,
            })
        };
        let q8_plan = plan_with(GgufKvCacheQuant::Q8_0);
        let q4_plan = plan_with(GgufKvCacheQuant::Q4_0);

        assert_eq!(q8_plan.context_length, q4_plan.context_length);
        assert_eq!(
            q8_plan.slots, q4_plan.slots,
            "lane count must not depend on KV quant under unified KV: q4={}, q8={}",
            q4_plan.slots, q8_plan.slots
        );
    }

    #[test]
    fn budget_driven_context_uses_measured_kv_and_compute() {
        // Roomy node: 16 GiB VRAM, 3 GiB weights. Here the measured KV
        // per-token cost (2 GiB / 16_384 = 131_072 B) is twice the q8
        // estimate (~69_632 B) the ladder assumes, so the budget-driven plan
        // correctly lands at 65_536 while the estimate-based ladder would
        // clamp at native 131_072. Measured reality cuts both ways: when the
        // real KV cost is higher than estimated, the correct plan is
        // shallower, not deeper. (Conversely, charging measured compute under
        // an 88% utilization target only buys depth over the 85% tax when
        // compute is under ~3% of free memory — the estimate is what binds
        // on roomy nodes, not the tax.)
        let metadata = gqa_metadata(131_072);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 3 * 1024 * 1024 * 1024,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: Some(MeasuredBufferFootprint {
                compute_bytes: 512 * 1024 * 1024,
                kv_bytes: 2 * 1024 * 1024 * 1024,
                context_length: 16_384,
                lane_count: 4,
            }),
        });
        assert_eq!(plan.context_length, 65_536);

        // Tight node where the ladder's estimate binds: 5 GiB free, the q8
        // estimate (65,536 B/tok for these dims) caps the ladder at 65_536,
        // while a measured KV cost half the estimate (32,768 B/tok measured
        // at 16_384) lets the budget-driven plan reach the full native
        // window.
        let tight = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 1 * 1024 * 1024 * 1024,
            vram_bytes: 6 * 1024 * 1024 * 1024,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: Some(MeasuredBufferFootprint {
                compute_bytes: 128 * 1024 * 1024,
                kv_bytes: 512 * 1024 * 1024,
                context_length: 16_384,
                lane_count: 4,
            }),
        });
        assert_eq!(tight.context_length, 131_072);

        let tight_ladder = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 1 * 1024 * 1024 * 1024,
            vram_bytes: 6 * 1024 * 1024 * 1024,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        });
        assert_eq!(tight_ladder.context_length, 65_536);
    }

    #[test]
    fn budget_driven_context_degrades_to_ladder_when_measurement_is_unusable() {
        let metadata = gqa_metadata(131_072);
        let base = RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        };

        // Zero-context, zero-KV, or zero-lane measurements cannot drive the
        // budget model; the plan must equal the static-ladder answer.
        let ladder = plan_runtime_resources(base).context_length;
        for unusable in [
            MeasuredBufferFootprint {
                compute_bytes: 512 * 1024 * 1024,
                kv_bytes: 0,
                context_length: 16_384,
                lane_count: 4,
            },
            MeasuredBufferFootprint {
                compute_bytes: 512 * 1024 * 1024,
                kv_bytes: 2 * 1024 * 1024 * 1024,
                context_length: 0,
                lane_count: 4,
            },
            MeasuredBufferFootprint {
                compute_bytes: 512 * 1024 * 1024,
                kv_bytes: 2 * 1024 * 1024 * 1024,
                context_length: 16_384,
                lane_count: 0,
            },
            // Compute larger than the whole utilized budget saturates the
            // budget to zero; the ladder answers rather than planning an
            // empty context.
            MeasuredBufferFootprint {
                compute_bytes: u64::MAX,
                kv_bytes: 2 * 1024 * 1024 * 1024,
                context_length: 16_384,
                lane_count: 4,
            },
        ] {
            let degraded = plan_runtime_resources(RuntimeResourcePlanInput {
                measured_buffers: Some(unusable),
                ..base
            });
            assert_eq!(degraded.context_length, ladder);
        }
    }

    #[test]
    fn budget_driven_compute_charge_scales_with_lane_count() {
        // Compute buffers scale ~linearly with lanes (measured 399/783/1551
        // MiB at 2/4/8 on the 5080). A footprint measured at 4 lanes must be
        // charged at 2x when the plan resolves 8 lanes, and at 0.5x when it
        // resolves 2 — and the context depth must follow the charge.
        let metadata = gqa_metadata(131_072);
        let footprint = MeasuredBufferFootprint {
            compute_bytes: 512 * 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 16_384,
            lane_count: 4,
        };
        let plan_at = |parallel_override: Option<usize>| {
            plan_runtime_resources(RuntimeResourcePlanInput {
                ctx_size_override: None,
                parallel_override,
                model_bytes: 3 * 1024 * 1024 * 1024,
                vram_bytes: 16 * 1024 * 1024 * 1024,
                metadata: Some(&metadata),
                kv_cache_quant: GgufKvCacheQuant::Q8_0,
                local_layer_fraction: None,
                planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
                measured_buffers: Some(footprint),
            })
            .context_length
        };

        let at2 = plan_at(Some(2));
        let at4 = plan_at(Some(4));
        let at8 = plan_at(Some(8));
        // Scaling the charge DOWN (fewer lanes than measured) frees budget and
        // can only deepen (or hold) the plan; scaling UP can only shallow it.
        assert!(at2 >= at4);
        assert!(at8 <= at4);
        // At the measured lane count the charge is exactly the measured value,
        // so this matches the roomy-node case of
        // budget_driven_context_uses_measured_kv_and_compute.
        assert_eq!(at4, 65_536);
    }

    #[test]
    fn reconciliation_exposes_overcharge_and_residual() {
        let breakdown = RuntimeResourcePlanBreakdown {
            vram_bytes: 16 * 1024 * 1024 * 1024,
            model_bytes: 3 * 1024 * 1024 * 1024,
            kv_budget_bytes: (13 * 1024 * 1024 * 1024) * 85 / 100,
            planned_kv_bytes: 2 * 1024 * 1024 * 1024,
            kv_bytes_per_token: 131_072,
            slots: 4,
            context_length: 16_384,
            slots_auto: true,
            context_auto: true,
        };

        // Without measurements the reconciliation still reports the charged
        // proxy so the log line carries the plan side of the story. The proxy
        // is what the 85% budget floor leaves behind, so derive it the same
        // way rather than as a separate 15% floor (85+15 != 100 in integer
        // math).
        let post_weight_bytes = 13 * 1024 * 1024 * 1024;
        let unmeasured = reconcile_memory_plan_with_measurements(&breakdown, None);
        assert_eq!(
            unmeasured.charged_compute_reserve_bytes,
            post_weight_bytes - post_weight_bytes * 85 / 100
        );
        assert_eq!(unmeasured.measured_compute_bytes, None);
        assert_eq!(unmeasured.residual_free_bytes, None);

        // With measurements: compute 0.5 GiB, KV 2.0 GiB over a 16 GiB node
        // holding 3 GiB of weights leaves 10.5 GiB residual — far above the
        // ~2 GiB proxy tax the planner charged.
        let measured = skippy_runtime::MeasuredNativeBuffers {
            compute_mib: Some(512.0),
            kv_mib: Some(2048.0),
            lane_count: Some(4),
        };
        let reconciled = reconcile_memory_plan_with_measurements(&breakdown, Some(measured));
        assert_eq!(reconciled.measured_compute_bytes, Some(512 * 1024 * 1024));
        assert_eq!(reconciled.measured_kv_bytes, Some(2048 * 1024 * 1024));
        assert_eq!(
            reconciled.residual_free_bytes,
            Some(10 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
        );
        assert_eq!(
            reconciled.charged_compute_reserve_bytes,
            unmeasured.charged_compute_reserve_bytes
        );
    }
}
