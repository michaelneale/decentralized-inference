use std::cmp::Ordering;

mod locked;

pub use locked::{LockedTopologyStage, plan_locked_topology};

const MINIMUM_AUTO_CONTEXT_LENGTH: u32 = 65_536;
const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];

/// Minimum context length per session for lane ceiling calculation.
/// This is the floor_ctx_per_session from the design.
const FLOOR_CTX_PER_SESSION: u32 = 4096;
const LLAMA_MAX_SEQ: usize = 256;
// Active lanes occupy one sequence id each. The native serving/cache layout
// reserves a second id range for auxiliary recurrent/speculative work and
// resident prefixes begin after `lane_count * 2`, so a lane ceiling must leave
// room for all three co-tenants under LLAMA_MAX_SEQ.
const SEQUENCE_IDS_PER_LANE_WITH_RESIDENT_CACHE: usize = 3;

/// Compute-buffer reserve applied to the KV term of each layer's placement
/// cost. Charging KV at 100/85 holds back 15% of a node's post-weight space for
/// llama.cpp compute-graph buffers and scratch — algebraically identical to the
/// single-node context planner's `usable_kv_cache_budget`, which grants KV 85%
/// of post-weight space (`context_planning.rs`). Without this, placement packed
/// a node with `weights + KV` alone and left the decode's transient buffers
/// nowhere to go, OOM-ing the stage or swapping the host. Because the reserve
/// rides on the KV term it scales with context length, matching how compute
/// buffers grow with `n_ctx`. A fixed per-node floor (see the coordinator's
/// node headroom) covers the context-independent minimum on top of this.
const KV_COMPUTE_RESERVE_NUMERATOR: u128 = 100;
const KV_COMPUTE_RESERVE_DENOMINATOR: u128 = 85;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlanningInput {
    pub native_context_length: u32,
    pub layer_count: u32,
    pub model_weight_bytes: u64,
    pub layer_weight_bytes: Vec<u64>,
    pub kv_bytes_per_token: u64,
    /// Fixed recurrent-state allocation for one configured lane, per layer.
    /// Dense/SWA-only layers use zero. The vector is layer-aligned when set.
    pub recurrent_bytes_per_sequence_by_layer: Vec<u64>,
    /// Sequence identifiers reserved for resident prefixes and auxiliary work.
    pub reserved_sequence_ids: usize,
    pub minimum_nodes: usize,
    pub nodes: Vec<TopologyNode>,
    pub context_length_override: Option<u32>,
    pub parallel_lanes_override: Option<usize>,
    pub target_decode_tpot_ms: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyNode {
    pub node_id: String,
    pub detected_vram_bytes: u64,
    pub max_vram_bytes: Option<u64>,
    pub runtime_headroom_bytes: u64,
    pub stage_transfer_latency_ms: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlan {
    pub context_length: u32,
    pub parallel_lanes: usize,
    pub stages: Vec<TopologyStagePlan>,
    pub estimated_decode_network_ms_per_token: Option<u32>,
    pub decode_tpot_target_met: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyStagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub parameter_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TopologyPlanError {
    #[error("topology planning requires native GGUF context length")]
    MissingNativeContext,
    #[error("topology planning requires at least one model layer")]
    MissingLayers,
    #[error("topology planning requires model weight bytes")]
    MissingModelWeights,
    #[error("topology planning requires KV bytes per token")]
    MissingKvBytesPerToken,
    #[error("topology planning requires at least one node")]
    MissingNodes,
    #[error("requested context {requested} is below minimum valid context {minimum}")]
    ContextBelowMinimum { requested: u32, minimum: u32 },
    #[error("requested context {requested} exceeds native context {native}")]
    ContextExceedsNative { requested: u32, native: u32 },
    #[error("requested parallel lanes must be greater than zero")]
    ZeroParallelLanes,
    #[error("no native sequence IDs remain after reservations")]
    NoSequenceIdCapacity,
    #[error("requested parallel lanes {requested} exceed sequence-id capacity {capacity}")]
    ParallelLanesExceedSequenceCapacity { requested: usize, capacity: usize },
    #[error("no topology can distribute all layers and keep context >= {minimum_context}")]
    NoValidTopology { minimum_context: u32 },
    #[error("locked topology must contain at least {minimum} stages; found {actual}")]
    LockedStageCount { minimum: usize, actual: usize },
    #[error("locked topology references unknown node {node_id}")]
    LockedUnknownNode { node_id: String },
    #[error("locked topology assigns node {node_id} more than once")]
    LockedDuplicateNode { node_id: String },
    #[error(
        "locked topology stage {stage_index} must start at layer {expected_start}; found {actual_start}"
    )]
    LockedNonContiguousRange {
        stage_index: usize,
        expected_start: u32,
        actual_start: u32,
    },
    #[error("locked topology stage {stage_index} has empty or reversed range {start}..{end}")]
    LockedInvalidRange {
        stage_index: usize,
        start: u32,
        end: u32,
    },
    #[error("locked topology ends at layer {actual_end}; model has {layer_count} layers")]
    LockedIncompleteCoverage { actual_end: u32, layer_count: u32 },
    #[error("locked topology cannot fit context >= {minimum_context}")]
    LockedTopologyDoesNotFit { minimum_context: u32 },
}

pub fn plan_topology(input: &TopologyPlanningInput) -> Result<TopologyPlan, TopologyPlanError> {
    plan_topology_with_required_stage0(input, None)
}

pub fn plan_topology_with_stage0(
    input: &TopologyPlanningInput,
    stage0_node_id: &str,
) -> Result<TopologyPlan, TopologyPlanError> {
    plan_topology_with_required_stage0(input, Some(stage0_node_id))
}

fn plan_topology_with_required_stage0(
    input: &TopologyPlanningInput,
    required_stage0_node_id: Option<&str>,
) -> Result<TopologyPlan, TopologyPlanError> {
    validate_input(input)?;

    let minimum_context = minimum_valid_context(input.native_context_length);
    let context_candidates = context_candidates(
        input.native_context_length,
        minimum_context,
        input.context_length_override,
    )?;
    let nodes = usable_nodes(&input.nodes);
    let latency_aware = latency_aware_planning(input, &nodes);

    let minimum_nodes = input.minimum_nodes.max(1);
    let mut best_latency_candidate: Option<CandidatePlan> = None;
    for context_length in context_candidates {
        let lane_candidates = parallel_lane_candidates(
            input.parallel_lanes_override,
            context_length,
            input.kv_bytes_per_token,
            input.reserved_sequence_ids,
        )?;
        for node_count in minimum_nodes..=nodes.len().min(input.layer_count as usize) {
            for parallel_lanes in lane_candidates.iter().copied() {
                let mut best_for_count: Option<CandidatePlan> = None;
                for_each_node_subset(&nodes, node_count, |subset| {
                    let Some(candidate) =
                        fit_candidate(input, subset, context_length, parallel_lanes)
                    else {
                        return;
                    };
                    if !candidate_has_required_stage0(&candidate, required_stage0_node_id) {
                        return;
                    }
                    if best_for_count
                        .as_ref()
                        .is_none_or(|current| candidate_better_for_same_shape(&candidate, current))
                    {
                        best_for_count = Some(candidate);
                    }
                });
                if let Some(candidate) = best_for_count {
                    if latency_aware {
                        if best_latency_candidate.as_ref().is_none_or(|current| {
                            latency_candidate_better(&candidate, current, input)
                        }) {
                            best_latency_candidate = Some(candidate);
                        }
                        continue;
                    }
                    return Ok(candidate.plan);
                }
            }
        }
    }

    if let Some(candidate) = best_latency_candidate {
        return Ok(candidate.plan);
    }

    Err(TopologyPlanError::NoValidTopology { minimum_context })
}

fn validate_input(input: &TopologyPlanningInput) -> Result<(), TopologyPlanError> {
    if input.native_context_length == 0 {
        return Err(TopologyPlanError::MissingNativeContext);
    }
    if input.layer_count == 0 {
        return Err(TopologyPlanError::MissingLayers);
    }
    if input.model_weight_bytes == 0 {
        return Err(TopologyPlanError::MissingModelWeights);
    }
    let recurrent_only = input.recurrent_bytes_per_sequence_by_layer.len()
        == input.layer_count as usize
        && input
            .recurrent_bytes_per_sequence_by_layer
            .iter()
            .all(|bytes| *bytes > 0);
    if input.kv_bytes_per_token == 0 && !recurrent_only {
        return Err(TopologyPlanError::MissingKvBytesPerToken);
    }
    if input.nodes.is_empty() {
        return Err(TopologyPlanError::MissingNodes);
    }
    Ok(())
}

fn context_candidates(
    native_context: u32,
    minimum_context: u32,
    override_context: Option<u32>,
) -> Result<Vec<u32>, TopologyPlanError> {
    if let Some(requested) = override_context {
        if requested > native_context {
            return Err(TopologyPlanError::ContextExceedsNative {
                requested,
                native: native_context,
            });
        }
        return Ok(vec![requested]);
    }

    let mut candidates = CONTEXT_STEPS
        .iter()
        .copied()
        .filter(|context| *context >= minimum_context && *context <= native_context)
        .collect::<Vec<_>>();
    candidates.push(native_context);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.reverse();
    Ok(candidates)
}

fn parallel_lane_candidates(
    override_lanes: Option<usize>,
    context_length: u32,
    _kv_bytes_per_token: u64,
    reserved_sequence_ids: usize,
) -> Result<Vec<usize>, TopologyPlanError> {
    let sequence_capacity = LLAMA_MAX_SEQ
        .saturating_sub(reserved_sequence_ids)
        .checked_div(SEQUENCE_IDS_PER_LANE_WITH_RESIDENT_CACHE)
        .unwrap_or(0);
    if sequence_capacity == 0 {
        return Err(TopologyPlanError::NoSequenceIdCapacity);
    }
    if let Some(lanes) = override_lanes {
        if lanes == 0 {
            return Err(TopologyPlanError::ZeroParallelLanes);
        }
        if lanes > sequence_capacity {
            return Err(TopologyPlanError::ParallelLanesExceedSequenceCapacity {
                requested: lanes,
                capacity: sequence_capacity,
            });
        }
        return Ok(vec![lanes]);
    }
    // Derive ceiling from KV pool size: n_seq_max = pool_cells / floor_ctx_per_session
    // pool_cells = context_length (total tokens in shared KV pool)
    // floor_ctx_per_session = minimum context needed per session
    let pool_cells = context_length as usize;
    let max_seq = pool_cells / FLOOR_CTX_PER_SESSION as usize;
    let max_seq = max_seq.clamp(1, sequence_capacity);
    Ok((1..=max_seq).rev().collect())
}

pub fn minimum_valid_context(native_context: u32) -> u32 {
    native_context.clamp(1, MINIMUM_AUTO_CONTEXT_LENGTH)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UsableNode {
    node_id: String,
    usable_vram_bytes: u64,
    stage_transfer_latency_ms: Option<u32>,
}

fn usable_nodes(nodes: &[TopologyNode]) -> Vec<UsableNode> {
    let mut nodes = nodes
        .iter()
        .map(|node| {
            let capped = node
                .max_vram_bytes
                .map(|max| node.detected_vram_bytes.min(max))
                .unwrap_or(node.detected_vram_bytes);
            UsableNode {
                node_id: node.node_id.clone(),
                usable_vram_bytes: capped.saturating_sub(node.runtime_headroom_bytes),
                stage_transfer_latency_ms: node.stage_transfer_latency_ms,
            }
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| {
        right
            .usable_vram_bytes
            .cmp(&left.usable_vram_bytes)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    nodes
}

fn for_each_node_subset(nodes: &[UsableNode], count: usize, mut visit: impl FnMut(&[UsableNode])) {
    let mut current = Vec::with_capacity(count);
    visit_node_subsets(nodes, count, 0, &mut current, &mut visit);
}

fn visit_node_subsets(
    nodes: &[UsableNode],
    count: usize,
    start: usize,
    current: &mut Vec<UsableNode>,
    visit: &mut impl FnMut(&[UsableNode]),
) {
    if current.len() == count {
        visit(current);
        return;
    }
    let needed = count - current.len();
    if nodes.len().saturating_sub(start) < needed {
        return;
    }
    for index in start..=nodes.len() - needed {
        current.push(nodes[index].clone());
        visit_node_subsets(nodes, count, index + 1, current, visit);
        current.pop();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CandidatePlan {
    plan: TopologyPlan,
    minimum_remaining_vram: u64,
    total_remaining_vram: u128,
}

impl Ord for CandidatePlan {
    fn cmp(&self, other: &Self) -> Ordering {
        self.minimum_remaining_vram
            .cmp(&other.minimum_remaining_vram)
            .then_with(|| self.total_remaining_vram.cmp(&other.total_remaining_vram))
            .then_with(|| {
                let left = self
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                let right = other
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                right.cmp(&left)
            })
    }
}

impl PartialOrd for CandidatePlan {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn fit_candidate(
    input: &TopologyPlanningInput,
    nodes: &[UsableNode],
    context_length: u32,
    parallel_lanes: usize,
) -> Option<CandidatePlan> {
    let layer_count = input.layer_count as usize;
    if nodes.len() > layer_count {
        return None;
    }

    let layer_weights = layer_weight_bytes(input);
    let kv_per_layer = input
        .kv_bytes_per_token
        .div_ceil(u64::from(input.layer_count));
    let recurrent_by_layer = recurrent_bytes_by_layer(input);
    let layer_required_bytes = layer_required_bytes(
        &layer_weights,
        &recurrent_by_layer,
        kv_per_layer,
        context_length,
        parallel_lanes,
    )?;

    let mut capacities = nodes.to_vec();
    capacities.sort_by(|left, right| {
        right
            .usable_vram_bytes
            .cmp(&left.usable_vram_bytes)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });

    let mut next_layer = 0u32;
    let mut stages = Vec::with_capacity(capacities.len());
    let mut minimum_remaining_vram = u64::MAX;
    let mut total_remaining_vram = 0u128;

    for (stage_index, node) in capacities.iter().enumerate() {
        let remaining_layers = input.layer_count - next_layer;
        let remaining_nodes = capacities.len() - stage_index;
        let min_for_later = remaining_nodes.saturating_sub(1) as u32;
        let assignable = remaining_layers.saturating_sub(min_for_later);
        let layer_span = assignable.min(max_contiguous_layers_from(
            &layer_required_bytes,
            next_layer as usize,
            assignable as usize,
            node.usable_vram_bytes,
        ) as u32);
        if layer_span == 0 {
            return None;
        }

        let layer_start = next_layer;
        let layer_end = layer_start + layer_span;
        let range = layer_start as usize..layer_end as usize;
        let parameter_bytes = sum_u64(&layer_weights[range.clone()]);
        let required_bytes = sum_u64(&layer_required_bytes[range]);
        if required_bytes > node.usable_vram_bytes {
            return None;
        }
        let remaining = node.usable_vram_bytes - required_bytes;
        minimum_remaining_vram = minimum_remaining_vram.min(remaining);
        total_remaining_vram += u128::from(remaining);
        stages.push(TopologyStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id: node.node_id.clone(),
            layer_start,
            layer_end,
            parameter_bytes,
        });
        next_layer = layer_end;
    }

    if next_layer != input.layer_count {
        return None;
    }

    let estimated_decode_network_ms_per_token = estimate_decode_network_ms_per_token(nodes);
    Some(CandidatePlan {
        plan: TopologyPlan {
            context_length,
            parallel_lanes,
            stages,
            estimated_decode_network_ms_per_token,
            decode_tpot_target_met: decode_tpot_target_met(
                estimated_decode_network_ms_per_token,
                input.target_decode_tpot_ms,
            ),
        },
        minimum_remaining_vram,
        total_remaining_vram,
    })
}

fn latency_aware_planning(_input: &TopologyPlanningInput, nodes: &[UsableNode]) -> bool {
    nodes
        .iter()
        .any(|node| node.stage_transfer_latency_ms.is_some())
}

fn candidate_has_required_stage0(
    candidate: &CandidatePlan,
    required_stage0_node_id: Option<&str>,
) -> bool {
    required_stage0_node_id.is_none_or(|required| {
        candidate
            .plan
            .stages
            .first()
            .is_some_and(|stage| stage.node_id == required)
    })
}

fn candidate_better_for_same_shape(candidate: &CandidatePlan, current: &CandidatePlan) -> bool {
    let candidate_estimate = candidate
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let current_estimate = current
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    candidate_estimate < current_estimate
        || (candidate_estimate == current_estimate && candidate.cmp(current) == Ordering::Greater)
}

fn latency_candidate_better(
    candidate: &CandidatePlan,
    current: &CandidatePlan,
    input: &TopologyPlanningInput,
) -> bool {
    latency_candidate_ordering(candidate, current, input) == Ordering::Greater
}

fn latency_candidate_ordering(
    left: &CandidatePlan,
    right: &CandidatePlan,
    input: &TopologyPlanningInput,
) -> Ordering {
    let left_estimate = left
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let right_estimate = right
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let left_target_met = decode_tpot_target_met(
        left.plan.estimated_decode_network_ms_per_token,
        input.target_decode_tpot_ms,
    )
    .unwrap_or(true);
    let right_target_met = decode_tpot_target_met(
        right.plan.estimated_decode_network_ms_per_token,
        input.target_decode_tpot_ms,
    )
    .unwrap_or(true);

    left_target_met
        .cmp(&right_target_met)
        .then_with(|| right_estimate.cmp(&left_estimate))
        .then_with(|| left.plan.context_length.cmp(&right.plan.context_length))
        .then_with(|| left.plan.parallel_lanes.cmp(&right.plan.parallel_lanes))
        .then_with(|| left.cmp(right))
}

fn estimate_decode_network_ms_per_token(nodes: &[UsableNode]) -> Option<u32> {
    let hop_latency = nodes
        .iter()
        .filter_map(|node| node.stage_transfer_latency_ms)
        .max()?;
    Some(hop_latency.saturating_mul(nodes.len() as u32))
}

fn decode_tpot_target_met(estimate: Option<u32>, target: Option<u32>) -> Option<bool> {
    Some(estimate? <= target?)
}

fn layer_weight_bytes(input: &TopologyPlanningInput) -> Vec<u64> {
    if input.layer_weight_bytes.len() == input.layer_count as usize {
        return input.layer_weight_bytes.clone();
    }
    let weight_per_layer = input
        .model_weight_bytes
        .div_ceil(u64::from(input.layer_count));
    vec![weight_per_layer; input.layer_count as usize]
}

fn candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    _parallel_lanes: usize,
) -> Option<u64> {
    // KV cache is a single shared allocation of size `n_ctx` — all lanes
    // share one unified cache via sequence IDs (kv_unified=true in
    // llama.cpp when lane_count > 1).  Do not multiply by lanes.
    let kv_bytes = u128::from(kv_per_layer).checked_mul(u128::from(context_length))?;
    // Charge KV at 100/85 so 15% of the node's post-weight space is held back
    // for llama.cpp compute-graph buffers/scratch (mirrors the single-node
    // context planner's `usable_kv_cache_budget`). This scales the reserve with
    // context length, matching how compute buffers grow with `n_ctx`.
    let kv_with_compute_reserve = kv_bytes
        .checked_mul(KV_COMPUTE_RESERVE_NUMERATOR)?
        .div_ceil(KV_COMPUTE_RESERVE_DENOMINATOR);
    let total = u128::from(weight_per_layer).checked_add(kv_with_compute_reserve)?;
    total.try_into().ok()
}

fn layer_required_bytes(
    layer_weights: &[u64],
    recurrent_bytes_by_layer: &[u64],
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> Option<Vec<u64>> {
    layer_weights
        .iter()
        .zip(recurrent_bytes_by_layer.iter().copied())
        .map(|(weight, recurrent_bytes)| {
            candidate_bytes_per_layer(*weight, kv_per_layer, context_length, parallel_lanes)
                .and_then(|base| {
                    recurrent_bytes
                        .checked_mul(parallel_lanes as u64)
                        .and_then(|recurrent| base.checked_add(recurrent))
                })
        })
        .collect()
}

fn recurrent_bytes_by_layer(input: &TopologyPlanningInput) -> Vec<u64> {
    if input.recurrent_bytes_per_sequence_by_layer.len() == input.layer_count as usize {
        return input.recurrent_bytes_per_sequence_by_layer.clone();
    }
    vec![0; input.layer_count as usize]
}

fn max_contiguous_layers_from(
    layer_required_bytes: &[u64],
    start: usize,
    limit: usize,
    capacity: u64,
) -> u64 {
    let mut total = 0u64;
    let mut count = 0u64;
    for bytes in layer_required_bytes.iter().skip(start).take(limit) {
        let next = total.saturating_add(*bytes);
        if next > capacity {
            break;
        }
        total = next;
        count += 1;
    }
    count
}

fn sum_u64(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0u64, |total, value| total.saturating_add(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const QWEN_CODER_480B_NATIVE_CONTEXT: u32 = 262_144;
    const QWEN_CODER_480B_LAYERS: u32 = 62;
    const QWEN_CODER_480B_WEIGHT_BYTES: u64 = 315_680_000_000;
    const QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN: u64 = 128 * 1024;
    const LOCAL_M1_ULTRA_METAL_BYTES: u64 = 115_448_725_504;
    const STUDIO_METAL_BYTES: u64 = 239_143_780_352;
    const STUDIO_RAM_BYTES: u64 = 274_877_906_944;

    fn node(id: &str, gib: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: gib * GIB,
            max_vram_bytes: None,
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    fn latency_node(id: &str, gib: u64, stage_transfer_latency_ms: u32) -> TopologyNode {
        TopologyNode {
            stage_transfer_latency_ms: Some(stage_transfer_latency_ms),
            ..node(id, gib)
        }
    }

    fn input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: 65_536,
            layer_count: 40,
            model_weight_bytes: 40 * GIB,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: 64 * 1024,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            reserved_sequence_ids: 16,
            minimum_nodes: 1,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
        }
    }

    fn qwen_coder_480b_input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: QWEN_CODER_480B_NATIVE_CONTEXT,
            layer_count: QWEN_CODER_480B_LAYERS,
            model_weight_bytes: QWEN_CODER_480B_WEIGHT_BYTES,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            reserved_sequence_ids: 16,
            minimum_nodes: 2,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
        }
    }

    fn qwen_node(index: usize, gib: u64) -> TopologyNode {
        node(&format!("qwen-node-{index:02}"), gib)
    }

    fn qwen_nodes(count: usize, gib: u64) -> Vec<TopologyNode> {
        (0..count).map(|index| qwen_node(index, gib)).collect()
    }

    #[test]
    fn lane_planning_rejects_exhausted_sequence_ids() {
        assert_eq!(
            parallel_lane_candidates(None, 65_536, 1, LLAMA_MAX_SEQ),
            Err(TopologyPlanError::NoSequenceIdCapacity)
        );
        assert_eq!(
            parallel_lane_candidates(Some(1), 65_536, 1, LLAMA_MAX_SEQ),
            Err(TopologyPlanError::NoSequenceIdCapacity)
        );
    }

    #[test]
    fn recurrent_pricing_rejects_plan_that_old_zero_cost_model_admitted() {
        // Falcon-H1 1.5B metadata: conv=4, inner=3072, state=256,
        // groups=1, 24 recurrent layers. Three state planes and two native
        // sequence slots per configured lane cost 19,132,416 bytes per layer.
        const FALCON_RECURRENT_BYTES_PER_LANE_PER_LAYER: u64 = 19_132_416;
        const LAYERS: u32 = 24;
        const LANES: usize = 64;

        let mut request = TopologyPlanningInput {
            native_context_length: 65_536,
            layer_count: LAYERS,
            model_weight_bytes: GIB,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: 24 * 1024,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            reserved_sequence_ids: 0,
            minimum_nodes: 1,
            nodes: vec![node("falcon-node", 1)],
            context_length_override: Some(65_536),
            parallel_lanes_override: Some(LANES),
            target_decode_tpot_ms: None,
        };
        let layer_weights = layer_weight_bytes(&request);
        let kv_per_layer = request.kv_bytes_per_token.div_ceil(u64::from(LAYERS));
        let old_required = sum_u64(
            &layer_required_bytes(
                &layer_weights,
                &vec![0; LAYERS as usize],
                kv_per_layer,
                request.native_context_length,
                LANES,
            )
            .unwrap(),
        );
        let recurrent_required =
            FALCON_RECURRENT_BYTES_PER_LANE_PER_LAYER * u64::from(LAYERS) * LANES as u64;

        // The old planner charged zero for recurrent state, so this budget
        // appears sufficient even though it covers only half of the real
        // fixed recurrent allocation.
        request.nodes[0].detected_vram_bytes = old_required + recurrent_required / 2;
        assert!(plan_topology(&request).is_ok());

        // The new planner rejects the same unsafe budget and accepts the
        // exact boundary once the complete recurrent allocation is present.
        request.recurrent_bytes_per_sequence_by_layer =
            vec![FALCON_RECURRENT_BYTES_PER_LANE_PER_LAYER; LAYERS as usize];
        assert_eq!(
            plan_topology(&request),
            Err(TopologyPlanError::NoValidTopology {
                minimum_context: 65_536,
            })
        );
        request.nodes[0].detected_vram_bytes = old_required + recurrent_required;
        assert!(plan_topology(&request).is_ok());
        assert_eq!(recurrent_required, 29_387_390_976);
    }

    #[test]
    fn pure_recurrent_topology_accepts_zero_kv_bytes() {
        let mut request = input(vec![node("mamba-node", 80)]);
        request.kv_bytes_per_token = 0;
        request.recurrent_bytes_per_sequence_by_layer = vec![1024; request.layer_count as usize];

        assert!(plan_topology(&request).is_ok());
    }

    #[test]
    fn zero_kv_bytes_rejects_dense_and_hybrid_metadata() {
        let mut dense = input(vec![node("dense-node", 80)]);
        dense.kv_bytes_per_token = 0;
        assert_eq!(
            plan_topology(&dense),
            Err(TopologyPlanError::MissingKvBytesPerToken)
        );

        let mut hybrid = input(vec![node("hybrid-node", 80)]);
        hybrid.kv_bytes_per_token = 0;
        hybrid.recurrent_bytes_per_sequence_by_layer = vec![1024; hybrid.layer_count as usize];
        hybrid.recurrent_bytes_per_sequence_by_layer[0] = 0;
        assert_eq!(
            plan_topology(&hybrid),
            Err(TopologyPlanError::MissingKvBytesPerToken)
        );
    }

    fn metal_node(id: &str, metal_recommended_bytes: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: metal_recommended_bytes,
            max_vram_bytes: Some(metal_recommended_bytes),
            // Metal recommendedMaxWorkingSetSize is already the usable budget
            // reported by the local runtime.
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    #[test]
    fn chooses_highest_context_then_parallelism() {
        let plan = plan_topology(&input(vec![node("a", 23), node("b", 23)])).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 16);
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn prefers_fewest_nodes_before_more_lanes() {
        let plan = plan_topology(&input(vec![
            node("a", 80),
            node("b", 80),
            node("c", 80),
            node("d", 80),
            node("e", 80),
            node("f", 80),
        ]))
        .unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.parallel_lanes, 16);
    }

    #[test]
    fn assigns_fewer_layers_to_lower_vram_node() {
        let mut request = input(vec![node("small", 16), node("large", 48)]);
        request.minimum_nodes = 2;
        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        let small = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "small")
            .unwrap();
        let large = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "large")
            .unwrap();
        assert!(small.layer_end - small.layer_start < large.layer_end - large.layer_start);
    }

    #[test]
    fn exact_layer_weights_allow_uneven_package_fit() {
        let mut request = input(vec![node("large", 12), node("small", 9)]);
        request.layer_count = 4;
        request.model_weight_bytes = 18 * GIB;
        request.layer_weight_bytes = vec![GIB / 8, GIB / 8, 9 * GIB, 8 * GIB];
        request.kv_bytes_per_token = 1;
        request.minimum_nodes = 2;

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![("large", 0, 3), ("small", 3, 4)]
        );
        assert_eq!(plan.stages[0].parameter_bytes, 9 * GIB + GIB / 4);
        assert_eq!(plan.stages[1].parameter_bytes, 8 * GIB);
    }

    #[test]
    fn exact_layer_capacity_is_evaluated_at_each_stage_boundary() {
        let mut request = input(vec![node("large", 11), node("small", 3)]);
        request.layer_count = 4;
        request.model_weight_bytes = 12 * GIB;
        request.layer_weight_bytes = vec![9 * GIB, GIB, GIB, GIB];
        request.kv_bytes_per_token = 1;
        request.minimum_nodes = 2;

        let plan = plan_topology(&request).unwrap();

        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (2, 4)]
        );
    }

    #[test]
    fn applies_max_vram_and_runtime_headroom_per_node() {
        let mut capped = node("capped", 80);
        capped.max_vram_bytes = Some(24 * GIB);
        capped.runtime_headroom_bytes = 8 * GIB;
        let mut request = input(vec![capped, node("peer", 48)]);
        request.minimum_nodes = 2;
        let plan = plan_topology(&request).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "capped")
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < 20);
    }

    #[test]
    fn latency_aware_planner_prefers_lower_tpot_over_native_context() {
        let mut request = input(vec![
            latency_node("a", 23, 10),
            latency_node("b", 23, 10),
            latency_node("c", 23, 10),
            latency_node("d", 23, 10),
        ]);
        request.native_context_length = 262_144;
        request.minimum_nodes = 2;
        request.target_decode_tpot_ms = Some(33);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(20));
        assert_eq!(plan.decode_tpot_target_met, Some(true));
    }

    #[test]
    fn latency_aware_planner_reports_target_miss_when_memory_requires_more_stages() {
        let mut request = qwen_coder_480b_input(qwen_nodes(4, 80));
        request
            .nodes
            .iter_mut()
            .for_each(|node| node.stage_transfer_latency_ms = Some(10));
        request.target_decode_tpot_ms = Some(33);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 4);
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(40));
        assert_eq!(plan.decode_tpot_target_met, Some(false));
    }

    #[test]
    fn rejects_below_minimum_context_floor() {
        let err = plan_topology(&input(vec![node("tiny-a", 8), node("tiny-b", 8)]))
            .expect_err("context below the 64k floor should be rejected");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 65_536
            }
        );
    }

    #[test]
    fn minimum_context_floor_caps_at_native_context() {
        assert_eq!(minimum_valid_context(16_384), 16_384);
        assert_eq!(minimum_valid_context(65_536), 65_536);
        assert_eq!(minimum_valid_context(262_144), 65_536);
    }

    #[test]
    fn accepts_explicit_context_override_below_auto_floor() {
        let mut request = input(vec![node("a", 80), node("b", 80)]);
        request.native_context_length = 262_144;
        request.context_length_override = Some(32_768);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 32_768);
    }

    #[test]
    fn rejects_context_override_above_native() {
        let mut request = input(vec![node("a", 80)]);
        request.context_length_override = Some(131_072);

        assert_eq!(
            plan_topology(&request),
            Err(TopologyPlanError::ContextExceedsNative {
                requested: 131_072,
                native: 65_536,
            })
        );
    }

    #[test]
    fn qwen_coder_480b_rejects_when_layers_cannot_fit_above_context_floor() {
        // Simulation: 4 x 70 GiB nodes.
        //
        // Expected topology: none.
        //
        // Why: the planner may degrade context only to the shared 64k floor
        // (65_536). At this machine size the full 62-layer package plus
        // 64k KV cannot be distributed, so launching would silently produce
        // an under-resourced split.
        let err = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 70)))
            .expect_err("four 70 GiB nodes cannot hold this layer package above the context floor");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 65_536
            }
        );
    }

    #[test]
    fn qwen_coder_480b_studio_james_and_studio_mic_form_native_topology() {
        // Simulation: meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers
        // split across studio-james and studio-mic.
        //
        // studio-james:
        //   Metal recommendedMaxWorkingSetSize = 115_448_725_504 bytes.
        //
        // studio-mic:
        //   Metal recommendedMaxWorkingSetSize = 239_143_780_352 bytes.
        //   RAM = 274_877_906_944 bytes. RAM is documented here because it is
        //   part of the fixture, but the planner must use Metal working set
        //   size, not total RAM.
        //
        // Expected topology: possible, 131_072 context, 32 lanes.
        //
        // Why: this is a fixture-driven simulation. The model package metadata
        // and each machine's Metal working-set budget are passed into the same
        // planner used by runtime orchestration, and the planner reports
        // whether a topology can be formed plus its context and lane count.
        //
        // Context is 131_072 rather than the model's 262_144 native maximum
        // because the planner reserves compute-buffer headroom (KV billed at
        // 100/85). The ~316 GB of weights plus full-native KV would pack the
        // combined ~354.6 GB working-set budget to within a few GB, leaving no
        // room for llama.cpp compute graphs; halving the context restores ~18 GB
        // of headroom across the two stages. This is the fix for stages that
        // previously loaded at native context and then OOM'd on the first token.
        assert_eq!(STUDIO_RAM_BYTES, 274_877_906_944);

        let planned = plan_topology(&qwen_coder_480b_input(vec![
            metal_node("studio-james", LOCAL_M1_ULTRA_METAL_BYTES),
            metal_node("studio-mic", STUDIO_METAL_BYTES),
        ]));
        let (split_possible, context_length, parallel_lanes) = match &planned {
            Ok(plan) => (true, Some(plan.context_length), Some(plan.parallel_lanes)),
            Err(_) => (false, None, None),
        };

        assert!(split_possible, "{planned:?}");
        assert_eq!(context_length, Some(131_072));
        assert_eq!(parallel_lanes, Some(32));

        let plan = planned.expect("studio-james and studio-mic should form a split topology");
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(
            plan.stages.last().unwrap().layer_end,
            QWEN_CODER_480B_LAYERS
        );
    }

    #[test]
    fn qwen_coder_480b_uses_context_floor_when_larger_contexts_do_not_fit() {
        // Simulation: 4 x 80 GiB nodes.
        //
        // Expected topology: 4 stages, 65_536 context, 16 lanes.
        //
        // Why: native 262_144 and 131_072 contexts do not fit across these
        // nodes, but the shared 64k floor does.  Lanes use a shared unified
        // KV cache and do not multiply memory cost, so the pool/floor formula
        // derives 16 lanes.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 80))).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 16);
        assert_eq!(plan.stages.len(), 4);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(
            plan.stages.last().unwrap().layer_end,
            QWEN_CODER_480B_LAYERS
        );
    }

    #[test]
    fn qwen_coder_480b_prefers_native_context_then_parallelism() {
        // Simulation: 5 x 80 GiB nodes.
        //
        // Expected topology: 5 stages, native 262_144 context, 64 lanes.
        //
        // Why: adding the fifth node makes native context fit.  Lanes use a
        // shared unified KV cache, so the pool/floor formula derives 64 lanes.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(5, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 64);
        assert_eq!(plan.stages.len(), 5);
    }

    #[test]
    fn qwen_coder_480b_prefers_fewest_nodes_then_maximizes_lanes() {
        // Simulation: 10 x 80 GiB nodes.
        //
        // Expected topology: 5 stages, native 262_144 context, 64 lanes.
        //
        // Why: the planner prefers fewest nodes before more lanes. Five nodes
        // is the minimum that can hold the full layer package at native
        // context.  Lanes use a shared unified KV cache, so the auto cap of
        // 64 applies regardless of extra VRAM headroom.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(10, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 64);
        assert_eq!(plan.stages.len(), 5);
    }

    #[test]
    fn qwen_coder_480b_excludes_bystander_nodes() {
        // Simulation: 7 x 80 GiB nodes plus 3 x 1 GiB bystanders.
        //
        // Expected topology: 5 stages, native 262_144 context, 64 lanes.
        //
        // Why: the planner prefers fewest nodes first. Five 80 GiB nodes
        // achieve native context. Bystander nodes (1 GiB) cannot carry even
        // one layer at this shape and are excluded entirely.
        let mut nodes = qwen_nodes(7, 80);
        nodes.extend((7..10).map(|index| qwen_node(index, 1)));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 64);
        assert_eq!(plan.stages.len(), 5);
        assert!(
            plan.stages
                .iter()
                .all(|stage| !stage.node_id.ends_with("07")
                    && !stage.node_id.ends_with("08")
                    && !stage.node_id.ends_with("09"))
        );
    }

    #[test]
    fn qwen_coder_480b_assigns_less_work_to_smaller_nodes() {
        // Simulation: 1 x 64 GiB node and 5 x 80 GiB nodes.
        //
        // Expected topology: native context with the 64 GiB node assigned
        // fewer layers than the largest stage.
        //
        // Why: KV and weights are layer-local. Assigning fewer layers to the
        // smaller node prevents it from forcing down the cluster-wide context.
        let mut nodes = vec![qwen_node(0, 64)];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let smallest_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(smallest_stage.layer_end - smallest_stage.layer_start < max_layers);
    }

    #[test]
    fn qwen_coder_480b_applies_max_vram_and_headroom_in_simulation() {
        // Simulation: one physically larger 120 GiB node capped to 80 GiB
        // with 16 GiB runtime headroom, plus 5 x 80 GiB nodes.
        //
        // Expected topology: the capped node receives fewer layers than the
        // largest stage, despite having 120 GiB physically detected.
        //
        // Why: planning must apply max-vram and local runtime headroom per
        // node before assigning layers. The capped node's usable budget is
        // 64 GiB, so it should be treated as smaller than the uncapped peers.
        let mut capped = qwen_node(0, 120);
        capped.max_vram_bytes = Some(80 * GIB);
        capped.runtime_headroom_bytes = 16 * GIB;
        let mut nodes = vec![capped];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < max_layers);
    }
}
