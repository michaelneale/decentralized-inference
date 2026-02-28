//! MoE expert sharding: split models across mesh nodes by expert assignment.
//!
//! Each node gets a GGUF with the full trunk (attention, norms, embeddings, head)
//! plus a subset of experts. The shared core (hottest experts by gate mass) is
//! replicated to every node. Remaining experts are distributed uniquely.
//!
//! No cross-node traffic during inference — each node runs independently.

use crate::download::MoeConfig;
use std::path::{Path, PathBuf};

/// Expert assignment for a single node: which expert IDs it should hold.
#[derive(Clone, Debug)]
pub struct NodeAssignment {
    /// All expert IDs for this node (shared core + unique shard), sorted.
    pub experts: Vec<u32>,
    /// How many of these are shared (replicated to every node).
    pub n_shared: usize,
    /// How many are unique to this node.
    pub n_unique: usize,
}

/// Compute expert assignments for N nodes using the overlap strategy.
///
/// - `ranking`: expert IDs sorted by gate mass descending (hottest first)
/// - `n_nodes`: number of mesh nodes to split across
/// - `min_experts`: minimum experts per node for coherent output
///
/// Returns one NodeAssignment per node. Every expert appears in at least one node.
pub fn compute_assignments(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    let min_exp = min_experts as usize;

    if n_nodes <= 1 || min_exp >= n_expert {
        // Single node or core covers everything — just give everyone all experts
        return vec![NodeAssignment {
            experts: ranking.to_vec(),
            n_shared: n_expert,
            n_unique: 0,
        }; n_nodes.max(1)];
    }

    // Shared core = top min_experts by gate mass
    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();

    // Remaining experts to distribute
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();
    let unique_per_node = remaining.len() / n_nodes;
    let leftover = remaining.len() % n_nodes;

    let mut assignments = Vec::with_capacity(n_nodes);
    let mut offset = 0;

    for i in 0..n_nodes {
        // Distribute leftover experts to first nodes (one extra each)
        let n_unique = unique_per_node + if i < leftover { 1 } else { 0 };
        let unique_shard: Vec<u32> = remaining[offset..offset + n_unique].to_vec();
        offset += n_unique;

        let mut experts = shared_core.clone();
        experts.extend_from_slice(&unique_shard);
        experts.sort();

        assignments.push(NodeAssignment {
            experts,
            n_shared: min_exp,
            n_unique,
        });
    }

    assignments
}

/// Format expert list as comma-separated string for moe-split --expert-list.
pub fn expert_list_arg(assignment: &NodeAssignment) -> String {
    assignment.experts.iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Path to the cached split GGUF for a given model + node count + node index.
pub fn split_path(model_path: &Path, n_nodes: usize, node_index: usize) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-splits")
        .join(format!("{stem}"))
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"))
}

/// Check if cached splits exist for this model + node count.
pub fn splits_exist(model_path: &Path, n_nodes: usize) -> bool {
    (0..n_nodes).all(|i| split_path(model_path, n_nodes, i).exists())
}

/// Run llama-moe-split to produce a split GGUF for one node.
pub fn run_split(
    bin_dir: &Path,
    model_path: &Path,
    assignment: &NodeAssignment,
    output_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let expert_list = expert_list_arg(assignment);
    let status = std::process::Command::new(bin_dir.join("llama-moe-split"))
        .args([
            "-m", &model_path.to_string_lossy(),
            "--expert-list", &expert_list,
            "-o", &output_path.to_string_lossy(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run llama-moe-split: {e}"))?;

    anyhow::ensure!(status.success(), "llama-moe-split exited with {status}");
    Ok(())
}

/// Full pipeline: ensure split GGUFs exist for this model + node count.
/// Returns paths to all split GGUFs.
pub fn ensure_splits(
    bin_dir: &Path,
    model_path: &Path,
    moe_config: &MoeConfig,
    n_nodes: usize,
) -> anyhow::Result<Vec<PathBuf>> {
    // Check cache
    if splits_exist(model_path, n_nodes) {
        let paths: Vec<PathBuf> = (0..n_nodes)
            .map(|i| split_path(model_path, n_nodes, i))
            .collect();
        tracing::info!("Using cached MoE splits for {n_nodes} nodes");
        return Ok(paths);
    }

    tracing::info!(
        "Splitting MoE model for {n_nodes} nodes ({} experts, top-{}, min {} per node)",
        moe_config.n_expert, moe_config.n_expert_used, moe_config.min_experts_per_node
    );

    let assignments = compute_assignments(
        moe_config.ranking,
        n_nodes,
        moe_config.min_experts_per_node,
    );

    let mut paths = Vec::with_capacity(n_nodes);
    for (i, assignment) in assignments.iter().enumerate() {
        let path = split_path(model_path, n_nodes, i);
        tracing::info!(
            "  Node {i}: {} experts ({} shared + {} unique) → {}",
            assignment.experts.len(), assignment.n_shared, assignment.n_unique,
            path.display()
        );
        run_split(bin_dir, model_path, assignment, &path)?;
        paths.push(path);
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assignments_2_nodes() {
        // 10 experts, min 4, 2 nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments(&ranking, 2, 4);

        assert_eq!(assignments.len(), 2);
        // Each node: 4 shared + 3 unique = 7 experts
        assert_eq!(assignments[0].experts.len(), 7);
        assert_eq!(assignments[1].experts.len(), 7);
        assert_eq!(assignments[0].n_shared, 4);
        assert_eq!(assignments[0].n_unique, 3);

        // Shared core (0-3) in both
        for e in 0..4 {
            assert!(assignments[0].experts.contains(&e));
            assert!(assignments[1].experts.contains(&e));
        }

        // Full coverage
        let mut all: Vec<u32> = assignments[0].experts.clone();
        all.extend(&assignments[1].experts);
        all.sort();
        all.dedup();
        assert_eq!(all, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_assignments_3_nodes() {
        // 128 experts, min 46, 3 nodes
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments(&ranking, 3, 46);

        assert_eq!(assignments.len(), 3);
        // 82 remaining / 3 = 27 each + 1 leftover
        // Nodes 0: 46+28=74, Node 1: 46+27=73, Node 2: 46+27=73
        assert_eq!(assignments[0].experts.len(), 74);
        assert_eq!(assignments[1].experts.len(), 73);
        assert_eq!(assignments[2].experts.len(), 73);

        // Full coverage
        let mut all: Vec<u32> = Vec::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        all.sort();
        all.dedup();
        assert_eq!(all, (0..128).collect::<Vec<u32>>());
    }

    #[test]
    fn test_single_node() {
        let ranking: Vec<u32> = (0..8).collect();
        let assignments = compute_assignments(&ranking, 1, 4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].experts.len(), 8); // gets everything
    }
}
