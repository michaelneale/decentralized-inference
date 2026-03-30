//! MoE expert sharding: split models across mesh nodes by expert assignment.
//!
//! Each node gets a GGUF with the full trunk (attention, norms, embeddings, head)
//! plus a subset of experts. The shared core (hottest experts by gate mass) is
//! replicated to every node. Remaining experts are distributed uniquely.
//!
//! No cross-node traffic during inference — each node runs independently.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ── GGUF MoE detection ──

/// MoE info extracted from a GGUF file header.
#[derive(Clone, Debug)]
pub struct GgufMoeInfo {
    pub expert_count: u32,
    pub expert_used_count: u32,
}

/// GGUF value types (matching gguf.h enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    /// Size in bytes for fixed-size types. Returns None for String and Array.
    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// Read a little-endian u32.
fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64(f: &mut std::fs::File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a little-endian i64.
fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a GGUF string: uint64 length + bytes.
fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let len = read_u64(f)? as usize;
    if len > 1_000_000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

/// Skip over a GGUF value of the given type.
fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_u64(f)? as usize;
            for _ in 0..count {
                skip_gguf_value(f, elem_type)?;
            }
        }
        other => {
            let size = other.fixed_size().unwrap_or(0);
            f.seek(SeekFrom::Current(size as i64))?;
        }
    }
    Ok(())
}

/// Read a GGUF KV value as u32 (handles uint32, int32, uint16, etc.).
fn read_gguf_value_as_u32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<u32>> {
    match typ {
        GgufType::Uint32 => Ok(Some(read_u32(f)?)),
        GgufType::Int32 => Ok(Some(read_u32(f)?)), // reinterpret
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            Ok(Some(u16::from_le_bytes(buf) as u32))
        }
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf)?;
            Ok(Some(buf[0] as u32))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

/// Detect MoE parameters from a GGUF file by reading its header KV pairs.
///
/// Scans for `*.expert_count` and `*.expert_used_count` keys.
/// Returns None if the file isn't MoE (no expert_count or expert_count <= 1).
/// Takes ~1ms for typical GGUF files — only reads the header, not tensor data.
pub fn detect_moe(path: &Path) -> Option<GgufMoeInfo> {
    let mut f = std::fs::File::open(path).ok()?;

    // Header: magic (4) + version (4) + n_tensors (8) + n_kv (8)
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }

    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None; // v1 not supported
    }

    let _n_tensors = read_i64(&mut f).ok()?;
    let n_kv = read_i64(&mut f).ok()?;

    let mut expert_count: Option<u32> = None;
    let mut expert_used_count: Option<u32> = None;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key.ends_with(".expert_count") {
            expert_count = read_gguf_value_as_u32(&mut f, vtype).ok()?;
        } else if key.ends_with(".expert_used_count") {
            expert_used_count = read_gguf_value_as_u32(&mut f, vtype).ok()?;
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }

        // Early exit once we have both
        if expert_count.is_some() && expert_used_count.is_some() {
            break;
        }
    }

    match (expert_count, expert_used_count) {
        (Some(ec), Some(euc)) if ec > 1 => Some(GgufMoeInfo {
            expert_count: ec,
            expert_used_count: euc,
        }),
        _ => None,
    }
}

fn read_gguf_value_as_f32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<f32>> {
    match typ {
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(Some(f32::from_le_bytes(buf)))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_string_opt(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<String>> {
    match typ {
        GgufType::String => Ok(Some(read_gguf_string(f)?)),
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufCompactMeta {
    pub architecture: String,
    pub context_length: u32,
    pub vocab_size: u32,
    pub embedding_size: u32,
    pub head_count: u32,
    pub layer_count: u32,
    pub feed_forward_length: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub tokenizer_model_name: String,
    pub rope_scale: f32,
    pub rope_freq_base: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

/// Scan a GGUF file header and return compact structural metadata.
/// Reads only the KV section, never tensor data. Returns None on any parse failure.
pub fn scan_gguf_compact_meta(path: &Path) -> Option<GgufCompactMeta> {
    let mut f = std::fs::File::open(path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }
    let _n_tensors = read_i64(&mut f).ok()?;
    let n_kv = read_i64(&mut f).ok()?;

    let mut meta = GgufCompactMeta::default();
    let mut kv_head_count: u32 = 0;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.architecture" {
            meta.architecture = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key == "tokenizer.ggml.model" {
            meta.tokenizer_model_name = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key.ends_with(".context_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.context_length = v;
            }
        } else if key.ends_with(".embedding_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.embedding_size = v;
            }
        } else if key.ends_with(".head_count") && !key.ends_with("_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.head_count = v;
            }
        } else if key.ends_with(".attention.head_count_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                kv_head_count = v;
            }
        } else if key.ends_with(".block_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.layer_count = v;
            }
        } else if key.ends_with(".feed_forward_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.feed_forward_length = v;
            }
        } else if key.ends_with(".attention.key_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.key_length = v;
            }
        } else if key.ends_with(".attention.value_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.value_length = v;
            }
        } else if key.ends_with(".rope.scale") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_scale = v;
            }
        } else if key.ends_with(".rope.freq_base") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_freq_base = v;
            }
        } else if key.ends_with(".vocab_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.vocab_size = v;
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_count = v;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_used_count = v;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    if meta.head_count > 0 {
        if meta.key_length == 0 {
            meta.key_length = meta.embedding_size / meta.head_count;
        }
        if meta.value_length == 0 {
            let effective_kv = if kv_head_count > 0 {
                kv_head_count
            } else {
                meta.head_count
            };
            meta.value_length = meta.embedding_size / effective_kv;
        }
    }

    Some(meta)
}

// ── GGUF assembler: combine trunk + expert files into a shard ──

// ── Ranking cache ──

/// Path to cached ranking CSV for a model.
/// Stored next to the model: `~/.models/moe-rankings/<model-stem>.csv`
pub fn ranking_cache_path(model_path: &Path) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-rankings").join(format!("{stem}.csv"))
}

/// Load a cached ranking CSV. Format: one expert_id per line, sorted by gate mass descending.
/// Also supports the full CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
pub fn load_cached_ranking(path: &Path) -> Option<Vec<u32>> {
    let content = std::fs::read_to_string(path).ok()?;
    let ranking: Vec<u32> = content
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("expert"))
        .filter_map(|l| {
            // Support both plain "42" and CSV "42,1234.5,0.03,500"
            l.split(',').next()?.trim().parse().ok()
        })
        .collect();
    if ranking.is_empty() {
        None
    } else {
        Some(ranking)
    }
}

// ── Expert assignment ──

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
/// Convenience wrapper for compute_assignments_with_overlap with overlap=1.
pub fn compute_assignments(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
) -> Vec<NodeAssignment> {
    compute_assignments_with_overlap(ranking, n_nodes, min_experts, 1)
}

/// Compute expert assignments with a configurable overlap factor.
///
/// - `overlap`: how many nodes each expert should live on (1 = no redundancy,
///   2 = every expert on at least 2 nodes, etc.). Capped at n_nodes.
///
/// Strategy:
/// 1. Shared core = top `min_experts` by gate mass → replicated to every node
/// 2. Remaining experts distributed with `overlap` copies each
///
/// With overlap=2, losing any single node doesn't orphan any expert —
/// at least one other node still has it.
pub fn compute_assignments_with_overlap(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
    overlap: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    let min_exp = min_experts as usize;
    let overlap = overlap.min(n_nodes).max(1);

    if n_nodes <= 1 || min_exp >= n_expert {
        // Single node or core covers everything — give everyone all experts
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    // Shared core = top min_experts by gate mass (replicated to every node)
    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();

    // Remaining experts to distribute with overlap
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();

    // With overlap, each expert goes to `overlap` nodes.
    // Total expert-slots = remaining.len() * overlap, distributed round-robin.
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        // Assign to `overlap` consecutive nodes (wrapping)
        for j in 0..overlap {
            let node = (i + j) % n_nodes;
            node_experts[node].push(expert_id);
        }
    }

    let mut assignments = Vec::with_capacity(n_nodes);
    for node_exps in node_experts {
        let n_unique = node_exps.len();
        let mut experts = shared_core.clone();
        experts.extend_from_slice(&node_exps);
        experts.sort();
        experts.dedup(); // in case overlap wraps and duplicates with shared core

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
    assignment
        .experts
        .iter()
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

fn resolve_split_binary(bin_dir: &Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        bin_dir.join("llama-moe-split"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-split"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }

    anyhow::bail!(
        "llama-moe-split not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    );
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
    let split_bin = resolve_split_binary(bin_dir)?;
    let status = std::process::Command::new(&split_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--expert-list",
            &expert_list,
            "-o",
            &output_path.to_string_lossy(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run {}: {e}", split_bin.display()))?;

    anyhow::ensure!(status.success(), "llama-moe-split exited with {status}");
    Ok(())
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
    fn test_ranking_cache_roundtrip() {
        let dir = std::env::temp_dir().join("moe-test-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        let ranking: Vec<u32> = vec![0, 26, 41, 69, 104, 3, 7, 99];
        let content: String = ranking
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, ranking);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_moe_analyze_csv() {
        // The CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
        let dir = std::env::temp_dir().join("moe-test-csv");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ranking.csv");
        std::fs::write(
            &path,
            "expert_id,total_mass,mass_fraction,selection_count\n\
            0,8365.69,0.250,15680\n\
            26,267.43,0.008,4800\n\
            41,250.11,0.007,4600\n",
        )
        .unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, vec![0, 26, 41]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_moe_qwen3() {
        let path = std::path::Path::new("/Users/micn/.models/Qwen3-30B-A3B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: model file not found");
            return;
        }
        let info = detect_moe(path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 128);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_olmoe() {
        let path =
            std::path::Path::new("/Users/micn/.models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf");
        if !path.exists() {
            eprintln!("Skipping: OLMoE model file not found");
            return;
        }
        let info = detect_moe(path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 64);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_dense_model() {
        // Qwen2.5-3B is dense (no experts) — should return None
        let path = std::path::Path::new("/Users/micn/.models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: dense model file not found");
            return;
        }
        assert!(
            detect_moe(path).is_none(),
            "Dense model should not be detected as MoE"
        );
    }

    #[test]
    fn test_single_node() {
        let ranking: Vec<u32> = (0..8).collect();
        let assignments = compute_assignments(&ranking, 1, 4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].experts.len(), 8); // gets everything
    }

    // ── Overlap tests ──

    #[test]
    fn test_overlap_2x_3_nodes() {
        // 128 experts, min 46, 3 nodes, 2× overlap
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 46, 2);

        assert_eq!(assignments.len(), 3);

        // Every expert should appear in at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }

        // Shared core (0..46) in all 3 nodes
        for e in 0..46 {
            assert!(
                *expert_count.get(&e).unwrap() >= 3,
                "Shared expert {e} should be in all nodes"
            );
        }
        // Remaining experts (46..128) in at least 2 nodes
        for e in 46..128 {
            assert!(
                *expert_count.get(&e).unwrap() >= 2,
                "Expert {e} should be in at least 2 nodes, got {}",
                expert_count[&e]
            );
        }
        // Full coverage
        assert_eq!(expert_count.len(), 128);
    }

    #[test]
    fn test_overlap_2x_2_nodes() {
        // With 2 nodes and 2× overlap, every remaining expert is on both nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 2, 4, 2);

        assert_eq!(assignments.len(), 2);
        // Both nodes should have all 10 experts (4 shared + 6 remaining × 2× = both)
        assert_eq!(assignments[0].experts.len(), 10);
        assert_eq!(assignments[1].experts.len(), 10);
    }

    #[test]
    fn test_overlap_1x_same_as_original() {
        // overlap=1 should give same results as compute_assignments
        let ranking: Vec<u32> = (0..128).collect();
        let a1 = compute_assignments(&ranking, 3, 46);
        let a2 = compute_assignments_with_overlap(&ranking, 3, 46, 1);

        for i in 0..3 {
            assert_eq!(a1[i].experts, a2[i].experts);
        }
    }

    #[test]
    fn test_overlap_capped_at_n_nodes() {
        // overlap=10 with 3 nodes should cap to 3 (every expert on every node)
        let ranking: Vec<u32> = (0..20).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 5, 10);

        // All 3 nodes should have all 20 experts
        for a in &assignments {
            assert_eq!(a.experts.len(), 20);
        }
    }

    #[test]
    fn test_overlap_glm5_10_nodes() {
        // GLM-5: 256 experts, min 96, 10 nodes, 2× overlap
        let ranking: Vec<u32> = (0..256).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 10, 96, 2);

        assert_eq!(assignments.len(), 10);

        // Full coverage
        let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        assert_eq!(all.len(), 256);

        // Every remaining expert on at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }
        for e in 96..256 {
            assert!(*expert_count.get(&e).unwrap() >= 2);
        }

        // Print sizes for verification
        for (i, a) in assignments.iter().enumerate() {
            eprintln!(
                "  Node {i}: {} experts ({} shared + {} unique)",
                a.experts.len(),
                a.n_shared,
                a.n_unique
            );
        }
    }
}
