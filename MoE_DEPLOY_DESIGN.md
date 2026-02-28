# MoE Auto-Deploy Design

## User Experience

```bash
# User just says what model they want. mesh-llm figures out the rest.
mesh-llm --model Qwen3-30B-A3B-Q4_K_M
```

That's it. The system detects it's MoE, analyzes expert routing, splits the GGUF
per node, and each node runs its own llama-server with its expert shard. Sessions
are hash-routed across nodes. No manual steps.

## How It Works

### Step 1: Detect MoE

When a GGUF is loaded or referenced, read its metadata:
- `{arch}.expert_count` > 0 → it's MoE
- Extract `expert_count`, `expert_used_count` from GGUF kv pairs
- This is a cheap read — just parse the GGUF header, no need to load the model

```rust
// In download.rs or a new moe.rs
fn detect_moe(gguf_path: &Path) -> Option<MoeInfo> {
    // Read GGUF header → look for expert_count kv
    // Returns MoeInfo { n_expert: 128, n_expert_used: 8 }
}
```

### Step 2: Decide — Solo vs MoE Split

In election.rs, after detecting MoE:

```
if model.is_moe && node_count >= 2 && !model_fits_single_node {
    → MoE split mode (each node gets different GGUF)
} else {
    → existing behavior (tensor split or solo)
}
```

Even if the model fits on one node, MoE split is useful with 2+ nodes because:
- Each node's GGUF is smaller → more KV cache headroom → longer contexts
- Each node runs independently → parallel sessions → higher throughput

### Step 3: Expert Analysis (one-time, cached)

First time a MoE model is deployed:
1. Run `llama-moe-analyze --export-ranking --all-layers` on the full GGUF
2. Cache the ranking CSV alongside the model: `~/.models/Qwen3-30B-A3B-Q4_K_M.ranking.csv`
3. This takes ~2-5 minutes (runs inference on 10 sample prompts, logs router decisions)

On subsequent deploys, the cached ranking is reused.

```rust
fn ensure_ranking(model_path: &Path, bin_dir: &Path) -> Result<PathBuf> {
    let ranking_path = model_path.with_extension("ranking.csv");
    if ranking_path.exists() {
        return Ok(ranking_path);
    }
    // Run moe-analyze
    let status = Command::new(bin_dir.join("llama-moe-analyze"))
        .args(["-m", model_path, "--export-ranking", &ranking_path, "--all-layers", "-n", "16"])
        .status()?;
    Ok(ranking_path)
}
```

### Step 4: Split for N Nodes

Given the ranking CSV and node count, determine expert assignment:

1. **Find minimum viable expert count** — binary search: split with N experts,
   test coherence (or use the heuristic: ~36% of total for Qwen3-style models,
   likely model-dependent so we calibrate once and cache).

2. **Compute overlap strategy:**
   - `shared_core` = top M experts (minimum viable)
   - `unique_per_node` = (total - M) / N
   - Each node gets: shared_core + unique_shard[i]

3. **Run `llama-moe-split`** for each node:
   ```
   llama-moe-split -m full.gguf --expert-list <node_i_experts> -o node-i.gguf
   ```

4. **Cache the splits** — store in `~/.models/splits/Qwen3-30B-A3B-Q4_K_M/2-nodes/`
   Invalidate when node count changes.

```rust
fn ensure_splits(
    model_path: &Path,
    ranking_path: &Path,
    n_nodes: usize,
    moe_info: &MoeInfo,
    bin_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let split_dir = model_dir.join(format!("splits/{}-nodes", n_nodes));
    if split_dir.join("node-0.gguf").exists() {
        // TODO: validate node count matches
        return Ok(collect_split_paths(&split_dir, n_nodes));
    }
    
    // Compute expert assignments from ranking
    let assignments = compute_overlap_assignments(ranking_path, n_nodes, moe_info)?;
    
    // Run moe-split for each node
    for (i, expert_list) in assignments.iter().enumerate() {
        let output = split_dir.join(format!("node-{i}.gguf"));
        Command::new(bin_dir.join("llama-moe-split"))
            .args(["-m", model_path, "--expert-list", expert_list, "-o", &output])
            .status()?;
    }
    Ok(collect_split_paths(&split_dir, n_nodes))
}
```

### Step 5: Distribute Shards to Nodes

This is the key architectural change. Currently all nodes have the same GGUF.
With MoE splitting, each node needs a DIFFERENT GGUF.

Options:
- **a) Split on host, push shards to workers via QUIC** — host runs moe-split,
  sends each worker its shard. Workers don't need the full GGUF.
- **b) Every node has the full GGUF, splits locally** — simpler but wastes storage.
  Each node runs moe-split with its assignment and discards the full model.
- **c) Host splits and serves shards via HTTP** — workers download their shard
  from the host. Like a mini CDN within the mesh.

**Recommendation: (a) for v1.** The host (highest VRAM, elected first) has the full
GGUF. It runs the ranking + split. Workers join, get assigned a shard index,
and receive their split GGUF over QUIC. Workers only store their shard.

### Step 6: Independent llama-servers

Unlike tensor-split mode (one host + RPC workers), MoE mode means:
- **Every node runs its own llama-server** with its split GGUF
- **No --rpc, no tensor splitting** — each node is fully independent
- **Session routing**: the mesh proxy hashes `session_id → node_index` and
  forwards all requests for that session to the same node
- **Each node has its own KV cache** — no cross-node state

This is a fundamentally different mode from the current host/worker model.

### Step 7: Session Routing in Proxy

```rust
// In proxy.rs — when routing a chat completion request:
fn route_moe_session(session_id: &str, moe_nodes: &[NodeId]) -> NodeId {
    let hash = hash(session_id) % moe_nodes.len();
    moe_nodes[hash]
}
```

New sessions get assigned to nodes round-robin or by hash. Once assigned,
all subsequent requests for that session go to the same node (sticky).

## Architecture Comparison

### Current: Tensor Split (dense models)
```
    Client → Proxy → Host (llama-server --rpc worker1,worker2)
                        ↕ RPC          ↕ RPC
                    Worker1          Worker2
```
- One llama-server, distributed computation
- Cross-node traffic per token (tensor activations)
- Single KV cache on host

### New: MoE Expert Split
```
    Client → Proxy ─→ Node0 (llama-server, shard-0.gguf)
                   ├→ Node1 (llama-server, shard-1.gguf)
                   └→ Node2 (llama-server, shard-2.gguf)
```
- N independent llama-servers
- Zero cross-node traffic during inference
- Per-node KV cache → N× total context capacity
- Proxy does session-sticky routing

## What Changes in mesh-llm Code

### New: `src/moe.rs`
- `detect_moe(path) → Option<MoeInfo>` — read GGUF metadata
- `ensure_ranking(path, bin_dir) → PathBuf` — run/cache moe-analyze
- `compute_assignments(ranking, n_nodes, moe_info) → Vec<Vec<expert_id>>` — overlap strategy
- `ensure_splits(path, ranking, n_nodes, bin_dir) → Vec<PathBuf>` — run/cache moe-split

### Modified: `src/election.rs`
- Detect MoE at election time
- MoE mode: don't elect a single host. Instead, every node is a "host" running
  its own llama-server with its shard.
- Coordinate shard assignment: node index based on join order or VRAM-sorted rank

### Modified: `src/proxy.rs`  
- MoE mode: session-sticky routing across N independent backends
- Parse session/conversation ID from request → hash → pick node
- Forward to that node's llama-server (via direct QUIC tunnel or local port)

### Modified: `src/download.rs` / `src/launch.rs`
- MoE split GGUFs as a download/distribution mechanism
- `start_llama_server` in MoE mode: no --rpc, just the shard GGUF

### NOT changed: `src/mesh.rs`, `src/tunnel.rs`, `src/nostr.rs`
- QUIC mesh, gossip, tunnel infrastructure stays the same
- Tunnels still used for proxy → node forwarding

## Open Questions

1. **Shard distribution**: QUIC file transfer of 10-13GB shards — how fast over local network?
   WiFi: ~30MB/s → ~6 minutes. Ethernet: 100MB/s+ → ~2 minutes. Acceptable?

2. **Node count changes**: When a 3rd node joins a 2-node MoE mesh, we need to re-split
   (new overlap strategy for 3 nodes). This means downtime during re-split + redistribution.
   Mitigation: keep serving with old shards while new ones are prepared?

3. **Minimum viable calibration**: The "36% shared core" number is Qwen3-30B-specific.
   Do we need to calibrate per model, or can we use a conservative default (50%)?
   Conservative default wastes some storage but guarantees quality.

4. **Can we skip moe-analyze?** If we default to 50% shared core (top half by gate norm —
   which IS cheap to compute from GGUF weights, no inference needed), that might be good
   enough. The ranking from actual inference is better, but adds minutes to first deploy.
   Gate norms were flat for Qwen3-30B, but might be more discriminative for other models.
