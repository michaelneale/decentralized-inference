# MoE Expert Sharding — Design & Status

Distribute MoE models across mesh nodes using **overlapping expert shards** with zero cross-node inference traffic. Each node holds the full trunk plus a subset of experts. Sessions are hash-routed to nodes.

See [ROADMAP.md](../../ROADMAP.md) for how this fits into mesh-llm.

## What's Implemented

All core phases are complete and integrated into mesh-llm.

### Detection (`moe.rs`)
- `detect_moe()` reads `expert_count` from GGUF header in ~1ms. Any MoE model works — no curated entry needed.
- Auto-detected in `election.rs` at model load time.

### Ranking
- **Curated models**: pre-computed expert gate mass rankings embedded from `models/metadata.toml`.
- **Cached rankings**: `moe::load_cached_ranking()` loads from `~/.models/moe-rankings/<model>.csv`.
- **Fallback**: no ranking → conservative 50% shared core with sequential expert IDs.
- **Tool**: `llama-moe-analyze` (in `llama.cpp/tools/moe-analyze/`) runs inference on sample prompts and exports per-expert gate mass CSV.

### Splitting (`moe.rs` + `llama-moe-split`)
- `compute_assignments()` implements the overlap strategy: shared core (top N experts by gate mass) replicated to every node, remaining experts distributed uniquely.
- `run_split()` calls `llama-moe-split` to produce per-node GGUFs (trunk + expert subset). Cached at `~/.models/moe-splits/<model>/<n>-nodes/node-<i>.gguf`.
- `llama-moe-split` (in `llama.cpp/tools/moe-split/`) slices expert tensors, gathers router gate rows, clamps `expert_used_count`. Supports `--groups`, `--expert-list`, `--ranking-file`.

### Mesh Integration (`election.rs`)
- `moe_election_loop()` handles the full lifecycle: detect MoE → compute assignments → split if needed → start llama-server with shard → rebuild on mesh changes.
- **Solo mode**: model fits locally → load full model, no splitting.
- **Multi-node mode**: model doesn't fit or `--split` forced → each node gets its own shard, runs its own llama-server independently.
- `moe_shard_index()` determines which shard this node gets based on sorted node IDs.
- `build_moe_targets()` publishes the MoE target map so the proxy knows all MoE nodes.

### Session Routing (`election.rs` + `proxy.rs`)
- `get_moe_target()` hashes a session hint (user field, session_id, or conversation_id) to pick a node. Pure hash — deterministic, sticky.
- `extract_session_hint()` parses the hint from HTTP request body.
- `MoeLocal` / `MoeRemote` variants in `InferenceTarget` handle local vs QUIC-tunneled forwarding.

### Tested
- OLMoE-1B-7B: 2 nodes over WAN (225ms RTT Sydney↔Sydney), both shards coherent.
- Qwen3-30B-A3B: local quality validation, 87/128 experts per node = excellent.
- GLM-4.7-Flash-Q4_K_M: MoE auto-detected (64 experts, top-4), fits locally → solo mode, no split. No pre-baked curated ranking yet — would use 50% fallback if split.

## What's NOT Implemented

### No probe-based session placement (planned)
The current design uses hash routing — sessions are assigned to nodes deterministically. The original plan proposed fan-out probes where each node scores "how well does my shard match this prompt" and the best node gets the session. This was unnecessary for the 2-node case with sufficient overlap (68%+) — both nodes produce equivalent quality. Probing becomes important with more nodes, less overlap, or sharper expert specialization. With scale testing on larger models coming soon, this is next on the list.

### No lazy `moe-analyze` for unknown models
Phase 4 in TODO. Unknown MoE models use the 50% shared core fallback with sequential expert IDs. Running `moe-analyze` automatically on first deploy (2-5 min of sample inference to compute proper rankings) is planned but not implemented. You can run it manually and the cached ranking will be picked up.

### No scale testing on large models
Phase 5 in TODO. Mixtral 8×22B (~80GB) and Qwen3-235B-A22B (~130GB) are the real targets where expert sharding provides value (models that don't fit on one machine). Not tested yet.

## Key Findings

From [MoE_SPLIT_REPORT.md](MoE_SPLIT_REPORT.md):

- **Expert 0 dominance**: In Qwen3-30B-A3B, expert 0 captures 25% of all gate mass. The top 46 experts (36%) form the minimum viable set.
- **Minimum viable threshold**: ~50% of experts needed per node for coherent output (model-dependent). Below that → degenerate loops.
- **Overlap makes probing unnecessary**: With 68%+ overlap, both nodes handle all prompt types equally well. Hash routing is sufficient.
- **Contiguous slicing is naive**: Expert importance isn't correlated with expert ID. Informed grouping (by gate mass ranking) is essential.
- **Split GGUFs are faster**: Smaller expert tensors → less memory pressure → ~8% speed improvement (110 vs 101 t/s on Qwen3-30B-A3B).

## Architecture

### MoE mode (multi-node)
```
Client → Proxy ─→ Node0 (llama-server, shard-0.gguf)
               ├→ Node1 (llama-server, shard-1.gguf)  
               └→ Node2 (llama-server, shard-2.gguf)
```
- N independent llama-servers, zero cross-node inference traffic
- Per-node KV cache → N× total context capacity
- Session-sticky hash routing

### vs Tensor split (dense models)
```
Client → Proxy → Host (llama-server --rpc worker1,worker2)
                    ↕ RPC          ↕ RPC
                Worker1          Worker2
```
- One llama-server, distributed computation
- Cross-node traffic per token (tensor activations)
- Single KV cache on host
