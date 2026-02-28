# MoE Expert Sharding Plan

Distribute MoE models across mesh nodes using **masked expert groups** with bounded network fanout (≤1 remote host per token per MoE layer, ideally 0). Each node holds the full trunk (attention + norms + router + embeddings + head) plus only its assigned expert shard. Sessions are pinned to nodes so decode runs entirely local.

See [MoE.md](MoE.md) for full design discussion. See [ROADMAP.md](ROADMAP.md) for how this fits into mesh-llm.

## Architecture Summary

```
Session starts → fan-out probe to N nodes → each scores "how well does my expert
shard match this prompt?" using unmasked router logits → pick best → pin session there

During decode: trunk + local experts only. No cross-host traffic per token.
```

Each node stores:
- **Trunk** (replicated everywhere): embeddings, attention, norms, router/gate, output head
- **Expert shard** (unique per node): 1/N of the expert FFN weights
- **Replicated "hot" experts** (optional overlay): small set of generalist experts on every node

Quality tradeoff: restricting routing to a subset of experts means the model can't always pick its "best" expert. Mitigated by: larger groups, replicated generalists, probe-based placement, same-group top-k.

---

## Phase 1: Single-Machine Masking Proof (Mixtral 8×7B) ← **START HERE**

**Goal**: Prove that restricting expert routing to a subset doesn't catastrophically degrade quality.

### Steps

1. **Get Mixtral 8×7B Instruct GGUF**
   - Download a Q4_K_M or Q5_K_M quant (fits easily on any dev machine)
   - Confirm it runs in our llama.cpp fork

2. **Understand the MoE routing code**
   - `build_moe_ffn()` in `llama-graph.cpp:1011` is the entry point
   - Gate logits → softmax/sigmoid → optional group masking → `argsort_top_k` → expert selection
   - llama.cpp already has group-masking for DeepSeek-V3 (`n_expert_groups > 1`)
   - Mixtral has `n_expert=8, n_expert_used=2, n_expert_groups=0` (no built-in groups)

3. **Add `--expert-mask` flag**
   - New CLI arg: `--expert-mask 0,1,2,3` (comma-separated list of allowed expert IDs)
   - Before `argsort_top_k`, set logits for masked-out experts to `-INFINITY`
   - This is ~10 lines of code in `build_moe_ffn()`

4. **Add unmasked gate logging**
   - Before masking, log which experts the router wanted and their gate weights
   - Only for first D MoE layers (D=2..4) to avoid bias from masked hidden states
   - Output: CSV or JSON with `layer_id, token_pos, expert_id, gate_weight`

5. **Run A/B comparison**
   - Prompt suite: 10 code + 10 chat + 10 reasoning + 10 instruction = 40 prompts
   - Generate 64 tokens per prompt under each condition:
     - **(A)** Baseline: no mask (all 8 experts)
     - **(B)** Masked: experts {0,1,2,3} only (50% restriction)
     - **(C)** Masked + replica: experts {0,1,2,3} + 1 hot expert from {4,5,6,7}
   - Compare average log-probability per condition
   - Log unmasked gate stats to identify hot experts

### Results (Qwen3-30B-A3B, 128 experts, top-8)

**Switched from Mixtral 8×7B to Qwen3-30B-A3B** — 128 experts gives much more realistic group sizes
for the target deployment scenario (vs Mixtral's 8 experts).

Routing analysis tool (`llama-moe-analyze`) built and run on 10 diverse prompts × 32 tokens:

**Expert popularity**: Expert 0 dominates at 32.6% of total gate mass (likely a "generalist").
Top 4 experts capture 34.9%, top 8 capture 37.7%, top 32 capture 52.4% of total mass.

**Group capture ratios** (best single group's top-8 mass / unrestricted top-8 mass):

| Groups | Experts/Group | Mean  | P25   | P50   | P5    |
|--------|--------------|-------|-------|-------|-------|
| 2      | 64           | 0.997 | 1.000 | 1.000 | 1.000 |
| 4      | 32           | 0.995 | 1.000 | 1.000 | 1.000 |
| 8      | 16           | 0.993 | 1.000 | 1.000 | 1.000 |

**Key finding**: Even with 8 groups (16 experts per group), the best group captures 99.3% of the
unrestricted top-8 mass on average, with P5 = 1.000. **Expert replication adds negligible benefit**
because capture is already near-perfect.

**Conclusion**: For this model, restricting routing to 1/N of the expert pool barely hurts router
mass capture. The experts are diverse enough that any reasonably-sized group contains good options.
Next step: verify this translates to actual generation quality (logprob comparison with masking).

### Deliverables
- ✅ Routing analysis tool (`tools/moe-analyze/`)
- ✅ Gate frequency/mass data showing expert popularity
- ✅ Group capture ratio analysis across different group sizes
- ✅ Quality delta via actual generation with masked routing (Phase 1b)
- ✅ `llama_model_set_expert_mask()` API added to llama.cpp

**Phase 1b results** (4 groups of 32 experts + 2 hot replicas, 5 prompts × 32 tokens):

| Group | Experts | Avg logprob delta |
|-------|---------|-------------------|
| 0 | 0-31 + hot | -0.945 |
| 1 | 32-63 + hot | -2.292 |
| 2 | 64-95 + hot | **-0.105** |
| 3 | 96-127 + hot | -0.829 |

**Key findings**:
1. **Routing mass capture (1a) is necessary but not sufficient** — 99%+ mass capture
   doesn't guarantee low quality loss. Actual generation testing is essential.
2. **Quality varies dramatically across groups** — Group 2 barely degrades (-0.1) while
   Group 1 is heavily degraded (-2.3). This confirms probe-based placement is critical.
3. **The best group is nearly lossless** — if sessions are placed on their best-matching
   group, quality loss can be very small. This validates the "fan-out probe then pin" approach.
4. Next: try larger groups (2 groups of 64), try more diverse placement scoring, and
   test with more prompts to understand variance.

---

## Phase 2: Per-Node GGUF Packaging ✅

**Goal**: Build offline tooling to produce one GGUF per node (trunk + expert group).

### Implementation: `tools/moe-split/moe-split.cpp`

Native C++ tool that operates directly on GGUF files (no Python/safetensors dependency).

**Capabilities**:
- Reads MoE GGUF → classifies tensors as trunk vs expert
- Slices packed expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) along expert dim
- Gathers router gate rows (`ffn_gate_inp`) to match selected experts
- Updates `{arch}.expert_count` and clamps `expert_used_count` in metadata
- Supports contiguous groups (`--groups N`) and custom expert lists (`--expert-list 64,65,72,...`)
- Non-contiguous gather path for informed expert assignment from moe-analyze data

**Usage**:
```bash
# Split into 4 contiguous groups
llama-moe-split -m model.gguf -g 4 --output-prefix node
# → node-0.gguf, node-1.gguf, node-2.gguf, node-3.gguf

# Single group
llama-moe-split -m model.gguf -g 4 --group-id 2 -o node2.gguf

# Custom expert selection (informed grouping from moe-analyze)
llama-moe-split -m model.gguf --expert-list 64,65,72,80,3,17 -o custom.gguf

# Dry run
llama-moe-split -m model.gguf -g 8 --dry-run
```

### Results (Qwen3-30B-A3B-Q4_K_M, 128 experts, top-8)

| Metric | Value |
|--------|-------|
| Input size | 17 GB |
| Output per node (4 groups) | 5.35 GB |
| Expert tensors sliced | 144 (48 layers × 3 tensors) |
| Router gates gathered | 48 (one per layer) |
| Generation speed | 104-106 t/s (vs 101 t/s full model) |

**Quality by contiguous group** (4 groups of 32 experts):

| Group | Experts | Quality |
|-------|---------|---------|
| 0 | 0-31 | ❌ garbage |
| 1 | 32-63 | ❌ garbage |
| **2** | **64-95** | **✅ coherent** (106.4 t/s) |
| 3 | 96-127 | ❌ garbage |

**Key findings**:
1. Split GGUFs load and run in unmodified llama.cpp — no loader changes needed
2. Quality varies dramatically by group (matches Phase 1b logprob analysis)
3. Contiguous slicing is a naive baseline — **informed grouping** (using moe-analyze
   data to select balanced expert mixes) is essential for production
4. Router gate rows are correctly gathered/reordered — expert renumbering works
5. Generation speed is actually slightly faster (smaller expert tensors → less memory pressure)

### Phase 2b: Overlap Strategy (validated on Qwen3-30B-A3B)

The key finding from Phase 2: **disjoint expert partitioning doesn't work**. You need a
shared core of hot experts replicated to every node, with unique long-tail experts split across nodes.

**Expert routing is dominated by a small core:**
- Expert 0: 25% of all gate mass, selected on 97.8% of tokens
- Top 46 experts (36%): minimum viable set for coherent output
- Remaining 82 experts: 44.4% of gate mass, individually ~0.5% each

**Validated overlap deployment (2 nodes):**

| Config | Per-node | Quality | Coverage |
|--------|----------|---------|----------|
| 87 experts (46 shared + 41 unique) | 12.9 GB | ✅ both excellent | 128/128 |
| Top-64 by mass | 9.8 GB | ✅ excellent | 64/128 |
| Bottom-64 by mass | 9.8 GB | ❌ garbage | 64/128 |
| Top-48 | 7.5 GB | ✅ coherent | 48/128 |
| Top-44 | ~7 GB | ❌ loops | 44/128 |

**The overlap strategy:**
1. `moe-analyze --export-ranking --all-layers` → get per-expert gate mass ranking
2. Shared core = top N experts (N ≈ minimum viable, ~36% for this model)
3. Each node = shared core + unique shard of remaining experts
4. No prompt probing needed — both nodes handle everything, hash-route sessions
5. Full expert coverage across the fleet

**Probing is NOT required** for the 2-node case with 68% overlap. Both nodes produce
equivalent quality on all tested prompts (general chat, code, math, translation, reasoning).
Probing becomes relevant only with more nodes / less overlap / sharper expert specialization.

**Tools updated:**
- `moe-analyze --export-ranking --all-layers` — exports gate mass CSV
- `moe-split --ranking-file FILE --replicate N` — ranking-based snake draft with hot replication
- `moe-split --expert-list` — arbitrary expert selection for custom splits

---

## Phase 3: Mesh Integration (2+ nodes) ← **NEXT**

**Goal**: Wire the split GGUFs into mesh-llm so two real nodes serve sessions.

### What's needed

1. **moe-split automation in mesh-llm**
   - Model catalog knows the full GGUF + ranking CSV
   - On `mesh deploy`, split GGUFs for each node based on node count
   - Push per-node GGUF to each node (or have nodes pull their shard)

2. **Session routing**
   - For N=2 with overlap: simple hash routing (both nodes are equivalent)
   - No probing infrastructure needed for v1
   - Session sticks to the assigned node for its lifetime (KV cache is local)

3. **llama-server per node**
   - Each node runs stock llama-server with its split GGUF
   - No code changes to llama-server — the split GGUF handles expert restriction
   - mesh-llm proxy routes incoming requests to the assigned node

4. **End-to-end test**
   - Two machines, each loading a split Qwen3-30B-A3B GGUF
   - Chat sessions routed via mesh-llm proxy
   - Verify: both nodes respond coherently, sessions are sticky, no cross-node traffic

### What's NOT needed for v1
- Probe-based placement (both nodes equivalent with overlap)
- Online gate logging / hot expert discovery
- Dynamic expert rebalancing

---

## Phase 4: Scale to Real Targets

Models where expert sharding provides actual value (model doesn't fit on one machine):

| Model | Experts | Top-K | Size (Q4) | Est. min core | Nodes | Per-node |
|-------|---------|-------|-----------|---------------|-------|----------|
| Mixtral 8×22B | 8 | 2 | ~80 GB | ~4 experts | 2 | ~45 GB |
| Qwen3-235B-A22B | 128 | 8 | ~130 GB | ~46? | 2-4 | ~35-70 GB |
| DeepSeek-V3 | 256 | 8 | ~350 GB | TBD | 4-8 | ~50-90 GB |
| Kimi-K2.5 | 384 | 8 | ~500 GB | TBD | 4-8 | ~65-130 GB |

**Same pipeline**: `moe-analyze` → ranking → `moe-split` with overlap → deploy to nodes.

Key question for larger models: does the "shared core" percentage shrink as expert count grows?
If so, overlap cost decreases and more nodes become viable. Need to test.

---

## Phase 5 (Later): PP for Trunk + Expert Groups

If trunk doesn't fit on one node (possible for Kimi-K2.5 at 1T total):
- Split trunk layers across PP stages
- Each PP stage holds its trunk layers + experts for the pinned group in those layers
- MoE never adds extra hops beyond PP activation handoff

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Expert distribution | Overlapping shared core + unique shards | Disjoint fails; overlap makes every node viable |
| Routing constraint | Router naturally adapts to available experts | Split GGUF only contains node's experts |
| Session binding | Hash-route, sticky to one node | Both nodes equivalent with overlap; no probing needed |
| Trunk placement | Replicated on every node | Keeps attention+KV local |
| Placement method | Hash (v1), probe later if needed | Overlap makes nodes equivalent for general use |
| Hot expert discovery | Offline: `moe-analyze --export-ranking` | Determines shared core + unique assignment |
| Packaging | Per-node GGUF (trunk + shared core + unique shard) | Works with existing llama-server |
| Starter model | Qwen3-30B-A3B | 128 experts (realistic group sizes), well-supported, 17GB Q4 |
