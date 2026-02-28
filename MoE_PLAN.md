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

## Phase 2: Per-Node GGUF Packaging

**Goal**: Build offline tooling to produce one GGUF per node (trunk + expert group).

### Steps

1. **Inspect Mixtral safetensors from HF**
   - `mistralai/Mixtral-8x7B-Instruct-v0.1` ships as sharded safetensors
   - Expert tensors: `model.layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}.weight`
   - Gate: `model.layers.{L}.block_sparse_moe.gate.weight`
   - Everything else: trunk

2. **Write repacker** (Python or Rust)
   - Read `model.safetensors.index.json` → classify trunk vs expert by name regex
   - Partition experts into N groups (e.g., 2 groups: {0,1,2,3} and {4,5,6,7})
   - Output: `trunk.safetensors`, `experts_group0.safetensors`, `experts_group1.safetensors`
   - Optional: `experts_replicated.safetensors` (hot experts)

3. **Convert to GGUF**
   - For each node: merge trunk + experts_groupN → convert → quantize → `nodeN.gguf`
   - Each node loads one self-contained GGUF (no llama.cpp loader changes needed)

4. **Distribution manifest**
   ```json
   {
     "model": "Mixtral-8x7B-Instruct-v0.1",
     "architecture": "moe",
     "n_experts": 8, "top_k": 2,
     "groups": {
       "0": {"experts": [0,1,2,3], "file": "node0.gguf"},
       "1": {"experts": [4,5,6,7], "file": "node1.gguf"}
     },
     "replicated_experts": [2, 5],
     "routing": "masked-same-group-topk"
   }
   ```

---

## Phase 3: Mesh Integration (2+ nodes)

**Goal**: Run Mixtral across mesh nodes with session pinning and probe-based placement.

### Steps

1. **Session placement in mesh-llm**
   - New concept: `expert_group` per session
   - Probe: on first request, send first 16–32 tokens to candidate nodes
   - Each node scores: `S = sum(top_k(softmax(gate_logits) * mask))`
   - Pin session to best-scoring node

2. **Expert masking enforced by GGUF contents**
   - Each node's GGUF only contains its group's experts
   - Still need `--expert-mask` or equivalent so router logits are properly masked

3. **Online gate logging for hot-expert discovery**
   - Each node logs unmasked gate top-k for early MoE layers during normal serving
   - Periodically aggregate → update replicated expert set → rebuild overlay

---

## Phase 4: Scale to Larger MoE

Move to models where expert sharding actually matters:

| Model | Total | Active | Experts | Groups needed |
|-------|-------|--------|---------|---------------|
| Mixtral 8×7B | 47B | ~13B | 8 | 2 |
| Qwen3-30B-A3B | 30B | 3B | many | 4–8 |
| Mixtral 8×22B | 141B | ~44B | 8 | 2–4 |
| Qwen3-235B-A22B | 235B | ~22B | 128 | 8–16 |
| Kimi-K2.5 | 1T | ~32B | 384 | 12–16 |

Same pipeline: inspect safetensors → classify → partition → per-node GGUF → masked routing + session pinning.

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
| Expert distribution | Non-all-to-all, always | RTT over mesh/WiFi/WAN kills interactive chat |
| Routing constraint | Group-masked top-k, same group | Bounded fanout by construction |
| Session binding | Sticky to one node/group | No per-token cross-host traffic |
| Trunk placement | Replicated on every node | Keeps attention+KV local |
| Placement method | Probe: score router mass per group | Content-aware, cheap |
| Hot expert discovery | Offline calibration + online gate logging | Identify generalists for replication |
| Packaging | Per-node GGUF (trunk + group + replicas) | Works with existing llama-server |
| Starter model | Qwen3-30B-A3B | 128 experts (realistic group sizes), well-supported, 17GB Q4 |
