# MoE Expert Splitting — Reality Check

## What We Tested

Model: **Qwen3-30B-A3B-Q4_K_M** (128 experts, top-8, 17 GB)
Hardware: Apple M4 Max
Full model baseline: 101-108 t/s generation, fully coherent output

## Expert Distribution (the core problem)

Expert 0 is a super-generalist: **25% of total gate mass**, selected on **97.8% of all tokens**.
All other 127 experts share the remaining 75% roughly equally (~0.6% each).

| Top N experts | Gate mass captured |
|----|------|
| 1 (expert 0) | 25.0% |
| 32 | 46.7% |
| 48 | 56.8% |
| 64 | 66.4% |
| 96 | 84.1% |
| 128 (all) | 100% |

## What Works and What Doesn't

### ✅ Works: 64 experts (2 nodes)

| Config | Size | Speed | Quality |
|--------|------|-------|---------|
| Full model (128) | 17 GB | 101 t/s | Baseline |
| Contiguous 64-127 | 9.8 GB | 110 t/s | Coherent, minor degradation |
| Top 64 by gate mass | 9.8 GB | 111 t/s | Coherent, minor degradation |
| Ranked snake draft, 2 groups | 9.2-9.8 GB | 110 t/s | **1 of 2 groups works** |

The minimum viable split for Qwen3-30B-A3B is **2 groups of 64 experts**.
Each node saves ~45% storage (9.8 vs 17 GB) and runs ~8% faster (less memory pressure).

### ❌ Fails: 32 experts (4 nodes)

Every 32-expert configuration tested produced garbage or degenerate loops:
- Contiguous groups: 3/4 garbage, 1/4 marginal (repetitive)
- Balanced by gate norms: 4/4 garbage
- Ranked snake draft + expert 0 replicated: 4/4 garbage
- Hand-picked top 32 by gate mass: degenerate loops

**32 experts is below the minimum viable threshold for this model.**

### ❌ Fails: Ranked 2-group with snake draft

Surprisingly, the snake-draft approach (distribute hot/cold experts evenly) actually produces
**1 working group and 1 broken group** — same as naive contiguous splitting. The problem is
that expert diversity isn't captured by aggregate gate mass. Experts that look equivalent by
mass stats may have very different specializations that matter for coherent generation.

## Why This Happens

1. **Expert 0 dominance**: One expert hogs 25% of all routing. When it's in your group, the
   router over-relies on it. When it's not, the router must redistribute. Neither is great
   at small group sizes.

2. **Minimum expert diversity**: The top-8 routing selects 8 experts per token. With only 32
   available, the router repeatedly picks the same ~8 experts for every token. The softmax
   concentrates mass on a handful of choices, causing repetitive/degenerate output.

3. **Expert specialization is invisible to mass stats**: Two experts can have identical aggregate
   gate mass but handle completely different types of tokens. Splitting them into different
   groups breaks both groups for certain inputs.

## What This Means for mesh-llm

### Qwen3-30B-A3B specifically
- **2-node split is viable**: each node holds ~10 GB, generates at 110 t/s
- **4-node split is NOT viable** without additional techniques
- The 45% size reduction is nice but not transformative — this model already fits on a single M4 Max
- This model isn't the right target for distributed expert sharding (it already runs fast locally)

### Larger MoE models (the real targets)

| Model | Experts | Top-K | Total Size | Min Groups | Node Size |
|-------|---------|-------|------------|------------|-----------|
| Qwen3-30B-A3B | 128 | 8 | 17 GB | 2 | ~10 GB |
| Mixtral 8×22B | 8 | 2 | ~80 GB Q4 | 2 | ~45 GB |
| Qwen3-235B-A22B | 128 | 8 | ~130 GB Q4 | 2-4? | ~35-70 GB |
| Kimi-K2.5 | 384 | 8 | ~500 GB Q4 | 4-8? | ~65-130 GB |

The minimum viable group size (in experts) likely scales with top-k:
- **Rule of thumb**: you need ≥ 4-8× top_k experts per group for coherent output
- Qwen3-30B: top-8, minimum ~64 experts → max 2 groups from 128
- Mixtral: top-2, could work with 4 experts → max 2 groups from 8
- Larger models with more experts may split further

### What's actually needed for mesh-llm distributed inference

For models that DON'T fit on one machine, the approach changes:

1. **Pipeline parallelism (PP) is the primary axis** — split layers across nodes
2. **Expert parallelism is a secondary optimization** on top of PP
3. **The real value of expert sharding is for models with 256+ experts** (Kimi-K2.5 class)
   where you have enough expert redundancy to split 4-8 ways

## Recommendations

1. **Don't pursue expert sharding for Qwen3-30B-A3B** — it already fits and runs fast locally.
   The 2-node split works but provides minimal benefit.

2. **The tooling is ready** — `moe-split` correctly slices/gathers expert tensors, handles
   non-contiguous expert selection, reads ranking files, and produces valid loadable GGUFs.
   It's ready for larger models.

3. **Next steps should focus on**:
   - Testing on Mixtral 8×22B (8 experts, 2 groups should work well)
   - Pipeline parallelism for large models (the real scaling axis)
   - Probe-based session placement (Phase 3) — this matters more with better-separable expert pools

4. **The expert mask API (Phase 1) has standalone value** — even on a single node, masking can
   speed up inference by reducing the effective expert pool (fewer memory reads per token).

## Artifacts

- `tools/moe-split/moe-split.cpp` — GGUF expert splitter with contiguous, balanced, ranked, and custom modes
- `tools/moe-analyze/moe-analyze.cpp` — routing analysis with `--export-ranking` and `--all-layers`
- `/tmp/expert-ranking.csv` — Qwen3-30B-A3B expert gate mass rankings (all 48 layers)
- Expert mask API in llama.cpp (`llama_model_set_expert_mask()`)
