# ci: agentic replay nightly regression — run 4191 — NEEDS ATTENTION

## Agentic replay nightly — NEEDS ATTENTION

**Nightly run:** [Actions run #4191](https://github.com/Mesh-LLM/mesh-llm/actions/runs/4191) · 2026-09-13 · source `f10c88e2`
**Regressing cohorts:** qwen3-next-80b-a3b × c1 × decode_tokens_per_second AND ttft_ms_mean; qwen3-next-80b-a3b × c2 × decode_tokens_per_second

## Regression evidence

```
regressions detected:
  - qwen3-next-80b-a3b c1: decode_tokens_per_second regressed -18.2% vs baseline median 24.60 (candidate 20.12)
  - qwen3-next-80b-a3b c1: ttft_ms_mean regressed +24.7% vs baseline median 455.0 (candidate 567.4)
  - qwen3-next-80b-a3b c2: decode_tokens_per_second regressed -15.9% vs baseline median 39.20 (candidate 32.97)
```

## Diagnosis

opencode analysis (low confidence): the regression appears with the hybrid-recurrent model at low
concurrency only, suggesting the recurrent-state (`KvRecurrent`) checkpoint path added in `a94e77b`
("recurrent: state checkpointing on prefix boundaries"). The attempted fix (avoid re-copying the SSM
state on cache-identical prefixes) did not clear the regression: c1 decode improved only -18.2% →
-14.1%, still outside tolerance. The agent could not isolate the remaining cost without deeper
profiling of the Metal recurrent kernels.

## Fix

Attempted (retained on this branch for reference): skip redundant SSM state copy when the prefix
cache identity matches. Insufficient — see re-run below. **Do not merge as-is; the diagnosis is
incomplete.**

- Changed: `crates/skippy-cache/src/recurrent.rs` (attempted)
- Rationale: partial mitigation only

## Re-run benchmark results (HF card format)

| Model | Class | Conc. | Decode tok/s | Δ vs baseline | TTFT mean | Cache hit % | Finish=length % |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 1 | 21.1 | **-14.1%** | 549 ms | — (cold) | 4.8 |
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 2 | 34.9 | **-11.0%** | 941 ms | 73.8 | 4.9 |

**Gate:** FAIL (bootstrap state: complete) — regression unresolved after automated repair attempt

## Provenance

| Field | Value |
|---|---|
| Runner | micstudio (M3 Ultra, 256 GB) — fingerprint verified |
| Nightly run | `4191` |
| History shard | `data/runs/2026-09-13/4191.jsonl` in meshllm/agentic-replay-nightly |
| Raw evidence | GitHub Actions artifact `replay-artifacts` (30-day retention) |
| Repair agent | opencode (agent mode), session log in run artifact |

## Reviewer checklist

- [ ] Diagnosis names a specific commit and mechanism — **partial; needs human profiling of Metal recurrent kernels**
- [x] Fix does not touch `ci/agentic-replay-nightly/` baselines, thresholds, or the harness itself
- [ ] Re-run results show the gate still failing — **needs attention**
- [x] `finish_reason_length_pct` stable (4.8–4.9%) — not an output-budget artifact
