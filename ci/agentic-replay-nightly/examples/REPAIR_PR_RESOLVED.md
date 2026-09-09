# ci: fix decode regression in shared-prefix cache admission (agentic replay nightly — fix verified)

## Agentic replay nightly — fix verified

**Nightly run:** [Actions run #4183](https://github.com/Mesh-LLM/mesh-llm/actions/runs/4183) · 2026-09-12 · source `8e2f1a b04d`
**Regressing cohorts:** qwen25-coder-14b × c4 × decode_tokens_per_second; qwen25-coder-14b × c2 × decode_tokens_per_second

## Regression evidence

```
regressions detected:
  - qwen25-coder-14b c4: decode_tokens_per_second regressed -11.4% vs baseline median 71.80 (candidate 63.61)
  - qwen25-coder-14b c2: decode_tokens_per_second regressed -9.8% vs baseline median 55.10 (candidate 49.70)
```

## Diagnosis

opencode analysis: commit `3d7c19e` ("cache: eager prefix promotion on batch join") promotes a shared
prefix to resident KV at batch-join time. The promotion now holds the cache lock across the batched
decode step, serializing decode for requests that share a prefix — which is exactly the c2/c4 cohort
shape. c1 and c8 (queue-dominated) are unaffected, matching the cohort pattern.

## Fix

Revert to lazy promotion (on first token of the joining request) and re-acquire the cache lock
per request rather than per batch.

- Changed: `crates/skippy-cache/src/admission.rs`, `crates/mesh-llm-native-runtime/src/scheduler.rs`
- Rationale: preserves the eager-promotion correctness property from `3d7c19e` while removing the
  lock scope that serialized shared-prefix decode.

## Re-run benchmark results (HF card format)

| Model | Class | Conc. | Decode tok/s | Δ vs baseline | TTFT mean | Cache hit % | Finish=length % |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B Q4_K_M | dense | 1 | 38.6 | +0.5% | 409 ms | — (cold) | 6.0 |
| Qwen2.5-Coder-14B Q4_K_M | dense | 2 | 55.4 | +0.5% | 684 ms | 69.9 | 6.1 |
| Qwen2.5-Coder-14B Q4_K_M | dense | 4 | 72.1 | +0.4% | 1,049 ms | 71.5 | 6.2 |
| Qwen2.5-Coder-14B Q4_K_M | dense | 8 | 76.5 | +0.4% | 1,876 ms | 70.8 | 6.2 |
| gpt-oss-20b MXFP4 | moe | 1–8 | unchanged | within noise | unchanged | unchanged | unchanged |
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 1–8 | unchanged | within noise | unchanged | unchanged | unchanged |

**Gate:** PASS (bootstrap state: complete, 14 baseline-eligible runs)

## Provenance

| Field | Value |
|---|---|
| Runner | micstudio (M3 Ultra, 256 GB) — fingerprint verified |
| Nightly run | `4183` |
| History shard | `data/runs/2026-09-12/4183.jsonl` in meshllm/agentic-replay-nightly |
| Raw evidence | GitHub Actions artifact `replay-artifacts` (30-day retention) |
| Repair agent | opencode (agent mode), session log in run artifact |

## Reviewer checklist

- [x] Diagnosis names a specific commit and mechanism (not just "made it faster")
- [x] Fix does not touch `ci/agentic-replay-nightly/` baselines, thresholds, or the harness itself
- [x] Re-run results table present
- [x] `finish_reason_length_pct` did not shift — a budget change is not a throughput fix
