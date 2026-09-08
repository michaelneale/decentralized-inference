# Agentic Replay Nightly Run — 2026-09-09

- **Source:** `mesh-llm` @ `c28f0e2cc0f9a3e7d1b24c5e8f6a90d21c3b47e5` (main)
- **Runner:** `micstudio` — Mac15,14, Apple M3 Ultra, 80 GPU cores, 256 GB unified memory, macOS 26.6.2 (Metal 4)
- **Backend binary SHA-256:** `9f1c…e7a2`
- **Replay:** checkpoint profile (one request per trajectory, early/middle/late/final), 4 trajectories/framework/pass, 2 passes, warm-up 4 ordered turns discarded
- **Workload:** Thoughtworks `agentic-coding-trajectories` (pinned parquet, SHA-256 `4b2d…91c0`), 36 measured requests per model per pass
- **Gate:** PASS — all cohorts within tolerance of baseline (baseline age 11 runs)

## Per-model results

| Model | Class | Concurrency | Decode tok/s (Δ vs baseline) | E2E tok/s | TTFT mean | TTFT p90 | Cache hit % | Finish=length % | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B Q4_K_M | dense | 1 | 38.4 (+0.8%) | 31.2 | 412 ms | 618 ms | 0 (cold) | 6.1 | 0 |
| Qwen2.5-Coder-14B Q4_K_M | dense | 4 | 71.9 (-1.2%) | 58.7 | 1,054 ms | 1,873 ms | 71.4 | 6.3 | 0 |
| gpt-oss-20b MXFP4 | moe | 1 | 44.1 (-0.5%) | 35.0 | 388 ms | 590 ms | 0 (cold) | 8.9 | 0 |
| gpt-oss-20b MXFP4 | moe | 4 | 86.3 (+1.7%) | 66.1 | 961 ms | 1,690 ms | 69.8 | 9.0 | 0 |
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 1 | 24.6 (+0.3%) | 19.8 | 455 ms | 702 ms | 0 (cold) | 4.8 | 0 |
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 4 | 52.8 (+2.1%) | 40.3 | 1,190 ms | 2,214 ms | 74.2 | 4.9 | 0 |

*Decode tok/s is token-weighted after first generated content; E2E includes prompt ingestion. Δ is median-of-passes vs the reviewed baseline cohort; ±5% tolerance per metric after bootstrap. `Finish=length` is the share of responses that hit the per-turn output cap — tracked so a token-budget change cannot silently masquerade as a throughput change.*

## Trend (decode tok/s @ concurrency 4, last 7 runs)

| Run date | Qwen2.5-Coder-14B | gpt-oss-20b | Qwen3-Next-80B-A3B |
|---|---:|---:|---:|
| 09-03 | 73.6 | 84.1 | 51.4 |
| 09-04 | 72.8 | 85.0 | 51.9 |
| 09-05 | 72.4 | 85.4 | 51.6 |
| 09-06 | 73.1 | 84.8 | 51.7 |
| 09-07 | 72.7 | 85.2 | 52.0 |
| 09-08 | 72.8 | 84.9 | 51.7 |
| 09-09 | **71.9** | **86.3** | **52.8** |

## Provenance

| Field | Value |
|---|---|
| Cohort key | sha256 over (source_sha, hardware fingerprint, model hash, replay parameters) |
| Artifact | `runs/2026-09-09/` (immutable shard, retained in HF dataset + GH Actions evidence) |
| Gate decision | PASS (bootstrap complete, 11 baseline-eligible runs) |
