# Agentic Replay Nightly Run — 2026-09-09

> Illustrative layout only. These values are not observed benchmark evidence.

- **Source:** `mesh-llm` @ `c28f0e2cc0f9a3e7d1b24c5e8f6a90d21c3b47e5` (main)
- **Runner:** `micstudio` — Mac15,14, Apple M3 Ultra, 80 GPU cores, 256 GB unified memory, macOS 26.6.2 (Metal 4)
- **Backend binary SHA-256:** `9f1c…e7a2`
- **Replay:** checkpoint profile (one request per trajectory, early/middle/late/final), 8 trajectories/framework/pass, 2 passes, warm-up 4 ordered turns discarded
- **Workload:** Thoughtworks `agentic-coding-trajectories` (pinned parquet, SHA-256 `4b2d…91c0`), 72 measured requests per model per pass (8 per framework x 3 frameworks x 3 concurrency cohorts)
- **Gate:** PASS — all cohorts within tolerance of baseline (baseline age 11 runs)

## Per-model results

| Model | Class | C1 decode (Δ) | C1 TTFT | C2 decode (Δ) | C4 decode (Δ) | C4 TTFT | C8 decode (Δ) | C8 TTFT | Cache hit % (c4) | Finish=length % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B Q4_K_M | dense | 38.4 (+0.9%) | 412 ms | 55.1 (+1.0%) | 71.9 (-1.2%) | 1,054 ms | 76.2 (-2.9%) | 1,884 ms | 71.4 | 6.3 |
| gpt-oss-20b MXFP4 | moe | 44.1 (-0.4%) | 388 ms | 63.7 (+0.6%) | 86.3 (+1.7%) | 961 ms | 91.0 (+0.8%) | 1,742 ms | 69.8 | 9.0 |
| Qwen3-Next-80B-A3B Q4_K_M | hybrid-recurrent | 24.6 (+0.3%) | 455 ms | 39.2 (+0.9%) | 52.8 (+2.1%) | 1,190 ms | 54.3 (+1.2%) | 2,290 ms | 74.2 | 4.9 |

Note the saturation curve: throughput climbs steeply from c1→c4 (batching
gains) then flattens at c8 — on a single M3 Ultra the decode-bound region
begins around 4–8 in-flight requests depending on model class. That knee is
itself a tracked signal: a regression that only appears at c8 means scheduler
admission/queueing; one that appears at c1 is pure kernel/attention-path.

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
