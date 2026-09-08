<!-- Published benchmark card — rendered on the HF dataset repo card and the mesh-llm repo card.
     Auto-refreshed nightly from the latest complete run; historical charts from the dataset trend. -->

# MeshLLM Coding-Agent Serving Benchmark

How fast can a Mac Studio M3 Ultra (256 GB, Metal 4) serve real agentic coding
workloads on mesh-llm? Measured nightly by replaying pinned Thoughtworks
agentic-coding trajectories (`opencode`/`goose`/`pi` framework traffic)
against each model below. Latest complete run: **2026-09-09** @
`c28f0e2cc`. Full history: [meshllm/agentic-replay-nightly](https://huggingface.co/datasets/meshllm/agentic-replay-nightly).

## Latest results — decode throughput (tok/s, concurrency 4)

| Model | Class | Decode tok/s | E2E tok/s | TTFT mean | Cache hit % |
|---|---|---:|---:|---:|---:|
| gpt-oss-20b MXFP4 | MoE | **86.3** | 66.1 | 962 ms | 69.8% |
| Qwen2.5-Coder-14B Q4_K_M | Dense | 71.9 | 58.7 | 1,054 ms | 71.4% |
| Qwen3-Next-80B-A3B Q4_K_M | Hybrid (recurrent) | 52.8 | 40.3 | 1,190 ms | 74.2% |

## Trend — decode tok/s @ concurrency 4

| | Sep 3 | Sep 4 | Sep 5 | Sep 6 | Sep 7 | Sep 8 | Sep 9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B | 73.6 | 72.8 | 72.4 | 73.1 | 72.7 | 72.8 | 71.9 |
| gpt-oss-20b | 84.1 | 85.0 | 85.4 | 84.8 | 85.2 | 84.9 | 86.3 |
| Qwen3-Next-80B-A3B | 51.4 | 51.9 | 51.6 | 51.7 | 52.0 | 51.7 | 52.8 |

*(SVG trend charts generated from the dataset by the nightly publish step.)*

## Methodology

- **Workload:** deterministic, hash-ordered sample of real agent sessions; one
  measured request per trajectory at early/middle/late/final checkpoints with
  the full recorded history as prompt; 2 passes, ABBA-ordered.
- **Server:** `mesh-llm serve --model <model> --log-format json` — no hidden
  context, KV, or backend tuning; identical across models.
- **Metrics:** token-weighted decode throughput after first generated content;
  end-to-end throughput including prefill; TTFT mean/p90; KV cache hit rate.
- **Fairness guard:** every run reports `finish_reason_length_pct` — the share
  of responses clipped by the per-turn output cap — so output-budget changes
  cannot masquerade as throughput changes.
- **Regression gate:** after a 3-run bootstrap, a cohort regressing beyond the
  reviewed per-metric tolerance breaks the nightly and opens a repair PR.

## Provenance

Every row in the history dataset carries source SHA, backend binary hash,
hardware fingerprint, model file SHA-256, dataset hash, and replay parameters.
Comparisons are made only between exact cohort matches.
