# Backend Benchmarking

Single-host comparison harness for `mesh-llm` backends.

This compares:

- `--backend llama`
- `--backend mlx`

through the same local OpenAI-compatible API on `:9337`.

## What it measures

For each backend and benchmark case:

- startup time until `/v1/models` is ready
- time-to-first-token
- total request time
- exact generated token count from a paired deterministic non-streaming usage check
- post-first-token tok/s

If a backend does not support streaming yet, the harness falls back to a non-streaming request:

- `ttft_ms` becomes `null`
- `tok_s` is computed from total request time instead of post-first-token time

The harness now also fails fast if a backend exits before it becomes ready, instead of waiting only on a port poll.

## What it does not measure

- distributed splits
- multi-node tensor/pipeline execution
- output quality
- exact apples-to-apples weights across formats

The benchmark is only as fair as the model pair you provide. In practice:

- `llama` backend uses GGUF
- `mlx` backend uses MLX-format model directories

So the most useful comparison is usually "same family, similar size, similar quantization class", not "identical bits".

## Default Cases

The default benchmark set lives in [default-cases.json](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/evals/backend-benchmarking/default-cases.json).

It now includes a mix of:

- short chat
- longer briefing-style context
- extraction from provided context
- code-oriented context

Cases are full OpenAI-style message arrays, not just one short prompt string.

## Usage

```bash
./evals/backend-benchmarking/bench-backends.sh \
  --llama-model ~/.models/Qwen3-0.6B-Q4_K_M.gguf \
  --mlx-model ~/.models/mlx/Qwen3-0.6B-bf16 \
  --runs 3
```

Or through `just`:

```bash
just bench-backends \
  ~/.models/Qwen3-0.6B-Q4_K_M.gguf \
  ~/.models/mlx/Qwen3-0.6B-bf16 \
  3
```

Run the checked-in reproducible matrix:

```bash
./evals/backend-benchmarking/run-matrix.sh --runs 3
```

Or through `just`:

```bash
just bench-backends-matrix 3
```

Use a custom cases file:

```bash
./evals/backend-benchmarking/bench-backends.sh \
  --llama-model ~/.models/Qwen3-0.6B-Q4_K_M.gguf \
  --mlx-model ~/.models/mlx/Qwen3-0.6B-bf16 \
  --cases-file ./evals/backend-benchmarking/default-cases.json
```

Append one extra simple case:

```bash
./evals/backend-benchmarking/bench-backends.sh \
  --llama-model ~/.models/Qwen3-0.6B-Q4_K_M.gguf \
  --mlx-model ~/.models/mlx/Qwen3-0.6B-bf16 \
  --case 'long-user|192|Summarize the following long note in four bullets: ...'
```

Append one full message-array case:

```bash
./evals/backend-benchmarking/bench-backends.sh \
  --llama-model ~/.models/Qwen3-0.6B-Q4_K_M.gguf \
  --mlx-model ~/.models/mlx/Qwen3-0.6B-bf16 \
  --case-json '{"label":"system-heavy","max_tokens":128,"messages":[{"role":"system","content":"Answer tersely."},{"role":"user","content":"Explain why streaming matters for latency benchmarks."}]}'
```

## Output

Results are written under:

```bash
evals/results/backend-benchmarking/<timestamp>/
```

Files:

- `results.jsonl` — one JSON object per measured run
- `summary.json` — median summary grouped by backend and case
- `llama.log` / `mlx.log` — backend startup logs

For matrix runs:

- `model-matrix.json` defines the checked-in family/model pairs
- `run-matrix.sh` runs each pair serially into its own output directory

## Notes

- The harness stops local `mesh-llm`, `mesh-llm-mlx`, `llama-server`, and `rpc-server` processes before each backend run.
- It warms each backend before recording measurements.
- It runs backends serially, never in parallel.
- It reuses the existing streaming measurement helper in `evals/latency-benchmarking/measure.py`.
- The default matrix intentionally enables only the most stable pairs by default. Experimental pairs stay in `model-matrix.json` with `"enabled": false` until the backend behavior is trustworthy.
