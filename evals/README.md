# mesh-llm Router Evals

## Compare Mesh releases and inference engines

`agentic-replay.py` is the durable entrypoint for comparing two or more Mesh
refs and, optionally, external llama.cpp, vLLM, and SGLang arms on one model.
It creates isolated detached worktrees, builds each Mesh release host and native
runtime, replays a pinned subset of the Thoughtworks
agentic-coding trajectories, and produces raw JSONL, CSV/JSON/Markdown tables,
SVG throughput and TTFT charts, logs, binary hashes, and an artifact inventory.

Inspect the exact build order and launch command without changing anything:

```bash
python3 evals/agentic-replay.py plan \
  --ref stable=v0.75.1 \
  --ref main=origin/main \
  --model '<model-uri>' \
  --trajectories-per-framework 4
```

Run the default fast A/B release gate after materializing the pinned parquet from
`thoughtworks/agentic-coding-trajectories`:

```bash
python3 evals/agentic-replay.py run \
  --ref stable=v0.75.1 \
  --ref main=origin/main \
  --model '<model-uri>' \
  --trajectories-per-framework 4 \
  --dataset-file /path/to/sessions.parquet \
  --output /path/to/artifact
```

The server command is always `mesh-llm serve --model <model> --log-format
json`. The runner never sets context size, Mesh execution lanes, KV budget, or
backend tuning; `--concurrency` controls simultaneous client requests only.
Trajectory count is deliberately explicit rather than hidden behind a default.
Each measured cohort must contain at least twice the maximum offered client
concurrency so the high-concurrency cells have more than one worker wave. With
the example above and client concurrency 1/2/4, the runner selects 36 measured
whole trajectories: four from each of the three recorded agent frameworks for
each disjoint concurrency cohort. It also selects a disjoint 12-trajectory
warm-up cohort and discards four ordered turns after every model-ready event.
Selection is deterministic by session ID hash within each framework.

The default checkpoint profile measures one request per trajectory: within each
framework's four trajectories, one deterministically represents each of early,
middle, late, and final stage. Every request contains the complete recorded
history before that checkpoint. Skipped assistant turns and tool observations
are appended from the dataset in strict order, so every experiment arm receives
an identical prefix without paying for discarded generations at every step.
This is 36 requests per ref and 72 for a two-ref A/B gate. Use `--passes 2` for
reverse-order ABBA confirmation when the first-pass result is ambiguous.

For nightly or research runs, `--replay-mode all --passes 2 --warmup-turns 14`
restores exhaustive assistant-turn replay. The manifest records all selected
session IDs, framework and turn counts, context bounds, and hashes. Use `report`
to regenerate tables and charts from a completed artifact without rerunning the
model.

The default per-turn output cap is 2,048 tokens. Reports lead with token-weighted
decode throughput measured after first generated content and show end-to-end
output throughput separately because it includes prompt ingestion. They also
show realized mean in-flight concurrency, slot utilization, failed requests,
and the per-pass range. Percent deltas are suppressed unless compared arms
fail the same request IDs.

This is a deterministic hash-ordered stress sample, not a stratified sample of
the corpus. Tool names are preserved but tool schemas are permissive stubs, and
generated output is measured but not fed back into the replay. The benchmark is
therefore a paired serving-performance comparison on reconstructed agent
prompts; it does not measure answer quality or claim byte-identical production
traffic. Treat small deltas as directional unless a larger cohort and repeated
passes show a consistent separation.

### KV-cache acceptance with captured harness traffic

The Thoughtworks corpus is useful for broad agentic load, but it is not a
substitute for the exact Buzz, OpenCode, Goose, or Pi prefix whose cache behavior
is being certified. `run` therefore also accepts `--trajectory-manifest` in
place of `--dataset-file`. Captured trajectories may include their exact `tools`
array; the runner sends those schemas unchanged because tool-schema tokens are
part of the reusable prefix identity.

Use `--replay-mode final` to measure each captured trajectory at its longest
recorded prefix. The optional gates make a qualifying run fail closed while
retaining its complete artifact:

```bash
python3 evals/agentic-replay.py run \
  --ref parent=<parent-commit> \
  --ref candidate=<candidate-commit> \
  --model '<model-uri>' \
  --trajectory-manifest /path/to/captured-harnesses.json \
  --replay-mode final \
  --passes 2 \
  --require-framework buzz \
  --require-framework opencode \
  --require-framework goose \
  --prompt-token-range 18000:22000 \
  --min-cache-pct 70 \
  --require-output-match \
  --max-ttft-regression-pct 5 \
  --output /path/to/artifact
```

The manifest uses the same `warmup`, `1`, `2`, and `4` cohort names as the
generated workload. Each trajectory has `session_id`, `source_dataset`,
`agent_framework`, nullable `recorded_model`, ordered `messages`, and an
optional exact OpenAI `tools` array. Include real captures from the required
harnesses as distinct `agent_framework` values. Do not commit private captures;
the artifact records and hashes the supplied manifest for reproducibility.

Each repeated `--require-framework` value must be represented in every measured
concurrency cohort. This preflight runs before any ref is built, preventing a
long acceptance run from silently omitting Buzz or one of the comparison agent
harnesses.

`--prompt-token-range` is checked against server-reported token counts for every
successful measured request, not a character estimate. `--min-cache-pct` checks
the aggregate server-reported cached-token share in every ref/concurrency cell.
`--require-output-match` compares generated-content hashes and failure identities
across refs and passes. `--max-ttft-regression-pct` bounds candidate median TTFT
relative to the first ref. Any configured failure is written into `run.json` and
the Markdown report before the command exits non-zero.

### Compare Mesh with llama.cpp, vLLM, and SGLang

Pass `--engine-config` to append external OpenAI-compatible server arms to the
same ordered replay. The two Mesh refs remain required, so release-versus-main
and cross-engine comparisons share one manifest, client, request order,
sampling parameters, warm-up policy, and report. Arms run sequentially on the
same host and port.

The JSON config is intentionally explicit. Each arm records its executable,
model and tokenizer locations, context capacity, maximum server concurrency,
cache policy, and extra launch arguments. Relative executable paths resolve the
way the launch runs them: a path with a separator is read against the arm's
`cwd`, a bare name is searched on the runner `PATH`. Before launch, the runner
queries the engine's reported version (`--version`, or Python package metadata
for SGLang, bounded by a 30-second timeout) on that same resolved executable
and uses SHA-256 of that version string as the arm identity. It does not gate
on executable, model, or tokenizer file hashes.

```json
{
  "schema_version": 1,
  "comparison": {"model": "hf://org/model"},
  "arms": [
    {
      "label": "llama-cpp",
      "engine": "llama.cpp",
      "executable": "/opt/llama.cpp/llama-server",
      "model": "/models/model.gguf",
      "tokenizer": "/models/tokenizer",
      "context_size": 131072,
      "max_concurrency": 8,
      "batch_size": 2048,
      "ubatch_size": 128
    },
    {
      "label": "vllm",
      "engine": "vllm",
      "executable": "/opt/vllm/bin/vllm",
      "model": "/models/model-vllm",
      "tokenizer": "/models/tokenizer",
      "context_size": 131072,
      "max_concurrency": 8
    },
    {
      "label": "sglang",
      "engine": "sglang",
      "executable": "/opt/sglang/bin/python",
      "model": "/models/model-sglang",
      "tokenizer": "/models/tokenizer",
      "context_size": 131072,
      "max_concurrency": 8
    }
  ]
}
```

Inspect the exact five-arm ABBA plan before running it:

```bash
python3 evals/agentic-replay.py plan \
  --ref stable=v0.75.1 \
  --ref main=origin/main \
  --engine-config /path/to/engines.json \
  --model 'hf://org/model' \
  --trajectory-manifest /path/to/captured-harnesses.json \
  --concurrency 8 \
  --passes 2
```

Absolute latency and throughput remain visible for every arm. Relative deltas
stay fail-closed unless successful request IDs, failed request IDs, and generated
content hashes match the first arm, preventing output divergence from being
presented as a paired speedup. vLLM and SGLang normally require a Linux/CUDA
host; the config deliberately does not hide that environmental difference.

For the pinned Mesh-versus-raw-llama.cpp scheduler matrix across CUDA, Metal,
dense/MoE/recurrent/hybrid models, llama-benchy, and Thoughtworks agent traces, see
[`docs/skippy/COMPETITIVE_BENCHMARK.md`](../docs/skippy/COMPETITIVE_BENCHMARK.md).

A/B comparison of pi agent performance through mesh-llm's multi-model router vs a frontier cloud model.

## Setup

### Mesh (local multi-model)
```bash
# 3 models on M4 Max 52GB (~27GB total, room for KV cache)
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm \
  --model Qwen2.5-32B-Instruct-Q4_K_M \
  --model Qwen2.5-Coder-7B-Instruct-Q4_K_M \
  --model Hermes-2-Pro-Mistral-7B-Q4_K_M
```

Router auto-classifies each request and picks the best model:
- **Qwen2.5-32B** (tier 3) — reasoning, chat, complex code, tool use
- **Qwen2.5-Coder-7B** (tier 2) — code generation/review, fast (85 tok/s)
- **Hermes-7B** (tier 2) — fast chat, simple Q&A (87 tok/s, no tool use)

`MESH_LLM_EPHEMERAL_KEY=1` uses a fresh identity so no external peers connect.

### Cloud baseline
Sonnet via `pi --provider anthropic --model claude-sonnet-4-20250514`.

## Scenarios

Multi-turn conversations that start with chat and progress to tool use:

| Scenario | Turns | What it tests |
|---|---|---|
| **chat-to-code** | 4 | Chat→write code→write tests→review (router must switch models) |
| **debug-session** | 4 | Read files→run code→find/fix bugs→verify (tool-heavy) |
| **edit-file** | 3 | Analyze→multi-step edits→verify (structured editing) |
| **html-app** | 3 | Generate code→validate→iterate (code generation) |
| **explore-repo** | 4 | Bash tools→read files→summarize (repo navigation) |
| **refactor** | 3 | Code review→refactor→verify (code quality) |

## Running

### Multi-turn (recommended — realistic)
```bash
# Single scenario
./evals/run-multi.sh mesh chat-to-code
./evals/run-multi.sh opus chat-to-code

# Compare results
./evals/compare.sh chat-to-code
```

### One-shot (quick, less realistic)
```bash
./evals/run.sh mesh edit-file
./evals/run.sh opus edit-file
```

## Results

Results go to `evals/results/<provider>/<scenario>/`:
- Working files (copied from scenario, edited by agent)
- `_output.txt` — full session capture
- `_screen_turnN.txt` — screen state after each turn
- `_time.txt` — wall clock seconds
- `_turns.txt` — number of turns completed

## What to look for

1. **Correctness** — Did it complete all turns? Are edits right?
2. **Tool use** — Did it use read/edit/bash appropriately?
3. **Routing** — Check `/tmp/mesh-llm-local.log` for which model handled each turn
4. **Speed** — Wall clock per scenario
5. **Model switching** — Does quality degrade when router changes models mid-conversation?
6. **Chat quality** — Are quick chat responses from Hermes comparable to 32B?

## Model capabilities (from testing)

| Model | Tool use | Code gen | Chat | Speed |
|---|---|---|---|---|
| Qwen2.5-32B | ✅ works | ✅ good | ✅ good | ~18 tok/s |
| Qwen2.5-Coder-7B | ✅ works | ✅ great | ⚠️ ok | ~85 tok/s |
| Hermes-7B | ❌ broken | ⚠️ basic | ✅ fast | ~87 tok/s |
| Qwen3-30B-A3B | ❌ thinking format | ✅ good | ❌ empty content | ~22 tok/s |
