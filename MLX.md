# MLX Notes

As of 2026-03-28, MLX is the long-term backend direction for Apple Silicon, but it is not yet at feature parity with the current `llama.cpp` path.

## Current Direction

- `vMLX` has been removed from the repo plan and runtime path.
- `--backend mlx` is intended to use our own MLX worker binary: `mesh-llm-mlx`.
- The current native worker remains narrow:
  - macOS only
  - local-host serving
  - early family support (`gemma2`, `gemma3_text`, `llama`, `mistral`, `qwen2`, `qwen3`)
  - no distributed tensor parity yet

## Why MLX Still Matters

MLX gives us the substrate we need on Apple Silicon:

- local inference
- `mx.distributed`
- `ring` and `jaccl` communication backends
- tensor/pipeline parallel building blocks

Useful references:

- [MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX tensor parallelism example](https://ml-explore.github.io/mlx/build/html/examples/tensor_parallelism.html)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [exo](https://github.com/exo-explore/exo)

## Current Native Path

Build the native worker:

```bash
just build-mlx
```

Run the MLX worker unit tests:

```bash
just test-mlx
```

This currently includes model-family regression tests for every supported family in the worker.

Run real local inference smoke tests:

```bash
just test-mlx-inference
```

Run one family at a time:

```bash
just test-mlx-family real_mistral_inference_smoke
```

For the Llama cache path specifically:

```bash
just test-mlx-family real_llama_prefix_cache_extension_smoke
```

These inference recipes run serially with `--test-threads=1` because the shared MLX runtime is not stable under parallel Rust test execution.

The ignored suite currently covers:

- real inference smoke tests for `gemma2`, `gemma3_text`, `llama`, `mistral`, `qwen2`, and `qwen3`
- a Gemma2 parameter-key coverage check that validates quantized checkpoint keys against the native module tree after loader normalization

By default the ignored inference tests look under `$HOME/.models/mlx/` for the currently supported families, then under the local Hugging Face cache if the repo snapshot already exists. You can override paths with:

- `MESH_LLM_MLX_TEST_GEMMA2_MODEL`
- `MESH_LLM_MLX_TEST_GEMMA3_TEXT_MODEL`
- `MESH_LLM_MLX_TEST_LLAMA_MODEL`
- `MESH_LLM_MLX_TEST_MISTRAL_MODEL`
- `MESH_LLM_MLX_TEST_QWEN2_MODEL`
- `MESH_LLM_MLX_TEST_QWEN3_MODEL`

If those paths are missing, the ignored inference tests can also download public fixtures from Hugging Face. Override the repo ids with:

- `MESH_LLM_MLX_TEST_GEMMA2_REPO`
- `MESH_LLM_MLX_TEST_GEMMA3_TEXT_REPO`
- `MESH_LLM_MLX_TEST_LLAMA_REPO`
- `MESH_LLM_MLX_TEST_MISTRAL_REPO`
- `MESH_LLM_MLX_TEST_QWEN2_REPO`
- `MESH_LLM_MLX_TEST_QWEN3_REPO`

Current defaults:

- Gemma2: `mlx-community/gemma-2-2b-it-4bit`
- Gemma3 Text: `mlx-community/gemma-3-270m-it-4bit`
- Llama: `mlx-community/Llama-3.2-1B-Instruct-bf16`
- Mistral: `mlx-community/Mistral-7B-Instruct-v0.2-4bit`
- Qwen2: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
- Qwen3: `mlx-community/Qwen3-0.6B-bf16`

Run the worker directly:

```bash
./target/release/mesh-llm-mlx \
  --model ~/.models/mlx/Qwen3-0.6B-bf16 \
  serve
```

Run through `mesh-llm`:

```bash
./target/release/mesh-llm \
  --backend mlx \
  --model ~/.models/mlx/Qwen3-0.6B-bf16
```

If `mesh-llm` cannot find the worker automatically, pass:

```bash
./target/release/mesh-llm \
  --backend mlx \
  --mlx-native-bin ./target/release/mesh-llm-mlx \
  --model ~/.models/mlx/Qwen3-0.6B-bf16
```

Profile one MLX request from inside the worker:

```bash
MESH_LLM_MLX_PROFILE=1 ./target/release/mesh-llm-mlx \
  --model ~/.models/mlx/Llama-3.2-1B-Instruct-bf16 \
  --listen 127.0.0.1:9348 \
  serve
```

Then send a request and read the worker stderr. The worker prints one line per
request with:

- `encode_ms`
- `render_ms`
- `build_ms`
- `ttft_ms`
- `reused_prompt_tokens`
- `first_chunk_ms`
- `decode_ms`
- `total_ms`
- `prompt_tokens`
- `completion_tokens`
- `finish_reason`

## Strategic Goal

The real target is not “MLX sidecar support.” The target is:

- MLX as the future default backend
- exo-style tensor-splitting parity on Apple Silicon
- broader family support over time
- eventual retirement of the MLX bootstrap experiments

The concrete plan is in [MLX_BACKEND_PLAN.md](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/MLX_BACKEND_PLAN.md).
