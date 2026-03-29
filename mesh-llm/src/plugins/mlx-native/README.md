# MLX Worker

This crate is the current owned MLX worker path for `mesh-llm`.

This plugin runs a local OpenAI-compatible HTTP endpoint backed by native MLX.

Current scope:

- local-only HTTP serving
- plugin control-plane status via `mesh-llm` plugins
- model families implemented first: `gemma2`, `gemma3_text`, `llama`, `mistral`, `qwen2`, and `qwen3`
- host-node integration through `mesh-llm --backend mlx`
- remote consumers can reach the host through the existing mesh HTTP tunnel path
- streaming chat and completions

Build with the native backend enabled:

```bash
just build-mlx
```

Run the worker tests:

```bash
just test-mlx
just test-mlx-inference
```

Run standalone:

```bash
target/release/mesh-llm-mlx \
  --model ~/.models/mlx/Qwen3-0.6B-bf16 \
  serve
```

The server binds `127.0.0.1:0` by default and prints health via the plugin
surface. Pass `--listen 127.0.0.1:9338` to choose a fixed port.

Run through `mesh-llm`:

```bash
target/release/mesh-llm \
  --backend mlx \
  --model ~/.models/mlx/Qwen3-0.6B-bf16
```

## Third-Party Attribution

Some MLX generation and prompt-handling logic in this crate is adapted from
[exo](https://github.com/exo-explore/exo), which is licensed under
Apache-2.0. See `NOTICE-exo.md` and `APACHE-2.0.txt` in this directory.
