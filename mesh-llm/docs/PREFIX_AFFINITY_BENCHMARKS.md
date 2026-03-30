# Prefix Affinity Benchmarks

This benchmark compares two routing modes for agentic traffic:

- `sticky-only`: disable prefix affinity and route by the existing sticky key
- `prefix-only`: route by the learned scaffold prefix, with no sticky fallback

The benchmark uses a local 3-node topology so routing happens on a passive client:

- `node1`: serving host
- `node2`: serving host
- `client1`: passive entrypoint that chooses a host and tunnels requests

## Workload

Each request keeps the same shared scaffold and changes only the first user turn:

- a long agent system prompt
- 10 tool definitions
- a short task-specific first user message

That models the common agent case where many independent tasks share the same expensive prompt prefix.

## How To Run

Build first:

```bash
just build
```

Then run the benchmark:

```bash
MODEL_PATH=$HOME/.models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
REQUESTS=8 \
ENTRY_MODE=passive \
NO_DRAFT=1 \
NODE1_API_PORT=9437 \
NODE2_API_PORT=9438 \
CLIENT1_API_PORT=9439 \
NODE1_CONSOLE_PORT=3231 \
NODE2_CONSOLE_PORT=3232 \
CLIENT1_CONSOLE_PORT=3233 \
BIND_PORT=7942 \
KEEP_TMP=1 \
just bench-prefix-affinity
```

Shortcut:

```bash
just bench-prefix-affinity
```

Useful overrides:

- `MODEL_PATH=/path/to/model.gguf`
- `REQUESTS=12`
- `WARMUP_REQUESTS=1`
- `ENTRY_MODE=passive`
- `NO_DRAFT=1`
- `KEEP_TMP=1`

The script writes logs and JSON summaries to a temporary directory and prints the path at startup.

## Example Result

Run configuration:

- model: `Qwen2.5-3B-Instruct-Q4_K_M`
- entry mode: `passive`
- requests per phase: `8`
- warmup requests: `1`
- draft decoding: disabled

Results:

| Mode | Mean prompt ms | Mean elapsed ms | Prefix hits | Prefix misses | Learned |
|---|---:|---:|---:|---:|---:|
| `sticky-only` | 2118.1 | 3697.4 | 0 | 0 | 0 |
| `prefix-only` | 128.4 | 1260.1 | 7 | 1 | 8 |

Observed improvement:

- prompt time: `93.9%` lower
- end-to-end elapsed time: `65.9%` lower

## Interpreting Host Counts

The script also annotates per-host route counts from the host logs.

In the example passive run:

- `sticky-only`: `node_9437=8`, `node_9438=0`
- `prefix-only`: `node_9437=8`, `node_9438=0`

That means this particular run did **not** win by spreading traffic differently across hosts. The gain came from keeping repeated scaffold-prefill requests on the same already-warm host after the first prefix miss.

## Notes

- The passive benchmark forces both serving hosts up with `MESH_LLM_FORCE_DUPLICATE_HOSTS=1` so the test measures routing rather than demand-driven duplicate-host promotion latency.
- For passive multi-host runs, make sure the HTTP tunnel half-close and first-byte timeout fixes are present; otherwise the benchmark can disconnect before completion on larger prompts.
