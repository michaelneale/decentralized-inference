# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple devices using llama.cpp's RPC backend.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Client (pi, curl, any OpenAI-compatible client)                │
│  POST http://localhost:8080/v1/chat/completions                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  llama-server  (orchestrator)                                   │
│  - Loads the GGUF model file                                    │
│  - Splits layers across devices by free memory                  │
│  - Sends weights to workers over TCP at startup                 │
│  - Dispatches compute subgraphs each forward pass               │
│  - Exposes OpenAI-compatible API on :8080                       │
└───────┬──────────────────┬──────────────────┬───────────────────┘
        │ TCP              │ TCP              │ local
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  rpc-server   │  │  rpc-server   │  │  Metal GPU    │
│  :50052       │  │  :50053       │  │  (built-in)   │
│  Metal GPU    │  │  CPU          │  │               │
│  layers 0-15  │  │  layers 16-31 │  │  layers 32-47 │
└───────────────┘  └───────────────┘  └───────────────┘
```

Workers are **stateless** — they don't need the model file. The orchestrator sends
everything over TCP: weights at startup, compute graphs per request, activations
between split boundaries.

### Local vs Remote

This demo runs everything on **one machine** to prove it works. For real distributed
inference across multiple machines, the only difference is:

- Run `rpc-server -H 0.0.0.0 -p 50052` on each remote host
- Point `llama-server --rpc 192.168.x.x:50052,192.168.y.y:50052` at their IPs

The protocol is the same — TCP sockets carrying tensor data and compute graphs.
This is how you'd run a model too large for a single machine's memory.

## Quick Start

### 1. Start the server

```bash
./demo.sh glm      # GLM-4.7-Flash 17GB — downloads model if needed, builds llama.cpp if needed
./demo.sh qwen3    # Qwen3-Coder-30B 18GB
./demo.sh stop     # shut everything down
```

This builds llama.cpp with RPC support, downloads the model to `~/.models/`,
starts two RPC workers and llama-server, and waits until it's ready.

### 2. Use it

**With pi** (add `llama-rpc` provider to `~/.pi/agent/models.json` — see below):
```bash
pi --provider llama-rpc --model "GLM-4.7-Flash-Q4_K_M.gguf"

# or one-shot
pi -p "explain quicksort" --provider llama-rpc --model "GLM-4.7-Flash-Q4_K_M.gguf"
```

**With curl:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

**With any OpenAI-compatible client** — point it at `http://localhost:8080/v1`.

### 3. Pi provider config

Add this to `~/.pi/agent/models.json` under `"providers"`:

```json
"llama-rpc": {
  "baseUrl": "http://localhost:8080/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "models": [
    {
      "id": "GLM-4.7-Flash-Q4_K_M.gguf",
      "name": "GLM 4.7 Flash (RPC Split)",
      "reasoning": false,
      "input": ["text"],
      "contextWindow": 202752,
      "maxTokens": 32768,
      "compat": {
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens",
        "supportsDeveloperRole": false
      }
    }
  ]
}
```

## Performance (Apple M4 Max, 64GB)

### Model Speeds (3 nodes, no latency)

| Model | Size | Generation Speed |
|---|---|---|
| GLM-4.7-Flash Q4_K_M | 17GB | ~60 tok/s |
| Qwen3-Coder-30B-A3B Q4_K_M | 18GB | ~44 tok/s |

### Scaling & Latency Benchmark

How throughput changes as you add more nodes and inject network latency.
Tested with GLM-4.7-Flash Q4_K_M using even tensor splits and a Python
TCP proxy (`latency-proxy.py`) that adds delay only on compute operations.

| Nodes | Latency | tok/s | vs baseline |
|-------|---------|------:|-------------|
| 3     | 0ms     | 60.2  | 1.00×       |
| 3     | 5ms     | 30.1  | 0.50×       |
| 3     | 10ms    | 21.4  | 0.36×       |
| 3     | 20ms    | 12.6  | 0.21×       |
| 3     | 30ms    |  9.4  | 0.16×       |
| 4     | 0ms     | 57.1  | 0.95×       |
| 4     | 5ms     | 21.8  | 0.36×       |
| 4     | 10ms    | 15.7  | 0.26×       |
| 4     | 20ms    |  8.0  | 0.13×       |
| 5     | 0ms     | 52.8  | 0.88×       |
| 5     | 5ms     | 18.4  | 0.31×       |

**Key observations:**

- **At 0ms (same machine):** adding nodes barely hurts — 3→5 nodes only drops
  60→53 tok/s. The overhead is just pipeline scheduling across more splits.
- **Latency × nodes multiplies:** each token requires a serial forward pass
  through every RPC node, so the per-token cost is roughly
  `compute_time + (N-1) × round_trip_latency`.
- **5ms is realistic** for same-datacenter machines and already halves throughput
  at 3 nodes. Cross-region latency (20-40ms) makes this impractical unless the
  model simply doesn't fit on fewer machines.
- **The sweet spot** for multi-machine inference is low-latency links (< 5ms)
  with as few nodes as possible — use it when the model doesn't fit in one
  machine's memory, not as a performance optimization.

## Latency Simulation

Inject artificial per-node latency to simulate real network conditions:

```bash
# Per-node latency (ms) — applied to GRAPH_COMPUTE ops only, not model loading
LATENCY1=5 LATENCY2=10 ./demo.sh glm

# Force specific layer split ratios
TENSOR_SPLIT="0.33,0.33,0.34" ./demo.sh glm

# Verbose logging — shows per-layer device assignment
VERBOSE=1 ./demo.sh glm

# Combine all
LATENCY1=10 LATENCY2=20 TENSOR_SPLIT="0.5,0.25,0.25" VERBOSE=1 ./demo.sh glm
```

The latency proxy (`latency-proxy.py`) parses the llama.cpp RPC binary protocol
and injects `time.sleep()` only on `GRAPH_COMPUTE` and `GRAPH_RECOMPUTE` commands.
Model loading (SET_TENSOR, ALLOC_BUFFER, etc.) passes through at full speed.

### Run the full benchmark

```bash
./bench.sh               # full matrix: 3/4/5 nodes × multiple latencies
NODES=4 LATENCY=10 ./bench.sh   # single data point
```

## Files

| File | Purpose |
|---|---|
| `README.md` | This file — overview and quick start |
| `demo.sh` | One-command setup: build, download, run (supports latency + split) |
| `bench.sh` | Automated benchmark across node counts and latencies |
| `latency-proxy.py` | TCP proxy that injects delay on RPC compute commands |
| `notes.md` | Detailed reference: build steps, architecture, gotchas, debug |
| `llama.cpp/` | Built from source with `-DGGML_RPC=ON` |
| `~/.models/` | Downloaded GGUF model files |
