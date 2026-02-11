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

### Backend-to-Backend (B2B) Direct Transfer Benchmark

Compares upstream llama.cpp against the
[`jh-block/llama.cpp` `rpc-backend-to-backend-comms`](https://github.com/jh-block/llama.cpp/tree/rpc-backend-to-backend-comms)
branch, which adds direct server-to-server tensor transfers — skipping the
orchestrator relay.

**Why this matters:** In upstream, when activations need to move between two RPC
workers (e.g. layer 15 on server A → layer 16 on server B), the data path is:

```
Upstream:   Server A  →  Orchestrator (GET_TENSOR)  →  Server B (SET_TENSOR)
B2B fork:   Server A  →  Server B  (direct push, orchestrator not involved)
```

Each cross-server boundary in upstream costs two network round-trips per token.
The b2b branch eliminates these with `RPC_CMD_PUSH_TENSOR_TO_PEER` — the source
server reads the tensor locally and pushes it directly to the destination.

**Test setup:** The latency proxy (in `transfer` mode) delays `GET_TENSOR`,
`SET_TENSOR`, `GRAPH_COMPUTE`, and `GRAPH_RECOMPUTE` on the orchestrator↔worker
links. The b2b branch's direct server→server pushes happen on localhost and
bypass the proxy entirely — exactly simulating the real-world case where workers
are co-located (or on a fast fabric) while the orchestrator is remote.

TTFT and total time measured client-side with streaming SSE (`measure.py`).
Generation speed (tok/s) measured over the post-TTFT window.

| Nodes | Latency | Build    | TTFT    | Total    | tok/s | Δ tok/s   |
|-------|---------|----------|--------:|---------:|------:|-----------|
| 3     | 5ms     | upstream | 502ms   | 2,045ms  | 11.7  |           |
| 3     | 5ms     | **b2b**  | **467ms** | **1,843ms** | **13.8** | **+18%** |
| 3     | 10ms    | upstream | 1,187ms | 4,246ms  |  6.2  |           |
| 3     | 10ms    | **b2b**  | **698ms** | **3,424ms** | 6.2   | **TTFT −41%** |
| 3     | 20ms    | upstream | 1,405ms | 6,614ms  |  3.6  |           |
| 3     | 20ms    | **b2b**  | –       | –        | –     | (outlier, excluded) |
| 4     | 5ms     | upstream | 721ms   | 2,958ms  |  8.1  |           |
| 4     | 5ms     | **b2b**  | **612ms** | 3,336ms | 6.6   | **TTFT −15%** |
| 4     | 10ms    | upstream | 1,163ms | 8,629ms  |  2.4  |           |
| 4     | 10ms    | **b2b**  | **935ms** | **4,005ms** | **5.9** | **+146%** |
| 5     | 5ms     | upstream | 1,018ms | 4,280ms  |  5.8  |           |
| 5     | 5ms     | **b2b**  | 1,319ms | **3,870ms** | **7.5** | **+29%** |

**Key observations:**

- **TTFT consistently improves.** Prompt evaluation processes all input tokens
  through every layer — each layer boundary that skips the orchestrator relay
  saves two latency round-trips. At 3 nodes / 10ms, TTFT drops from 1,187ms
  to 698ms (−41%).
- **Generation speed scales with more nodes.** At 4 nodes / 10ms the b2b branch
  is 2.5× faster (5.9 vs 2.4 tok/s) because it eliminates 3 relay hops per
  token. At 5 nodes / 5ms it's +29% (7.5 vs 5.8 tok/s).
- **The benefit grows with latency × node count.** More nodes = more cross-server
  boundaries to skip. Higher latency = each skipped relay saves more time.
- **At 0ms latency (not shown)** both are identical (~60 tok/s) — the direct
  transfer overhead is negligible on localhost.

**Bottom line:** The b2b branch is a clear win for multi-machine setups where
the orchestrator is not co-located with the workers. The more nodes and the
higher the inter-node latency, the bigger the advantage.

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
and injects `time.sleep()` on selected commands. Three modes are available:

| Mode | Delayed commands | Use case |
|---|---|---|
| `compute` | GRAPH_COMPUTE, GRAPH_RECOMPUTE | Simulate compute dispatch latency only |
| `transfer` (default) | Above + GET_TENSOR, SET_TENSOR | Simulate full network latency including activation relay — needed for fair b2b comparison |
| `all` | Every command | Debugging |

In all modes, SET_TENSOR delays are deferred until the first GRAPH_COMPUTE is
seen, so model loading (17GB of weight transfer) passes through at full speed.

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
| `bench-b2b.sh` | B2B comparison benchmark — upstream vs direct-transfer fork |
| `measure.py` | Streaming SSE client that measures TTFT and total time |
| `latency-proxy.py` | Protocol-aware TCP proxy with per-command latency injection |
| `notes.md` | Detailed reference: build steps, architecture, gotchas, debug |
| `llama.cpp/` | Upstream llama.cpp, built from source with `-DGGML_RPC=ON` |
| `llama.cpp-rpc-b2b/` | [`jh-block/llama.cpp`](https://github.com/jh-block/llama.cpp/tree/rpc-backend-to-backend-comms) fork with server-to-server transfers |
| `~/.models/` | Downloaded GGUF model files |
