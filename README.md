# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple machines using llama.cpp's RPC backend, with patches that eliminate model weight transfer and reduce per-token RPC chatter.

## The Problem

Stock llama.cpp RPC transfers **all model weights** from the orchestrator to every worker over TCP at startup (e.g. 17GB for GLM-4.7-Flash). Over WiFi or WAN this takes 10-15+ minutes. Then during inference, hundreds of redundant RPC round-trips per token kill throughput.

## The Fix: Local GGUF Loading + Chatter Reduction

Our patches to `ggml-rpc.cpp` ([fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf)):

1. **`SET_TENSOR_GGUF`** — worker loads tensors from its own local copy of the GGUF file. Zero bytes transferred over the network for model weights. Falls back to wire transfer if the worker doesn't have the file.

2. **Skip probing for RPC backends** — `weight_buft_supported()` was doing `ALLOC_BUFFER(0)` + `FREE_BUFFER` round-trips for every tensor. Since RPC's `supports_op()` always returns `true`, short-circuit it.

3. **Cache `get_alloc_size`** — the scheduler re-queried alloc sizes for every tensor on every token. Now cached by tensor config. 558 → 8 round-trips per token.

4. **Skip GGUF lookup for non-weights** — don't waste round-trips trying to find compute graph intermediates in the GGUF index.

## Results

### Model Loading (17GB GLM-4.7-Flash, WiFi)
| | Stock RPC | Patched |
|---|---|---|
| Data over network | 16.88 GB | **0 bytes** |
| Time to load | 14+ minutes | **~30 seconds** |

### Inference (M4 Max worker ↔ Mac Mini orchestrator, WiFi)
| | Stock RPC | Patched |
|---|---|---|
| Prompt eval | 1.4-2.4 tok/s | **36-59 tok/s** |
| Generation | 1.3-5.5 tok/s | **9-24 tok/s** |
| RPC round-trips/token | ~600 | **~7** |

## Quick Start

### Quick Start (same machine)
```bash
just local          # clones fork, builds, downloads model, starts worker + server
just test           # run a test inference
just stop           # kill everything
```

### Two machines
```bash
just build          # on both machines
just download-model # on both machines
```

**Worker** (big GPU machine):
```bash
just worker --host 0.0.0.0
```

**Orchestrator** (any machine):
```bash
just serve rpc=<worker-ip>:50052
```

### Test
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

### Manual build (no just)
```bash
git clone https://github.com/michaelneale/llama.cpp.git
cd llama.cpp && git checkout rpc-local-gguf
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Orchestrator (Mac Mini, 12GB RAM)                   │
│  llama-server --rpc <worker>:50052 --model model.gguf│
│  - Reads GGUF metadata                               │
│  - Assigns layers to RPC worker                      │
│  - Sends SET_TENSOR_GGUF (tensor name only)          │
│  - Worker loads from its own local GGUF              │
│  - Dispatches GRAPH_COMPUTE per token                │
└──────────────────────┬──────────────────────────────┘
                       │ TCP (tiny: names, graphs, activations)
                       ▼
┌─────────────────────────────────────────────────────┐
│  Worker (M4 Max, 55GB VRAM)                          │
│  rpc-server --gguf model.gguf -d MTL0               │
│  - Receives tensor name via SET_TENSOR_GGUF          │
│  - Looks up in local GGUF index → reads from NVMe   │
│  - Executes GRAPH_COMPUTE on Metal GPU               │
│  - Returns activations/logits                        │
└─────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches (branch `rpc-local-gguf`) |
| `PLAN.md` | Design analysis and benchmark results |
| `notes.md` | Detailed reference: build, architecture, gotchas |
| `demo.sh` | One-command local setup: build, download, run |
| `bench.sh` | Benchmark across node counts and latencies |
| `latency-proxy.py` | Protocol-aware TCP proxy for latency simulation |
| `mesh-inference/` | Earlier experiment: iroh QUIC mesh (archived) |

## Latency Simulation (single-machine benchmarks)

| Nodes | Latency | tok/s |
|-------|---------|------:|
| 3     | 0ms     | 60.2  |
| 3     | 5ms     | 30.1  |
| 3     | 10ms    | 21.4  |
| 3     | 20ms    | 12.6  |
| 4     | 0ms     | 57.1  |
| 5     | 0ms     | 52.8  |

```bash
LATENCY1=10 LATENCY2=20 ./demo.sh glm
```
