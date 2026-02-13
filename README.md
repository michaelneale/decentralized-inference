# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple machines using llama.cpp's RPC backend, with patches that eliminate model weight transfer and reduce per-token RPC chatter. Includes a QUIC mesh for NAT traversal and peer discovery.

## The Problem

Stock llama.cpp RPC transfers **all model weights** from the orchestrator to every worker over TCP at startup (e.g. 17GB for GLM-4.7-Flash). Over WiFi or WAN this takes 10-15+ minutes. Then during inference, hundreds of redundant RPC round-trips per token kill throughput.

## The Fix: Local GGUF Loading + Chatter Reduction

Our patches to `ggml-rpc.cpp` ([fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf)):

1. **`SET_TENSOR_GGUF`** — worker loads tensors from its own local copy of the GGUF file. Zero bytes transferred over the network for model weights.
2. **Skip probing for RPC backends** — eliminates hundreds of alloc/free round-trips during startup.
3. **Cache `get_alloc_size`** — 558 → 8 round-trips per token.
4. **Skip GGUF lookup for non-weights** — don't look up compute graph intermediates in the GGUF index.

## Quick Start

### Prerequisites

Both machines need the same GGUF model file:
```bash
just build            # clones fork, builds with Metal + RPC
just download-model   # downloads GLM-4.7-Flash Q4_K_M (~17GB)
```

### Same machine (test everything works)
```bash
just local            # builds, downloads model, starts worker + server
just test             # run a test inference
just stop             # kill everything
```

### Two machines (raw TCP)

**Worker** (big GPU machine):
```bash
just worker --host 0.0.0.0
```

**Orchestrator** (any machine):
```bash
just serve rpc=<worker-ip>:50052
```

### Two machines (QUIC mesh with NAT traversal)

The `mesh-inference` binary handles peer discovery, QUIC tunneling, and process management. Same binary on every node.

**Machine A** — starts mesh, waits for peers:
```bash
just mesh-worker
# prints an invite token
```

**Machine B** — joins mesh and starts llama-server:
```bash
just mesh-serve join=<token-from-A>
```

Both machines load model weights from local disk. Only activations (~10KB/token) cross the network.

> **Firewall note**: The machine that can accept incoming UDP should start first (print the token). The firewalled machine should be the one that joins. On managed Macs with stealth mode, that machine must always be the joiner.

### Test
```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## Configurations

### A) Big machine as orchestrator (simplest, fastest)

The machine with the most VRAM runs `--serve`. It uses its local GPU directly for the bulk of compute, remote workers contribute extra VRAM via RPC.

```
┌─ M4 Max (55GB) ──────────────┐     QUIC tunnel     ┌─ Mac Mini (12GB) ────────┐
│ mesh-inference --serve 8090   │◄──────────────────►│ mesh-inference            │
│ llama-server (82% local Metal)│  ~10KB/token        │ rpc-server (18% Metal)   │
│ rpc-server (unused)           │                     │                          │
└───────────────────────────────┘                     └──────────────────────────┘
```

**16 tok/s** generation. Orchestrator handles most layers locally at full speed.

### B) Small machine as orchestrator (model doesn't fit on one machine)

When the model is too large for any single machine, the orchestrator can be the smaller machine. Use `--tensor-split` to control how much stays local and `--no-mmap` (automatic) prevents the full model from being memory-mapped.

```
┌─ Mac Mini (12GB) ─────────────┐     QUIC tunnel     ┌─ M4 Max (55GB) ─────────┐
│ mesh-inference --serve 8090   │◄──────────────────►│ mesh-inference            │
│ llama-server (15% local Metal)│  ~10KB/token        │ rpc-server (85% Metal)   │
│ rpc-server (unused)           │                     │                          │
│ --tensor-split 0.85,0.15      │                     │                          │
└───────────────────────────────┘                     └──────────────────────────┘
```

**21 tok/s** generation. The bulk compute (85%) runs on M4 Max with zero network cost. Mini handles 15% locally. `--no-mmap` ensures only 2.6GB is allocated on Mini's Metal (not the full 17GB model).

### C) All remote compute (orchestrator has no GPU)

With `GGML_METAL_DEVICES=0`, the orchestrator contributes zero GPU compute. Everything goes through RPC. Only useful when the orchestrator truly has no GPU.

**2.5 tok/s** — every operation serializes through the network. Avoid this if possible.

## Benchmarks

### Model Loading (17GB GLM-4.7-Flash, WiFi)
| | Stock RPC | Patched |
|---|---|---|
| Data over network | 16.88 GB | **0 bytes** |
| Time to load | 14+ minutes | **~9 seconds** |

### Inference (M4 Max + Mac Mini, WiFi, QUIC mesh)
| Configuration | Prompt eval | Generation |
|---|---|---|
| Config A: M4 Max orchestrator (82% local + 18% remote) | 25 tok/s | **16 tok/s** |
| Config B: Mini orchestrator (15% local + 85% remote) | 27 tok/s | **21 tok/s** |
| Config C: Mini orchestrator, no local GPU | 3 tok/s | 2.5 tok/s |
| Local only (M4 Max, no mesh) | 160 tok/s | 68 tok/s |

### Latency Simulation (single-machine, raw TCP)
| Nodes | Latency | tok/s |
|-------|---------|------:|
| 3     | 0ms     | 60.2  |
| 3     | 5ms     | 30.1  |
| 3     | 10ms    | 21.4  |
| 3     | 20ms    | 12.6  |

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Orchestrator                                        │
│  llama-server --rpc <tunnel>:port --model model.gguf │
│  - Reads GGUF metadata, assigns layers to backends   │
│  - Sends SET_TENSOR_GGUF (tensor name only, no data) │
│  - Workers load weights from their own local GGUF    │
│  - Dispatches GRAPH_COMPUTE per token                │
└──────────────────────┬──────────────────────────────┘
                       │ QUIC tunnel (only activations, ~10KB/token)
                       ▼
┌─────────────────────────────────────────────────────┐
│  Worker                                              │
│  rpc-server --gguf model.gguf -d MTL0               │
│  - Receives tensor name via SET_TENSOR_GGUF          │
│  - Looks up in local GGUF index → reads from NVMe   │
│  - Executes GRAPH_COMPUTE on Metal GPU               │
│  - Returns activations/logits                        │
└─────────────────────────────────────────────────────┘

mesh-inference handles:
  - Peer discovery via iroh QUIC + gossip
  - NAT traversal (direct UDP or relay fallback)
  - TCP↔QUIC tunneling for RPC connections
  - Process management (rpc-server, llama-server)
```

## Files

| File | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches |
| `mesh-inference/` | Rust binary: QUIC mesh, tunneling, process management |
| `Justfile` | Build and run tasks |
| `PLAN.md` | Design analysis and detailed benchmark data |
