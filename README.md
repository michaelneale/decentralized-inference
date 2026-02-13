# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple machines using llama.cpp's RPC backend, with patches that eliminate model weight transfer and reduce per-token RPC chatter. Includes a QUIC mesh for NAT traversal, peer discovery, and direct worker-to-worker tensor transfers.

## The Problem

Stock llama.cpp RPC transfers **all model weights** from the orchestrator to every worker over TCP at startup (e.g. 17GB for GLM-4.7-Flash). Over WiFi or WAN this takes 10-15+ minutes. Then during inference, hundreds of redundant RPC round-trips per token kill throughput.

## The Fix

Two things working together:

**llama.cpp patches** ([fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf)):
- `SET_TENSOR_GGUF` — workers load tensors from their own local GGUF copy. Zero bytes over the network.
- Skip probing for RPC backends — eliminates hundreds of alloc/free round-trips at startup.
- `get_alloc_size` cache — cuts per-token RPC calls from ~290 to 7.
- B2B direct transfers — workers push activation tensors directly to each other, bypassing the orchestrator.

**mesh-inference** (Rust sidecar):
- QUIC mesh via iroh — NAT traversal, peer discovery, gossip.
- TCP↔QUIC tunneling — llama.cpp just sees localhost TCP sockets.
- REGISTER_PEER rewriting — enables B2B direct transfers through the mesh.
- `--client` lite mode — join the mesh from any machine and use the API, no GPU needed.

## Quick Start

```bash
just build            # clones fork, builds llama.cpp + mesh-inference
just download-model   # downloads GLM-4.7-Flash Q4_K_M (~17GB)
```

### Same machine (test everything works)
```bash
just local            # starts worker + server on localhost
just test             # run a test inference
just stop             # kill everything
```

### Two machines over the mesh

Both machines need the same GGUF model file.

**Machine A** (worker — starts first, prints invite token):
```bash
just mesh-worker
```

**Machine B** (orchestrator — joins and starts llama-server):
```bash
just mesh-serve join=<token-from-A>
```

### Lite client (no GPU, no model needed)

Any machine can join the mesh as a lightweight API proxy:

```bash
mesh-inference --client --join <token> --port 8080
```

This exposes `http://localhost:8080` — an OpenAI-compatible API that tunnels through the QUIC mesh to the host's llama-server. Works with curl, the OpenAI Python library, or any HTTP client. SSE streaming works transparently.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

> **Firewall note**: The machine that can accept incoming UDP should start first and print the invite token. The firewalled machine should be the joiner (`--join`). On managed Macs with stealth mode, that machine must always join.

## Node Roles

| Role | What it does | Flags |
|------|-------------|-------|
| **Worker** | Runs rpc-server, provides GPU compute | `mesh-inference --model ...` |
| **Host** | Runs llama-server + rpc-server, orchestrates inference, serves HTTP API | `mesh-inference --serve PORT --model ...` |
| **Client** | No GPU, no model. Proxies API requests through the mesh to a host | `mesh-inference --client --join TOKEN` |

Roles are advertised via gossip — clients automatically discover which node is the host.

## Configurations & Benchmarks

All benchmarks: M4 Max (55GB VRAM) + Mac Mini M4 (12GB VRAM), WiFi, GLM-4.7-Flash-Q4_K_M (17GB).

### Model Loading
| | Stock RPC | Patched |
|---|---|---|
| Data over network | 16.88 GB | **0 bytes** |
| Time to load | 14+ minutes | **~9 seconds** |

### Inference
| Configuration | Generation |
|---|---|
| Mini orchestrator, 85% remote on M4 Max (`--tensor-split 0.85,0.15`) | **21 tok/s** |
| M4 Max orchestrator, 82% local + 18% remote | **16 tok/s** |
| 3-node: Mini orchestrator + 2 workers (40/40/20 split) | **12-13 tok/s** |
| Local only (M4 Max, no mesh) | 68 tok/s |

QUIC mesh adds zero measurable overhead vs raw TCP.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Host (orchestrator)                                 │
│  llama-server --rpc <tunnels> --model model.gguf     │
│  - Assigns layers to backends, dispatches compute    │
│  - Workers load weights from local GGUF (no transfer)│
│  - Only activations (~10KB/token) cross the network  │
└──────────────────────┬──────────────────────────────┘
                       │ QUIC tunnel
                       ▼
┌─────────────────────────────────────────────────────┐
│  Worker                                              │
│  rpc-server --gguf model.gguf -d MTL0               │
│  - Loads weights from local GGUF                     │
│  - Computes assigned layers on Metal GPU             │
│  - B2B: pushes activations directly to next worker   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Lite Client                                         │
│  mesh-inference --client --join <token>              │
│  - No GPU, no model                                  │
│  - localhost:8080 → QUIC → host's llama-server       │
│  - Full OpenAI-compatible API (including SSE)        │
└─────────────────────────────────────────────────────┘
```

## Files

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches (SET_TENSOR_GGUF, chatter reduction, B2B) |
| `mesh-inference/` | Rust sidecar: QUIC mesh, tunneling, B2B rewriting, lite client proxy |
| `Justfile` | Build and run tasks |
