# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple machines using llama.cpp's RPC backend, with patches that eliminate model weight transfer and reduce per-token RPC chatter. Includes a QUIC mesh for NAT traversal, peer discovery, and direct worker-to-worker tensor transfers.

## The Problem

Stock llama.cpp RPC transfers **all model weights** from the orchestrator to every worker over TCP at startup (e.g. 17GB for GLM-4.7-Flash). Over WiFi or WAN this takes 10-15+ minutes. Then during inference, hundreds of redundant RPC round-trips per token kill throughput.

## The Fix

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

Any machine can join the mesh as a lightweight API proxy. Only the `mesh-inference` binary is needed — no llama.cpp binaries, no model file.

```bash
mesh-inference --client --join <token> --port 8080
```

This exposes `http://localhost:8080` — an OpenAI-compatible API tunneled through QUIC to the host's llama-server. SSE streaming works transparently.

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## Node Roles

| Role | What it does | Flags |
|------|-------------|-------|
| **Worker** | Runs rpc-server, provides GPU compute | `mesh-inference --model PATH` |
| **Host** | Runs llama-server + rpc-server, orchestrates inference, serves HTTP API | `mesh-inference --serve PORT --model PATH` |
| **Client** | No GPU, no model. Proxies API requests through the mesh to a host | `mesh-inference --client --join TOKEN` |

Roles are advertised via gossip — clients automatically discover which node is the host.

## All Options

```
--model PATH         GGUF model file (worker + host only)
--serve PORT         Run llama-server on this HTTP port (makes this node a host)
--client             Run as lite client (no GPU, no model, no rpc-server)
--join TOKEN         Join mesh via invite token (repeatable)
--port PORT          Local HTTP port for --client mode (default: 8080)
--min-peers N        Wait for N peers before starting llama-server (default: 1)
--tensor-split R,R   Layer distribution ratios (e.g. "0.85,0.15")
--bin-dir PATH       Directory with rpc-server + llama-server binaries
--device DEV         GPU device for rpc-server (default: MTL0 on macOS)
```

## Benchmarks

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

## Firewall Notes

- The machine that can accept incoming UDP should start first (print the invite token)
- The firewalled machine should be the joiner (`--join`)
- macOS stealth mode drops incoming UDP — that machine must always join, never listen
- iroh provides relay fallback if direct UDP fails, but relay is rate-limited

## Deploying to Another Machine

```bash
just bundle    # creates /tmp/mesh-bundle.tar.gz with all binaries + dylibs
scp /tmp/mesh-bundle.tar.gz user@remote:/tmp/
ssh user@remote "cd /tmp && tar xzf mesh-bundle.tar.gz"
# Then: /tmp/mesh-bundle/mesh-inference --model ... (or --client --join ...)
```

For `--client` mode, only the `mesh-inference` binary is needed.

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

## Justfile Targets

Requires [just](https://github.com/casey/just). All targets run from the project root.

| Target | Description |
|---|---|
| `just build` | Clone fork + build llama.cpp and mesh-inference |
| `just download-model` | Download GLM-4.7-Flash Q4_K_M (~17GB) |
| `just local` | Build + start worker + server on localhost for testing |
| `just test` | Quick inference test against a running server |
| `just stop` | Kill all running processes |
| `just mesh-worker` | Start a mesh worker (prints invite token) |
| `just mesh-serve join=TOKEN` | Start mesh host (orchestrator + llama-server) |
| `just mesh-client join=TOKEN` | Start lite client (local HTTP proxy, no GPU) |
| `just bundle` | Create portable tarball for deployment to another machine |
| `just worker` | Start raw TCP rpc-server (no mesh) |
| `just serve rpc=HOST:PORT` | Start raw TCP llama-server (no mesh) |
| `just diff` | Show fork patches vs upstream |

## Files

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches (SET_TENSOR_GGUF, chatter reduction, B2B) |
| `mesh-inference/` | Rust sidecar: QUIC mesh, tunneling, B2B rewriting, lite client proxy ([design](mesh-inference/DESIGN.md)) |
| `Justfile` | Build and run targets (see above) |
| `PLAN.md` | Historical design notes and benchmark data |
