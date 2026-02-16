# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple machines using llama.cpp's RPC backend, with patches that eliminate model weight transfer and reduce per-token RPC chatter. Includes a QUIC mesh for NAT traversal, peer discovery, auto host election, and direct worker-to-worker tensor transfers.

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
- Auto host election — highest-VRAM node becomes the host automatically.
- Auto tensor-split — VRAM-proportional layer distribution, no manual config.
- Dynamic membership — workers join/leave, llama-server auto-restarts.
- `--client` lite mode — join the mesh from any machine and use the API, no GPU needed.
- Web console — browser UI for managing mesh, no CLI knowledge needed.

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

### Two machines over the mesh (CLI)

Both machines need the same GGUF model file in `~/.models/`.

**Machine A** (starts first, prints invite token):
```bash
mesh-inference --model ~/.models/model.gguf
# Prints: Invite token: eyJ...
```

**Machine B** (joins Machine A):
```bash
mesh-inference --model ~/.models/model.gguf --join <token-from-A>
```

The mesh auto-elects a host (highest VRAM) and starts llama-server. The other machine becomes a worker. Tensor split is calculated automatically based on VRAM.

If a worker joins or leaves later, llama-server restarts automatically with the new configuration.

### Two machines over the mesh (Web Console)

The web console provides a browser UI — no CLI knowledge needed.

**Machine A:**
```bash
mesh-inference console
# Opens http://localhost:3131
```

**Machine B:**
```bash
mesh-inference console
# Open http://localhost:3131, paste Machine A's invite token, click Join
```

On both machines: click **Start**, select a model. The mesh auto-elects a host and starts inference. Models can be downloaded directly from the console UI.

### Lite client (no GPU, no model needed)

Any machine can join the mesh as a lightweight API proxy:

```bash
mesh-inference --client --join <token> --port 8080
```

This exposes `http://localhost:8080` — an OpenAI-compatible API tunneled through QUIC to the host's llama-server. SSE streaming works transparently.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## How It Works

Every node with a model runs `rpc-server` (contributes GPU to the mesh). The mesh gossips VRAM, models, and roles. The node with the most VRAM automatically becomes the host, runs `llama-server`, and uses all other nodes as workers.

```
┌─────────────────────────────────────────────────────┐
│  Elected Host                                        │
│  rpc-server + llama-server --rpc <tunnels>           │
│  - Highest VRAM → auto-elected                       │
│  - Uses all workers via QUIC tunnels                 │
│  - Auto tensor-split proportional to VRAM            │
│  - Workers load weights from local GGUF (0 transfer) │
└──────────────────────┬──────────────────────────────┘
                       │ QUIC mesh
                       ▼
┌─────────────────────────────────────────────────────┐
│  Worker (every other node with a model)              │
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

### Dynamic Mesh

- **Worker joins**: Host detects new peer, restarts llama-server with updated `--rpc` list and recalculated tensor-split.
- **Worker leaves/crashes**: Detected in ~40s via QUIC timeout. Host restarts with remaining workers (or local-only if none left).
- **Host leaves**: Next highest-VRAM node auto-promotes to host.
- **All automatic** — no manual intervention needed.

### Auto Host Election

Election is deterministic from gossip state — no consensus protocol:
1. If an existing host is in the mesh, everyone else stays as worker (stability).
2. If no host exists, highest VRAM wins. Tie-break: highest node ID.
3. Re-election only happens when the current host disappears.

### VRAM-Proportional Tensor Split

Each node reports its available VRAM via gossip. The host calculates `--tensor-split` automatically:

```
3 nodes: 64GB + 16GB + 8GB = 88GB total
Split: 0.73, 0.18, 0.09
```

No manual `--tensor-split` needed (but you can override it).

## Web Console

```bash
mesh-inference console [--port 3131]
```

The console serves a single-page dashboard at `http://localhost:3131`:
- **Join mesh** — paste a token from another node
- **Start** — select a model, start rpc-server, enter auto-election
- **Just Connect** — lite client mode (no GPU)
- **Live peer list** — shows each peer's role, VRAM, and models
- **Model catalog** — download models directly from Hugging Face
- **Built-in chat** — test the LLM right from the browser

Available models in the built-in catalog:
| Model | Size | Description |
|---|---|---|
| Qwen2.5-3B-Instruct | 2GB | Small & fast general chat |
| Qwen2.5-Coder-7B-Instruct | 4.7GB | Code generation & completion |
| GLM-4.7-Flash | 17GB | Large general chat with reasoning |

Or place any GGUF file in `~/.models/` and it appears in the model selector.

## CLI Reference

### mesh-inference (main)

```
mesh-inference [OPTIONS]

--model PATH         GGUF model file
--serve PORT         Run llama-server on this HTTP port (forces this node to be host)
--client             Run as lite client (no GPU, no model, no rpc-server)
--join TOKEN         Join mesh via invite token (repeatable)
--port PORT          Local HTTP port for --client mode (default: 8080)
--min-peers N        Wait for N peers before starting llama-server (default: 1)
--tensor-split R,R   Layer distribution ratios (e.g. "0.85,0.15")
--bin-dir PATH       Directory with rpc-server + llama-server binaries
--device DEV         GPU device for rpc-server (default: MTL0 on macOS)
```

### mesh-inference console

```
mesh-inference console [--port PORT]

--port PORT          HTTP port for the web console (default: 3131)
```

The console starts a mesh node and manages everything via the browser. No other flags needed.

Console API endpoints (for scripting):
```bash
# Status
curl http://localhost:3131/api/status

# Join a mesh
curl -X POST http://localhost:3131/api/join -d '{"token":"eyJ..."}'

# Start (rpc-server + election)
curl -X POST http://localhost:3131/api/start -d '{"model":"/path/to/model.gguf"}'

# Stop everything
curl -X POST http://localhost:3131/api/stop

# Download a model from the catalog
curl -X POST http://localhost:3131/api/download-model -d '{"model":"Qwen2.5-3B-Instruct-Q4_K_M"}'

# List available models in the catalog
curl http://localhost:3131/api/catalog

# SSE stream of status updates
curl http://localhost:3131/api/events
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
| Mini orchestrator, 85% remote on M4 Max | **21 tok/s** |
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
| `mesh-inference/` | Rust sidecar: QUIC mesh, tunneling, election, B2B rewriting, console ([details](mesh-inference/README.md)) |
| `Justfile` | Build and run targets (see above) |
| `PLAN.md` | Historical design notes and benchmark data |
