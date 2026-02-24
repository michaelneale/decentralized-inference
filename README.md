# Mesh LLM

![Mesh LLM](mesh.png)

Pool spare GPU capacity to run LLMs at larger scale. Split inference across machines over QUIC — models can be larger than any single machine's VRAM. Each node loads only its assigned layers from a local GGUF copy (zero network transfer for weights).

## Quick start (macOS Apple Silicon)

```bash
curl -fsSL https://github.com/michaelneale/decentralized-inference/releases/latest/download/mesh-llm-aarch64-apple-darwin.tar.gz | tar xz && sudo mv mesh-bundle/* /usr/local/bin/
```

```bash
mesh-llm --model Qwen2.5-32B --console    # downloads model (~20GB), starts API + web console
mesh-llm --model Qwen2.5-3B --console     # or a small model first (~2GB)
```

Add another machine:
```bash
mesh-llm --join <token>                    # token printed by the first machine
```

Or discover and join public meshes:
```bash
mesh-llm --auto                            # find and join the best mesh via Nostr
mesh-llm --auto --client                   # join as API-only client (no GPU)
```

## How it works

Every node gets an OpenAI-compatible API at `http://localhost:9337/v1`.

**Solo mode** — if a model fits on one machine, it runs there. Full speed, no network overhead.

**Tensor split** — if a model doesn't fit, layers are distributed across nodes proportional to VRAM. llama-server runs on the highest-VRAM node and coordinates via RPC. Each rpc-server loads only its assigned layers from local disk. Latency-aware: peers are selected by lowest RTT first, with an 80ms hard cap — high-latency nodes stay in the mesh as API clients but don't participate in splits.

**Multi-model** — different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node via QUIC tunnel. `/v1/models` lists everything available.

**Demand-aware rebalancing** — request rates are tracked per model and shared via gossip. Standby nodes auto-promote to serve hot models (≥10 req/min, ≥3x imbalance). Nodes also auto-promote when a model loses its last server (within ~60s).

**Latency design** — the key insight is that HTTP streaming is latency-tolerant while RPC is latency-multiplied. llama-server always runs on the same box as the GPU. The mesh tunnels HTTP, so cross-network latency only affects time-to-first-token, not per-token throughput. RPC only crosses the network for tensor splits where the model physically doesn't fit on one machine.

### Network optimizations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111s → 5s.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8.
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client.
- **Speculative decoding** — draft model runs locally on the host, proposes tokens verified in one batched forward pass. +38% throughput on code (75% acceptance).

## Usage

### Solo
```bash
mesh-llm --model Qwen2.5-32B
# API ready at http://localhost:9337
```

### Distributed
```bash
# Machine A
mesh-llm --model Qwen2.5-32B
# Prints invite token

# Machine B — learns model from gossip, downloads if needed
mesh-llm --join <token>
```

### Multi-model
```bash
# Request two models — node picks one based on mesh needs
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash

# Route by model name
curl localhost:9337/v1/chat/completions -d '{"model":"GLM-4.7-Flash-Q4_K_M", ...}'
```

### Client (no GPU)
```bash
mesh-llm --client --join <token>
# Proxies to mesh via QUIC
```

### Share via Nostr
```bash
mesh-llm --model Qwen2.5-3B --publish --mesh-name "My Mesh" --region AU
mesh-llm discover                          # browse meshes
mesh-llm discover --model GLM --region AU  # filter
```

### Browse and join interactively
```bash
mesh-llm                                   # opens console on :3131 for discovery/joining
```

## Web console

```bash
mesh-llm --model Qwen2.5-32B --console    # opens dashboard on :3131
```

Live topology, VRAM bars per node, model picker, built-in chat. Everything comes from `/api/status` (JSON) and `/api/events` (SSE).

## Using with agents

mesh-llm prints launch commands when ready:
```
pi:    pi --provider mesh --model Qwen2.5-32B-Instruct-Q4_K_M
goose: GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen2.5-32B-Instruct-Q4_K_M goose session
```

## Benchmarks

GLM-4.7-Flash-Q4_K_M (17GB), M4 Max + Mac Mini M4, WiFi:

| Configuration | tok/s |
|---|---|
| Solo (no mesh) | 68 |
| 2-node split (85/15) | 21 |
| 3-node split (62/31/8) | 12-13 |

Cross-network (Sydney ↔ Queensland, ~20ms RTT): 10-25 tok/s. Overhead dominated by per-token RPC latency.

Stock llama.cpp RPC transfers 16.88GB on connect. This fork: **0 bytes, ~9 seconds**.

## Model catalog

```bash
mesh-llm download           # list models
mesh-llm download 32b       # Qwen2.5-32B (~20GB)
mesh-llm download 72b --draft  # Qwen2.5-72B + draft model
```

Draft pairings for speculative decoding:

| Model | Size | Draft | Draft size |
|-------|------|-------|------------|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |

## CLI Reference

```
mesh-llm [OPTIONS]
  --model NAME|PATH    Model to serve (can specify multiple)
  --client             API-only client (no GPU)
  --join TOKEN         Join mesh via invite token
  --auto               Discover and join via Nostr
  --publish            Publish mesh to Nostr relays
  --mesh-name NAME     Human-readable mesh name
  --region REGION      Geographic region (AU, US-West, EU-West, ...)
  --max-clients N      Delist from Nostr when N clients connected
  --port PORT          API port (default: 9337)
  --bind-port PORT     Pin QUIC to fixed UDP port (for NAT)
  --max-vram GB        Cap VRAM advertised to mesh
  --split              Force tensor split
  --console [PORT]     Web dashboard (default: 3131)
  --device DEV         GPU device (default: MTL0)
  --draft PATH         Draft model for speculative decoding
  --no-draft           Disable auto draft detection

mesh-llm download [NAME] [--draft]
mesh-llm discover [--model M] [--region R] [--auto]
mesh-llm drop <model>
mesh-llm rotate-key
```

## Deploying

```bash
just bundle                                    # creates /tmp/mesh-bundle.tar.gz
scp /tmp/mesh-bundle.tar.gz user@remote:
ssh user@remote 'tar xzf mesh-bundle.tar.gz && mesh-bundle/mesh-llm --model Qwen2.5-3B'
```

Same architecture required (arm64 macOS → arm64 macOS). Bundle includes mesh-llm + llama.cpp binaries. For WAN: forward `--bind-port` UDP on the router — only the originator needs it.

## Building

```bash
just build            # clones llama.cpp fork, builds everything
just bundle           # portable tarball
```

For `--client` mode only the `mesh-llm` binary is needed.

## Developer releases

Cross-platform `mesh-llm` release binaries are published by GitHub Actions from tags.

### Local checks (before tagging)

```bash
just build            # full local build (llama.cpp + mesh-llm)
just build-mesh aarch64-apple-darwin
```

Optional: follow the detailed packaging/smoke-test checklist in `RELEASE.md`.

### Create a release

```bash
git add -A
git commit -m "v0.X.0: <summary>"
git tag v0.X.0
git push origin main --tags
```

This triggers the GitHub Actions release workflow, which builds `mesh-llm` with `just` for:

- macOS (`aarch64-apple-darwin`)
- Linux (`x86_64-unknown-linux-gnu`)
- Windows (`x86_64-pc-windows-msvc`)

The workflow then creates/updates the GitHub release for that tag and uploads the binaries as release assets.

### CI workflows

- `.github/workflows/build-cross-platform.yml` builds PRs/pushes and uploads CI artifacts
- `.github/workflows/release-mesh-llm.yml` builds tag releases and publishes GitHub Release assets
## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with zero-transfer RPC patches |
| `mesh-llm/` | Rust QUIC mesh ([internals](mesh-llm/README.md)) |
