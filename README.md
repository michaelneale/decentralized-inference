# Mesh LLM - a new kind of LLM

Donate or pool your spare capacity so you can run LLMs at larger scale. 

Split LLM inference across multiple machines over QUIC. Models can be larger than any single machine's VRAM — each node only loads the layers assigned to it by the tensor split. Weights are read from each node's own local GGUF copy (zero network transfer for model loading). The mesh auto-elects a host, calculates tensor split from VRAM, and restarts when nodes join or leave.

## Quick start (macOS Apple Silicon)

```bash
curl -fsSL https://github.com/michaelneale/decentralized-inference/releases/latest/download/mesh-llm-aarch64-apple-darwin.tar.gz | tar xz && sudo mv mesh-bundle/* /usr/local/bin/
```

Then run:
```bash
mesh-llm --model Qwen2.5-32B --console    # downloads model on first run (~20GB), starts API + web console
mesh-llm --model Qwen2.5-3B --console     # or try a small model first (~2GB)
```

To add another machine to the mesh:
```bash
mesh-llm --model Qwen2.5-32B --join <token>    # token is printed by the first machine
```

Or join without a GPU:
```bash
mesh-llm --client --join <token>
```

## How it works
A common question is around latency and networks, there are a few ways that are addressed.
This uses mesh tech (quic) to distribute inference workload: 

* use UDP to establish a mesh
* Zero-transfer GGUF loading (SET_TENSOR_GGUF) — Instead of sending model weights over the network, the RPC server reads them
 directly from a local copy of the GGUF file on disk. Dropped model load time from 111s → 5s on localhost. Also skips unnecessary
 op-support probing for RPC backends.
* RPC round-trip reduction — Caches get_alloc_size responses (deterministic for a given tensor shape/op) and skips GGUF lookups
 for tiny intermediate compute tensors. Reduced per-token round-trips from 558 → 8, boosting generation from ~3 tok/s to ~15
 tok/s over WiFi.
* Direct server-to-server tensor transfers — When a model is split across multiple RPC servers, intermediate tensors are
 pushed directly between servers via TCP instead of relaying through the client. Adds REGISTER_PEER, PUSH_TENSOR_TO_PEER, and
 PEER_TENSOR_DATA protocol commands, with automatic fallback to the old client-relay path if the remote server doesn't support* 
* draft/predictive models are used from the host so that in parallel many completions are evaluated reducing potential round trips when it guesses correctly (a bit like branch prediction but for LLMs!) - for some repetitive tasks (ie code!) that hit rate is 85%!
* The mesh is automatically rebalanced, and a "host" is elected to run the head instance to distribute the work. 

Limitations: the more spread and the higher latency, the lower the enumber of tok/s. Nearby (city) and networks that are friendly work best, and minimal nodes. It will use just one node to serve if it can fit that model in comfortably.


## Usage

### Solo

```bash
mesh-llm --model ~/.models/model.gguf
# API ready at http://localhost:9337
```

### Distributed (two or more machines)

```bash
# Machine A — starts the mesh, sets the model
mesh-llm --model Qwen2.5-32B
# Prints: Invite token: eyJ...

# Machine B — joins and auto-discovers the model
mesh-llm --join <token>
```

The joining node learns the model from the mesh via gossip. If the model file isn't already in `~/.models/`, it downloads automatically from HuggingFace (for catalog models). You can also specify `--model` explicitly on the joiner if you prefer.

Both get `localhost:9337`. The host (highest VRAM) runs llama-server with `--rpc` across all nodes. Tensor split is automatic. When nodes join or leave, the mesh re-elects and restarts.

### Lite client (no GPU needed)

```bash
mesh-llm --client --join <token>
# API ready at http://localhost:9337
```

Just the binary — no model, no llama.cpp. Proxies to the mesh via QUIC.

### Models larger than one machine

If the model doesn't fit on the first machine, it waits for more peers:

```
⏳ Waiting for more peers — need 55.0GB VRAM for model, have 51.5GB
# ... another machine joins ...
🗳 Elected as host (154.6GB VRAM available for 50.0GB model)
```

No single node needs to fit the entire model. Each loads only its assigned layers (`--no-mmap`).

## Using with agents

mesh-llm prints launch commands when the LLM is ready (and shows them in `--console`):

```
pi:    pi --provider mesh --model Qwen2.5-32B-Instruct-Q4_K_M
goose: GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen2.5-32B-Instruct-Q4_K_M goose session
```

**pi** requires a `"mesh"` provider in `~/.pi/agent/models.json` (the console shows the snippet to copy). **goose** just needs env vars.

## Web console

Add `--console` to any run to open a browser dashboard on `:3131`:

```bash
mesh-llm --model ~/.models/model.gguf --console
mesh-llm --model ~/.models/model.gguf --join <token> --console
```

Shows the live state of the running process:

- **Cluster bar** — nodes sized by VRAM, split percentages
- **Model info** — model, draft, total VRAM, API port
- **Agent commands** — copy-paste commands for pi and goose
- **Chat** — test the API with streaming responses

## How It Works

```
┌──────────────────────────────────────────────────────┐
│  Host (highest VRAM)                                  │
│  rpc-server (GPU) + llama-server --rpc <all nodes>    │
│  :9337 → local llama-server                           │
└──────────────────────┬───────────────────────────────┘
                       │ QUIC
┌──────────────────────┴───────────────────────────────┐
│  Worker                          │  Lite Client       │
│  rpc-server (GPU)                │  (no GPU)          │
│  :9337 → QUIC → host             │  :9337 → QUIC → host│
└──────────────────────────────────┴───────────────────┘
```

- **Zero-transfer model loading**: each rpc-server loads its assigned layers from its own local GGUF file (`--gguf` flag on our llama.cpp fork). The host's llama-server sends a small `SET_TENSOR_GGUF` command (tensor name + offset, no weight data). Stock llama.cpp transfers the full model over RPC (~17GB for a 32B model); this fork transfers 0 bytes.
- **Models larger than one machine**: the tensor split assigns layers across nodes. Each rpc-server only loads its slice. A 72B model (47GB) can run across two 32GB machines — neither needs to fit it alone.
- **Election**: highest VRAM wins, deterministic, re-runs on every mesh change
- **Tensor split**: auto from VRAM (e.g. 103GB + 51GB → 0.67, 0.33)
- **RTT gating**: peers with >80ms round-trip are skipped for tensor split (stay in mesh as API clients). Measured during gossip exchange.
- **VRAM cap**: `--max-vram 10` advertises 10GB to the mesh regardless of actual VRAM — limits how much work gets split to you
- **Concurrent queries**: both ends can query simultaneously (llama-server request queue)

## Networking

Connections use [iroh](https://iroh.computer) QUIC with NAT traversal via STUN + relays.

For WAN with port forwarding (best latency):
```bash
mesh-llm --model model.gguf --bind-port 7842  # pins QUIC to fixed UDP port
```

The joining side doesn't need port forwarding. If relays are blocked: `--relay <url>`.

## Benchmarks

GLM-4.7-Flash-Q4_K_M (17GB), M4 Max + Mac Mini M4, WiFi.

| Configuration | tok/s |
|---|---|
| Local only (no mesh) | 68 |
| Mini orchestrator, 85% on M4 Max | **21** |
| M4 Max orchestrator, 82% local | **16** |
| 3-node (40/40/20) | **12-13** |

Stock llama.cpp RPC transfers 16.88GB of weights on connect (14+ min). This fork: **0 bytes, ~9 seconds**.

## CLI Reference

```
mesh-llm [OPTIONS]
  --model PATH         GGUF model file
  --client             Lite client (no GPU/model)
  --join TOKEN         Join mesh via invite token
  --port PORT          API port (default: 9337)
  --bind-port PORT     Pin QUIC to fixed UDP port
  --max-vram GB        Cap VRAM advertised to mesh (limits work split to you)
  --split              Force tensor split even if model fits on host
  --relay URL          Override relay URLs
  --tensor-split R,R   Manual split ratios
  --bin-dir PATH       Directory with rpc-server + llama-server
  --device DEV         GPU device (default: MTL0)
  --draft PATH         Draft model for speculative decoding (auto-detected from catalog)
  --draft-max N        Max draft tokens per speculation (default: 8)
  --no-draft           Disable auto draft detection
  --console [PORT]     Web dashboard (default: 3131)

mesh-llm download [NAME] [--draft]
```

## Speculative Decoding

A small "draft" model runs on the host GPU and proposes candidate tokens. The distributed
main model verifies them in one batched forward pass, accepting multiple tokens per
network round-trip. This helps most for code generation and when network latency is high.

Draft models are auto-detected from the catalog — if you download a model with `--draft`,
the draft model is found automatically on launch:

```bash
mesh-llm download 32b --draft    # downloads Qwen2.5-32B + 0.5B draft
mesh-llm --model ~/.models/Qwen2.5-32B-Instruct-Q4_K_M.gguf
# Auto-detected draft model: ~/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Or specify explicitly: `--draft <path>`, `--draft-max N`, `--no-draft`.

**Benchmarks** (Qwen2.5-32B, 2 nodes, 0.67/0.33 split, 20ms RTT):

| Task | Baseline | With draft | Improvement |
|------|----------|------------|-------------|
| Prose (200 tok) | 5.3 tok/s | 6.2 tok/s (56% accept) | +17% |
| Code (1000 tok) | 5.3 tok/s | 7.3 tok/s (75% accept) | +38% |

Code has higher acceptance because it's more predictable. The draft model (491MB) costs
almost nothing and is purely local — no extra network traffic.

### Model catalog

```bash
mesh-llm download           # list all models
mesh-llm download 72b       # download Qwen2.5-72B (47GB, needs 2+ machines)
mesh-llm download 72b --draft  # also download the paired draft model
```

Models with tested draft pairings:

| Model | Size | Draft | Draft size |
|-------|------|-------|------------|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen2.5-Coder-32B | 20GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |
| GLM-4.7-Flash (MoE) | 17GB | — | No compatible draft |

## Deploying to a remote node

Build locally and copy the bundle:

```bash
just bundle                          # creates /tmp/mesh-bundle.tar.gz
scp /tmp/mesh-bundle.tar.gz user@remote:
```

On the remote machine:

```bash
mkdir -p ~/bin && tar xzf mesh-bundle.tar.gz -C ~/bin --strip-components=1
# Installs: mesh-llm, rpc-server, llama-server, *.dylib into ~/bin/
```

Download a model and start:

```bash
~/bin/mesh-llm download 32b --draft   # downloads to ~/.models/
~/bin/mesh-llm --model Qwen2.5-32B --bind-port 7842
# Prints invite token — paste on the joining machine
```

**Requirements**: same architecture (arm64 macOS → arm64 macOS). The bundle includes all llama.cpp dylibs. Models go in `~/.models/` by convention. `--bin-dir` defaults to the directory containing the `mesh-llm` binary.

For WAN: forward the `--bind-port` UDP port on the router. Only one side needs port forwarding.

## Building

```bash
just build            # clones llama.cpp fork, builds everything
just download-model   # downloads GLM-4.7-Flash Q4_K_M (~17GB)
just bundle           # portable tarball for another machine
```

For `--client` mode only the `mesh-llm` binary is needed.

## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC local-GGUF patches |
| `mesh-llm/` | Rust QUIC mesh ([details](mesh-llm/README.md), [design](mesh-llm/DESIGN.md)) |

<details>
<summary>Multi-model serving</summary>

The mesh can serve multiple models simultaneously. Different nodes load different models, and a single API endpoint routes requests by model name.

```bash
# Node 1: seed mesh with two models, serves the first itself
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash

# Node 2: joins without --model, auto-assigned to GLM (needed by mesh, already on disk)
mesh-llm --join <token>

# Any node's API port routes to the right model
curl localhost:9337/v1/models                    # lists both models
curl localhost:9337/v1/chat/completions \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M", ...}'    # routed to node 2 via QUIC
```

**How models are balanced:** Each node serves exactly one model. When a node joins without `--model`, the mesh assigns it automatically — preferring models that nobody is serving yet and that the node already has on disk (no download wait), then falling back to the least-served model. Nodes scan `~/.models/` on startup and advertise what they have via gossip.

**No accidental tensor split:** If a model fits on one node, it runs solo — its own independent llama-server. Two nodes both serving Qwen2.5-3B = two independent servers, not a tensor split. Splitting only happens when a model genuinely doesn't fit on any single node, or when `--split` is explicitly passed.

**Big models still split across nodes:** For a model too large for one machine, nodes serving that model form a group. The highest-VRAM node becomes host and runs llama-server with `--rpc` pointing at the others. This is the same tensor-split behavior as before, just scoped to one model's group.

Other features:
- `/v1/models` returns all served models (standard OpenAI API)
- `mesh-llm drop <model>` to stop serving a model
- Console: model picker to chat with any model, nodes highlight when selected
- `--client` works alongside GPU nodes on the same machine
- Dead peers detected and cleaned up in ~15s

</details>

<details>
<summary>Future ideas</summary>

### Load balancing across hosts
When multiple nodes independently serve the same model (solo mode), the proxy could round-robin requests across them. Currently it picks one deterministically.

### P2P model transfer
Nodes that already have a model could serve GGUF chunks over QUIC to new joiners. Faster than HuggingFace on LAN, doesn't depend on the catalog, works for any GGUF.

### Usage-aware rebalancing
Track per-model request rates via gossip. Automatically unload idle models and reassign nodes to busy ones. Currently assignments are static after join.

### 405B-class models
Tested Hermes-3-Llama-3.1-405B IQ2_M (137GB, 4 split files) across 2× M4 Max nodes. The mesh handled it correctly — auto-split, VRAM gating, split GGUF loading, no mmap. But generation was 0.04 tok/s (27s per token). The bottleneck is raw compute, not network — 405B parameters through 126 transformer layers is too much for 2 Apple Silicon chips. Would need 4+ high-end nodes or wait for faster hardware.

</details>
