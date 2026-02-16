# Distributed LLM Inference with llama.cpp RPC

Split LLM inference across multiple machines over QUIC. Workers load model weights from their own local GGUF — nothing transfers over the network. The mesh auto-elects a host, calculates tensor split from VRAM, and restarts when nodes join or leave.

## Quick Start

### Build

```bash
just build            # clones llama.cpp fork, builds rpc-server + llama-server + mesh-inference
just download-model   # downloads GLM-4.7-Flash Q4_K_M (~17GB) to ~/.models/
```

### Deploy to another machine

```bash
just bundle           # creates /tmp/mesh-bundle.tar.gz with all binaries + dylibs
scp /tmp/mesh-bundle.tar.gz user@remote:~/
ssh user@remote "tar xzf mesh-bundle.tar.gz && mv mesh-bundle ~/bin"
```

For `--client` mode only the `mesh-inference` binary is needed.

### Test locally

```bash
just local            # starts worker + server on localhost
just test             # run inference
just stop             # kill everything
```

## Usage

### Solo (single machine)

Run on one machine with a model — it elects itself as host and starts serving:

```bash
mesh-inference --model ~/.models/model.gguf
# API ready at http://localhost:9337

curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

### Two machines (distributed inference)

Both machines need the same GGUF model file. Each loads weights from its own local copy — nothing transfers over the network.

**Machine A** (starts first):
```bash
mesh-inference --model ~/.models/model.gguf
# Prints: Invite token: eyJ...
# API ready at http://localhost:9337
```

**Machine B** (joins):
```bash
mesh-inference --model ~/.models/model.gguf --join <token>
# API ready at http://localhost:9337
```

The mesh auto-elects a host (highest VRAM wins). The host runs llama-server with `--rpc` pointing at all nodes. Tensor split is calculated from VRAM automatically. Both machines can `curl localhost:9337` — the host serves directly, the worker proxies to the host.

When a node joins or leaves, llama-server is killed and restarted with the new configuration. No manual intervention needed.

Both machines can query `localhost:9337` simultaneously — llama-server handles concurrent requests with its built-in request queue. This lets you share one distributed inference cluster across multiple users or applications.

### Lite client (no GPU, no model)

Join the mesh from any machine as a lightweight API proxy:

```bash
mesh-inference --client --join <token> --port 9337
# API ready at http://localhost:9337

curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Only the `mesh-inference` binary is needed — no llama.cpp binaries, no model file.

### Web console

```bash
mesh-inference console    # opens http://localhost:3131
```

Browser UI for joining meshes, selecting models, downloading from HuggingFace, and chatting.

## Networking

Connections use [iroh](https://iroh.computer) QUIC. By default iroh handles NAT traversal via relay servers and STUN. Direct UDP is preferred (lowest latency), relays are fallback.

### WAN with port forwarding

For the best latency, forward a **UDP** port on the router to the machine that will accept incoming connections:

```bash
# Machine with port forwarding (e.g. UDP 7842 forwarded on router)
mesh-inference --model ~/.models/model.gguf --bind-port 7842
```

`--bind-port` does two things:
1. Pins the QUIC endpoint to a fixed UDP port (otherwise iroh picks a random port each run)
2. Triggers a STUN lookup to discover the public IP, so the invite token includes it

The joining machine doesn't need port forwarding — it connects outbound.

### Relay override

If iroh's default relays are unreachable (DNS sinkhole, corporate firewall):

```bash
mesh-inference --relay https://staging-use1-1.relay.iroh.network./ --model ...
```

### Connection order

Most of the time **order doesn't matter** — iroh handles NAT traversal via STUN and UDP hole punching, and two nodes behind normal NAT can connect in either direction.

Order matters when one side has a restrictive network:
- **Symmetric NAT** (mapped port changes per destination, so hole punching fails)
- **Firewall dropping unsolicited inbound UDP** (e.g. macOS stealth mode)
- **DNS sinkhole** blocking iroh's relay/STUN servers

In these cases, the **reachable** node should start first (print the token) and the restricted node should join (`--join`). Once the QUIC connection is established it's fully bidirectional — the host can open streams back through the existing connection regardless of NAT.

- **Both sides normal NAT** → either order works
- **One side restricted** → reachable side starts first, restricted side joins
- **Neither side reachable** → falls back to iroh relay (~200ms RTT vs ~20ms direct)

## How It Works

Every node with a model runs `rpc-server` (contributes GPU). The mesh gossips VRAM and roles. On every mesh change:

1. llama-server is killed (wherever it's running)
2. Election runs (deterministic — highest VRAM wins, every node computes the same answer)
3. Winner starts llama-server with `--rpc` pointing at all nodes (including itself)
4. mesh-inference owns `:9337` on every node — host proxies to local llama-server, workers proxy via QUIC to the host

```
┌────────────────────────────────────────────────────────┐
│  Elected Host (highest VRAM)                            │
│  rpc-server (GPU) + llama-server --rpc <all nodes>      │
│  :9337 → local llama-server (ephemeral port)            │
└──────────────────────┬─────────────────────────────────┘
                       │ QUIC mesh
                       ▼
┌────────────────────────────────────────────────────────┐
│  Worker (every other node with a model)                 │
│  rpc-server (GPU)                                       │
│  :9337 → QUIC tunnel → host's llama-server              │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Lite Client (--client, no GPU/model needed)            │
│  :9337 → QUIC tunnel → host's llama-server              │
└────────────────────────────────────────────────────────┘
```

### Election rules

1. Highest VRAM wins. Tie-break: highest node ID.
2. Deterministic — no consensus protocol, every node computes the same answer from gossip.
3. Re-election on every mesh change (join/leave).

### Tensor split

Calculated automatically from VRAM:
```
2 nodes: 103GB + 51GB = 154GB total
Split: 0.67, 0.33
```

### Running models larger than one machine

The mesh pools VRAM across nodes, so you can run models that wouldn't fit on any single machine. Before starting llama-server, the elected host checks that total mesh VRAM exceeds the model size. If not, it waits for more peers to join:

```
⏳ Waiting for more peers — need 55.0GB VRAM for model, have 51.5GB
# ... another machine joins ...
🗳 Elected as host (154.6GB VRAM available for 50.0GB model)
```

No single node needs to fit the entire model. Each node loads only its assigned layers from its local GGUF copy (`--no-mmap` prevents unified memory from trying to mmap the entire file).

## CLI Reference

```
mesh-inference [OPTIONS]

  --model PATH         GGUF model file. Starts rpc-server, enters auto-election.
  --client             Lite client — no GPU, no model, API proxy only.
  --join TOKEN         Join mesh via invite token (repeatable).
  --port PORT          Local API port (default: 9337).
  --bind-port PORT     Pin QUIC to a fixed UDP port (for router port forwarding).
  --relay URL          Override iroh relay URLs (repeatable).
  --tensor-split R,R   Manual layer split ratios (e.g. "0.85,0.15").
  --bin-dir PATH       Directory with rpc-server + llama-server binaries.
  --device DEV         GPU device for rpc-server (default: MTL0).

mesh-inference console [--port PORT]

  --port PORT          Web console port (default: 3131).
```

## Benchmarks

M4 Max + Mac Mini M4, WiFi, GLM-4.7-Flash-Q4_K_M (17GB).

| | Stock RPC | Patched |
|---|---|---|
| Weight transfer | 16.88 GB | **0 bytes** |
| Load time | 14+ min | **~9 seconds** |

| Configuration | tok/s |
|---|---|
| Mini orchestrator, 85% on M4 Max | **21** |
| M4 Max orchestrator, 82% local | **16** |
| 3-node (40/40/20) | **12-13** |
| Local only (no mesh) | 68 |

## Justfile

### Build & Deploy

| Target | Description |
|---|---|
| `just build` | Clone fork + build everything |
| `just download-model` | Download default model to `~/.models/` |
| `just bundle` | Portable tarball with all binaries for another machine |

### Run

| Target | Description |
|---|---|
| `just mesh-worker` | Start mesh node (auto-election, prints invite token) |
| `just mesh-join join=TOKEN` | Join mesh (auto-election) |
| `just mesh-client join=TOKEN` | Lite client (no GPU) |

### Dev

| Target | Description |
|---|---|
| `just local` | Worker + server on localhost |
| `just test` | Quick inference test |
| `just stop` | Kill all processes |
| `just diff` | Show llama.cpp fork patches |

## Using with pi

mesh-inference serves an OpenAI-compatible API, so any tool that speaks that protocol can use it. To use it as a model provider in [pi](https://github.com/mariozechner/pi-coding-agent), add a `mesh` provider to `~/.pi/agent/models.json`:

```json
"mesh": {
  "baseUrl": "http://localhost:9337/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "models": [
    {
      "id": "GLM-4.7-Flash-Q4_K_M.gguf",
      "name": "GLM 4.7 Flash (Mesh)",
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

Then start mesh-inference (solo or distributed) and select the model in pi's model picker:

```bash
# Solo
mesh-inference --model ~/.models/GLM-4.7-Flash-Q4_K_M.gguf

# Or distributed with a remote machine
mesh-inference --model ~/.models/GLM-4.7-Flash-Q4_K_M.gguf --join <token>

# Or lite client (no GPU, proxies to a remote mesh)
mesh-inference --client --join <token>
```

All three modes serve the same API on `localhost:9337`.

## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches |
| `mesh-inference/` | Rust QUIC mesh sidecar ([details](mesh-inference/README.md), [design](mesh-inference/DESIGN.md)) |
| `PLAN.md` | Design notes, benchmarks, WAN testing notes |
