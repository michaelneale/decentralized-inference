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

## Mesh Usage

Both machines need the same GGUF model file. Each loads weights from its own local copy.

**Machine A** (starts first):
```bash
mesh-inference --model ~/.models/model.gguf
# Prints: Invite token: eyJ...
```

**Machine B** (joins):
```bash
mesh-inference --model ~/.models/model.gguf --join <token>
```

Highest VRAM becomes the host automatically. The other machine contributes GPU as a worker.

### Lite client (no GPU, no model)

```bash
mesh-inference --client --join <token> --port 8080
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

### Web console

```bash
mesh-inference console    # opens http://localhost:3131
```

Browser UI for joining meshes, selecting models, downloading from HuggingFace, and chatting.

## Networking

Connections use [iroh](https://iroh.computer) QUIC. By default iroh handles NAT traversal via relay servers and STUN. Direct UDP is preferred (lowest latency), relays are fallback.

### WAN with port forwarding

For the best latency, forward a **UDP** port on the router to the machine that starts first:

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

## CLI Reference

```
mesh-inference [OPTIONS]

  --model PATH         GGUF model file. Starts rpc-server, enters auto-election.
  --client             Lite client — no GPU, no model, API proxy only.
  --join TOKEN         Join mesh via invite token (repeatable).
  --port PORT          HTTP port for client proxy or auto-elected host (default: 8080/8090).
  --bind-port PORT     Pin QUIC to a fixed UDP port (for router port forwarding).
  --relay URL          Override iroh relay URLs (repeatable).
  --serve PORT         Force host on this port (skip auto-election).
  --tensor-split R,R   Manual layer split ratios (e.g. "0.85,0.15").
  --bin-dir PATH       Directory with rpc-server + llama-server binaries.
  --device DEV         GPU device for rpc-server (default: MTL0).
  --min-peers N        Wait for N peers before starting (--serve mode only).

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

See [PLAN.md](PLAN.md) for detailed notes.

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
| `just mesh-worker` | Start mesh node (auto-election) |
| `just mesh-serve join=TOKEN` | Manual host mode |
| `just mesh-client join=TOKEN` | Lite client |

### Dev

| Target | Description |
|---|---|
| `just local` | Worker + server on localhost |
| `just test` | Quick inference test |
| `just stop` | Kill all processes |
| `just diff` | Show llama.cpp fork patches |

## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC patches |
| `mesh-inference/` | Rust QUIC mesh sidecar ([details](mesh-inference/README.md), [design](mesh-inference/DESIGN.md)) |
| `PLAN.md` | Design notes, benchmarks, WAN testing notes |
