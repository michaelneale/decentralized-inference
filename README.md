# Distributed LLM Inference with llama.cpp RPC

Split LLM inference across multiple machines over QUIC. Each machine loads model weights from its own local GGUF — nothing transfers over the network. The mesh auto-elects a host, calculates tensor split from VRAM, and restarts when nodes join or leave.

## Usage

### Solo

```bash
mesh-inference --model ~/.models/model.gguf
# API ready at http://localhost:9337
```

### Distributed (two or more machines)

Both machines need the same GGUF file locally.

```bash
# Machine A
mesh-inference --model ~/.models/model.gguf
# Prints: Invite token: eyJ...

# Machine B
mesh-inference --model ~/.models/model.gguf --join <token>
```

Both get `localhost:9337`. The host (highest VRAM) runs llama-server with `--rpc` across all nodes. Tensor split is automatic. When nodes join or leave, the mesh re-elects and restarts.

### Lite client (no GPU needed)

```bash
mesh-inference --client --join <token>
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

## Using with pi

mesh-inference serves an OpenAI-compatible API. To use with [pi](https://github.com/mariozechner/pi-coding-agent), add to `~/.pi/agent/models.json`:

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

Start mesh-inference in any mode (solo, distributed, or lite client), then select the model in pi.

## Web console

```bash
mesh-inference console    # opens http://localhost:3131
```

Browser UI for joining meshes, selecting models, and chatting. Can search and download models from HuggingFace directly.

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

- **Election**: highest VRAM wins, deterministic, re-runs on every mesh change
- **Tensor split**: auto from VRAM (e.g. 103GB + 51GB → 0.67, 0.33)
- **Concurrent queries**: both ends can query simultaneously (llama-server request queue)
- **llama-server always uses --rpc**: even solo — same code path always

## Networking

Connections use [iroh](https://iroh.computer) QUIC with NAT traversal via STUN + relays.

For WAN with port forwarding (best latency):
```bash
mesh-inference --model model.gguf --bind-port 7842  # pins QUIC to fixed UDP port
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
mesh-inference [OPTIONS]
  --model PATH         GGUF model file
  --client             Lite client (no GPU/model)
  --join TOKEN         Join mesh via invite token
  --port PORT          API port (default: 9337)
  --bind-port PORT     Pin QUIC to fixed UDP port
  --relay URL          Override relay URLs
  --tensor-split R,R   Manual split ratios
  --bin-dir PATH       Directory with rpc-server + llama-server
  --device DEV         GPU device (default: MTL0)

mesh-inference console
  --port PORT          Console port (default: 3131)
```

## Building

```bash
just build            # clones llama.cpp fork, builds everything
just download-model   # downloads GLM-4.7-Flash Q4_K_M (~17GB)
just bundle           # portable tarball for another machine
```

For `--client` mode only the `mesh-inference` binary is needed.

## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with RPC local-GGUF patches |
| `mesh-inference/` | Rust QUIC mesh ([details](mesh-inference/README.md), [design](mesh-inference/DESIGN.md)) |
