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

Browser dashboard for managing everything without the CLI:

- **Download models** from the catalog (with optional draft model)
- **Join a mesh** by pasting an invite token
- **Start as a GPU worker** — pick a model, auto-elect host, serve the LLM
- **Connect as a lite client** — no GPU needed, proxy to a mesh host
- **Chat** to test the API with streaming responses
- **Live status** — see peers, roles, VRAM, election state via SSE

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
  --draft PATH         Draft model for speculative decoding (auto-detected from catalog)
  --draft-max N        Max draft tokens per speculation (default: 8)
  --no-draft           Disable auto draft detection

mesh-inference download [NAME] [--draft]
mesh-inference console [--port PORT]
```

## Speculative Decoding

A small "draft" model runs on the host GPU and proposes candidate tokens. The distributed
main model verifies them in one batched forward pass, accepting multiple tokens per
network round-trip. This helps most for code generation and when network latency is high.

Draft models are auto-detected from the catalog — if you download a model with `--draft`,
the draft model is found automatically on launch:

```bash
mesh-inference download 32b --draft    # downloads Qwen2.5-32B + 0.5B draft
mesh-inference --model ~/.models/Qwen2.5-32B-Instruct-Q4_K_M.gguf
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
mesh-inference download           # list all models
mesh-inference download 72b       # download Qwen2.5-72B (47GB, needs 2+ machines)
mesh-inference download 72b --draft  # also download the paired draft model
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
