# Distributed LLM Inference with llama.cpp RPC

Run large language models split across multiple devices using llama.cpp's RPC backend.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Client (pi, curl, any OpenAI-compatible client)                │
│  POST http://localhost:8080/v1/chat/completions                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  llama-server  (orchestrator)                                   │
│  - Loads the GGUF model file                                    │
│  - Splits layers across devices by free memory                  │
│  - Sends weights to workers over TCP at startup                 │
│  - Dispatches compute subgraphs each forward pass               │
│  - Exposes OpenAI-compatible API on :8080                       │
└───────┬──────────────────┬──────────────────┬───────────────────┘
        │ TCP              │ TCP              │ local
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  rpc-server   │  │  rpc-server   │  │  Metal GPU    │
│  :50052       │  │  :50053       │  │  (built-in)   │
│  Metal GPU    │  │  CPU          │  │               │
│  layers 0-14  │  │  layers 15-30 │  │  layers 31-47 │
└───────────────┘  └───────────────┘  └───────────────┘
```

Workers are **stateless** — they don't need the model file. The orchestrator sends
everything over TCP: weights at startup, compute graphs per request, activations
between split boundaries.

### Local vs Remote

This demo runs everything on **one machine** to prove it works. For real distributed
inference across multiple machines, the only difference is:

- Run `rpc-server -H 0.0.0.0 -p 50052` on each remote host
- Point `llama-server --rpc 192.168.x.x:50052,192.168.y.y:50052` at their IPs

The protocol is the same — TCP sockets carrying tensor data and compute graphs.
This is how you'd run a model too large for a single machine's memory.

## Quick Start

### 1. Start the server

```bash
./demo.sh glm      # GLM-4.7-Flash 17GB — downloads model if needed, builds llama.cpp if needed
./demo.sh qwen3    # Qwen3-Coder-30B 18GB
./demo.sh stop     # shut everything down
```

This builds llama.cpp with RPC support, downloads the model to `~/.models/`,
starts two RPC workers and llama-server, and waits until it's ready.

### 2. Use it

**With pi** (add `llama-rpc` provider to `~/.pi/agent/models.json` — see below):
```bash
pi --provider llama-rpc --model "GLM-4.7-Flash-Q4_K_M.gguf"

# or one-shot
pi -p "explain quicksort" --provider llama-rpc --model "GLM-4.7-Flash-Q4_K_M.gguf"
```

**With curl:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

**With any OpenAI-compatible client** — point it at `http://localhost:8080/v1`.

### 3. Pi provider config

Add this to `~/.pi/agent/models.json` under `"providers"`:

```json
"llama-rpc": {
  "baseUrl": "http://localhost:8080/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "models": [
    {
      "id": "GLM-4.7-Flash-Q4_K_M.gguf",
      "name": "GLM 4.7 Flash (RPC Split)",
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

## Performance (Apple M4 Max, 64GB)

| Model | Size | Generation Speed |
|---|---|---|
| GLM-4.7-Flash Q4_K_M | 17GB | ~61 tok/s |
| Qwen3-Coder-30B-A3B Q4_K_M | 18GB | ~44 tok/s |

## Files

| File | Purpose |
|---|---|
| `README.md` | This file — overview and quick start |
| `demo.sh` | One-command setup: build, download, run |
| `notes.md` | Detailed reference: build steps, architecture, gotchas, debug |
| `llama.cpp/` | Built from source with `-DGGML_RPC=ON` |
| `~/.models/` | Downloaded GGUF model files |
