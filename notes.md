# llama.cpp RPC Split Inference

Split a model across multiple devices using RPC workers over TCP.
Workers are stateless — they don't need the model file. `llama-server` sends weights and compute graphs over the wire.

- Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- Enabled by: https://github.com/ggml-org/llama.cpp/pull/6829

## Step 1: Clone and Build

Homebrew's `llama.cpp` is built **without** RPC support. Must build from source.

### Prerequisites

```bash
brew install cmake
```

### Clone

```bash
cd /Users/micn/Documents/code/deez
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Tested at commit `612db6188` (tag ~b7991). To pin to that:
```bash
git checkout 612db6188
```

### Build

```bash
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

`-DGGML_METAL=ON` enables Apple GPU. `-DGGML_RPC=ON` enables the RPC backend.

This produces two key binaries:
```
llama.cpp/build/bin/rpc-server      # RPC worker (exposes a device over TCP)
llama.cpp/build/bin/llama-server    # OpenAI-compatible server (orchestrates inference)
```

## Step 2: Get a Model

Models must be GGUF format with an architecture llama.cpp supports.

### Option A: Download from HuggingFace

```bash
mkdir -p ~/.models
curl -L -o ~/.models/GLM-4.7-Flash-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
```

This is GLM-4.7-Flash (17GB, Q4_K_M quant, `deepseek2` arch, supports thinking mode).

Good GGUF sources on HuggingFace: **unsloth**, **bartowski**, **lmstudio-community**, **mradermacher**.

### Option B: Reuse an Ollama model

Ollama stores GGUFs as blobs. Find the path:
```bash
cat ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/latest | python3 -m json.tool
# Look for "application/vnd.ollama.image.model" — the digest is the blob filename:
# ~/.ollama/models/blobs/<digest>
```

Example — Qwen3-Coder-30B works directly:
```
~/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a
```

**Caveat:** Some ollama models use ollama-specific arch strings that upstream doesn't recognise (e.g. GLM-4.7-Flash uses `glm4moelite` in ollama but needs `deepseek2` in upstream). Use the HuggingFace GGUF instead.

## Step 3: Start RPC Servers

Each `rpc-server` exposes one device as a TCP endpoint. Run two on different ports with different devices:

```bash
# Terminal 1 — Metal GPU worker
cd /Users/micn/Documents/code/deez/llama.cpp/build
./bin/rpc-server -d MTL0 -p 50052

# Terminal 2 — CPU worker
cd /Users/micn/Documents/code/deez/llama.cpp/build
./bin/rpc-server -d CPU -p 50053
```

Or backgrounded:
```bash
cd /Users/micn/Documents/code/deez/llama.cpp/build
nohup ./bin/rpc-server -d MTL0 -p 50052 > /tmp/rpc-50052.log 2>&1 &
nohup ./bin/rpc-server -d CPU -p 50053 > /tmp/rpc-50053.log 2>&1 &
```

**Important:** On the same machine, you must assign different devices via `-d`. If both default to all devices, the BLAS backend crashes on unsupported ops (e.g. `RMS_NORM`).

Verify they're listening:
```bash
lsof -i :50052 -i :50053 | grep LISTEN
```

## Step 4: Start llama-server

Point `llama-server` at the model and both RPC endpoints:

```bash
cd /Users/micn/Documents/code/deez/llama.cpp/build
./bin/llama-server \
  -m ~/.models/GLM-4.7-Flash-Q4_K_M.gguf \
  --rpc 127.0.0.1:50052,127.0.0.1:50053 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

Key flags:
- `-m` — path to GGUF model
- `--rpc` — comma-separated list of `host:port` RPC endpoints
- `-ngl 99` — offload all layers to GPU/RPC (use a lower number for partial offload)
- `--host 0.0.0.0` — listen on all interfaces
- `--port 8080` — HTTP port

It takes 30-60s to load (sends ~17GB of weights to RPC workers over TCP). You'll see `server is listening on http://0.0.0.0:8080` when ready.

## Step 5: Test

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

It exposes a standard OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, etc).

## Helper Script

`demo.sh` automates all of the above (build, download, start servers):

```bash
./demo.sh glm           # GLM-4.7-Flash (downloads to ~/.models/ if needed)
./demo.sh qwen3         # Qwen3-Coder-30B-A3B (downloads to ~/.models/ if needed)
./demo.sh /path/to.gguf # any GGUF
./demo.sh stop          # kill everything
```

## How the Split Works

1. `llama-server` loads the GGUF and discovers all backends: local Metal, local CPU, + each RPC endpoint
2. Layers are assigned to devices proportional to free memory
3. Weights are sent to each worker over TCP (`SET_TENSOR`) at startup
4. Each forward pass: scheduler runs subgraphs on each device (`GRAPH_COMPUTE`) and shuffles activations at split boundaries
5. Workers just execute ggml ops — zero knowledge of the model

### Example split (GLM-4.7-Flash, 47 layers, M4 Max 64GB):

```
CPU (embed) → RPC0:50052 (layers 0-14) → RPC1:50053 (layers 15-30) → MTL0 (layers 31-47 + output)
```

Note: `llama-server` also registers its local Metal as a 3rd device — you get 3 compute devices, not 2.

Override proportions with `--tensor-split 0.5,0.5,0`.

## Tested Models

| Model | Source | Size | Arch | Speed (M4 Max 64GB) |
|---|---|---|---|---|
| GLM-4.7-Flash Q4_K_M | [unsloth](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) | 17GB | deepseek2 | ~61 tok/s |
| Qwen3-Coder-30B-A3B Q4_K_M | [unsloth](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18GB | qwen3moe | ~44 tok/s |

Both are downloaded to `~/.models/` by `demo.sh`.

## Gotchas

- Two RPC servers on the same machine must use different `-d` device flags or one will crash
- Workers receive all weights over TCP at startup. Use `rpc-server -c` to cache on disk for faster restarts
- For multi-machine: run `rpc-server -H 0.0.0.0` on each host and point `--rpc` at their IPs
- Ollama blobs sometimes use unsupported arch strings — prefer HuggingFace GGUFs

## Debug

```bash
# Per-layer device assignment
LLAMA_LOG_VERBOSITY=10 ./bin/llama-server -m model.gguf --rpc ... -ngl 99

# RPC protocol debug
GGML_RPC_DEBUG=1 ./bin/rpc-server -p 50052
```
