# mesh-inference

P2P mesh sidecar for distributed LLM inference. Connects machines over QUIC (via [iroh](https://iroh.computer/)) and tunnels llama.cpp RPC traffic between them — no port forwarding, no VPN, no config.

```
┌──────────────────────────┐          QUIC (iroh)          ┌──────────────────────────┐
│  Machine A               │◄────────────────────────────►│  Machine B               │
│                          │   NAT traversal / relay       │                          │
│  rpc-server -d MTL0      │                               │  rpc-server -d CPU       │
│  (GPU compute)           │                               │  llama-server            │
│                          │                               │  (orchestrator + API)    │
└──────────────────────────┘                               └──────────────────────────┘
```

Every node runs the same binary. Each node starts a local `rpc-server` and joins a mesh. One node can optionally run `llama-server` to expose an OpenAI-compatible API, using all mesh peers as compute backends.

## Download

Pre-built releases include `mesh-inference`, `rpc-server`, `llama-server`, and all required libraries — no build step needed.

| Platform | File |
|----------|------|
| macOS Apple Silicon (M1/M2/M3/M4) | [mesh-inference-v0.1.0-aarch64-apple-darwin.tar.gz](https://github.com/michaelneale/decentralized-inference/releases/tag/v0.1.0) |

The release bundles binaries from the [llama.cpp B2B fork](https://github.com/nicholasgasior/llama.cpp) which is a superset of upstream llama.cpp. You can also use upstream llama.cpp — see [Using upstream llama.cpp](#using-upstream-llamacpp) below.

## Quick Start

### Install (run on each machine)

```bash
curl -L -o mesh-inference.tar.gz https://github.com/michaelneale/decentralized-inference/releases/download/v0.1.0/mesh-inference-v0.1.0-aarch64-apple-darwin.tar.gz
mkdir -p mesh-inference && tar xzf mesh-inference.tar.gz -C mesh-inference
cd mesh-inference/bin
```

### Two Machines

Start `mesh-inference` on both machines. Either one can go first — every node prints an invite token on startup.

**Machine A** (has GPU):

```bash
./mesh-inference --device MTL0
```

**Machine B** (orchestrator — serves the API):

```bash
./mesh-inference \
  --device CPU \
  --join <TOKEN_FROM_A> \
  --serve 8080 \
  --model ~/.models/your-model.gguf
```

Copy the invite token from whichever node started first and pass it to the other via `--join`. Direction doesn't matter — A can join B or B can join A.

Once the model loads:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

### Three+ Machines

Start `mesh-inference` on each machine. Any node's token works to join — peers discover each other via gossip, so only one token is needed to bootstrap the whole mesh:

```bash
# Machine C (another GPU — use any existing node's token)
./mesh-inference --device MTL0 --join <ANY_TOKEN>
```

The `--serve` node automatically uses all discovered peers as RPC backends.

### Which Machine Should Be the Orchestrator?

The `--serve` node runs `llama-server` which handles:
- Loading the model file (reads from local disk)
- Splitting layers across all RPC backends by available memory
- Embedding lookup (always runs on CPU locally)
- Scheduling compute graphs

It doesn't need a powerful GPU — a small Mac or even a CPU-only machine works fine. The heavy compute (matrix multiplications) happens on the remote GPU workers.

## CLI Reference

```
mesh-inference [OPTIONS]

Options:
  -j, --join <TOKEN>      Join mesh via invite token (repeatable)
      --serve <PORT>      Start llama-server on this HTTP port (requires --model)
      --model <PATH>      Path to GGUF model file
      --min-peers <N>     Wait for N peers before starting llama-server [default: 1]
      --bin-dir <PATH>    Path to directory containing rpc-server and llama-server
                          [default: same directory as mesh-inference binary]
      --device <DEVICE>   Device for local rpc-server: MTL0, CPU, CUDA0, etc.
                          [default: MTL0 on macOS, CPU otherwise]
  -h, --help              Print help
```

## Using Upstream llama.cpp

mesh-inference is just a TCP tunnel — it works with any llama.cpp build that has RPC enabled. The bundled release uses the B2B fork for future peer-to-peer tensor transfer support, but upstream llama.cpp works identically for standard orchestrator→worker inference.

To use upstream llama.cpp:

```bash
# Build upstream llama.cpp with RPC
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
cd ../..

# Point mesh-inference at the upstream binaries
./mesh-inference --bin-dir ./llama.cpp/build/bin --device MTL0
```

Everything works the same — mesh formation, tunneling, model splitting. The only difference is the B2B fork additionally supports direct worker-to-worker tensor transfers (which mesh-inference will rewrite through tunnels in a future release).

## How It Works

1. Every node starts a local `rpc-server` (llama.cpp's RPC backend for compute)
2. Nodes connect over QUIC using iroh — handles NAT traversal automatically
3. Peer discovery propagates via gossip (connect to one peer, learn about all)
4. For each peer, a local TCP tunnel port is allocated (`127.0.0.1:<port>` → QUIC → peer's rpc-server)
5. When `--serve` is set, `llama-server` is launched with all tunnel ports (+ local rpc-server) as `--rpc` endpoints
6. llama.cpp auto-splits the model across all backends by available memory

Each RPC command from llama.cpp opens a TCP connection to a tunnel port. The tunnel opens a QUIC bi-stream to the remote peer, relays the request to the remote rpc-server, and streams the response back. Large tensor transfers (8+ GB during model loading) flow through as raw byte streams.

## Same-Machine Testing

You can test locally by running two instances in separate terminals:

```bash
# Terminal 1
./mesh-inference --device MTL0

# Terminal 2 (paste token from terminal 1, or vice versa)
./mesh-inference \
  --device CPU \
  --join <TOKEN> \
  --serve 8080 \
  --model ~/.models/your-model.gguf
```

Note: even on the same machine, traffic goes through iroh's relay, so model loading will be slower than direct localhost RPC. The real benefit is cross-machine where direct TCP isn't possible.

## Building from Source

```bash
cd mesh-inference
cargo build --release
# Binary at target/release/mesh-inference

# Point it at your llama.cpp build
./target/release/mesh-inference --bin-dir /path/to/llama.cpp/build/bin --device MTL0
```

## Environment Variables

- `RUST_LOG=mesh_inference=info` — standard logging (peer joins, tunnel creation)
- `RUST_LOG=mesh_inference=debug` — verbose logging (every tunnel stream, byte counts)

## Logs

- `/tmp/mesh-inference-rpc-<port>.log` — local rpc-server output
- `/tmp/mesh-inference-llama-server.log` — llama-server output (when using `--serve`)

## Models

Any GGUF model works. Some good options:

```bash
mkdir -p ~/.models

# GLM-4.7-Flash Q4_K_M (~17GB) — fast, good quality
curl -L -o ~/.models/GLM-4.7-Flash-Q4_K_M.gguf \
  https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf

# Qwen3-Coder-30B-A3B Q4_K_M (~18GB) — MoE, great for coding
curl -L -o ~/.models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```
