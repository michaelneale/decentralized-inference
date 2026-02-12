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

## Quick Start

### Prerequisites

Build the [llama.cpp B2B fork](https://github.com/nicholasgasior/llama.cpp) with RPC support:

```bash
git clone https://github.com/nicholasgasior/llama.cpp llama.cpp-rpc-b2b
cd llama.cpp-rpc-b2b
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_RPC=ON   # macOS with Metal
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

Download a GGUF model (e.g. [GLM-4.7-Flash Q4_K_M](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF), ~17GB):

```bash
mkdir -p ~/.models
curl -L -o ~/.models/GLM-4.7-Flash-Q4_K_M.gguf \
  https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf
```

### Two-Machine Setup

**Machine A** (has GPU — starts the mesh):

```bash
mesh-inference --bin-dir ./llama.cpp-rpc-b2b/build/bin --device MTL0
```

This prints an invite token. Copy it.

**Machine B** (orchestrator — joins and serves the API):

```bash
mesh-inference \
  --bin-dir ./llama.cpp-rpc-b2b/build/bin \
  --device CPU \
  --join <PASTE_TOKEN> \
  --serve 8080 \
  --model ~/.models/GLM-4.7-Flash-Q4_K_M.gguf
```

Once the model loads, query it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

### Three+ Machines

Start Machine A as above. Then join any number of additional machines with the same token:

```bash
# Machine C (another GPU)
mesh-inference --bin-dir ./llama.cpp-rpc-b2b/build/bin --device MTL0 --join <TOKEN>
```

Peers discover each other via gossip — only one token is needed to bootstrap. The `--serve` node automatically uses all discovered peers as RPC backends.

### Same-Machine Testing

You can test locally by running two instances (simulates two network-separated machines):

```bash
# Terminal 1: GPU worker
mesh-inference --bin-dir ./llama.cpp-rpc-b2b/build/bin --device MTL0

# Terminal 2: CPU orchestrator (paste token from terminal 1)
mesh-inference \
  --bin-dir ./llama.cpp-rpc-b2b/build/bin \
  --device CPU \
  --join <TOKEN> \
  --serve 8080 \
  --model ~/.models/GLM-4.7-Flash-Q4_K_M.gguf
```

Note: same-machine traffic goes through iroh's relay server, so model loading will be slower than direct localhost RPC. This is expected — the real benefit is cross-machine.

## CLI Reference

```
mesh-inference [OPTIONS]

Options:
  -j, --join <TOKEN>      Join mesh via invite token (repeatable)
      --serve <PORT>      Start llama-server on this HTTP port (requires --model)
      --model <PATH>      Path to GGUF model file
      --min-peers <N>     Wait for N peers before starting llama-server [default: 1]
      --bin-dir <PATH>    Path to llama.cpp build/bin directory
                          [default: ../llama.cpp-rpc-b2b/build/bin]
      --device <DEVICE>   Device for local rpc-server: MTL0, CPU, CUDA0, etc.
                          [default: auto-detect]
  -h, --help              Print help
```

## How It Works

1. Every node starts a local `rpc-server` (llama.cpp's RPC backend)
2. Nodes connect over QUIC using iroh — handles NAT traversal automatically
3. Peer discovery propagates via gossip (connect to one peer, learn about all)
4. For each peer, a local TCP tunnel port is allocated (`127.0.0.1:<port>` → QUIC → peer's rpc-server)
5. When `--serve` is set, `llama-server` is launched with all tunnel ports (+ local rpc-server) as `--rpc` endpoints
6. llama.cpp auto-splits the model across all backends by available memory

Each RPC command from llama.cpp creates a TCP connection to a tunnel port. The tunnel opens a QUIC bi-stream to the remote peer, relays the request to the remote rpc-server, and streams the response back. Large tensor transfers (during model loading) flow through as raw byte streams.

## Building from Source

```bash
cd mesh-inference
cargo build --release
```

Binary is at `target/release/mesh-inference`.

## Environment Variables

- `RUST_LOG=mesh_inference=debug` — verbose logging (tunnel streams, peer discovery)
- `RUST_LOG=mesh_inference=info` — standard logging (peer joins, tunnel creation)

## Logs

- `/tmp/mesh-inference-rpc-<port>.log` — local rpc-server output
- `/tmp/mesh-inference-llama-server.log` — llama-server output (when using `--serve`)
