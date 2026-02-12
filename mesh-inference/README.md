# mesh-inference

P2P mesh for distributed LLM inference. Connects machines over QUIC (via [iroh](https://iroh.computer/)) and tunnels llama.cpp RPC traffic between them — no port forwarding, no VPN.

```
┌──────────────────────────┐          QUIC (iroh)          ┌──────────────────────────┐
│  Machine A               │◄────────────────────────────►│  Machine B               │
│  rpc-server (GPU)        │   NAT traversal / relay       │  rpc-server (CPU)        │
│                          │                               │  llama-server (API)      │
└──────────────────────────┘                               └──────────────────────────┘
```

Every node runs the same binary. Each starts a local `rpc-server` and joins a mesh. One node runs `llama-server` to expose an OpenAI-compatible API, splitting the model across all peers.

## Install

macOS Apple Silicon:

```bash
curl -L -o mesh-inference.tar.gz \
  https://github.com/michaelneale/decentralized-inference/releases/download/v0.1.0/mesh-inference-v0.1.0-aarch64-apple-darwin.tar.gz
mkdir -p mesh-inference && tar xzf mesh-inference.tar.gz -C mesh-inference
cd mesh-inference/bin
```

Self-contained — includes `mesh-inference`, `rpc-server`, `llama-server`, and all required libraries.

## Get a Model

```bash
mkdir -p ~/.models
curl -L -o ~/.models/GLM-4.7-Flash-Q4_K_M.gguf \
  https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf
```

This downloads GLM-4.7-Flash (~17GB). Any GGUF model works.

## Run

You need two machines. Run the install step on both.

**Machine A** — GPU worker:

```bash
./mesh-inference --device MTL0
```

It prints an invite token. Copy it.

**Machine B** — orchestrator (runs `llama-server`):

```bash
./mesh-inference \
  --device CPU \
  --join <PASTE_TOKEN> \
  --serve 8080 \
  --model ~/.models/GLM-4.7-Flash-Q4_K_M.gguf
```

The model file only needs to be on Machine B (the `--serve` node). It will load the model and push tensor data to all peers over the mesh.

Wait for `llama-server ready` then:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

Either machine can start first. Direction of `--join` doesn't matter.

### Three+ Machines

Any node's token works — peers discover each other via gossip:

```bash
./mesh-inference --device MTL0 --join <ANY_TOKEN>
```

The `--serve` node automatically uses all discovered peers as RPC backends.

## CLI

```
mesh-inference [OPTIONS]

Options:
  -j, --join <TOKEN>      Join mesh via invite token
      --serve <PORT>      Run llama-server on this HTTP port (requires --model)
      --model <PATH>      Path to GGUF model file
      --min-peers <N>     Wait for N peers before starting [default: 1]
      --bin-dir <PATH>    Directory containing rpc-server/llama-server
                          [default: same directory as mesh-inference]
      --device <DEVICE>   Device for local rpc-server: MTL0, CPU, CUDA0, etc.
                          [default: MTL0 on macOS]
```

## Using Upstream llama.cpp

The release bundles the [B2B fork](https://github.com/nicholasgasior/llama.cpp) (superset of upstream). Stock llama.cpp works too — build with `-DGGML_RPC=ON` and point `--bin-dir` at it:

```bash
./mesh-inference --bin-dir /path/to/llama.cpp/build/bin --device MTL0
```

## Building from Source

```bash
cd mesh-inference
cargo build --release
./target/release/mesh-inference --bin-dir /path/to/llama.cpp/build/bin --device MTL0
```

## Logs

- `/tmp/mesh-inference-rpc-<port>.log` — rpc-server
- `/tmp/mesh-inference-llama-server.log` — llama-server
- `RUST_LOG=mesh_inference=info` for mesh/tunnel logging
- `RUST_LOG=mesh_inference=debug` for per-stream byte counts
