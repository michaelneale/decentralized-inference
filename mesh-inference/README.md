# mesh-inference

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for full usage and setup.

## Building

```bash
cargo build --release
```

Requires rpc-server and llama-server from the [patched llama.cpp fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) (not needed for `--client` mode):
```bash
git clone -b rpc-local-gguf https://github.com/michaelneale/llama.cpp.git
cmake -B build -S llama.cpp -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build --config Release
```

Or from the project root: `just build`

## Architecture

See [DESIGN.md](DESIGN.md) for internals — stream types, B2B rewriting, tunnel manager, etc.

```
src/
├── main.rs      CLI, startup, role branching (worker/host/client)
├── mesh.rs      iroh QUIC endpoint, gossip (roles, VRAM, models), peer management
├── tunnel.rs    TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs   REGISTER_PEER interception and endpoint rewriting
├── launch.rs    rpc-server and llama-server process management
├── console.rs   Web console: HTTP server, SSE, API, auto-election, model catalog
└── console.html Embedded single-page dashboard (dark theme, chat, peer list)
```

### Key Concepts

**Gossip**: Each node broadcasts its `EndpointAddr`, `NodeRole`, available VRAM, and local model names. All peers exchange this on connect and propagate to the mesh.

**Auto-election**: Every Active node (has a model + rpc-server running) participates. Highest VRAM wins host. Deterministic — no consensus protocol needed, every node computes the same answer from gossip state.

**Dynamic membership**: The elected host watches `peer_change_rx`. When worker count changes, it kills llama-server, recalculates tensor-split from VRAM gossip, and relaunches. Dead peers are detected in ~40s (QUIC keep-alive at 10s, idle timeout at 30s).

**Tunnel manager**: Creates a local TCP listener per peer. llama.cpp connects to `127.0.0.1:<tunnel_port>`, traffic is relayed over QUIC to the peer's rpc-server. The `rpc_port` is an `AtomicU16` so the tunnel can start before rpc-server exists.

**Console**: A subcommand (`mesh-inference console`) that starts an HTTP server with embedded HTML. The mesh node starts once on boot and never restarts — role changes just update gossip and start/stop child processes. Token stays stable.
