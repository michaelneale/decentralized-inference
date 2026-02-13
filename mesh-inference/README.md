# mesh-inference

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage and setup.

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
├── mesh.rs      iroh QUIC endpoint, gossip with roles, tunnel map exchange
├── tunnel.rs    TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs   REGISTER_PEER interception and endpoint rewriting
└── launch.rs    rpc-server and llama-server process management
```
