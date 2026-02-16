# mesh-inference

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage.

See [DESIGN.md](DESIGN.md) for internals — stream types, B2B rewriting, tunnel manager, etc.

```
src/
├── main.rs        CLI, startup, mode dispatch
├── mesh.rs        iroh QUIC endpoint, gossip, peer management, STUN
├── election.rs    Auto host election by VRAM, dynamic llama-server lifecycle
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.rs     Web console: HTTP server, SSE, API, model catalog
└── console.html   Embedded single-page dashboard
```
