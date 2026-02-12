# mesh-inference (experiment — archived)

> **Status: Experiment that kind of worked, but not viable for real use.**
>
> This was a proof-of-concept for tunneling llama.cpp RPC traffic over iroh QUIC
> between machines. It successfully formed meshes, tunneled RPC, and ran inference
> on a single machine. Cross-machine inference partially worked but hit fundamental
> limitations — see below.
>
> The approach of tunneling stock llama.cpp RPC is being replaced by a custom RPC
> protocol that avoids transferring model weights over the network entirely.
> See `../PLAN.md` for the new direction.

## What Worked
- Mesh formation over iroh QUIC with gossip-based peer discovery
- Bidirectional QUIC stream multiplexing (gossip + tunnel on single connection)
- Same-machine inference: model loaded in ~111s, ~51 tok/s
- Self-contained release tarball with llama.cpp binaries + dylibs
- Persistent node identity (key saved to `~/.mesh-inference/key`)

## What Didn't Work
- **iroh direct UDP holepunching**: Only works in one direction. macOS firewall
  stealth mode silently drops incoming UDP, preventing path validation. WebRTC ICE
  handles this fine; iroh's QUIC multipath doesn't.
- **iroh relay throughput**: Public relay servers are rate-limited to ~4 MB/s.
  Unusable for transferring 13GB+ of model weights.
- **iroh QUIC throughput**: Even with direct connection, only achieved 3-5 MB/s
  (echo test) vs 30 MB/s raw TCP on the same WiFi link. 10MB transfers stalled.
- **llama.cpp RPC model loading**: The protocol sends all tensor weights from
  orchestrator to worker over TCP. 13GB at 15 MB/s = 14+ minutes. The `-fit`
  auto-probing phase also does hundreds of synchronous round-trips.

## Key Insight

The transport doesn't matter when you're transferring 13GB of weights. The fix is
**not transferring weights at all** — both machines have the GGUF on disk, so the
worker should load tensors from its own local file. Only activations (~8KB per layer
per token) need to cross the network.

## Architecture (for reference)

```
┌──────────────────────────┐          QUIC (iroh)          ┌──────────────────────────┐
│  Machine A               │◄────────────────────────────►│  Machine B               │
│  rpc-server (GPU)        │   NAT traversal / relay       │  rpc-server (CPU)        │
│                          │                               │  llama-server (API)      │
└──────────────────────────┘                               └──────────────────────────┘
```

Same binary on every node. Each starts `rpc-server`, joins mesh, tunnels RPC over QUIC.
One node runs `llama-server` as orchestrator.

## Building

```bash
cargo build --release
./target/release/mesh-inference --bin-dir /path/to/llama.cpp/build/bin --device MTL0
```

## Benchmarks collected

| Transport | Ping | Throughput | Notes |
|-----------|------|------------|-------|
| Raw TCP (python) | — | 30 MB/s | WiFi ceiling |
| WebRTC (p2p-llm) | ~48ms | ~15 MB/s | Works both directions |
| iroh QUIC direct | ~14ms | 3-5 MB/s | One direction only, stalls on large transfers |
| iroh QUIC relay | ~400ms | ~2 MB/s | Rate-limited, unusable |
