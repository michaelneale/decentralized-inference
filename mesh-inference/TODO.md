# mesh-inference TODO

## High Priority

- [x] **Increase llama-server health timeout** — was 120s, now 600s.
- [x] **Stale peer connect timeout** — was 5 minutes (iroh default), now 10s.
- [ ] **Persist secret key to disk** — every restart generates a new key = new invite token. Should save to `~/.mesh-inference/key` so the token is stable across restarts.
- [ ] **B2B REGISTER_PEER rewriting** — `rewrite.rs` is scaffolded but not wired into the tunnel relay. Without this, B2B worker-to-worker tensor transfers won't go through the mesh (workers try direct TCP to each other using orchestrator-side addresses). Need to intercept command byte 17 in the QUIC→TCP path and rewrite the endpoint field to the local tunnel port.
- [ ] **Auto-detect device** — don't require `--device CPU` or `--device MTL0`. rpc-server with no `-d` flag auto-detects (picks Metal on Mac, CPU otherwise). Just don't pass `-d` by default and let it figure it out.

## Medium Priority

- [ ] **Retry/reconnect on QUIC connection drop** — if the tunnel dies mid-transfer, everything fails. Should reconnect and resume.
- [ ] **Model loading timeout is brutal** — 17GB over relay takes ~100s. Would be much faster with direct UDP (iroh supports it). Investigate why it's going through relay even on LAN.
- [ ] **Reduce tunnel log spam** — every 3278-byte HELLO exchange logs two lines. At info level, hundreds of these during model fitting. Should be debug level.
- [ ] **Graceful shutdown** — Ctrl-C should kill child rpc-server and llama-server processes. Currently they may orphan.

## Low Priority

- [ ] **Linux / CUDA support** — build and test on Linux with CUDA backends
- [ ] **x86_64 macOS binary** — cross-compile or CI build for Intel Macs
- [ ] **Performance profiling** — measure tunnel overhead vs direct TCP for large tensor transfers
- [ ] **Multiple models** — support different models on different nodes
- [ ] **Web UI** — show mesh topology, peer status, tunnel throughput
