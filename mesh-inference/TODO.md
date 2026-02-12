# mesh-inference TODO

## High Priority

- [x] **Increase llama-server health timeout** — was 120s, now 600s.
- [x] **Stale peer connect timeout** — was 5 minutes (iroh default), now 10s.
- [ ] **Persist secret key to disk** — every restart generates a new key = new invite token. Should save to `~/.mesh-inference/key` so the token is stable across restarts.
- [ ] **B2B REGISTER_PEER rewriting** — `rewrite.rs` is scaffolded but not wired into the tunnel relay. Without this, B2B worker-to-worker tensor transfers won't go through the mesh (workers try direct TCP to each other using orchestrator-side addresses). Need to intercept command byte 17 in the QUIC→TCP path and rewrite the endpoint field to the local tunnel port.
- [ ] **Auto-detect device** — don't require `--device CPU` or `--device MTL0`. rpc-server with no `-d` flag auto-detects (picks Metal on Mac, CPU otherwise). Just don't pass `-d` by default and let it figure it out.

- [ ] **Pre-seed / cache model tensors** — if the model is already on a worker machine, don't re-send tensors across the mesh. Could hash tensor data and skip `SET_TENSOR` for matching buffers, or pre-load from local GGUF. Also: make tensor transfer resumable so a timeout/restart doesn't mean re-sending 17GB from scratch.
- [ ] **HTTP gateway mode** — any node could act as a gateway: no model, no GPU, just `--gateway 8080`. Gives you `localhost:8080` that proxies HTTP through the QUIC mesh to whichever node is running `--serve`. Flow: client → gateway(localhost:8080) → QUIC bi-stream (type 0x03) → serve node(llama-server:8080) → RPC to GPU workers. Already proven to work in `../p2p-llm` (the `tunnel serve`/`tunnel proxy` pattern) using WebRTC+signal server. With iroh it's simpler — no signal server, the mesh connection already exists. Just add a new stream type and a lightweight HTTP reverse proxy. Each HTTP request maps to one QUIC bi-stream, same multiplexing pattern as the RPC tunnels.

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
