# Distributed LLM Inference — Design Notes & Results

Historical notes from development. See README.md for current usage.

## Transport Benchmarks (same LAN, WiFi)

| Transport | Ping (median) | Throughput | Notes |
|-----------|--------------|------------|-------|
| Raw ICMP ping | ~30-80ms | n/a | WiFi baseline, jittery |
| Raw TCP | n/a | **30 MB/s** | WiFi ceiling |
| iroh QUIC direct | ~10-14ms | **3-5 MB/s** | Only works one direction with macOS stealth mode |
| iroh QUIC relay | ~400ms | **2 MB/s** | Rate-limited, unusable for model data |

### Key finding

iroh QUIC throughput is lower than raw TCP, but for the actual inference workload (activations are ~10KB/token, not GB) it doesn't matter. **QUIC mesh adds zero measurable overhead to inference tok/s** vs raw TCP — confirmed by back-to-back testing of all configurations.

The real bottleneck was always per-token RPC round-trip latency over WiFi (~40ms × 2 RTTs = ~80ms/token network cost), not throughput.

## The Core Insight

Both machines have the model file on disk. There is **zero need to transfer model weights over the network**. The worker loads tensors from its own local GGUF at NVMe speed.

### What crosses the network
- **Control plane** (~KB): graph compute commands, tensor metadata
- **Activations** (~10KB/token): intermediate results at graph split boundaries
- **NOT weights** (~17GB): these come from local disk

## llama.cpp Patches

Branch: `rpc-local-gguf` in `michaelneale/llama.cpp`

### 1. SET_TENSOR_GGUF (zero-transfer weight loading)
- Client sends tensor name instead of tensor data
- Server looks up tensor in its local GGUF index, reads from disk
- **Result**: 17GB model loads in ~9s (NVMe) vs 14+ min (WiFi transfer)

### 2. Probing skip for RPC backends
- `weight_buft_supported()` short-circuits for "RPC" prefix buffers
- Eliminates hundreds of alloc/free probing round-trips at startup

### 3. get_alloc_size cache + non-weight GGUF skip
- FNV hash cache for `get_alloc_size` responses: 558 → 8 calls per token
- Skip GGUF lookup for intermediate tensors (only weights have GGUF entries)
- **Result**: 2-18x improvement in tok/s from eliminating ~550 wasted RPC round-trips

### 4. B2B direct server-to-server transfers
- `REGISTER_PEER`: orchestrator tells workers about each other
- `PUSH_TENSOR_TO_PEER`: source worker pushes activation data directly to destination worker
- `PEER_TENSOR_DATA`: the actual data transfer (reuses SET_TENSOR wire format)
- Threaded client handling in rpc-server (needed for concurrent peer connections)
- **Result**: activation tensors at graph split boundaries flow directly worker→worker, bypassing orchestrator

## Inference Benchmarks

### 2-node mesh (M4 Max + Mac Mini, WiFi, GLM-4.7-Flash 17GB)

| Configuration | Prompt eval | Generation |
|---|---|---|
| Mini orchestrator (15% local + 85% remote, `--tensor-split 0.85,0.15`) | 27 tok/s | **21 tok/s** |
| M4 Max orchestrator (82% local + 18% remote) | 25 tok/s | **16 tok/s** |
| Mini orchestrator, all remote (`GGML_METAL_DEVICES=0`) | 3 tok/s | 2.5 tok/s |
| Local only (M4 Max, no mesh) | 160 tok/s | 68 tok/s |

### 3-node mesh (Mini orchestrator + 2 M4 Max workers, 40/40/20 split)

| Configuration | Generation |
|---|---|
| 3-node with B2B | **12-13 tok/s** |
| 3-node without B2B | 12 tok/s |

B2B shows no improvement in this test because both workers are on the same machine (the transfers are already localhost). The benefit appears when workers are on separate machines.

### Per-token steady state
- 7 RPC calls per token (down from ~290 before chatter reduction)
- 2 blocking network round-trips per token (irreducible with current protocol)
- WiFi ~40ms RTT × 2 = ~80ms network cost per token

## Key Design Decisions

- **iroh over libp2p**: simpler API, built-in NAT traversal
- **Same binary everywhere**: worker/host/client role determined by flags
- **Single ALPN + single QUIC connection per peer**: multiplexed by first byte (gossip/tunnel/map/http)
- **Model weights never transfer over the network**: both sides have the GGUF locally
- **`--no-mmap` always used**: prevents full-file mmap crash on unified memory Macs with less VRAM than model size
- **Orchestrator uses local GPU directly**: not through a local rpc-server (avoids redundant RPC overhead)
- **Mini as orchestrator is faster than M4 Max as orchestrator**: because the bulk compute (85%) runs on M4 Max with zero network cost
- **B2B via tunnel rewriting**: each worker's sidecar intercepts REGISTER_PEER and rewrites endpoints to local tunnel ports, enabling direct worker-to-worker transfers through the mesh
- **Lite client via raw byte relay**: HTTP and SSE streaming work transparently through the QUIC tunnel with no protocol awareness
