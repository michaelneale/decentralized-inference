# Distributed LLM Inference — Plan & Results

## What We Learned

### Transport benchmarks (same LAN, WiFi)
| Transport | Ping (median) | Throughput | Notes |
|-----------|--------------|------------|-------|
| Raw ICMP ping | ~30-80ms | n/a | WiFi baseline — jittery |
| Raw TCP (python) | n/a | **30 MB/s** | WiFi ceiling |
| WebRTC (p2p-llm) | ~48ms | **~15 MB/s** | Works both directions, survives stealth mode |
| iroh QUIC direct | ~10-14ms | **3-5 MB/s** echo | Only works one direction (firewall), 10MB stalls |
| iroh QUIC relay | ~400ms | **2 MB/s** | Rate-limited, unusable for model data |

### Key findings
1. **iroh is unsuitable** — QUIC multipath holepunching is fragile, throughput is 6x worse than raw TCP even when direct works, 10MB transfers stall, and it can't penetrate macOS stealth mode in both directions. WebRTC ICE handles stealth mode fine.
2. **The transport isn't the real problem** — even at 30 MB/s, loading a 13GB model takes 7+ minutes. The real fix is **not transferring model weights over the network at all**.
3. **llama.cpp RPC is designed wrong for distributed** — it treats remote GPU like a dumb buffer. The orchestrator reads the GGUF, allocates remote buffers, then sends tensor data over TCP. Every `SET_TENSOR` is a synchronous round-trip. The `SET_TENSOR_HASH` + cache optimization exists but requires first-time transfer + cache warmup.
4. **The `-fit` probing issue**: `weight_buft_supported()` calls `ggml_backend_buft_alloc_buffer(buft, 0)` to test op support. For RPC backends, this means: for every tensor × every possible op, it does `ALLOC_BUFFER(0 bytes)` + `supports_op` round-trips. With ~400ms RTT through relay, hundreds of probes = minutes of waiting. `-fit off` skips this.

## The Right Architecture

### Core insight: Both machines have the model file on disk

If every node has the GGUF file locally (or can download it from e.g. HuggingFace), there is **zero need to transfer tensor weights over the network**. The orchestrator tells the worker "load tensor X from your local GGUF at offset Y, size Z" and the worker reads it from its own disk at NVMe speed.

### What needs to go over the network
1. **Control plane** (~KB): "load this model", "here's the compute graph", "here's the result"
2. **Activations** (~MB per token): intermediate results flowing between layers during inference
3. **NOT weights** (~GB): these come from local disk

### Design: Custom RPC protocol

Replace llama.cpp's `ggml-rpc` with a new protocol designed for distributed inference:

#### Phase 1: Model Loading (zero network transfer)
```
Orchestrator                           Worker
    |                                     |
    |-- LOAD_MODEL(gguf_path, layers) --> |
    |                                     | (reads GGUF from local disk)
    |                                     | (allocates GPU buffers)
    |                                     | (loads tensors into GPU)
    |<-- MODEL_READY(layer_info) -------- |
```

- Worker loads its assigned layers from its own local copy of the GGUF
- Model metadata (layer count, dimensions, vocab) is tiny — send once
- No probing, no supports_op round-trips, no fit calculation
- Load time: seconds (NVMe read speed), not minutes

#### Phase 2: Inference (pipeline parallelism)
```
Orchestrator (layers 0-15)             Worker (layers 16-30)
    |                                     |
    | [compute layers 0-15]               |
    |-- ACTIVATIONS(hidden_state) ------> |
    |                                     | [compute layers 16-30]
    |<-- ACTIVATIONS(output) ------------ |
    | [sample token]                      |
```

- Only activations cross the network: `hidden_dim × batch_size × sizeof(f16)`
- For a 4096-dim model: 4096 × 1 × 2 = **8 KB per layer transition per token**
- Even at 15 MB/s WiFi throughput: 8KB takes **0.5ms** — negligible

#### Phase 3: The actual implementation approach

Rather than rewriting llama.cpp's backend system from scratch, we have two options:

**Option A: Patch ggml-rpc to support local loading**
- Add `RPC_CMD_LOAD_GGUF` command: worker loads tensors from local GGUF file
- Skip `SET_TENSOR` entirely for weights — only use it for activations  
- Keep the rest of the RPC protocol for `GRAPH_COMPUTE`, etc.
- Pros: Minimal changes to llama.cpp, still uses its backend scheduler
- Cons: Still limited by the orchestrator-centric model

**Option B: Split-brain architecture (preferred)**
- Each node runs llama-server with its assigned layers
- A lightweight coordinator routes requests through the pipeline
- Each node loads its own layers from its own GGUF copy
- Inter-node communication is just activation tensors
- Pros: Each node is fully autonomous, no complex RPC
- Cons: Need to implement layer splitting logic

**Option C: Patch ggml-rpc with "GGUF-aware set_tensor"**
- Keep llama.cpp's orchestrator model but intercept `SET_TENSOR`
- When the orchestrator tries to send tensor data, instead send the tensor name + GGUF file path
- Worker looks up the tensor in its local GGUF and loads from disk
- This is essentially making the `SET_TENSOR_HASH` + cache path work without first-time transfer
- Pros: Simplest to implement, minimal llama.cpp changes
- Cons: Orchestrator still does the scheduling

### Recommended: Option C (simplest, biggest bang for buck)

#### Implementation:
1. **New RPC command**: `RPC_CMD_SET_TENSOR_GGUF`
   - Request: `{ tensor_name, gguf_path, offset_in_file, size }`
   - Worker opens the GGUF file, seeks to offset, reads data into GPU buffer
   - Falls back to regular `SET_TENSOR` if file not found locally

2. **Orchestrator-side change**: In `ggml_backend_rpc_buffer_set_tensor()`:
   - Before sending tensor data, try `SET_TENSOR_GGUF` with the tensor name
   - If worker has the file → instant load, skip network transfer
   - If worker doesn't have the file → fall back to `SET_TENSOR` (regular network transfer)

3. **Skip fit probing**: Always use `-fit off` equivalent — the orchestrator should know the split ahead of time based on declared memory, not probing

4. **Transport**: Use WebRTC (from p2p-llm) or plain TCP (for LAN). iroh is out.

### Data flow with Option C:
```
Time to load 13GB model:
  Current: 13GB / 15 MB/s = 14 minutes (network transfer)
  With Option C: 13GB / 3 GB/s = 4 seconds (local NVMe read)
  
Time per inference token (activation transfer):
  Hidden state: 4096 × 2 bytes = 8 KB
  Network: 8KB / 15 MB/s = 0.5ms
  Compute: ~20-50ms per layer
  Network overhead: ~1-2% of compute time
```

## Results — Option C Implemented ✅

Branch: `rpc-local-gguf` in `llama.cpp/`

### Changes (197 lines, 4 files)
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — new `RPC_CMD_SET_TENSOR_GGUF` command + server handler
- `ggml/include/ggml-rpc.h` — `ggml_backend_rpc_start_server_with_gguf()` API
- `tools/rpc/rpc-server.cpp` — `--gguf PATH` CLI flag
- `src/llama-model.cpp` — skip probing round-trips for RPC backends

### Same-machine test (localhost TCP)
- **Before:** 111s to load 17GB model (transferring 11.8GB over TCP)
- **After:** 5s to load (zero network transfer, local NVMe reads)
- 427 tensors / 8.3 GB loaded from GGUF, 0 via wire

### Cross-machine test (M4 Max worker ↔ Mac Mini orchestrator, WiFi)
- **Worker:** M4 Max (192.168.86.250), `rpc-server --gguf` on port 50052
- **Orchestrator:** Mac Mini (192.168.86.60), `llama-server --rpc`
- **843 tensors / 16.88 GB loaded from local GGUF on worker — zero bytes over WiFi**
- **0 tensors transferred over network**
- Model ready in ~30s (buffer alloc + disk reads + warmup), vs 14+ min before
- Inference: ~2 tok/s (WiFi GRAPH_COMPUTE round-trip latency ~480ms/token)

### Inference benchmarks (warmed up, WiFi)
| Run | Prompt eval | Generation | Notes |
|-----|-------------|------------|-------|
| Cold (first request) | 2.4 tok/s | 1.3 tok/s | First request after startup |
| Warm (cache hit) | 6.7-7.2 tok/s | 7.2-8.7 tok/s | Same prompt, prompt cache hit |
| New prompt, 50 tok gen | 1.4 tok/s | 4.0 tok/s | Fresh prompt |
| New prompt, 100 tok gen | 1.5 tok/s | **5.5 tok/s** | Longer gen amortizes round-trips |

### Remaining bottleneck: GRAPH_COMPUTE round-trips
Every token generation requires at least one synchronous `GRAPH_COMPUTE` RPC
round-trip. On WiFi (~80ms avg ping), that's 100-500ms per token depending on
graph complexity. Longer generation batches amortize better (5.5 tok/s at 100 tokens).
See "Next Steps" below for analysis.

## Next Steps

1. **Reduce inference round-trip cost** — see analysis below
2. **Upstream PR** — clean up the patch and submit to llama.cpp upstream. The `SET_TENSOR_GGUF` + probing skip are fully backward compatible.
3. **Mesh/discovery** — integrate with mesh-inference or p2p-llm for NAT traversal and auto-discovery.
4. **Model distribution** — auto-download GGUF from HuggingFace on workers that don't have it yet.
