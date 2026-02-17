# Testing mesh-llm

## Permutations

### 1. Solo (single node)

```bash
mesh-llm --model Qwen2.5-3B --console
```

- API on `:9337`, console on `:3131`
- Console: `host=true, peers=0`
- llama-server has 1 RPC entry (self)

### 2. Two GPU nodes, model fits on host

```bash
# node A (more VRAM, becomes host)
mesh-llm --model Qwen2.5-32B --bind-port 7842
# node B (joins)
mesh-llm --model Qwen2.5-32B --join <TOKEN>
```

- Host elected by highest VRAM — best node always runs it
- Model fits on host → loads solo, 1 RPC
- Workers stay in mesh but idle (prints "Use --split to force")
- API works from both nodes on `:9337` (worker proxies to host)

### 3. Two GPU nodes, forced split

```bash
# host with --split
mesh-llm --model Qwen2.5-32B --bind-port 7842 --split
# worker joins
mesh-llm --model Qwen2.5-32B --join <TOKEN>
```

- `--split` forces tensor split even when model fits on host
- llama-server has 2 RPC entries
- Tensor split proportional to VRAM (e.g. `0.67,0.33`)
- Draft model auto-detected and used
- Check `draft_n_accepted` in response timings

### 4. Two GPU nodes, model too big for one

When the model exceeds host VRAM, split happens automatically without `--split`.
Tested with Hermes-3-Llama-3.1-405B — see experiment notes below.

### 5. Lite client (no GPU)

```bash
mesh-llm --client --join <TOKEN> --console
```

- API tunneled to host over QUIC
- Console shows CLIENT role, "API tunnel", "QUIC · HTTP"
- VRAM total excludes client
- Host stays solo (no tensor split)

### 6. Connection resilience

- Connection drops trigger reconnect (10s timeout)
- Reconnect success: gossip re-exchange → re-election → split restored
- Reconnect failure: peer removed → re-elect solo
- Model load (~20s) should not cause peer loss (120s idle timeout, 10s keepalive)

### 7. Console in all modes

- Diagram reflects actual role (HOST/WORKER/CLIENT)
- Chat proxy works (sidebar panel)
- SSE updates on peer changes

## Deploy to remote node

```bash
just bundle
# scp, then on remote:
codesign -s - ~/bin/mesh-llm ~/bin/rpc-server ~/bin/llama-server
xattr -cr ~/bin/mesh-llm ~/bin/rpc-server ~/bin/llama-server
```

Must codesign + xattr after every scp or macOS kills the binary (exit 137).

## Benchmarks (v0.6.0, Qwen2.5-32B, 20ms RTT)

| Mode | Task | Gen tok/s | Draft acceptance |
|------|------|-----------|------------------|
| Solo (103GB node) | Prose | ~19 | 62% |
| Solo (103GB node) | Code | ~25 | 85% |
| Split (2 nodes) | Prose | 6–9 | 54–63% |
| Split (2 nodes) | Code | ~13 | 87% |

Split overhead from 20ms RTT × 2 RPC round-trips per token. Speculative decoding (draft-max 8) amortizes this for high-acceptance content.

## Experiment: Hermes-3-Llama-3.1-405B-IQ2_M (2025-02-17)

**Setup**: bartowski/Hermes-3-Llama-3.1-405B-GGUF IQ2_M quantization.
4 split files (39.7 + 40 + 40 + 17 GB = 136.7 GB total).

**Why**: 136.7GB doesn't fit on either node alone (103GB / 52GB VRAM).
This is a genuine split-or-nothing scenario — exactly what the mesh is for.

**What worked**:
- VRAM gating correctly detected 150.3GB needed (136.7 × 1.1), waited for second node
- Tensor split 0.67/0.33 across both nodes
- Split GGUF support: rpc-server indexed all 4 split files locally (1138 tensors)
- No mmap, no network transfer for tensor weights — all loaded from local disk
- Model loaded successfully in ~140 seconds
- Inference produced correct output ("Hello! How")

**What didn't work**:
- Generation speed: **0.04 tok/s** (27 seconds per token)
- This is NOT a network/RTT problem — 40ms of RTT is 0.15% of the 27s per token
- It's raw compute: 405B parameters × 126 transformer layers is just too much work for two M4 Max chips
- IQ2_M dequantization is also more expensive per byte than Q4_K_M
- Metal kernel JIT compilation on first inference took several minutes

**Conclusion**: The mesh handles 405B correctly — auto-split, VRAM gating, split GGUF loading all work.
But 405B needs more GPU horsepower than two Apple Silicon chips can provide at usable speed.
A 70B model would be the practical upper limit for this hardware combo.

## Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

Always kill all three — child processes can orphan.
