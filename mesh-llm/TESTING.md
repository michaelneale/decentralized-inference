# Testing mesh-llm

## Permutations

### 1. Solo (single node)

```bash
mesh-llm --model Qwen2.5-3B --console
```

- API on `:9337`, console on `:3131`
- Console: `host=true, peers=0`
- llama-server has 1 RPC entry (self)

### 2. Split (2 GPU nodes)

First node (starts the mesh):
```bash
mesh-llm --model Qwen2.5-32B --bind-port 7842
```

Second node (joins as GPU worker):
```bash
mesh-llm --model Qwen2.5-32B --join <TOKEN> --console
```

- llama-server has 2 RPC entries
- Tensor split proportional to VRAM (e.g. `0.67,0.33`)
- API works from both nodes on `:9337`
- Draft model auto-detected and used
- Check `draft_n_accepted` in response timings

### 3. Lite client (no GPU)

```bash
mesh-llm --client --join <TOKEN> --console
```

- API tunneled to host over QUIC
- Console shows CLIENT role, "API tunnel", "QUIC · HTTP"
- VRAM total excludes client
- Host stays solo (no tensor split)

### 4. Connection resilience

- Connection drops trigger reconnect (10s timeout)
- Reconnect success: gossip re-exchange → re-election → split restored
- Reconnect failure: peer removed → re-elect solo
- Model load (~20s) should not cause peer loss (120s idle timeout, 10s keepalive)

### 5. Console in all modes

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

## Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

Always kill all three — child processes can orphan.
