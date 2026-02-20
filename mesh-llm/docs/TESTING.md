# Testing mesh-llm

## Single-model permutations

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

- Both nodes run solo (no split) — each is its own host
- API works from both nodes on `:9337`

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

### 4. Two GPU nodes, model too big for one

When the model exceeds host VRAM, split happens automatically without `--split`.

### 5. Lite client (no GPU)

```bash
mesh-llm --client --join <TOKEN> --port 9555
```

- Uses ephemeral key (unique identity, works on same machine as GPU node)
- `/v1/models` lists all served models from gossip
- API tunneled to correct host per model via QUIC
- VRAM total excludes client

## Multi-model permutations

### 6. Two nodes, different models

```bash
# node A: seeds mesh with two models, serves 3B
mesh-llm --model Qwen2.5-3B --model GLM-4.7-Flash --console
# node B: joins, auto-assigned to GLM (needed, on disk)
mesh-llm --join <TOKEN>
```

- `/v1/models` on either node lists both models
- Requesting GLM from node A routes via QUIC to node B
- Requesting 3B from node B routes via QUIC to node A
- Both run solo (no tensor split)
- Console shows both models warm with node counts

### 7. Auto-assignment

```bash
# seeder declares two models
mesh-llm --model Qwen2.5-3B --model GLM-4.7-Flash
# joiner with no --model
mesh-llm --join <TOKEN>
```

- Joiner scans `~/.models/`, picks unserved model already on disk
- Log: "Assigned to serve GLM-4.7-Flash (needed by mesh, already on disk)"

### 8. Lite client with multi-model

```bash
# GPU nodes running as above
mesh-llm --client --join <TOKEN> --port 9555
```

- Client sees all models via gossip (ephemeral key = unique identity)
- `/v1/models` lists all served models
- Routes to correct host per model
- Streaming works through cross-model QUIC tunnel

### 9. Drop a model

```bash
mesh-llm drop GLM-4.7-Flash-Q4_K_M
```

- Node serving that model exits cleanly
- Other nodes unaffected
- Model goes cold in console

### 10. Console model picker

- Dropdown appears when >1 warm model
- Switching models highlights the serving node in topology view
- Chat routes to selected model via API proxy

## Resilience

### 11. Dead peer cleanup

- Kill a node with `kill -9`
- Heartbeat detects it in ~120s (60s interval, 2 consecutive failures)
- On-use detection is faster: tunnel failure → immediate death broadcast
- Dead model goes cold, peer removed from list, death broadcast to mesh
- Dead peer won't be re-added by gossip (dead_peers set)
- Console updates automatically

### 12. Node rejoin

- Kill a node, restart it with `--join <token>`
- Rejoin loop (60s) reconnects to bootstrap if connection drops
- Inbound reconnection clears dead_peers entry
- Model goes warm again, cross-model routing resumes

### 13. Gossip stability

- Regossip after becoming host should NOT cause restart loops
- Log should show "still host, no restart needed" on re-check
- llama-server starts exactly once per election (not 5-9 times)
- Heartbeat gossip doesn't re-discover dead peers (discover_peers=false)

## Single-machine testing with ephemeral keys

Set `MESH_LLM_EPHEMERAL_KEY=1` to give a second process a unique identity on the same machine.
Without this, both processes share `~/.mesh-llm/key` and appear as the same node.

### 14. Forced split on one machine

```bash
# Terminal 1: host with --split
mesh-llm --model Qwen2.5-3B --port 9337 --split --console

# Terminal 2: worker with ephemeral key
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm --model Qwen2.5-3B --join <TOKEN> --port 9338 --split --max-vram 1
```

- Host starts solo, then re-elects with split when worker joins
- Worker becomes rpc-server, proxies API to host
- Tensor split proportional to VRAM (e.g. `0.98,0.02`)
- Kill worker → host detects via heartbeat (~60s), reverts to solo mode

### 15. Passive client on one machine

```bash
# Terminal 1: host
mesh-llm --model Qwen2.5-3B --port 9337

# Terminal 2: passive client (--client uses ephemeral key automatically)
mesh-llm --client --join <TOKEN> --port 9338
```

- Client connects without gossip (no peer list entry on host)
- `/v1/models` returns models from routing table
- Inference routes through QUIC tunnel to host
- Host does NOT see client in its peer list (zero per-client state)

## Deploy to remote node

```bash
just bundle
# scp, then on remote:
codesign -s - ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
xattr -cr ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
```

Must codesign + xattr after every scp or macOS kills the binary (exit 137).

## Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

Always kill all three — child processes can orphan.
