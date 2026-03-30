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

### 9a. Local runtime load/unload and process view

```bash
# Running node
mesh-llm --model Qwen2.5-0.5B-Instruct-Q4_K_M --console

# Operator surface
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm models
mesh-llm ps
mesh-llm drop Llama-3.2-1B-Instruct-Q4_K_M

# REST surface
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

- `mesh-llm models` shows the local models currently backed by running inference processes
- `mesh-llm ps` shows backend, port, pid, and whether each process is startup-managed or runtime-managed
- `GET /api/runtime` and `GET /api/runtime/processes` agree with the CLI output
- Loading a small local model adds it to `/v1/models` without restarting the node
- Dropping the runtime-loaded model removes it cleanly while the primary model stays available

### 10. Console model picker

- Dropdown appears when >1 warm model
- Switching models highlights the serving node in topology view
- Chat routes to selected model via API proxy

## Mesh Identity

### 16. Mesh ID generation (originator)

```bash
# With --mesh-name (deterministic ID)
mesh-llm --model Qwen2.5-3B --mesh-name "test-mesh"
```

- Log: `📌 Mesh ID: <hex>`
- `~/.mesh-llm/last-mesh` contains the same hex
- Restart with same `--mesh-name` → same mesh ID (deterministic)
- Different `--mesh-name` → different mesh ID

### 17. Mesh ID propagation (joiner)

```bash
# Originator
mesh-llm --model Qwen2.5-3B --mesh-name "test-mesh"
# Joiner
mesh-llm --model Qwen2.5-3B --join <TOKEN>
```

- Joiner log: `📌 Mesh ID: <same hex as originator>`
- Joiner's `~/.mesh-llm/last-mesh` matches originator's mesh ID
- Mesh ID arrives via gossip (worker nodes) or routing table (passive clients)

### 18. Sticky mesh preference

- Join a mesh → `~/.mesh-llm/last-mesh` saved
- On next `--auto`, `score_mesh()` adds +500 for meshes with matching `mesh_id`
- If that mesh is dead (not on Nostr), scoring proceeds normally without bonus

## Bootstrap Proxy

### 19. Instant API during GPU bootstrap

```bash
# Originator (already running)
mesh-llm --model Qwen2.5-3B --port 8090
# Joiner
mesh-llm --model Qwen2.5-3B --join <TOKEN> --port 8091
```

- Joiner log: `⚡ API ready (bootstrap): http://localhost:8091`
- BEFORE `rpc-server` or `llama-server` starts on joiner:
  - `curl localhost:8091/v1/models` → lists mesh models
  - `curl localhost:8091/v1/chat/completions` → inference via tunnel to originator
- Log: `⚡ Bootstrap proxy handing off to full API proxy`
- After handoff, API continues working (now served locally or via election)

### 20. Bootstrap proxy not started for originator

```bash
mesh-llm --model Qwen2.5-3B
```

- No `⚡ API ready (bootstrap)` message (only joiners get bootstrap proxy)
- API port opens only after election resolves

## Idle Mode & Management API

### 21. Idle mode (no args)

```bash
mesh-llm
```

- Log: `mesh-llm v0.19.0 — 52GB VRAM, 7 models on disk` + suggested commands
- Console on `:3131`, inference port `:9337` returns 503
- `curl localhost:3131/api/status` → JSON with `model_name: "(idle)"`, 0 peers
- `curl localhost:3131/api/discover` → Nostr mesh listings (JSON array)
- **Dormant QUIC**: peers from previous sessions cannot reconnect (no ghost peers)


### 22. Join via console

```bash
mesh-llm    # idle mode
# In browser: http://localhost:3131 → Discover → Join
# Or via API:
curl -X POST localhost:3131/api/join -H 'Content-Type: application/json' -d '{"token":"..."}'
```

- `/api/join` triggers full flow: connect → gossip → assign model → download → serve
- Console updates: status, peers, model name all reflect new state
- Inference port starts working after model loads

### 23. Management API while serving

```bash
mesh-llm --auto
# After serving:
curl localhost:3131/api/status   # JSON: node, peers, models, mesh_id, mesh_name
curl localhost:3131/api/events   # SSE stream
curl localhost:3131/api/discover # Nostr meshes (current mesh marked by mesh_id)
```

- `/api/status` includes `mesh_id` and `mesh_name`
- SSE events push every 2s and on topology changes
- Discover results can be matched to current mesh by `mesh_id`

### 24. HTTP proxy single-request connection contract

- Send a routed inference request, then pipeline or reuse the same TCP
  connection for a second request.
- Verify only the first request reaches the selected upstream.
- Verify the proxy closes the routed connection after the first response.
- Verify the upstream-observed request includes `Connection: close`.

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
