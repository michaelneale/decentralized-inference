# mesh-llm TODO

## Test forced tensor split
Verify split mode still works after event-driven mesh changes.
- [ ] Pick Qwen2.5-3B (2.1GB, fits on both Local and Mini easily)
- [ ] Force split: hack VRAM detection or add a `--force-split` flag so model doesn't run solo
- [ ] Verify: rpc-server starts on worker, host does tensor split, inference works
- [ ] Verify: solo mode still works when model fits (no accidental split)
- [ ] Verify: death broadcast + heartbeat work correctly with split group (host dies → workers notice)

## Scaling to 1000s of passive nodes

### Problem
Passive nodes (clients, standby GPU) currently join gossip and get added to active
nodes' peer lists. 1000 passive nodes = 1000 peer entries, 1000 heartbeat probes/60s.

### Plan
Passive nodes connect but skip gossip. Active nodes don't track them.

1. **Passive connect (no gossip):**
   - Passive node calls `endpoint.connect(host_addr, ALPN)` — QUIC connection opens
   - Passive node sends a STREAM_ROUTE_REQUEST instead of STREAM_GOSSIP as first interaction
   - Active node handles the route request, returns routing table
   - Active node does NOT call `add_peer()` — no peer list entry, no heartbeat
   - The QUIC connection stays alive for the passive node to reuse for tunneling

2. **Passive node lifecycle:**
   - On start: connect to bootstrap, get routing table, cache it
   - On request: open HTTP tunnel stream to appropriate host (from cached routing table)
   - Every 30-60s: refresh routing table from any known active node
   - On tunnel failure: remove that host from local cache, try next host, refresh table
   - No gossip, no peer list membership, no heartbeat from active side

3. **Active node changes:**
   - `handle_incoming()`: accept connection but only add to peer list if gossip stream arrives
   - Currently: every inbound connection → store in `connections` → dispatch streams
   - Change: store connection for stream dispatch, but only `add_peer()` after gossip exchange
   - HTTP tunnel and route request streams work without peer list entry

4. **What stays the same:**
   - Active ↔ Active: full gossip, peer list, heartbeat, election (small group, 5-20 nodes)
   - Passive node holds its own connection(s) — can tunnel to any active host
   - Hash-based host selection uses routing table, not peer list

### Impact
- Active node memory: O(active_nodes) instead of O(all_nodes)
- Heartbeat cost: O(active_nodes²) instead of O(all_nodes × active_nodes)  
- 1000 passive nodes: zero per-client state on server, just QUIC connections (lightweight)
- Passive nodes are truly stateless from the server's perspective

## Reactive rebalancing (nice to have)
On topology changes, standby nodes self-decide whether to promote:
- On death broadcast: check if dead node's model is now unserved, promote if capable
- On join: check routing table for gaps
- Most logic exists in `pick_model_assignment()`, needs topology event trigger

## Don't download what won't fit
- Before downloading via `--model`, check if node VRAM >= model_size * 1.1
- Skip download if model won't fit and no split peers available

## Console VRAM usage bar per node
Add second bar showing model_size / node_vram per node. Data already in gossip.
