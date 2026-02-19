# mesh-llm TODO

## Event-driven mesh (in progress)
Replaced aggressive 15s health check polling with event-driven death detection:
- [x] Slow heartbeat (60s) catches silent failures
- [x] Death broadcast: when tunnel fails, broadcast "X is dead" to all peers
- [x] Peers verify death before removing (prevents split-brain false positives)
- [x] Clean shutdown broadcast (STREAM_PEER_LEAVING)
- [x] `open_http_tunnel` failure triggers `handle_peer_death` → broadcast
- [ ] Test: kill -9 a node, verify other nodes detect + remove within ~60s
- [ ] Test: clean shutdown (ctrl-c), verify peers remove immediately

## Unified passive mode
Clients and standby GPU nodes use same lightweight path:
- Get routing table from any active node (STREAM_ROUTE_REQUEST)
- Route requests by tunneling to hosts
- No gossip participation
- `--client` = passive + never promote (0 VRAM)
- Standby GPU = passive + can promote when needed
- [ ] Refactor `run_client()` and `run_idle_gpu()` into single `run_passive()` path
- [ ] Hash-based host selection: `hash(client_id + model) % hosts.len()`
- [ ] Periodic routing table refresh (30s) instead of gossip

## Reactive rebalancing
On topology changes, nodes self-decide whether to promote:
- [ ] On join: check routing table, serve unserved model if possible
- [ ] On death broadcast: standby checks if dead node's model is now unserved
- [ ] Standby with matching model promotes itself to active

## Don't download what won't fit
- [ ] Before downloading via `--model`, check if node VRAM >= model_size * 1.1
- [ ] Skip download if model won't fit and no split peers available

## Console VRAM usage bar per node
Add a second bar showing how much of node's VRAM is used by its model.
Data is already in gossip — just UI work in console.html.

## Test forced tensor split
- [ ] Pick a small model (e.g. Qwen2.5-3B)
- [ ] Force split across two nodes even though it fits on one (test `--split` flag or hack VRAM detection)
- [ ] Verify rpc-server workers and tensor split still work correctly with new event-driven mesh
- [ ] Confirm solo mode still works (no accidental split when model fits locally)
