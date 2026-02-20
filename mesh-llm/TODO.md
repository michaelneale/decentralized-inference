# mesh-llm TODO

## Done
- [x] Event-driven mesh: 60s heartbeat + death broadcasts (replaced 15s aggressive polling)
- [x] Death/leaving broadcasts: peers notified immediately on failure or clean shutdown
- [x] Dead peers set: prevents gossip from re-adding killed nodes
- [x] Rejoin loop: re-connects to bootstrap token every 60s
- [x] Routing table protocol (STREAM_ROUTE_REQUEST)
- [x] Unified passive mode: `run_passive()` replaces `run_client()` + `run_idle_gpu()`
- [x] Hash-based host selection: `host_for_model()` distributes across multiple hosts
- [x] `--auto` flag: discover mesh via Nostr and join in one command
- [x] `any_host()` fallback for requests with no model match
- [x] Console JS fix (servingSel ordering)

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
- [ ] Force split across two nodes even though it fits on one
- [ ] Verify rpc-server workers and tensor split still work with event-driven mesh
- [ ] Confirm solo mode still works (no accidental split when model fits locally)

## Scaling to 1000s (future)
- [ ] Passive nodes: stateless connections (connect per request, not persistent)
- [ ] Active nodes don't track individual clients (zero per-client server state)
- [ ] Routing table caching on clients (30-60s TTL, fetch from any active node)
