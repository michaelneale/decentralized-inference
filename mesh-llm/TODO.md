# mesh-llm TODO

## Done
- [x] Event-driven mesh: 60s heartbeat + death broadcasts (replaced 15s aggressive polling)
- [x] Death/leaving broadcasts: peers notified immediately on failure or clean shutdown
- [x] Dead peers set: prevents gossip from re-adding killed nodes
- [x] Rejoin loop: re-connects to bootstrap token every 60s
- [x] Routing table protocol (STREAM_ROUTE_REQUEST)
- [x] Unified passive mode: `run_passive()` replaces `run_client()` + `run_idle_gpu()`
- [x] Hash-based host selection: distributes across multiple hosts for same model
- [x] `--auto` flag: discover mesh via Nostr and join in one command
- [x] Passive node scaling: `join_passive()` — no gossip, no peer tracking
- [x] Route table refresh: passive nodes poll every 30s, stay in sync
- [x] On-demand QUIC connect: passive nodes connect to hosts from routing table
- [x] Active nodes don't track passive clients (zero per-client server state)
- [x] Console JS fix (servingSel ordering)

## Test forced tensor split
Verify split mode still works after event-driven mesh changes.
- [ ] Pick Qwen2.5-3B, force split across two nodes
- [ ] Verify rpc-server workers and tensor split still work
- [ ] Verify solo mode still works (no accidental split)
- [ ] Verify death broadcast works with split group (host dies → workers notice)

## Reactive rebalancing (nice to have)
- [ ] On death broadcast: standby checks if dead node's model is now unserved
- [ ] On join: check routing table for gaps, promote if capable
- [ ] Logic mostly exists in `pick_model_assignment()`, needs topology event trigger

## Don't download what won't fit
- [ ] Before downloading via `--model`, check if node VRAM >= model_size * 1.1
- [ ] Skip download if model won't fit and no split peers available

## Console VRAM usage bar per node
- [ ] Second bar showing model_size / node_vram per node (data in gossip, just UI)
