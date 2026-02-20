# mesh-llm TODO

## Nice to have

### Reactive rebalancing
When a host dies, standby nodes with that model on disk could self-promote.
- On death broadcast: check if dead node's model is now unserved, promote if capable
- On join: check routing table for gaps
- Logic mostly exists in `pick_model_assignment()`, needs topology event trigger

### Don't download what won't fit
Before downloading via `--model`, check if node VRAM >= model_size * 1.1.
Skip download if model won't fit and no split peers available.

### Console VRAM usage bar
Second bar per node showing model_size / node_vram. Data already in gossip, just UI.
