# Model Demand: Unified Want System

## Core data structure

```rust
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    /// Unix timestamp of the most recent request or declaration
    pub last_active: u64,
    /// Total requests seen across the mesh (merged via max)
    pub request_count: u64,
    /// Is this model explicitly declared via --model by a live node?
    /// (Not gossiped — each node computes locally by checking if any
    /// peer or self has it in requested_models)
    #[serde(skip)]
    pub pinned: bool,
}
```

Gossiped in `PeerAnnouncement`:
```rust
/// Demand signals for models — replaces request_rates.
/// Keyed by model name. Merged across peers via max(last_active), max(request_count).
#[serde(default)]
model_demand: HashMap<String, ModelDemand>,

/// Keep requested_models for backward compat with old nodes
#[serde(default)]
requested_models: Vec<String>,
```

## Where demand entries are created/refreshed

1. **`--model` flag at startup** → entry with `last_active = now, request_count = 0`
   (pinned while this node is alive)

2. **API request for a served model** (existing `record_request`) → refresh `last_active = now`, increment `request_count`

3. **API request for an unserved model** (currently just 503s) → same as above.
   The request is a signal even if we can't serve it yet.

4. **Receiving gossip from old nodes**: if `model_demand` is empty but
   `requested_models` has entries, synthesize demand entries from them
   with `last_active = now, request_count = 0`.

## Merge strategy on gossip receive

For each model in the received `model_demand`:
- `last_active = max(ours, theirs)`
- `request_count = max(ours, theirs)`

This means demand info only grows/refreshes — never decreases. The
"decay" happens naturally: if nobody refreshes `last_active`, it gets
old and eventually falls below the TTL.

## TTL / decay

```rust
const DEMAND_TTL_SECS: u64 = 7200; // 2 hours
```

`pick_model_assignment()` only considers models where:
- `now - last_active < DEMAND_TTL_SECS`, OR
- `pinned` (a live node declared it via --model)

## What replaces what

| Old | New |
|-----|-----|
| `requested_models` (per-node, gossiped) | Kept for backward compat, but new nodes use `model_demand` |
| `mesh_wanted` (local HashSet, never gossiped) | Removed — `model_demand` IS the mesh-wide wanted set |
| `request_rates` (per-node, gossiped) | Subsumed by `model_demand.request_count` |
| Nostr `wanted` computation | Derived from `model_demand`: entries where model is not served and `last_active` is recent |

## pick_model_assignment() changes

Instead of:
```
for m in mesh_wanted:
    if serving_count[m] == 0 and local_models.contains(m): ...
```

Now:
```
active_demands = model_demand where (now - last_active < TTL) or pinned
sort by request_count desc (hottest first)
for m in active_demands:
    if serving_count[m] == 0 and (local_models.contains(m) or downloadable(m)):
        assign(m)
    elif serving_count[m] < needed_replicas(m):  // future: scale by demand
        assign(m)
```

## Backward compat

- Old nodes ignore `model_demand` (unknown field, serde skips)
- Old nodes still send `requested_models` and `request_rates`
- New nodes receiving from old nodes: merge `requested_models` into
  `model_demand` as synthetic entries, merge `request_rates` into
  `request_count`
- Mixed mesh works, just with less rich demand data flowing through old nodes

## What this fixes

1. **Node leaves, want disappears**: No — demand was gossiped to all nodes,
   they keep spreading it. Decays naturally after TTL if no new requests.

2. **Client asks for unserved model**: Creates demand entry, gossips out,
   standby node picks it up.

3. **"What does the mesh want?"**: One answer — `model_demand` sorted by
   activity. No more three competing concepts.

4. **Spare VRAM sits idle**: Standby nodes look at demand and pick the
   hottest unserved model they can handle.
