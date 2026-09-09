# mesh-llm Design

> Current serving note: mesh serving embeds the Skippy/llama.cpp stage runtime
> in `mesh-llm`. Operator runbooks are available in
> [`../MESHES.md`](../MESHES.md) and
> [`../SKIPPY_SPLITS.md`](../SKIPPY_SPLITS.md).

The proposed graph-derived replacement for model-aware stage filtering is
specified in [`SKIPPY_GRAPH_FILTER_V2.md`](SKIPPY_GRAPH_FILTER_V2.md).

mesh-llm connects nodes over QUIC (via [iroh](https://iroh.computer)), gossips
capabilities and model state, routes OpenAI-compatible requests through the
mesh, and can coordinate package-backed Skippy stage execution for models that
need to be split across peers.

## Architecture

The workspace is split across many crates (see the root `AGENTS.md` for the
full map). The shipped `mesh-llm` binary is a thin entry point: Clap parsing
lives in `mesh-llm-cli`, one-shot command handlers in `mesh-llm-commands`, and
the host-side runtime in `mesh-llm-host-runtime`, whose module layout carries
most of the behavior described in this document:

```text
crates/mesh-llm-host-runtime/src/
├── lib.rs                   Crate entry; runtime entrypoints called by the binary
├── api/                     Management API (:3131): status, models, search, events, discover
├── crypto/                  Key management, envelope encryption, keychain
├── inference/
│   ├── election.rs          Per-model host election and split planning
│   ├── skippy/              Embedded staged runtime integration
│   ├── virtual_llm.rs       Inter-model collaboration hooks
│   └── pipeline.rs          Inference pipeline coordination
├── mesh/mod.rs              Node struct, QUIC endpoint, gossip, peer management, mesh identity
├── models/
│   ├── capabilities.rs      Vision/audio/multimodal/reasoning capability inference
│   ├── catalog.rs           Model catalog and HuggingFace downloads
│   ├── resolve/             Model reference resolution, mmproj lookup
│   └── ...                  GGUF parsing, inventory, search, topology
├── network/
│   ├── proxy.rs             HTTP proxy: request parsing, model routing, response helpers
│   ├── router.rs            Request classification, model scoring, multimodal routing
│   ├── tunnel.rs            TCP ↔ QUIC relay, B2B tunnel map
│   ├── nostr.rs             Nostr discovery, score_mesh(), smart_auto()
│   ├── affinity.rs          Sticky fallback and cache-evidence routing
│   └── openai/              OpenAI transport glue
├── logging/                 Local request lifecycle, persistence, replay, retention, audit, webhooks
├── plugin/                  Plugin host, runtime, transport, MCP bridge
├── plugins/
│   └── blobstore/           Request-scoped media object storage for multimodal
├── protocol/                Wire protocol types, protobuf encoding/decoding
├── runtime/                 Top-level process orchestration, startup coordination
├── runtime_data/            Internal collector snapshots for management API fan-in
└── system/                  Hardware detection, benchmarking, self-update
```

## Topology Roles

```rust
enum NodeRole {
    Worker,                      // provides staged GPU compute for a model
    Host { http_port: u16 },     // runs the local serving runtime, serves HTTP API
    Client,                      // no compute, just API access via tunnel
}
```

Roles are exchanged via gossip. Live-state badges are separate and use `Client`, `Standby`, `Loading`, and `Serving`. Preferred peers use `meshllm.node.v1` protobuf on QUIC ALPN `mesh-llm/1`; legacy peers may still negotiate `mesh-llm/0` and use the older JSON gossip payloads. A node transitions Worker → Host when elected.

A newly connected peer is quarantined until it sends a valid `GossipFrame` with `gen = 1` (quarantine-until-gossip admission model). Only streams 0x01 (GOSSIP) and 0x05 (ROUTE_REQUEST) are accepted before admission. All other streams are rejected until the peer is admitted.

## Control-Plane Protocol

The control plane uses QUIC ALPN `mesh-llm/1` with the `meshllm.node.v1` protobuf schema. Scoped control-plane streams use 4-byte LE framing followed by protobuf bytes.


See [message_protocol.md](message_protocol.md) for the full wire format specification.

## QUIC Stream Types

Single QUIC connection per peer, multiplexed by 1-byte prefix:

| Byte | Type | Purpose | Format |
|------|------|---------|--------|
| 0x01 | GOSSIP | Peer announcements (role, serving, VRAM, models, explicit interest, demand, mesh_id) | protobuf `GossipFrame` |
| 0x02 | TUNNEL | TCP relay for remote runtime compute traffic | raw TCP relay |
| 0x03 | TUNNEL_MAP | B2B tunnel port map exchange | protobuf `TunnelMap` |
| 0x04 | TUNNEL_HTTP | TCP relay to a remote node's HTTP API | raw TCP relay |
| 0x05 | ROUTE_REQUEST | Routing table for passive nodes (hosts + models) | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | Death broadcast (immediate, from any node that detects a death) | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | Clean shutdown broadcast (ctrl-c) | protobuf `PeerLeaving` |

Streams 0x02 and 0x04 are raw TCP relay tunnels and are not subject to protobuf framing or generation validation.

## Multi-Model

Different nodes serve different models. The API proxy on each node peeks at
the `model` field in POST bodies and routes to the correct host via QUIC tunnel.

- **One model per node** — no VRAM double-commitment
- **Solo by default** — if VRAM ≥ model_size × 1.1, run solo
- **Per-model election groups** — nodes serving the same model elect a host independently
- **Auto-assignment** — joiners without `--model` get assigned based on mesh needs and what's on disk

### HTTP/1.1 Connection Contract

For routed inference requests, the proxy buffers and routes exactly one HTTP
request per client connection:

- The full request is framed first (`Content-Length` or chunked) before routing.
- The forwarded upstream request is rewritten to `Connection: close`.
- After the buffered request is written upstream, the proxy only relays the
  response back to the client.
- Additional client bytes on the same connection are ignored and dropped when
  the connection closes; they are not replayed to the already-selected
  upstream.

This is an intentional safety tradeoff. The proxy does not currently implement
per-request routing for persistent HTTP/1.1 keep-alive or pipelined multi-
request connections. Clients should open a fresh connection for each routed
inference request.

## Mesh Identity

Every mesh has a stable `mesh_id`:
- **Requirement-aware mesh**: canonical genesis policy hash, deterministic from
  the immutable creation-time requirements policy
- **Legacy named unrestricted mesh**: `hash(name + originator_nostr_pubkey)`,
  deterministic and unique per creator
- **Legacy unnamed unrestricted mesh**: random UUID, persisted to
  `~/.mesh-llm/mesh-id`

For requirement-aware meshes, the immutable inputs are node-version bounds,
protocol-generation bounds, and release-attestation policy. Local owner-trust
policy is excluded from the mesh identity hash.

Propagated via gossip (`PeerAnnouncement.mesh_id`) and routing table (`RoutingTable.mesh_id`).
Published in Nostr listings (`MeshListing.mesh_id`).
Saved to `~/.mesh-llm/last-mesh` on successful join for sticky preference scoring.

Mesh metadata and admission facts are different things:

- Node version is self-advertised metadata in gossip and status surfaces.
- Protocol generation is a negotiated and validated transport fact.
- Build attestation is release certification proof, not proof that a remote
  process is unmodified official code running with trusted hardware or OS state.
- Certified-build admission is not remote runtime attestation.

### Release provenance

The shipped `mesh-llm` executable uses embedded release attestation, and the
release-signing trust root is separate from owner trust. This applies only to
the packaged `mesh-llm` binary, not SDK, XCFramework, or other native
artifacts. `missing` is expected for unstamped local and dev builds, `valid`
means the packaged binary matches a trusted release signer, and `invalid` means
the bytes changed after packaging. Operators can verify stamped packaged
binaries with `cargo run -p xtask -- release-attestation inspect --binary <path-to-packaged-mesh-llm> --public-key-file <release-signing-public-key.json>`.
Bare `inspect --binary ...` is only sufficient for unstamped binaries that
should classify as `missing`; stamped binaries require `--public-key-file` and
otherwise report `invalid` with an explicit error.

This is provenance and admission hardening, not runtime integrity proof. Mesh
requirements can require a certified build at admission time through
`require_release_attestation` and `release_signer_keys`, but that does not prove
the remote process is running unmodified code on trusted hardware or OS state.

Changing mesh requirements creates a new mesh.

## Bootstrap Proxy

When joining an existing mesh, a tunnel-only API proxy starts immediately on the
local port — before the local serving runtime is ready. Requests are tunneled to
mesh hosts via QUIC. When the real `api_proxy` is ready, it takes over the listener.

This gives instant API access (within seconds of `mesh-llm serve --join`) while the local
GPU loads its model in the background.

## Local Node Config

`mesh-llm serve` owns startup model configuration. By default it reads
`~/.mesh-llm/config.toml`, which now serves as the unified local node config for:

- startup models under `[[models]]`
- local GPU startup policy under `[gpu]`
- plugin declarations under `[[plugin]]`

Phase 2 keeps this config intentionally local-node only. There is no authored mesh-wide
`[[nodes]]` state yet.

CLI precedence is by concern:

- explicit `--model` or `--gguf` ignores configured `[[models]]` for model
  selection and tuning; an exact, unique `--model` ref may still inherit its
  configured pinned GPU selector
- explicit `--ctx-size` overrides configured `ctx_size`
- explicit `--device` wins over configured device selectors; stable IDs and
  backend names resolve before startup, `CPU` bypasses GPU preflight, and
  `auto` keeps the inherited selector
- plugin config continues to load from the same file

Pinned GPU startup is also local-node only:

- `[gpu].assignment = "pinned"` means each configured `[[models]]` entry must carry its own `gpu_id`
- valid IDs come from the local `mesh-llm gpus` / `mesh-llm gpus --json` inventory surface
- pin resolution is host-local and fail-closed: missing, ambiguous, unsupported, or stale IDs abort startup and config push for that node instead of silently falling back to auto placement
- explicit CLI `--model` / `--gguf` keeps the selected artifact, context, and
  projector choices. An exact, unique `--model` ref carries only its effective
  pinned GPU selector. Unmatched refs and `--gguf` paths carry no configured
  model identity but may inherit only `defaults.hardware.device`; duplicate
  configured refs are rejected because the CLI has no profile selector
- an explicit `--device` is resolved even when `[gpu].assignment = "auto"`

Bare `mesh-llm serve` is the config-owned path. If `[[models]]` is empty, it warns,
prints help, and exits cleanly. Background services use that path directly.

Creation-time mesh requirement fields live under `[mesh_requirements]` in
`~/.mesh-llm/config.toml`:

- `min_node_version`
- `max_node_version`
- `min_protocol_version`
- `max_protocol_version`
- `require_release_attestation`
- `release_signer_keys`

Those fields are evaluated at mesh creation and join time. Changing them creates
a new requirement-aware mesh; it does not mutate a running mesh.

## Passive Mode

Two flavors, one code path (`run_passive()`):
- **`--client`**: pure consumer, ephemeral key, no gossip, routing table only
- **Standby GPU**: has VRAM + models on disk, watches for topology changes, promotes when needed

Passive nodes get routing tables via `STREAM_ROUTE_REQUEST` (0x05), not full gossip.
Scales to hundreds of clients without O(n²) gossip cost.

## Demand-Aware Rebalancing

- `record_request(model)` increments per-model counter on every API proxy request
- `snapshot_request_rates()` computes delta each gossip cycle (requests/min)
- Rates gossipped in `PeerAnnouncement.request_rates`
- Standby nodes check on 60s timer + topology changes via `tokio::select!`
- Promotion triggers: (1) model with 0 servers, (2) ≥3x demand imbalance + ≥10 req/min, (3) single hot model ≥10 req/min

## Latency-Aware Tensor Split

When a model requires splitting across nodes:
1. Filter candidates by `rtt_ms < 80ms`
2. Sort by RTT ascending (unknown RTT sorts last)
3. Greedily accumulate VRAM until `≥ model_size × 1.1`
4. Stop — don't add unnecessary high-latency peers

## Event-Driven Peer Management

- **Reconnect-gossip-probe** — when a QUIC connection drops, the node reconnects and awaits gossip with a 10s timeout. If gossip fails, the peer is removed immediately. Dead peer cleanup typically completes in ~41s after `kill -9`.
- **60s heartbeat** with 2-consecutive-failure threshold (fallback path)
- **Death broadcasts** (`STREAM_PEER_DOWN`, protobuf) for immediate notification
- **Clean shutdown** (`STREAM_PEER_LEAVING`, protobuf) on ctrl-c — only removes the sender, not other peers
- **Dead peers set** prevents gossip from re-adding killed nodes
- **Tunnel failure detection** triggers immediate death broadcast

## B2B Direct Transfer

When the model is split across workers, activation data flows directly
between workers (1 hop) instead of through the host (2 hops). Each node
broadcasts `{EndpointId → tunnel_port}` via `STREAM_TUNNEL_MAP` so peers can
open direct worker-to-worker tunnels. In the current embedded staged runtime,
stage-to-stage activation traffic uses the Skippy binary stage transport over
these direct paths; the legacy llama.cpp RPC rewrite path (`rewrite.rs`
intercepting `REGISTER_PEER`) survives only in the `mesh-client` compatibility
surface.

## Operator request logging

Request logging is a host-local subsystem, not a mesh protocol. Ingress and
runtime lifecycle code submit bounded lifecycle facts to the logging service;
the service holds active state, persists terminal summaries and optional
artifact pointers, and publishes replayable request, operation, and system
events. Persistence and artifact work run outside the serving hot path so a
logging failure can be represented without blocking an inference response.

The embedded console reads this state through trusted-local `/api/logs/**`
management routes. The list endpoint merges active state with durable history;
the detail route can read a terminal request immediately after completion.
Artifact pointer metadata is distinct from content retrieval, and only an
explicit redacted-artifact capture policy makes content eligible for an
operator opt-in download. The local log API is never advertised in gossip,
added to ALPN, or substituted for existing status/runtime SSE.

The Logs page hydrates from the ledger and owns a separate SSE lifecycle for
`/api/logs/events`. Standard SSE event IDs allow native browser reconnect;
the stream sends channels, backend-supported request-ID filters, and the last
received logging replay cursor. Ledger-only filters reopen the stream with
that last cursor and trigger REST hydration instead of being serialized into
the SSE request. A replay gap, including an omitted or `null` recovery cursor,
causes an authoritative ledger refresh, while a disconnected stream eventually
uses bounded polling. Unsupported hosts keep this lifecycle inert. The logging
route therefore owns its own connection states without changing the status or
runtime event contracts.

Retention bounds durable terminal records, pointers, audit records, webhook
delivery records, persistence queues, replay capacity, exports, and artifacts.
Scoped cleanup, request deletion, export, and dead-letter retry are audited
management operations: the host validates the request, including retry delivery
context/input, and the UI only renders the returned receipt. Artifact download
is explicit and limited to available redacted captures. Metrics export only
bounded lifecycle and maintenance outcomes; it never exports local log data or
payloads. The production `OutputEvent` path is a separate local projection of
bounded request/event IDs, replay channel/sequence, outcomes, status, duration,
and token counts; it does not add those IDs or payloads to mesh/network or OTLP
telemetry.

See [the operator logging guide](../LOGGING.md) for settings and recovery, and
[telemetry.md](../plugins/telemetry.md) for the explicit metrics privacy
boundary.

## Management API (port 3131)

Separate from the inference API (port 9337). Serves mesh management endpoints
and the embedded web dashboard.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Lightweight management liveness and local mesh/serving summary (JSON); returns 200 whenever the management process answers |
| `/api/status` | GET | Live mesh state (JSON): node, peers, routing, targets |
| `/api/models` | GET | Mesh model inventory for the dashboard and operators |
| `/api/search` | GET | Search the built-in catalog or Hugging Face with the same JSON payload shape as `mesh-llm models search --json` |
| `/api/model-interests` | GET, POST | Read back or register local explicit interest keyed by canonical model refs |
| `/api/model-interests/{model_ref}` | DELETE | Clear local explicit interest for one canonical model ref |
| `/api/model-targets` | GET | Ranked model targets from explicit interest, active demand, and serving visibility |
| `/api/events` | GET | SSE stream of status updates (2s interval + on change) |
| `/api/discover` | GET | Browse Nostr-published meshes, or LAN mDNS advertisements when `--mesh-discovery-mode mdns` is active |
| `/api/discovery/lan-details` | POST | Return local LAN discovery detail after invite-token proof; advertised only when the management API is LAN-reachable, and the raw token is never returned |
| `/api/chat` | POST | Proxy to inference API (`/v1/chat/completions`) |
| `/` | GET | Embedded web dashboard |

The dashboard is a thin client. Live node state comes from `/api/status` and
`/api/events`, while model inventory comes from `/api/models`. `/api/search`
provides the same read-only model search payload as `mesh-llm models search --json`
to operators and future UI flows without requiring CLI output parsing.
`/api/model-interests` is intentionally local-node-only in phase 2: it stores
explicit interest on the connected host and advertises those canonical refs
through gossip for read-only mesh target ranking. `/api/model-targets` combines
local and peer explicit interest with active demand and current serving
visibility; it does not launch, unload, or auto-assign models by itself. Entries
should use canonical refs such as `org/repo@rev:variant`. Mesh management works
without the HTML via curl/scripts.

Runtime reconciliation is opt-in. When `[runtime] reconcile_model_targets = true`
is set, the local runtime may load an already-present local GGUF for a locally
registered explicit interest, but only when `/api/model-targets` says the target
is wanted, unserved, and a single-node capacity fit for the current node. A
host that also sets `reconcile_model_target_demand_upgrades = true` may replace
a less-demanded local model with a locally present, higher-ranked unserved
target once fresh active request demand crosses
`model_target_demand_upgrade_min_requests`. Stale demand older than
`model_target_demand_upgrade_max_age_secs` is advisory only. Runtime
reconciliation does not download models, start split serving, or act on
requested-only seed interest.

`/api/model-targets` keeps raw inputs and computed hints separate. Each target
reports `signals` from observed mesh state (`explicit_interest_count`,
`request_count`, `last_active_secs_ago`, `serving_node_count`, and `requested`)
alongside `derived` hints (`target_rank`, `wanted`, and optional
`wanted_reason`) plus `capacity_advice`. Capacity advice is also advisory: it
uses existing catalog size and node VRAM signals to report whether a target is
already served, has a single-node fit, is a split-capable aggregate candidate,
has a known shortfall, or cannot be judged because model size or node capacity
is unknown. Client-role exclusions and missing VRAM are reported separately so
operators can distinguish "this node cannot host" from "this node did not
advertise usable capacity." Ranking is advisory and deterministic: explicit
interest is ordered first, active request demand second, requested-only unserved
models third, then recent activity, display name, and canonical ref. A
`requested` signal does not add a second rank boost when request demand is
already present; it only breaks into the ranking as its own requested-only
signal. `wanted` means the model is currently unserved and has at least one
explicit-interest, active demand, or requested-model signal. It is not a
desired replica count, an unload/load command, or proof that a split package is
ready on every participant.

Always enabled on port 3131 (configurable with `--console <port>`).

### Runtime Data Collector

Broad management API reads are assembled through the internal `runtime_data/`
collector. Subsystems still own their source-of-truth state: `runtime/` owns
process and local-instance observations, `models::inventory` owns GGUF scans and
metadata, `network::metrics` plus `mesh::Node` own routing counters, and
`plugin::PluginManager` / plugin runtime own plugin actions and lifecycle. Those
owners publish small snapshots into the collector through subsystem-local
producer handles instead of moving ownership into the API layer.

The collector stores runtime status/process rows, local-instance and inventory
snapshots, routing metrics, and passive plugin report snapshots. API routes keep
their public JSON shapes stable by adapting collector views back into the
existing `api/status.rs` payload types. `/api/events` sends an initial
`/api/status` payload and then wakes from the collector's single versioned
`watch` stream; producer updates mark dirty bits synchronously and never await
collector locks on hot request paths.

This refactor is intentionally internal-only. It does not add public HTTP
fields, gossip fields, or plugin protocol messages. If a broad API read needs a
new data source, prefer adding a subsystem-local producer publication and a
collector snapshot/view adapter over rejoining subsystem internals directly in
route handlers.

## No-Arg Behavior

`mesh-llm` with no arguments prints the standard CLI help and exits.

Management API and inference listeners are only started by active modes such as
`--model`, `--join`, `--auto`, or `--client`. This avoids surprising port binds
for users who run the binary just to check usage.

## Hardware Detection

`hardware.rs` collects GPU and host info at startup via the `Collector` trait:

```rust
trait Collector {
    fn collect(&self) -> Vec<Metric>;
}
```

| Implementation | Platform | Source |
|---|---|---|
| `DefaultCollector` | macOS (Metal/CPU) | `system_profiler`, `vm_stat` |
| `DefaultCollector` | Linux NVIDIA | `/proc/driver/nvidia`, `nvidia-smi` |
| `DefaultCollector` | Linux AMD | `/sys/class/drm`, `rocm-smi` |
| `TegraCollector` | Jetson / Tegra | sysfs + `tegrastats` |

The shipped `mesh-llm` binary builds `mesh-llm-system` with `skippy-devices`.
In that mode, runtime-selectable GPU inventory is authoritative from the
embedded Skippy/llama backend device API. Platform collectors remain legacy
fallbacks for non-Skippy builds and diagnostic surfaces, but they must not
invent GPU count, backend identity, or usable runtime capacity when the embedded
backend reports no selectable GPU.

`survey()` calls all applicable collectors and returns a `HardwareSurvey` with `gpu_name`, `gpu_vram` (per-GPU bytes), `gpu_reserved` (per-GPU reserved or unavailable bytes when the platform reports a true reserved/unavailable metric), `vram_bytes` (total), `hostname`, `is_soc`, per-device `GpuFacts` entries, `system_ram_bytes` (total system RAM when the platform reports it), and `ram_offload_bytes` (the share of `vram_bytes` backed by system RAM rather than accelerator memory, derived once in `query`). Benchmark-derived memory-bandwidth and compute-throughput hints are attached later when cached or freshly measured results are available. ROCm `rocm-smi --showmeminfo` and Intel `xpu-smi` discovery expose live used-memory counters, so mesh-llm intentionally omits `gpu_reserved` for those backends instead of reinterpreting used bytes as reserved memory.

### Gossip Fields

`PeerAnnouncement` fields carried in the `meshllm.node.v1` protobuf `GossipFrame`:

| Field | Type | Description |
|---|---|---|
| `gpu_name` | `Option<String>` | Comma-separated GPU model names |
| `hostname` | `Option<String>` | System hostname |
| `is_soc` | `Option<bool>` | True for Tegra/Jetson (unified memory) |
| `gpu_vram` | `Option<String>` | Comma-separated per-GPU VRAM in bytes |
| `gpu_reserved_bytes` | `Option<String>` | Comma-separated per-GPU reserved bytes when the platform reports a true reserved/unavailable metric |
| `hardware.memory` | `optional MemoryInfo` | Itemized capacity behind `vram_bytes`: total, driver reserve, platform reserve, configured reserve, usable, plus system RAM and the RAM-backed share of the local budget as informational items |
| `gpu_mem_bandwidth_gbps` | `Option<String>` | Comma-separated per-GPU memory bandwidth measurements or cached benchmark results |
| `gpu_compute_tflops_fp32` | `Option<String>` | Comma-separated per-GPU FP32 compute-throughput hints |
| `gpu_compute_tflops_fp16` | `Option<String>` | Comma-separated per-GPU FP16 compute-throughput hints |
| `available_model_metadata` | `repeated CompactModelMetadata` | GGUF-derived metadata per available model |
| `available_model_sizes` | `map<string, uint64>` | File sizes in bytes per model name |
| `mesh_id` | `optional string` | Stable mesh identity (self entry only) |
| `demand` | `repeated ModelDemandEntry` | Per-model demand entries (self entry only) |

GGUF-derived metadata (architecture, quantization type, tokenizer, RoPE parameters, expert counts) is transported via `CompactModelMetadata` in the `available_model_metadata` field. This lets peers learn model capabilities without downloading the file. The `ScannedModel` type in the proto schema carries the same information for catalog-level model listings. Current gossip sanitization still strips `available_models`, `available_model_metadata`, and `available_model_sizes` before sending announcements on the wire, so these schema fields remain compatibility surface rather than a second transitive model-inventory source.

### `--no-enumerate-host` Flag

By default, nodes broadcast their GPU name, hostname, VRAM capacity, reserved bytes, and the itemized capacity block (`hardware.memory`) to all mesh peers. Pass `--no-enumerate-host` to suppress this hardware identification. `is_soc` and the `vram_bytes` budget are always sent. Benchmark-derived bandwidth and compute hints remain additive optional fields when available. `gpu_reserved_bytes` stays omitted on backends such as ROCm and Intel where the tooling does not report a true reserved/unavailable memory metric.

```
--no-enumerate-host    # opt out: suppress GPU name and hostname from gossip
```

### API Shape

`GET /api/status` — self node:
```json
{
  "my_hostname": "carrack",
  "my_is_soc": false,
  "gpus": [{"name": "NVIDIA RTX 5090", "vram_bytes": 34359738368, "reserved_bytes": 1073741824, "mem_bandwidth_gbps": 1792.0, "compute_tflops_fp32": 104.8, "compute_tflops_fp16": 209.6}]
}
```

For ROCm and Intel hosts, `reserved_bytes` is omitted because their standard CLI telemetry exposes live used-memory counters rather than a true reserved/system-memory value.

Routing health in `/api/status.routing_affinity.target_reputation` is local to
the process that served the management API. It records behavioral routing
signals such as penalized targets and routes reordered away from penalized
targets. This is a local availability/reliability aid only: it is not gossiped,
not a cross-mesh trust score, and not a replacement for owner attestation or
model-output verification. The operator-facing behavior is specified in
[Local Node Reputation](../NODE_REP.md).

`peers[]` entries (only when peer has not passed `--no-enumerate-host`):
```json
{"hostname": "lemony-28", "is_soc": true, "gpus": [{"name": "Tegra AGX Orin", "vram_bytes": 0}]}
```

## Nostr Discovery

Opt-in mesh advertisement via Nostr relays (NIP-89, kind 31990):
- `--publish`: republish listing every 60s (TTL 120s)
- `--auto`: discover meshes, score them, health-probe, join best
- Publish watchdog: if publisher dies, another node takes over
- `score_mesh()`: region match (+200), capacity, node count, VRAM, sticky preference (+500)
- `smart_auto()`: picks best mesh or recommends starting new one with models for your VRAM
