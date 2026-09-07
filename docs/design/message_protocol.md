# mesh-llm Message Protocol

Use this reference for public mesh communication over the `meshllm.node.v1`
protobuf schema on QUIC ALPN `mesh-llm/1` and for the separate private
owner-control lane on `mesh-llm-control/1`.

## ALPN

Control-plane connections prefer ALPN `mesh-llm/1`.


## Stream Types

Each QUIC connection carries multiple logical streams, distinguished by a 1-byte prefix:

| Byte | Name | Direction | Format |
|------|------|-----------|--------|
| 0x01 | GOSSIP | bidirectional | protobuf `GossipFrame` |
| 0x02 | TUNNEL | bidirectional | raw TCP relay (not protobuf) |
| 0x03 | TUNNEL_MAP | send | protobuf `TunnelMap` |
| 0x04 | TUNNEL_HTTP | bidirectional | raw TCP relay (not protobuf) |
| 0x05 | ROUTE_REQUEST | bidirectional | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | send | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | send | protobuf `PeerLeaving` |
| 0x08 | PLUGIN_CHANNEL | bidirectional | plugin protocol |
| 0x09 | PLUGIN_BULK_TRANSFER | send | plugin protocol bulk data |
| 0x0a | PLUGIN_MESH_STREAM | bidirectional | plugin protocol mesh stream |
| 0x0b | CONFIG_SUBSCRIBE | reserved | legacy mesh-plane config stream ID; do not reuse |
| 0x0c | CONFIG_PUSH | reserved | legacy mesh-plane config stream ID; do not reuse |
| 0x0d | SUBPROTOCOL | bidirectional | protobuf `MeshSubprotocolOpen`, then subprotocol-owned framing |

Streams 0x02 and 0x04 are raw TCP relay tunnels. They carry llama.cpp RPC and HTTP traffic respectively and are not subject to protobuf framing or generation validation.

## Framing

All protobuf control-plane streams use the same framing:

```
[1 byte stream type][4 bytes LE length][N bytes protobuf body]
```

Maximum frame size: 8 MiB (`MAX_CONTROL_FRAME_BYTES`). Frames exceeding this
limit are rejected before allocation on read and before a prefix or body is
written on send.

Config and inventory mutation are exclusive to `mesh-llm-control/1`. The former mesh-plane config stream IDs `0x0b` and `0x0c` remain reserved for wire compatibility bookkeeping, but `mesh-llm/1` no longer dispatches protobuf request/response handlers for them. Skippy layer-package artifact transfer is not a mesh-owned schema. Gossip advertises `skippy-stage/2` feature support through `PeerAnnouncement.subprotocols`; mesh stream `0x0d` opens the advertised subprotocol, and the request/response schema plus artifact byte framing remain owned by `skippy-protocol`.

## Private owner-control (`mesh-llm-control/1`)

Owned-node commands use a separate QUIC ALPN and do not consume a public mesh
stream type. Each bidirectional owner-control stream carries 4-byte
little-endian length-prefixed `OwnerControlEnvelope` messages: a handshake,
then typed request/response envelopes. Current unary client calls (`get_config`,
`apply_config`, and `refresh_inventory`) open one authenticated stream per
command and expect one response before closing. `watch_config` sends one
request, then receives either an initial `accepted` response
(`include_snapshot=false`) or `snapshot` response (`include_snapshot=true`),
followed by zero or more `update` responses until either side closes the stream.
The same 8 MiB inbound and outbound limit applies.

The client must receive an explicit owner-control endpoint token out of band.
Control endpoints are never derived from peer IDs, gossip, Nostr, route tables,
or `/api/status`. QUIC authenticates the explicitly pinned target endpoint;
the handshake contains signed ownership bound to the requester's live QUIC
endpoint identity. The server verifies same-owner authorization before decoding
and dispatching a command, then verifies the common requester and target node
IDs for every typed request.

The request and response oneofs are the owned-node command registry. Current
request tags are:

| Tag | Protobuf operation | Execution shape |
|---:|---|---|
| 2 | `get_config` | unary |
| 3 | `watch_config` | accepted stream |
| 4 | `apply_config` | unary |
| 5 | `refresh_inventory` | unary scan |
| 6 | `load_model` | unary (session intent) |
| 7 | `unload_model` | unary (session intent) |
| 8 | `ensure_model` | unary (session intent) |
| 9 | `drain_model` | unary (session intent) |

`scan-refresh` is the operator-facing CLI/API name for the existing wire
operation `refresh_inventory = 5`; tag 5 is not renamed or renumbered. The
server converts the oneof into an exhaustive typed `OwnedNodeCommand` and uses
one authenticated dispatcher for requester binding, target binding, request
ID, execution shape, deadline policy, and bounded response handling. There are
no opaque command names or JSON payloads on this ALPN.

### Inventory refresh response compatibility

`OwnerControlRefreshInventoryResponse.snapshot = 1` remains the legacy result.
Current servers add `inventory = 2`, containing:

```proto
message OwnerControlRefreshInventory {
  repeated OwnerControlInventoryEntry entries = 1;
  OwnerControlRefreshInventoryDisposition disposition = 2;
}

message OwnerControlInventoryEntry {
  string canonical_model_ref = 1;
  optional string display_name = 2;
  uint64 total_size_bytes = 3;
  CompactModelMetadata metadata = 4;
}
```

Entries are strictly sorted by `canonical_model_ref`. Metadata is optional and
retains the compact GGUF-derived fields needed by future private management
clients. `EXECUTED` means the caller led the scan; `COALESCED` means it joined
an in-flight scan. All successful joiners receive the same inventory snapshot.
The successful scan snapshot is also the sole source used to update the node's
available-model projection; there is no second disk scan.

Old clients ignore tag 2 and continue reading tag 1. New clients accept an old
server's snapshot-only response as a successful, compatibility-limited result
without inventing inventory or disposition data. Released peers are not
assumed to understand the new response field. A failed scan reports one
structured error to all waiters and preserves the last good runtime snapshot
and advertised availability.

Rich inventory entries are private command output. They are not persisted as a
raw response in `PeerInfo`, public gossip, runtime status, or `/api/status`, and
endpoint tokens are not advertised. Existing public model-availability fields
remain a separate projection of a successful local scan.

### Bounds and deadlines

- Client connect: 8 seconds; stream open: 2 seconds; handshake write: 2
  seconds; request write: 2 seconds.
- Client get/apply unary response: 5 seconds; client inventory response: 35
  seconds, covering the server's 30-second scan deadline plus margin; watch
  acceptance: 5 seconds. An accepted watch has no unary response deadline.
- Client lifecycle-command unary response: 5 seconds.
- Server handshake read: 2 seconds; request read: 5 seconds.
- At most 32 owner-control stream workers are active per connection. The
  permit is acquired before spawning request work.
- Request IDs never use zero, including after `u64` wraparound.
- A response that would exceed 8 MiB is replaced with the existing
  `CONTROL_UNAVAILABLE` error before transport write.

### Model lifecycle commands (tags 6-9)

`load_model` (tag 6), `unload_model` (tag 7), `ensure_model` (tag 8), and
`drain_model` (tag 9) are shipped additive owner-control commands on
`mesh-llm-control/1`. They create session-only desired intents on the target
node's reconciler and never mutate durable config.

- **`load_model`**: one-shot present intent. Accepts one canonical model
  reference. Returns accepted/current lifecycle state within the five-second
  lifecycle-command unary deadline.
- **`ensure_model`**: maintained present intent with bounded retry. Accepts one
  canonical model reference. Survives transient load failures for the session.
- **`unload_model`**: absent intent. Accepts one canonical model reference or
  instance ID.
- **`drain_model`**: draining-then-absent intent. Accepts one canonical model
  reference or instance ID. Already-admitted work finishes; new work is
  rejected. Unloads at zero in-flight or force-cancels at the configured drain
  deadline (default 30s, capped by `drain_timeout_max_secs`).

Responses return request/intent ID, accepted/current lifecycle state, resolved
model/instance when known, and bounded typed errors. They acknowledge queueing
within the lifecycle-command unary deadline and never wait for a model load to
complete.

Old peers that do not implement these commands return `UNKNOWN_COMMAND`, which
new clients translate to typed `CONTROL_UNSUPPORTED`. There is no silent
fallback to the public mesh.

### Adding future owned-node commands

Start/stop inference, configuration additions beyond the current typed
get/watch/apply operations, and long-running operation queries are design
targets only. Those future operations are **not shipped** by the owner-control
command surface described here beyond the model lifecycle commands above.

Every future command addition must include all of the following:

1. Additive typed protobuf request and response oneof variants with new tags;
   never repurpose an existing field or add an opaque command/payload registry.
2. A new exhaustive typed decoder/dispatcher arm and a small command executor
   under `mesh/owner_control/commands/`.
3. Shared requester identity, single target identity, non-zero request ID,
   same-owner authorization, execution shape, deadline, frame-limit, and
   structured-error policy.
4. Defined idempotency behavior. Long-running or mutating operations must also
   define progress, cancellation, disconnect, timeout, and process-restart
   semantics before implementation.
5. Explicit old-peer behavior, normally a structured unsupported-command
   result; never a silent public-mesh fallback or compatibility claim for an
   older binary.
6. Typed client support plus loopback-only REST and CLI facades, including any
   required compatibility alias.
7. A privacy review covering logs, public gossip, runtime status, endpoint
   tokens, command inputs, and command results.
8. Unit/compatibility tests and two-node evidence for success, wrong-owner and
   wrong-target rejection, bounds/deadlines, failure preservation, and public
   mixed-version join/routing coexistence.

## Protocol Generation

`NODE_PROTOCOL_GENERATION = 1`

Every protobuf message that carries a `gen` field must have `gen == 1`. Frames with any other value are rejected with a `BadGeneration` error. This applies to:

- `GossipFrame.gen`
- `RouteTableRequest.gen`
- `RouteTable.gen`
- `PeerDown.gen`
- `PeerLeaving.gen`

## Admission (Quarantine-Until-Gossip)

A newly connected peer is quarantined until it sends a valid `GossipFrame` with `gen = 1`. Until admission:

- Only stream 0x01 (GOSSIP) and 0x05 (ROUTE_REQUEST) are accepted.
- All other streams are rejected and the stream is closed.
- The QUIC connection itself stays open so gossip can complete.

A peer is admitted when its `GossipFrame` decodes successfully and passes validation checks.

## Stream 0x01 — Gossip (`GossipFrame`)

Carries peer announcements. Both sides send a `GossipFrame` and read the other's frame.

```proto
message GossipFrame {
  uint32 gen = 1;                      // must equal NODE_PROTOCOL_GENERATION (1)
  repeated PeerAnnouncement peers = 2; // all known peers including self
  bytes sender_id = 3;                 // exactly 32 bytes; must match QUIC peer identity
}
```

Validation:
1. `gen == 1` — rejects legacy or future frames
2. `sender_id.len() == 32` — structural check
3. `sender_id == QUIC TLS peer identity` — anti-spoofing
4. Per peer: `endpoint_id.len() == 32`; HOST role requires `http_port` present

### PeerAnnouncement

Each `PeerAnnouncement` describes one node's state. Fields:

| Field | Description |
|-------|-------------|
| `endpoint_id` | 32-byte Ed25519 public key (node identity) |
| `role` | `WORKER`, `HOST`, or `CLIENT` |
| `http_port` | Required when role is HOST |
| `version` | Software version string |
| `gpu_name` | Comma-separated GPU model names when host enumeration is enabled |
| `hostname` | Hostname of the node |
| `is_soc` | `true` if running on a system-on-chip (e.g. Apple Silicon) |
| `gpu_vram` | Comma-separated per-GPU VRAM values in bytes |
| `gpu_reserved_bytes` | Comma-separated per-GPU reserved bytes when the platform reports a true reserved/unavailable metric |
| `gpu_mem_bandwidth_gbps` | Comma-separated per-GPU memory-bandwidth values in GB/s (gigabytes/sec) when known; the field name is retained for wire compatibility |
| `gpu_compute_tflops_fp32` | Comma-separated per-GPU FP32 compute-throughput hints when known |
| `gpu_compute_tflops_fp16` | Comma-separated per-GPU FP16 compute-throughput hints when known |
| `vram_bytes` | Advertised accelerator capacity in bytes: the placement budget peers size against |
| `hardware.memory` | Itemized view of `vram_bytes` (`MemoryInfo`, tag 4 of `HardwareInfo`, additive): `total_bytes`, `reserved_bytes`, `platform_reserve_bytes`, `configured_reserve_bytes`, `usable_bytes`, plus the informational `system_ram_bytes` and `ram_offload_bytes`; sent only when host enumeration is enabled |
| `model_source` | Source identifier for the model (e.g. HuggingFace repo) |
| `primary_serving` | Primary model being served; backward-compat alias for `serving` |
| `serving_models` | Models currently being served |
| `available_models` | Models on disk, available to serve |
| `catalog_models` | This node's contribution to the mesh model catalog |
| `mesh_id` | Stable mesh identity (self entry only) |
| `requested_models` | Models this node has requested to load |
| `experts_summary` | legacy expert usage summary (`ExpertsSummary`; self entry only) |
| `rtt_ms` | Round-trip time to the reporting node in milliseconds |
| `demand` | Per-model demand entries (self entry only) |
| `available_model_metadata` | GGUF-derived metadata for each available model |
| `available_model_sizes` | File sizes in bytes per model name |
| `serialized_addr` | JSON-serialized `EndpointAddr` for peer discovery |
| `admission` | Optional coarse inference admission state (tag 49, additive) |

These GPU telemetry fields are additive and optional. Older peers continue to interoperate by ignoring unknown `/1` protobuf fields, and the richer hardware reporting does not replace the existing model-metadata flow. For the shipped Skippy-enabled binary, GPU telemetry represents devices the embedded backend reports as runtime-selectable; platform probes are not a fallback source for advertised GPU count or usable capacity when Skippy reports no backend GPU. For clarity, `gpu_mem_bandwidth_gbps` values are serialized in GB/s (gigabytes/sec), matching benchmark output and CLI formatting; only the field name still carries the older `gbps` suffix for backward compatibility. ROCm `rocm-smi --showmeminfo` and Intel `xpu-smi` discovery expose used-memory counters rather than a true reserved/system-memory value, so `gpu_reserved_bytes` is intentionally omitted for those backends.

`hardware.memory` itemizes the capacity behind `vram_bytes`. `total_bytes` is the enumerated accelerator memory (the sum of device VRAM, or the unified working set on SoCs), `reserved_bytes` the driver/runtime reserve when the platform reports a true value, `platform_reserve_bytes` what the platform keeps back by policy on unified-memory hosts (the Tegra collector budgets 90% of physical RAM; zero for discrete GPUs, and zero on Metal while the survey reports the working set as the device memory), `configured_reserve_bytes` what the node owner withholds (the effective `defaults.hardware.safety_margin_gb`, plus whatever a `max_vram_gb` cap leaves out), and `usable_bytes` the remainder, so that `total_bytes = reserved_bytes + platform_reserve_bytes + configured_reserve_bytes + usable_bytes`. `system_ram_bytes` and `ram_offload_bytes` describe the node's local fit budget (its total system RAM, and the share of that budget backed by RAM) and are never counted as accelerator capacity. `vram_bytes` remains the placement budget: the block explains it rather than replacing it, and peers that predate it ignore the field.

### Admission advertisement (tag 49)

The optional `admission` field (protobuf tag 49 on `PeerAnnouncement`) carries
a coarse inference admission enum so peers can route around nodes that are
temporarily not accepting work. The field is additive; old peers that do not
recognize it treat its absence as `UNSPECIFIED` and continue to use legacy
eligibility.

Enum values:

| Value | Meaning |
|-------|---------|
| `UNSPECIFIED` | Legacy/unknown; treated as eligible by old peers |
| `ACCEPTING` | Node is accepting new inference work |
| `ACCEPTING_DEPRIORITIZED` | Node is accepting but deprioritized |
| `REMOTE_PAUSED` | Remote/mesh inference is paused; local work continues |
| `ALL_PAUSED` | All inference admission is paused; management/owner-control remain reachable |

Advertisement modes (configured via `[runtime.activity].advertisement`):

- `none`: emit nothing. Old peers see no change.
- `availability_only`: publish hosted/serving availability as
  explicitly-known-and-empty while non-admitting. Does not emit the enum.
- `coarse_state` (default): emit the admission enum and also publish
  known-empty availability for old peers that do not recognize tag 49.
- `private_coarse_state`: emit the enum only on private meshes; use
  known-empty availability publicly.

**Privacy**: the admission field encodes only the coarse state above. No raw
activity data, input events, app/window names, usernames, idle durations,
timestamps, or detector errors are ever encoded in gossip.

#### ExpertsSummary

```proto
message ExpertsSummary {
  uint32 total_experts = 1;
  uint32 expert_count_used = 2;
  repeated uint32 top_expert_ids = 3;
}
```

#### ModelDemandEntry

```proto
message ModelDemandEntry {
  string model_name = 1;
  uint64 last_active = 2;
  uint32 request_count = 3;
}
```

### GGUF Metadata in Gossip

Model metadata derived from GGUF headers is transported via `CompactModelMetadata` in the `available_model_metadata` field of each `PeerAnnouncement`. This lets peers learn model capabilities without downloading the file.

```proto
message CompactModelMetadata {
  string model_key = 1;
  string architecture = 10;          // e.g. "llama", "qwen2", "glm"
  string quantization_type = 18;     // e.g. "Q4_K_M", "IQ4_XS", "F16"
  string tokenizer_model_name = 11;
  repeated SpecialToken special_tokens = 12;
  float rope_scale = 13;
  float rope_freq_base = 14;
  bool is_moe = 15;
  uint32 expert_count = 16;
  uint32 used_expert_count = 17;
  // ... context_length, vocab_size, embedding_size, head_count, layer_count, etc.
}
```

Fields covered: architecture, quantization type, tokenizer, special tokens, RoPE parameters, expert counts, and standard transformer dimensions.

#### SpecialToken

```proto
message SpecialToken {
  string name = 1;
  int32 token_id = 2;
}
```

## Stream 0x03 — Tunnel Map (`TunnelMap`)

Sent after admission. Maps peer identities to local tunnel ports for B2B direct transfers.

```proto
message TunnelMap {
  bytes owner_peer_id = 1;       // exactly 32 bytes; must match QUIC sender identity
  repeated TunnelEntry entries = 2;
}

message TunnelEntry {
  bytes target_peer_id = 1;      // exactly 32 bytes
  optional bytes relay_peer_id = 2;
  uint32 tunnel_port = 3;        // must be in range [1, 65535]
}
```

`owner_peer_id` must match the QUIC connection identity. Frames with a mismatched owner are rejected.

## Stream 0x05 — Route Table (`RouteTableRequest` / `RouteTable`)

Used by passive clients and standby nodes to learn the current routing table without full gossip participation.

**Request:**
```proto
message RouteTableRequest {
  bytes requester_id = 1;  // 0 or exactly 32 bytes
  uint32 gen = 2;          // must equal NODE_PROTOCOL_GENERATION (1)
}
```

**Response:**
```proto
message RouteTable {
  repeated RouteEntry entries = 1;
  optional string mesh_id = 2;  // passive callers learn mesh identity here
  uint32 gen = 3;               // must equal NODE_PROTOCOL_GENERATION (1)
}

message RouteEntry {
  bytes endpoint_id = 1;  // exactly 32 bytes
  string model = 2;       // model being served (empty if not serving)
}
```

Serving a route table does not admit the requester. The requester is never added to `state.peers`.

## Stream 0x06 — Peer Down (`PeerDown`)

Broadcast when a node detects that another peer is unreachable. Requires reachability confirmation before the dead peer is removed from state.

```proto
message PeerDown {
  bytes peer_id = 1;  // exactly 32 bytes; the peer being reported as unreachable
  uint32 gen = 2;     // must equal NODE_PROTOCOL_GENERATION (1)
}
```

A node never broadcasts `PeerDown` for itself. The receiver confirms reachability (3s timeout) before acting on the report.

## Stream 0x07 — Peer Leaving (`PeerLeaving`)

Sent on clean shutdown (ctrl-c). Only removes the sender from peer state — not any other peer.

```proto
message PeerLeaving {
  bytes peer_id = 1;  // exactly 32 bytes; must match the QUIC sender identity
  uint32 gen = 2;     // must equal NODE_PROTOCOL_GENERATION (1)
}
```

`peer_id` must match the QUIC connection identity. Forged `PeerLeaving` frames (where `peer_id` names a different node) are rejected without any state change.

## Skippy Stage Artifact Transfer

Used by a Skippy worker to fetch missing Hugging Face layer-package artifacts from the coordinating node before falling back to normal local/HF package resolution. Mesh gossip only advertises support:

```proto
message MeshSubprotocol {
  string name = 1;              // "skippy-stage"
  uint32 major = 2;             // 2
  repeated string features = 3; // includes "artifact-transfer"
}

message MeshSubprotocolOpen {
  uint32 gen = 1;
  string name = 2;              // "skippy-stage"
  uint32 major = 3;             // 2
}
```

Outbound transfer uses mesh stream `0x0d` (`STREAM_SUBPROTOCOL`), followed by a length-prefixed `MeshSubprotocolOpen { name: "skippy-stage", major: 2 }`, the Skippy-owned stream kind `0x03`, a length-prefixed `StageArtifactTransferRequest`, a length-prefixed `StageArtifactTransferResponse`, and raw artifact bytes when accepted. Generation 7 requires this mesh-subprotocol path for control and artifacts; the dedicated stage ALPN accepts activation transport only.

**Request:**
```proto
message StageArtifactTransferRequest {
  uint32 gen = 1;
  bytes requester_id = 2;
  string topology_id = 3;
  string run_id = 4;
  string stage_id = 5;
  string package_ref = 6;
  string manifest_sha256 = 7;
  string relative_path = 8;
  uint64 offset = 9;
  optional uint64 expected_size = 10;
  optional string expected_sha256 = 11;
}
```

**Response:**
```proto
message StageArtifactTransferResponse {
  uint32 gen = 1;
  bool accepted = 2;
  uint64 total_size = 3;
  optional string sha256 = 4;
  optional string error = 5;
}
```

Privacy and safety properties:

- The capability is advertised as a boolean only; nodes do not gossip package inventories, file names, local paths, or cache contents.
- Requests are admitted-peer only and the `requester_id` must match the QUIC peer identity.
- Only `hf://namespace/repo@revision` package refs are served.
- Non-manifest artifact paths must be declared by the cached `model-package.json`, with matching manifest SHA, artifact size, and artifact SHA.
- Absolute paths, parent-directory traversal, and symlink escapes outside the managed Hugging Face repo cache are rejected.
- Transfers are streamed in bounded chunks; the protocol carries an offset, and
  the current client installs only freshly verified complete artifacts.
- Body reads use an idle timeout so a stalled peer cannot hang stage
  preparation indefinitely.

## Out-of-Scope Streams

The following are explicitly NOT protobuf and are not described here:

- **0x02 / 0x04** — raw TCP relay for llama.cpp RPC and HTTP. No framing changes.
- **Nostr discovery payloads** — remain JSON (NIP-89 kind 31990).
- **Plugin streams (0x08 / 0x09)** — PLUGIN_CHANNEL and PLUGIN_BULK_TRANSFER; separate protocol, unchanged.
- **Invite/join token encoding** — unchanged.

## Compatibility

`mesh-llm/1` is the only supported control-plane protocol.

- Nodes advertise `mesh-llm/1` on accept.
- Scoped `mesh-llm/1` control-plane streams (0x01, 0x03, 0x05, 0x06, 0x07, 0x0b, 0x0c, 0x0d) use protobuf framing for their mesh-owned open/control frames.
- Skippy artifact transfer uses `STREAM_SUBPROTOCOL` (0x0d) to open `skippy-stage/2`, then Skippy-owned stream type `0x03`; only its request/response metadata is protobuf-framed, followed by raw artifact bytes when accepted.
