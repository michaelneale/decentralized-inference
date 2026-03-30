# mesh-llm Message Protocol

This document describes the wire protocol for control-plane communication between mesh-llm nodes. Control-plane traffic prefers the `meshllm.node.v1` protobuf schema on QUIC ALPN `mesh-llm/1`, with backward-compatible support for the legacy `mesh-llm/0` JSON/raw payloads.

## ALPN

Control-plane connections prefer ALPN `mesh-llm/1`.

Peers may also negotiate the legacy ALPN `mesh-llm/0` for backward compatibility. Mixed meshes containing both `/0` and `/1` nodes are supported, but `/0` links only exchange the legacy field set.

## Stream Types

Each QUIC connection carries multiple logical streams, distinguished by a 1-byte prefix:

| Byte | Name | Direction | Format |
|------|------|-----------|--------|
| 0x01 | GOSSIP | bidirectional | protobuf `GossipFrame` |
| 0x02 | TUNNEL_RPC | bidirectional | raw TCP relay (not protobuf) |
| 0x03 | TUNNEL_MAP | send | protobuf `TunnelMap` |
| 0x04 | TUNNEL_HTTP | bidirectional | raw TCP relay (not protobuf) |
| 0x05 | ROUTE_REQUEST | bidirectional | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | send | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | send | protobuf `PeerLeaving` |

Streams 0x02 and 0x04 are raw TCP relay tunnels. They carry llama.cpp RPC and HTTP traffic respectively and are not subject to protobuf framing or generation validation.

## Framing

All protobuf control-plane streams (0x01, 0x03, 0x05, 0x06, 0x07) use the same framing:

```
[1 byte stream type][4 bytes LE length][N bytes protobuf body]
```

Maximum frame size: 8 MB (`MAX_CONTROL_FRAME_BYTES`). Frames exceeding this limit are rejected.

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
- All other streams (0x02, 0x03, 0x04, 0x06, 0x07) are rejected and the stream is closed.
- The QUIC connection itself stays open so gossip can complete.

A peer is admitted when its negotiated gossip payload decodes successfully and passes validation checks. On `/1` this is a protobuf `GossipFrame`; on `/0` this is the legacy JSON gossip payload.

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

Each `PeerAnnouncement` describes one node's state. Key fields:

| Field | Description |
|-------|-------------|
| `endpoint_id` | 32-byte Ed25519 public key (node identity) |
| `role` | `WORKER`, `HOST`, or `CLIENT` |
| `http_port` | Required when role is HOST |
| `vram_bytes` | Total GPU VRAM in bytes |
| `serving_models` | Models currently being served |
| `available_models` | Models on disk, available to serve |
| `catalog_models` | This node's contribution to the mesh model catalog |
| `mesh_id` | Stable mesh identity (self entry only) |
| `demand` | Per-model demand entries (self entry only) |
| `available_model_metadata` | GGUF-derived metadata for each available model |
| `available_model_sizes` | File sizes in bytes per model name |
| `serialized_addr` | JSON-serialized `EndpointAddr` for peer discovery |

### GGUF Metadata in Gossip

Model metadata derived from GGUF headers is transported via `CompactModelMetadata` in the `available_model_metadata` field of each `PeerAnnouncement`. This lets peers learn model capabilities without downloading the file.

```proto
message CompactModelMetadata {
  string model_key = 1;
  string architecture = 10;          // e.g. "llama", "qwen2", "glm"
  string quantization_type = 18;     // e.g. "Q4_K_M", "IQ4_XS", "F16"
  string tokenizer_model_name = 11;
  float rope_scale = 13;
  float rope_freq_base = 14;
  bool is_moe = 15;
  uint32 expert_count = 16;
  uint32 used_expert_count = 17;
  // ... context_length, vocab_size, embedding_size, head_count, layer_count, etc.
}
```

Fields covered: architecture, quantization type, tokenizer, RoPE parameters, expert counts (for MoE models), and standard transformer dimensions.

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

## Out-of-Scope Streams

The following are explicitly NOT protobuf and are not described here:

- **0x02 / 0x04** — raw TCP relay for llama.cpp RPC and HTTP. No framing changes.
- **Nostr discovery payloads** — remain JSON (NIP-89 kind 31990).
- **Plugin streams** — separate protocol, unchanged.
- **Invite/join token encoding** — unchanged.

## Compatibility

`mesh-llm/1` remains the preferred protocol, but negotiation keeps older nodes working:

- Nodes advertise both `mesh-llm/1` and `mesh-llm/0` on accept.
- Connectors prefer `/1` and offer `/0` as a fallback when needed.
- All five scoped control-plane streams (0x01, 0x03, 0x05, 0x06, 0x07) use protobuf framing on `/1` and the legacy JSON/raw formats on `/0`.
