# Plugins

This document is the reference for the `mesh-llm` plugin system.

It is written for two audiences:

- Mesh users and operators who want to enable, disable, inspect, and use plugins.
- Plugin developers, whether human or agent, who want to build a plugin that `mesh-llm` can launch and talk to.

## What A Plugin Is

A plugin is a local executable process launched by `mesh-llm`.

`mesh-llm` owns:

- mesh transport
- peer routing
- local plugin process launch
- plugin IPC
- management API exposure
- MCP exposure

A plugin owns:

- its own feature logic
- its tool implementations
- any plugin-local state
- any plugin-specific `ChannelMessage` payloads

`blackboard` is the reference plugin. It is auto-registered by the host and launched as:

```text
mesh-llm --plugin blackboard
```

## For Mesh Users

### Default Behavior

`blackboard` is enabled by default.

You do not install it separately. `mesh-llm` auto-registers it unless you disable it in config.

### Config File

Default config path:

```text
~/.mesh-llm/config.toml
```

Override for development:

```text
MESH_LLM_CONFIG=/some/other/path/config.toml
```

### Config Shape

Plugins are configured explicitly.

Example:

```toml
[[plugin]]
name = "blackboard"
enabled = false

[[plugin]]
name = "notes"
command = "/Users/jdumay/.mesh-llm/plugin/notes-plugin"
args = ["--workspace", "default"]
config = { workspace = "default" }
enabled = true
```

Field meanings:

- `name`: expected runtime plugin identity.
- `enabled`: whether the plugin is active on this node. Omit to use the default.
- `command`: executable to launch for a custom plugin.
- `args`: startup arguments passed to that executable.
- `config`: plugin-specific config object. This is serialized to JSON and sent to the plugin in `InitializeRequest`.

### Runnable Inference Plugin Example

The built-in `inference` plugin is opt-in. It is not enabled by default.

This proof-of-concept hosts its own HTTP listeners from inside the plugin process and exposes the startup context the host passed in.

Example config:

```toml
[[plugin]]
name = "inference"
enabled = true
config = { api_port = 19400, console_port = 19401, bind_address = "127.0.0.1" }
```

You can copy this directly from:

```text
examples/inference-plugin.toml
```

Run:

```text
MESH_LLM_CONFIG=examples/inference-plugin.toml mesh-llm
```

Inspect:

```text
curl http://127.0.0.1:19400/startup-context
open http://127.0.0.1:19401/
```

This uses plugin-owned listeners so it does not rely on the main process serving those HTTP endpoints.

### Built-In Default Plugins

`blackboard` is a host-provided default plugin spec.

To disable it:

```toml
[[plugin]]
name = "blackboard"
enabled = false
```

`blackboard` may not override `command` or `args` in config because it is served by the main `mesh-llm` binary.

### Suggested Plugin Binary Location

External plugin binaries can live anywhere, but the suggested location is:

```text
~/.mesh-llm/plugin/
```

Examples:

```text
~/.mesh-llm/plugin/notes
~/.mesh-llm/plugin/tickets
```

This is only a convenience location. It is not an auto-discovery mechanism.

### Listing Plugins

To see the resolved plugin specs for the current node:

```text
mesh-llm plugin list
```

### Management API

Running plugins are exposed through the management API.

Endpoints:

- `GET /api/plugins`
- `GET /api/plugins/<name>/tools`
- `POST /api/plugins/<name>/tools/<tool-name>`

The POST body is passed through as the plugin tool's `arguments_json`.

### MCP Exposure

`mesh-llm` can also expose plugin tools over MCP.

Current blackboard entry point:

```text
mesh-llm --client --join <token> blackboard --mcp
```

Protocol boundary:

- MCP client to `mesh-llm`: JSON-RPC
- `mesh-llm` to plugin process: framed protobuf over local IPC
- mesh peer to mesh peer plugin traffic: framed protobuf over QUIC streams

Important rule:

- MCP is JSON-RPC only at the outer edge.
- Inside the system, plugin traffic stays on the protobuf plugin protocol.
- The MCP adapter does not proxy through `/api/plugins`.

### Crash Behavior Today

Today, plugin crash handling is basic.

If a plugin process exits or disconnects:

- in-flight requests fail
- future tool calls fail
- the plugin is not automatically restarted

In practice, that means a crashed plugin stays dead until `mesh-llm` is restarted.

## For Plugin Developers

### Mental Model

A plugin is a named executable that speaks the `mesh-llm` plugin protocol over a local IPC stream.

The plugin executable does not join the mesh directly. `mesh-llm` remains the mesh node and forwards plugin channel traffic on the plugin's behalf.

### Runtime Identity

Config is authoritative for what should be launched.

Handshake is authoritative for what was actually launched.

Startup rule:

1. `mesh-llm` resolves enabled plugins from config plus host defaults.
2. `mesh-llm` launches the configured executable.
3. `mesh-llm` sends `InitializeRequest`.
4. The plugin returns `InitializeResponse`.
5. `InitializeResponse.plugin_id` must match the configured `name`.
6. If it does not match, the host rejects the plugin.

### Compatibility Policy

When changing the plugin protocol, always consider compatibility.

- If a change may be breaking, confirm whether that break is intended.
- If a change is not intended to be breaking, the previous protocol version must continue to be supported.
- Do not silently strand older hosts or plugins.

This applies both to the protobuf schema and to the semantic meaning of existing messages and fields.

### Launch Environment

The host launches the plugin and sets:

- `MESH_LLM_PLUGIN_ENDPOINT`
- `MESH_LLM_PLUGIN_TRANSPORT`
- `MESH_LLM_PLUGIN_NAME`

The plugin connects back to the host using those values.

### Local IPC Transport

The transport depends on the operating system.

macOS and Linux:

- Unix domain sockets
- recommended socket path: `~/.mesh-llm/run/plugins/<plugin-name>.sock`

Windows:

- named pipes
- recommended pipe name: `\\.\pipe\mesh-llm-<plugin-name>`

The transport changes by OS, but the framing and protobuf schema stay the same.

### Framing

Every message is sent as:

1. `u32` little-endian frame length
2. protobuf payload bytes

This is intentionally simpler than gRPC for local subprocess IPC.

### Wire Schema

The canonical schema lives in:

- [`mesh-llm/proto/plugin.proto`](/Users/jdumay/.codex/worktrees/16af/decentralized-inference/mesh-llm/proto/plugin.proto)

Rust code generation in this repo uses `prost`.

Core message families:

- initialize
- health
- shutdown
- MCP RPC requests
- MCP RPC responses
- MCP RPC notifications
- channel messages
- bulk transfer
- mesh events
- structured errors

### Handshake

During initialization:

- host sends `InitializeRequest`
- plugin returns `InitializeResponse`

The plugin must provide:

- `plugin_id`
- `plugin_protocol_version`
- `plugin_version`
- `server_info_json`
- `capabilities`

The host currently requires protocol version equality.

### Mesh Events

The host can send `MeshEvent` envelopes into every running plugin.

Current event kinds:

- `PEER_UP`
- `PEER_DOWN`
- `PEER_UPDATED`
- `LOCAL_ACCEPTING`
- `LOCAL_STANDBY`
- `MESH_ID_UPDATED`

Peer events carry a `MeshPeer` snapshot. Local-state events populate the mesh-level fields and leave
`peer` empty.

The current `MeshPeer` payload includes:

- `peer_id`
- `version`
- `role`
- `vram_bytes`
- `models`
- `serving_models`
- `available_models`
- `requested_models`
- `rtt_ms`
- `model_source`

Every `MeshEvent` also carries:

- `local_peer_id`
- `mesh_id`
- `detail_json`

Plugins should treat these events as advisory runtime state from the local host.

On initial attach, the host now sends:

- one local readiness event: `LOCAL_ACCEPTING` or `LOCAL_STANDBY`
- `MESH_ID_UPDATED` when the node already knows its mesh id
- a `PEER_UP` snapshot for every known peer

At runtime, external plugins are supervised by the host. If the plugin disconnects, stops
responding, or fails health checks, the host marks it as restarting and attempts to relaunch it on
the next supervision round before redelivering control, bulk, or mesh events.

### MCP RPC

Plugins expose standard MCP server methods through generic RPC envelopes.

Common methods include:

- `tools/list`
- `tools/call`
- `prompts/list`
- `prompts/get`
- `resources/list`
- `resources/templates/list`
- `resources/read`
- `resources/subscribe`
- `resources/unsubscribe`
- `completion/complete`
- `logging/setLevel`
- task methods such as `tasks/list`, `tasks/get`, `tasks/result`, and `tasks/cancel`

Plugins can also send host-directed MCP client requests and notifications over the same envelope
surface, including:

- `roots/list`
- `sampling/createMessage`
- `elicitation/create`
- standard MCP notifications such as resource updates and list-changed events

The `server_info_json` field in `InitializeResponse` carries the canonical MCP capability
declaration for the plugin. The host uses that MCP server info for:

- management API discovery
- aggregated MCP server exposure

### Tool Calls

Tool execution is now standard MCP:

- `tools/list`
- `tools/call`

`arguments_json` is passed through as JSON text.

The plugin returns:

- `content_json`
- `is_error`

Guidance for plugin authors:

- keep tool inputs and outputs JSON-friendly
- return stable JSON shapes
- prefer explicit validation errors over ambiguous failures

### Channel Messages

Plugins can send and receive `ChannelMessage` values through the host.

This is the mechanism for plugin-defined mesh traffic.

Each `ChannelMessage` contains:

- `channel`
- `source_peer_id`
- `target_peer_id`
- `content_type`
- `body`
- `message_kind`
- `correlation_id`
- `metadata_json`

The host wraps these in `MeshChannelFrame` for mesh-wide forwarding.

### Bulk Transfer

Plugins can also send and receive `BulkTransferMessage` values through the host.

This is the generic path for larger payloads where ordinary control messages are
not a good fit.

Each `BulkTransferMessage` contains:

- `kind`
- `transfer_id`
- `channel`
- `source_peer_id`
- `target_peer_id`
- `content_type`
- `correlation_id`
- `metadata_json`
- `total_bytes`
- `offset`
- `body`
- `final_chunk`

The host wraps these in `MeshBulkFrame` for mesh-wide forwarding.

### Mesh-Wide Forwarding

The host reserves generic mesh streams for plugin traffic.

Every forwarded frame contains:

- `plugin_id`
- `message_id`
- `ChannelMessage`

For bulk transfer, every forwarded frame contains:

- `plugin_id`
- `message_id`
- `BulkTransferMessage`

Forwarding behavior:

- `message_id` is deduplicated at each node
- a message is delivered locally if `target_peer_id` is empty or matches the local peer
- a message is forwarded outward if it is a broadcast or is targeted at another peer

This is how the external blackboard plugin replicates across the mesh.

### Health And Shutdown

The protocol includes:

- `HealthRequest`
- `HealthResponse`
- `ShutdownRequest`
- `ShutdownResponse`

Current implementation note:

- these messages exist
- health supervision and automatic restart policy are not fully implemented yet

### Current Failure Semantics

Today, if the plugin crashes or disconnects:

- the host connection loop exits
- pending requests fail as disconnected
- later requests fail because the plugin is no longer accepting requests
- the host does not automatically restart the plugin

Plugin authors should assume:

- abrupt disconnects surface as request failures
- reconnect is not currently supported without restarting the host

### Why This Uses Protobuf Framing Instead Of gRPC

This system intentionally does not use full gRPC.

Reasons:

- plugins are local subprocesses, not network services
- HTTP/2 is unnecessary overhead here
- framed protobuf is enough for the needed RPC surface
- host supervision stays simpler

The stack is:

- protobuf schema
- `prost` codegen
- manual framed reads and writes

Not:

- `tonic`
- HTTP/2
- full gRPC transport

### Why This Does Not Use `dlopen`

Shared-library loading is not the preferred plugin model.

Reasons:

- weak crash isolation
- Rust ABI stability concerns
- harder compatibility and upgrade handling
- harder debugging and restart behavior

The subprocess model is safer and easier to evolve.

## Blackboard Reference Plugin

`blackboard` is the first real plugin using this system.

It is:

- auto-registered by the host
- launched as `mesh-llm --plugin blackboard`
- synchronized across the mesh through plugin `ChannelMessage` forwarding
- exposed through both the management API and MCP

The runtime reuses shared blackboard data types from:

- [`mesh-llm/src/plugins/blackboard/mod.rs`](/Users/jdumay/code/mesh-llm/mesh-llm/src/plugins/blackboard/mod.rs)

The generic MCP adapter for plugins is in:

- [`mesh-llm/src/plugin_mcp.rs`](/Users/jdumay/code/mesh-llm/mesh-llm/src/plugin_mcp.rs)

## Example Surface Plugin

The repo also includes a standalone example plugin executable under:

- [`mesh-llm/examples/plugin-surface`](/Users/jdumay/code/mesh-llm/mesh-llm/examples/plugin-surface)

It demonstrates:

- tool schemas and tool calls
- MCP exposure through the generic plugin MCP adapter
- targeted and broadcast `ChannelMessage` delivery
- targeted and broadcast `BulkTransferMessage` delivery
- inbound mesh event observation

It is useful for validating plugin behavior on a real mesh without inventing a new production feature first.

It is not built into `mesh-llm`, and it is not part of the normal `mesh-llm` distribution. To use it, compile the example executable separately and point a config entry at that binary.

## Implementation Checklist For New Plugins

If you are building a new plugin, the minimum checklist is:

1. Implement an executable that can be launched locally.
2. Read the host-provided environment variables.
3. Connect to the host over the provided local IPC transport.
4. Speak the framed protobuf protocol from [`plugin.proto`](/Users/jdumay/code/mesh-llm/mesh-llm/proto/plugin.proto).
5. Respond correctly to `InitializeRequest`.
6. Return stable tool schemas and tool outputs.
7. If using mesh traffic, define a stable `channel` name and message body format.
8. Decide whether any protocol changes you need are breaking or non-breaking before changing the shared protocol.

## Current Gaps

The current plugin system is functional, but not finished.

Known gaps:

1. Mesh events currently cover peer lifecycle snapshots, but broader topology and cluster-level signaling is still limited.
2. Restart policy and crash supervision are not implemented.
3. A standalone example plugin outside the main `mesh-llm` executable is still missing.
4. Cross-platform transport testing, especially Windows named pipes, needs more coverage.
5. Plugin protocol version negotiation is still stricter than the long-term design likely wants.
