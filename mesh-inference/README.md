# mesh-inference

P2P mesh for distributed llama.cpp inference over QUIC. Handles peer discovery, NAT traversal, TCP↔QUIC tunneling, and B2B direct worker-to-worker tensor transfers. Also provides a lite client mode for accessing the API from any machine without a GPU.

Same binary on every node. Three roles:

| Role | What runs | Flags |
|------|-----------|-------|
| **Worker** | rpc-server + mesh | `--model PATH` |
| **Host** | rpc-server + llama-server + mesh | `--model PATH --serve PORT` |
| **Client** | mesh only (no GPU, no model) | `--client --join TOKEN` |

## Usage

### Worker (offers GPU compute)
```bash
mesh-inference --model ~/.models/model.gguf --bin-dir /path/to/llama.cpp/build/bin
```
Prints an invite token. Other nodes use this to join.

### Host (orchestrator — runs llama-server)
```bash
mesh-inference --model ~/.models/model.gguf --bin-dir /path/to/llama.cpp/build/bin \
  --serve 8090 --join <invite-token> --tensor-split 0.85,0.15
```

### Lite client (no GPU — just proxies the API locally)
```bash
mesh-inference --client --join <invite-token> --port 8080
```

Exposes `http://localhost:8080` as a local OpenAI-compatible API. All requests tunnel through the QUIC mesh to the host's llama-server. SSE streaming works.

```bash
# Check it's working
curl http://localhost:8080/v1/models
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

The client discovers the host automatically via gossip — no need to specify which node is the host.

### All options
```
--model PATH         GGUF model file (worker + host only)
--serve PORT         Run llama-server on this HTTP port (makes this node a host)
--client             Run as lite client (no GPU, no model, no rpc-server)
--join TOKEN         Join mesh via invite token (repeatable)
--port PORT          Local HTTP port for --client mode (default: 8080)
--min-peers N        Wait for N peers before starting llama-server (default: 1)
--tensor-split R,R   Layer distribution ratios (e.g. "0.85,0.15")
--bin-dir PATH       Directory with rpc-server + llama-server binaries
--device DEV         GPU device for rpc-server (default: MTL0 on macOS)
```

## How it works

### Mesh formation
1. Node starts an iroh QUIC endpoint (Ed25519 keypair)
2. Joins a peer via `--join <token>` (or waits for inbound connections)
3. Peers exchange their known peer lists via gossip (includes node roles)
4. Full mesh forms — every node has a QUIC connection to every other node

### Tunneling
For each peer, the sidecar allocates a local TCP port. When llama.cpp connects to it, the sidecar opens a QUIC bi-stream to the peer and relays bytes bidirectionally. llama.cpp sees localhost TCP; the mesh handles everything else.

### B2B direct transfers
When llama-server splits the model across multiple workers, activation tensors need to flow between workers at each graph split boundary. Without B2B, these go: Worker 1 → orchestrator → Worker 2 (two network hops). With B2B:

1. Each node broadcasts its tunnel port map to all peers
2. The orchestrator's `REGISTER_PEER` commands are intercepted on the inbound tunnel
3. The sidecar rewrites the endpoint addresses to local tunnel ports
4. Workers push activations directly to each other through the mesh

### Lite client
The client joins the mesh, discovers a host via role-aware gossip, then accepts local TCP connections and tunnels each through a QUIC bi-stream to the host's llama-server HTTP port. Pure byte relay — no protocol awareness needed for HTTP or SSE.

## Firewall notes

- The machine that can accept incoming UDP should start first (print the token)
- The firewalled machine should join (`--join`)
- macOS stealth mode drops incoming UDP — that machine must always be the joiner
- iroh provides relay fallback if direct UDP fails, but relay is rate-limited

## Building

```bash
cargo build --release
```

Requires rpc-server and llama-server from the patched llama.cpp fork (not needed for `--client` mode):
```bash
git clone -b rpc-local-gguf https://github.com/michaelneale/llama.cpp.git
cmake -B build -S llama.cpp -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build --config Release
```

## Deploying to another machine

```bash
just bundle    # creates /tmp/mesh-bundle.tar.gz with all binaries + dylibs
scp /tmp/mesh-bundle.tar.gz user@remote:/tmp/
ssh user@remote "cd /tmp && tar xzf mesh-bundle.tar.gz"
# Then: /tmp/mesh-bundle/mesh-inference --model ... (or --client --join ...)
```

For `--client` mode, only the `mesh-inference` binary is needed — no llama.cpp binaries or model files.
