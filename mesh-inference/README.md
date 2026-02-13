# mesh-inference

P2P mesh for distributed llama.cpp inference over QUIC. Handles peer discovery, NAT traversal, and TCP↔QUIC tunneling so llama.cpp RPC traffic can cross firewalls and NATs without manual port forwarding.

## How It Works

Same binary on every node. Each node:
1. Starts a local `rpc-server` with `--gguf` for zero-transfer weight loading
2. Joins the QUIC mesh (iroh) and discovers peers via gossip
3. Creates local TCP tunnel ports that bridge to remote peers' rpc-servers
4. Optionally runs `llama-server` as orchestrator with `--rpc` pointing at tunnel ports

## Usage

### Worker (offers GPU compute to the mesh)
```bash
mesh-inference --model ~/.models/model.gguf --bin-dir /path/to/llama.cpp/build/bin
```
Prints an invite token. Other nodes use this to join.

### Orchestrator (runs llama-server, joins an existing mesh)
```bash
mesh-inference --model ~/.models/model.gguf --bin-dir /path/to/llama.cpp/build/bin \
  --serve 8090 --join <invite-token>
```

### Options
```
--model PATH         GGUF model file (both worker and orchestrator need this)
--serve PORT         Run llama-server on this HTTP port
--join TOKEN         Join mesh via invite token (can be repeated)
--min-peers N        Wait for N peers before starting llama-server (default: 1)
--tensor-split R,R   Layer distribution ratios (e.g. "0.85,0.15")
--bin-dir PATH       Directory with rpc-server + llama-server binaries
--device DEV         GPU device for rpc-server (default: MTL0 on macOS)
```

## Configurations

### Big machine as orchestrator
The machine with the most VRAM runs `--serve`. llama-server uses its local GPU directly for most layers, remote workers contribute extra VRAM.

```bash
# Machine A (55GB VRAM) — orchestrator
mesh-inference --model model.gguf --serve 8090 --join <token-from-B>

# Machine B (12GB VRAM) — worker, starts first
mesh-inference --model model.gguf
```

### Small machine as orchestrator
When the model doesn't fit on one machine, the smaller machine can orchestrate. Use `--tensor-split` to limit local GPU allocation.

```bash
# Machine B (12GB VRAM) — orchestrator, starts first
mesh-inference --model model.gguf --serve 8090 --tensor-split 0.85,0.15

# Machine A (55GB VRAM) — worker, joins
mesh-inference --model model.gguf --join <token-from-B>
```

`--tensor-split 0.85,0.15` means 85% of layers go to the remote worker, 15% stay local. The orchestrator automatically uses `--no-mmap` to prevent memory-mapping the full model file.

## Firewall Notes

- The machine that can accept incoming UDP should start first and print the invite token
- The firewalled machine should be the joiner (`--join`)
- macOS stealth mode drops incoming UDP silently — that machine must always join, never listen
- iroh provides relay fallback if direct UDP fails, but relay is rate-limited

## Building

```bash
cargo build --release
```

Requires `rpc-server` and `llama-server` from the patched llama.cpp fork:
```bash
git clone -b rpc-local-gguf https://github.com/michaelneale/llama.cpp.git
cmake -B build -S llama.cpp -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build --config Release
```

## Deploying to Another Machine

```bash
just bundle    # creates /tmp/mesh-bundle.tar.gz with all binaries + dylibs
scp /tmp/mesh-bundle.tar.gz user@remote:/tmp/
ssh user@remote "cd /tmp && tar xzf mesh-bundle.tar.gz"
# Then run: /tmp/mesh-bundle/mesh-inference --model ...
```
