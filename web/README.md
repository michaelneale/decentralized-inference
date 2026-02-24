# mesh-llm web console

Public read-only dashboard for the default mesh-llm mesh. Runs as a lightweight client — discovers the mesh via Nostr, proxies API requests to GPU nodes.

## What it does

- Joins the default "mesh-llm" mesh as a client (no GPU needed)
- Console on `:3131` — live topology, models, chat
- API on `:9337` — OpenAI-compatible, routes to mesh nodes via QUIC

## Run locally

```bash
# Native (easiest)
mesh-llm --client --auto

# Or on custom ports (to avoid conflict with a local mesh node)
mesh-llm --client --auto --port 9338 --console 3132
```

## Deploy to Fly.io

From the repo root:

```bash
fly launch --config web/fly.toml --dockerfile web/Dockerfile
fly deploy --config web/fly.toml --dockerfile web/Dockerfile
```

## Docker

```bash
docker build -f web/Dockerfile -t mesh-llm-web .
docker run -p 3131:3131 -p 9337:9337 mesh-llm-web
```

## Architecture

```
Browser → Fly (HTTPS) → mesh-llm --client → QUIC tunnel → GPU nodes
                         ├── :3131 console (read-only dashboard)
                         └── :9337 API (proxied to mesh)
```

The client has zero GPU cost. It maintains a QUIC connection to mesh nodes and tunnels HTTP requests. Topology shown in the console comes from gossip.
