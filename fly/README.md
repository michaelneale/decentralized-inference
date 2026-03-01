# mesh-llm Fly.io Apps

Two Fly apps, one shared Dockerfile. Both run mesh-llm in `--client` mode — no GPU, just QUIC tunnels to mesh nodes.

| App | URL | Purpose | Fly config |
|---|---|---|---|
| **console** | [mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev) | Dashboard + chat + API | `fly/console/fly.toml` |
| **api** | mesh-llm-api.fly.dev | Public OpenAI-compatible API (rate-limited) | `fly/api/fly.toml` |

## Architecture

```
                              ┌─────────────────────────┐
Browser/curl ──HTTPS──→ Fly   │  mesh-llm --client      │
                              │  discovers mesh via      │──QUIC──→ GPU nodes
                              │  Nostr, tunnels requests │
                              └─────────────────────────┘
```

**Console** exposes `:3131` (dashboard, chat, topology) and `:9337` (API).
**API** exposes only `:9337` with Fly concurrency limits (soft: 3, hard: 5 in-flight requests).

## Deploy

All commands run from the **repo root**:

```bash
# Console (dashboard + API)
fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile

# API only (rate-limited)
fly launch --config fly/api/fly.toml --dockerfile fly/Dockerfile    # first time
fly deploy --config fly/api/fly.toml --dockerfile fly/Dockerfile    # updates
```

## Run locally

```bash
# Same as what the Fly apps run — no Docker needed
mesh-llm --client --auto
```

## Rate limiting (API app)

The API app uses [Fly Proxy concurrency limits](https://fly.io/docs/reference/concurrency/):

- **soft_limit: 3** — Fly starts routing new requests to other machines (if available)
- **hard_limit: 5** — Fly returns 503 for requests beyond this

These limits are per-machine. With `min_machines_running = 1`, the single machine handles up to 5 concurrent inference requests. Adjust in `fly/api/fly.toml`.

## Docker (local)

```bash
# Console
docker build -f fly/Dockerfile -t mesh-llm-console .
docker run -p 3131:3131 -p 9337:9337 mesh-llm-console

# API
docker build -f fly/Dockerfile --build-arg CMD=api -t mesh-llm-api .
docker run -p 9337:9337 mesh-llm-api
```
