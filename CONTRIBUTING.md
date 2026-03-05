# Contributing

This file covers local build and development workflows for this repository.

## Prerequisites

- macOS Apple Silicon for the default local workflow
- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js + npm (for UI development)

## Build from source

Build everything (llama.cpp fork, mesh binary, and UI production build):

```bash
just build
```

Create a portable bundle:

```bash
just bundle
```

## UI development workflow

Use this two-terminal flow for UI development.

Terminal A (run `mesh-llm` yourself):

```bash
mesh-llm --port 9337 --console 3131
```

If `mesh-llm` is not on your `PATH`:

```bash
./mesh-llm/target/release/mesh-llm --port 9337 --console 3131
```

Terminal B (run Vite with HMR):

```bash
just ui-dev
```

Open:

```text
http://127.0.0.1:5173
```

`ui-dev` defaults:

- Serves on `127.0.0.1:5173`
- Proxies `/api/*` to `http://127.0.0.1:3131`

Overrides:

```bash
# Different backend API origin for /api proxy
just ui-dev http://127.0.0.1:4141

# Different Vite dev port
just ui-dev http://127.0.0.1:3131 5174
```

## InferenceHub integration (local dev)

Use this flow when developing `mesh-llm` + `inferencehub` together.

Assumes:

- `decentralized-inference` at `/Users/jdumay/code/mesh/decentralized-inference`
- `inferencehub.cc` at `/Users/jdumay/code/mesh/inferencehub.cc`

Terminal A (`inferencehub.cc`):

```bash
cd /Users/jdumay/code/mesh/inferencehub.cc
npm install
cp .env.example .env
cp .dev.vars.example .dev.vars
wrangler d1 migrations apply inferencehub --local
npm run dev
```

Terminal B (`mesh-llm` with hub API target):

```bash
cd /Users/jdumay/code/mesh/decentralized-inference
MESH_LLM_HUB_BASE_URL=http://127.0.0.1:8787 \
  ./mesh-llm/target/release/mesh-llm --auto --inferencehub
```

Open:

- InferenceHub web: `http://127.0.0.1:5173`
- mesh-llm console: `http://127.0.0.1:3131`

From the mesh-llm console:

1. Click `Login with InferenceHub`
2. Complete device approval in local InferenceHub
3. Link the current mesh or leave local mesh and join/create on InferenceHub

Notes:

- In hub-linked mode, local mesh invites and local publish/discovery flows are disabled.
- If your worker port differs from `8787`, update `MESH_LLM_HUB_BASE_URL`.
- For release binaries, replace the local executable path with `mesh-llm` on `PATH`.

## Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just --list           # list all recipes
```
