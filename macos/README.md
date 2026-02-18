# Mesh LLM macOS Menu Bar App

This directory contains the macOS menu bar app in Swift that controls `mesh-llm` directly (no virtualization).

## Current Status
- Menu bar item with actions:
  - `Start Mesh`
  - `Stop Mesh`
  - `Restart Mesh`
  - `Copy Token`
  - `Open Console`
  - `Open Logs`
  - `Settings...`
  - `Quit`
- State-driven enable/disable behavior for menu actions
- `mesh-llm` process manager wired to start/stop/restart a local binary
- Status icon tint changes by state (running/error/transition/stopped) using the same base icon
- App packaging bundles `mesh-llm` into `.app/Contents/Resources/mesh-llm`
- Real health/token flow via local endpoints (`/health`, console `/api/status`)
- Writes logs to `~/Library/Logs/MeshLLMMenuBar/mesh-llm.log` and can open them in Console.app
- Startup waits for mesh health and may take a long time on first run while model download/loading completes

## Build (dev)
```bash
swift build
```

## Run (dev)
```bash
swift run MeshLLMMenuBar
```

The app runs as an accessory app (menu bar only).

## Runtime Configuration
Configure via `Settings...` in the menu.

Defaults:
- Model: `Qwen2.5-32B-Instruct-Q4_K_M`
- API port: `9337`
- Console port: `3131`

Environment variables still override settings:

- `MESH_LLM_MODEL_PATH` (path to GGUF/model arg for `--model`)
- `MESH_LLM_JOIN_TOKEN` (token for `--join`)
- `MESH_LLM_API_PORT` (default `9337`)
- `MESH_LLM_CONSOLE_PORT` (default `3131`, needed for token fetch)

Example:
```bash
MESH_LLM_MODEL_PATH="$HOME/.models/GLM-4.7-Flash-Q4_K_M.gguf" \
MESH_LLM_API_PORT=9337 \
MESH_LLM_CONSOLE_PORT=3131 \
swift run MeshLLMMenuBar
```

## Build + Bundle App With mesh-llm
This is the main workflow for distributable app packaging:

From `external/decentralized-inference/macos`:

```bash
./scripts/package-app.sh
```

From `external/decentralized-inference`:

```bash
cd macos
./scripts/package-app.sh
```

What this does:
1. Runs `../mesh-llm/scripts/build-mesh-binary.sh`
2. Builds release binary (`swift build -c release`)
3. Creates `dist/Mesh LLM.app` and bundles `mesh-llm` into `Contents/Resources`
4. Bundles `rpc-server` and `llama-server` from `../llama.cpp/build/bin`
5. Bundles app icon from `Sources/Resources/MeshLLM.icns`

## Project Layout
- `Sources/App`: app entry + app state
- `Sources/MenuBar`: menu bar controller and action wiring
- `Sources/Persistence`: settings storage + defaults
- `Sources/Process`: mesh process manager and bundled binary discovery
- `Sources/Mesh`: health/token provider
- `Sources/Utilities`: utility services (clipboard)
- `scripts/package-app.sh`: packaging pipeline (build mesh + build app + bundle mesh)
- `../mesh-llm/scripts/build-mesh-binary.sh`: local mesh binary build contract
