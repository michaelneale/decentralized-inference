# Image Generation via stable-diffusion.cpp

Add image generation (FLUX, SD3.5, etc.) to mesh-llm by managing `sd-server`
as a second backend alongside `llama-server`. No pipeline splitting — image
models run solo on a single node, just like small LLMs.

## Why it fits

- **`sd-server`** is an off-the-shelf HTTP binary from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), same pattern as llama-server
- **OpenAI-compatible API**: `POST /v1/images/generations` → base64 image response
- **Metal/CUDA/Vulkan** backends, GGUF quantized weights, runs FLUX.1-dev q4_k in ~6.4GB VRAM
- **One model per process**, same as our "one model per node" design
- No tensor split needed — diffusion models don't benefit from multi-node splitting

## Architecture

```
               ┌─────────────────────────────────┐
               │         mesh-llm node            │
               │                                  │
  :9337 ──────►│  API proxy                       │
               │    ├─ /v1/chat/completions ──► llama-server (:N)
               │    ├─ /v1/images/generations ─► sd-server (:M)
               │    └─ /v1/images/edits ───────► sd-server (:M)
               │                                  │
               │  Process manager (launch.rs)     │
               │    ├─ start_llama_server()        │
               │    └─ start_sd_server()     NEW  │
               └─────────────────────────────────┘
```

In the mesh case, the API proxy already routes by model name across QUIC
tunnels. Image models just become another entry in the routing table — if the
current node doesn't serve images, the request tunnels to the node that does.

## sd-server invocation

```bash
sd-server \
  --diffusion-model flux1-dev-q4_k.gguf \
  --vae ae.safetensors \
  --clip_l clip_l.safetensors \
  --t5xxl t5xxl_fp16.safetensors \
  --clip-on-cpu \
  --fa \
  --listen-port 1234
```

Health check: `GET /` returns `"Stable Diffusion Server is running"` (200 OK).

## Model bundles

Unlike LLMs (single .gguf), diffusion models are multi-file bundles:

| Component | File | Size | GPU? |
|-----------|------|------|------|
| Diffusion model | `flux1-dev-q4_k.gguf` | ~6.4 GB | Yes |
| VAE | `ae.safetensors` | ~167 MB | Yes (or `--vae-on-cpu`) |
| CLIP-L | `clip_l.safetensors` | ~235 MB | CPU ok (`--clip-on-cpu`) |
| T5XXL | `t5xxl_fp16.safetensors` | ~9.5 GB | CPU ok (`--clip-on-cpu`) |

### Bundle convention

Store bundles in `~/.models/` with a directory per model:

```
~/.models/
├── GLM-4.7-Flash-Q4_K_M.gguf          # LLM (single file, existing)
├── flux1-dev/                           # Image model bundle (new)
│   ├── flux1-dev-q4_k.gguf
│   ├── ae.safetensors
│   ├── clip_l.safetensors
│   └── t5xxl_fp16.safetensors
└── flux1-schnell/
    ├── flux1-schnell-q4_k.gguf
    ├── ae.safetensors                   # shared VAE, could symlink
    ├── clip_l.safetensors
    └── t5xxl_fp16.safetensors
```

Detection: a directory in `~/.models/` containing a `*.gguf` + `ae.safetensors`
(or `*vae*`) = image model bundle. `scan_local_models()` returns these alongside
LLM GGUFs with a type tag.

## Implementation plan

### Phase 1: Build & launch (standalone)

1. **Justfile**: Add `build-sd` recipe — clone stable-diffusion.cpp, cmake with
   `-DSD_METAL=ON`, build `sd-server` binary.

2. **`launch.rs`**: Add `start_sd_server()`:
   - Takes bundle path (directory), listen port
   - Resolves component files (diffusion model, vae, clip_l, t5xxl)
   - Spawns `sd-server` with appropriate flags
   - Health-checks `GET /` until 200
   - Same log-to-file, detach pattern as `start_llama_server()`

3. **`--offline` mode**: Detect image bundles alongside LLMs. Start sd-server
   for each (or first) image bundle found. Route by path in the proxy.

### Phase 2: Mesh integration

4. **Model type in gossip**: Extend `PeerAnnouncement.serving` to include a
   model type field (`llm` | `image`). Existing peers ignore unknown types
   (forward-compatible).

5. **API proxy routing**: In `proxy.rs`, route by HTTP path:
   - `/v1/chat/completions`, `/v1/models` → llama-server (local or tunneled)
   - `/v1/images/generations`, `/v1/images/edits` → sd-server (local or tunneled)

6. **Election**: Image models participate in the same election logic. They just
   use `start_sd_server()` instead of `start_llama_server()`. No tensor split —
   image models always run solo (VRAM must fit on one node).

### Phase 3: Download & catalog

7. **`download.rs`**: Add image model entries to catalog. Download is a bundle
   (4 files) rather than a single GGUF. Could use HuggingFace repo download
   or individual file URLs. Preconverted GGUF bundles available from:
   - [leejet/FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf)
   - [leejet/FLUX.1-schnell-gguf](https://huggingface.co/leejet/FLUX.1-schnell-gguf)
   - VAE/CLIP/T5 from [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders)

8. **Bundle command**: `mesh-llm download flux-dev` fetches all 4 files into
   `~/.models/flux1-dev/`.

### Phase 4: Bundle & deploy

9. **`just bundle`**: Include `sd-server` binary in the portable tarball.

## VRAM budget

For FLUX.1-dev q4_k with `--clip-on-cpu`:
- Diffusion model: ~6.4 GB GPU
- VAE: ~167 MB GPU (or CPU with `--vae-on-cpu`)
- CLIP + T5: CPU only
- **Total GPU: ~6.6 GB** — fits on 8GB nodes

This means a 24GB Mac can serve an LLM + an image model simultaneously if the
LLM is small enough. A 128GB Mac can easily host both a 70B LLM and FLUX.

## API shape

Request (OpenAI-compatible):
```json
POST /v1/images/generations
{
  "prompt": "a cat astronaut floating in space",
  "size": "1024x1024",
  "n": 1
}
```

Response:
```json
{
  "created": 1709000000,
  "data": [
    { "b64_json": "iVBORw0KGgo..." }
  ]
}
```

Also supports `/v1/images/edits` (img2img with mask) via multipart form data,
and A1111-compatible `/sdapi/v1/txt2img` and `/sdapi/v1/img2img`.

## Non-goals (for now)

- **Splitting diffusion models across nodes**: Not needed — even the largest
  FLUX models fit in 12GB quantized. If giant video models (Wan2.1, etc.)
  need splitting later, that's a separate design.
- **Streaming/progress**: sd-server doesn't stream — it blocks until the image
  is done (typically 5-30s). Fine for now.
- **LoRA hot-loading**: sd-server supports LoRA but we won't manage LoRA
  libraries initially. Just serve the base model.
- **Video generation**: stable-diffusion.cpp supports video (Wan2.1) but we'll
  start with images only.

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [sd-server source](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/main.cpp)
- [FLUX docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md)
- [C API: stable-diffusion.h](https://github.com/leejet/stable-diffusion.cpp/blob/master/include/stable-diffusion.h)
- [FLUX.1-dev GGUF weights](https://huggingface.co/leejet/FLUX.1-dev-gguf)
