# Usage Guide

Use this operational reference for installation details, setup, service mode,
model storage, and runtime control.

For command-by-command CLI usage, model resolution rules, and JSON automation examples, see [CLI.md](./CLI.md). For the field-by-field `config.toml` reference, see the [canonical configuration reference](https://meshllm.cloud/docs/pages/config-reference/).

## Installation details

Install the latest release executable:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash
```

On Windows, use PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.ps1 | iex
```

To opt into the latest published prerelease bundle instead:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash -s -- --pre-release
```

The installer puts `mesh-llm` on your `PATH`. After install, run `mesh-llm setup`
to finish runtime configuration and, on supported macOS and Linux machines,
optionally install the background service.

Source builds must use `just`:

```bash
git clone https://github.com/Mesh-LLM/mesh-llm
cd mesh-llm
just build
```

Requirements:

- `just`
- `cmake`
- Rust toolchain
- Node.js 24 + npm

Backend-specific notes:

- NVIDIA builds require `nvcc`
- AMD builds require ROCm/HIP
- Vulkan builds require the Vulkan development files and `glslc`
- CPU-only and Jetson/Tegra are also supported

For full build details, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Common commands

```bash
mesh-llm setup
mesh-llm serve --auto
mesh-llm serve --model Qwen2.5-32B
mesh-llm serve --join <token>
mesh-llm serve --discover "my-mesh"
mesh-llm serve --model MiniMax-M2.5-Q4_K_M --mesh-guardrails metrics
mesh-llm share http://localhost:11434
mesh-llm client --auto
mesh-llm gpus
mesh-llm discover
mesh-llm discover --name "my-mesh"
```

Mesh workflow details live in [MESHES.md](MESHES.md). Big-model split serving
lives in [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md).

If you run `mesh-llm` with no arguments, it prints `--help` and exits. It does not start the console or bind ports until you choose a mode.
Bare `mesh-llm serve` loads startup models from `[[models]]` in `~/.mesh-llm/config.toml`.

## Benchmark tuning

`mesh-llm benchmark tune` measures local model-serving throughput for already-downloaded local models. It resolves local targets, plans safe startup settings, creates temporary per-trial configs, starts isolated local `mesh-llm serve` children, sends OpenAI-compatible chat-completion requests, reports decode tok/s plus setup/readiness/request/shutdown/total timing stats for each context/batch/ubatch/mmap/mlock/flash-attention/speculative-decoding candidate, and keeps trial logs under `target/gpu-tune/`.

Benchmark tune reports the raw highest-throughput trial, the Pareto frontier for decode tok/s versus `ctx_size`, and a recommended trial. By default, the recommendation treats candidates within `10.0%` of the raw best decode tok/s as throughput-equivalent, then chooses the largest context window among those candidates.

```bash
mesh-llm benchmark tune --model /models/qwen3-8b.gguf
mesh-llm benchmark tune --models /models/qwen3-8b.gguf,/models/mixtral.gguf --json
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --ctx-sizes 4096,8192,16384 --batch-sizes 1024,2048 --ubatch-sizes 256,512
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --mmap-values auto,true,false --mlock-values true,false
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --flash-attention on,off
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types auto
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types mtp --debug-telemetry --json
mesh-llm benchmark tune --model /models/qwen3-mtp.gguf --speculative-types mtp,mtp-ngram,disabled --spec-draft-max-tokens 4,8,16
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --throughput-tolerance-pct 2.5
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --apply
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --apply --replace-existing
mesh-llm benchmark tune --model /models/qwen3-8b.gguf --launch-args
```

If `--mmap-values` is omitted, benchmark tune tries `auto`, `true`, and `false`. If `--mlock-values` is omitted, it tries `false` and only tries `true` when the current mlock limit can cover the evaluated budget. If `--flash-attention` is omitted, flash attention is not varied during the sweep; when supplied (e.g. `--flash-attention on,off`), trial count doubles and the recommendation applies the best flash attention setting.
If `--speculative-types` is omitted, benchmark tune uses `auto`: native MTP and the bounded MTP + request-local N-gram cache composite are tried for MTP-looking targets, locally discoverable draft models are tried when available, and a disabled baseline is included for comparison. Use `--speculative-types mtp,mtp-ngram,draft,disabled` to force an explicit speculative sweep, or `--no-speculative-tune` to run only the disabled baseline.
Use `--apply` to write the recommended settings into `~/.mesh-llm/config.toml`, and combine with `--replace-existing` to overwrite existing writable recommendation fields. `--launch-args` prints generated `mesh-llm serve` arguments for local launch without writing config.
Use `--debug-telemetry` when proving speculative decoding behavior: each trial log includes Skippy debug telemetry, including `llama_stage.native_mtp.*` summary attributes for MTP drafted, accepted, rejected, and accept-rate counts.

Use `mesh-llm gpus detect` when you want to refresh the raw hardware fingerprint, bandwidth, and compute hints rather than benchmark model-serving throughput.

## Setup

Use `mesh-llm setup` after the executable is installed. It configures the native runtime and can install the background service on supported macOS and Linux machines.

See [CLI.md](./CLI.md) for the setup flags and the service options.

## Model catalog

List or fetch models from the built-in catalog:

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download 72b --draft
```

Draft pairings for speculative decoding:

| Model | Size | Draft | Draft size |
|---|---|---|---|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |

## Specifying models

`mesh-llm serve --model` accepts several formats. Hugging Face-backed models are cached in the standard Hugging Face cache on first use.

```bash
mesh-llm serve --model Qwen3-8B
mesh-llm serve --model Qwen3-8B-Q4_K_M
mesh-llm serve --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --gguf ~/my-models/custom-model.gguf
mesh-llm serve --gguf ~/my-models/qwen3.5-4b.gguf --mmproj ~/my-models/mmproj-BF16.gguf
```

## Startup config

`mesh-llm serve` also loads startup models from `~/.mesh-llm/config.toml` by default.

Use the persisted TOML for future starts or reloads. It does not rewrite active
sessions in place, and request payload values still win over any request
defaults from the file.

### Split a pre-copied local GGUF across nodes

Use `source_policy = "local-required"` when every split-serving node already
has the same GGUF bytes and the runtime must never fetch or copy the model. Use
the same logical `model` value on every node; `hardware.model_path` is
node-local and may be a different absolute path on each host.

Node A can use `/etc/mesh-llm/node-a.toml`:

```toml
[[models]]
model = "orca-70b-q4-k-m"

[models.hardware]
model_path = "/srv/models/orca/orca-70b-q4-k-m.gguf"

[models.skippy]
source_policy = "local-required"
```

Node B can use the same logical model with its own path in
`/etc/mesh-llm/node-b.toml`:

```toml
[[models]]
model = "orca-70b-q4-k-m"

[models.hardware]
model_path = "/mnt/fast-storage/llms/orca-70b-q4-k-m.gguf"

[models.skippy]
source_policy = "local-required"
```

Start the private mesh on node A, then use its invite token on node B:

```bash
# node A
mesh-llm serve --config /etc/mesh-llm/node-a.toml --split

# node B
mesh-llm serve --config /etc/mesh-llm/node-b.toml --split --join <token>
```

For a split GGUF set, point `model_path` at the first shard on each node. Every
numbered shard must be present beside it and use the standard GGUF shard
naming. The directory may still differ per node:

```toml
[models.hardware]
model_path = "/srv/models/orca/orca-70b-q4-k-m-00001-of-00004.gguf"
```

The CLI naming form is also supported. Put the policy in a small config file:

```toml
# /etc/mesh-llm/strict-local.toml
[defaults.skippy]
source_policy = "local-required"
```

Then give every node the same plain alias and its own local source path:

```bash
# node A
mesh-llm serve --config /etc/mesh-llm/strict-local.toml \
  --gguf /srv/models/orca/orca-70b-q4-k-m.gguf \
  --model orca-70b-q4-k-m --split

# node B
mesh-llm serve --config /etc/mesh-llm/strict-local.toml \
  --gguf /mnt/fast-storage/llms/orca-70b-q4-k-m.gguf \
  --model orca-70b-q4-k-m --split --join <token>
```

Strict local-source behavior has these operator-visible invariants:

- A source must be an absolute, regular, non-symlink local GGUF path. A
  sharded source must start at `-00001-of-000NN.gguf`; every shard is part of
  the identity. Direct GGUFs are supported, but `hf://` layer packages are not
  valid strict-local sources.
- A plain GGUF is identified by its SHA-256. For a sharded set, mesh-llm hashes
  every shard and derives an order-sensitive aggregate SHA-256 from the shard
  ordinal, size, and digest. The resulting `local-gguf://sha256/...` identity
  and synthetic manifest contain no absolute path or filename.
- The first strict content-addressed verification of a source snapshot hashes
  every source byte and never consults the persistent digest sidecar. Within
  one process, later inventory/load checks may reuse that verified identity
  only while file size, modification time, inode change time, device, and
  inode remain unchanged; otherwise they rehash.
  The persistent cache at `~/.mesh-llm/cache/hashes` (or
  `MESH_LLM_HASH_CACHE_DIR`) remains limited to legacy path-local identities in
  `fallback` mode. The registered path is only a node-local locator.
- Strict inventory and load never consult the model catalog or Hugging Face
  and never use peer artifact transfer. A missing file, incomplete shard set,
  or SHA-256 mismatch excludes that node or fails startup instead of fetching
  replacement bytes.
- Treat the configured GGUF set as immutable for the lifetime of a strict
  topology. mesh-llm verifies its identity during inventory and immediately
  before load, but the native runtime subsequently opens the operator-owned
  path; this mode is not a sandbox against another local process replacing the
  file in that final interval.
- Nodes must advertise local-GGUF content-ID support before they can enter a
  strict split topology. Older nodes, where the additive capability is absent,
  are excluded before coordinator election and revalidated again before
  package inventory. They may remain in the mesh for compatible work, but
  cannot host this strict-local split.
- Omitting `source_policy`, or setting it to `"fallback"`, preserves the
  existing resolver and mixed-version behavior, including catalog, Hugging
  Face, and peer artifact transfer where eligible. A model-level policy
  overrides `[defaults.skippy]`; leaving it unset inherits the default.

## Request logging

The embedded console includes a local **Logs** tab for request history,
details, and bounded maintenance operations. It is not a public mesh service:
the log routes accept trusted-local management access only. The request ledger
shows active requests and terminal outcomes, and uses a dedicated log event
stream with an authoritative REST refresh and bounded polling recovery when
needed. Ledger filters are applied through REST hydration; only supported
request-ID filters are sent to the stream, and the last cursor is reused when
the stream reopens. A host without the logs capability remains inert rather
than retrying the old status stream.

Logging retains metadata by default. Redacted artifact capture is an explicit
configuration choice and never restores data that was not captured. For
retention, exports, scoped cleanup, webhook retry, privacy, and recovery
guidance, use [LOGGING.md](LOGGING.md).

## Runtime mode and daemon lifecycle

The `[runtime]` section controls daemon-level behavior: operating mode, startup
failure policy, drain timeouts, and opt-in host activity adaptation.

```toml
[runtime]
mode = "serve"                        # "client" | "serve" (default) | "on_demand"
startup_failure_policy = "best_effort" # "best_effort" (default) | "fail_fast"
drain_timeout_secs = 30               # 1..=3600, default 30
drain_timeout_max_secs = 300          # 1..=3600, default 300, must be >= drain_timeout_secs

[runtime.activity]
enabled = false                       # opt-in, default false
idle_after_secs = 300                 # 30..=86400, default 300
poll_interval_secs = 5                # 1..=60, default 5
resume_debounce_secs = 30             # 0..=300, default 30
response = "pause_remote"             # "pause_remote" (default) | "pause_all" | "reduce_priority"
advertisement = "coarse_state"        # "none" | "availability_only" | "coarse_state" (default) | "private_coarse_state"
```

### Operating modes

- **`serve`** (default, also when the field is absent): the daemon starts mesh
  gossip, discovery, tunnels, management, owner-control, OpenAI ingress, and
  plugins before resolving models. Configured and CLI models load eagerly as
  startup intents.
- **`client`**: read-only safety boundary. The node joins the mesh and routes
  requests but never serves local models. Persisted `client` mode conflicts
  with explicit `serve` or `--model`/`--gguf`/`--mmproj` flags and fails
  **before** listeners start. Remediation: change the config mode to `serve`
  or `on_demand`, or remove the model flags.
- **`on_demand`**: the daemon starts worker-capable but idle. Configured models
  are preference/candidate metadata, not eager intents. Models load only when
  explicitly requested through local commands, owner-control lifecycle
  commands, advisory mesh demand, or explicit CLI `--model`/`--gguf`
  arguments. Explicit CLI models remain eager startup intents.

### Startup failure policy

- **`best_effort`** (default): continue starting if a configured model fails to
  load. The daemon stays alive and degraded; the error is logged.
- **`fail_fast`**: abort startup if any eager startup model fails to load.
  Applies **only** to eager startup intents. The daemon waits for terminal
  outcomes, then orderly closes listeners and metadata and exits nonzero with a
  bounded summary. Owner and advisory failures never kill the daemon.

### Drain timeout

- `drain_timeout_secs` (default 30): seconds before forcibly unloading a
  draining instance after new work is rejected.
- `drain_timeout_max_secs` (default 300): maximum allowed drain timeout cap.
  Per-command overrides are capped by this value.
- Force drain uses deadline 0 (immediate unload).

### Host activity policy

Activity adaptation is **opt-in** (`enabled = false` by default). When enabled,
the daemon detects host activity and adapts inference admission:

- `idle_after_secs` (default 300): seconds of inactivity before transitioning
  to idle.
- `poll_interval_secs` (default 5): how often to poll the activity detector.
- `resume_debounce_secs` (default 30): seconds to wait after activity resumes
  before re-enabling admission.
- `response`: what to do when activity is detected:
  - `pause_remote` (default): reject new inbound QUIC HTTP and stage transport
    work; local API/OpenAI/plugin work continues.
  - `pause_all`: also reject local OpenAI/plugin model dispatch. Management,
    owner-control, health, status, and unload/drain remain reachable.
  - `reduce_priority`: keep admissions open and invoke the best-effort priority
    controller. May surface degraded status on failure.
- `advertisement`: how to advertise admission state to mesh peers:
  - `none`: emit nothing.
  - `availability_only`: publish hosted/serving availability as
    explicitly-known-and-empty while non-admitting.
  - `coarse_state` (default): emit the admission enum and also publish
    known-empty availability for old peers.
  - `private_coarse_state`: emit the enum only on private meshes; use
    known-empty availability publicly.

**Platform support**: unsupported or headless platforms report `Unknown` and
never infer idle. Manual override (`Auto`/`Active`/`Idle`) is session-only and
not persisted in config. `reduce_priority` is best-effort: it captures the
original process state, applies only through a safe platform capability, and
restores on idle/shutdown.

**Privacy**: no raw owner payloads, input events, app/window names, usernames,
idle durations, timestamps, or detector errors appear in gossip, public status,
logs, or telemetry. Only the coarse admission enum and known-empty availability
are advertised.

### Daemon states

`/api/status` includes an optional `runtime.daemon_state` field derived with
this precedence:

1. `stopping` — shutdown in progress
2. `degraded` — terminal failure or priority restoration failure
3. `ready_serving` — local model serving inference
4. `ready_proxying` — healthy remote/plugin route, no local serving
5. `ready_idle` — listeners ready, no models loaded
6. `starting` — not yet ready

Coexistence is represented by capability booleans (`worker_capable`,
`local_serving`, `proxying`, `plugin_ingress`, `accepting_local`,
`accepting_remote`), not extra enum combinations.

### Runtime status and activity API routes

- `GET /api/runtime/intents` — filtered intent list, capped at 256 entries.
  Shows durable/configured and session intents, but never raw owner payloads or
  detector details.
- `GET /api/runtime/activity` — current activity policy status.
- `PUT /api/runtime/activity/override` — set manual override (`auto`/`active`/`idle`).
- `DELETE /api/runtime/activity/override` — restore auto.

### Compatibility

Additive defaulted TOML needs no config-version bump. Canonical config still
travels as TOML. Public `/0` and `/1` ALPN remain unchanged. Owner-control ALPN
remains unchanged. Old peers treat missing admission as eligible (legacy
behavior). New lifecycle commands may return typed `CONTROL_UNSUPPORTED` on
older hosts.

The example below shows every configuration section with annotations. All
sections and fields are optional unless noted.

```toml
# ~/.mesh-llm/config.toml
#
# Comprehensive configuration reference.
#
# Precedence (highest → lowest):
#   explicit request field value
#   → per-model config ([[models]] entry)
#   → [defaults.*] global config
#   → family / topology policy
#   → built-in runtime defaults
#
# Request defaults are merged ONLY at the OpenAI frontend boundary when the
# incoming request field is absent or null. They never enter StageConfig,
# protobuf, or any lower runtime layer.

version = 1

# ---------------------------------------------------------------------------
# GPU assignment policy
# ---------------------------------------------------------------------------

[gpu]
# "auto"   — let the planner pick the best visible device (default)
# "pinned" — require an explicit device= in every model or in [defaults.hardware]
assignment = "auto"
parallel   = 2        # total parallel inference slots across all models

# ---------------------------------------------------------------------------
# Node identity and network
# ---------------------------------------------------------------------------

[owner_control]
bind           = "0.0.0.0:7447"          # QUIC listen address
advertise_addr = "203.0.113.10:18443"    # address announced to peers

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

[telemetry]
enabled  = true
endpoint = "http://localhost:4317"       # OTLP collector

# ---------------------------------------------------------------------------
# Global defaults — applied to every model that does not override the field
# ---------------------------------------------------------------------------

# --- Context, batching, and KV cache -------------------------------------
[defaults.model_fit]
ctx_size         = 8192          # context window size (tokens)
batch            = 512           # n_batch — prompt-processing chunk
ubatch           = 128           # n_ubatch — micro-batch within a batch
cache_type_k     = "auto"        # KV key dtype: auto f16 f32 bf16 q8_0 q4_0 …
cache_type_v     = "auto"        # KV value dtype (same enum)
flash_attention  = "auto"        # auto on off
kv_cache_policy  = "balanced"    # macro preset: auto quality balanced saver
                                 #   quality  → f16/f16, no forced RAM cap
                                 #   balanced → preserve runtime defaults
                                 #   saver    → low-memory dtypes + offload
                                 # explicit cache_type_k/v always wins over preset
kv_offload       = "auto"        # bool or "auto" — KV residency / offload policy
kv_unified       = "auto"        # bool or "auto" — unified KV layout (schema-reserved)
cache_ram_mib    = 0             # byte cap for KV cache in MiB; 0 = no cap (schema-reserved)
cache_idle_slots = 0             # idle slot retention count (schema-reserved)
prompt_cache     = "auto"        # bool or "auto" — reuse previous prompt KV
swa_full         = false         # sliding-window attention (model-family specific)

# exact-prefix cache sub-section
[defaults.model_fit.prefix_cache]
enabled              = true
max_entries          = 64
max_bytes            = 0         # 0 = no explicit byte cap
min_tokens           = 64
shared_stride_tokens = 32        # stride for shared-prefix record matching
shared_record_limit  = 4         # max retained shared-prefix records
payload_mode         = "auto"    # resident-kv kv-recurrent full-state auto

# Schema-reserved fields (accepted but not yet wired to runtime):
# keep_tokens          = 256     # session prompt retention
# context_shift        = "auto"  # long-context shift
# checkpoint_interval  = 100     # KV checkpoint cadence
# checkpoint_count     = 5       # KV checkpoint retention
# lookup_cache_static  = "/path/to/static.cache"
# lookup_cache_dynamic = "/path/to/dynamic.cache"

# --- Hardware and model loading ------------------------------------------
[defaults.hardware]
model_runtime    = "auto"        # backend: auto cpu cuda rocm metal vulkan
device           = "auto"        # device id/index, e.g. "cuda:0" or "0"
gpu_layers       = "auto"        # integer >= -1, or "auto" (all layers)
placement        = "auto"        # planner placement strategy enum
split_mode       = "auto"        # multi-GPU split: auto none layer row
main_gpu         = 0             # primary device index for split_mode tuning
safety_margin_gb = 2.0           # reserved headroom; maps to fit_target_mib
fit_target_mib   = 0             # explicit allocatable-memory target (MiB)
                                 # do NOT write derived values back into TOML
fit_context      = "auto"        # bool or "auto" — estimator context-fit mode
mmap             = "auto"        # bool or "auto" — memory-mapped model load
mlock            = false         # pin model pages in RAM
direct_io        = false         # bypass page cache for model reads
repack           = false         # backend-specific repack flag
op_offload       = false         # backend-specific op-offload flag
no_host_buffer   = false         # backend-specific host-buffer flag
warmup           = "auto"        # bool or "auto" — post-load warmup pass
check_tensors    = false         # tensor-validation at load time (debug)

# multi-GPU tensor split (per-GPU ratio list or backend-native string)
# tensor_split = [0.6, 0.4]

# staged (skippy) layer ownership — set by planner; override only when manual
# stage_layer_start = 0
# stage_layer_end   = 15

# model artifact — typically set per-model; unusual in [defaults]
# model_path  = "/models/default.gguf"
# hf_repo     = "org/model-GGUF"
# hf_file     = "model-q4_k_m.gguf"
# mmproj      = "mmproj-f16.gguf"

# LoRA adapters and control vectors
# lora_adapters   = ["/adapters/adapter-1.gguf"]
# control_vectors = ["/vectors/cv-1.gguf"]

# MoE (Mixture-of-Experts) routing
# cpu_moe   = "auto"   # bool or "auto"
# n_cpu_moe = 0        # number of experts to route to CPU

# --- Throughput, scheduling, and CPU -------------------------------------
[defaults.throughput]
parallel             = 1           # concurrent request slots
continuous_batching  = "auto"      # bool or "auto"
threads              = 8           # CPU inference thread count
threads_batch        = 4           # CPU batch-processing thread count
tuning_profile       = "balanced"  # macro preset: throughput balanced saver
                                   #   throughput → larger batch/ubatch, more parallel
                                   #   balanced   → preserve runtime defaults
                                   #   saver      → smaller batch/ubatch, lower parallel
                                   # explicit low-level fields always win over preset
slot_prompt_similarity = 0.5       # slot-reuse heuristic threshold
priority               = "normal"  # scheduler priority hint (integer or string)

# CPU affinity and NUMA (advanced — usually leave unset)
# cpu_affinity = "0-7"
# numa         = "distribute"
# poll         = "auto"   # bool or "auto" — polling strategy

# Rejected in model config — stays operational/host-level:
# threads_http       — HTTP worker pool
# sleep_idle_seconds — power-management idle

# --- Skippy staged serving -----------------------------------------------
[defaults.skippy]
source_policy                   = "fallback" # default; use local-required only for pre-copied GGUFs
binary_stage_transport          = "auto"    # auto on off
prefill_chunking                = "fixed"   # fixed schedule none
prefill_chunk_size              = 512       # tokens per prefill chunk
lifecycle_startup_timeout_ms    = 30000     # stage startup grace period (ms)
lifecycle_readiness_interval_ms = 250       # readiness poll interval (ms)
lifecycle_health_interval_ms    = 5000      # health-check interval (ms)

# Staged-only / manual topology (set by planner; override carefully)
# stage_model_path       = "/packages/stage-0.pkg"
# stage_role             = "prefill"
# stage_topology         = "2-stage-split"
# prefill_chunk_schedule = "128,256,512"   # custom progressive schedule

# --- Speculative decoding ------------------------------------------------
[defaults.speculative]
strategy                   = "auto"          # auto disabled mtp or a package strategy id
mode                       = "auto"          # external draft-model mode: auto disabled draft
draft_selection_policy     = "auto"          # auto manual heuristic
pairing_fault              = "warn_disable"  # warn_disable fail_open fail_closed
draft_acceptance_threshold = 0.0             # 0.0 = use runtime default
spec_default               = "auto"          # bool or "auto"

# Draft model source (per-model is more typical; these are global fallbacks)
# draft_model = "org/draft-GGUF:Q4_K_M"
# draft_hf_repo    = "org/draft-GGUF"
# draft_hf_file    = "draft-q4_k_m.gguf"

# Native MTP strategy override
# strategy = "mtp"  # force native model MTP when available
# strategy = "disabled"       # disable package/model native MTP
# draft_max_tokens = 1        # MTP/draft max draft-token window (native MTP defaults to 1)
# draft_min_tokens = 0        # MTP/draft min draft-token window

# Draft hardware (leave unset to share host model's device)
# draft_gpu_layers   = -1
# draft_device       = "cuda:1"
# draft_threads      = 4
# draft_cache_type_k = "q8_0"
# draft_cache_type_v = "q8_0"

# N-gram proposer and MTP extension.
# `cache` is request-local and requires ngram_max <= 4; `suffix` is a pure-Rust
# longest-suffix (prompt-lookup) matcher allowing ngram_max <= 64.
# ngram_proposer            = "cache"  # cache | suffix
# ngram_min                 = 2
# ngram_max                 = 4
# ngram_max_proposal_tokens = 6        # output budget, separate from ngram_max
# extension_max_tokens      = 6        # fixed request-local continuation horizon

# Target VerifyWindow and native-MTP recovery controls
# verify_window_min_tokens                    = 1
# verify_window_max_tokens                    = 6
# verify_window_pipeline_depth                = 2
# native_mtp_reject_cooldown_tokens           = 4
# native_mtp_suppress_cooldown_drafts         = true
# native_mtp_suppress_cooldown_draft_limit    = 1

# --- Request defaults (merged at OpenAI frontend only) -------------------
[defaults.request_defaults]
# Sampling — explicit request values always win
temperature       = 0.8
top_p             = 0.95
top_k             = -1           # -1 = disabled
min_p             = 0.05
typical_p         = 1.0
top_nsigma        = 0.0
dynatemp_range    = 0.0
dynatemp_exponent = 1.0
repeat_penalty    = 1.1
repeat_last_n     = 64
presence_penalty  = 0.0
frequency_penalty = 0.0
seed              = -1           # -1 = random

# Mirostat sampling (alternative to top_p/top_k)
mirostat_mode          = 0       # 0 off, 1 v1, 2 v2
mirostat_entropy       = 5.0
mirostat_learning_rate = 0.1

# Stop sequences (string or list of strings)
stop = ["<|im_end|>", "</s>"]

# Token budget
max_tokens = 2048
ignore_eos = false

# Sampler ordering (leave unset to use runtime default)
# samplers         = ["top_k", "top_p", "temperature"]
# sampler_sequence = "kpt"

# Logit bias: token_id → bias delta (TOML inline table)
# logit_bias = { "12345" = -2.0, "67890" = 1.5 }

# Reasoning (for thinking models)
reasoning_format  = "auto"   # auto none deepseek deepseek-legacy hidden
reasoning_enabled = "auto"   # bool or "auto" / "on" / "off"
reasoning_budget  = "auto"   # integer token budget, or "auto"

# Chat template (leave unset to use model's embedded template)
# chat_template      = "chatml"
# chat_template_file = "/path/to.jinja"
# jinja              = false
# skip_chat_parsing  = false

# System prompt injected at the start of every conversation
# system_prompt = "You are a helpful assistant."

# Schema-reserved (accepted, not yet wired):
#   dry, xtc, adaptive  — advanced sampler bags
#   backend_sampling    — raw backend sampling passthrough
#   grammar, json_schema, logprobs
#   prefill_assistant, chat_template_kwargs

# --- Multimodal ----------------------------------------------------------
[defaults.multimodal]
mmproj           = "default-mmproj-f16.gguf"  # vision projector path or HF ref
# mmproj_url     = "https://huggingface.co/org/repo/resolve/main/mmproj.gguf"
mmproj_offload   = "auto"                     # bool or "auto"
image_min_tokens = 1
image_max_tokens = 4096
media_marker = "<__media__>"
batch_max_tokens = 1024
glm_dsa_policy = "auto"                       # "auto" or "v1"
generation_signal_window = 16

# Schema-reserved (accepted, not yet wired):
#   embeddings, reranking, pooling, vocoder

# --- Advanced server (operational — reject most in model config) ---------
[defaults.advanced.server]
alias = "my-cluster"   # friendly name shown in /api/status
# host, port, reuse_port, timeout, metrics, slots, props, and api_prefix are
# operational or rejected here, not model-settings controls.

# ===========================================================================
# Per-model entries — each [[models]] block overrides specific defaults
#
# The optional `profile` field distinguishes multiple entries for the same
# model artifact. When omitted, the entry uses the default (unnamed) profile.
# Two entries with the same `model` but different `profile` load as
# independent serving instances — each with its own settings and its own
# copy of the model weights.
#
# At the routing layer, named profiles appear as `{model_ref}#{profile}`.
# For example, `Qwen/Qwen3-8B:Q4_K_M#chat`.
# The default profile (no `#` suffix) keeps the bare model ref for backward
# compatibility.
# ===========================================================================

# ---------------------------------------------------------------------------
# Example 1: GPU-heavy model with staged serving and speculative decoding
# ---------------------------------------------------------------------------

[[models]]
model = "Qwen/Qwen3-8B:Q4_K_M"

[models.model_fit]
ctx_size        = 16384
batch           = 1024
ubatch           = 256
cache_type_k    = "f16"
cache_type_v    = "f16"
kv_cache_policy = "quality"    # overrides global "balanced"
flash_attention  = "on"
prompt_cache     = true

[models.model_fit.prefix_cache]
enabled     = true
max_entries = 128
min_tokens  = 128

[models.hardware]
device            = "cuda:0"
gpu_layers        = 99          # all layers on GPU
fit_target_mib    = 22528       # 22 GiB target
stage_layer_start = 0           # staged split: this node owns layers 0–15
stage_layer_end   = 15
split_mode        = "layer"
tensor_split      = [0.6, 0.4]  # two-GPU split ratios
main_gpu          = 0
mmap              = true
warmup            = true
lora_adapters     = ["/adapters/qwen-chat-v2.gguf"]

[models.throughput]
parallel            = 4
continuous_batching = true
tuning_profile      = "throughput"
threads             = 16
threads_batch       = 8

[models.skippy]
source_policy          = "fallback" # or "local-required" for identical pre-copied GGUFs
prefill_chunking       = "schedule"
prefill_chunk_size     = 256
prefill_chunk_schedule = "128,256,512,1024"

[models.speculative]
mode                   = "draft"
draft_model            = "org/qwen3-0.6b-draft:Q8_0"
draft_selection_policy = "manual"
pairing_fault          = "warn_disable"
draft_max_tokens       = 8
draft_gpu_layers       = 28
draft_device           = "cuda:1"
draft_cache_type_k     = "q8_0"
draft_cache_type_v     = "q8_0"

[models.request_defaults]
temperature      = 0.7
top_p            = 0.9
repeat_penalty   = 1.05
max_tokens       = 4096
reasoning_format = "hidden"
reasoning_budget = 512
system_prompt    = "You are a helpful coding assistant."
stop             = ["<|im_end|>"]

[models.multimodal]
mmproj           = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
mmproj_offload   = true
image_max_tokens = 8192

[models.advanced.server]
alias = "qwen3-8b"

# ---------------------------------------------------------------------------
# Example 2: CPU-only small model, minimal config
# ---------------------------------------------------------------------------

[[models]]
model = "bartowski/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf"

[models.hardware]
model_runtime = "cpu"
gpu_layers    = 0
mmap          = true

[models.model_fit]
ctx_size = 4096
batch    = 128
ubatch   = 64

[models.throughput]
threads        = 4
threads_batch  = 4
tuning_profile = "saver"

[models.request_defaults]
temperature = 0.9
max_tokens  = 512

[models.advanced.server]
alias = "gemma-tiny"

# ---------------------------------------------------------------------------
# Example 3: MoE model with CPU expert offload
# ---------------------------------------------------------------------------

[[models]]
model = "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf"

[models.hardware]
device         = "cuda:0"
gpu_layers     = 32
cpu_moe        = true
n_cpu_moe      = 4             # route 4 experts to CPU
split_mode     = "row"
placement      = "auto"
fit_target_mib = 20480

[models.model_fit]
ctx_size        = 8192
kv_cache_policy = "saver"

[models.throughput]
parallel = 2
threads  = 8

[models.advanced.server]
alias = "mixtral-8x7b"

# ---------------------------------------------------------------------------
# Example 4: Vision model from Hugging Face
# ---------------------------------------------------------------------------

[[models]]
model = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"

[models.hardware]
hf_repo    = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF"
hf_file    = "qwen2.5-vl-7b-instruct-q4_k_m.gguf"
device     = "cuda:0"
gpu_layers = 99

[models.multimodal]
mmproj           = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
mmproj_offload   = true
image_min_tokens = 16
image_max_tokens = 16384

[models.model_fit]
ctx_size = 8192

[models.advanced.server]
alias = "qwen-vl"

# ---------------------------------------------------------------------------
# Example 5: Multi-profile — same model, different serving configurations
# ---------------------------------------------------------------------------

[[models]]
model = "Qwen/Qwen3-8B:Q4_K_M"
profile = "deep-context"

[models.model_fit]
ctx_size = 32768
prompt_cache = true

[models.throughput]
parallel = 1
tuning_profile = "balanced"

[[models]]
model = "Qwen/Qwen3-8B:Q4_K_M"
profile = "interactive"

[models.model_fit]
ctx_size = 8192

[models.throughput]
parallel = 4
tuning_profile = "throughput"

[models.hardware]
device = "cuda:0"

# The first profile ("deep-context") dedicates a large context window with
# conservative parallelism for document analysis. The second ("interactive")
# prioritizes throughput for chat-style usage. Each loads independently and
# appears as a separate model in /v1/models:
#
#   Qwen/Qwen3-8B:Q4_K_M             ← default profile (if defined separately)
#   Qwen/Qwen3-8B:Q4_K_M#deep-context ← named profile
#   Qwen/Qwen3-8B:Q4_K_M#interactive   ← named profile
#
# Weight sharing between profiles is not yet supported — each loads its own
# copy of the model weights.

# ---------------------------------------------------------------------------
# Plugin declarations
# ---------------------------------------------------------------------------

[[plugin]]
name    = "blackboard"
enabled = true
command = "mesh-llm-plugin-blackboard"

# [[plugin]]
# name    = "openai-endpoint"
# url     = "http://localhost:8000/api/v1"
#
# [plugin.startup]
# connect_timeout_secs = 75
# init_timeout_secs = 90
# optional = true
# lazy_start = true
```

Use the default config:

```bash
mesh-llm serve
```

If no startup models are configured, `mesh-llm serve` remains alive as a
healthy zero-model daemon. It reports `ready_idle` while no local, plugin, or
remote route is available, and can load a model later without restarting.

Or an explicit path:

```bash
mesh-llm serve --config /path/to/config.toml
```

Config precedence:

- Request values override per-model config, which override `[defaults.*]`, which
  override family or topology policy, which finally override built-in runtime
  defaults.
- Request defaults only fill missing or null request fields at the OpenAI
  frontend boundary. Explicit request values win, and those defaults never
  become `StageConfig`, runtime load structs, protobuf payloads, or lower-layer
  runtime settings.
- Explicit `--model` or `--gguf` ignores configured `[[models]]` for model
  selection and tuning. An exact, unique `--model` ref may still inherit its
  configured pinned GPU selector. Unmatched `--model` refs and `--gguf` paths
  carry no configured model identity but may inherit only
  `defaults.hardware.device`.
- Explicit `--ctx-size` overrides configured `ctx_size` for the selected startup
  models.
- Explicit `--device` overrides the inherited device selector. Stable GPU IDs
  and backend names such as `CUDA0` resolve before native startup, including
  under automatic assignment. `--device CPU` selects CPU without requiring a
  GPU inventory, while `--device auto` keeps the inherited selector.
- Explicit `--mesh-guardrails <disabled|metrics|enforce>` seeds the
  server-side mesh guardrail mode for hosted Skippy startup models and later
  runtime-loaded models.
- `mmproj` is optional and only used when that startup model needs a projector
  sidecar.
- `skippy.*` staged-serving controls stay staged-only. Prefill controls,
  speculative draft controls, and manual stage layer ranges apply only when
  the model is started in staged mode. Activation frames always use raw f32.
- `safety_margin_gb` resolves to `hardware.fit_target_mib` by subtracting the
  reserved MiB from detected allocatable memory, and the derived target is not
  written back into TOML.
- Changing this file affects future starts or reloads, not active sessions.
- Plugin entries stay in the same file.
- `[plugin.startup]` controls how long mesh-llm waits for an external plugin to
  connect and initialize. `optional = true` records a missing installed plugin
  as inactive instead of rejecting the config, and `lazy_start = true` defers
  process launch until direct plugin use. This is useful for very slow legacy
  hosts or emulator-assisted startup paths.

## Speculative decode configuration

Configure speculative decoding under `[defaults.speculative]` for all staged
models, or under `[models.speculative]` to override one configured model. CLI
flags have the highest precedence, followed by the selected model, then
`[defaults.speculative]`; package strategies supply the remaining declared
defaults. The resolved plan is validated once before Skippy starts.

Set `strategy = "auto"` to use a package recommendation, `"disabled"` for
the no-speculation baseline, or `"mtp"` for native MTP. A package may also
publish stable names such as `mtp-cache`; that name is valid only for the
package that declares it. Direct GGUF serving can use `ngram-cache` or
`ngram-suffix` when it supplies valid N-gram bounds.

```toml
[[models]]
model = "meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M"

[models.speculative]
strategy = "mtp"
ngram_min = 2
ngram_max = 4
ngram_max_proposal_tokens = 6
extension_max_tokens = 6
verify_window_min_tokens = 1
verify_window_max_tokens = 6
verify_window_pipeline_depth = 2
```

`ngram_min` and `ngram_max` determine the history match length.
`ngram_max_proposal_tokens` is separately the maximum continuation length.
The request-local cache is limited to `ngram_max <= 4`. N-gram settings may run
standalone or, with native MTP, form one composite proposal. All combinations
are verified together by the target, so tuning these values changes speculative
work, not output correctness.

The `suffix` proposer is a pure-Rust longest-suffix matcher
("prompt-lookup decoding"). Unlike `cache` it is not bound by
llama.cpp's 4-token match window, so it can match long verbatim repeats in the
context (up to `ngram_max <= 64`) and copy long, high-confidence drafts. It is
designed for input-grounded, repetitive workloads — re-emitting a file with a
small edit, echoed tool output, repeated identifiers — and stays silent when no
sufficiently long match exists. Benchmark the target workload before assuming
an uplift or neutrality on freeform prose. `ngram_min` is the minimum verbatim
match length before it drafts; draft length scales with match length up to
`ngram_max_proposal_tokens`.

```toml
[models.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
extension_max_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
```

Suffix can also run without MTP by setting `strategy = "ngram-suffix"` and
omitting the extension controls. Layer packages may declare `ngram-suffix` as
a request-local proposer and standalone strategy. See
[Suffix N-gram Proposer](skippy/SUFFIX_NGRAM_PROPOSER.md) for the lookup
contract, telemetry, and benchmark requirements.

For package-authoring rules, see
[Layer Package Repositories](specs/layer-package-repos.md#generation-defaults).
For strategy diagrams, CLI overrides, and the VerifyWindow telemetry used to
evaluate a configuration, see
[Pipelined VerifyWindow Decode](skippy/PIPELINED_VERIFY_WINDOW.md).

## Share an LLM server you already run

If Ollama, LM Studio, LiteLLM, vLLM, or any other OpenAI-compatible server is
already running, publish its models to your mesh with one command:

```bash
mesh-llm share http://localhost:11434
```

The upstream server keeps running the models. Mesh forwards requests to it and
advertises its models under their own names. This node does not install a
plugin, does not load a native inference runtime, does not load a model of its
own, and does not rewrite your config file.

Like `serve`, this starts a **private** mesh. Make the choice explicit to go
further:

```bash
mesh-llm share http://localhost:11434 --join <token>
mesh-llm share http://localhost:11434 --publish
```

Check what is being served:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

No native inference runtime is required, so this works on a machine with no
supported GPU.

Current limits:

- One HTTP OpenAI-compatible upstream per node, fixed for the run. To change
  it, stop the command and start it again with the new URL.
- `https://` upstreams are rejected, so authenticated cloud providers are not
  supported yet.
- Ctrl-C stops sharing. It does not stop the upstream server, which keeps
  running and serving its own clients.

For a provider you want recorded durably in `config.toml` and restored on every
start, use the `openai-endpoint` plugin instead — see
[plugins/README.md](plugins/README.md).

## Lemonade integration

Use the `openai-endpoint` plugin to route requests to a local [Lemonade Server](https://lemonade-server.ai) through the same `http://localhost:9337/v1` API that mesh-llm exposes.

Start Lemonade first, either with the Lemonade Desktop app or with the CLI:

```bash
lemonade-server serve
curl -s http://localhost:8000/api/v1/models | jq '.data[].id'
```

Install the plugin:

```bash
mesh-llm plugins install openai-endpoint
```

You can also install directly from GitHub:

```bash
mesh-llm plugins install Mesh-LLM/openai-endpoint
```

Then enable the plugin in `~/.mesh-llm/config.toml`:

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "openai-endpoint"
url = "http://localhost:8000/api/v1"
```

Plugins that declare a host-projected web UI may independently disable that
console projection while leaving the plugin process and endpoint behavior
enabled:

```toml
[[plugin]]
name = "example-plugin"
enabled = true
web_ui_enabled = false
```

`web_ui_enabled` is meaningful only for a plugin that declares a web UI. It
does not install, start, stop, or disable the plugin process.

If you are running the plugin binary yourself instead of using
`mesh-llm plugins install`, set `command = "openai-endpoint"` in the same
plugin block.

Start mesh-llm normally:

```bash
mesh-llm serve
```

No `[[models]]` entry or placeholder local model is required. `on_demand`
prevents any configured local models from loading eagerly while preserving the
ability to load one later.

After startup, mesh-llm should include Lemonade-hosted models in its own model list:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Requests sent to mesh-llm with a Lemonade model ID are forwarded to Lemonade:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-GGUF",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

Notes:

- mesh-llm does not start or supervise Lemonade; run it separately with the Desktop app or CLI.
- Use the exact model ID returned by Lemonade's `/api/v1/models`.
- mesh-llm passes the configured URL to the plugin through `MESH_LLM_PLUGIN_URL`.
- Plugin-process health and Lemonade endpoint health are separate; verify both
  before reinstalling the plugin.

Useful model commands:

```bash
mesh-llm models recommended
mesh-llm models installed
mesh-llm models search qwen 8b
mesh-llm models search --catalog qwen
mesh-llm models show Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models download Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates Qwen/Qwen3-8B-GGUF
mesh-llm models cleanup
mesh-llm models prune
```

## Model storage

- Hugging Face repo snapshots are the canonical managed model store.
- Managed model scans use Hugging Face repo snapshots.
- Arbitrary local GGUF files still work through `mesh-llm serve --gguf`.
- Skippy materialized stage GGUFs are derived cache and can be preview-pruned
  with `mesh-llm models prune`.

Model downloads validate the Hugging Face Hub cache and Xet working cache
before worker threads start. If either location is read-only, mesh-llm warns
with the original operating-system error and uses a writable application-data
directory instead. `MESH_LLM_DATA_DIR` chooses that fallback root;
`HF_HUB_CACHE` and `HF_XET_CACHE` configure the two caches directly.

## Inspect local GPUs

```bash
mesh-llm gpus
mesh-llm gpus --json
mesh-llm gpus detect --json
```

This prints the local GPU inventory with stable IDs, backend device names, VRAM, unified-memory status, and cached bandwidth when a benchmark fingerprint is already present. Add `--json` for machine-readable inventory output, or run `mesh-llm gpus detect --json` to refresh the cached fingerprint and print the benchmark summary as JSON.

## Local runtime control

Local hot load and unload target the running daemon on this machine.

```bash
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
mesh-llm runtime guardrails --mode enforce --port 3131
```

Management API endpoints:

```bash
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
curl -X POST localhost:3131/api/runtime/mesh-guardrails \
  -H 'Content-Type: application/json' \
  -d '{"mode":"enforce"}'
curl -s localhost:3131/api/status | jq '.runtime.openai_guardrails'
```

The guardrail mode update is also node-local. It changes the shared
server-side `GuardrailPolicy.mode` without restarting the process, so existing
hosted Skippy backends and future local runtime loads observe the new mode.
Single-owner remote load, ensure, unload, and drain are available through the
explicitly targeted owner-control commands below. Autonomous mesh-wide
placement and rebalancing remain future work; owner-control is not a public
mesh-wide load/unload mechanism.

## Owner-control plane

Owner-control is the private operator lane for commands directed at exactly one
owner-attested node. It does **not** replace the public mesh plane used for
join, gossip, routing, or inference. Config and inventory mutation are
exclusive to `mesh-llm-control/1`; the old mesh-plane config stream IDs are
reserved but no longer carry protobuf request/response handling.

`scan-refresh` is the first public owned-node command. It asks the explicitly
targeted remote node to rescan its managed model inventory, republishes the
model names from that exact scan, and returns the refreshed inventory to the
requester. The compatible protobuf operation remains named
`refresh_inventory` on the wire.

### Bootstrap contract

- New control clients need an explicit owner-control endpoint token. The token
  identifies and cryptographically pins one target; it is not inferred from a
  peer ID, public gossip, Nostr, routing state, or `/api/status`.
- Read a target node's local bootstrap policy from
  `GET /api/runtime/control-bootstrap` or
  `mesh-llm runtime bootstrap --json` on that node, then transfer the endpoint
  token to the controlling node out of band.
- The controlling node must have a valid owner key for the same owner. The
  target verifies requester ownership against the actual QUIC connection
  identity before dispatching a command.
- If no explicit endpoint is supplied, the current client contract returns `ControlEndpointRequired`.
- If an explicit endpoint is configured and fails, the client stays on owner-control and reports a structured failure. It does **not** silently fall back to mesh-plane config streams.

### Transport and fallback matrix

| Caller / target | Result |
|---|---|
| New client + explicit endpoint | Use `mesh-llm-control/1` only; no silent legacy downgrade |
| New client + no endpoint | `ControlEndpointRequired` |
| New client ↔ old node with no endpoint | `ControlEndpointRequired` by default |
| Old client + new node | Legacy mesh-plane config stream IDs are reserved but rejected as unsupported/unknown |
| Old node ↔ new node public mesh join/routing | Public mesh ALPN negotiation, gossip, and routing remain compatible; owner-control is not required for join/routing |
| Old client + old node | Unchanged old-node behavior outside this release |

### Operator commands

Inspect the local bootstrap policy:

```bash
mesh-llm runtime bootstrap --port 3131 --json
curl -s localhost:3131/api/runtime/control-bootstrap | jq .
```

Run owner-control requests through the local management API using an explicit endpoint token:

```bash
mesh-llm runtime get-config --port 3131 --endpoint '<control-endpoint>' --json
mesh-llm runtime scan-refresh --port 3131 --endpoint '<control-endpoint>'
mesh-llm runtime scan-refresh --port 3131 --endpoint '<control-endpoint>' --json
mesh-llm runtime apply-config \
  --port 3131 \
  --endpoint '<control-endpoint>' \
  --expected-revision 7 \
  --config /absolute/path/to/config.toml \
  --json
```

Owner lifecycle commands create session-only intents on the target node. They
never mutate durable config or TOML — use `apply-config` for persistent changes.

```bash
mesh-llm runtime load-model --port 3131 --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M
mesh-llm runtime unload-model --port 3131 --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M
mesh-llm runtime ensure-model --port 3131 --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M
mesh-llm runtime drain-model --port 3131 --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M
```

- `load-model`: one-shot present intent. Returns accepted lifecycle state.
- `ensure-model`: maintained present intent with bounded retry. Survives
  transient load failures for the session.
- `unload-model`: absent intent. The model is unloaded.
- `drain-model`: draining-then-absent intent. Already-admitted work finishes;
  new work is rejected. Unloads at zero in-flight or force-cancels at the
  configured drain deadline.

Legacy hosts that do not implement these commands return typed
`ControlUnsupported`, not a silent fallback to the public mesh.

Equivalent REST calls:

```bash
curl -s -X POST localhost:3131/api/runtime/control/get-config \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/scan-refresh \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .

# Compatibility alias: retains the legacy snapshot-only response shape.
curl -s -X POST localhost:3131/api/runtime/control/refresh-inventory \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/apply-config \
  -H 'Content-Type: application/json' \
  -d '{
    "endpoint":"<control-endpoint>",
    "expected_revision":7,
    "config":{"version":1}
  }' | jq .

curl -s -X POST localhost:3131/api/runtime/control/load-model \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>","model":"Qwen3-8B-Q4_K_M"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/unload-model \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>","model":"Qwen3-8B-Q4_K_M"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/ensure-model \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>","model":"Qwen3-8B-Q4_K_M"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/drain-model \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>","model":"Qwen3-8B-Q4_K_M"}' | jq .
```

The local REST facade is loopback-only. The public CLI spelling is
`runtime scan-refresh`; the old `runtime refresh-inventory` spelling remains a
hidden compatibility alias and continues to return the legacy config snapshot.
Neither facade discovers a target implicitly: both require `--endpoint` (or
the REST `endpoint` field).

### Scan-refresh result

The JSON response contains `target_node_id`, `disposition`, and `inventory`.
Inventory entries are sorted by `canonical_model_ref` and contain an optional
`display_name`, `total_size_bytes`, and optional compact model metadata. The
metadata includes the canonical model key plus GGUF-derived architecture,
quantization, tokenizer, dimensions, RoPE, special-token, and MoE fields when
known. `--json` prints this response unchanged; human output summarizes the
disposition, target, model count, total bytes, and sorted model references.

`disposition` is `executed` when this request performed the scan and
`coalesced` when it joined an already-running scan. Joined callers receive the
same successful inventory payload. A new client talking to an older
owner-control server may receive only the legacy wire snapshot; the command
still succeeds, but the REST fields `disposition` and `inventory` are `null`
and human output labels the result `compatibility-limited`. This does not mean
released nodes support the richer response fields.

Scan failures are returned to all joined callers and preserve the last good
inventory and model advertisements. Rich inventory results stay on the private
owner-control response path: they are not copied wholesale into peer state,
public gossip, runtime status, or `/api/status`. Endpoint tokens and raw command
results must not be logged or advertised. The node continues to publish only
its existing availability projection from a successful scan.

### Owner-control limits

- Inbound and outbound protobuf frames are limited to 8 MiB. An oversized
  generated response becomes `ControlUnavailable` before any oversized body is
  written.
- Client connect, stream-open, handshake, and request-write waits are bounded
  at 8 seconds, 2 seconds, 2 seconds, and 2 seconds respectively.
- Get/apply unary responses have a 5-second bound; inventory scans have a
  30-second bound. Watch acceptance has a 5-second bound, after which an
  accepted watch remains streaming without a unary deadline.
- The server bounds handshake and request reads at 2 and 5 seconds and admits
  at most 32 concurrent owner-control stream workers per connection.
- Request IDs are non-zero. Authentication, requester binding, target binding,
  request validation, deadline selection, and response-size enforcement occur
  in the common dispatcher path.

### Failure modes

| Error | Meaning | Typical operator action |
|---|---|---|
| `ControlEndpointRequired` / `control_endpoint_required` | No explicit endpoint was supplied | Read `runtime bootstrap`, then retry with the advertised endpoint token |
| `ControlUnsupported` / `control_unsupported` | Target accepted the connection path but does not speak `mesh-llm-control/1` | Verify the endpoint token targets an owner-control listener |
| `ControlUnavailable` / `control_unavailable` | Endpoint token, listener, network path, or local owner key loading failed | Verify the endpoint token, listener status, and local owner keystore/passphrase |
| `Unauthorized` / `unauthorized` | Same-owner handshake failed | Check that both nodes use the same owner identity and that the local key can be unlocked |
| `RevisionConflict` / `revision_conflict` | Apply request used a stale `expected_revision` | Re-read config, merge, and retry with the current revision |
| `LegacyJsonUnsupported` / `legacy_json_unsupported` | A legacy mesh-plane frame hit `mesh-llm-control/1` | Fix the caller to use owner-control protobuf frames |

### Transition note

Treat owner-control as the only lane for operator config and inventory clients. Legacy mesh-plane config stream IDs remain reserved for compatibility bookkeeping, but current nodes do not handle config subscribe/push requests on `mesh-llm/1`.

### Mixed-version QA harness

Use the task harness when you need executable evidence for mixed-version routing or owner-control bootstrap:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence
```

Loopback-only routing/owner-control smoke:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence \
  --local-only
```

Owner-control bootstrap lane only:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence \
  --local-only \
  --config-only
```

To validate the harness contract without starting processes or writing evidence, add `--print-plan`; it prints the planned public, loopback, and owner-control result names as JSON.

`--config-only` skips public-mesh probes and focuses on the owner-control migration lane:

- loopback released/current private-mesh coexistence in both directions
- current-branch proof that new clients prefer `mesh-llm-control/1`
- current-branch proof that missing endpoints fail with `ControlEndpointRequired`
- current-node `runtime bootstrap` / `runtime get-config` evidence when owner-control is enabled

Each real run writes a timestamped evidence directory with `manifest.json`, `commands.jsonl`, `results.jsonl`, `summary.md`, `summary.json`, `versions/*.txt`, process logs, and grouped status/model/chat/control payloads.

If the local bootstrap payload reports `enabled=false`, the harness records a `PREREQ` result explaining that a signed same-owner keystore is required before runtime owner-control requests can be proven on that machine. That is an explicit prerequisite report, not a silent pass.
