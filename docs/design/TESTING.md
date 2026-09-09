# Testing mesh-llm

## Local inspection

### Process-environment mutation contract

Rust 2024 makes `std::env::set_var` and `std::env::remove_var` unsafe on
platforms where another thread could read the process environment. Tests that
need an environment override therefore follow one contract across the
workspace:

- mark the owning test `#[serial]` (from `serial_test`);
- snapshot and restore each key with a semantically owned guard/helper in that
  crate; and
- keep a `// SAFETY:` comment beside every mutation explaining the serialized
  scope.

The crates intentionally do not share a Rust dev-dependency just for this
test helper. Build scripts have an isolated process boundary; runtime startup
code retains explicit safety notes and audit TODOs until its single-threaded
ordering is proven. The repository-level census check covers
all 17 files from the original 128-comment audit and rejects a test mutation
that is not covered by `#[serial]`:

```bash
just check-env-mutation-contract
```

The same check runs through `scripts/tests` in the normal CI validation path.

### 0. Inspect local GPUs

```bash
mesh-llm gpus
mesh-llm gpus --json | jq .
mesh-llm gpus detect --json | jq .
```

- Prints local runtime-selectable GPU entries with stable IDs, backend devices, VRAM, unified-memory status, and cached bandwidth when a fingerprint is available
- In the shipped Skippy-enabled binary, platform tools alone are not enough: if the embedded backend does not enumerate a GPU, the node should report CPU-only rather than advertising probe-visible GPU capacity
- `--json` emits machine-readable inventory and benchmark payloads suitable for automation

### 0a. Startup config smoke

Create `~/.mesh-llm/config.toml`:

```toml
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen2.5-3B"

[[models]]
model = "/absolute/path/to/qwen2.5-vl.gguf"
mmproj = "/absolute/path/to/mmproj.gguf"
ctx_size = 8192
```

Then start:

```bash
mesh-llm serve
```

- Both configured startup models should be considered for launch
- If `[[models]]` is empty, `mesh-llm serve` should print a `⚠️` warning, show help, and exit cleanly
- Explicit `--model` or `--gguf` should ignore configured `[[models]]` for
  model selection and tuning, except that an exact, unique `--model` ref may
  inherit its configured pinned GPU selector
- Explicit `--ctx-size` should override configured `ctx_size`
- Explicit `--device` should override persisted device selectors under pinned
  or automatic assignment. Backend names resolve to a detected device, `CPU`
  bypasses GPU-only preflight, and `auto` retains the inherited selector.
- `mesh-llm benchmark tune` is the measured local model-serving tuning companion for these startup configs. It only accepts already-downloaded targets, rejects remote-only or not-downloaded refs without fetching them, and runs isolated throughput trials. For speculative decoding changes, run a small sweep that includes the disabled baseline plus `mtp`, `mtp-ngram`, or draft candidates as applicable, then inspect trial logs/telemetry for native MTP or draft acceptance statistics in addition to decode tok/s.

### 0b. Pinned startup smoke

First inspect the valid local IDs:

```bash
mesh-llm gpus
mesh-llm gpus --json | jq .
```

Then create `~/.mesh-llm/config.toml` with a real pinnable stable ID from that output (for example `pci:*`, `uuid:*`, or `metal:*`, not fallback IDs like `index:*` or backend-device names):

```toml
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen2.5-3B"
gpu_id = "pci:0000:65:00.0"
```

Start the node:

```bash
mesh-llm serve
```

- Startup should succeed only when `gpu_id` matches a valid local pinnable stable ID from `mesh-llm gpus`
- If the pinned ID is missing, ambiguous, unsupported, or stale, startup should fail closed before local launch
- Explicit `mesh-llm serve --model ...` keeps the CLI model path, context, and
  projector choices. When the ref exactly matches one configured model, only
  that model's effective pinned GPU selector is carried forward and resolved
  from its stable ID to the backend device name. Unmatched refs and all
  `--gguf` paths carry no configured model identity but may inherit only
  `defaults.hardware.device`; duplicate configured refs are rejected as
  ambiguous because the CLI has no profile selector.
- Do not use GPU indexes, `index:*`, or backend-device names like `CUDA0` / `HIP0` / `MTL0` as `gpu_id`
- Backend-device names are accepted only through the explicit CLI `--device`
  override. Persisted `gpu_id` values remain stable IDs.

### 0c. Requirement-aware mesh smoke

Create an unrestricted mesh:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M
```

Create a release-attestation-required public mesh:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M --publish \
  --require-release-attestation \
  --release-signer-key ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --owner-key ~/.mesh-llm/owner-keystore.json \
  --owner-required \
  --trust-policy require-owned \
  --node-label lab-a
```

The release-attestation flags are creation-time mesh requirements. The owner and
trust-policy flags exercise local owner-identity behavior and must not change
the mesh requirements hash.

Or with config-backed creation requirements:

```toml
[mesh_requirements]
require_release_attestation = true
release_signer_keys = ["ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"]
```

Join via signed bootstrap token:

```bash
mesh-llm serve --join <signed-bootstrap-token>
```

- `mesh-llm runtime bootstrap --port 3131` should show the local owner-control
  bootstrap policy used by the node.
- `curl -s http://localhost:3131/api/runtime/control-bootstrap | jq .` should
  report the same local bootstrap policy in JSON form.
- Admission failures for nodes that miss the certified-build gate should be
  deterministic. In human-facing prose this is "certified build required";
  the machine reason codes surfaced by logs, status, and evidence are
  `certified_binary_required`, `build_proof_invalid`, and
  `release_signer_untrusted`.
- Legacy unrestricted meshes still accept the older unsigned invite-token path,
  while requirement-aware meshes require signed bootstrap tokens.

- For release smoke, always inspect the packaged `mesh-llm` executable with `cargo run -p xtask -- release-attestation inspect --binary /tmp/test-bundle/mesh-llm --public-key-file /tmp/mesh-release-key.pub`, never the raw `target/release/mesh-llm` path. Packaged release archives can report `valid`, unstamped local or dev builds report `missing`, and a binary mutated after download reports `invalid`, but default startup still allows it because this is provenance and admission hardening, not runtime integrity proof. Bare `inspect --binary ...` is only sufficient for unstamped binaries that should classify as `missing`; stamped release binaries require `--public-key-file` and otherwise report `invalid` with an explicit error.
- Smoke and release evidence should come from the packaged archive contents, not raw release output, because the release pipeline now sources publishable binaries from extracted packaged archives.

### 0d. Terminal dashboard smoke

The pretty dashboard uses raw mode and the alternate screen when both stdin and stderr are interactive TTYs and `TERM` supports a real terminal. It should leave native terminal text selection available and fall back to line-oriented pretty output when stdin is not a TTY, stderr is not a TTY, or `TERM` is empty / `dumb`.

Run these manual checks after changes to `runtime/interactive.rs` or `cli/output/mod.rs`:

| Shell | Setup | Expected result |
|---|---|---|
| Plain terminal | `mesh-llm serve --model Qwen2.5-3B` | Dashboard renders, resizes cleanly, `Tab`/`Shift-Tab` focus panels, `Enter` or `z` opens the focused panel full screen, `Esc` or `z` returns to the multi-panel view, terminal text selection works, `h`/`i`/`q` work, and exit restores the prompt. |
| Piped stdin | `true | mesh-llm serve --model Qwen2.5-3B` | No line reader is spawned; pretty output stays line-oriented. |
| Unsupported terminal | `TERM=dumb mesh-llm serve --model Qwen2.5-3B` | Dashboard is disabled and pretty output uses fallback lines. |
| tmux, mouse off | `tmux new 'mesh-llm serve --model Qwen2.5-3B'` | Dashboard renders and exits cleanly; keyboard navigation works. |
| tmux, mouse on | Inside tmux: `set -g mouse on`, then run mesh-llm | Dashboard renders and exits cleanly; terminal/tmux text selection remains usable. |
| GNU screen default | `screen mesh-llm serve --model Qwen2.5-3B` | If the alternate screen is unavailable, fallback behavior or clean restoration is acceptable. |
| GNU screen altscreen | In `~/.screenrc`: `altscreen on`, then run mesh-llm | Dashboard enters/leaves the alternate screen cleanly. |

For terminal restoration QA:

- Resize during startup, after embedded runtime readiness, and while the dashboard has focus on different panels.
- Open the mesh events/log panel full screen and verify long log lines wrap within the panel.
- Detach and reattach tmux/screen while the dashboard is active.
- Select visible dashboard text with the terminal mouse selection gesture and verify it can be copied.
- Press `q` and `Ctrl+C`; the cursor should be visible and the shell prompt should not remain in raw mode.
- A `SIGKILL` (`kill -9`) cannot run in-process cleanup. If a terminal is left corrupted after a hard kill, recover with `reset` or by closing the terminal pane.

### 0d. Agent tool-call reliability

Run the direct tool-call probe when changing OpenAI chat-completions routing,
agent integrations, MoA reducer behavior, or anything that may affect
`tools` / `tool_calls` / tool-result continuation:

```bash
scripts/qa-agent-tool-call-reliability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models auto,mesh \
  --attempts 3 \
  --output target/agent-tool-call-reliability/results.jsonl
```

- `tool_call` and `stream_tool_call` require real OpenAI-style function calls,
  not prose that says a tool would be used
- `tool_result` and `stream_tool_result` verify that the assistant can continue
  after a matching tool result and include the deterministic fixture value
- `--print-plan` is side-effect-free and emits machine-readable JSON
- use `--skip-streaming` only when intentionally narrowing a local diagnosis to
  non-streaming chat-completions

### 0e. Nightly stability harness

Use the repeatable stability harness when you want black-box evidence that a
live mesh endpoint stays usable across repeated chat, streaming, tool-call, and
optional agent-client checks:

```bash
scripts/qa-nightly-stability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models auto,mesh \
  --attempts 5 \
  --agent-smokes opencode,pi,goose \
  --output-dir target/nightly-stability/local
```

- the harness attaches to an existing OpenAI-compatible `/v1` endpoint; it does
  not start nodes, load models, publish to the public mesh, or change routing
  policy
- direct OpenAI surface probes write `results.jsonl` with `/v1/models`,
  non-streaming chat, streaming chat, HTTP status, elapsed time, and TTFT where
  applicable
- the merged tool-call probe remains the canonical tool-call validator; this
  harness invokes it and records its command/log path instead of reimplementing
  tool-call parsing
- optional OpenCode, Pi, and Goose smokes reuse the existing CI agent fixtures;
  missing optional CLIs are recorded as `PREREQ` rather than a stability failure
- every run writes `manifest.json`, `commands.jsonl`, `results.jsonl`,
  `summary.json`, `summary.md`, and `logs/`
- `--print-plan` is side-effect-free and shows the exact checks/artifacts that
  would run

The scheduled GitHub workflow is opt-in through
`MESH_NIGHTLY_STABILITY_ENABLED=1` and a configured
`MESH_NIGHTLY_STABILITY_BASE_URL` (or the existing agent endpoint variables).
The scheduled/manual wrapper calls the reusable `nightly-stability-run.yml`
workflow so maintainers can reuse the same harness execution from other
workflows or lab jobs. The job summary includes the timing snapshot from
`summary.md`, so day-over-day drift can be checked without opening JSONL
artifacts. The reusable workflow is fixed to GitHub-hosted Ubuntu; it never
accepts caller-provided runner labels because it checks out the caller's
repository content. It is intentionally evidence-producing and non-required:
failed nightlies should guide stabilization work, not block unrelated pull
requests.

The reusable run also invokes `qa-kv-tool-loop-stability.py` by default. Use
`MESH_NIGHTLY_KV_MODELS` to select the direct model IDs and the bounded
`MESH_NIGHTLY_KV_{ATTEMPTS,PRESSURE_TURNS,OVERLAP_REQUESTS,MIN_CACHED_TOKENS,SUFFIX_PREFILL_LIMIT}`
variables to tune the live probe. A manual run may set `skip_kv_tool_loop` for
a deliberately narrower diagnosis. Both harnesses run to completion, publish
their summaries and evidence, and then preserve either failure as the job
result.

### 0f. KV/tool-loop stability certification

Run the KV/tool-loop certification probe when changing Skippy KV slot cleanup,
prefix-cache lookup, agent tool-loop behavior, or any runtime path related to
issues where repeated tool calls eventually hit `llama_decode failed` or low
same-prefix cache reuse.

```bash
scripts/qa-kv-tool-loop-stability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m \
  --attempts 5 \
  --pressure-turns 8 \
  --timeout 180 \
  --min-cached-tokens 2048 \
  --suffix-prefill-limit 256 \
  --native-log ~/.mesh-llm/runtime/<pid>/logs/skippy-native.log \
  --output-dir target/kv-tool-loop-stability/local
```

- the probe attaches to an existing `/v1/chat/completions` endpoint; it does
  not start nodes, load models, join a mesh, or change routing policy
- each attempt runs a growing non-streaming tool-result conversation with a
  long stable system prefix, repeated pressure turns, a second forced tool
  call, and final recall of both deterministic tool facts
- `same_prefix_cache` warms the same long prefix with one tail and measures a
  different tail; `exact_prefix_cache` verifies the identical-body cache path
  still works
- `--min-cached-tokens 2048` matches the known Qwen 3B reproduction shape where
  healthy same-prefix reuse is near the shared prefix, not the 256-token floor
- `--native-log` scans Skippy native logs for `failed to find a memory slot`,
  `llama_decode failed`, and proactive eviction errors appended after the
  certification starts, so stale failures in long-lived logs do not fail a
  clean run
- every run writes `manifest.json`, `results.jsonl`, `summary.json`,
  `summary.md`, and sanitized transcript JSONL under `transcripts/`; the
  transcript directory is reset at run start to keep repeated `latest` runs
  auditable
- `--print-plan` is side-effect-free and emits the exact models, thresholds,
  runtime options, checks, and evidence files that would be produced

This certification is deliberately a lab/release-confidence check, not a
required PR gate. Use it to prove KV/cache stability on a real direct-model
endpoint after local unit tests and before relying on agent workloads such as
Goose, Pi, or OpenCode for broad smoke coverage.

The separate `nightly-kv-coverage.yml` schedule expands the deterministic
radix-lease and blob-ownership state machines on a pinned public CPU image. It
records exact seed and step budgets plus the source SHA, and preserves the
seed/step trace in its uploaded log. Repository variables
`MESH_NIGHTLY_KV_STATE_MACHINE_SEEDS` and
`MESH_NIGHTLY_KV_STATE_MACHINE_STEPS` may raise or lower the bounded corpus;
set `MESH_NIGHTLY_KV_COVERAGE_ENABLED=0` to disable the scheduled run. Manual
dispatch still executes trusted `main` on GitHub-hosted infrastructure.

The unchanged-pin daily llama canary uses the `nightly` cadence in
`ci/llama-canary/family-certified.json`: Qwen3 dense, Falcon-H1 hybrid,
Qwen3Next composite, and Mamba recurrent. Llama bumps and explicit forced
certification retain the full 33-family battery.

### 0g. Logging workflow certification

Use the request logging checks after changing the trusted-local logging service,
the Logs console, logging configuration, terminal projection, or its privacy
boundary. Run the UI gates serially from the UI crate:

```bash
pnpm run lint
pnpm run typecheck
pnpm run test
pnpm run build
pnpm run test:e2e
```

The browser suite must use the embedded console routes and exercise the real
typed logging client paths. Its deterministic contract fixture covers an active
request reaching a terminal outcome, immediate details, stream filtering,
replay-gap recovery with a `null` cursor, bounded polling fallback, explicit
download of a redacted artifact, missing artifacts, metadata-only export,
previewed cleanup with an audit receipt, validated manual webhook retry input,
and an older host that reports the logs API as unsupported and stays inert. The
dedicated stream repeats the backend-supported ledger filters `from`, `to`,
`model`, `provider`, `engine`, `route`, and `outcome`; `source` stays REST-only
and is never serialized as an SSE filter. The focused hook/schema/component
tests additionally cover ledger-filter reopen with the last cursor, both
omitted and `null` recovery cursors, and success-only maintenance refresh: only
`completed` or `partial` cleanup/delete receipts refetch the active ledger. A
partial receipt with failed artifact deletion work retries with its frozen
operation ID and audit reason, while previews and failed mutations do not
refresh. Dead-letter retry requires a manually entered, validated delivery ID;
it must not claim a request-details delivery context. It must not use the
existing status or runtime SSE stream as a substitute.

For a live-host certification, run the matching workflow against an isolated
trusted-local host after the normal build succeeds:

```bash
just build
```

Verify that an active request reaches exactly one terminal outcome, detail is
available immediately, and the dedicated stream reconnects. Then force a
replay gap and an unavailable stream, confirm the authoritative ledger refresh
and bounded polling indication, review redacted/missing artifact states, and
perform a metadata-only export plus a preview-and-confirm cleanup using a
meaningful audit reason. Verify that only a returned `completed` or `partial`
maintenance receipt refreshes the active ledger and that a partial artifact
deletion result retries with the same operation ID and audit reason. Test a
host without the logs API separately and expect the console's accessible
unsupported state. Do not place prompts, completions, credentials, artifact
bodies, raw identifiers, local paths, or URL query data in certification
evidence.

Visual and accessibility certification covers the Logs list and details pages
at 375, 768, and 1280 CSS pixels in light and dark themes. Check no horizontal
overflow or clipped controls, keyboard-only list-to-details and modal focus
return, reduced-motion behavior, and an axe scan. The dedicated Logs axe suite
also runs all 14 theme/accent combinations (two themes × seven accent modes),
including active and terminal status badges. A clean certification has no
serious or critical violations. If a shared UI finding is already present,
preserve the exact axe output, identify the affected shared token or primitive,
and record the result as prerequisite-limited; do not disable the rule or
represent the run as clean. Browser screenshots assist visual review but do not
replace the functional assertions above.

Report the two results separately: a functional E2E pass establishes the
lifecycle, recovery, operations, and unsupported-host contract only; it is not
a waiver for a serious or critical accessibility finding. Until the shared
finding is fixed, record the functional result as passed and the visual and
accessibility result as prerequisite-limited, including the exact affected
route, theme, viewport, selector, and axe rule in the certification evidence.

## Single-model permutations

### 1. Solo (single node)

```bash
mesh-llm serve --model Qwen2.5-3B --console 3131
```

- API on `:9337`, console on `:3131`
- Console: `host=true, peers=0`
- Embedded runtime reports one local serving route for the model

### 1a. Headless mode (API-only, no embedded UI)

```bash
mesh-llm serve --model Qwen2.5-3B --headless --console 3131
```

- API on `:9337`, management API on `:3131`
- `GET /api/status` returns 200 with normal JSON
- `GET /` returns 404 (web console routes are disabled)
- `GET /dashboard`, `GET /chat`, and `/assets/*` also return 404
- The management API (`/api/*`) remains fully accessible

This mode is intended for headless server deployments where the embedded web UI is not needed.

### 2. Two GPU nodes, model fits on host

```bash
# node A (more VRAM, becomes host)
mesh-llm serve --model Qwen2.5-32B --bind-port 7842
# node B (joins)
mesh-llm serve --model Qwen2.5-32B --join <TOKEN>
```

- Both nodes run solo (no split) — each is its own host
- API works from both nodes on `:9337`

### 3. Two GPU nodes, forced split

```bash
# node A (coordinator)
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --bind-port 7842 \
  --port 9447 \
  --console 3232

# node B (worker)
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --join <TOKEN> \
  --bind-port 7843 \
  --port 9447 \
  --console 3232
```

- `--split` forces staged execution even when the model fits on the host
- Use a published layer package (`*-layers`) so each node downloads only the
  shared artifacts and layer files assigned to its stage.
- Embedded runtime assigns participating stage routes
- Stage placement is proportional to available VRAM (e.g. `0.67,0.33`)
- If `--max-vram` is too low, planning should fail before startup with the
  required memory estimate. In lab testing, `3` GB was too low for this Qwen3
  8B package at the default context, while `5` GB per node reached readiness.
- Wait for `/api/status` on the coordinator to show `node_state="serving"`,
  `llama_ready=true`, a ready runtime stage, and one peer.
- `GET /v1/models` should list the full layer-package model id.
- A short `/v1/chat/completions` request should return an OpenAI-shaped
  response from the layer-package model.

> **CI coverage:** `two_node_split_smoke` runs
> `scripts/ci-two-node-split-smoke.sh` against the Linux inference binary in two
> model lanes: dense SmolLM2-135M and recurrent Qwen3.5-0.8B. Each lane starts
> two serving nodes (the recurrent lane fixes a 4096-token context), waits for
> a topology with stages on two distinct nodes, checks `/v1/models`, then sends
> three progressively longer `/v1/chat/completions` prompts with one shared
> prefix through stage 0. The smoke requires the reported cached-token count to
> increase after each request so either model-state path cannot silently fall
> back to cold prefill for prefix matches. A follow-up request that restores
> nothing is the one outcome a loaded runner can produce without a regression,
> because the host answers before the stage lane releases; the smoke retries the
> whole sequence from a fresh cold prefix up to
> `MESH_TWO_NODE_SPLIT_PREFIX_ATTEMPTS` times (3 by default) for that case only.
> Reuse that is present but not growing fails immediately.
>
> Other nearby CI coverage:
>
> - `scripts/ci-two-node-client-serving-smoke.sh` — two nodes, but only tests
>   `client` -> `serve` routing. The model is held entirely on one node.
> - `scripts/skippy-ci-smoke.sh` — exercises 3-stage layer splits via
>   `skippy-correctness chain`, but all stages run on `127.0.0.1` in a single
>   runner. It validates the staged-runtime ABI, not mesh-llm node-to-node
>   split serving over QUIC.
>
> Use real flags only: `--split`, `--max-vram`, `--join`, `--bind-port`,
> `--port`, `--console`. There is no `--layer-range` flag — layer placement
> is normally decided by the runtime from advertised VRAM. Maintainer runs that
> require exact placement can supply a complete topology manifest with
> `--split-topology-lock <path>`; see `docs/SKIPPY_SPLITS.md`.

#### Multi-interface Docker/Linux bind-IP validation

For host-network Docker or multi-NIC Linux hosts, validate the selected
host-to-host interface explicitly:

```bash
# seed on the routable management IP
mesh-llm serve --model Qwen2.5-32B --split --bind-ip 10.1.2.3 --bind-port 7842

# worker joins the printed token
mesh-llm serve --model Qwen2.5-32B --split --join <TOKEN>
```

- Decode the invite token and confirm Docker/CNI bridge addresses such as
  `172.17.0.1` or `172.23.0.1` are absent when `--bind-ip` is set.
- The seed sees the worker in `/api/status`; it must not stay at
  `Waiting for peers...`.
- `/v1/models` becomes non-empty after split startup and inference works.
- `--listen-all` is not a substitute for this test; it only affects local
  HTTP API/console listeners.

#### Split-package preflight diagnostics

Before starting a package-backed split run, preflight the local package
directory and then certify the immutable published ref:

```bash
skippy-model-package preflight ./model-package --stages 2 --verify-sha256
mesh-llm models certify hf://namespace/repo@revision --package-only --report-out target/skippy-preflight/cert.json
```

Clean packages should pass with a machine-readable report. Deliberately broken
packages or refs should fail before the model is advertised through `/v1/models`
and should name the blocked manifest field, missing artifact, size/SHA mismatch,
sidecar, or stage part.

### 4. Two GPU nodes, model too big for one

When the model exceeds host VRAM, split happens automatically without `--split`.

### 5. Lite client (no GPU)

```bash
mesh-llm client --join <TOKEN> --port 9555
```

- Uses ephemeral key (unique identity, works on same machine as GPU node)
- `/v1/models` lists all served models from gossip
- API tunneled to correct host per model via QUIC
- VRAM total excludes client

## Multi-model permutations

### 6. Two nodes, different models

```bash
# node A: seeds mesh with two models, serves 3B
mesh-llm serve --model Qwen2.5-3B --model GLM-4.7-Flash --console 3131
# node B: joins, auto-assigned to GLM (needed, on disk)
mesh-llm serve --join <TOKEN>
```

- `/v1/models` on either node lists both models
- Requesting GLM from node A routes via QUIC to node B
- Requesting 3B from node B routes via QUIC to node A
- Both run solo (no tensor split)
- Console shows both models warm with node counts

Compatibility result:
- Verified on 2026-04-02 with the current `codex/model-identity-design` branch on node 1 and the latest GitHub release `v0.54.0` on node 2.
- Node 1 served `Llama-3.2-1B-Instruct-Q4_K_M`; node 2 served `Qwen3-4B-Q4_K_M`.
- `/api/models` and `/v1/models` agreed on the same warm model list from both nodes.
- Chat from node 1 to node 2's model succeeded, and chat from node 2 to node 1's model succeeded.

### 7. Auto-assignment

```bash
# seeder declares two models
mesh-llm serve --model Qwen2.5-3B --model GLM-4.7-Flash
# joiner with no --model
mesh-llm serve --join <TOKEN>
```

- Joiner scans the Hugging Face cache and picks an unserved model already on disk
- Log: "Selected to serve GLM-4.7-Flash (needed by mesh, already on disk)"

### 8. Lite client with multi-model

```bash
# GPU nodes running as above
mesh-llm client --join <TOKEN> --port 9555
```

- Client sees all models via gossip (ephemeral key = unique identity)
- `/v1/models` lists all served models
- Routes to correct host per model
- Streaming works through cross-model QUIC tunnel

### 9. Unload a model

```bash
mesh-llm unload GLM-4.7-Flash-Q4_K_M
```

- Node serving that model exits cleanly
- Other nodes unaffected
- Model goes cold in console

### 9a. Local runtime load/unload and local status view

```bash
# Running node
mesh-llm serve --model Qwen2.5-0.5B-Instruct-Q4_K_M --console 3131

# Operator surface
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M

# REST surface
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

- `mesh-llm status` shows the local models currently backed by running inference processes, including PID when present
- `GET /api/runtime` and `GET /api/runtime/processes` agree with the CLI output
- Loading a small local model adds it to `/v1/models` without restarting the node
- Unloading any local model removes it cleanly without terminating the mesh-llm process

### 10. Console model picker

- Dropdown appears when >1 warm model
- Switching models highlights the serving node in topology view
- Chat routes to selected model via API proxy

### 11. Console live-state and wakeable capacity

```bash
cd crates/mesh-llm-ui/
npm run test:run
npm run typecheck
just build
```

- Live badges show only `Client`, `Standby`, `Loading`, and `Serving`
- Wakeable capacity renders in a separate section from topology peers and live nodes
- Wakeable entries do not appear in the topology peer list
- Validation uses `npm run test:run`, `npm run typecheck`, and `just build`
- `just build` must leave `target/debug/mesh-llm` with exactly one adjacent
  `target/debug/native-runtimes/<runtime-id>/` tree. Validate it with
  `./target/debug/mesh-llm --log-format json runtime list`; CI also starts the
  composed client noninteractively, observes either the JSON `Client ready`
  message or the structured `passive_mode`/`status=ready`/`role=client` event,
  and requires bounded graceful SIGTERM shutdown on Unix or CTRL_BREAK_EVENT
  on Windows. Interactive Ctrl-C/SIGINT behavior is tested separately because
  noninteractive background shell children may inherit SIGINT as ignored.

## Mesh Identity

### 16. Mesh ID generation (originator)

```bash
# With --mesh-name (deterministic ID)
mesh-llm serve --model Qwen2.5-3B --mesh-name "test-mesh"
```

- Log: `📌 Mesh ID: <hex>`
- `~/.mesh-llm/last-mesh` contains the same hex
- Restart with same `--mesh-name` → same mesh ID (deterministic)
- Different `--mesh-name` → different mesh ID

### 17. Mesh ID propagation (joiner)

```bash
# Originator
mesh-llm serve --model Qwen2.5-3B --mesh-name "test-mesh"
# Joiner
mesh-llm serve --model Qwen2.5-3B --join <TOKEN>
```

- Joiner log: `📌 Mesh ID: <same hex as originator>`
- Joiner's `~/.mesh-llm/last-mesh` matches originator's mesh ID
- Mesh ID arrives via gossip (worker nodes) or routing table (passive clients)

### 18. Sticky mesh preference

- Join a mesh → `~/.mesh-llm/last-mesh` saved
- On next `--auto`, `score_mesh()` adds +500 for meshes with matching `mesh_id`
- If that mesh is dead (not on Nostr), scoring proceeds normally without bonus

## Bootstrap Proxy

### 19. Instant API during GPU bootstrap

```bash
# Originator (already running)
mesh-llm serve --model Qwen2.5-3B --port 8090
# Joiner
mesh-llm serve --model Qwen2.5-3B --join <TOKEN> --port 8091
```

- Joiner log: `⚡ API ready (bootstrap): http://localhost:8091`
- BEFORE the joiner finishes its local embedded runtime startup:
  - `curl localhost:8091/v1/models` → lists mesh models
  - `curl localhost:8091/v1/chat/completions` → inference via tunnel to originator
- Log: `⚡ Bootstrap proxy handing off to full API proxy`
- After handoff, API continues working (now served locally or via election)

### 20. Bootstrap proxy not started for originator

```bash
mesh-llm serve --model Qwen2.5-3B
```

- No `⚡ API ready (bootstrap)` message (only joiners get bootstrap proxy)
- API port opens only after election resolves

## No-Arg Behavior & Management API

### 21. No-arg help

```bash
mesh-llm
```

- Prints the same usage/help text as `mesh-llm --help`
- No ports are bound
- `curl localhost:3131/api/status` fails to connect


### 22. Join and observe via console

```bash
mesh-llm client --auto
# In browser: http://localhost:3131 → observe status/discovery
# Or join explicitly from the CLI:
mesh-llm client --join <token>
```

- CLI join triggers full flow: connect → gossip → assign model → download → serve
- Console updates: status, peers, model name all reflect new state
- Inference port starts working after model loads

### 23. Management API while serving

```bash
mesh-llm serve --auto
# After serving:
curl localhost:3131/api/status   # JSON: node, peers, routing, mesh_id, mesh_name
curl localhost:3131/api/events   # SSE stream
curl 'localhost:3131/api/search?q=qwen&catalog=true&artifact=gguf&limit=5' # JSON search results
curl -X POST localhost:3131/api/model-interests \
  -H 'Content-Type: application/json' \
  -d '{"model_ref":"Qwen3-Coder-Next-Q4_K_M","source":"ui"}'
curl localhost:3131/api/model-interests
curl localhost:3131/api/model-targets
curl localhost:3131/api/discover # Nostr meshes (current mesh marked by mesh_id)
```

- `/api/status` includes `mesh_id` and `mesh_name`
- SSE events push every 2s and on topology changes
- `/api/search` returns 200 JSON with canonical model refs for matching results
- `/api/model-interests` stores and returns local explicit-interest entries keyed by canonical model refs
- `/api/model-targets` returns ranked targets with explicit-interest counts, request counts, serving-node counts, `wanted` for targets not currently served, and derived `capacity_advice` without changing ranking or routing behavior
- If `[runtime] reconcile_model_targets = true` is enabled, unserved local explicit interests that are already present on disk and fit the current node may be runtime-loaded automatically. If `reconcile_model_target_demand_upgrades = true` is also enabled, an already-serving host may replace a lower-demand local model with a locally present, higher-demand unserved target whose active demand is still within `model_target_demand_upgrade_max_age_secs`. Leave these unset for read-only advisory checks.
- Discover results can be matched to current mesh by `mesh_id`
- In `--mesh-discovery-mode mdns`, `/api/discover` must show LAN advertisements without raw invite tokens; token fingerprints are expected, while proof challenges and `/api/discovery/lan-details` should appear only when the publishing node's management API is LAN-reachable, such as with `--listen-all`.
- POST `/api/discovery/lan-details` must reject missing or wrong-token proof and must return local detail without echoing the raw invite token.

### 24. HTTP proxy single-request connection contract

- Send a routed inference request, then pipeline or reuse the same TCP
  connection for a second request.
- Verify only the first request reaches the selected upstream.
- Verify the proxy closes the routed connection after the first response.
- Verify the upstream-observed request includes `Connection: close`.

### 25. OpenAI guardrail corpus smoke

Enable the server-side guardrail mode first. Either start the runtime in that
mode or update the running process without restarting it:

```bash
mesh-llm serve --model MiniMax-M2.5-Q4_K_M --mesh-guardrails metrics
# or, for an already-running node:
mesh-llm runtime guardrails --mode metrics --port 3131
curl -s localhost:3131/api/status | jq '.runtime.openai_guardrails'
```

```bash
python3 scripts/run-openai-guardrail-corpus.py \
  --base-url http://127.0.0.1:9337/v1 \
  --model MiniMax-M2.5-Q4_K_M \
  --guardrail-mode metrics \
  --trials 20 \
  --out .sisyphus/evidence/openai-guardrail-corpus.json
```

- Phase 0 stays off by default. `metrics` and `enforce` are opt-in.
- `--guardrail-mode` only records request intent and sends the matching
  `mesh_guardrails` request override. It does not reconfigure the server; use
  `--mesh-guardrails`, `mesh-llm runtime guardrails`, or the management API
  for server-side activation.
- If the runtime is unavailable, the script falls back to deterministic
  fake-backend mode and still writes the expected JSON artifact.
- The corpus covers streaming pass-through, native tool-call validation,
  structured `_mesh_respond` output, strict structured output, and the
  unsupported real tools plus strict structured combination.
- The command is a reliability check, not a hard constrained decoding promise.

If a Python sidecar baseline is available, you may optionally run a smoke
comparison on the same corpus to compare behavior before and after this
adaptation. Treat it as a local comparison aid only; it is not a mandatory
implementation gate.

## Resilience

### 11. Dead peer cleanup

- Kill a node with `kill -9`
- Cleanup happens in ~41s via the reconnect-gossip-probe mechanism:
  1. QUIC detects connection drop (~5-30s depending on idle timeout and relay state)
  2. Reconnect attempt with 10s gossip probe timeout
  3. Gossip probe fails → `remove_peer` called immediately
- Heartbeat also detects dead peers (60s interval, 2 consecutive failures) as a fallback
- On-use detection: tunnel failure → immediate death broadcast via stream 0x06
- Dead model goes cold, peer removed from list, death broadcast to mesh
- Dead peer won't be re-added by gossip (dead_peers set)
- Console updates automatically

### 12. Node rejoin

- Kill a node, restart it with `--join <token>`
- Rejoin loop (60s) reconnects to bootstrap if connection drops
- Inbound reconnection clears dead_peers entry
- Model goes warm again, cross-model routing resumes

### 13. Gossip stability

- Regossip after becoming host should NOT cause restart loops
- Log should show "still host, no restart needed" on re-check
- The embedded runtime starts exactly once per election (not 5-9 times)
- Heartbeat gossip doesn't re-discover dead peers (discover_peers=false)

## Control-Plane Protocol (Protobuf v1)

The control plane uses QUIC ALPN `mesh-llm/1` with the `meshllm.node.v1` protobuf schema. Scoped control-plane streams use 4-byte LE framing followed by protobuf bytes. Skippy control/artifact streams are advertised through gossip subprotocol features and run through mesh `STREAM_SUBPROTOCOL` (0x0d); activation transport stays on `skippy-stage/2`.

| Stream | Type | Format |
|--------|------|--------|
| 0x01 | GOSSIP | protobuf `GossipFrame` |
| 0x03 | TUNNEL_MAP | protobuf `TunnelMap` |
| 0x05 | ROUTE_REQUEST | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | protobuf `PeerLeaving` |
| 0x0b | CONFIG_SUBSCRIBE | reserved legacy mesh-plane config stream ID; do not reuse |
| 0x0c | CONFIG_PUSH | reserved legacy mesh-plane config stream ID; do not reuse |
| 0x0d | STREAM_SUBPROTOCOL | protobuf `MeshSubprotocolOpen`, then subprotocol-owned bytes |

Config and inventory mutation must use `mesh-llm-control/1`; `mesh-llm/1` no longer dispatches request/response handlers for the reserved 0x0b/0x0c config stream IDs. Raw TCP relay streams (0x02 RPC, 0x04 HTTP) are unchanged.


### Verifying protobuf gossip in logs

After two nodes connect, look for log lines indicating gossip was exchanged:

```
DEBUG mesh: gossip received from <peer_id>
DEBUG mesh: admitted peer <peer_id>
```

Absence of JSON-related log lines for streams 0x01/0x03/0x05/0x06/0x07 confirms the protobuf path is active.

### Verifying Skippy peer artifact transfer

For a layer-package split where the coordinator already has the HF package
cached and a worker does not:

- Current/current mesh: the worker may use mesh `STREAM_SUBPROTOCOL` (0x0d)
  to open `skippy-stage/2`, then Skippy artifact-transfer stream 0x03, to
  fetch only its assigned package files before the normal HF fallback path.
- Current/released mixed mesh: a coordinator without the complete
  `stage-generation-8` control/status/content-identity/admission bundle must not
  be selected for a generation-8 split topology. Missing `artifact-transfer`
  only prevents peer cache
  sourcing; the worker may still participate when local/HF package resolution
  provides an independent source.
- Default public-mesh safety: with `MESH_LLM_ARTIFACT_TRANSFER` unset, the node
  must advertise no `artifact-transfer` feature, reject inbound artifact
  transfer requests, and continue through local/HF fallback resolution.
- Trusted-owner opt-in: with `MESH_LLM_ARTIFACT_TRANSFER=trusted`, artifact
  transfer is eligible only for same-owner or explicitly trusted-owner peers.
- Explicit lab opt-in: with `MESH_LLM_ARTIFACT_TRANSFER=open`, the node may
  advertise and serve artifact transfer to any peer that is authorized by the
  active split topology.
- Privacy check: gossip/status output must not include local package inventory,
  cache roots, or artifact file lists; only subprotocol feature support is
  advertised.
- Integrity check: corrupt or same-sized cached artifacts must be refetched or
  rejected by SHA-256 verification before stage load.

### 13. Mixed-version owner-control coexistence

Owner-control QA needs to prove three things at the same time:

1. Public-mesh join and routed inference still work across released/current binaries.
2. Explicit endpoint bootstrap and `mesh-llm-control/1` work for current nodes while released peers continue to coexist on the public mesh plane for join, gossip, routing, and inference. Config and inventory mutation stay on owner-control only.
3. A current same-owner requester can run the public `runtime scan-refresh`
   command against exactly one current remote target and receive the sorted,
   metadata-bearing result. Released targets are not expected to return the new
   private response field.

Use the dedicated harness for the full mixed-version pass:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary /absolute/path/to/released/mesh-llm \
  --current-binary /absolute/path/to/current/mesh-llm \
  --evidence-dir /absolute/path/to/evidence
```

For a loopback-only smoke on one machine:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary /absolute/path/to/released/mesh-llm \
  --current-binary /absolute/path/to/current/mesh-llm \
  --evidence-dir /absolute/path/to/evidence \
  --local-only
```

For the owner-control bootstrap lane only:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary /absolute/path/to/released/mesh-llm \
  --current-binary /absolute/path/to/current/mesh-llm \
  --evidence-dir /absolute/path/to/evidence \
  --local-only \
  --config-only
```

The harness writes a timestamped run directory containing logs, status payloads, owner-control bootstrap evidence, owner-control get-config and scan-refresh evidence, and a markdown/json summary.
It also writes `manifest.json`, `commands.jsonl`, `results.jsonl`, `versions/*.txt`, and grouped `status/`, `models/`, `chat/`, and `control/` payloads so a reviewer can audit which binaries, commands, and local endpoints produced the evidence.
Use `--print-plan` when CI or review automation needs to validate the planned checks without starting mesh processes or writing evidence; planned check names match the `name` values emitted to `results.jsonl`.

Expected checks:

- Public mode: both binaries must bring up `/api/status`, list at least one model from `/v1/models`, and complete a routed chat request against the public mesh.
- Loopback mode: the harness runs both mixed-version directions (`current -> released` and `released -> current`) on a private local mesh, waits for peers to appear on both nodes, then runs `mesh-llm runtime bootstrap --json` plus `mesh-llm runtime get-config --json` against the current-version node's explicit endpoint. When two current same-owner nodes are available, it also records `runtime scan-refresh --json` against the explicitly bootstrapped remote target.
- Config-only mode: the harness skips public probes, runs the same loopback coexistence checks, then executes the compatibility tests that prove new clients require explicit endpoints, use `mesh-llm-control/1`, reject legacy frames on the owner-control ALPN, and accept a legacy snapshot-only refresh response without claiming rich inventory support.
- The current-version bootstrap payload must keep `requires_explicit_remote_endpoint=true` and expose an endpoint token when owner-control is enabled.
- If the bootstrap payload reports `enabled=false`, the harness records a `PREREQ` result showing that a signed same-owner keystore is required before runtime owner-control requests can be proven on that machine.

Manual spot checks if the harness fails:

```bash
mesh-llm runtime bootstrap --port 3131 --json
mesh-llm runtime get-config --port 3131 --endpoint '<control-endpoint>' --json
mesh-llm runtime scan-refresh --port 3131 --endpoint '<control-endpoint>' --json
curl -s localhost:3131/api/runtime/control-bootstrap | jq .
curl -s -X POST localhost:3131/api/runtime/control/get-config \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .
curl -s -X POST localhost:3131/api/runtime/control/scan-refresh \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .
```

For a manual current/current scan-refresh check:

1. Start two current nodes with owner-control enabled and signed by the same
   owner. Obtain the target's endpoint from the target node's loopback
   `runtime bootstrap --json` output and transfer it to the requester out of
   band. Do not derive it from public status or a peer ID.
2. From the requester, run `runtime scan-refresh --endpoint <token> --json`.
   Assert `target_node_id` is the target, `disposition` is `executed` or
   `coalesced`, inventory entries are strictly sorted by
   `canonical_model_ref`, sizes are present, and any compact metadata remains
   attached to the matching entry.
3. Start two concurrent scans. Assert one may report `executed` and the joined
   caller reports `coalesced`, both payloads contain the same entries, and the
   target performs only one loader scan.
4. Retry from a requester signed by a different owner and with a request bound
   to the wrong target ID. Assert structured `unauthorized` and
   `target_node_mismatch` failures respectively, with no inventory mutation.
5. Exercise the hidden `runtime refresh-inventory` or
   `POST /api/runtime/control/refresh-inventory` compatibility alias and assert
   its snapshot-only response shape is unchanged. Separately decode a frozen
   snapshot-only wire response with the new client and assert success with
   `inventory = null` / no disposition.
6. Force a scan loader error and assert all joined callers fail while the last
   good runtime inventory and model advertisement remain unchanged.
7. Inspect logs, `/api/status`, peer state, and public gossip evidence. Endpoint
   tokens and raw inventory command results must not appear there; only the
   existing successful available-model projection may change.

Transport-focused evidence must also cover rejection of outbound and inbound
frames over 8 MiB, 2-second open/handshake/write deadlines, 5-second unary and
watch-accept deadlines, the 30-second scan deadline, non-zero request IDs after
wraparound, and the 32-worker per-connection ceiling. An accepted watch must
remain open rather than inherit a unary timeout.

Failure interpretation:

- `control_endpoint_required`: you did not use the explicit bootstrap endpoint.
- `control_unsupported`: the target does not negotiate `mesh-llm-control/1`; this should not silently downgrade.
- `control_unavailable`: listener, token, network path, or local owner-key loading failed.
- `unauthorized`: same-owner authentication failed.
- `revision_conflict`: the apply CAS revision is stale; re-read before retrying.
- `legacy_json_unsupported`: a legacy mesh-plane frame was sent to the owner-control ALPN.

Config-only result interpretation:

- `config-missing-endpoint-required`: executable proof that config bootstrap without an explicit owner-control endpoint is rejected when cargo-backed protocol tests run.
- `config-new-client-owner-control`: executable proof that new clients prefer `mesh-llm-control/1` when cargo-backed protocol tests run.
- `config-control-rejects-legacy-frames`: executable proof that owner-control does not silently accept legacy frames on the wrong ALPN when cargo-backed protocol tests run.
- `config-current-scan-refresh`: a same-owner current requester used an
  explicit current target endpoint and received a sorted private inventory
  response.
- `config-current-scan-refresh-wrong-owner`: a requester with a different owner was
  rejected without changing target inventory state.
- `config-owner-control-client-hardening`: the new client accepted a frozen
  snapshot-only response and passed its timeout, framing, and request lifecycle
  tests; public released/current coexistence is recorded independently.
- `PREREQ config-cargo-tests`: cargo-backed protocol tests were skipped or cargo was unavailable.
- `PREREQ config-runtime-bootstrap`: runtime owner-control requests could not be proven because the local node did not expose a signed owner-control endpoint.

Reserved-ID note: mesh-plane stream IDs 0x0b and 0x0c are kept reserved, but current nodes should not advertise or rely on legacy config subscribe/push behavior there.

## Daemon lifecycle certification

The daemon lifecycle harness certifies the runtime model lifecycle rollout:
zero-model serve, all modes, best-effort/fail-fast, runtime load/unload/ensure/drain,
activity overrides/admission, and privacy — without downloading models or publishing a mesh.

### Running the harness

```bash
# Local daemon lifecycle certification
scripts/qa-runtime-daemon-lifecycle.sh \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence/task-15-local
```

The harness uses temporary homes (`$TMP_ROOT/mesh-runtime-daemon-lifecycle.*`) and ports (base `19740+`) with scoped cleanup. Each test scenario isolates a fresh `$HOME` via `MESH_LLM_EPHEMERAL_KEY=1` so multiple nodes coexist on one machine.

### Planned checks

Each check produces a PASS/FAIL/PREREQ record in `results.jsonl`:

| Check | Description |
|---|---|
| `prereq.current-binary` | Binary is executable and reports a version string |
| `prereq.owner-identity` | Temporary owner keystore can be created for authenticated lifecycle checks |
| `zero_model_serve_ready` | Start the daemon with `--headless` and explicit API, console, and bind ports but no model; verify management API alive, `/v1/models` returns success (empty list acceptable), and the process stays alive |
| `runtime_mode_serve` | Default serve mode starts and responds on management API within timeout |
| `runtime_mode_on_demand` | On-demand mode starts idle without a model; if unsupported, records PREREQ |
| `best_effort_startup` | Start with nonexistent model under explicit best-effort policy — daemon stays alive rather than crashing |
| `fail_fast_startup` | Start with nonexistent model under explicit fail-fast policy — process exits promptly |
| `runtime_load_model` | Authenticated owner-control `POST /api/runtime/control/load-model` returns an accepted intent |
| `runtime_unload_model` | Authenticated owner-control `POST /api/runtime/control/unload-model` returns an accepted intent |
| `runtime_ensure_model` | Authenticated owner-control `POST /api/runtime/control/ensure-model` returns an accepted intent |
| `runtime_drain_model` | Authenticated owner-control `POST /api/runtime/control/drain-model` returns an accepted intent |
| `activity_override` | `PUT /api/runtime/activity/override` with bare JSON string `"active"`, then `DELETE`, round-trips successfully |
| `privacy_no_raw_data` | Assert `/api/runtime/activity` exposes exactly `effective_state`, `override_mode`, and `detector_category`, each limited to its documented coarse enum |
| `clean_process_teardown` | After stopping, verify no leaked mesh-llm processes via tracked instance metadata or fallback pkill |

The lifecycle POST checks first read `/api/runtime/control-bootstrap`, then send
`Content-Type: application/json` requests to `/api/runtime/control/{load,unload,ensure,drain}-model`
with body `{"endpoint":"<control-endpoint>","model":"qa.invalid/model@main:missing.gguf"}`.
The harness expects HTTP 200, `accepted=true`, the same `model`, no `instance_id`,
and `accepted_state` of `present`, `absent`, `present`, and `draining`, respectively.

### Print plan (side-effect-free)

```bash
scripts/qa-runtime-daemon-lifecycle.sh \
  --current-binary ./target/debug/mesh-llm \
  --print-plan
```

Produces compact JSON without starting any processes:

```json
{
  "base_port": 19740,
  "checks": ["prereq.current-binary", "zero_model_serve_ready", ...],
  "current_binary": "./target/debug/mesh-llm",
  "evidence_dir": ".sisyphus/evidence",
  "max_wait_seconds": 60,
  "script": "qa-runtime-daemon-lifecycle.sh"
}
```

### Evidence directory structure

Each run creates a timestamped directory:

```text
.sisyphus/evidence/runtime-daemon-lifecycle-20260724T123456Z-1234/
  manifest.json      Run inputs and mode flags
  commands.jsonl     Commands executed by the harness and their logs
  results.jsonl      Machine-readable PASS/FAIL/PREREQ records
  summary.md         Human-readable final summary
  summary.json       Machine-readable final summary (overall: pass|fail|prereq)
  versions/current.txt   Captured binary version string
  logs/*.log           Per-test process output
  status/*.json        Collected /api/status payloads
  control/*.json       Lifecycle command responses
```

### Prerequisites and failure modes

`PREREQ` is never reported as PASS. Checks that reach execution record missing
runtime prerequisites as `PREREQ`. An invalid `--current-binary` is rejected
earlier as a usage error (exit 2), before an evidence directory is created:

```bash
# Missing binary — exits 2 with an executable-path usage error and no evidence files
scripts/qa-runtime-daemon-lifecycle.sh \
  --current-binary /nonexistent/path/mesh-llm \
  --evidence-dir .sisyphus/evidence/prereq-test
test "$?" -eq 2
```

After any run, `pgrep -f 'mesh-runtime-daemon-lifecycle'` must find no leaked processes. The cleanup trap kills all harness-owned process trees and verifies zero survivors.

## Mixed-version lifecycle certification

The mixed-version harness is extended to probe owner lifecycle commands
(`load-model`, `unload-model`, `ensure-model`, `drain-model`) and typed
unsupported behavior for new commands on released hosts, while preserving
all existing scan-refresh/public mesh checks.

### Running the harness

```bash
# Mixed-version with lifecycle commands (local-only)
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --local-only \
  --evidence-dir .sisyphus/evidence/task-15-mixed-local

# Public mesh certification (preserves existing probes)
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --require-public \
  --evidence-dir .sisyphus/evidence/task-15-mixed-public

# Inspect planned checks (side-effect-free)
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --local-only \
  --print-plan | python3 -c "import json,sys; d=json.load(sys.stdin); [print(c) for c in d['checks'] if 'lifecycle' in c]"
```

### New lifecycle probes (local mode only)

When `--local-only` is set, the harness adds five new checks after existing
owner-control bootstrap and scan-refresh tests:

| Check | Description | Expected result |
|---|---|---|
| `lifecycle-load-model` | Probe current-host owner lifecycle load command | PASS only when the authenticated command returns HTTP 200 with `accepted=true` |
| `lifecycle-unload-model` | Probe current-host owner lifecycle unload command | PASS only when the authenticated command returns HTTP 200 with `accepted=true` |
| `lifecycle-ensure-model` | Probe current-host owner lifecycle ensure command | PASS only when the authenticated command returns HTTP 200 with `accepted=true` |
| `lifecycle-drain-model` | Probe current-host owner lifecycle drain command | PASS only when the authenticated command returns HTTP 200 with `accepted=true` |
| `lifecycle-legacy-unsupported` | Send a lifecycle command through the current controller to the released target | PASS only for a structured response that explicitly reports unsupported/unknown command; a generic 404 is a failure |

`run_local_direction` and the `config-current-controller` setup assign the
available console ports to `CURRENT_HOST_CONSOLE` and `RELEASED_HOST_CONSOLE`;
the lifecycle probes do not scan ports. When `CURRENT_HOST_CONSOLE` is empty,
all five checks record `PREREQ`. When only `RELEASED_HOST_CONSOLE` is empty, the
four current-host lifecycle command probes still run and only
`lifecycle-legacy-unsupported` records `PREREQ`.

### Existing behavior preserved

`--require-public` continues to produce PASS for existing public mesh probes (status, models, chat) without lifecycle command extensions — the `--local-only` flag gates the new probes so they do not affect public-mesh certification runs.

### Expected outcomes

The local-only run produces PASS for owner-control/scan compatibility and typed unsupported for new lifecycle commands on released hosts. The `--require-public` run produces PASS for existing public mesh probes. Both mixed-version `summary.json` files must contain zero FAIL records when prerequisites are satisfied (PREREQ is acceptable for optional checks).

### Failure mode: prerequisite verification

Run each harness with a nonexistent required binary or documented missing prerequisite:

```bash
# Missing released binary — exits nonzero with PREREQ in summary.json
scripts/qa-control-plane-mixed-version.sh \
  --released-binary /nonexistent/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --local-only \
  --evidence-dir .sisyphus/evidence/prereq-test

# Verify summary shows prereq status, not pass
cat .sisyphus/evidence/control-plane-mixed-version-*/*/summary.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'overall={d[\"overall\"]}, counts={d[\"counts\"]}')"

# No leaked processes after failure
pgrep -f 'mesh-control-plane-mixed-version|task-15-' || echo "no leaks"
```

## Single-machine testing with ephemeral keys

Set `MESH_LLM_EPHEMERAL_KEY=1` to give a second process a unique identity on the same machine.
Without this, both processes share `~/.mesh-llm/key` and appear as the same node.

### 14. Forced split on one machine

```bash
# Terminal 1: host with --split
mesh-llm serve --model Qwen2.5-3B --port 9337 --console 3131 --split

# Terminal 2: worker with ephemeral key
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm serve --model Qwen2.5-3B --join <TOKEN> --port 9338 --console 3145 --split --max-vram 1
```

- Host starts solo, then re-elects with split when worker joins
- Worker receives a stage assignment and proxies API requests to the host
- Stage placement is proportional to VRAM (e.g. `0.98,0.02`)
- Kill worker → host detects via heartbeat (~60s), reverts to solo mode

#### Split startup + worker-loss recovery certification

For large-model/manual release QA, use the opt-in certification harness after
building a binary and preparing a model or layer-package ref:

```bash
scripts/certify-split-startup-recovery.sh \
  target/release/mesh-llm \
  /absolute/path/to/model.gguf
```

The harness starts one seed and two workers with isolated homes, ephemeral mesh
keys, private join-token bootstrapping, fixed local ports, and per-process
runtime roots. It then proves:

- a multi-node split topology becomes routable via `/api/runtime/stages` and
  `/v1/models`;
- an active downstream worker is terminated;
- the coordinator reaches the requested recovery outcome without using the
  killed worker.

By default the expected outcome is replacement:

```bash
MESH_SPLIT_CERT_EXPECT=replacement \
scripts/certify-split-startup-recovery.sh target/release/mesh-llm /models/qwen.gguf
```

Supported expectations are:

- `replacement` — a new ready split topology appears without the killed node;
- `withdraw` — the split route is explicitly withdrawn after the worker loss;
- `local-fallback` — serving remains available through a single local route;
- `any` — accept the first stable explicit recovery state for exploratory QA.

Useful knobs:

```bash
MESH_SPLIT_CERT_WORKERS=2
MESH_SPLIT_CERT_SEED_MAX_VRAM=10
MESH_SPLIT_CERT_WORKER_MAX_VRAM=9,10
MESH_SPLIT_CERT_CTX_SIZE=2048
MESH_SPLIT_CERT_DISCOVERY_MODE=mdns
MESH_SPLIT_CERT_RUN_INFERENCE=1
MESH_SPLIT_CERT_KEEP_LOGS=1
MESH_SPLIT_CERT_WORK_DIR=target/split-recovery-cert/manual
MESH_SPLIT_CERT_PROCESS_ROOT=/tmp/mesh-split-cert-manual
```

Expected output includes machine-readable `PASS`/`FAIL` lines and a
`result.jsonl` evidence file in the work directory. The per-process homes and
runtime roots default to a short temp directory to avoid Unix socket path length
limits on macOS. The script never publishes a mesh and does not run by default
in CI.

### 15. Passive client on one machine

```bash
# Terminal 1: host
mesh-llm serve --model Qwen2.5-3B --port 9337

# Terminal 2: client (the client surface uses an ephemeral key automatically)
mesh-llm client --join <TOKEN> --port 9338
```

- Client connects without gossip (no peer list entry on host)
- `/v1/models` returns models from routing table
- Inference routes through QUIC tunnel to host
- Host does NOT see client in its peer list (zero per-client state)

### 16. Release host/runtime/product boundary

For release, installer, SDK, or packaging changes, validate the three layers
without relying on a pre-existing user runtime:

```bash
just release-host-build
just release-runtime-build metal # choose the platform backend
product_out="$(mktemp -d)"
just release-bundle "v$(./target/release/mesh-llm --version | awk '{print $NF}')" "$product_out"
```

Required evidence:

- `scripts/verify-host-dependencies.py target/release/mesh-llm` reports no
  rejected backend imports.
- Extracting the product archive yields one host, one runtime tree,
  `product-manifest.json`, and `host-imports.json`; all recorded digests match.
- With isolated `HOME`, XDG cache, and
  `MESH_LLM_NATIVE_RUNTIME_CACHE_DIR`, the extracted host passes `--version`,
  `--help`, and `runtime list`. The adjacent runtime is listed and the isolated
  cache remains empty.
- A product with an incompatible MeshLLM version or Skippy ABI is rejected
  before loading.
- `client --auto --log-format json --no-console` starts without a native
  runtime or GPU driver, emits a real ready event, and the noninteractive
  service probe stops cleanly on SIGTERM (Unix) or CTRL_BREAK_EVENT (Windows).

Run the command-surface smoke on every product platform without device
passthrough. Separately qualify backend loading and a minimal operation on
suitable CUDA, ROCm, Vulkan, or Metal hardware. Preserve unique temp paths and
ports and verify process/listener cleanup.

## Deploy to remote node

```bash
just bundle
# scp, then on remote:
codesign -s - ~/mesh-bundle/mesh-llm
xattr -cr ~/mesh-bundle/mesh-llm
```

Must codesign + xattr after every scp or macOS kills the binary (exit 137).

## Cleanup

```bash
mesh-llm stop || pkill -f mesh-llm
```

Prefer `mesh-llm stop` for tracked local instances. If the runtime is wedged,
kill any remaining mesh-llm process and then verify no stale process is still
bound to the test ports.
