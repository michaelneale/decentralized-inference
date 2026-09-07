# CLI User Guide

This is a practical user guide to the `mesh-llm` CLI.
It explains what to run for common tasks, then documents each command and switch.

Catalog id definition: a catalog id is the model id shown in `mesh-llm models recommended` (for example `Qwen3-0.6B-Q4_K_M`).

## Get help

```bash
mesh-llm --help
mesh-llm <command> --help
mesh-llm setup --help
mesh-llm uninstall --help
mesh-llm doctor --help
mesh-llm serve --help
mesh-llm client --help
mesh-llm --help-advanced
mesh-llm models --help
mesh-llm models <subcommand> --help
```

`serve --help` and `client --help` show concise runtime-entrypoint help for the
most common serving and client-only options. Use `--help-advanced` when you need
the complete runtime option surface.

### Logging and local capture

`mesh-llm --help-advanced` also documents the node-local logging store, capture
modes, retention setting, and terminal-event navigation. `--log-format pretty`
is the default operator view; `--log-format json` emits one stable operational
event per stdout line. The local ledger remains the source of truth for request
details and artifacts; CLI output is a bounded, privacy-safe process view.

For the trusted-local ledger, retention, and capture guidance, see
[Operator request logging](/docs/LOGGING/).

## Check the running version

```bash
mesh-llm --version
```

Release builds report the released package version, such as `mesh-llm 0.76.0-rc9`.
Local source builds may include build metadata, such as `mesh-llm 0.76.0-rc9+gABCDEF.dirty`, so you can tell exactly which commit produced the binary. Compatibility checks, native-runtime cache paths, and release identity still use the plain release version.

## Start here (common tasks)

If you want to:

1. Finish a fresh install:

```bash
mesh-llm setup
```

2. Start serving on this machine:

```bash
mesh-llm serve --model Qwen3-0.6B-Q4_K_M
```

3. Join the public mesh:

```bash
mesh-llm serve --auto
```

4. Find a model you can run:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx
```

5. Inspect a model before downloading:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

6. Download a model:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

7. Check what is already installed:

```bash
mesh-llm models installed
```

8. Remove the executable and setup-owned files:

```bash
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
```

## Runtime entrypoints (`serve` / `client`)

If you want to start serving, join a mesh, or run as an API-only client, start here.

Examples:

```bash
mesh-llm setup
mesh-llm serve
mesh-llm serve --model Qwen3-0.6B-Q4_K_M
mesh-llm client --auto
```

## Runtime lifecycle and modes

Bare `mesh-llm serve` is valid. It starts the durable API, console, mesh,
plugins, and owner-control surfaces even when no startup model is configured.

Persisted `[runtime].mode` controls configured models:

- `serve` (default) loads configured and explicit CLI models eagerly.
- `on_demand` keeps configured models as candidate metadata but does not load
  them eagerly. Explicit `--model` and `--gguf` remain eager.
- `client` is routing-only and disables local loading. Persisted client mode
  conflicts with explicit `serve` and local model/serving flags.

`startup_failure_policy = "best_effort"` is the default and keeps durable
surfaces running after an eager model failure. `fail_fast` exits instead.

Local `mesh-llm load` and `mesh-llm unload` target the daemon on this machine.
Remote `mesh-llm runtime load-model`, `ensure-model`, `unload-model`, and
`drain-model` require an explicit same-owner endpoint and target exactly one
remote node. They are not mesh-wide placement commands.

Lifecycle changes may be asynchronous. Observe them with:

```bash
mesh-llm status
curl -s localhost:3131/api/runtime/intents | jq .
curl -s localhost:3131/api/status | jq '.runtime'
curl -s localhost:9337/v1/models | jq '.data[].id'
```

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) for the complete state,
draining, activity, privacy, and compatibility model.

### Native serving integrations

`mesh-llm serve --local-model-only` can host one native serving integration for
an embedded product component. Mesh owns model execution, tokenization,
verification, and the OpenAI-compatible endpoint; the integration receives the
authoritative generation lifecycle and may submit proposals before Mesh's
configured hard deadline. It cannot delay normal decoding: late or unavailable
work is ignored and generation continues normally.

This is a versioned ABI for components compiled for the same Mesh release, not
the general managed-plugin system. It requires all four explicit values below:

```bash
mesh-llm serve --local-model-only \
  --native-serving-plugin /absolute/path/to/libintegration.dylib \
  --native-serving-plugin-config /absolute/path/to/integration.json \
  --native-serving-plugin-state /absolute/path/to/state \
  --native-serving-plugin-deadline-ms 8
```

Mesh validates the ABI and the complete local-model contract before starting.
The state directory belongs to the integration and must be durable. Use the
ordinary `serve` command without these options when no native integration is
needed.

### `setup`

Use this to finish a fresh install after the executable is on your `PATH`.

`mesh-llm setup` downloads and configures the native runtime, can install and
enable the background service on supported macOS and Linux machines, and only
shows the GitHub star prompt when it is interactive and eligible. The star
prompt defaults to Yes, and `--yes` or `--no-interactive` skip it without
starring anything. Default output is concise; use `--verbose` when you want
service paths, commands, log locations, and detailed setup status.

Usage:

```bash
mesh-llm setup
mesh-llm setup --service
mesh-llm setup --no-service --skip-runtime
mesh-llm setup --yes
mesh-llm setup --verbose
```

Switches:

- `--yes`: automatically answer yes to setup prompts. This accepts the service prompt and skips the GitHub star prompt.
- `--no-interactive`: run without prompting. When service is not requested, setup prints guidance to rerun with `--service`.
- `--service`: install and enable the background service.
- `--no-service`: skip installing and enabling the background service.
- `--skip-runtime`: skip downloading or configuring the native runtime.
- `--verbose`: print detailed service paths, commands, log locations, and setup status.

On Windows, `--service` is unsupported.

### `uninstall`

Use this to remove a Mesh executable install and setup-owned service/runtime files from a machine.

By default, uninstall stops tracked `mesh-llm` processes, disables and removes
the per-user service when present, removes setup-owned service helper files,
removes the native-runtime cache, and removes the executable last. It preserves
`~/.mesh-llm` configuration and identity data unless you explicitly pass
`--purge-config`.

Usage:

```bash
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
mesh-llm uninstall --yes --keep-cache
mesh-llm uninstall --yes --purge-config
mesh-llm uninstall --verbose --dry-run
```

Switches:

- `--dry-run`: print the cleanup plan without changing the machine.
- `--yes`: run without a confirmation prompt.
- `--json`: print dry-run plans and outcomes as JSON.
- `--verbose`: print detailed cleanup steps and removed paths.
- `--keep-cache`: preserve downloaded native runtimes.
- `--keep-service-files`: preserve setup-owned service helper files.
- `--purge-config`: remove `~/.mesh-llm` configuration and identity data.
- `--keep-config`: explicitly preserve configuration and identity data.
- `--binary-path <PATH>`: remove a specific executable path.

If the setup service configuration directory contains unrelated files,
uninstall leaves that directory in place and reports a warning instead of
recursively deleting it. Default text output is concise; use `--verbose` when
you want the full cleanup plan or exact removed paths.

### `doctor`

Use this only when troubleshooting a failed install or runtime problem. It gathers local status, runtime diagnostics, and logs.

Usage:

```bash
mesh-llm doctor
```

Switches:

- `--json`: machine-readable output.

### Common runtime options

- `--join <TOKEN>`: join a specific mesh using an invite token (repeatable).
- `--discover [NAME]`: discover a mesh via Nostr and join it. With a name, joins the mesh matching that name. Without a name, behaves like `--auto`.
- `--mesh-discovery-mode <nostr|mdns>`: choose public Nostr or LAN mDNS discovery. mDNS is LAN-scoped and still requires an invite token for joining.
- `--auto`: auto-join the best discovered mesh.
- `--model <MODEL>`: model to serve (catalog id from `models recommended`, HF ref/URL, or path).
- `--gguf <GGUF>`: serve a specific local GGUF file directly (repeatable).
- `--port <PORT>`: API port (default `9337`).
- `--client`: API-only mode (no GPU/model serving).
- `--console <CONSOLE>`: console/API management port (default `3131`).
- `--headless`: disable the embedded web UI; keep the management API on the `--console` port.
- `--publish`: publish your mesh for discovery.
- `--mesh-name <MESH_NAME>`: friendly mesh name in discovery.
- `--region <REGION>`: region hint for discovery.
- `--name <NAME>`: display name for this node.
- `--max-vram <MAX_VRAM>`: cap VRAM used for planning and fit decisions.
- `--llama-flavor <LLAMA_FLAVOR>`: force backend binary flavor (`cpu|cuda|rocm|vulkan|metal`).
- `--config <CONFIG>`: explicit config file path.
- `--owner-key <OWNER_KEY>`: keystore used to attest this runtime node.
- `--owner-required`: fail startup if owner attestation cannot be loaded.
- `--node-label <NODE_LABEL>`: attach a human label to this runtime node certificate.
- `--trust-policy <TRUST_POLICY>`: override peer ownership trust policy.
- `--trust-owner <TRUST_OWNER>`: add trusted owner IDs on top of the local trust store.

### Locked split topology

Automatic split planning chooses nodes and layer boundaries from the capacity
currently advertised by the mesh. For controlled lab benchmarks, use a
topology lock so every host runs the same node order and layer ranges across
repeated runs, branches, and binaries.

Copy the same versioned JSON file to every serving host:

```json
{
  "version": 1,
  "model": "hf://meshllm/example-layers@immutable-revision",
  "manifest_sha256": "<sha256 of model-package.json>",
  "stages": [
    {
      "node": "micstudio.local",
      "layer_start": 0,
      "layer_end": 31
    },
    {
      "node": "studio54-3.local",
      "layer_start": 31,
      "layer_end": 47
    }
  ]
}
```

Then pass the lock with `--split` on every host:

```bash
mesh-llm serve \
  --model hf://meshllm/example-layers@immutable-revision \
  --split \
  --split-topology-lock /path/to/topology-lock.json
```

The runtime verifies the resolved package and manifest digest, resolves each
node selector uniquely, requires contiguous ranges covering the full model,
and applies the normal context, KV-cache, headroom, and VRAM checks to those
exact assignments. A node selector may be a full iroh endpoint ID or an
advertised hostname. Ranges are half-open: `layer_start` is inclusive and
`layer_end` is exclusive.

The lock is fail-closed, not a placement hint. If the requested topology cannot
be reproduced, startup fails. If an assigned stage is later lost, mesh-llm
withdraws the route after the normal grace period instead of replanning or
falling back to a local model. This prevents benchmark results from silently
mixing different execution topologies.

### Speculative decoding overrides

Advanced `serve` invocations can temporarily override a package or config-file
speculative decoding plan. CLI values have highest precedence; fields you omit
continue to come from the selected model, defaults, or model package.

```bash
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft \
  --speculative-strategy mtp \
  --speculative-ngram-min 2 \
  --speculative-ngram-max 4 \
  --speculative-ngram-max-proposal-tokens 32 \
  --speculative-extension-initial-tokens 4 \
  --speculative-extension-max-tokens 32 \
  --speculative-verify-window-pipeline-depth 8
```

- `--speculative-strategy <STRATEGY>`: select `auto`, `disabled`, `mtp`, or a strategy declared by the model package. N-gram is a request-local extension of native MTP, not a standalone strategy.
- `--speculative-ngram-min <N>` / `--speculative-ngram-max <N>`: set the request-local cache history match bounds used to extend native MTP.
- `--speculative-ngram-max-proposal-tokens <N>`: cap the N-gram continuation proposed at once.
- `--speculative-extension-initial-tokens <N>` / `--speculative-extension-max-tokens <N>`: set the adaptive N-gram tail bounds when extending native MTP.
- `--speculative-extension-tail-backoff-proposals <N>`: pause extension attempts after a rejected N-gram tail.
- `--speculative-native-mtp-reject-cooldown-tokens <N>`: set the generated-token cooldown after native MTP rejection.
- `--speculative-native-mtp-suppress-cooldown-drafts`: suppress native drafts during cooldown; `--speculative-native-mtp-allow-cooldown-drafts` explicitly disables a configured suppression policy.
- `--speculative-native-mtp-suppress-cooldown-draft-limit <N>`: cap the native drafts suppressed by one cooldown.
- `--speculative-verify-window-min-tokens <N>` / `--speculative-verify-window-max-tokens <N>`: set adaptive verification window bounds.
- `--speculative-verify-window-pipeline-depth <N>`: set the global maximum in-flight verification windows. Live request heads consume this capacity; optional N-gram windows are admitted with bounded, fair credits and fall back to native MTP when requests already fill the pipeline.

## Commands

### `models`

Start with `models` when you’re working with models: finding them, checking details, downloading them, or checking update state.

Subcommands:

- `recommended`
- `installed`
- `cleanup`
- `prune`
- `search`
- `show`
- `download`
- `package`
- `certify`
- `updates`
- `delete`

### `models recommended`

Run this when you want the official built-in model IDs (catalog IDs) and sizes.

Switches:

- `--json`: machine-readable output.

### `models installed`

Run this when you want to see what’s already on your machine.

Switches:

- `--json`: machine-readable output.

### `models cleanup`

Preview or remove stale managed model-cache entries:

```bash
mesh-llm models cleanup
mesh-llm models cleanup --unused-since 30d --yes
```

Use `--json` for machine-readable output. The default is a preview; `--yes`
applies the removal.

### `models prune`

Preview or remove stale derived Skippy stage artifacts:

```bash
mesh-llm models prune
mesh-llm models prune --yes
```

The default is a preview and active or pinned stage artifacts are preserved.

### `models search`

Use this to find something you can actually download and run (GGUF or MLX).

Usage:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx --limit 5
mesh-llm models search qwen --catalog
```

Switches:

- `--gguf`: GGUF-only search (default).
- `--mlx`: MLX-only search.
- `--catalog`: search only built-in catalog.
- `--limit <LIMIT>`: max results (default `20`).
- `--json`: machine-readable output.

### `models show`

Use this when you want to sanity-check one exact model ref before you download or serve it.

Usage:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models show mlx-community/SmolLM-135M-8bit
```

Switches:

- `--json`: machine-readable output.

### `models download`

Use this when you’re ready to download one specific resolved model.

Usage:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

Switches:

- `--draft`: also download the recommended draft model (if available).
- `--direct`: download the exact HuggingFace GGUF file directly, bypassing catalog layer-package resolution.
- `--json`: machine-readable output.

### `models package`

Plan or submit a Hugging Face Job that splits a source GGUF into a Skippy
layer-package repository. The default is a dry run; `--confirm` is required to
submit a spend-bearing job.

```bash
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --confirm --follow
mesh-llm models package --status <JOB_ID>
```

Pass `--experimental` to publish a public package marked experimental: the
package README carries an experimental warning and the Hugging Face catalog PR
is opened but left unmerged until the package is certified.

Use `--help` for the full planning, status, logs, cancel, and publishing
options.

### `models certify`

Use this when you want a repeatable Skippy layer-package confidence report
before treating a split package as ready for a release or rollout.

Choose exactly one mode: use `--package-only` for package integrity and local
stage materialization, or pass `--api-base` to also prove an already running
OpenAI-compatible mesh endpoint. Runtime certification checks the model list and
requires real text-bearing responses from both chat completions and Responses
API smoke requests, not only successful HTTP status codes.

Usage:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers --package-only --report-out cert.json
mesh-llm models certify unsloth/Qwen3-8B-GGUF:Q4_K_M --api-base http://127.0.0.1:9337 --json
```

Switches:

- `--package-only`: verify package resolution, artifact integrity, and local stage materialization without claiming runtime OpenAI readiness.
- `--api-base <URL>`: run `/v1/models`, `/v1/chat/completions`, and `/v1/responses` smoke gates against an already running mesh-llm API. The URL must be an `http` or `https` base URL.
- `--report-out <PATH>`: write the JSON certification report to disk.
- `--prompt <PROMPT>`: prompt for runtime smoke gates.
- `--max-tokens <N>`: max tokens for runtime smoke gates. Must be greater than zero when runtime gates are enabled.
- `--json`: print the certification report.

### `models updates`

Use this when you want to check for new upstream revisions or refresh cached repo metadata.

Usage:

```bash
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates unsloth/gemma-4-31B-it-GGUF
```

Switches:

- `--all`: operate on all cached HF repos.
- `--check`: check only; do not refresh cache.
- `--json`: machine-readable output.

### `models delete`

Remove a managed model entry. Run `mesh-llm models delete --help` first to
review the current confirmation and selection options.

### `download`

Use this to quickly download by built-in catalog ID or shorthand.

Usage:

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download Qwen3-0.6B-Q4_K_M --draft
```

Switches:

- `--draft`: download recommended draft model too.

### `update`

Use this to update mesh-llm and exit.

Switches:
- `--version <VERSION>`: install a specific release tag or version, for example `v0.60.0`.
- `--flavor <FLAVOR>`: install or switch to a specific release bundle flavor (`cpu`, `cuda`, `rocm`, `vulkan`, or `metal`).
- `--detect-flavor`: re-detect the best host backend flavor before selecting the release bundle. Cannot be combined with `--flavor`.
- `--auto-update`: available on most commands; when set, mesh-llm checks for a newer bundled release before proceeding.

### `runtime`

Inspect and manage installed native runtimes, or run supported owner-control
operations against an explicitly targeted node:

```bash
mesh-llm runtime list
mesh-llm runtime list --available
mesh-llm runtime list --installed
mesh-llm runtime install
mesh-llm runtime install cuda13
mesh-llm runtime remove <RUNTIME_ID>
mesh-llm runtime prune --active-only
mesh-llm runtime scan-refresh --endpoint '<control-endpoint>'
mesh-llm runtime scan-refresh --endpoint '<control-endpoint>' --json
mesh-llm runtime load-model --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime unload-model --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime ensure-model --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime drain-model --endpoint '<control-endpoint>' --instance-id '<instance-id>'
```

Plain `mesh-llm runtime list` lists locally discoverable native runtimes. Use
`mesh-llm runtime list --available` to list the release-manifest and bundled
runtimes together: an adjacent bundle (for example the cpu runtime shipped
next to the Windows binary) no longer hides the downloadable GPU runtimes of
the release catalog. `--installed` is the explicit compatibility spelling for
the default local-discovery behavior.

Use `--json` for machine-readable output. Runtime selection is constrained by
the running Mesh version, platform, backend, and Skippy ABI.

With `--available`, the JSON output is an object rather than a bare array:
`catalogs` says what was consulted (the manifest file or URL and how many
artifacts it listed, the bundle directories and how many runtimes they added,
and the remote error when the release catalog was unreachable and the bundles
carried the load), and `runtimes` holds the rows. `runtime install --json`
reports the same `catalogs` object on success. When no runtime can be
selected, its `error` object carries a `resolution` field with the catalogs,
the candidates that were plausible for this host and their rejection reasons,
the number of candidates set aside, and whether the candidates could not be
enumerated at all; `resolution` is `null` for other failures, and `context`
keeps the cause chain in every case.

#### `runtime scan-refresh`

Use this to ask exactly one remote, owner-attested node to rescan its managed
model inventory. It uses the private `mesh-llm-control/1` lane; it does not
change public mesh join, gossip, routing, or inference behavior.

The target node must expose owner-control and the requester must use an owner
key for the same owner. On the target node, read the endpoint token locally:

```bash
mesh-llm runtime bootstrap --port 3131 --json
```

Transfer that token to the controlling node out of band, then run:

```bash
mesh-llm runtime scan-refresh \
  --port 3131 \
  --endpoint '<control-endpoint>'
```

Switches:

- `--endpoint <TOKEN>`: required token that identifies and pins one target
  node. Mesh does not infer it from a peer ID, public gossip, discovery, or
  status output.
- `--port <PORT>`: management API port on the controlling node (default
  `3131`). The CLI sends the request through this local, loopback-only API.
- `--json`: print the API response unchanged.

Human output includes the execution disposition, target node, model count,
total bytes, and sorted canonical model references. JSON output includes
`target_node_id`, `disposition`, and `inventory`. A disposition of `executed`
means this request ran the scan; `coalesced` means it joined an in-progress
scan and received the same result.

The older hidden `runtime refresh-inventory` spelling remains available for
compatibility and returns its legacy snapshot-only shape. New clients also
accept snapshot-only responses from older owner-control servers without
claiming that detailed inventory metadata was returned.

#### Owner-control model lifecycle

Use the lifecycle subcommands to manage models on exactly one remote,
owner-attested node:

```bash
mesh-llm runtime load-model \
  --endpoint '<control-endpoint>' \
  --model 'org/model:file.gguf'

mesh-llm runtime ensure-model \
  --endpoint '<control-endpoint>' \
  --model 'org/model:file.gguf' \
  --profile low-ctx

mesh-llm runtime unload-model \
  --endpoint '<control-endpoint>' \
  --model 'org/model:file.gguf'

mesh-llm runtime drain-model \
  --endpoint '<control-endpoint>' \
  --instance-id '<instance-id>'
```

`load-model` and `ensure-model` require a canonical model reference and accept
an optional `--profile`. `unload-model` and `drain-model` require exactly one
of `--model` or `--instance-id`. All four commands accept `--port <PORT>`
(default `3131`) and `--json`.

The endpoint token and ownership requirements are the same as for
`runtime scan-refresh`. A successful response means the target accepted the
lifecycle intent; use runtime status to observe the resulting instance state.
`drain-model` stops new admission, waits for in-flight work within the target
policy, and then unloads the selected model or instance.

These intents last only for the target daemon session and never edit its
configuration. Older targets return a typed unsupported result; the client
does not retry over the public mesh plane.


### `gpus`

Use this to inspect local GPU identity and capacity, including per-device VRAM, unified-memory state, and cached benchmark-derived bandwidth when present.

### `config validate`

Use this to validate a mesh-llm config file before starting a node or applying
the file through owner-control.

Usage:

```bash
mesh-llm config validate
mesh-llm config validate --config-path ~/.mesh-llm/config.toml
mesh-llm config validate --config-path ./mesh.toml --json
```

Switches:

- `--config-path <CONFIG_PATH>`: config TOML file to validate. If omitted,
  mesh-llm uses the global `--config` path, then `MESH_LLM_CONFIG`, then
  `~/.mesh-llm/config.toml`.
- `--json`: print a machine-readable validation report.

The JSON report uses this shape:

```json
{
  "ok": false,
  "path": "./mesh.toml",
  "diagnostics": [
    {
      "code": "missing_required_value",
      "severity": "error",
      "source": "schema",
      "path": "plugin[\"example\"].settings.api_key",
      "message": "required plugin setting is missing"
    }
  ]
}
```

Validation checks built-in settings and installed plugin config schemas. Warning
diagnostics are printed but do not make the command fail; error diagnostics and
TOML load/parse failures exit nonzero.


### `load`

Use this to load a model into an already-running local mesh-llm runtime.

Usage:

```bash
mesh-llm load Qwen3-0.6B-Q4_K_M
```

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

The command targets the local runtime. Use `status`,
`GET /api/runtime/intents`, and `/v1/models` to observe completion.

### `unload`

Use this to unload a model from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

The command targets the local runtime. Remote same-owner control uses
`runtime unload-model`.

### `status`

Use this to inspect model status from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `discover`

Use this to discover meshes via Nostr and optionally select one automatically.

Switches:

- `--name <NAME>`: filter by mesh name (case-insensitive exact match).
- `--model <MODEL>`: filter discovered meshes by model name substring.
- `--min-vram <MIN_VRAM>`: filter by minimum VRAM (GB).
- `--region <REGION>`: filter by region.
- `--auto`: print best invite token (useful for piping).
- `--relay <RELAY>`: custom relay URL(s).

### `benchmark`

Use this to benchmark model-serving throughput and import prompt corpora. The
`benchmark` command has two subcommands: `tune` and `import-prompts`.

#### `benchmark tune`

Tune model-serving settings by running isolated throughput trials against one or
more local model targets. Trials sweep candidate values and recommend the best
configuration.

Usage:

```bash
mesh-llm benchmark tune
mesh-llm benchmark tune --model Qwen3-0.6B-Q4_K_M
mesh-llm benchmark tune --models Qwen3-0.6B-Q4_K_M,gemma-4-31B-it-Q4_K_M
mesh-llm benchmark tune --model Qwen3-0.6B-Q4_K_M --ctx-sizes 4096,8192 --batch-sizes 512,1024 --ubatch-sizes 256,512
mesh-llm benchmark tune --model Qwen3-0.6B-Q4_K_M --apply
mesh-llm benchmark tune --model Qwen3-0.6B-Q4_K_M --apply --replace-existing
mesh-llm benchmark tune --model Qwen3-0.6B-Q4_K_M --launch-args
```

Core tuning switches:

- `--model <MODEL>`: tune one specific local/configured model target.
- `--models <MODELS>`: tune multiple local/configured model targets (comma-separated). Conflicts with `--model`.
- `--json`: print machine-readable JSON output.
- `--apply`: persist the recommended settings to the local config file (`~/.mesh-llm/config.toml`).
- `--replace-existing`: when persisting, overwrite existing writable recommendation fields instead of preserving current values.
- `--launch-args`: print the exact `mesh-llm serve` arguments generated by the tune path instead of performing config application.
- `--ctx-sizes <SIZES>`: context sizes to benchmark (comma-separated token counts).
- `--batch-sizes <SIZES>`: batch sizes to benchmark (comma-separated).
- `--ubatch-sizes <SIZES>`: micro-batch sizes to benchmark (comma-separated).
- `--mmap-values <VALUES>`: mmap settings to benchmark independently (`auto`, `enabled`, `disabled`; comma-separated).
- `--mlock-values <VALUES>`: mlock settings to benchmark independently (`enabled`, `disabled`; comma-separated).

Speculative decoding tuning switches:

- `--speculative-types <TYPES>`: speculative decoding types to sweep (`auto`, `disabled`, `mtp`, `draft`, `mtp-ngram`; comma-separated). Conflicts with `--no-speculative-tune`.
- `--no-speculative-tune`: disable speculative decoding sweeps and only benchmark the disabled baseline.
- `--spec-draft-models <PATHS>`: candidate draft GGUF paths for speculative draft mode (comma-separated).
- `--spec-draft-max-tokens <N>`: candidate maximum draft-token windows for MTP and draft speculation (comma-separated).
- `--spec-draft-min-tokens <N>`: candidate minimum draft-token windows for MTP and draft speculation (comma-separated).
- `--spec-ngram-min <N>`: candidate minimum cache match lengths for `mtp-ngram` (comma-separated).
- `--spec-ngram-max <N>`: candidate maximum cache match lengths for `mtp-ngram` (comma-separated).

Additional switches:

- `--throughput-tolerance-pct <PCT>`: treat candidates within this percent of the raw best tok/s as throughput-equivalent (default `10.0`).
- `--max-tokens <N>`: maximum generated tokens per benchmark request (default `128`).
- `--startup-timeout-secs <SECS>`: startup wait limit for each benchmark trial (default `600`).
- `--request-timeout-secs <SECS>`: HTTP request timeout for each benchmark request (default `600`).
- `--debug-telemetry`: capture Skippy debug telemetry in each trial log.
- `--prompt <PROMPT>`: prompt sent during benchmark trials (default `"Write a concise paragraph about distributed GPU inference."`).

#### `benchmark import-prompts`

Import a prompt corpus from a supported online source into local JSONL.

Usage:

```bash
mesh-llm benchmark import-prompts --source mt-bench --output ./corpus.jsonl
mesh-llm benchmark import-prompts --source gsm8k --limit 50 --max-tokens 512 --output ./eval.jsonl
```

Switches:

- `--source <SOURCE>`: online source to import (`mt-bench`, `gsm8k`, `humaneval`).
- `--limit <LIMIT>`: maximum number of prompts to import (default `20`).
- `--max-tokens <N>`: optional per-prompt decode budget hint written into the corpus.
- `--output <PATH>`: output JSONL path (required).


### `goose`

Use this to launch Goose already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `claude`

Use this to launch Claude Code already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `opencode`

Use this to launch OpenCode already wired to mesh-llm's OpenAI-compatible endpoint.

It injects a temporary OpenCode config through `OPENCODE_CONFIG_CONTENT` at launch time, so it does not edit persistent OpenCode config files unless you explicitly pass `--write`.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--host <HOST|HOST:PORT|URL>`: OpenCode target host or URL (default `127.0.0.1:9337`). Bare host forms assume `http`, default inference port `9337`, and default management port `3131`.
- `--write`: write a merged `~/.config/opencode/opencode.json` that preserves unrelated root keys and sibling providers. If only `opencode.jsonc` exists, mesh-llm errors and tells you to rename or migrate it to `opencode.json` first.

### `pi`

Use this to launch Pi already wired to mesh-llm's OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--host <HOST|HOST:PORT|URL>`: Pi target host or URL (default `127.0.0.1:9337`). Bare host forms assume `http`, default inference port `9337`, and default management port `3131`.

### `stop`

Use this to stop local `mesh-llm` instances tracked in the runtime root.


### `blackboard` (plugin)

Shared mesh notes — post, search, and read notes across the mesh. Blackboard was moved from a built-in command to an [installable plugin](/docs/pages/plugins/#use-plugin-features):

```bash
mesh-llm plugins install blackboard
```

Once installed, it runs as a managed plugin process when mesh-llm starts. See the [plugins documentation](/docs/pages/plugins/#use-plugin-features) for configuration and usage.

### `plugins` (alias: `plugin`)

Use this to install, manage, and inspect plugins.

Both `mesh-llm plugins` and `mesh-llm plugin` work.

Subcommands:

- `plugins install <reference>`: install from a catalog name, GitHub
  `owner/repository`, or GitHub URL.
- `plugins install --archive <PATH> --name <NAME> [--version <VERSION>]`:
  install a local `.tar.gz` or `.zip` release archive. `--name` is required;
  `--version` defaults to `dev`. These flags conflict with `<reference>`.
- `plugins update <name>`: update an installed plugin to the latest compatible release.
- `plugins enable <name>`: mark an installed plugin runnable by mesh-llm.
- `plugins disable <name>`: keep the plugin on disk but prevent host startup from launching it.
- `plugins delete <name>`: remove the extracted files and local metadata.
- `plugins info <name>`: show source, version, target, path, and latest known status.
- `plugins search [query]`: search the plugin catalog.
- `plugins list`: list installed, auto-registered, and configured plugins.

For plugins that declare a console projection, `web_ui_enabled` and the
Configuration → Plugins toggle affect only that projection; they do not change
the plugin process state.

Local archives are for authoring and validation. Rebuild and reinstall them;
`plugins update` remains a GitHub release workflow.

See [plugins documentation](/docs/pages/plugins/#use-plugin-features) for more detail.


### `auth`

Use this to manage owner identity and keystore files.

Subcommands:

- `auth init`: generate/save owner keypair.
- `auth status`: show identity/keystore status.
- `auth sign-node`: sign the current node identity with the owner key.
- `auth renew-node`: renew the local node ownership certificate.
- `auth verify-node`: verify a node ownership certificate and trust policy.
- `auth rotate-node`: rotate the local node identity key and optionally revoke
  the previous certificate.
- `auth revoke-owner`: revoke an owner in the local trust store.
- `auth revoke-node`: revoke a node certificate or node ID in the local trust
  store.
- `auth rotate-owner`: rotate the owner keystore identity.
- `auth trust add <OWNER_ID> [--label <LABEL>] [--trust-store <PATH>]`: add an
  owner to the local trust allowlist.
- `auth trust remove <OWNER_ID> [--trust-store <PATH>]`: remove an owner from
  the local trust allowlist.
- `auth trust list [--trust-store <PATH>]`: show the current trust store.

`auth init` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--force`: overwrite existing keystore.
- `--no-passphrase`: leave keys unencrypted.
- `--keychain`: store random unlock passphrase in OS keychain.

`auth status` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

`auth sign-node` / `auth renew-node` / `auth rotate-node` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--node-label <NODE_LABEL>`: attach a human label to the signed node certificate.

`auth rotate-owner` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

`auth trust` switches:

- `--trust-store <PATH>`: use a specific trust store instead of the default.
- `auth trust add <OWNER_ID> --label <LABEL>`: attach a human-readable label to
  a trusted owner.

## Model reference formats

Supported for `models show`, `models download`, and `serve --model`:

1. Catalog id (an id from `mesh-llm models recommended`):

```bash
mesh-llm models show Qwen3-0.6B-Q4_K_M
```

2. HF repo or GGUF selector:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

3. HF URL:

```bash
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
```

4. Revision pin:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@main
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@<commit-sha>
mesh-llm models show mlx-community/SmolLM-135M-8bit@<commit-sha>
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main
```

For MLX, use repo shorthand (not `/model`):

```bash
mesh-llm models show mlx-community/SmolLM-135M-8bit
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

## Model resolution behavior

Resolution order:

1. exact catalog id
2. exact HF ref
3. HF URL
4. bare-name discovery

GGUF behavior:

1. GGUF search uses Hub `gguf` pre-filter.
2. Excludes sidecars like `mmproj*.gguf`.
3. Split GGUF uses first shard (`-00001-of-...`) for selection/display.
4. `repo` with no selector uses fit-aware ranking against local VRAM.
5. `repo:SELECTOR` resolves exact quant/variant.

MLX behavior:

1. MLX search uses Hub `mlx` pre-filter.
2. Model must include weight files (`model.safetensors` or split first shard).
3. `model.safetensors.index.json` by itself is not treated as a model artifact.
4. Display reference stays repo shorthand.

## Machine-readable output (`--json`)

All `models` subcommands support `--json`.

Examples:

```bash
mesh-llm models search smoll --mlx --limit 1 --json | jq .
mesh-llm models show mlx-community/SmolLM-135M-8bit --json | jq .
mesh-llm models download Qwen3-0.6B-Q4_K_M --json | jq .
mesh-llm models installed --json | jq .
mesh-llm models recommended --json | jq .
mesh-llm models updates --check --json | jq .
```

Shape summary:

- `search --json`: `{ filter, query, machine, results[] }`
- `show --json`: resolved model + `variants[]`
- `download --json`: requested/resolved refs + local `path`
- `installed --json`: `{ cache_dir, results[] }`
- `recommended --json`: `{ source, results[] }`
- `updates --json`: check/update results

Automation tips:

1. Prefer explicit refs in scripts.
2. Pin `@<commit-sha>` when reproducibility matters.
3. Parse stable keys such as `type`, `ref`, `fit`, `path`, and `results`.
