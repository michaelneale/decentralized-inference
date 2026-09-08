---
title: Config Reference
description: The canonical field-by-field reference for ~/.mesh-llm/config.toml, with wiring status and CLI equivalents
---

# Config Reference

This page is the canonical, schema-derived reference for every
`~/.mesh-llm/config.toml` field. Each field appears exactly once, with its
type, allowed values, default (including `auto` semantics), whether it lives
under `[defaults]`, `[[models]]`, or both, its restart behavior, its current
wiring status, and whether a CLI flag exists for it.

**No field in this stack adds a CLI flag.** The CLI-equivalent column states
`none` for every field without one; that does not change in this PR series.

Status values used below:

- **wired** — the field changes runtime behavior today.
- **partial** — part of the field's path, or only some of its values, take effect.
- **unwired** — the schema accepts and validates the value, but no consumer reads it yet.
- **rejected** — `mesh-llm config validate` fails the field; see [Unsupported and reserved settings](#unsupported-and-reserved-settings).

Precedence for every field below: request value (where applicable), then
`[[models]]`, then `[defaults]`, then family or topology policy, then the
built-in runtime default.

## Environment variables

| Variable | Effect |
|---|---|
| `MESH_LLM_CONFIG` | Full path to the config file, instead of `~/.mesh-llm/config.toml` |
| `MESH_LLM_LIFECYCLE_LOG_PARSER` | Overrides `runtime.lifecycle_log_parser`; accepts `auto`, `enabled`, or `disabled` |

## Managing config via CLI

```bash
mesh-llm config validate
mesh-llm config validate --config-path ./mesh.toml
mesh-llm config validate --config-path ./mesh.toml --json
```

`config validate` checks the TOML file without starting a node.

## Group 1: file, schema, and reload

| Key path | Type | Default / `auto` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|
| `version` | integer | must be `1` | process restart | wired | none |

The file lives at `~/.mesh-llm/config.toml` unless `MESH_LLM_CONFIG` overrides
it. It is a future-start and owner-control-reload artifact, not a live session
mutation layer: most fields apply on the next process restart or model
reload, never mid-request. `request_defaults.*` is the one exception: it
merges request-time, only for absent or null request fields (see
[Group 8](#group-8-sampling-chat-templates-reasoning-and-request-defaults)).
Validation runs on load and on `mesh-llm config validate`; an invalid file
produces a clear startup error rather than a partial start.

## Group 2: node identity, mesh admission, owner control, telemetry, logging, runtime lifecycle

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `gpu.assignment` | enum | `auto` (default), `pinned` | node-level | process restart | wired | none |
| `gpu.parallel` | integer | optional total parallel slot count; unset lets the runtime choose | node-level | process restart | wired | none |
| `mesh_requirements.min_node_version`<br>`mesh_requirements.max_node_version` | string (semver) | optional peer version bounds; unset means no bound | node-level | process restart | wired | none |
| `mesh_requirements.min_protocol_version`<br>`mesh_requirements.max_protocol_version` | integer | `0` means no bound | node-level | process restart | wired | none |
| `mesh_requirements.require_release_attestation` | boolean | `false` | node-level | process restart | wired | none |
| `mesh_requirements.release_signer_keys` | array of string | `[]`; when non-empty, only those signer keys admit peers | node-level | process restart | wired | none |
| `owner_control.bind` | socket address | unset (auto); e.g. `[::]:7447` | node-level | process restart | wired | none |
| `owner_control.advertise_addr` | socket address | unset (auto-detected) | node-level | process restart | wired | none |
| `telemetry.enabled` | boolean | `false` | node-level | process restart | wired | none |
| `telemetry.service_name` | string | `mesh-llm` | node-level | process restart | wired | none |
| `telemetry.endpoint` | URL | unset | node-level | process restart | wired | none |
| `telemetry.headers` | object (string map) | `{}` | node-level | process restart | wired | none |
| `telemetry.export_interval_secs` | integer | `10` | node-level | process restart | wired | none |
| `telemetry.queue_size` | integer | `2048` | node-level | process restart | wired | none |
| `telemetry.prompt_shape_metrics` | boolean | `false` | node-level | process restart | wired; exports token-count histograms only when explicitly enabled | none |
| `telemetry.metrics.endpoint` | URL | unset (falls back to `telemetry.endpoint`) | node-level | process restart | wired | none |
| `logging.audit.enabled` | boolean | unset | node-level | process restart | wired | none |
| `logging.audit.log_path` | path | unset | node-level | process restart | wired | none |
| `logging.audit.log_format` | enum | `json_lines` | node-level | process restart | wired | none |
| `logging.audit.log_level` | enum | `info`, `warn`, `error`, `critical` | node-level | process restart | wired | none |
| `logging.audit.max_file_size_mb` | integer | unset | node-level | process restart | wired | none |
| `logging.audit.max_files` | integer | unset | node-level | process restart | wired | none |
| `logging.enabled` | boolean | `true` | node-level | process restart | wired | none |
| `logging.application_state_root` | path | unset (runtime default path) | node-level | process restart | wired | none |
| `logging.summary_line_limit` | integer | `2048`; 1–65536 | node-level | process restart | wired | none |
| `logging.event_buffer_size` | integer | `10000`; 50–100000 | node-level | process restart | wired | none |
| `logging.retention_ttl_secs` | integer | `129600` (36h); 3600–7776000 | node-level | applies dynamically | wired | none |
| `logging.retention_max_rows` | integer | `100000`; 1–1000000 | node-level | process restart | wired | none |
| `logging.replay_capacity` | integer | `128`; 1–10000, and no greater than `event_buffer_size` | node-level | applies dynamically | wired | none |
| `logging.queue_capacity` | integer | `4096`; 64–131072 | node-level | process restart | wired | none |
| `logging.export_limit_bytes` | integer | 5 MiB; 64 KiB–100 MiB | node-level | process restart | wired | none |
| `logging.cleanup_cadence_secs` | integer | `3600`; 300–86400 | node-level | process restart | wired | none |
| `logging.artifact.capture_mode` | enum | `metadata_only` (default), `redacted_artifacts` | node-level | process restart | wired | none |
| `logging.artifact.byte_limit_bytes` | integer | 256 KiB; 1024–16 MiB | node-level | process restart | wired | none |
| `logging.artifact.aggregate_limit_bytes` | integer | 8 MiB; 512 KiB–500 MiB | node-level | process restart | wired | none |
| `logging.webhook.enabled` | boolean | `false` | node-level | process restart | wired | none |
| `logging.webhook.url` | URL | unset | node-level | process restart | wired | none |
| `logging.webhook.max_attempts` | integer | runtime default | node-level | process restart | wired | none |
| `logging.webhook.timeout_secs` | integer | runtime default | node-level | process restart | wired | none |
| `logging.webhook.dead_letter_retention_secs` | integer | runtime default | node-level | process restart | wired | none |
| `runtime.debug` | boolean | `false` | node-level | process restart | wired | none |
| `runtime.listen_all` | boolean | `false` | node-level | process restart | wired | none |
| `runtime.mode` | enum | `serve` (default), `on_demand`, `client` | node-level | process restart | wired | none |
| `runtime.startup_failure_policy` | enum | `best_effort` (default), `fail_fast` | node-level | process restart | wired | none |
| `runtime.lifecycle_log_parser` | enum | `auto` (default), `enabled`, `disabled`; `auto` keeps only parser categories without equivalent native lifecycle events | node-level | process restart | wired | none |
| `runtime.drain_timeout_secs` | integer | `30`; 1–3600, must not exceed the max | node-level | process restart | wired | none |
| `runtime.drain_timeout_max_secs` | integer | `300`; 1–3600 | node-level | process restart | wired | none |
| `runtime.activity.enabled` | boolean | `false` | node-level | process restart | wired | none |
| `runtime.activity.idle_after_secs` | integer | `300`; 30–86400 | node-level | process restart | wired | none |
| `runtime.activity.poll_interval_secs` | integer | `5`; 1–60 | node-level | process restart | wired | none |
| `runtime.activity.resume_debounce_secs` | integer | `30`; 0–300 | node-level | process restart | wired | none |
| `runtime.activity.response` | enum | `pause_remote` (default), `pause_all`, `reduce_priority` | node-level | process restart | wired | none |
| `runtime.activity.advertisement` | enum | `none`, `availability_only`, `coarse_state` (default), `private_coarse_state` | node-level | process restart | wired | none |
| `runtime.reconcile_model_targets` | boolean | `false` | node-level | process restart | wired | none |
| `runtime.reconcile_model_target_demand_upgrades` | boolean | `false` | node-level | process restart | wired | none |
| `runtime.native_runtime.mesh_version`<br>`runtime.native_runtime.skippy_abi`<br>`runtime.native_runtime.selection` | string | unset (auto-selected) | node-level | process restart | wired | none |
| `runtime.model_target_demand_upgrade_min_requests` | integer | `2` | node-level | process restart | wired | none |
| `runtime.model_target_demand_upgrade_max_age_secs` | integer | `3600` | node-level | process restart | wired | none |
| `advanced.server.alias` | string | unset; per-model alias overrides the default | both | model reload | wired; becomes the served identity used by `/v1/models` and routing | none |

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/#runtime-modes) for mode
behavior and [Activity-aware admission](/docs/pages/runtime-lifecycle/#activity-aware-admission)
for the activity policy and privacy boundary.

## Group 3: model sources, context, KV cache, memory, and prompt caching

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `model` | string | required; catalog id, Hugging Face reference, URL, or local path | `[[models]]` only | model reload | wired | positional `--model` / `--gguf` on the ad-hoc single-model path |
| `hardware.model_path` | path | unset | both | model reload | wired | `--gguf` sets it for the ad-hoc single-model path |
| `hardware.hf_repo`<br>`hardware.hf_file` | string | unset | both | model reload | wired | none |
| `model_fit.ctx_size` | integer | `0` = auto | both | model reload | wired | `--ctx-size` on the ad-hoc single-model path |
| `model_fit.batch` | integer | `0` = auto (`n_batch`) | both | model reload | wired | none |
| `model_fit.ubatch` | integer | `0` = auto (`n_ubatch`); should not exceed `batch` | both | model reload | wired | none |
| `model_fit.cache_type_k`<br>`model_fit.cache_type_v` | enum (dtype) | `auto`, `f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`; explicit value overrides `kv_cache_policy` | both | model reload | wired | none |
| `model_fit.kv_cache_policy` | enum | `auto`, `quality`, `balanced`, `saver`; expands into cache dtypes | both | model reload | wired | none |
| `model_fit.kv_offload` | bool-or-`auto` | `auto` | both | model reload | wired | none |
| `model_fit.kv_unified` | bool-or-`auto` | `auto` | both | model reload | wired (recurrent/hybrid architectures still force this true natively) | none |
| `model_fit.cache_ram_mib` | integer | unset (no cap) | both | model reload | unwired (any positive value fails at model load) | none |
| `model_fit.cache_idle_slots` | integer | unset (unbounded) | both | model reload | wired | none |
| `model_fit.prompt_cache` | bool-or-`auto` | `auto` | both | model reload | wired | none |
| `model_fit.prefix_cache.enabled` | boolean | unset (disabled) | both | model reload | wired | none |
| `model_fit.prefix_cache.max_entries` | integer | runtime default | both | model reload | wired | none |
| `model_fit.prefix_cache.max_bytes` | integer | `0`/unset = no cap | both | model reload | wired | none |
| `model_fit.prefix_cache.min_tokens` | integer | runtime default | both | model reload | wired | none |
| `model_fit.prefix_cache.shared_stride_tokens` | integer | runtime default | both | model reload | wired | none |
| `model_fit.prefix_cache.shared_record_limit` | integer | runtime default | both | model reload | wired | none |
| `model_fit.prefix_cache.payload_mode` | enum | `resident-kv`, `kv-recurrent`, `full-state`, `auto` (default) | both | model reload | wired | none |
| `model_fit.keep_tokens` | integer | unset | both | model reload | unwired | none |
| `model_fit.context_shift` | bool-or-`auto` | `auto` | both | model reload | unwired | none |
| `model_fit.swa_full` | boolean | unset | both | model reload | wired | none |
| `model_fit.checkpoint_interval`<br>`model_fit.checkpoint_count` | integer | unset (disabled) | both | model reload | unwired | none |
| `model_fit.lookup_cache_static`<br>`model_fit.lookup_cache_dynamic` | string | unset | both | model reload | unwired | none |
| `model_fit.flash_attention` | enum | `auto` (default), `disabled`, `enabled` | both | model reload | wired | none |

Missing TOML for this group: GGUF metadata `kv_overrides`. There is no
schema key for it yet; do not expect an override path until a later PR adds
one.

## Group 4: device selection, GPU offload, multi-GPU, CPU MoE, and loading behavior

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `hardware.model_runtime` | enum | `auto`, `cpu`, `cuda`, `rocm`, `metal`, `vulkan`; all rejected when set | both | not applicable | rejected (selected by the installed native runtime and hardware resolver, not by config) | none |
| `hardware.device` | string | unset (auto) | both | model reload | wired | `--device` |
| `hardware.gpu_layers` | integer-or-`auto` | `auto`; `-1` means all where the backend supports it | both | model reload | wired | `--gpu-layers` |
| `hardware.stage_layer_start`<br>`hardware.stage_layer_end` | integer | planner-generated unless pinned; `start >= 0`, `end > start` | both | model reload | wired (staged mode only) | none |
| `hardware.placement` | enum | `auto` (default), `pooled`, `separated` | both | model reload | unwired (passes validation, then fails at model load) | none |
| `hardware.tensor_split` | comma ratios or string | unset | both | model reload | unwired (passes validation, then fails at model load; the separate `--tensor-split` CLI flag has its own plumbing) | `--tensor-split` (separate code path, does not read this field) |
| `hardware.split_mode` | enum | `auto` (default), `none`, `layer`, `row`, `tensor` | both | model reload | partial (single-node resolution only; multi-node stages use auto) | none |
| `hardware.main_gpu` | integer | unset (auto); overrides the derived GPU index when `split_mode` resolves to `none` | both | model reload | partial (single-node resolution only; multi-node stages use auto) | none |
| `hardware.cpu_moe`<br>`hardware.n_cpu_moe` | bool-or-`auto` / integer | family or planner default | both | model reload | unwired | none |
| `hardware.fit_target_mib` | integer | optional | both | model reload | wired (GPU tuner authors it and Skippy fit planning consumes it) | derived from allocatable memory when unset |
| `hardware.safety_margin_gb` | float | not a saved key; documented mapping only | both | model reload | wired (feeds `fit_target_mib` derivation) | none |
| `hardware.fit_context` | bool-or-`auto` | unsupported | both | not applicable | rejected | none |
| `hardware.lora_adapters` | array of string | unsupported | both | not applicable | rejected | none |
| `hardware.control_vectors` | array of string | unsupported | both | not applicable | rejected | none |
| `hardware.check_tensors` | boolean | `false` | both | model reload | partial (single-node resolution only; multi-node stages use disabled default) | none |
| `hardware.checkpoint_quantization` | enum | `preserve` (default), `F32`, `F16`, `BF16`, `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ1_S`, `IQ1_M`, `TQ1_0`, `TQ2_0`, `Q2_K`, `Q2_K_S`, `IQ3_XS`, `IQ3_XXS`, `IQ3_S`, `IQ3_M`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_NL`, `IQ4_XS`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `MXFP4_MOE` | both | model reload | partial (single-node direct SafeTensors checkpoints only) | none |
| `hardware.checkpoint_imatrix` | string path | unset; relative paths resolve from the checkpoint directory | both | model reload | partial (single-node direct SafeTensors checkpoints only); required by importance-aware recipes | none |
| `hardware.mmap` | bool-or-`auto` | `auto` | both | model reload | wired | `--mmap` |
| `hardware.use_mmap_prefetch` | boolean | unsupported | both | not applicable | rejected (the native model loader does not consume it) | none |
| `hardware.use_mmap_buffer` | boolean | unsupported | both | not applicable | rejected (the native model loader does not consume it) | none |
| `hardware.mlock` | boolean | `false` | both | model reload | wired | `--mlock` |
| `hardware.direct_io` | boolean | `false`; takes precedence over `mmap`/`mlock` when `true` | both | model reload | partial (single-node resolution only; multi-node stages use disabled default) | none |
| `hardware.repack` | boolean | `false` | both | model reload | partial (single-node resolution only; multi-node stages use disabled default) | none |
| `hardware.op_offload` | boolean | unset preserves llama.cpp's derived default (currently enabled); `false`/`true` force it | both | model reload | partial (single-node resolution only; multi-node stages use auto) | none |
| `hardware.no_host_buffer` | boolean | `false` | both | model reload | partial (single-node resolution only; multi-node stages use disabled default) | none |
| `hardware.warmup` | bool-or-`auto` | unsupported | both | not applicable | rejected | none |

For one-off SafeTensors serving, `mesh-llm serve --quant <RECIPE>` overrides
`hardware.checkpoint_quantization`; `--checkpoint-imatrix <PATH>` supplies the
optional importance matrix. Run `mesh-llm serve --help-advanced` for the full
recipe list.

Missing TOML for this group: LoRA per-adapter scale, control-vector scale and
layer range (`il_start`/`il_end`), RoPE/YaRN load-time overrides, and general
per-tensor device overrides. None of these has a schema key yet.

## Group 5: concurrency, batching, threads, and performance profiles

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `throughput.parallel` | integer | `1` | both | model reload | wired | `--parallel` |
| `throughput.continuous_batching` | bool-or-`auto` | `auto` | both | model reload | wired (disabled mode limits scheduler iterations to one active request; enabled/auto uses all configured lanes) | none |
| `throughput.threads` | integer | `0` = auto from host CPU count | both | model reload | wired | `--threads` |
| `throughput.threads_batch` | integer | `0` = defaults to `threads` | both | model reload | wired | none |
| `throughput.priority` | integer-or-string | unsupported | both | not applicable | rejected (no model-scoped scheduling or OS-priority consumer) | none |
| `throughput.poll` | bool-or-enum | unsupported | both | not applicable | rejected (the embedded runtime exposes no polling policy) | none |
| `throughput.cpu_affinity` | string or list | unsupported | both | not applicable | rejected (no inference-thread affinity consumer) | none |
| `throughput.numa` | string | unsupported | both | not applicable | rejected (the embedded runtime exposes no NUMA policy) | none |
| `throughput.slot_prompt_similarity` | float | unsupported | both | not applicable | rejected (Skippy prefix reuse has no similarity threshold) | none |
| `throughput.tuning_profile` | enum | `throughput`, `balanced` (default), `saver` | both | model reload | wired (expands batch, ubatch, parallel, and continuous batching) | none |

## Group 6: Skippy staged serving, transport, topology, and lifecycle

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `skippy.stage_model_path` | path | unsupported | both | not applicable | rejected (stage artifacts come from the verified model package) | none |
| `skippy.source_policy` | enum | `fallback` (default), `local-required` | both | model reload | wired (`local-required` admits only workers with the exact pre-copied, content-verified local GGUF and never fetches replacement bytes) | none |
| `skippy.stage_role` | string | unsupported | both | not applicable | rejected (the typed planner derives stage roles) | none |
| `skippy.stage_topology` | string | unsupported | both | not applicable | rejected (issue #1052 owns typed per-model topology; untyped strings are not accepted) | none |
| `skippy.binary_stage_transport` | string | unsupported | both | not applicable | rejected (binary transport is the only staged transport and is selected automatically) | none |
| `skippy.lifecycle_startup_timeout_ms` | integer | `900000` ms | both | model reload | wired (bounds downstream stage load) | none |
| `skippy.lifecycle_readiness_interval_ms`<br>`skippy.lifecycle_health_interval_ms` | integer | `2000` ms / `30000` ms | both | model reload | wired (controls source-readiness polling / coordinator health checks) | none |
| `topology.mode` | enum | unset; `locked` when configured | both | model reload | wired (selects fail-closed locked planning) | none |
| `topology.manifest_sha256` | string | unset | both | model reload | wired (must match the resolved package manifest) | none |
| `topology.stages` | array of typed stage objects | unset | both | model reload | wired (unique endpoint ID or hostname selectors and contiguous half-open ranges) | none |
| `skippy.prefill_chunking` | enum | `auto` (normalizes to `fixed`), `fixed`, `schedule`, `adaptive-ramp` | both | model reload | wired (staged mode only) | none |
| `skippy.prefill_chunk_size` | integer | runtime default | both | model reload | wired (staged mode only) | none |
| `skippy.prefill_chunk_schedule` | string | unset | both | model reload | wired (staged mode only; monotonic comma-separated positive integers) | none |

Staged-only controls stay staged-only: lifecycle intervals, prefill controls,
and manual stage layer ranges only take effect
when the model actually runs in staged mode. The hidden
`--split-topology-lock` remains a maintainer compatibility flag. New operator
configuration should use typed per-model `topology`; explicit `--model` and
`--gguf` continue to bypass configured `[[models]]` entries and their topology.

## Group 7: speculative decoding

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `speculative.strategy` | string | `auto` | both | model reload | wired | none |
| `speculative.mode` | enum | `auto` (default), `disabled`, `draft` | both | model reload | wired (legacy draft-model and standalone N-gram compatibility path; prefer `strategy` for native-MTP and package selection) | none |
| `speculative.draft_model` | path | unset | both | model reload | wired (resolves an explicit path/model reference) | none |
| `speculative.draft_hf_repo`<br>`speculative.draft_hf_file` | string | unset | both | model reload | wired (resolves a Hugging Face repo/file pair) | none |
| `speculative.draft_selection_policy` | enum | `manual`, `auto` | both | model reload | wired (`manual` uses the configured source; `auto` may select a sibling draft/EAGLE GGUF) | none |
| `speculative.pairing_fault` | enum | `warn_disable`, `fail-open`/`fail_open`, `fail-closed`/`fail_closed` | both | model reload | wired | none |
| `speculative.draft_max_tokens` | integer | native MTP uses `1` unless overridden | both | model reload | wired | none |
| `speculative.draft_min_tokens` | integer | `0` | both | model reload | wired | none |
| `speculative.draft_acceptance_threshold` | float | `0.0`–`1.0` | both | model reload | wired (minimum verified draft acceptance ratio; `0.0` keeps exact-token verification) | none |
| `speculative.draft_split_probability` | float | `0.0`–`1.0` | both | model reload | wired (deterministic per-window probability for splitting draft verification spans) | none |
| `speculative.draft_gpu_layers` | integer | planner or runtime default | both | model reload | wired (may propagate on the staged draft path) | none |
| `speculative.draft_device`<br>`speculative.draft_threads` | string / integer | target device / runtime default | both | model reload | wired (overrides the draft model's native loader device and thread counts) | none |
| `speculative.draft_cache_type_k`<br>`speculative.draft_cache_type_v` | enum (dtype) | `auto`, `f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` | both | model reload | wired (draft-model KV cache dtypes) | none |
| `speculative.ngram_min`<br>`speculative.ngram_max` | integer | required for a direct N-gram plan; `0 < min <= max` | both | model reload | wired | none |
| `speculative.ngram_proposer` | enum | `cache` (default), `suffix` | both | model reload | wired | none |
| `speculative.ngram_max_proposal_tokens` | integer | N-gram maximum | both | model reload | wired | none |
| `speculative.extension_max_tokens` | integer | N-gram output budget | both | model reload | wired (requires native MTP plus an N-gram proposer) | none |
| `speculative.native_mtp_reject_cooldown_tokens`<br>`speculative.native_mtp_suppress_cooldown_drafts`<br>`speculative.native_mtp_suppress_cooldown_draft_limit` | integer / boolean | runtime defaults | both | model reload | wired | none |
| `speculative.verify_window_min_tokens`<br>`speculative.verify_window_max_tokens`<br>`speculative.verify_window_pipeline_depth` | integer | package policy or runtime defaults; `min <= max` | both | model reload | wired | none |
| `speculative.spec_default` | bool-or-`auto` | `auto` | both | model reload | wired (`false` disables automatic speculation; `true`, `auto`, and omission enable supported automatic defaults) | none |

## Group 8: sampling, chat templates, reasoning, and request defaults

Request precedence: request payload values still win. Config only supplies a
fallback for fields the request body leaves absent or null. Sampling values
are encoded into the stage protocol and native sampler chain; chat defaults
are applied by the embedded OpenAI frontend before prompt rendering.

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `request_defaults.max_tokens` | integer | unset | both | request-time | wired | none |
| `request_defaults.stop` | string or list | unset | both | request-time | wired | none |
| `request_defaults.temperature` | float | unset | both | request-time | wired | none |
| `request_defaults.top_p` | float | `0.0`–`1.0` | both | request-time | wired | none |
| `request_defaults.top_k` | integer | `>= 0` | both | request-time | wired | none |
| `request_defaults.min_p` | float | `0.0`–`1.0` | both | request-time | wired | none |
| `request_defaults.typical_p` | float | `0.0`–`1.0` | both | request-time | wired | none |
| `request_defaults.top_nsigma` | float | backend range | both | request-time | wired | none |
| `request_defaults.dynatemp_range`<br>`request_defaults.dynatemp_exponent` | float | `>= 0.0` | both | request-time | wired | none |
| `request_defaults.repeat_penalty` | float | `>= 0.0` | both | request-time | wired | none |
| `request_defaults.repeat_last_n` | integer | `>= -1` | both | request-time | wired | none |
| `request_defaults.presence_penalty`<br>`request_defaults.frequency_penalty` | float | backend range | both | request-time | wired | none |
| `request_defaults.dry` | object | typed multiplier, base, length, window, and sequence breakers | both | request-time | wired | none |
| `request_defaults.xtc` | object | probability and threshold | both | request-time | wired | none |
| `request_defaults.mirostat_mode` | integer-or-enum | `disabled`, `1`, `2` | both | request-time | wired | none |
| `request_defaults.mirostat_entropy`<br>`request_defaults.mirostat_learning_rate` | float | `> 0.0` | both | request-time | wired | none |
| `request_defaults.samplers`<br>`request_defaults.sampler_sequence` | array of string / string | supported native sampler names | both | request-time | wired | none |
| `request_defaults.seed` | integer | unset | both | request-time | wired | none |
| `request_defaults.logit_bias` | object | unset | both | request-time | wired | none |
| `request_defaults.ignore_eos` | boolean | `false` | both | request-time | wired | none |
| `request_defaults.reasoning_format` | enum | `auto` (default), `none`, `deepseek`, `deepseek-legacy`, `hidden` | both | request-time | wired | none |
| `request_defaults.reasoning_enabled` | bool-or-enum | `auto`, `off`, `on` | both | request-time | wired | none |
| `request_defaults.reasoning_budget` | integer-or-enum | `auto`, `low`, `medium`, `high` | both | request-time | wired | none |
| `request_defaults.chat_template`<br>`request_defaults.chat_template_file`<br>`request_defaults.jinja` | string / path / boolean | backend auto-detection default | both | request-time | wired | none |
| `request_defaults.chat_template_kwargs` | object | unset | both | request-time | wired | none |
| `request_defaults.skip_chat_parsing` | boolean | `false` | both | request-time | wired | none |
| `request_defaults.prefill_assistant` | string or object | unset | both | request-time | wired | none |
| `request_defaults.system_prompt` | string | unset | both | request-time | wired | none |
| `request_defaults.grammar` | string | unset; mutually exclusive with `json_schema` | both | request-time | wired | none |
| `request_defaults.json_schema` | object | unset; mutually exclusive with `grammar` | both | request-time | wired | none |

Conflict: `request_defaults.mirostat_mode` overrides `top_p`/`top_k`/`min_p`
sampling at the backend when it selects mode `1` or `2`.

## Group 9: multimodal

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `multimodal.mmproj`<br>`hardware.mmproj` | path | unset | both | model reload | wired | `--mmproj` |
| `multimodal.mmproj_url` | Hugging Face file URL | unset | both | model reload | wired; downloads only the projector file | none |
| `multimodal.mmproj_offload`<br>`hardware.mmproj_offload` | bool-or-`auto` | `auto` | both | model reload | wired | none |
| `multimodal.image_min_tokens`<br>`multimodal.image_max_tokens` | integer | family or backend default; `min <= max` | both | model reload | wired | none |
| `multimodal.image_marker` | string | unsupported; use `media_marker` | both | not applicable | rejected during validation | none |
| `multimodal.media_marker` | non-empty string | mtmd default when unset | both | model reload | wired | none |
| `multimodal.batch_max_tokens` | integer `1..=2147483647` | mtmd default when unset | both | model reload | wired | none |
| `multimodal.glm_dsa_policy` | string | `auto` (default) or `v1` | both | model reload | wired | none |
| `multimodal.generation_signal_window` | integer `1..=4096` | `16` when unset | both | model reload | wired | none |

## Group 10: plugins

| Key path | Type | Allowed values / default (`auto`) | Scope | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `plugin.<name>.name` | string | required | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.enabled` | boolean | `true` | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.web_ui_enabled` | boolean | unset (follows the plugin's declared default) | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.command` | string | required unless `url` is set | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.args` | array of string | `[]` | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.url` | URL | unset | plugin entry | plugin process restart | wired for HTTP(S) adapter URLs; `tcp://` control is rejected because no authenticated capability handshake exists | none |
| `plugin.<name>.startup.connect_timeout_secs` | integer | host default | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.startup.init_timeout_secs` | integer | host default | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.startup.optional` | boolean | `false` | plugin entry | plugin process restart | wired; command resolution, spawn, connect, and initialize failures become inactive error summaries instead of stopping host startup | none |
| `plugin.<name>.startup.lazy_start` | boolean | `false` | plugin entry | plugin process restart | wired | none |
| `plugin.<name>.settings.*` | plugin-defined | plugin manifest schema | plugin entry | plugin process restart | wired (validated against the plugin's declared `config_schema`) | none |

Ad-hoc `--model`/`--gguf` selectors preserve their existing CLI precedence and
inherit applicable `[defaults]` values for GPU selection, batching, cache
types, flash attention, context size, and projector selection. Explicit
`--ctx-size` and `--mmproj` values still win.

## Unsupported and reserved settings

These fields are recognized by the parser but rejected by
`mesh-llm config validate`. Each row states the diagnostic and where the operator should
configure the underlying behavior instead. A model-scoped key is never routed
into an unrelated host-global setting.

| Key path | Type | Allowed values / default (`auto`) | `[defaults]` / `[[models]]` | Restart | Status | CLI equivalent |
|---|---|---|---|---|---|---|
| `hardware.rpc_backend` | object | unsupported; the legacy RPC-backend escape hatch is unavailable in the embedded runtime | both | not applicable | rejected (`RejectedField`) | none |
| `throughput.threads_http` | integer | unsupported; dedicated HTTP worker tuning stays host-owned | both | not applicable | rejected (`RejectedField`) | none |
| `throughput.sleep_idle_seconds` | integer | unsupported; power-management idle sleep stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `skippy.openai_frontend_mode` | object | unsupported; frontend mode belongs to deployment or service config | both | not applicable | rejected (`RejectedField`) | none |
| `request_defaults.backend_sampling` | object | unsupported until a backend sampling toggle exists end to end | both | not applicable | rejected (`RejectedField`) | none |
| `request_defaults.adaptive` | object | unsupported until adaptive sampling is implemented end to end | both | not applicable | rejected (`RejectedField`) | none |
| `request_defaults.logprobs` | object | unsupported until logprobs response handling is implemented | both | not applicable | rejected (`RejectedField`) | none |
| `multimodal.embeddings` | object | unsupported; embeddings is not an implemented product mode | both | not applicable | rejected (`RejectedField`) | none |
| `multimodal.reranking` | object | unsupported; reranking is not an implemented product mode | both | not applicable | rejected (`RejectedField`) | none |
| `multimodal.pooling` | object | unsupported; pooling is not an implemented product mode | both | not applicable | rejected (`RejectedField`) | none |
| `multimodal.vocoder` | object | unsupported; vocoder/audio generation is not an implemented product mode | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.host` | string | unsupported; listen address is deployment/service scope | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.port` | integer | unsupported; listen port is deployment/service scope | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.reuse_port` | boolean | unsupported; socket tuning stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.timeout` | integer | unsupported; timeout ownership stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.metrics` | boolean | unsupported; metrics exposure stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.slots` | boolean | unsupported; the slots debug endpoint stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.props` | boolean | unsupported; the props debug endpoint stays operational | both | not applicable | rejected (`RejectedField`) | none |
| `advanced.server.api_prefix` | string | unsupported; route prefix stays deployment-owned | both | not applicable | rejected (`RejectedField`) | none |

Not modeled as config keys at all, by design: API keys and key files, TLS
certificate/key material, static/media filesystem paths, and product
admin/router UI toggles. These stay in the operator's secret store or
deployment environment and must never be persisted in `config.toml`.

## Internal status matrix

[`docs/skippy/CONFIGURATION.md`](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/skippy/CONFIGURATION.md)
is the internal, per-field status inventory this reference is generated
against: owner module, translation target, supported serving modes, and test
evidence for every key path above. Keep both in sync; a compile-time test in
`mesh-llm-config` fails the build when they drift apart.

## Deep dives

| Page | What it covers |
|---|---|
| [Config File](/docs/pages/config-toml/) | File location, schema version, top-level sections |
| [Config Defaults](/docs/pages/config-defaults/) | `[defaults]` runnable examples by intent |
| [Config Models & Plugins](/docs/pages/config-models/) | `[[models]]` and `[[plugin]]` runnable examples |
| [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) | Modes, startup policy, draining, activity admission |
