# Telemetry And Metrics Plugin

mesh-llm exports metrics-only OTLP/HTTP telemetry from host runtime code when
`[telemetry]` config enables an explicit endpoint. No collector or project-owned
destination is hard-coded.

The external `metrics` plugin lives at [Mesh-LLM/metrics](https://github.com/Mesh-LLM/metrics).
It advertises metrics support through the plugin API, but it does not receive
prompts, completions, logs, traces, endpoint URLs, or raw host identifiers.

## Configuration

Configure an OTLP metrics endpoint:

```toml
[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"
```

Install and enable the optional metrics plugin when you want the plugin
capability advertised:

```bash
mesh-llm plugins install metrics
```

```toml
[[plugin]]
name = "metrics"

[plugin.startup]
connect_timeout_secs = 75
init_timeout_secs = 90
optional = true
lazy_start = true
```

The startup block is optional. It is useful on slow legacy machines where the
plugin process may take longer than the default startup budget, or where metrics
should be advertised only after the plugin is actually used.

Endpoint precedence is:

1. `telemetry.metrics.endpoint`
2. `telemetry.endpoint` normalized to `/v1/metrics`
3. `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, only when `telemetry.enabled = true`
4. `OTEL_EXPORTER_OTLP_ENDPOINT` normalized to `/v1/metrics`, only when
   `telemetry.enabled = true`

If no endpoint is configured, telemetry export stays disabled. Ambient OTel
environment variables are not consumed unless telemetry is explicitly enabled in
mesh-llm config.

## Exported Metrics

Request and route metrics are emitted per fronting node. A collector or
dashboard can aggregate `mesh_llm_requests_inflight` across nodes for a
mesh-wide in-flight request view.

Counters:

- `mesh_llm_model_launch_total`
- `mesh_llm_model_launch_success_total`
- `mesh_llm_model_launch_failure_total`
- `mesh_llm_model_unload_total`
- `mesh_llm_model_exit_unexpected_total`
- `mesh_llm_model_request_total`
- `mesh_llm_route_attempt_total`
- `mesh_llm_guardrail_decision_total`
- `mesh_llm_guardrail_outcome_total`
- `mesh_llm_logging_lifecycle_terminal_total`
- `mesh_llm_logging_persistence_queue_dropped_total`
- `mesh_llm_logging_persistence_failure_total`
- `mesh_llm_logging_persistence_shutdown_loss_total`
- `mesh_llm_logging_replay_evicted_total`
- `mesh_llm_logging_replay_gap_total`
- `mesh_llm_logging_replay_dropped_total`
- `mesh_llm_logging_cleanup_total`
- `mesh_llm_logging_webhook_delivery_total`
- `mesh_llm_logging_webhook_attempt_total`
- `mesh_llm_logging_artifact_capture_total`

Gauges:

- `mesh_llm_loaded_models`
- `mesh_llm_model_loaded`
- `mesh_llm_model_context_length`
- `mesh_llm_requests_inflight`
- `mesh_llm_logging_persistence_outstanding`

Histograms:

- `mesh_llm_model_launch_duration_ms`
- `mesh_llm_model_uptime_s`
- `mesh_llm_prompt_tokens` (only when `telemetry.prompt_shape_metrics = true`)
- `mesh_llm_completion_tokens` (only when `telemetry.prompt_shape_metrics = true`)

## Privacy Boundary

Runtime telemetry exports metrics only. The external metrics plugin advertises a
capability only. Neither path exports prompts, completions, logs, traces,
hostnames, mesh gossip, relay messages, raw node IDs, raw GPU stable IDs,
endpoint URLs, or prompt hashes.

Prompt-shape telemetry is a separate, default-off opt-in. It records exact
prompt and completion token counts as histogram values, never as attributes.
Its attributes are restricted to the finite `mesh_llm.route_service` and
`mesh_llm.request_outcome` enums. It does not inspect or export model or node
identities, prompt text, completion text, token IDs, request IDs, endpoint URLs,
local paths, or content hashes.

Guardrail telemetry follows the same boundary. It exports only bounded labels for
guardrail mode, contract kind, decision, bypass reason, outcome, and retry
bucket. It does not export prompt text, completion text, schemas, tool
arguments, raw tool names, reserved sentinel names, request paths, endpoints, or
hostnames.

Logging telemetry is process-local until the host runtime's explicit telemetry
configuration installs its adapter. It exports only counters, one outstanding
queue gauge, and the closed lifecycle/cleanup/webhook/artifact outcome labels
listed below. Logging metrics never include event payloads, prompts,
completions, request or delivery IDs, URLs, local paths, raw identifiers,
tokens, or hashes. Replay eviction, gap, and drop metrics have no attributes.
It is not a log transport: request history, lifecycle details, artifact
metadata, and maintenance receipts remain on the trusted-local logging service
and are never mirrored into OTLP. See [the operator logging guide](../LOGGING.md)
for that local workflow.

Guardrail v1 validates native runtime output, not hard constrained decoding. Streaming is
pass-through, no tool execution happens inside the guardrail layer, and real
tools plus strict structured output stays unsupported in v1. See
`docs/design/OPENAI_GUARDRAILS.md` for the rollout contract and evidence path.

External request model values are not exported by lifecycle, request, route, or
prompt-shape metrics. Configured model lifecycle metrics use a stable
pseudonymous identifier derived from the canonical configured model selector.
The identifier correlates launch, loaded-state, unload, and unexpected-exit
records for one selector without exporting the raw selector, repository name,
filename, path, alias, or profile. It is a domain-separated, deterministic
128-bit SHA-256 prefix rendered as fixed-length lowercase hex. It is
pseudonymous rather than anonymous metadata: a collector can correlate the
same configured selector across events and process restarts. Its fixed width
bounds label size, while per-process series cardinality is bounded by the
configured models represented in lifecycle telemetry. GPU stable IDs and node
IDs are likewise exported as stable pseudonymous hashes, not raw identifiers.
Route-attempt metrics label local, remote, and endpoint target kinds; remote
target IDs are exported only as stable hashes so collectors can aggregate
node-to-node traffic without exposing raw peer IDs.

Telemetry attributes are intentionally allowlisted in code. Any new exported
attribute must update the allowlist, tests, and this document before it is added
to an OTLP record.

| Attribute | Used by | Privacy handling |
|---|---|---|
| `mesh_llm.launch_kind` | lifecycle | Bounded enum. |
| `mesh_llm.model_selector_id` | lifecycle | Stable pseudonymous identifier for the canonical configured model selector; `sha256:` plus 32 lowercase hex characters. Correlates launch, loaded-state, unload, and unexpected-exit series without exporting selector components. Cardinality is bounded by configured models represented in lifecycle telemetry. |
| `mesh_llm.gpu_count` | lifecycle | Count only. |
| `mesh_llm.is_soc` | lifecycle | Boolean only. |
| `mesh_llm.service_version` | lifecycle, request, route, in-flight | Build version only. |
| `mesh_llm.architecture` | lifecycle | GGUF architecture string when available. |
| `mesh_llm.quantization` | lifecycle | Derived quantization label. |
| `mesh_llm.gpu_name` | lifecycle | Hardware product label; no hostname or stable device ID. |
| `mesh_llm.gpu_stable_id` | lifecycle | Stable pseudonymous hash of the GPU ID. |
| `mesh_llm.backend_device` | lifecycle | Backend-local slot label such as `CUDA0`, `ROCm0`, `Vulkan0`, or `MTL0`. |
| `mesh_llm.backend` | lifecycle | Runtime/backend label. |
| `mesh_llm.context_bucket` | lifecycle | Bucketed context length, not the exact configured value. |
| `mesh_llm.failure_reason` | lifecycle | Bounded enum. |
| `mesh_llm.source_node_role` | request, route, in-flight | Bounded node role label such as `client` or `worker`. |
| `mesh_llm.source_node_id` | request, route, in-flight | Stable pseudonymous hash of the source node ID. |
| `mesh_llm.route_service` | request, prompt shape | Bounded service label: `local`, `remote`, `endpoint`, or `unavailable`. |
| `mesh_llm.request_outcome` | request, prompt shape | Bounded enum: `success`, `rejected`, or `unavailable`. |
| `mesh_llm.route_attempt_bucket` | request | Bounded retry bucket: `1`, `2`, `3_4`, or `5_plus`. |
| `mesh_llm.target_kind` | route | Bounded target kind: `local`, `remote`, or `endpoint`. |
| `mesh_llm.target_node_id` | route | Stable pseudonymous hash for local/remote node targets; omitted for endpoint targets. |
| `mesh_llm.attempt_outcome` | route | Bounded enum. |
| `mesh_llm.guardrail.mode` | guardrail decision, guardrail outcome | Bounded enum: `disabled`, `metrics`, or `enforce`. |
| `mesh_llm.guardrail.contract` | guardrail decision, guardrail outcome | Bounded enum: `tools` or `structured`. |
| `mesh_llm.guardrail.decision` | guardrail decision | Bounded enum: `eligible`, `bypassed`, `unsupported`, or `rejected`. |
| `mesh_llm.guardrail.bypass_reason` | guardrail decision | Bounded enum: `disabled`, `streaming`, `no_contract`, `unsupported_surface`, `reserved_collision`, or `mixed_tools_structured`. Omitted when no bypass reason applies. |
| `mesh_llm.guardrail.outcome` | guardrail outcome | Bounded enum: `pass_through`, `valid`, `retried`, `failed`, or `metrics_only_failure`. |
| `mesh_llm.guardrail.attempt_bucket` | guardrail outcome | Bounded retry bucket: `1`, `2`, or `3_plus`. |
| `mesh_llm.logging_terminal_outcome` | logging lifecycle terminal | Bounded enum: `completed`, `failed`, `rejected`, `cancelled`, or `dropped`. |
| `mesh_llm.logging_cleanup_outcome` | logging cleanup | Bounded enum: `completed`, `failed`, or `skipped_unavailable`. |
| `mesh_llm.logging_webhook_delivery_outcome` | logging webhook delivery | Bounded enum: `delivered`, `retry_scheduled`, `dead_lettered`, or `fenced_out`; no delivery ID, endpoint, or HTTP status code. |
| `mesh_llm.logging_webhook_attempt_state` | logging webhook attempt | Bounded enum: `claimed`; no delivery ID or endpoint. |
| `mesh_llm.logging_artifact_capture_status` | logging artifact capture | Bounded enum: `written`, `disabled`, or `failed`; no artifact ID, request ID, kind, path, or content metadata. |
| `llama_stage.verify_window.direct_return_upstream_opened` | Skippy decode summary | Boolean indicating that the preferred upstream-opened v10 prediction-return sink completed its handshake. |
| `llama_stage.verify_window.direct_return_reverse_fallback` | Skippy decode summary | Boolean indicating that the final stage used the bounded reverse-open v10 prediction-return fallback after the preferred sink was unavailable. |
| `llama_stage.linear_proposal.source_queue_wait_us` | Skippy linear proposal source | Queue wait in microseconds; contains no request, session, token, or plugin identity. |
| `llama_stage.linear_proposal.source_callback_us` | Skippy linear proposal source | Plugin callback duration in microseconds; contains no request, session, token, or plugin identity. |
| `llama_stage.linear_proposal.source_outcome` | Skippy linear proposal source | Bounded outcome enum: `ready`, `abstained`, `host_deadline_exceeded`, `queue_full`, `deadline_exceeded_before_dispatch`, `deadline_exceeded_in_plugin`, `candidate_returned_too_late`, or `source_error`. Ready/abstained events use token-debug sampling; deadline, pressure, late-candidate, and source-error outcomes are emitted unconditionally. |

## Runtime Event System Telemetry (v1)

The structured runtime-event system (`.omo/specs/event-system.md`) has its
own opt-in OTLP metrics consumer, separate from the request/route/lifecycle
instruments above: `crates/mesh-llm-host-runtime/src/runtime_events/telemetry.rs`
(engine-side bridge: bounded sample queue, an ingress-latency-observing
`RuntimeEventIngress` decorator, and periodic engine-health/occupancy
sampling) plus `crates/mesh-llm-host-runtime/src/runtime/survey/runtime_events.rs`
(the OTLP-specific consumer: instruments, recording, the attribute
allowlist). It shares the same `[telemetry]` endpoint/interval
configuration and the same disabled/failed-exporter-never-fails-startup
contract as every instrument above.

**v1 scope decision (plan-owner privacy call, not a spec mandate):**
runtime-event telemetry is metrics-only and ID-free. It exports no
request/session/operation IDs, no prompt/completion content, and no
arbitrary labels; it never feeds the reducer or readiness decisions. This is
narrower than spec §12.3, which is the plan's intentional v1 restriction.

**Existing Skippy telemetry stays out of scope.** This pipeline never
touches `crates/skippy-server/src/telemetry.rs` or the attribute vocabulary
in `crates/skippy-metrics`. Converting that existing Skippy telemetry into a
consumer of the runtime-event pipeline is recorded in the plan's deferred
follow-on register, triggered when these instruments have one release of
certification evidence -- it is not part of this pipeline today.

Counters:

- `mesh_llm_runtime_events_class_outcome_total` (`mesh_llm.runtime_events.delivery_class`, `mesh_llm.runtime_events.outcome`)
- `mesh_llm_runtime_events_reservation_exhausted_total`
- `mesh_llm_runtime_events_reducer_error_total`
- `mesh_llm_runtime_events_subscriber_lag_total`
- `mesh_llm_runtime_events_telemetry_samples_dropped_total`

Gauges:

- `mesh_llm_runtime_events_reservation_occupied`

Histograms:

- `mesh_llm_runtime_events_ingress_duration_us` (`mesh_llm.runtime_events.delivery_class`, `mesh_llm.runtime_events.outcome`) -- the certification ingress p99 instrument (spec's frozen 100µs budget).

| Attribute | Used by | Privacy handling |
|---|---|---|
| `mesh_llm.runtime_events.delivery_class` | class outcome, ingress duration | Bounded enum: `terminal`, `state_transition`, `progress`, or `diagnostic`. |
| `mesh_llm.runtime_events.outcome` | class outcome, ingress duration | Bounded enum: `accepted`, `coalesced`, `dropped_progress`, `dropped_diagnostic`, `rejected_shutting_down`, or `terminal_delivery_failed`. |

This allowlist is enforced in **every build profile**, not only debug --
`assert_runtime_event_attrs_allowlisted` in `runtime/survey/runtime_events.rs`
has no `cfg(debug_assertions)` gate, unlike the broader
`TELEMETRY_ATTRIBUTE_ALLOWLIST` above. The reservation-exhausted,
reducer-error, subscriber-lag, and telemetry-samples-dropped counters, and
the reservation-occupied gauge, carry no attributes at all: they are
coalesced deltas of the engine's own `EngineHealth` counters and reservation
occupancy, sampled at most once per second (the same cadence
`EngineHealth::publish_at` already enforces), never per-event.

The ingress-latency/class-outcome instruments are fed live in production:
`RuntimeEventTelemetry::start` installs its sample queue directly onto the
process's `RuntimeEventEngine` (`RuntimeEventEngine::install_telemetry_queue`).
Every producer's `try_submit` call -- via `ScopedIngress` or
`UnreservedIngress`, minted by `reserve_root`/`reserve_child`/
`unreserved_ingress` -- already funnels through that engine's `submit`
dispatch, so installing the queue there observes real traffic from every
task 9-12 producer without any producer file needing to change. A producer
that wants a locally-scoped decorator instead can still opt in with
`crate::runtime_events::telemetry::ObservingIngress`, which shares the same
`record_class_outcome` recording path; nothing in-tree uses that decorator
directly, since the engine-level hook already covers every real call site.

## Review Checklist

Before adding, renaming, or removing OTLP metrics or attributes:

1. Run the repo-local telemetry privacy review skill:
   `.agents/skills/telemetry-privacy-review/SKILL.md`.
2. Keep export destination behavior explicit: no default collector and no ambient
   OTel env export unless `telemetry.enabled = true`.
3. Update `TELEMETRY_ATTRIBUTE_ALLOWLIST` in
   `crates/mesh-llm-host-runtime/src/runtime/survey.rs`.
4. Update the attribute inventory above.
5. Add or update focused tests proving private paths, raw node IDs, raw GPU
   stable IDs, endpoint URLs, prompts, and completions are not exported. Prompt
   shape changes must also capture the actual exported histograms and assert the
   bounded service/outcome labels at the exporter boundary.
6. Keep guardrail corpus evidence under `.sisyphus/evidence/`, separate from
   OTLP export and from the telemetry metric payloads themselves.

## Runtime Safety

Telemetry exporter setup failures disable telemetry without failing inference
startup. Runtime events are buffered through a bounded queue; when the queue is
full, the oldest event is dropped instead of blocking inference.
