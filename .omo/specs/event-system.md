# Skippy Runtime Event System Specification

Status: ratified (plan: event-system)

Last updated: 2026-07-18

## 1. Purpose

This specification defines a structured event system for observations produced
by the patched llama.cpp Skippy runtime and by the Rust serving stack around it.
The system is intended to support:

- native runtime and model lifecycle visibility;
- backend-neutral node, model, stage, and topology state management;
- model availability and readiness decisions owned by Rust;
- prompt processing, prefill, decode, and generation visibility;
- session, KV/cache, device, and resource health;
- structured warnings, recoveries, and failures;
- CLI, TUI, JSON, management API, and telemetry consumers; and
- removal of normal-runtime dependence on parsing llama.cpp debug and log text.

The design preserves the existing native/Rust ownership boundary:

> Native code emits bounded facts. Rust owns policy, state transitions,
> readiness, retries, routing, presentation, telemetry, and supervision.

This specification does not require every event to originate in native code.
Facts that only Rust can know, such as routing identity, model availability,
HTTP readiness, and authoritative ABI call outcomes, remain Rust-owned events.

## 2. Decision Summary

The native C callback remains the ABI ingress mechanism. It is not replaced by
a C++ event bus.

The current direct callback-consumption pattern evolves into a callback-fed,
Rust-owned event pipeline:

```text
patched llama.cpp
  -> synchronous operation-scoped callback
  -> minimal Rust FFI trampoline
  -> validated, owned native fact
  -> bounded nonblocking Rust ingress
  -> typed dispatcher and state reducer
       -> lifecycle and availability state
       -> CLI/TUI/JSON presentation
       -> management API event views
       -> telemetry
       -> diagnostic subscribers
```

The native callback MUST do no policy work. The Rust FFI trampoline MUST do no
formatting, blocking I/O, telemetry export, state mutation requiring
application locks, or reentry into Skippy.

The normal ABI return value remains authoritative. Callback events are
observations and MUST NOT override the result of the operation that emitted
them.

## 3. Goals

The event system MUST:

- preserve a small, stable, append-only native ABI boundary;
- keep native callback execution short and nonblocking;
- protect model loading and inference throughput from slow event consumers;
- retain structured native data until all interested Rust consumers can use it;
- support multiple consumers without coupling producers to presentation;
- provide explicit event priority, buffering, coalescing, sampling, and drop
  behavior;
- preserve terminal correctness when callbacks or queued observations are
  missing;
- correlate events across model, stage, topology, session, and request scopes;
- expose enough information for Rust to derive backend-neutral availability;
- replace parsed native logs with structured facts for normal operation;
- bound memory, CPU, cardinality, and exporter work;
- remain useful when telemetry is disabled or unavailable;
- support mixed-version native runtimes through feature probing; and
- make the event system itself observable.

## 4. Non-Goals

This specification does not introduce:

- a C++ event bus, scheduler, telemetry exporter, or readiness state machine;
- native routing, retry, supervision, or node-availability policy;
- an unbounded event queue;
- per-tensor, per-layer, per-kernel, or graph-node event streams;
- production per-token text, logits, or token-event streams;
- detailed speculative or MTP proposal and acceptance accounting;
- detailed chat-template or grammar setup events;
- prompt text, completion text, tool arguments, request bodies, or media
  contents;
- a requirement that optional telemetry export succeed;
- asynchronous events as a replacement for ABI return values;
- a requirement to preserve arbitrary upstream llama.cpp log strings; or
- backend-specific state in public node or management API contracts.

Raw native logs MAY remain available as an opt-in debug artifact. They MUST NOT
be required for normal lifecycle, availability, progress, warning, or resource
state.

## 5. Current State

The current Skippy runtime event ABI exposes five model-open observations:

- model open started;
- model open progress;
- backend device selected;
- model open finished; and
- handled model-open failure.

The callback is operation-scoped, synchronous, best effort, and may execute on
the model-open thread or a native worker thread. Rust copies the native event
and immediately invokes a caller-supplied mutable closure. The mesh host then
projects known native events into generic output information or warning
messages.

The current boundary has useful properties:

- a small C-compatible ABI;
- explicit event version and struct size;
- fixed-width fields;
- borrowed string data copied during the callback;
- no callback after the model-open entrypoint returns;
- no native policy ownership; and
- authoritative success or failure through the normal return path.

Its limitations are:

- consumer work runs inline on the native emitter thread;
- one callback effectively serves one directly coupled consumer;
- no event buffering, fan-out, replay, or reducer exists;
- progress, lifecycle, and diagnostics have no distinct backpressure policy;
- structured fields are flattened before other consumers can use them;
- unknown future event kinds are dropped by the host projection;
- model-open callbacks are opt-in and not used by every Skippy caller;
- session, prefill, decode, generation, KV, warning, and unload event families
  are absent; and
- model, backend, memory, KV, and tokenizer visibility still partly depends on
  parsing upstream log text.

Rust-owned Skippy telemetry already records higher-level request, prefill,
decode, generation, KV, and lifecycle spans. That telemetry is useful prior art
for bounded asynchronous delivery, but it is not the event system:

- telemetry may be disabled;
- telemetry is an optional export consumer;
- telemetry events do not provide authoritative local state;
- many observations are reconstructed around synchronous calls rather than
  emitted from native execution; and
- lower-level Skippy consumers do not necessarily use the server telemetry
  path.

## 6. Architectural Principles

### 6.1 Facts and state are separate

Producers emit facts. A Rust reducer derives state.

Examples:

- Native may report that model tensors finished loading.
- Rust decides whether the model is available to serve.
- Native may report device memory pressure.
- Rust decides whether capacity is reduced or routing should avoid the model.
- Native may report decode cancellation.
- Rust reconciles it with the request cancellation and ABI return outcome.

Events MUST NOT directly encode routing or retry policy.

### 6.2 Native ingress stays minimal

The native callback and Rust trampoline MUST be safe for native worker-thread
execution.

The trampoline MUST:

1. validate the event pointer, ABI version, and readable struct size;
2. copy all borrowed native data required after callback return;
3. normalize integer discriminants without assuming unknown values are valid
   Rust enum variants;
4. attach operation-scoped Rust correlation when available;
5. submit the owned fact through a nonblocking ingress; and
6. return promptly.

The trampoline MUST NOT:

- block on queue capacity;
- acquire application-wide locks;
- write to files, terminals, sockets, or telemetry exporters;
- format user-facing messages;
- call an arbitrary application subscriber directly;
- call back into Skippy; or
- unwind through FFI.

### 6.3 Rust owns the event pipeline

Rust owns:

- buffering and backpressure;
- event classification and priority;
- coalescing and sampling;
- operation-result reconciliation;
- state reduction;
- subscriber fan-out;
- persistence or debug recording;
- management API projection;
- output formatting;
- telemetry transformation; and
- event-system health.

### 6.4 Public state is backend-neutral

Native Skippy details MAY remain in internal fact types. State exposed to the
mesh, management API, routing, and UI MUST use backend-neutral model, stage,
request, session, capacity, and availability concepts.

### 6.5 Return values remain authoritative

For every native operation:

- the ABI return value or process outcome determines authoritative success or
  failure;
- callback observations provide progress and diagnostic facts;
- Rust MUST reconcile a terminal Rust operation event from the authoritative
  outcome;
- a missing native terminal callback MUST NOT leave an operation permanently
  active;
- a native success observation followed by a failed return is a failed
  operation;
- a native failure observation followed by a successful return is a successful
  operation with a contradictory diagnostic fact; and
- a process crash is handled by supervision, not assumed to have emitted a
  final callback.

## 7. Event Model

### 7.1 Event layers

The system has three related event layers.

#### Native facts

Native facts are direct observations copied from the Skippy ABI callback.
They retain native sequence, timestamp, category, kind, progress, status, and
detail fields.

#### Runtime domain events

Runtime domain events are typed Rust events with logical identities and
backend-neutral meaning. They may originate from native facts, Rust-owned
operations, or reconciliation of both.

#### Consumer projections

Consumer projections include:

- state transitions;
- output events;
- telemetry spans and measurements;
- management API records; and
- optional diagnostic recordings.

Consumer projections MUST NOT be fed back as native facts.

### 7.2 Common envelope

Every runtime domain event MUST support the relevant subset of:

- schema version;
- event ID;
- event category and kind;
- producer source;
- severity;
- wall-clock timestamp;
- process monotonic timestamp;
- native monotonic timestamp, when present;
- native sequence number, when present;
- operation ID;
- logical model ID;
- topology ID;
- stage ID and stage index;
- session ID;
- request ID;
- node-local device ID;
- previous and current state, for state-transition events;
- progress current, total, and unit;
- outcome and stable reason code;
- duration;
- bounded numeric summaries; and
- a bounded human-readable summary.

Fields not meaningful for an event MUST be absent rather than filled with
ambiguous sentinel values.

### 7.3 Identity rules

- Native pointer addresses MUST NOT be used as durable model or session IDs.
- Rust SHOULD assign an operation ID before invoking an instrumented native
  operation.
- Logical model IDs MUST use the model identity already understood by the
  owning Rust runtime.
- Session and request identifiers MUST be generated or normalized by Rust.
- Stage index zero MUST remain distinguishable from an absent stage identity.
- Correlation IDs MUST not be derived from prompt or completion contents.

### 7.4 Outcome and reason codes

Terminal and warning events MUST use stable machine-readable reason codes.
Human-readable detail is supplementary and MUST NOT be parsed to drive state.

Reason codes SHOULD distinguish at least:

- invalid configuration;
- unsupported capability;
- missing artifact;
- artifact I/O failure;
- model format or model-load failure;
- backend initialization failure;
- device unavailable;
- resource allocation failure;
- out of memory;
- context exhausted;
- stage unavailable;
- timeout;
- cancellation;
- process crash;
- incompatible ABI or feature set;
- internal runtime failure; and
- unknown failure.

## 8. Event Families

The event families below define the required high-level vocabulary. Exact Rust
type names and numeric ABI values are left to implementation planning, but the
behavioral distinctions MUST be preserved.

### 8.1 Native runtime lifecycle

Required events:

- runtime resolution started;
- runtime resolution completed or failed;
- native library loaded, rejected, or unavailable;
- ABI and feature compatibility established or failed;
- runtime initialized;
- runtime stopping;
- runtime stopped; and
- runtime crashed.

Useful elements:

- runtime artifact identity;
- ABI version;
- advertised feature set;
- selected backend class;
- compatibility outcome; and
- failure reason.

These events support node startup diagnostics and distinguish model failure
from absence of a usable native runtime.

### 8.2 Model acquisition and preparation

Required events:

- model queued;
- model resolution started/completed/failed;
- model download started/progress/completed/failed/cancelled;
- model materialization or package preparation
  started/progress/completed/failed/cancelled; and
- model preparation completed.

Useful elements:

- logical model ID;
- source class without exposing credentials or private URLs;
- downloaded and total bytes;
- artifact or package count;
- cache reuse outcome; and
- duration.

These events are primarily Rust-owned and SHOULD reuse existing model catalog
and materialization observations.

### 8.3 Model loading

Required events:

- model load requested;
- model load started;
- model load phase changed;
- model load progress;
- backend or device selected;
- model memory allocation summary or pressure;
- native model load completed;
- model load failed; and
- model load cancelled.

Model loading SHOULD use the following coarse phases:

- preparing;
- reading metadata;
- loading tensors;
- assigning or offloading tensors;
- initializing execution context;
- initializing cache or recurrent state;
- initializing auxiliary model components; and
- finalizing.

Useful elements:

- progress current, total, and unit;
- bytes, tensors, or normalized steps where available;
- selected backend and device;
- coarse device placement or offload summary;
- allocated model/context/cache memory;
- phase duration;
- recoverability; and
- failure reason.

The normal event stream MUST NOT emit individual tensor, tensor-name, layer, or
kernel events.

### 8.4 Model availability and readiness

Required events:

- native model loaded;
- Rust backend initialization started;
- model available;
- model degraded;
- model unavailable;
- model recovery started/completed/failed; and
- model capacity changed.

Availability reasons SHOULD include:

- loading;
- ready;
- draining;
- resource constrained;
- backend failure;
- stage unavailable;
- unloading; and
- failed.

Native model-load completion MUST NOT imply model availability. Rust emits
model availability only after all required Rust serving surfaces and dependent
stages are usable.

### 8.5 Model unloading

Required events:

- unload requested;
- unload started;
- session draining started/completed;
- unload completed;
- unload failed; and
- forced unload.

Useful elements:

- active and draining session counts;
- unload reason;
- duration;
- resources released, when cheaply available; and
- forced-cleanup reason.

### 8.6 Stage and topology lifecycle

Required events:

- stage starting;
- stage loading;
- stage ready;
- stage degraded;
- stage unavailable;
- stage stopping;
- stage stopped;
- stage failed;
- topology assembling;
- topology ready;
- topology degraded;
- topology unavailable; and
- upstream or downstream stage connection
  established/lost/recovered.

Useful elements:

- topology and stage identity;
- layer range;
- required and available peer stages;
- connection state;
- stage capacity;
- degraded or unavailable reason; and
- recovery duration.

These events support split-serving readiness and MUST be projected into
backend-neutral model availability.

### 8.7 Session lifecycle

Required events:

- session requested;
- session created;
- session active;
- session idle or reusable;
- session reset;
- session trimmed;
- session restored from prefix cache or checkpoint;
- session draining;
- session closed;
- session failed; and
- session abandoned or reclaimed.

Useful elements:

- session identity;
- owning model and stage;
- lane identity;
- current token count;
- restored or trimmed token count;
- active/idle duration;
- close reason; and
- failure reason.

### 8.8 Request and admission lifecycle

Required events:

- request received;
- request queued;
- request admitted;
- request rejected;
- request execution started;
- request completed;
- request cancelled;
- request timed out; and
- request failed.

Useful elements:

- internal request ID;
- logical model ID;
- selected topology or serving mode;
- queue depth and queue wait;
- execution duration;
- outcome;
- rejection, cancellation, timeout, or failure reason; and
- retry/attempt count where owned by Rust.

No prompt, completion, tool argument, or media content may be included.

### 8.9 Prompt processing and prefill

Required events:

- prompt processing started;
- tokenization completed or failed;
- prefill started;
- prefill progress;
- prefill completed;
- prefill cancelled;
- prefill failed;
- media prefill started/completed/failed when applicable; and
- prompt-cache restore hit/miss/partial/error.

Useful elements:

- input token count;
- cached or restored token count;
- tokens requiring computation;
- completed and total tokens;
- chunk count;
- duration;
- coarse prompt throughput;
- cache outcome; and
- failure or cancellation reason.

Normal production events SHOULD aggregate prefill chunks. Per-chunk events MAY
exist in an explicit debug class, but availability and telemetry summaries MUST
not require them.

### 8.10 Decode and generation

Required events:

- generation started;
- first token produced;
- generation progress;
- generation completed;
- generation cancelled;
- generation timed out;
- generation failed; and
- stop condition reached.

Useful elements:

- generated token count;
- time to first token;
- elapsed generation duration;
- coarse decode throughput;
- stop reason;
- whether speculative or native draft execution was active;
- batching mode; and
- failure or cancellation reason.

Production generation progress MUST be aggregated by a bounded time or token
interval. Per-token text, logits, or token IDs MUST NOT appear in the normal
event stream. Optional internal token observations MUST use an explicit debug
class with sampling or batching.

### 8.11 KV and runtime-state lifecycle

Required events:

- KV/cache initialization started/completed/failed;
- cache lookup hit/miss/partial/error;
- prefix or checkpoint restored;
- cache record completed/failed;
- cache trim or eviction;
- cache reset;
- cache pressure crossed/cleared;
- context capacity approaching limit;
- context exhausted; and
- runtime state import/export completed/failed.

Useful elements:

- current occupancy or token count;
- capacity;
- restored or cached token count;
- evicted token or session count;
- pressure level;
- affected session count;
- duration; and
- reason.

Cache event cardinality MUST be bounded by model, stage, and operation scope.
Cache keys and prompt-derived hashes MUST NOT be exported through the normal
event envelope.

### 8.12 Backend, device, and resource health

Required events:

- backend initialization started/completed/failed;
- device selected;
- device ready;
- device degraded;
- device unavailable;
- device recovered;
- resource allocation completed/failed;
- memory pressure crossed/cleared;
- out-of-memory condition;
- backend fallback activated;
- CPU fallback activated;
- compute failure; and
- device lost or reset.

Useful elements:

- backend class;
- Rust-owned node-local device identity;
- selected device count;
- available, allocated, and requested memory;
- coarse offload or placement mode;
- recoverability;
- fallback target; and
- failure or fallback reason.

The public event projection MUST NOT expose native pointer values, raw stable
hardware identifiers, or absolute device paths.

### 8.13 Warnings, recoveries, and errors

Required events:

- warning raised;
- warning cleared;
- recoverable native failure;
- fallback applied;
- degraded operation entered;
- degraded operation exited;
- fatal native failure; and
- invariant or protocol violation.

Every structured warning or error MUST include:

- stable category;
- stable machine-readable code;
- severity;
- scope;
- bounded human-readable summary;
- recoverable flag;
- operation or resource correlation when available; and
- resulting fallback or degraded state when applicable.

Rust MUST NOT parse the human-readable summary to determine behavior.

### 8.14 Node serving availability

Required derived state events:

- node starting;
- node accepting requests;
- node degraded;
- node unavailable;
- node draining;
- node stopped;
- available model set changed;
- available stage set changed;
- request capacity changed;
- lane or session capacity changed; and
- resource pressure changed.

These events are reducer-owned Rust events. They support routing, management
API status, UI presentation, and any bounded availability projection shared
with peers.

Remote nodes MUST receive only the existing protocol-compatible availability
projection required by routing. This specification does not authorize
gossiping the internal event stream.

### 8.15 Event-system health

Required events or counters:

- ingress queue pressure;
- events coalesced;
- events sampled;
- events dropped by event class;
- subscriber lagging;
- subscriber disconnected;
- reducer error;
- telemetry exporter degraded/recovered;
- event schema incompatibility; and
- unknown native event received.

Event-system health reporting MUST itself be rate-limited and MUST NOT create a
recursive event storm.

## 9. Delivery Classes and Backpressure

One queue policy is insufficient for all event families. Events MUST be
classified into at least the following service classes.

### 9.1 Terminal lifecycle

Examples:

- operation completed or failed;
- model available or unavailable;
- stage ready or failed;
- fatal native failure.

Rules:

- maintain reserved ingress capacity or an equivalent bounded priority
  mechanism;
- preserve per-operation order where practical;
- never rely solely on callback delivery for authoritative completion;
- reconcile terminal state from the operation result; and
- record terminal delivery failure.

### 9.2 State transitions

Examples:

- model degraded;
- device unavailable;
- session created or closed;
- cache pressure crossed.

Rules:

- preserve the latest state for each bounded resource key;
- coalesce redundant identical transitions;
- avoid unbounded histories in the control plane; and
- make current state queryable independently of event retention.

### 9.3 Progress

Examples:

- model load progress;
- download progress;
- prefill progress;
- generation progress.

Rules:

- coalesce to the most recent value per operation;
- suppress duplicate or regressive values unless regression is itself an
  error;
- rate-limit presentation and export;
- allow intermediate progress to be dropped; and
- preserve terminal completion independently.

### 9.4 Diagnostics

Examples:

- recoverable fallback details;
- detailed memory summaries;
- optional chunk or token observations.

Rules:

- diagnostics are droppable;
- high-rate diagnostics require sampling or batching;
- diagnostic loss MUST be counted;
- diagnostics MUST never block model loading or inference; and
- diagnostic subscribers MUST not own readiness or state.

## 10. Ordering and Concurrency

- Ordering is guaranteed per operation where the producer supplies a sequence.
- No global total order across independent models, stages, sessions, or
  requests is required.
- Native sequence numbers MUST be retained but MUST NOT be assumed globally
  unique.
- Rust event IDs SHOULD be process-unique and monotonically assigned.
- Consumers MUST tolerate missing progress and diagnostic events.
- Consumers MUST tolerate a terminal Rust result without a native terminal
  observation.
- Consumers MUST tolerate unknown event kinds and newer append-only fields.
- Concurrent native callbacks MUST not access an unsynchronized mutable Rust
  closure.
- The ingress implementation MUST be safe for callbacks from multiple native
  threads.

## 11. State Reduction

The event system MUST provide or feed a Rust-owned reducer that maintains
bounded current state for:

- native runtime compatibility and health;
- model loading and availability;
- stage and topology readiness;
- model and node request capacity;
- active, idle, draining, and failed sessions;
- request admission and inflight counts;
- KV/cache pressure;
- device and resource health; and
- event-system health.

The reducer MUST:

- use authoritative Rust operation outcomes for terminal state;
- reject invalid state regressions or record them as diagnostics;
- remain correct when progress and diagnostic events are dropped;
- make current state queryable without replaying an unbounded history;
- keep backend-specific details internal;
- emit backend-neutral state transitions to public consumers; and
- avoid holding locks used by native callback ingress.

Readiness is advanced only by Rust-owned reducer transitions after all required
dependencies are ready.

## 12. Consumer Contracts

### 12.1 CLI, TUI, and JSON output

Presentation consumes typed events or reducer transitions and formats them at
the edge.

Output MUST:

- preserve stable machine-readable event names in JSON;
- avoid reducing structured events to generic `info` or `warning` names before
  JSON serialization;
- rate-limit progress display;
- keep visibility events from directly advancing readiness; and
- continue to work when telemetry is disabled.

### 12.2 Management API

The management API SHOULD expose:

- current reduced state;
- bounded recent lifecycle and warning events;
- active model-load progress;
- model, stage, and topology availability;
- current request and session capacity; and
- event-system drop/pressure summaries.

The management API MUST NOT expose the unbounded internal stream or sensitive
event fields.

### 12.3 Telemetry

Telemetry is an optional consumer.

Rules:

- no configured exporter means telemetry consumption is a no-op;
- exporter failures never fail startup or inference;
- telemetry transformation occurs outside the native callback;
- event attributes follow the existing telemetry privacy allowlist;
- high-cardinality request and session IDs are limited to request-scoped spans,
  not aggregate metrics;
- progress is summarized rather than exported at native callback frequency;
  and
- telemetry drop/export counters remain visible locally.

### 12.4 Diagnostics

Optional diagnostic consumers MAY:

- record bounded structured event traces;
- retain raw native logs in an explicit debug file;
- subscribe to sampled debug events; and
- correlate callback facts with authoritative outcomes.

Diagnostic consumers MUST be removable without changing state correctness.

## 13. Privacy and Cardinality

The normal event system MUST NOT collect or export:

- prompt or completion text;
- tool arguments or results;
- request bodies;
- media contents;
- prompt-derived cache keys or hashes;
- token text, token IDs, or logits;
- absolute local paths;
- private repository credentials or signed URLs;
- native pointer addresses;
- raw stable hardware identifiers;
- unbounded arbitrary attribute maps; or
- arbitrary upstream log lines as structured state.

Bounded numeric summaries such as token counts, durations, memory sizes,
capacity, queue depth, and progress are permitted.

Logical model references and device identities MUST follow existing
normalization and telemetry privacy rules before export.

## 14. ABI and Compatibility

Native event extensions MUST be append-only whenever possible.

Requirements:

- add a native feature bit for each independently optional event family or one
  versioned capability group where the family is inseparable;
- bump the Skippy ABI version when the native event ABI changes;
- keep native and Rust ABI constants synchronized;
- use explicit integer discriminants at the FFI boundary;
- represent unknown values safely in Rust;
- validate incoming event ABI version and struct size;
- accept larger compatible structs by reading only known fields;
- copy borrowed strings during the callback;
- preserve legacy no-event entrypoints;
- fall back to authoritative return values when event symbols or feature bits
  are absent; and
- do not require all optional event-family symbols before enabling an unrelated
  supported family.

Mixed-version mesh protocol behavior is unchanged. Internal runtime events are
not added to gossip by this specification.

## 15. Native Log Parser Retirement

The current native log parser may be removed from normal-runtime output and
state only after structured events cover the following:

- model load start, phase, progress, completion, and failure;
- backend initialization and selected device;
- model/context/cache memory allocation and pressure;
- tensor loading and offload at a coarse summary level;
- KV/cache initialization and pressure;
- tokenizer or required auxiliary component readiness/failure at a high level;
- recoverable warnings and fallback;
- fatal native errors; and
- authoritative Rust-side operation completion.

Retirement rules:

- parsed native log messages MUST first stop affecting node state;
- normal CLI/TUI/JSON progress MUST switch to structured events;
- telemetry MUST switch to structured events or Rust-owned operation spans;
- a compatibility period MAY retain parsed logs as debug-only visibility;
- parser-specific progress and metadata state MUST then be removed;
- raw native log redirection MAY remain for explicit debugging; and
- tests MUST prove normal lifecycle and availability work with native log
  forwarding disabled.

## 16. Migration Plan

### Phase 1: Rust event core

- Introduce typed runtime domain events.
- Add bounded ingress and dispatch.
- Add service classes, coalescing, and drop counters.
- Preserve the full current native event envelope.
- Route existing five model-open callbacks through the new ingress.
- Reconcile terminal model-open outcomes from ABI returns.
- Keep existing presentation behavior through an adapter.

### Phase 2: State reducer and consumer separation

- Add backend-neutral model, stage, session, resource, and event-system state.
- Move output formatting out of callback closures.
- Feed CLI/TUI/JSON and management API from typed events or reducer
  transitions.
- Convert telemetry into an optional consumer.
- Ensure no consumer can block native ingress.

### Phase 3: Native model and resource coverage

- Add structured model-load phases and progress.
- Add backend/device and resource allocation events.
- Add KV/cache initialization and pressure events.
- Add structured warning, fallback, and fatal-error events.
- Add model unload observations where native facts are required.

### Phase 4: Inference lifecycle

- Add session lifecycle events.
- Add prompt/prefill lifecycle and progress.
- Add generation start, first-token, aggregated progress, completion,
  cancellation, timeout, and failure.
- Add request and cache summaries from Rust where Rust already owns the facts.

### Phase 5: Native log parser retirement

- Compare structured event coverage against parsed-log output.
- Switch all normal consumers to structured events.
- Disable parsed logs in normal state and output paths.
- Retain raw logs only for opt-in debugging.
- Remove parser aggregation and parser-specific tests after compatibility
  evidence is complete.

## 17. Testing Requirements

### 17.1 Native ABI tests

Tests MUST cover:

- event layout and discriminants;
- reporter ABI version and struct-size validation;
- single-path and multipart model loading;
- actual callback order for success and handled failure;
- missing optional events;
- callbacks from permitted native threads;
- unknown event kinds and larger compatible structs;
- callback detail lifetime and Rust copying;
- no callback after native return; and
- feature and symbol fallback.

At least one integration test MUST invoke the real patched native runtime event
entrypoint rather than only simulating raw callbacks in Rust.

### 17.2 Ingress and backpressure tests

Tests MUST cover:

- concurrent callback ingress;
- full progress and diagnostic queues;
- terminal capacity under progress pressure;
- progress coalescing;
- sampling and drop accounting;
- subscriber lag and failure;
- dispatcher shutdown and draining; and
- no blocking on native producer threads.

### 17.3 Reconciliation tests

Tests MUST cover:

- callback success plus successful return;
- callback failure plus failed return;
- missing terminal callback;
- callback success plus failed return;
- callback failure plus successful return;
- process crash or worker termination;
- old native runtime without event support; and
- event loss while final state remains correct.

### 17.4 Reducer tests

Tests MUST cover:

- model availability only after Rust readiness;
- stage/topology degradation and recovery;
- session count and capacity transitions;
- KV and resource pressure transitions;
- invalid or duplicate transitions;
- unknown future events;
- dropped progress;
- model unload and forced cleanup; and
- node availability derived from multiple model and stage states.

### 17.5 Performance tests

Performance validation MUST compare:

- model loading with events disabled and enabled;
- normal inference with event consumers disabled and enabled;
- aggregated generation progress versus optional debug token observations;
- slow or failed telemetry consumers;
- full diagnostic queues; and
- multiple simultaneous model or session producers.

The production event configuration MUST not materially regress decode
throughput or time-to-first-token. Exact thresholds are left to the benchmark
plan, but callback ingress latency and drop/coalescing behavior MUST be
measured.

## 18. Acceptance Criteria

The event-system migration is complete when:

- native callback work is limited to validation, copying, correlation, and
  nonblocking ingress;
- arbitrary application closures are no longer invoked inline from native
  callbacks;
- typed Rust events preserve the full native envelope;
- terminal native operations are reconciled from return values;
- model load, availability, unload, stage, session, prefill, generation,
  KV/cache, backend/resource, warning, and node-availability families are
  represented at the high level defined here;
- state remains correct when progress and diagnostic events are dropped;
- CLI/TUI/JSON output no longer depends on parsing native logs;
- normal node state no longer depends on parsing native logs;
- telemetry is an optional nonblocking consumer;
- event pressure, coalescing, sampling, and drops are locally observable;
- mixed-version native runtimes retain a safe no-event fallback;
- public state remains backend-neutral;
- privacy and cardinality constraints are tested; and
- raw native logs are debug-only.

## 19. Implementation Boundaries

This specification intentionally leaves the following to implementation
planning:

- exact Rust crate and module placement;
- exact event enum and reducer type names;
- queue implementation;
- queue capacities;
- progress coalescing intervals;
- debug sampling rates;
- recent-event retention limits;
- management API response shape;
- exact native feature-bit grouping; and
- exact performance regression thresholds.

Those choices must preserve the ownership, boundedness, compatibility,
privacy, and terminal-correctness requirements defined above.
