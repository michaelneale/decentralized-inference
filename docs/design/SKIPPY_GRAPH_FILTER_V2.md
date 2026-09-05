# Generic Skippy Graph-Derived Stage Splitting Plan

Status: proposed for sign-off.

## Decision Request

Approve replacing Skippy's three independent descriptions of a stage with one
graph-derived stage plan:

1. Build the normal unsplit `ggml_cgraph` in metadata-only mode.
2. Partition it using generic, stage-independent block and state metadata.
3. Derive the executable node set, exact tensor dependency closure, activation
   boundary, and state ownership from that partition.
4. Load only the derived tensor set from a new model-package format.
5. Validate the realized stage before topology publication.

No model family names, architecture enums, tensor-name heuristics, or mutable
filtering diagnostics may participate in stage selection.

## Problem Statement

Today the same `[layer_start, layer_end)` request is interpreted independently
by three systems:

- package planning chooses artifacts using layer, role, endpoint, and name
  heuristics;
- native loading filters tensors using coarse ownership classes and family
  exceptions;
- model builders contain copied stage-range, activation-input, output, and
  early-return branches.

Correctness depends on all three interpretations agreeing. Granite exposed one
generic failure mode: mutable loader state suppressed separate Q/K/V weights
while the graph still referenced them. Other families can diverge through
shared weights, experts, recurrent state, sidebands, hyper-connections, tied
outputs, or optional execution paths.

## Proposed Design Invariants

These invariants are non-negotiable implementation gates:

1. **One computation graph.** Reuse the actual `ggml_cgraph`; do not create a
   parallel model IR that can drift from execution.
2. **No stage-aware model builders.** Builders construct their normal unsplit
   computation. They never receive a stage range and never choose stage input,
   output, or weight ownership.
3. **Stage-independent annotations only.** Builders or common graph helpers may
   register block input/output, layer ordinal, aliases, and persistent-state
   effects because those are properties of the unsplit computation.
4. **No model-aware splitter.** The partitioner cannot inspect family names,
   architecture enums, tensor names, fused/separate projection types, or
   model-specific flags.
5. **Exact dependency closure.** Every parameter referenced by a retained op is
   loaded exactly once with matching identity, dtype, and shape.
6. **Effects are dependencies.** KV/recurrent reads and writes, aliases,
   ordering constraints, and lifetimes participate in the slice closure.
7. **Boundaries come from crossing edges.** Activation and state planes are
   derived from graph edges crossing the selected block frontier, not declared
   by family policy.
8. **Persistent state stays local.** KV or recurrent state remains owned by the
   stage executing the relevant layers unless a real graph edge crosses the
   cut.
9. **Plans cover execution modes.** The realized dependency set is the
   conservative union across supported prefill, decode, batch, speculative,
   and optional paths. A single traced shape cannot certify a stage.
10. **Legal cuts are derived.** A numbered layer boundary is not automatically
    legal. Unsupported boundaries fail closed with a structured reason.
11. **Validation precedes publication.** No partial or inconsistent stage is
    advertised to the topology.
12. **Deterministic planning.** The same package identity, runtime profile, and
    requested range produce the same normalized plan and digest independent of
    tensor-probe order.

## Target Data Flow

```mermaid
flowchart LR
  P["Model package v2\nfull metadata and tensor index"]
  B["Normal model builder\nno stage knowledge"]
  G["Full ggml_cgraph\nno_alloc"]
  M["Block boundaries\nstate, effect, alias metadata"]
  U["Union of supported\nexecution profiles"]
  R["Requested layer range"]
  S["Generic partitioner"]
  X["Sliced executable graph"]
  D["Exact tensor dependencies"]
  A["Derived boundary ABI"]
  L["Materialize and load"]
  V{"Closure and ABI valid?"}
  T["Publish topology and run"]
  F["Structured rejection"]

  P --> B --> G --> U --> S
  B --> M --> S
  R --> S
  S --> X --> V
  S --> D --> L --> V
  P --> L
  S --> A --> V
  V -- yes --> T
  V -- no --> F
```

## Workstream 0: Baseline and Ownership

- Treat Astrid's immediate QKV correction as a separate correctness fix; do not
  duplicate or block it in this structural project.
- Start the structural work from the integration commit containing the current
  staged-runtime behavior and the immediate correctness fix.
- Record exact baseline commits for Mesh and the prepared llama.cpp tree.
- Capture current package corpus identities, supported-family registry, stage
  protocol generation, and end-to-end parity results.
- Use a dedicated worktree. Do not modify the default branch.

**Exit gate:** reproducible baseline and no ambiguous ownership with the
Granite fix or activation-plane work.

## Workstream 1: Package Format v2

Create a new package format and make it the only accepted format for the new
runtime path. Do not add long-lived v1 runtime compatibility.

### Manifest contents

- package schema version and immutable package identity;
- source model identity and complete model hyperparameter metadata required for
  `no_alloc` graph construction;
- artifact table: stable artifact id, path/reference, byte size, and SHA-256;
- tensor table with one entry per source tensor:
  - stable tensor identity/name;
  - dtype and complete dimensions;
  - optional stage-independent layer ordinal used for physical organization and
    diagnostics, never for dependency correctness;
  - artifact id, data offset, stored length, alignment, and tensor checksum or
    artifact-bound integrity proof;
- tokenizer, projector, and generation-sidecar identities where applicable;
- Skippy native ABI requirements and package-generator version.

### Physical layout

- Per-layer GGUF artifacts may remain because they are efficient physical
  containers.
- Replace semantic `embeddings` and `output` ownership assumptions with generic
  shared artifacts. Physical grouping cannot decide stage ownership.
- Every source tensor appears exactly once in the catalog unless an explicit
  alias record proves shared storage.
- Package creation is independent of stage count and cut locations.

### Creation and validation

- Generate artifacts and the catalog from the same source inventory.
- Re-open written artifacts and compare exact tensor identity, dtype, shape,
  offsets, and integrity—not only tensor count and aggregate bytes.
- Build the metadata-only unsplit graph from the package catalog as a package
  certification step.
- Reject missing, duplicate, mismatched, overlapping, or out-of-bounds tensor
  records.
- Provide an optional offline v1-to-v2 converter. It may reuse existing GGUF
  bytes only when a full scan proves exact coverage; otherwise it refuses with
  the missing tensor list and requires a rebuild.

**Exit gate:** v2 packages can reconstruct the full metadata model without
reading weight payloads, and the current certification corpus has been rebuilt
or explicitly converted.

## Workstream 2: Metadata-Only Unsplit Graph

- Prototype the existing llama.cpp `no_alloc` path against representative
  models. Prove that it creates all required tensor handles without reading or
  allocating weight payloads.
- Build the exact normal graph used for execution rather than a second model
  description.
- Remove the stage filter from planning builds; every model builder emits its
  unsplit computation.
- Add a planning profile matrix covering at minimum:
  - prompt prefill;
  - single-token decode;
  - multi-sequence/batched execution;
  - enabled speculative/MTP paths;
  - enabled multimodal or other optional paths that affect the language graph.
- Normalize and union graph facts across profiles. Reject unstable or
  unexplained shape-dependent tensor closures.
- Measure planning time and metadata memory. Assert that no full-model weight
  allocation or KV allocation occurs merely to plan.

**Exit gate:** the same executable graph machinery builds in metadata-only mode
for the certification fixtures, with bounded planning resources and no weight
payload reads.

## Workstream 3: Generic Graph Semantics

Extend the executable graph with only the semantics required for a correct
partition. Do not encode staging policy.

### Block structure

- Register a canonical input and output boundary for every transformer or
  recurrent block with its layer ordinal.
- Prefer common builder/context helpers such as `begin_block` / `end_block`.
  Model-file edits may be mechanical registration only.
- Add a validation/lint gate: every declared layer has exactly one ordered block
  entry and exit for each supported graph profile.
- Do not infer block boundaries from node names.

### Parameters and aliases

- Identify parameter leaves by stable catalog identity.
- Preserve view/base relationships and storage aliases.
- Ensure an alias cannot make an unloaded base allocation reachable.

### Effects and persistent state

- Represent KV/recurrent state reads, writes, and ordering dependencies in the
  graph or in a side table owned by that exact `ggml_cgraph`.
- Attach each persistent state component to its stage-independent layer owner.
- Distinguish persistent local state from values that genuinely cross a stage
  boundary.
- Reject any family whose necessary state remains invisible to the planner.

**Exit gate:** generic metadata fully describes block frontiers, parameters,
aliases, and effects for the fixture set. No family-specific splitter code is
introduced.

## Workstream 4: Pure Graph Partitioner

Implement the partitioner as a deterministic pure function of:

```text
(normalized graph profiles, package tensor catalog, requested range,
 runtime capabilities) -> StageSlicePlan | UnsupportedStage
```

### Algorithm

1. Resolve the requested start and end to registered block frontiers.
2. Treat graph values produced before the start frontier and consumed inside
   the slice as typed imports.
3. Retain computation from the start frontier through the end frontier.
4. Preserve all effect nodes, ordering edges, alias bases, and lifetimes needed
   by the retained computation.
5. Treat values produced inside and consumed after the end frontier as typed
   exports.
6. Walk the retained graph to derive exact parameter identities.
7. Union dependencies and boundary shapes across all supported graph profiles.
8. Classify persistent state ownership from registered state effects.
9. Validate that every dependency resolves exactly once in the package catalog.
10. Normalize and hash the plan.

### `StageSlicePlan`

- descriptor version;
- requested and realized layer range;
- legal-cut result and structured rejection reason;
- retained graph/profile identifiers;
- exact sorted tensor dependency identities and digest;
- resident parameter bytes;
- typed input and output boundary planes;
- persistent KV/recurrent state ownership and geometry;
- supported execution profiles, sequence/batch limits, and backend constraints;
- plan digest bound to package identity and runtime ABI.

The partitioner cannot call or depend on model-family helpers.

**Exit gate:** unit/property tests prove closure, deterministic output, boundary
completeness, and effect preservation for valid and invalid cuts.

## Workstream 5: Exact Realization and Loading

- Resolve `StageSlicePlan.tensor_dependencies` through the v2 tensor catalog.
- Initially materialize a stage-local GGUF if that minimizes changes to the
  existing loader; direct callback/range loading can follow without changing
  the plan contract.
- During real graph construction, create metadata tensor handles for the full
  model but allocate/read payloads only for the dependency closure.
- Instantiate the executable slice from the same normalized plan used for
  loading.
- Make representation selection structural: the unsplit graph references fused
  QKV or separate Q/K/V according to what exists; the loader never makes that
  choice.
- Return the realized descriptor through the native ABI.
- Before topology publication, verify:
  - every retained op operand is resolved;
  - every loaded tensor is in the closure;
  - no closure tensor is absent;
  - dtype, shape, alias, and backend placement match;
  - input/output plane descriptors match neighboring stages;
  - state ownership and supported operations are complete.
- Fail with a structured `UnsupportedStage` / `InvalidPackage` error, never an
  assertion or null operand during graph execution.

**Exit gate:** load-time tensor identity exactly equals planned closure and all
negative fixtures fail before topology publication.

## Workstream 6: Protocol and ABI

### Native ABI

- Add a versioned realized-stage descriptor carrying the normalized
  `StageSlicePlan` fields needed by the host.
- Prefer additive ABI entry points during development; remove obsolete
  stage-filter APIs only at final cutover.

### Network protocol

- Do not block the internal design on a wire change.
- Continue stage protocol generation 7 when the derived boundary is exactly
  representable as the current single raw-F32 activation plus supported flags.
- Fail closed when a derived boundary cannot be represented by generation 7.
- In a separate reviewed workstream, add a negotiated protocol generation for
  typed activation bundles:
  - repeated plane descriptors with stable semantic ids;
  - dtype, layout, dimensions/strides, byte offset, and byte length;
  - bounded payload framing and integrity validation;
  - exact producer/consumer descriptor matching.
- Preserve mixed-version meshes through capability negotiation until a
  coordinated protocol cutoff is explicitly approved. This is required by
  repository policy even though package v1 compatibility is intentionally not
  retained.

**Exit gate:** current single-plane models run unchanged on generation 7;
multi-plane models are either negotiated correctly or rejected before topology
publication.

## Workstream 7: Migration and Deletion

Use a shadow-first migration so the old path can be compared, then deleted.

1. Add the new planner behind a development-only switch.
2. In shadow mode, build the current stage and independently compute the new
   plan. Compare exact tensor identities, realized boundaries, state ownership,
   and resident bytes without changing execution.
3. Enable execution from the new plan for compact dense fixtures.
4. Expand to fused-QKV, separate-QKV, MoE/shared-weight, hybrid/recurrent,
   hyper-connected, sideband, MTP, and multimodal fixtures.
5. Run the full registry canary and every-cut certification.
6. Rebuild or convert the production package corpus to v2.
7. Atomically cut over the runtime to v2 plus graph-derived planning.
8. Delete:
   - thread-local and mutable stage-filter diagnostics;
   - loader family retention branches;
   - package role/name stage-selection policy;
   - stage-range branches and stage-boundary early returns from model builders;
   - development shadow mode and v1 package acceptance.

Rollback is by reverting to the previous binary and v1 package corpus, not by
shipping two permanent runtime implementations.

**Exit gate:** searches over the splitter, loader, package planner, and model
builders find no model-family staging policy and no stage-filter control flow.

## Workstream 8: Verification Matrix

### Pure partitioner tests

- first, middle, final, single-layer, and invalid cuts;
- parameter-free ops at boundaries;
- residual, skip, fan-out, and alias edges;
- effect-only KV/recurrent writes;
- deterministic output under tensor and node enumeration changes;
- closure union across prefill/decode/batch/optional profiles;
- unsupported hidden state and unrepresentable boundary failures.

### Package tests

- exact full-source tensor coverage;
- missing, duplicate, mismatched dtype/shape, corrupt offset, alias, and checksum
  failures;
- v1 rejection with an actionable conversion command;
- v1-to-v2 conversion success only for complete packages;
- no weight payload reads during metadata planning.

### Native/runtime integration fixtures

- dense separate-QKV model, including Granite or Granite-Hybrid;
- fused-QKV model;
- MoE and shared-expert model;
- recurrent and hybrid model with persistent state;
- sideband/hyper-connected model;
- MTP/speculative path;
- multimodal language path where it changes graph dependencies.

For every legal cut:

- package load succeeds;
- one prompt prefill succeeds;
- multiple decode steps succeed;
- split and unsplit logits/selected tokens satisfy the existing correctness
  tolerance;
- neighboring boundary descriptors match exactly;
- planned dependency identities equal loaded identities;
- resident parameter bytes equal the catalog sum;
- state export/import and prefix-cache behavior remain correct where supported.

Run the same behavioral matrix through direct GGUF, callback/SafeTensors, and
model-package v2 sources where those paths are supported.

### Required repository validation

- full tests for every touched Rust package, never module-scoped substitutes;
- native llama.cpp staged-runtime tests and patch-application checks;
- `cargo test -p skippy-ffi --lib` and `cargo test -p skippy-runtime --lib` for
  ABI changes;
- `cargo test -p skippy-protocol --lib` and
  `cargo test -p mesh-llm-host-runtime --lib` for protocol/control changes;
- real two-node inference for user-visible integration behavior;
- mixed-version node validation whenever the network protocol changes;
- upstream llama.cpp pin/canary validation across the supported registry.

## Delivery Sequence

Deliver as reviewable, independently gated changes rather than one giant patch:

1. Metadata-only full-graph feasibility prototype and resource measurements.
2. Package v2 schema, writer, reader, and exact validation.
3. Generic block/state/effect/alias metadata.
4. Pure partitioner plus `StageSlicePlan` and property tests.
5. Exact dependency realization and fail-closed native descriptor.
6. Shadow comparison against the current runtime.
7. Cross-family execution rollout and every-cut certification.
8. Production package rebuild/conversion.
9. Old stage-filter/family-policy deletion and v2-only cutover.
10. Typed multi-plane protocol generation, only if required boundaries cannot
    be represented by generation 7.

Each change must state its base commit, tests, observed fixture coverage, and
whether it changes package, native ABI, or network compatibility.

## Completion Criteria

The project is complete only when all of the following are true:

- package v2 is the production package format and the active corpus is
  available in v2;
- model builders contain no stage ranges, endpoint ownership, or stage early
  returns;
- package planning and native loading contain no family-name or tensor-name
  stage-retention rules;
- the executable graph, tensor closure, boundary ABI, and persistent-state
  ownership all come from one normalized partition result;
- every supported model passes every legal cut through load, prefill, and
  multiple decode steps;
- unsupported models/cuts fail before topology publication with structured
  reasons;
- planning reads metadata only and does not allocate full-model weights or KV;
- current generation-7 boundaries remain operational, and any typed-plane
  generation has explicit compatibility evidence;
- old staging implementation, development flags, and v1 runtime compatibility
  have been removed.

## Non-Goals

- arbitrary cuts inside a transformer/recurrent block;
- tensor, expert, or pipeline parallelism within a single block;
- changing quantization formats;
- changing topology optimization or device-placement policy beyond consuming
  accurate resident-byte and capability descriptors;
- using family certification fixtures as runtime family allowlists.

## Principal Risks and Mitigations

- **`no_alloc` is not sufficient end to end.** Prove it before designing around
  it; stop if planning still requires weight or KV allocation.
- **Graph shape changes dependencies.** Use a conservative profile union and
  reject unexplained instability.
- **Hidden side effects escape reachability.** Make state/effect registration a
  certification gate; do not silently accept invisible state.
- **Boundary annotation becomes staging by another name.** Limit annotations to
  unsplit block structure and effects; prohibit stage ranges and endpoint
  ownership in builders.
- **Package v2 omits legacy bytes.** Exact identity/type/shape validation against
  the source inventory prevents false certification.
- **Large llama.cpp patch burden.** Prefer common graph-context helpers and
  mechanical registration; keep the partitioner outside model files.
- **Protocol work expands scope.** Gate current-compatible models first and run
  typed activation planes as a separate negotiated change.
- **A shadow mismatch is normalized away.** Compare exact tensor identities and
  boundary descriptors; aggregate counts are not evidence.

## Sign-Off Checklist

Approval is requested for these decisions:

- [ ] Existing `ggml_cgraph` is the sole computation representation; no second
      model IR.
- [ ] Model builders may register stage-independent block and state semantics,
      but receive no staging configuration.
- [ ] The splitter, package selector, and loader contain no model-family policy.
- [ ] Model package v2 is a deliberate hard format cutover; no permanent v1
      runtime compatibility.
- [ ] An offline converter may reuse old weight artifacts only after exact
      coverage validation.
- [ ] Legal cuts are discovered and certified; not every numbered boundary is
      promised.
- [ ] Generation-7 wire behavior remains for representable boundaries; typed
      activation planes are a separate negotiated protocol change.
- [ ] Old split logic is deleted only after shadow comparison and the full
      cross-family every-cut gate passes.

No implementation starts under this plan until these decisions are approved.

## Evidence Base

- `crates/skippy-model-package/src/plan.rs`
- `crates/skippy-model-package/src/package.rs`
- `crates/skippy-model-package/src/write.rs`
- `crates/skippy-runtime/src/package.rs`
- `crates/skippy-runtime/src/types.rs`
- `crates/skippy-protocol/proto/stage.proto`
- `crates/skippy-protocol/src/binary/types.rs`
- `third_party/llama.cpp/patches/0001-Add-staged-model-graph-and-family-support.patch`
