# Mixture-of-Agents (MoA) Gateway

Fan out requests to multiple heterogeneous LLM endpoints in parallel,
arbitrate responses with deterministic logic, manage tool call lifecycles,
and return one coherent OpenAI-compatible response.

**Crate:** `crates/mesh-mixture-of-agents/`
**Automatic directive:** `model: "mesh"` (`auto` is a deprecated alias; advertised when a model is available)
**Status:** Integrated into mesh proxy, live-tested with mesh peers

---

## Why MoA exists

If you have N nodes on a mesh, the obvious thing is to shard one large
model across them (Skippy split). That gives the best quality per
VRAM-dollar when the network is good — but the network sits on the
critical path of every output token. Past a certain RTT, loss, or peer-
flakiness, split-large stops producing interactive performance at all.

MoA is the "use the mesh anyway" path for that region. Each worker runs
fully local on one node, so per-token latency stays local. The network
is only touched at fan-out, output collection, and the reducer. A slow
link slightly slows the whole turn instead of stalling every token.

The aim is **not** to beat split-large on quality. The aim is to keep
the mesh useful when split can't be — by mixing several mid-size models
in parallel and arbitrating one coherent answer, at quality at least as
good as a single mid-size local model would have produced.

Split and MoA are complementary, not competitive. Network conditions
decide which is appropriate. See [Diverse-mesh vs split-large: the
operating-envelope question](#diverse-mesh-vs-split-large-the-operating-envelope-question)
for the experimental framing.

---

## How it works

```
       Agent / Goose / Claude Code / pi
              │
              │  POST /v1/chat/completions
              │  { model: "mesh", messages, tools, stream }
              ▼
    ┌──────────────────────────────────────────────────────┐
    │  Mesh proxy (ingress.rs)                             │
    │  - intercepts model == "mesh"                        │
    │  - build_moa_config(callable models in mesh)         │
    └──────────────────────┬───────────────────────────────┘
                           │ handle_turn(config, body)
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  MoA gateway (crate: mesh-mixture-of-agents)         │
    │                                                      │
    │  session.classify_turn()                             │
    │    │                                                 │
    │    ├─ Fresh / Continuation ──► fan-out path          │
    │    └─ ToolResult ────────────► reducer-only path     │
    └────┬─────────────────────────────┬───────────────────┘
         │ fan-out path                │ tool-result path
         ▼                             │
    [LLM call #1]                      │
    assign_roles(N) → fire N workers   │
    in parallel:                       │
                                       │
       ┌─── fast       (smallest)      │
       ├─── specialist                 │
       ├─── specialist  …  (N-2 of)    │
       └─── strong     (biggest)       │
                                       │
    wall-clock = slowest worker        │
         │                             │
         ▼                             │
    [code] arbiter                     │
         │                             │
         ├─ consensus → emit one       │
         │  worker's output  ──────► done (no reducer)
         │                             │
         └─ conflict ─────┐            │
                          ▼            ▼
                       [LLM call #2] reducer
                       hedged candidate ladder
                       (top-2 strongest from same pool)
                       1 call usually, 2 if primary slow
                          │
                          ▼
                       final response
```

The client thinks it talks to one model. `mesh` is a routing directive
like `auto` — the proxy intercepts it before normal model routing.

### How many models, and how many LLM calls

The number of workers is **not fixed** — it scales with whatever's
callable in the mesh.

| Callable models | Workers fanned out | Roles assigned |
|---:|:---:|:---|
| 0 | — | degrades to ordinary model selection; 503 if none exists |
| 1 | 1 | one-worker Mesh gateway |
| 2 | 2 | fast + strong |
| 3 | 3 | fast + specialist + strong |
| 4 | 4 | fast + specialist + specialist + strong |
| N | N | fast + (N-2) specialists + strong |

On an **answer** turn every worker is packed at the full budget rather
than its role budget: the role tiers exist so the cheap worker can answer
the grace fast-path quickly, but a truncated draft is an *input* to
synthesis, so brevity there drags the aggregated answer down.

Tiering comes from **verified parameter counts**, not the alias string.
The serving node sums its GGUF tensor element counts and gossips the
result via `ServedModelMetadata.parameter_count_b`; the gateway reads
that and splits at `SMALL_TIER_MAX_B` (10B). A model with no verified
size ranks small, so an unparseable alias can never pose as a big-tier
worker and displace a real one. Name parsing is no longer used — it
mis-tiered real models (e.g. `gemma-4-E4B` stores 7.5B, not 4B).

Pool shaping, in order (all measured; see
`evals/moa-openrouter/RESULTS.md`):

- **Admission**: verified-small workers are dropped only when ≥2
  verified-big remain. A lone big + smalls keeps the mix, because
  dropping the smalls there would collapse the committee to a solo model.
- **All-small pools do not convene a committee.** Measured through the
  shipped path, an 8B-class pool with an 8B reducer never beat its best
  member and lost about a third of decided trials (2×8B 0W/37L; 6×8B
  5W/23L, p=0.0009). Such a pool collapses to its strongest member and
  runs as a one-worker Mesh gateway. This is a statement about a weak
  reducer synthesizing weak drafts; a capable pool is the opposite
  (71W/8T/1L, p<0.0001). As soon as a mesh gains a big-tier model, the
  pool is no longer all-small and MoA engages.
- **Committee cap**: 6 for all-small, 4 once a verified big is present.
  Fan-out costs ~2N+1 calls per turn and quality is flat past ~4.

**The reducer is not a separate fan-out slot.** It's the same strong
model (or next-strongest if the primary is slow/broken), invoked
*after* fan-out only when the arbiter says workers disagreed.

So serially, a worst-case MoA turn is **2 LLM round-trips**:

1. **Fan-out:** N workers in parallel — wall-clock equals the *slowest*
   worker, not the sum (bounded by `worker_timeout`).
2. **Reducer:** 1 strong model (sequential, blocks the response) — fires
   only on arbiter conflict; hedged to up to 2 overlapping calls if the
   primary stalls.

Happy paths collapse to 1 round-trip:

- **Early-exit / unanimous answer:** workers agree → no reducer.
- **Tool-result turn:** skip fan-out entirely → 1 reducer call.

This shape is why reducer streaming is the right TTFT lever: it's the
one serial LLM call on the critical path that produces user-visible
text. Workers are already parallelized; their wall-clock is bounded by
the slowest one regardless of streaming (arbiter needs full outputs to
compare).

---

## Transport

The crate defines a `ModelBackend` trait and ships a default `HttpBackend`
for standalone/test use. The host runtime implements mesh-native backends:

| Backend | Transport | Where |
|---------|-----------|-------|
| `LocalModelBackend` | Direct HTTP to `127.0.0.1:{skippy_port}` | Local model on this node |
| `RemoteModelBackend` | HTTP-over-QUIC tunnel via `node.open_http_tunnel()` | Remote mesh peer |
| `HttpBackend` | Plain HTTP to any URL | Standalone testing |

Same-model self-fill keeps two distinct physical endpoints per committee.
Concurrent turns at one gateway reserve the least-loaded clones within each
leading context/throughput-equivalent tier; a slower or unknown-context clone
does not displace an available higher-ranked clone merely because it is idle.
Reservations last while the turn's backend references survive, including
worker/reducer calls, and release on completion or cancellation. Self-fill
slots do not fail over onto sibling clones. This is process-local accounting,
not global admission, and does not change heterogeneous-model replica selection.

All worker requests set `mesh_hooks: false` to prevent recursive virtual
LLM consultations (MoA → model → hook → consult another model → ...).

---

## Context packing

Workers get slices of the real agent context, not synthetic prompts. The
agent's actual system prompt, messages, and tool definitions flow through.
The gateway varies depth per role, not content.

| Role | System prompt | Messages | Tools | Max tokens |
|------|:---:|:---:|:---:|:---:|
| Fast | ✅ agent's + one-line preamble | last user msg | names only in system prompt | 256 |
| Specialist | ✅ agent's + one-line preamble | last 4 | name + description | 512 |
| Strong | ✅ agent's + one-line preamble | last 10 | full schemas (native) | 1024 |
| Reducer | ✅ agent's + one-line preamble | worker outputs | full schemas (native) | 2048 |

---

## Arbitration

### Early-exit consensus

Workers are raced in parallel via `gather_workers_incremental()`. After
each response, `try_early_decision()` checks whether enough evidence
exists to return immediately:

| Condition | Action |
|-----------|--------|
| 2+ answers agree, confidence ≥ 0.5 | Return immediately, cancel remaining |
| 2+ tool proposals agree on same tool | Return immediately |
| 1 survivor, all others failed/timed out | Return sole survivor |
| Tool proposal vs answer conflict | Escalate to reducer |
| Low confidence answers | Wait for more workers |

### Deterministic arbiter

The full arbiter (`arbitrate()`) runs when early-exit doesn't fire:

- **Unanimous answers** → highest confidence wins
- **Unanimous tool, same function** → emit tool_call with best arguments
- **Tool vs answer conflict** → escalate to reducer
- **Conflicting tools** → escalate to reducer
- **All uncertain** → escalate to reducer

The reducer step uses a **hedged candidate ladder** (`hedged_reducer_call`)
over the ordered list from `reducer_candidates` — big-tier models first
(multi-digit B, or names with no size), then small-tier as last-resort.

- Start candidate 0 immediately.
- If candidate 0 hasn't replied within `hedge_delay` (5s by default),
  start candidate 1 **alongside** it — don't cancel candidate 0, race them.
- If a candidate errors fast, start the next one immediately (no hedge wait).
- First success wins; cancel the rest.
- All-fail falls back to the best worker output already gathered.

Cost shape: 1 backend call on the happy path (free), up to 2 overlapping
calls when the first is slow, N calls only when everything is failing.
End-to-end wall-clock for the worst case is bounded by
`reducer_timeout + (N-1)·hedge_delay` rather than `N·reducer_timeout`.

---

## Tool calling

Tool results go to reducer only, not re-broadcast to all workers.

```
Turn 1: Client sends "Read README.md" + tools: [read_file]
  → Workers fan out, strong worker proposes read_file({"path":"README.md"})
  → Arbiter: tool consensus → emit tool_call
  → Client gets: tool_calls: [{read_file, {"path":"README.md"}}]

Turn 2: Client sends tool_result with file contents
  → Gateway detects TurnType::ToolResult → reducer only (no fan-out)
  → Reducer synthesizes tool output into final answer
  → Client gets: content: "The README contains..."
```

---

## Normalization

Models produce unreliable output. The normalizer tries in order:

1. **JSON object parse** — structured envelope with kind/confidence/payload
2. **Line-based KV extraction** — `key: value` lines
3. **Heuristic classification** — pattern matching for tool proposals,
   confidence markers, uncertainty signals

Think tags (`<think>...</think>`) and GLM-style reasoning preambles are
stripped throughout the pipeline.

---

## Crate structure

`crates/mesh-mixture-of-agents/` — zero mesh dependency.

| Module | LOC | Tests | Purpose |
|--------|----:|------:|---------|
| `lib.rs` | 454 | 0 | `handle_turn()` orchestration, `GatewayConfig`, `TurnResult`, response builders |
| `backend.rs` | 277 | 6 | `ModelBackend` trait, `HttpBackend`, `SamplingParams`, `call_backend()` |
| `reducer.rs` | 390 | 4 | Reducer candidate ordering + hedged ladder |
| `fanout.rs` | 119 | 0 | Incremental worker gathering with early-exit |
| `arbiter.rs` | 560 | 15 | Deterministic arbitration + early-exit decisions |
| `normalize.rs` | 650 | 13 | 3-tier dirty output parsing |
| `session.rs` | 487 | 4 | Canonical transcript, tool tracking, turn classification |
| `context.rs` | 560 | 8 | Role-shaped context packing |
| `worker.rs` | 298 | 5 | Role assignment, tier sort, think-tag stripping |
| `tool_guard.rs` | 116 | 4 | Allowed-tool enforcement on reducer outputs |

---

## Integration

The MoA intercept lives in `ingress.rs` (~line 234). When `model == "mesh"`:

1. `build_moa_config()` collects all models from `ModelTargets` + `callable`
   list, deduplicates aliases, wraps each in `LocalModelBackend` or
   `RemoteModelBackend`
2. `handle_turn()` runs the stateless MoA pipeline
3. Response is sent as SSE (streaming clients) or plain JSON

Activation: `model=mesh` enters the Mesh gateway whenever at least one worker
is admitted. With two or more eligible workers it can form a committee; with one,
the gateway runs that worker without rewriting the virtual model. Only a
zero-worker request degrades to ordinary real-model selection (and returns 503
if no model is available).

### Buzz terminal-reply rescue

For Buzz agent requests, the gateway recognizes the trusted `[Context]` frame,
its validated channel/reply IDs, and the declared Buzz shell tool. Successful
terminal prose is converted into one deterministic `buzz messages send` tool
call so small models cannot silently finish without publishing their answer.
Intermediate tool calls remain untouched, and the rescue imposes no generic
tool-count limit; the existing repeated-identical-call detector handles actual
loops. Error responses are never converted into channel posts. This behavior is
restricted to the virtual `model=mesh`; pinned concrete-model requests pass
through unchanged.

---

## Relationship to existing systems

| System | What it does | Relationship to MoA |
|--------|-------------|---------------------|
| `auto` / `mesh` | Same automatic directive | Committee-capable chat uses MoA; streaming, media, non-chat, or unavailable workers use ordinary routing |
| Hooks (`virtual_llm.rs`) | Reactive during inference (entropy/drift/image) | MoA is proactive before inference |
| Consult (`consult.rs`) | Single peer consultation over QUIC | MoA does parallel multi-peer |
| Pipeline (`pipeline.rs`) | 2-model plan→execute for code tasks | Complementary, used at ingress line 279 |

Hooks and MoA are independent. Hooks fire reactively during inference.
MoA fires proactively before inference. The two are intentionally kept
separate — worker requests set `mesh_hooks: false` so the hook pipeline
never re-enters a worker call.

---

## Test & evaluation plan

### Unit tests (run in CI)

```bash
cargo test -p mesh-mixture-of-agents --lib
```

Currently 210 unit tests across the crate. Coverage areas:

* `arbiter` voting and decision rules.
* `normalize` parsing across JSON, KV, and heuristic strategies
  (including NaN/Inf confidence sanitization and OpenAI-shape inline
  tool-call extraction).
* `session` turn classification (Fresh vs ToolResult) on caller-provided
  history.
* `worker` role assignment (small vs big tier) and `strip_thinking`.
* `reducer` hedged-attempt scheduling.
* `backend` response extraction and retry-after parsing.
* `lib::response_builder_tests` for `best_answer` NaN safety and
  `tool_call_response` argument coercion.

Integration tests under `crates/mesh-mixture-of-agents/tests/` exercise
fan-out simulation, all-workers-fail handling, tool-result routing, and
worker accounting.

When adding tests, update this count by running
`cargo test -p mesh-mixture-of-agents --lib` and copying the final
`test result: ok. N passed` value here so future readers see the real
number.

### Local smoke test

Start mesh with at least one model + discover mesh peers:

```bash
mesh-llm serve --model "unsloth/GLM-4.7-Flash-GGUF:Q4_K_M" --auto
```

Wait for ≥2 models in `curl -s http://localhost:9337/v1/models`.

**Factual accuracy:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is the capital of Japan? One word only."}],"max_tokens":32,"stream":false}'
```
Expected: "Tokyo" (possibly with reasoning preamble from GLM).

**Tool calling:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"Read the file README.md"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"max_tokens":256,"stream":false}'
```
Expected: `tool_calls` with `read_file({"path":"README.md"})`.

**Streaming:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":16,"stream":true}'
```
Expected: SSE chunks with `data: {...}` lines, final `data: [DONE]`.

**Reasoning:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? One sentence."}],"max_tokens":128,"stream":false}'
```
Expected: Correct "No" with valid syllogistic reasoning.

### Agent integration test

```bash
# Goose
GOOSE_PROVIDER=mesh GOOSE_MODEL=mesh goose run

# pi
# Set model to mesh in provider config
```

Verify: tool calls work end-to-end, multi-turn context retained by agent
client, no recursive hook loops.

### Performance characteristics

| Scenario | Expected behavior |
|----------|-------------------|
| 2+ workers agree quickly | Early-exit, faster than single-model |
| 1 worker much faster than others | Returns fast worker if confident |
| Remote peer timeout (60s worker / 60s reducer) | Grace ships what arrived at the 10s window; a dead peer costs 10s, not 60s |
| First reducer candidate slow / cold KV | Hedges to second candidate after 5s, races for first OK |
| First reducer candidate broken (502s) | Fast-fails to next candidate immediately, no hedge wait |
| Only 1 model available | Runs that worker inside the Mesh gateway |
| No workers admitted | Degrades to ordinary real-model selection; 503 if none exists |
| All workers fail | Returns error response |

### What to watch for

- **GLM chain-of-thought leaking** — GLM uses numbered markdown lists for
  reasoning, not `<think>` tags. When GLM is sole survivor, its internal
  deliberation can leak into the response.
- **Sole-survivor wait** — with exactly two workers and one slow/dead,
  the survivor waits up to the worker timeout (15s) before being released by
  the majority-failed early-exit. With 3+ workers this rarely bites.
- **Remote peer 503s** — mesh peers can be flaky. Gateway degrades to
  fewer workers but this limits model diversity.

---
## Diverse-mesh vs split-large: the operating-envelope question

> **Status: research plan, not yet run.** No comparison numbers exist.
> This section is the experimental design — including what would falsify
> the framing — so we cannot claim a win after the fact by moving the
> goalposts.

### What this is and is not about

This is **not** a benchmark fight. We are not trying to show that a
diverse mix of mid-size models scores higher on a quality rubric than a
single large model sharded across the same hardware. In a tight
interconnect with no network constraint, split-large should win on
quality and is the right tool — that is not in dispute.

This **is** about *operating envelopes*. Split-large has a hard
practical ceiling on a real mesh: every cross-node hop sits on the
critical path of every output token, and that path is bounded by the
worst link in the route. There is a point — defined by mesh size,
network latency, packet loss, and peer reliability — beyond which
split-large stops producing acceptable interactive performance at all.

MoA (`model: "mesh"`) has a much higher ceiling. Each worker runs
fully local on one node, so within a single worker's inference the
network is not on the critical path. The network is only touched at
fan-out, output collection, and the reducer call. That changes the
shape of the failure mode entirely: a slow link slows the *whole turn*
slightly, instead of stalling *every token*.

The load-bearing claim is therefore:

> **MoA stays inside the acceptable operating envelope for interactive
> agent use across a much wider range of mesh conditions than
> split-large does. Where both are viable, split-large generally wins
> on quality. Where only MoA is viable, MoA produces acceptable answers
> that split-large cannot produce at all.**

That is a *complementary* positioning, not a competitive one. The
network condition decides which tool is correct. We use split when we
can; we use mix when we can't.

### What "acceptable" means — pick this before measuring anything

Every later claim depends on a concrete definition of acceptable. For
interactive agent use (goose, Claude Code, pi):

| Dimension | Acceptable threshold (proposed) |
|---|---|
| Time to first token | ≤ 3 s on a fresh turn |
| Total turn wall-clock | ≤ 45 s for typical agent tasks |
| Turn failure rate | ≤ 2% over a representative session |
| Quality | ≥ baseline of a single mid-size local model on the same task |

These numbers are starting proposals, not gospel. They should be
ratified before any matrix run — otherwise we will retrofit
"acceptable" to whatever the data happens to show. A configuration
that exceeds **any** of these thresholds is **off the envelope** for
that operating point, regardless of how it scores on the others.

The quality floor is deliberately set at "≥ single mid-size local
model on the same task." MoA's job in the MoA-only region is not to
match a hypothetical 70B running in a datacenter — it is to be at
least as good as the best thing the user could otherwise run locally,
while staying inside the latency and reliability budget.

### The three configurations to measure

All on the same physical hardware, fixed aggregate VRAM budget:

| Config | What it is | Failure mode |
|---|---|---|
| **Split-large** | One large model (e.g. 70B Q4) sharded across nodes via Skippy | Network on every token's critical path; per-token latency grows with RTT and boundary count |
| **Mix-diverse** | MoA over 2–3 mid-size models, each fully local on one node | Per-worker latency stays local; turn latency = slowest worker + reducer |
| **Single-mid** | One mid-size model on one node (no mesh use of the second machine) | Sanity baseline; the floor any other config must beat to justify itself |

Single-mid is the **quality floor**. If MoA isn't above it, we have no
story — MoA is just adding latency without value. It is in the matrix
explicitly so we cannot accidentally ship "MoA = single-best + overhead."

### Primary axis: network conditions

This is the real experiment. Run all three configs across:

- **LAN baseline** — same switch, sub-ms RTT, no induced loss
- **+20 ms RTT** — typical metro-to-metro link
- **+50 ms RTT** — typical cross-region link
- **1% packet loss** at +20 ms — flaky residential link
- **5% packet loss** at +20 ms — broken-but-functional link
- **Peer churn** — one peer dropped mid-turn, returned mid-turn

The deliverable is **not** a Pareto curve of who's higher. It is a
**viability map**: for each (config × network condition) cell, is the
config inside the acceptable envelope at all?

```
                  LAN    +20ms  +50ms  1%loss  5%loss  churn
split-large       ok     ok     warn   no      no      no
mix-diverse       ok     ok     ok     ok      warn    ok
single-mid        ok     ok     ok     ok      ok      n/a
```

(Cells filled in illustratively, not measured.) The interesting
boundary is the **first column where split-large drops to `no` and
mix-diverse stays `ok`** — that is the operating range where MoA is
the only viable mesh-aggregated option, and the entire feature's
justification.

In cells where both are viable, the secondary question — quality
comparison — kicks in and matters. In cells where only one is viable,
the comparison is moot.

### Secondary axis: mesh size

Re-run the network sweep at 2, 3, and 4 nodes. Predictions if the
framing is right:

- Split-large's viable range **shrinks** as nodes are added — each
  additional boundary multiplies network exposure per token
- Mix-diverse's viable range stays roughly **flat** — adding nodes
  adds workers but doesn't change per-worker locality

If split-large's viability *doesn't* shrink with mesh size, our claim
that "split has practical limits on a real mesh" is wrong as stated
and needs to be retracted or qualified.

### Tertiary axis: quality, only inside the viable region

Quality measurement is **demoted** here. It exists to answer one
specific question:

> In the region where MoA is viable and split-large is not, are MoA's
> answers above the single-mid quality floor?

If yes, MoA's value-prop holds: it extends the operating envelope to
mesh conditions where split-large can't go, while producing answers at
least as good as what the user could run locally on one node.

If no — if MoA stays inside the latency envelope but its answers are
worse than a single mid-size local model would have produced — then
the user is better off ignoring the mesh and just running locally, and
MoA's claim collapses.

The quality measurement itself uses the same machinery the earlier
draft described: adversarial scenarios spanning decomposable
correctness, single coherent thread, and failure-mode amplification;
agent-as-grader with position swap and dual-grader checks; real-task
replay as the strongest defense against scenario cherry-picking.
Those mechanics are not the headline — they are how we estimate the
quality dimension of the envelope.

### What would falsify this framing

Pre-committed, so we don't retreat to softer claims after seeing data:

- **Split-large's viable region does not shrink as the mesh degrades
  or grows.** If split-large stays inside the envelope at +50 ms,
  modest loss, or 4-node configs, then "MoA extends the operating
  envelope past split's limits" is wrong — there is no envelope to
  extend.
- **MoA's viable region does not extend past split-large's.** If both
  fail in the same cells, MoA is not buying envelope expansion, it's
  just a different way to fail.
- **MoA quality in the MoA-only region falls below the single-mid
  floor.** If MoA only "works" by producing answers worse than the
  user's local fallback, the feature is net-negative.
- **Same-model-N-workers performs equivalently to different-model-N-
  workers** *and* MoA still extends the envelope past split. Then the
  envelope extension is real but the value source is variance
  reduction, not mixture — the feature should be re-described as such
  and "diverse" is misleading.

If any of those hold, the framing in this section is wrong and the
doc must be updated to reflect what the data actually showed, not
softened to fit.

### Reporting discipline

Every result must include:

- Acceptable-envelope thresholds in force (TTFT, total turn, failure
  rate, quality floor)
- Exact mesh composition (model list per node, total VRAM per node)
- Network conditions in force (RTT, loss, churn schedule)
- Per-config viability flag (`ok` / `warn` / `no`) against each
  threshold
- Quality numbers reported only for cells where the config is viable;
  reporting quality for a non-viable cell is misleading

A claim like "MoA wins" is not reportable. Reportable claims look
like: *"At +20 ms RTT with 1% loss across 3 nodes, split-large
exceeds the TTFT threshold (measured 7.2 s, threshold 3 s) and is
non-viable; mix-diverse stays inside all thresholds and scores 3.9 on
the quality rubric vs single-mid's 3.6."* That sentence is concrete,
falsifiable, and locates the result on the viability map.

### Worker-set knob (deferred, harness-side only)

To run the same-model-N-workers ablation cleanly, the eval needs a
way to restrict MoA's worker pool. The current `model: "mesh"` uses
every callable model.

When needed, this belongs in the eval harness, not the crate: either
pre-flight filter `/v1/models` to an allowed set before launching, or
have the harness send a header (e.g. `x-moa-workers-include: A,B`)
that the gateway honors. No semantic change to the crate, just a
knob.

### Status / next step

No matrix has been run. The concrete first step is **pinning the
acceptable-envelope thresholds** above with the team — TTFT, total
turn time, failure rate, quality floor — *before* any measurement
starts. Those numbers are the load-bearing definition of the entire
experiment. Everything else flows from them.

The second step, only after thresholds are pinned, is a single-cell
measurement at LAN baseline with the three configs, to confirm the
machinery works and that split-large does in fact dominate quality
when the network isn't constraining it. If split-large doesn't win
at LAN baseline, our entire framing of "split is best when network
allows" is wrong and we need to rethink before running the network
sweep.
