# Mixture-of-Agents in mesh-llm

## Background

Two key papers:

1. **MoA** (arxiv 2406.04692, Together AI) — Fan-out + synthesis. All models answer in parallel, strongest model aggregates into one refined response. Layered: repeat with each model seeing previous responses. Simple, fast. Beat GPT-4o on AlpacaEval 2.0 (65.1% vs 57.5%) using only open models.

2. **NSED/MoM** (arxiv 2601.16863, Peeramid Labs) — Deliberation protocol. Parallel proposals → anonymous cross-evaluation (diagonal mask) → quadratic voting → recurrent refinement. 3-7 rounds. Consumer ensemble (<20B models) matched DeepSeek-R1 on AIME 2025 (84% vs 84.2%).

## Our Experiments (on `mom-experiment` branch)

### NSED results (mom.py, ran on Studio)
- Tool use: ✅ ensemble correctly picked MiniMax (only model calling both tools)
- Reasoning: ✅ multi-round deliberation improved structure
- Math/Code: ❌ no improvement when all models solve correctly
- Latency: 3-10x overhead (27 API calls for 3 rounds × 3 models)
- Qwen3-8B: 0 wins — too weak to lead, only useful as evaluator

### Agent eval (eval-agent.sh, pi harness)
- Claude Sonnet 4: 12s, 5/5 — read file, edited correctly, compiles
- Qwen3-8B local: 84s, 3/5 — hallucinated, no edit made
- Gap is massive for agentic work (tool use, file editing)

## What We're Building

**MoA as a client-side model in mesh-llm.** Request `model=moa` and the client node:

1. Fans out to all mesh models in parallel (streaming)
2. Collects their full responses
3. Injects all responses as references (per MoA pattern)
4. Sends to aggregator (strongest model by tier) for synthesis
5. Streams aggregator response back to caller

This is the MoA approach — simple synthesis, no voting. It works at the proxy level in the client, hosts are unaware.

### Why MoA over NSED

- MoA is ~2x latency (1 fan-out + 1 aggregation). NSED is 3-10x (multiple rounds).
- For mesh-llm, users are waiting for a chat response. 2x is tolerable, 10x is not.
- NSED's voting helps for tool routing, but that's a separate router feature, not a model.
- MoA's synthesis is what matters for quality — the aggregator sees all perspectives.

### Algorithms Implemented

All run client-side in `mesh-llm/src/moa.rs`. Select via model name:

| Model name | Strategy | How it works | Expected latency |
|---|---|---|---|
| `moa` | Synthesize | Fan-out → aggregator synthesizes all responses | ~2x solo |
| `best-of-n` | Best-of-N | Fan-out → aggregator picks best response verbatim | ~2x solo |
| `moa-2` | Two-layer | Fan-out → all models refine → aggregator synthesizes | ~3x solo |

**Synthesize** (`moa`): MoA paper approach. Simple and effective. Aggregator sees all perspectives and produces a refined answer.

**Best-of-N** (`best-of-n`): NSED-inspired selection. Aggregator picks the single best response, doesn't rewrite. Cheaper than synthesis. Preserves the original model's voice/format. Good for tool routing — if one model nails the tool call, just return it.

**Two-layer** (`moa-2`): Advanced MoA. Each model refines having seen all other models' Layer 1 responses, then aggregator synthesizes Layer 2. The paper showed this improves quality on benchmarks, at the cost of another full fan-out round.

### Not yet implemented

- **MoA + tool calls**: The current approach breaks tool-use because the aggregation prompt confuses the agent's tool protocol. Fix: run MoA for the *reasoning/planning* step, then do a separate model pass to extract tool calls from the synthesized answer. Alternatively, MoA produces a plan/answer, and a dedicated "tool extractor" pass converts it into the correct tool-call format for the harness. This is the key unsolved problem.
- **Full NSED** (3-7 round deliberation with voting): Too slow for interactive use. Our experiments showed 3-10x latency for marginal improvement on most tasks. The one exception is tool routing, which Best-of-N handles.
- **Speculative routing**: Already exists as `pipeline.rs` — small model pre-plans, big model executes. Orthogonal to MoA.
- **Confidence-based early exit**: Fan out but stop collecting when first 2 models agree. Would reduce latency but adds complexity.

### Implementation

- `mesh-llm/src/moa.rs` — all strategies, ~300 lines
- `mesh-llm/src/proxy.rs` — intercepts MoA model names in `handle_mesh_request`
- `mesh-llm/src/main.rs` — intercepts MoA model names in `api_proxy`
- Uses `reqwest` to make HTTP calls back through local proxy (routes via QUIC to hosts)
- Streaming for fan-out collection and aggregation response
- Picks aggregator by highest tier from router profiles
- Virtual models (`moa`, `best-of-n`, `moa-2`) appear in `/v1/models` when 2+ real models available

### How to test

1. Start local mesh client: `mesh-llm --client --auto --port 9337`
2. Verify models: `curl localhost:9337/v1/models` — should show `moa`, `best-of-n`, `moa-2`
3. Test MoA: `curl localhost:9337/v1/chat/completions -d '{"model":"moa","messages":[{"role":"user","content":"..."}]}'`
4. Test Best-of-N: same but `"model":"best-of-n"`
5. Test with pi: point `models.json` at `localhost:9337`, use any strategy as model id
6. Run eval: `mom/eval-agent.sh` — compares Claude vs MiniMax solo vs MoA

### Eval plan

Compare on the same agentic task (add display_name() to Rust struct):

| Run | What | Why |
|---|---|---|
| `claude` | Claude Sonnet 4 (Anthropic API) | Ceiling — best agentic model |
| `minimax-solo` | MiniMax-253B via mesh | Solo baseline — strongest mesh model |
| `moa` | MoA (synthesize) via mesh | Does synthesis improve agentic quality? |
| `best-of-n` | Best-of-N via mesh | Does selection beat solo? |

We already know Claude scores 5/5 and Qwen3-8B scores 3/5.
The question: does MiniMax-253B match Claude? Does MoA/Best-of-N improve?

## Files

```
mom/
  PLAN.md              — this file
  RESULTS.md           — experiment results and conclusions
  REFERENCE.md         — MoA/MoM/NSED/RouteMoA literature notes
  eval-agent.sh        — pi-based agentic eval (Claude vs mesh models vs MoA)
  local-moa-test.py    — fast local algorithm comparison (8 strategies)
  mom.py               — NSED protocol experiment (ran on Studio)
  historical/          — superseded Python prototypes
  results/             — eval outputs
mesh-llm/src/
  moa.rs               — MoA strategies (synthesize, best-of-n, moa-2), client-side
  proxy.rs             — MoA interception in handle_mesh_request
  main.rs              — MoA interception in api_proxy
```
