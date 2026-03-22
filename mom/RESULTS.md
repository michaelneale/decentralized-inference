# MoM Experiment Results

## Setup

3 models on Studio (M4 Max, 206GB):
- **Alpha**: MiniMax-M2.5-Q4_K_M (253B MoE) — strongest model
- **Beta**: Qwen3-8B-Q4_K_M — small general model  
- **Gamma**: Qwen3-30B-A3B-Q4_K_M (30B MoE, 3B active) — small MoE model

Protocol: NSED (N-Way Self-Evaluating Deliberation), up to 3 rounds.
- Parallel proposal generation from all 3 agents
- Anonymous cross-evaluation (diagonal mask — can't score own proposal)
- Quadratic voting aggregation (sqrt of scores dampens extremes)
- Winner's answer becomes context for next round

Thinking mode disabled (`enable_thinking: false`) — direct answers only.

## Results

| Test | Solo Alpha | Solo Beta | Solo Gamma | Ensemble | Rounds | Winner |
|------|-----------|----------|-----------|----------|--------|--------|
| Math | 128.3s | 12.1s | 13.9s | 191.5s | 3 | Gamma→Alpha→Gamma |
| Code | 23.1s | 6.5s | 6.3s | 123.3s | 3 | Gamma→Gamma→Gamma |
| Tool Use | 1.4s | 0.4s | 0.3s | 9.9s | 2 ✅ | Alpha→Alpha (converged) |
| Reasoning | 33.9s | 15.4s | 9.7s | 155.7s | 3 | Alpha→Gamma→Alpha |

**Agent win totals**: Alpha: 5 wins, Beta: 0 wins, Gamma: 6 wins

## Key Findings

### 1. Tool Use: Clear winner — MiniMax dominates
Alpha (MiniMax) was the ONLY model that correctly called both tools in a single response. Beta and Gamma only called `get_weather` and ignored the calculation. The ensemble correctly identified Alpha's multi-tool answer as best and converged in 2 rounds. **This is the strongest MoM result** — the ensemble picked the right specialist.

### 2. Math: All models get the same approach
All three models use the same completing-the-square method. The ensemble doesn't improve the answer — it mostly polishes presentation. MiniMax was slowest (128s) because of its size, even without thinking mode.

### 3. Code: All models produce correct O(n log n) LIS
Gamma wins every round, but the code is functionally identical across all three. The ensemble adds no value here — any solo model produces a correct `bisect_left`-based solution.

### 4. Reasoning: Genuine improvement through deliberation
The TCP vs QUIC comparison shows real improvement. Round 1 picks Alpha's detailed analysis, Round 2 Gamma refines it, Round 3 Alpha produces a more precise version that addresses peer review concerns. The final answer is notably better structured with explicit assumptions (TCP + TLS 1.3, IETF QUIC) and practical focus.

### 5. Beta (Qwen3-8B) never wins
Zero wins across all tests. It's outclassed by both MiniMax and the 30B MoE. In a MoM ensemble, weak models become pure evaluators — they add voting power but never lead.

### 6. Latency cost is high
Ensemble is 3-10x slower than the best solo model per task. Each round requires N proposals + N*(N-1) evaluations = 3 + 6 = 9 API calls. With 3 rounds: 27 API calls per query.

## Conclusions

**Does MoM work?** Partially.

✅ **Tool use**: The ensemble correctly identifies the model that fully satisfies the request. This is the most promising direction — routing to the right model via peer evaluation.

✅ **Reasoning**: Multi-round deliberation improves structure and rigor for open-ended analysis.

❌ **Math/Code**: When all models can solve the problem, the ensemble adds latency without improving correctness. The "convergent" problems don't benefit.

❌ **Latency**: 3-10x overhead is too much for interactive use. Could work for batch/background tasks.

## Implications for mesh-llm

1. **Smart routing via MoM evaluation is promising**. Instead of full deliberation loops, a single evaluation round could help pick which model's answer is best. This is essentially "try all models, vote on the best" — 1 round instead of 3.

2. **MoM for tool calls specifically**: When multiple models disagree on tool selection, cross-evaluation picks the most complete one. Could be a mesh router feature: if `auto` model and tools present, try N models in parallel, vote on winner.

3. **Don't do full NSED for simple queries**. The paper's AIME results required 6-7 rounds of 20K-token thinking models. Our practical tasks don't benefit from that depth.

4. **Model diversity matters more than model count**. MiniMax (MoE, strong tools) + Qwen3-30B (MoE, fast) complement each other. Adding a third weak model (Qwen3-8B) didn't help — it never won.

## Claude vs Qwen3-8B: Agentic Comparison

### Real agentic task (pi harness, multi-turn with tools)

Task: Add a `display_name()` method to a Rust User struct. Agent must read files, edit the right one, produce compiling code.

Ran via `pi -p` (print mode) with `PI_CODING_AGENT_DIR` isolation — same approach as Open Model Gym.

| | Claude Sonnet 4 | Qwen3-8B (local) |
|--|---|----|
| Time | **12s** | 84s |
| Score | **5/5** | 3/5 |
| Edited file? | ✅ Yes, correct | ❌ No |
| Compiles? | ✅ | ✅ (unchanged) |
| Hallucinated? | No | **Yes** — claimed method already existed |

**Claude** read `user.rs`, added the method in the right place with `format!("{} {}", self.first_name, self.last_name)`, preserved all existing methods. 12 seconds, clean.

**Qwen3-8B** hallucinated that the method was already present and produced no edit. The file was untouched. 84 seconds wasted on nothing.

### Single-turn comparison (raw API, no tools)

Same code review task (find bugs in a Python statistics module):

| | Claude Sonnet 4 | Qwen3-8B (local) |
|--|---|----|
| Time | 34s | 29s |
| Bugs found | 10 | 7 |
| Quality | Excellent | Good |

Claude found more subtle edge cases (NaN, infinity, booleans-as-ints, string-as-iterable). Qwen3-8B caught the main bugs but missed nuance.

### Verdict

**The gap is massive for agentic work.** Single-turn, the models are in the same ballpark — Qwen3-8B finds 70% of what Claude finds. But when you need the model to actually use tools (read files, edit code, verify), Qwen3-8B falls apart. It hallucinates rather than doing the work.

This is the real challenge for mesh-llm as an agent backend. The models can reason but they can't reliably execute multi-step tool-use workflows. MoM doesn't help here — you can't vote on "which model correctly edited a file" if none of them did.

## MiniMax-253B via Mesh: Matches Claude

Same task (add display_name() to Rust struct), pi -p harness, working directory fix applied.

| | Claude Sonnet 4 | MiniMax-253B (mesh) | Qwen3-8B (local) |
|--|---|---|---|
| Time | **15s** | 24s | 84s |
| Score | **5/5** | **5/5** | 3/5 |
| Edited file? | ✅ | ✅ | ❌ |
| Correct impl? | ✅ `format!("{} {}", ...)` | ✅ identical | N/A |
| Hallucinated? | No | No | Yes |

**MiniMax-253B matches Claude on this agentic task.** It read the file, found the right location, added the method with the correct signature and implementation, and produced compiling code. The edit is character-for-character identical to Claude's.

**MoA failed** — 0 bytes output. The fan-out to 3 models waits for the slowest (Qwen3.5-9B, ~80s for thinking), which likely exceeded pi's internal timeout. MoA as an agent backend needs either: (a) a time-bounded fan-out that proceeds with whoever responded, or (b) fewer/faster reference models.

### Implications

1. **Model scale is the answer for agentic work.** MiniMax-253B (free, local) matches Claude Sonnet 4 (paid API). Qwen3-8B (also free, also local) completely fails. The gap is model capability, not the harness.

2. **MoA latency kills agentic use.** An agent harness (pi, Claude Code, Goose) makes many small requests. Each MoA request waits for 3 models + aggregation. If each takes 30-90s, a 10-turn agent session becomes 5-15 minutes. Solo MiniMax at 24s/turn is already fast enough.

3. **MoA's value is for single-turn quality** — where you want the best possible answer to one question. For agentic flows (many turns, tool use), solo strong model wins on speed.

## Local Algorithm Comparison (3 models, no mesh)

3 local models: Qwen3-30B-A3B (17GB MoE), Qwen2.5-Coder-7B (4.4GB), Mistral-Small-24B (13GB).

### Code review task (find bugs in merge/search functions)

| Strategy | Time | Finding |
|---|---|---|
| Solo Qwen3-30B | 19.5s | Found real bug + some false positives |
| Solo Coder-7B | **6.9s** | Found real bug immediately, but also hallucinated hi=len(arr) is wrong |
| Solo Mistral | 22.2s | Found real bug correctly |
| MoA (synthesize) | 66.5s | Combined all findings, including hallucinated ones |
| Best-of-N (pick) | 66.6s | Picked best solo response |
| Rank-and-pick | 70.0s | Ranked and picked |

**Key finding**: MoA synthesis amplifies hallucinations — if one model says something wrong, the aggregator includes it. Best-of-N is better for correctness because it selects rather than merges.

### Monty Hall / Sieve of Eratosthenes

All models get these right. MoA adds 3.5x latency for zero quality improvement. Confirms: when all models can solve the problem, ensembling is pure waste.

## MoA with Strong Models Only (MiniMax + Qwen2.5-72B)

After filtering to tier 3+ models only (dropping Qwen3.5-9B), MoA fan-out drops from 80s to 3s. Second agentic eval:

| | Claude | MiniMax solo | MoA (2 strong) |
|--|---|---|---|
| Time | **13s** | 18s | 45s |
| Score | **5/5** | **5/5** | 3/5 |
| Issue | — | — | Aggregation prompt broke tool-use |

MoA's aggregation prompt (`"synthesize these responses..."`) confused MiniMax into using its own XML tool-call format (`<minimax:tool_call>`) instead of pi's tools. The synthesis context injection is fundamentally incompatible with agentic tool-use flows.

**MiniMax solo = Claude** confirmed again. Consistently 5/5 across two runs.

## Conclusions

### What works
- **Solo strong model (MiniMax-253B)** for agentic tasks — matches Claude at 18-24s
- **MoA for single-turn chat quality** — synthesizes diverse perspectives
- **Best-of-N for correctness** — picks best response without amplifying hallucinations
- **Tier filtering** — only fan out to strong models (tier 3+)

### What doesn't work
- **MoA for agentic/tool-use** — synthesis prompt breaks tool-use protocol
- **Including weak models** — Qwen3.5-9B adds 80s latency, never contributes quality
- **MoA synthesis for code review** — amplifies hallucinated bugs from weaker models
- **Multi-round deliberation (NSED)** — 3-10x latency, marginal improvement

### Recommendations for mesh-llm
1. **Default `model=auto` should pick the strongest single model** — already implemented, working
2. **`model=moa` is useful for chat** — non-agentic single-turn questions where quality matters
3. **`model=best-of-n` for code review/analysis** — when you want the best answer, not a merged one
4. **Filter MoA to tier 3+ models** — implemented, prevents weak model drag
5. **Don't use MoA as an agent backend** — it breaks tool-use flows

## Next Steps

- [ ] Test MoA on harder tasks where models genuinely disagree (edge cases, ambiguous requirements)
- [ ] Try MoA-2 (two-layer) to see if quality justifies 3x latency for chat
- [ ] Consider `model=auto` using MoA only for single-turn non-tool requests
- [ ] Profile MoA latency breakdown: fan-out vs aggregation vs thinking overhead
