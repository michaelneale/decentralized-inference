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

## Claude vs Qwen3-8B Comparison

Direct head-to-head on a code review task (find bugs in a statistics module, fix them, add error handling, suggest tests).

| | Claude Sonnet 4 | Qwen3-8B (local) |
|--|---|----|
| Time | 34.0s | 28.7s |
| Output | 8,987 chars | 7,460 chars |
| Bugs found | 10 | 7 |
| Quality | Excellent | Good |

### Quality Analysis

**Claude advantages:**
- Found more edge cases (NaN, infinity, boolean-as-number, string input, iterator input)
- Produced complete working fixed code with a shared `_validate_input()` helper
- Test cases are specific pytest assertions with `pytest.raises`
- Identified the subtle issue of booleans being treated as int (True=1, False=0)

**Qwen3-8B:**
- Found the main bugs (empty list, mode ties, percentile range, non-numeric)
- Chose `return None` pattern instead of raising exceptions — debatable design choice but less Pythonic
- Test cases are more repetitive (same `describe([])` test repeated)
- Missed: NaN handling, infinity, boolean-as-number, string-as-iterable, iterator support

**Verdict:** Claude is meaningfully better — finds more edge cases, produces more idiomatic Python, better test coverage. But Qwen3-8B catches the most important bugs. For a code review, Claude at ~$0.05 beats Qwen3-8B (free), but the gap isn't enormous.

**MoM implication:** An ensemble of Qwen3-8B + Qwen3-30B + MiniMax might close the gap with Claude by combining diverse perspectives. The MoM tool-use result already showed the ensemble picking the best model's answer.

## Next Steps

- [ ] Try single-round MoM (parallel generate → vote → best answer) — same quality gain, 1/3 the latency
- [ ] Run same code review through MoM ensemble, compare with Claude
- [ ] Test with harder problems where models genuinely disagree
- [ ] Consider MoM as a mesh-llm router strategy for `model=auto`
