# llama.cpp Family Parity

This is the cheap certification lane for reaching practical parity with the
pinned llama.cpp model-family surface.

The goal is one small GGUF representative per llama.cpp `src/models/*.cpp`
implementation. Passing this lane does not mean every model size and quant for
that family is certified. It means the family has at least one reviewed artifact
that proves the stage ABI, activation handoff, topology policy, and cache policy
work together.

For new large-model requests that do not already have a cheap representative,
start with [NEW_MODEL_ONBOARDING.md](NEW_MODEL_ONBOARDING.md). Keep those models
as onboarding candidates until package, runtime, and family evidence is strong
enough to promote.

## Workflow

1. Join the pinned llama.cpp inventory with the candidate manifest:

   ```bash
   scripts/skippy-llama-parity.py inventory
   ```

   The manifest must classify every pinned `src/models/*.cpp` implementation:

   ```bash
   scripts/skippy-llama-parity.py validate
   ```

   When `.deps/llama.cpp` is prepared, validation also rejects architecture-
   named runtime-slice admission gates and requires the structural and realized
   graph-boundary checks that make staged loading fail closed.

2. See which cheap representatives are not downloaded yet:

   ```bash
   scripts/skippy-llama-parity.py download-commands
   ```

   To use an external disk, set `HF_HOME` or `HF_HUB_CACHE` before downloading
   and before running this script.

3. Run local candidates through the existing certification harness:

   ```bash
   scripts/skippy-llama-parity.py run \
     --limit 1 \
     --prefix-token-count 8 \
     --cache-hit-repeats 2 \
     --borrow-resident-hits
   ```

   Supplying `--prefix-token-count` makes the wrapper select the production
   cache lane for the family: `ResidentKv` for dense families and
   `KvRecurrent` for recurrent/hybrid families. Omitting those cache flags runs
   only the split/debug lane and does not reproduce the production cache
   evidence below.

4. Keep the Rust regression lane green. `crates/skippy-correctness/tests/parity_models.rs`
   has one module per P0/P1 family and a cheap coverage test that fails when a
   P0/P1 manifest row does not have a module:

   ```bash
   LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-abi-cpu" \
     cargo test -p skippy-correctness --test parity_models
   ```

   The expensive checks are ignored by default. Run one family while debugging:

   ```bash
   SKIPPY_PARITY_DOWNLOAD=1 \
   LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-abi-cpu" \
     cargo test -p skippy-correctness --test parity_models \
     p0_qwen2_qwen2 -- --ignored
   ```

   Before patch-queue rebuilds or broad family promotions, run the full ignored
   lane. It downloads missing representatives, compares a three-stage local
   split against full-model decode, and verifies cache restore from one
   session/lane into another session/lane with native sequence remapping:

   ```bash
   SKIPPY_PARITY_DOWNLOAD=1 \
   LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-abi-cpu" \
     cargo test -p skippy-correctness --test parity_models -- --ignored
   ```

   Heavy parity tests default `SKIPPY_PARITY_N_GPU_LAYERS` to `999`, matching
   the family-certification lane; set it to `0` for an explicit CPU-only repro.
   Use `SKIPPY_PARITY_DOWNLOAD=0` when the Hugging Face cache must be treated as
   fixed input. Package-only rows that point at monolithic GGUFs are not
   downloaded by default, because some representatives are larger than 100GB.
   Set `SKIPPY_PARITY_DOWNLOAD_PACKAGE_ONLY=1` to fetch those artifacts, and
   set `SKIPPY_PARITY_REQUIRE_PACKAGE_ONLY=1` when a missing local package-only
   artifact should fail the run instead of being reported as a skipped
   package-only proof.

5. Promote passing rows into:

   - `docs/skippy/FAMILY_STATUS.md`
   - `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`

Raw run artifacts stay under `target/family-certify/...`.

### Representative cut lattice

The default `activation_handoff_matches_full_model` lane exercises exactly one
split pair per family. A Granite-class failure can hide in any untested cut,
because the native builders hand-copy the stage-filter snippet per model file.
`p0_llama_representative_cut_lattice_matches_full_model` covers a lattice of
cuts for the P0 `llama` representative instead: the full ordered-pair lattice
when `layer_count <= 12`, and otherwise the deduplicated union of all adjacent
pairs `(s, s+1)`, first-edge pairs `(1, s2)`, final-edge pairs `(s1, L-1)`,
plus the reviewed manifest pair. Coverage mode and pair count are printed on
every run, and a failing cut reports model, `(s1, s2)`, layer count, and the
stage ranges so a null-weight crash names its cut:

```bash
LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-abi-cpu" \
  cargo test -p skippy-correctness --test parity_models \
  p0_llama_representative_cut_lattice_matches_full_model -- --ignored
```

## Tracking Plan

We track each llama.cpp family through separate gates. A family is not promoted
just because one early gate passed.

| Gate | What It Proves | Required For Promotion |
| --- | --- | --- |
| Candidate selected | We have one cheap representative GGUF for the llama.cpp model implementation. | Yes |
| Downloaded/local | The representative is available in the local Hugging Face cache. | Yes |
| Inspect | GGUF metadata yields architecture, layer count, activation width, and split plan. | Yes |
| Text split | `single-step` and `chain` pass with the fixed raw-f32 wire. | Yes for text serving |
| Exact state | `state-handoff` accepts, or the family has a documented reason to reject exact state mobility. | Yes |
| Cache policy | The family is assigned `ResidentKv`, `KvRecurrent`, package-only `ResidentKv`, or cache disabled. | Yes |
| Cache smoke | Serving-path exact-prefix cache hits and misses behave correctly for that family/policy. | Yes for cache-on-by-default |
| Multimodal | Projector/media sidebands pass, where applicable. | Yes for multimodal serving |
| Promotion | `FAMILY_STATUS.md` and reviewed topology records are updated. | Final |

## Cache Scope

Yes, cache support is part of this parity program, but it is a separate gate
from stage-split correctness.

The rule is:

| Family Shape | Default Cache Target | Promotion Rule |
| --- | --- | --- |
| Dense causal decoder | `ResidentKv` | Promote after text split, exact-state decision, and serving cache smoke pass. |
| Recurrent/hybrid causal decoder | `KvRecurrent` | Promote only after recurrent ranges or sticky ownership are documented and cache smoke proves KV plus recurrent state is exact. |
| Package-backed huge model | package-local `ResidentKv` | Promote against materialized stage artifacts; do not require monolithic full-GGUF llama-server baseline. |
| Multimodal decoder | family-specific, usually `ResidentKv` plus sidebands | Text cache smoke is not enough; projector/media-token path must pass. |
| Encoder/embedding/non-causal | disabled or separate future lane | Do not promote as causal stage-split cache support. |
| Unknown/uncertain state layout | disabled | No implicit cache reuse. |

`FullState` remains a certification/debugging payload. It is not the production
cache target for parity. Production evidence should use `ResidentKv` or
`KvRecurrent`.

## Cache Correctness Exit Gate

Before this branch is considered done, prove exact-prefix cache correctness by
family/state-layout class. The important invariant is native sequence remap:
state recorded from one runtime session/lane must restore exactly into a
different runtime session/lane with a different backend sequence id.

Required scenarios:

1. Dense `ResidentKv` families:
   - Use representative local GGUFs for Llama, Qwen dense, Gemma, GLM, OLMo,
     or the closest locally available dense reviewed families.
   - Use `ResidentKv` only.
   - Record a prefix from session A, restore or borrow it into session B, and
     verify the next token/logits match normal no-cache prefill.
   - Verify suffix prefill after restore also matches normal prefill.
   - Run one-stage and at least one split-stage topology.
2. Hybrid/recurrent `KvRecurrent` families:
   - Use Falcon-H1 and Qwen3Next/Qwen3.6 recurrent when available.
   - Include MiniMax, RWKV, Mamba-like, Jamba, or LFM2 representatives when
     available.
   - Use `KvRecurrent` only.
   - Verify attention KV plus recurrent/SSM state restores exactly into a
     different target session/lane.
   - Verify immediate decode and suffix-prefill-then-decode.
   - Report payload bytes, imported tokens, hit/miss counters, and whether
     recurrent bytes are non-zero.
3. Negative policy checks:
   - Confirm recurrent/stateful GGUFs do not select `ResidentKv`.
   - Check whether current recurrent guards detect tensor names containing
     `.ssm`, `ssm_`, `time_mix`, `recurrent`, or `rwkv`.
   - If a stateful family uses different tensor naming, flag it and propose the
     detection rule before enabling cache reuse.
4. Split-stage correctness:
   - Verify cache restore in staged serving, not only single-stage.
   - Include stage 0, middle-stage, and final-stage restore where the harness
     can observe the relevant cache/activation boundary.
   - Vary layer ranges enough to catch stage-local KV or recurrent state shape
     issues.

Pass criteria:

- Cached restore produces the same next token/logits as normal prefill.
- Restore target is a different runtime session/lane from the recording source.
- Repeated cache hits remain stable.
- Unknown or failing families remain disabled or recompute-only.

The final evidence table should include one row per family/model ref with:
payload mode, one-stage result, split-stage result, source native seq id,
target native seq id, imported tokens, payload bytes, recurrent bytes, hit/miss
counters, repeated-hit stability, suffix-prefill result, and promotion decision.

## Certification Levels

| Level | Meaning |
| --- | --- |
| `candidate` | Cheap GGUF exists or is proposed; run full family certification before promotion. |
| `candidate_stateful` | Recurrent/hybrid family; default to sticky recurrent ownership. |
| `candidate_multimodal` | Text lane can be checked cheaply; projector/image/audio lanes must pass before multimodal promotion. |
| `certified` | Already in the reviewed support matrix. Re-running is allowed when llama.cpp or the ABI changes. |
| `certified_package_only` | Full source model is too large; package/stage evidence is the support contract. |
| `implementation_base` | Shared llama.cpp implementation helper, not a standalone GGUF architecture row. Certify concrete derived families instead. |
| `needs_runtime_slice_support` | Causal llama.cpp family exists, but skippy's stage ABI does not yet support its graph/tensor filtering. Add runtime-slice support before certification. |
| `non_causal_aux` | Encoder, embedding, decoder-only audio, or other auxiliary path; do not certify as causal stage-split serving. |
| `package_or_remote_only` | No cheap local representative exists; certification needs package/stage evidence on larger hardware. |
| `needs_candidate` | No cheap representative has been selected yet; rows with a concrete `repo`/`include` should use `candidate`, `candidate_stateful`, `candidate_multimodal`, or `package_or_remote_only` so the download workflow includes them. |

## Policy

Activation transport is fixed to raw `f32`. The single-step and chain lanes
prove staged execution parity without a lower-precision wire policy. Older
tables below retain f16/q8 measurements as historical evidence; those formats
are no longer selectable protocol modes.

For recurrent and hybrid families, the cheap lane should use `--recurrent-all`
until exact recurrent layer ranges are known. Activations may cross topology
boundaries; recurrent state defines sticky topology affinity.

For multimodal families, a text-only pass is not enough. The row can move to
text stage-split support after the text lane passes, but multimodal support
requires projector/token-sideband evidence.

The candidate manifest lives at
`docs/skippy/llama-parity-candidates.json`.

## Popularity Priority

The manifest keeps every pinned llama.cpp architecture classified, but the
active support queue is filtered by popularity and real Mesh demand:

| Priority | Meaning | Current Rows |
| --- | --- | ---: |
| `p0` | Must support: very popular GGUF ecosystems, current Mesh deployment targets, staged package families, and strategic multimodal/recurrent families. | 41 |
| `p1` | Should support: common vendor, coding, recurrent, enterprise, and historically important GGUF families. | 39 |
| `p2` | Best effort: long-tail or niche families that stay classified but are not release blockers unless demand appears. | 51 |

The P0/P1 set is the parity gate. It is based on Hugging Face GGUF
text-generation search sorted by downloads, current Mesh deployment demand, and
the pinned llama.cpp architecture inventory. At the time of review, top GGUF
download signals were dominated by Llama, Qwen, Gemma, MiniMax, GPT-OSS,
Qwen3.5/Qwen3.6 MoE, Mistral, GLM, Phi, and related coder/VL variants.

Use priority filters when working the active queue:

```bash
python3 scripts/skippy-llama-parity.py inventory --priority p0
python3 scripts/skippy-llama-parity.py inventory --priority p1
python3 scripts/skippy-llama-parity.py download-commands --priority p0
scripts/download-skippy-parity-candidates.sh --dry-run
```

## Current Coverage Summary

`scripts/skippy-llama-parity.py validate` now requires every pinned
llama.cpp `src/models/*.cpp` implementation to have a manifest row, and it
checks architecture-independent runtime-slice admission when the prepared
llama.cpp checkout is available. Current classification:

| Status | Count | Meaning |
| --- | ---: | --- |
| `certified` | 89 | Cheap representative passed the promoted split-serving evidence for this branch. |
| `certified_package_only` | 7 | Huge model has package/stage evidence rather than a monolithic local baseline. |
| `candidate_multimodal` | 3 | Needs projector/media sideband certification before multimodal promotion. |
| `non_causal_aux` | 14 | Encoder, embedding, audio, or other non-causal serving lane. |
| `implementation_base` | 4 | Shared implementation helper, not a standalone GGUF architecture. |
| `package_or_remote_only` | 9 | No cheap local representative; certify on larger hardware or package artifacts. |
| `no_public_gguf_candidate` | 5 | No public GGUF matching that llama.cpp architecture was found. |

The active P0/P1 queue has no `needs_candidate`, `candidate_multimodal`, or
`needs_runtime_slice_support` rows. The remaining `candidate_multimodal` rows
are long-tail non-P0/P1 families; remaining package/remote rows need larger
hardware or package artifact evidence before promotion.

## Family Board

This board is the running checklist. Update it whenever a family moves forward
or a blocker is discovered.

| Family | llama.cpp Model | Candidate | Downloaded | Text Split | Exact State | Cache Policy | Cache Smoke | Promotion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2` | `qwen2` | selected | yes | pass | pass | `ResidentKv` | pass | ready for reviewed promotion |
| `deepseek` | `deepseek` | selected | yes | pass | pass | `ResidentKv` | pass | promoted |
| `mistral` | `mistral3` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `mistral4` | `mistral4` | package selected | yes | package validated | package validated | package-local `ResidentKv` target | package validated | `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF:IQ2_XXS` package-only certified; 36-layer package materialized and validated with all 579 tensors accounted for |
| `lfm2` | `lfm2` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent cache restore ready; keep normal decode ownership sticky |
| `gpt2` | `gpt2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `gemma` | `gemma` | selected | yes | pass with raw f32 wire | pass | `ResidentKv` | pass | cache restore ready |
| `gemma3n` | `gemma3n` | selected | yes | pass | pass | `ResidentKv` text path | pass | certified with AltUp sideband activation frames; reviewed topology keeps KV-reuse layers with their KV-owner layers (`0..10,10..15,15..30`) |
| `mpt` | `mpt` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `olmo2` | `olmo2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `olmoe` | `olmoe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `qwen3vl` | `qwen3vl` | selected | yes | pass | FullState blocked by M-RoPE; `ResidentKv` pass | `ResidentKv` for text | pass + split multimodal pass | text split/cache and split multimodal ready |
| `phi` | `phi3` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `phi2` | `phi2` | selected | yes | pass | rejected-too-large | `ResidentKv` | pass | runtime-slice and resident cache ready; full-state payload is too large for mobility |
| `granite` | `granite` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `granite_hybrid` | `granite-hybrid` | selected | yes | pass | rejected-too-large | `KvRecurrent` | pass | runtime-slice and recurrent cache restore ready; keep normal decode ownership sticky |
| `granite_moe` | `granite-moe` | layout probe selected | yes | pass | pass | `ResidentKv` | pass | tiny random layout probe only; replace with a real small artifact when available |
| `hunyuan_dense` | `hunyuan-dense` | selected | yes | pass | pass | `ResidentKv` | pass | runtime-slice and cache restore ready |
| `hunyuan_moe` | `hunyuan-moe` | selected | yes | pass | pass | `ResidentKv` | pass | real A13B MoE runtime-slice and cache restore ready |
| `hunyuan_vl` | `hunyuan-vl` | selected | yes | pass | untested | split media path | split multimodal pass | HunyuanOCR projector split multimodal ready |
| `bloom` | `bloom` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `gptneox` | `gptneox` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `openai_moe` | `openai-moe` / `gpt-oss` | selected | yes | pass | pass | `ResidentKv` | pass | GPT-OSS 20B split/cache ready |
| `baichuan` | `baichuan` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `exaone` | `exaone` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `exaone4` | `exaone4` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `ernie4_5_moe` | `ernie4-5-moe` | selected | yes | pass | pass | `ResidentKv` | pass | ERNIE 4.5 MoE split/cache ready |
| `nemotron_h_moe` | `nemotron-h-moe` | package selected | yes | package validated | rejected-too-large | `KvRecurrent` | package validated | `lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:Q4_K_M` package-only certified; 52-layer package materialized and validated with all 401 tensors accounted for |
| `command_r` | `command-r` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `cohere2` | `cohere2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `jamba` | `jamba` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent cache restore ready; middle stage can be recurrent-only |
| `kimi_linear` | `kimi-linear` | selected | yes | pass | rejected-too-large full state; `KvRecurrent` pass | `KvRecurrent` | pass | KDA recurrent ranges plus sparse K-only MLA KV pages split/cache ready |
| `falcon` | `falcon` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `internlm2` | `internlm2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `stablelm` | `stablelm` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `starcoder2` | `starcoder2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `mamba` | `mamba` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `mamba2` | `mamba2` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `rwkv6` | `rwkv6` | replacement selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `qwen2vl` | `qwen2vl` | selected | yes | pass | FullState blocked by M-RoPE; `ResidentKv` pass | `ResidentKv` for text | pass + split multimodal pass | text split/cache and split multimodal ready |
| `qwen2moe` | `qwen2moe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `qwen3moe` | `qwen3moe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `qwen3vlmoe` | `qwen3vlmoe` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` for text | pass + split multimodal pass | Qwen3-VL-MoE split/cache and split multimodal ready |
| `arcee` | `arcee` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready; q8 rejected |
| `chatglm` | `chatglm` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready; q8 rejected |
| `codeshell` | `codeshell` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready; q8 rejected |
| `deci` | `deci` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready; q8 rejected |
| `qwen35` | `qwen35` | selected | yes | pass | `KvRecurrent` pass; FullState too large | `KvRecurrent` | pass | recurrent cache restore ready; keep normal decode ownership sticky; q8 validated |
| `qwen35moe` | `qwen35moe` | package selected | yes | package validated | package validated | `KvRecurrent` | package validated | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL` package-only certified; 40-layer package materialized and validated with all 733 tensors accounted for |
| `xverse` | `xverse` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready; q8 validated |
| `maincoder` | `maincoder` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `mimo2` | `mimo2` | package/remote only | no | package/remote pending | package/remote pending | package-local `ResidentKv` target | pending | HF re-audit found MiMo-V2.5 and MiMo-V2-Flash GGUF artifacts; package-sized multimodal model requiring projector parity |
| `openelm` | `openelm` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `minicpm` | `minicpm` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `minicpm3` | `minicpm3` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `plamo3` | `plamo3` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `plm` | `plm` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `refact` | `refact` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `smallthinker` | `smallthinker` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `smollm3` | `smollm3` | selected | yes | pass | `ResidentKv` pass | `ResidentKv` | pass | cache restore ready |
| `llama4` | `llama4` | package selected | yes | package validated | package validated | package-local `ResidentKv` target | package validated | `ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M` package-only certified; 48-layer split-GGUF package materialized and validated with all 627 owned tensors accounted for |
| `seed_oss` | `seed-oss` | package selected | yes | package validated | package validated | package-local `ResidentKv` target | package validated | `lmstudio-community/Seed-OSS-36B-Instruct-GGUF:Q4_K_M` package-only certified; 64-layer package materialized and validated with all 771 tensors accounted for |
| `deepseek2ocr` | `deepseek2ocr` | selected | yes | pass | pass | `ResidentKv` | pass + split multimodal pass | DeepSeek-OCR split/cache and split multimodal ready |

Broader coverage lives in `docs/skippy/llama-parity-candidates.json`. The board
above tracks the active certification queue rather than every pinned llama.cpp
implementation.

Any lower-precision wire remarks retained in this document or the candidate
inventory are historical experiment evidence only. The shipping activation
wire is fixed raw f32 and no longer has a family-specific dtype certification.

## Next Batch

1. Finish package or remote certification for MiMo-V2 and Exaone-MoE.
2. Add Gemma3n runtime-slice graph support before attempting text or multimodal
   promotion; its current graph carries 3D AltUp/per-layer state and has no
   stage-filter hooks.
3. Keep non-causal auxiliary rows in their own serving lanes instead of
   promoting them as causal stage-split serving.
4. Keep the remaining long-tail multimodal rows in `candidate_multimodal` until
   they have real projector/media sideband split evidence.

## Current Local Evidence

These rows were collected primarily on the local Mac Studio against the Metal
stage ABI. They are cheap text-split and cache-smoke evidence, not full
promotion by themselves until the reviewed topology records are updated.
Rows with distributed evidence call out the second backend explicitly.

| Family | Artifact | Text Split | Historical q8 Wire | Exact State | Cache |
| --- | --- | --- | --- | --- | --- |
| `qwen2` | `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 10.82x cache-hit speedup |
| `deepseek` | `Morgen0052/deepseek-llm-7b-chat-Q4_K_M-GGUF` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1.58x cache-hit speedup |
| `openai_moe` | `ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` state handoff passed; llama.cpp model file is `openai-moe`, GGUF architecture is `gpt-oss` |
| `ernie4_5_moe` | `lmstudio-community/ERNIE-4.5-21B-A3B-PT-GGUF:Q4_K_M` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` state handoff passed |
| `laguna` | `meshllm/laguna-s-2.1-Q4_K_M-layers@0c467ad441ee94cb5a76f626294d963c4048507d` | package-backed `single-step` and three-stage `chain` parity passed; ordinary three-stage CUDA Mesh serving completed two 128,080-token prompts | untested | accepted through bounded recurrent checkpoint/replay | Three-stage CUDA placement `0..25 / 25..39 / 39..48` used a 131,072-token allocation, Q4_0 K/V, F16 activation wire, and package-default suffix N-gram depth 2. The two 128K cold prompts ran at 956.12/958.05 prompt tok/s and 12.45/12.59 generation tok/s. A 44,460-token exact replay restored 44,416 tokens; structured OpenAI tool call and result turns passed; 12/12 subsequent stability requests passed. This evidence promotes only the pinned Q4_K_M package. The 128K repeats reported zero cached tokens; 256K, Q8 wire, other quants, and Metal at 128K remain unproven. |
| `qwen3next` (Inkling) | `meshllm/inkling-UD-Q2_K_XL-layers@9b4b91a7ddd978dd7a01679bc977f6e53777f2c7` | experimental two-stage CUDA text serving passed at a 131,072-token allocation | rejected; F32 required | exact replay accepted; recurrent verification recovery exercised | Automatic placement `0..65 / 65..66` on a direct approximately 5 ms path generated at 14.98 tok/s and completed two sequential native OpenAI tool loops. Exact replay restored 3,531/3,531 prompt tokens and the native fatal-pattern scan passed. This lopsided research topology reserved about 589 GiB of head host workspace and is not a deployment recommendation. Overlapping admission and changed-tail same-prefix cache reuse still fail the formal harness; native MTP, multimodal serving, and clean Metal teardown remain unproven. |
| `llama4` | `ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M` | package validated | untested | untested | package-only validation passed: 48 layers, 627 owned tensors, 51 artifacts, no missing/duplicate tensors |
| `mistral4` | `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF:IQ2_XXS` | package validated | untested | untested | package-only validation passed: 36 layers, 579 tensors, 39 artifacts, no missing/duplicate tensors |
| `nemotron_h_moe` | `lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:Q4_K_M` | package validated | untested | rejected-too-large | package-only validation passed: 52 layers, 401 tensors, 55 artifacts; `KvRecurrent` target |
| `seed_oss` | `lmstudio-community/Seed-OSS-36B-Instruct-GGUF:Q4_K_M` | package validated | untested | untested | package-only validation passed: 64 layers, 771 tensors, 67 artifacts, no missing/duplicate tensors |
| `glm4_moe` | see `target/family-certify/llama-parity-glm4-moe-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` native-sequence remap cache smoke passed |
| `mistral3` | `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 88.40x cache-hit speedup |
| `baichuan` | see `target/family-certify/llama-parity-baichuan-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 130.11x cache-hit speedup |
| `phi` | see `target/family-certify/llama-parity-phi-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 2645.30x cache-hit speedup |
| `phimoe` | see `target/family-certify/llama-parity-phimoe-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed after PhiMoE staged-graph support | validated | accepted | `ResidentKv` native-sequence remap cache smoke passed |
| `bloom` | see `target/family-certify/llama-parity-bloom-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 328.66x cache-hit speedup |
| `gptneox` | see `target/family-certify/llama-parity-gptneox-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 282.70x cache-hit speedup |
| `stablelm` | see `target/family-certify/llama-parity-stablelm-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 211.80x cache-hit speedup |
| `phi2` | see `target/family-certify/llama-parity-phi2-runtime-slice-2`, `target/family-certify/llama-parity-phi2-resident-kv-1`, and `/Volumes/External/tmp/skippy-phi2-resident-kv-stage1-20260506.json` | `single-step`, `chain`, and dtype matrix passed | validated | rejected-too-large for full-state | `ResidentKv` one-stage and split-final cache restore passed; stage0 cache probe remaps resident KV but cannot produce logits because it is a non-output slice |
| `cohere2` | see `target/family-certify/llama-parity-cohere2-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 63.48x cache-hit speedup |
| `command_r` | none | blocked | blocked | blocked | Lumia101/c4ai Command-R 7B reports GGUF architecture `cohere2`; find a true `command-r` artifact before promotion |
| `arcee` | `bartowski/arcee-ai_AFM-4.5B-GGUF:IQ2_M` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 8-token prefix |
| `chatglm` | `mradermacher/chatglm3-6b-i1-GGUF:IQ2_M` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 8-token prefix |
| `codeshell` | `mradermacher/CodeShell-7B-i1-GGUF:IQ2_M` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 8-token prefix |
| `deci` | `mradermacher/DeciLM-6b-instruct-i1-GGUF:IQ2_M` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 8-token prefix |
| `qwen35` | `mradermacher/UnifiedReward-Edit-qwen35-4b-i1-GGUF:IQ2_M` | `single-step`, `chain`, and dtype matrix passed | validated | FullState rejected-too-large | `KvRecurrent` smoke passed, 8-token prefix; keep normal decode ownership sticky |
| `xverse` | `xverse/XVERSE-7B-Chat-GGUF:q2_k` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 8-token prefix |
| `falcon` | see `target/family-certify/llama-parity-falcon-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1753.04x cache-hit speedup |
| `exaone` | see `target/family-certify/llama-parity-exaone-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 868.76x cache-hit speedup |
| `exaone4` | see `target/family-certify/llama-parity-exaone4-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 220.01x cache-hit speedup |
| `internlm2` | see `target/family-certify/llama-parity-internlm2-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 80.78x cache-hit speedup |
| `granite` | see `target/family-certify/llama-parity-dense-tranche-2-granite-fix2` and `/tmp/skippy-cache-correctness-dense-medium` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed; fixed staged activation rescaling |
| `granite_hybrid` | see `target/family-certify/llama-parity-granite-hybrid-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | rejected-too-large for full-state | `KvRecurrent` cache smoke passed; fixed Granite-Hybrid graph stage filtering |
| `granite_moe` | see `target/family-certify/llama-parity-granite-moe-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed on tiny random layout probe; fixed Granite-MoE staged-graph support |
| `hunyuan_dense` | see `target/family-certify/llama-parity-hunyuan-dense-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed; fixed Hunyuan-Dense graph stage filtering |
| `hunyuan_moe` | see `target/family-certify/llama-parity-hunyuan-moe-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed on real A13B MoE GGUF; fixed Hunyuan-MoE graph stage filtering |
| `kimi_linear` | see `target/family-certify/llama-parity-kimi-linear-20260507h` | `single-step`, `chain`, and dtype matrix passed after sparse K-only MLA KV export/import support | validated | rejected-too-large for full-state | `KvRecurrent` handoff passed with source/target sequence remap, suffix prefill, repeated hit stability, 64,512 KV bytes, and 44,892,668 recurrent bytes |
| `starcoder2` | see `target/family-certify/llama-parity-starcoder2-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 198.10x cache-hit speedup |
| `gpt2` | see `target/family-certify/llama-parity-gpt2-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1535.17x cache-hit speedup |
| `gemma` | see `target/family-certify/llama-parity-gemma-f32-wire-1` and `/tmp/skippy-cache-correctness-dense-gemma` | `single-step`, `chain`, and dtype matrix passed with `f32` only | rejected | accepted | `ResidentKv` cache smoke passed with `f32`; `f16` predicted token `0`, `q8` predicted token `107` |
| `mpt` | see `target/family-certify/llama-parity-mpt-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1657.77x cache-hit speedup |
| `olmo2` | see `target/family-certify/llama-parity-olmo2-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 159.19x cache-hit speedup |
| `olmoe` | see `target/family-certify/llama-parity-olmoe-runtime-slice-1` and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 197.09x cache-hit speedup; MoE expert-stage smoke passed for one-stage, split-middle, and split-final |
| `qwen2vl` | see `target/family-certify/llama-parity-candidate-multimodal-20260506b`; split smoke via `cargo test -p skippy-server real_multimodal_split_smoke_when_fixture_is_set` with `Qwen2-VL-2B-Instruct-Q4_K_M.gguf`, `mmproj-Qwen2-VL-2B-Instruct-f16.gguf`, and `test-1.jpeg` | text `single-step`, `chain`, f32, and f16 passed after allowing tied output embeddings in final runtime slices; split multimodal passed after sampled final media prefill and position sideband forwarding | rejected: q8 predicted token `362` vs baseline `11` | FullState restore blocked by M-RoPE native-position rules; production `ResidentKv` text cache passed | `ResidentKv` one-stage borrowed-hit smoke passed, 8-token prefix, 229,376 resident bytes, 2.43x cache-hit speedup; split multimodal smoke passed |
| `qwen3vl` | see `target/family-certify/llama-parity-candidate-multimodal-20260506b`; split smoke via `cargo test -p skippy-server real_multimodal_split_smoke_when_fixture_is_set` with `Qwen3VL-2B-Instruct-Q4_K_M.gguf`, `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`, and `test-1.jpeg` | text `single-step`, `chain`, and dtype matrix passed after allowing tied output embeddings in final runtime slices; split multimodal passed after native input-width activation padding, sampled final media prefill, and position sideband forwarding | validated | FullState restore blocked by M-RoPE native-position rules; production `ResidentKv` text cache passed | `ResidentKv` one-stage borrowed-hit smoke passed, 8-token prefix, 917,504 resident bytes, 1.83x cache-hit speedup; split multimodal smoke passed |
| `hunyuan_vl` | split smoke via `cargo test -p skippy-server real_multimodal_split_smoke_when_fixture_is_set` with `HunyuanOCR-Q8_0.gguf`, `mmproj-HunyuanOCR-Q8_0.gguf`, and `test-1.jpeg` | split multimodal passed with the shared Hunyuan-Dense/VL graph filter | untested | untested | projector/media sideband split smoke passed over raw f32 activation transport |
| `deepseek2ocr` | see `target/family-certify/llama-parity-new-mm-candidates-20260507b`; split smoke via `cargo test -p skippy-server real_multimodal_split_smoke_when_fixture_is_set` with `DeepSeek-OCR-Q8_0.gguf`, `mmproj-DeepSeek-OCR-Q8_0.gguf`, and `test-1.jpeg` | text `single-step`, `chain`, dtype matrix, and split multimodal passed | rejected | accepted | `ResidentKv` handoff and split multimodal smoke passed |
| `qwen3vlmoe` | see `target/family-certify/llama-parity-new-mm-candidates-20260507b`; split smoke via `cargo test -p skippy-server real_multimodal_split_smoke_when_fixture_is_set` with `Qwen3-VL-30B-A3B-Instruct-1M-MXFP4_MOE.gguf`, `mmproj-F16.gguf`, and `test-1.jpeg` | text `single-step`, `chain`, dtype matrix, and split multimodal passed | rejected | accepted | `ResidentKv` handoff and split multimodal smoke passed; stage readiness wait allows large local fixtures |
| `gemma3n` | see `target/family-certify/llama-parity-gemma3n-cpu-20260507d` | `single-step`, chain split `0..10,10..15,15..30`, dtype matrix, and state handoff passed after adding AltUp sideband activation frames | validated | accepted for text `ResidentKv` policy | multimodal/projector smoke remains separate, but text split serving is certified |
| `exaone_moe` | `LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF:Q4_K_M` | package-only certified from remote GGUF metadata: architecture `exaone-moe`, 49 layers, width 6144, 128 experts, 8 active experts, 143471722240 bytes | untested | rejected-too-large | full monolithic correctness is too large for cheap local hardware; package/stage-policy evidence is the support contract |
| `bert` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | non-causal aux; downloaded and inspected | not applicable | not applicable | GGUF reports `nomic-bert`, 12 blocks, width 768; do not promote as causal stage-split serving |
| `t5` | `cyanic-selkie/flan-t5-small-Q4_K_M-GGUF` | non-causal aux; downloaded and inspected | not applicable | not applicable | GGUF reports `t5`, 8 blocks, width 512; encoder-decoder support needs a separate serving lane |
| `qwen2moe` | see `target/family-certify/llama-parity-qwen2moe-runtime-slice-4`, `/tmp/skippy-cache-correctness-dense-medium`, and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` cache smoke and MoE expert-stage smoke passed |
| `qwen3moe` | see `target/family-certify/llama-parity-qwen3moe-runtime-slice-2`, `/tmp/skippy-cache-correctness-dense-medium`, and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke and MoE expert-stage smoke passed |
| `lfm2` | see `target/family-certify/llama-parity-lfm2-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed; keep recurrent ownership sticky for normal decode |
| `jamba` | see `target/family-certify/llama-parity-jamba-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed; middle-stage recurrent-only slices are valid |
| `mamba` | see `target/family-certify/llama-parity-mamba-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes |
| `mamba2` | see `target/family-certify/llama-parity-mamba2-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes |
| `rwkv6` | see `target/family-certify/llama-parity-rwkv6-runtime-slice-3` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes; keep recurrent ownership sticky |
| `rwkv7` | `Mungert/rwkv7-191M-world-GGUF` plus `target/family-certify/rwkv7-sideband-*.json` | `single-step`, `chain`, and dtype matrix passed | validated on sampled artifact | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes; activation-frame sideband carries layer-0 `v_first` |

Raw run directories:

- `target/family-certify/llama-parity-cheap-qwen2-full-4`
- `target/family-certify/llama-parity-cheap-mistral-full`
- `target/family-certify/llama-parity-cheap-lfm2`
- `target/family-certify/llama-parity-cheap-olmoe`
- `target/family-certify/llama-parity-cheap-qwen3vl-text`
- `target/family-certify/llama-parity-cheap-available`
- `target/family-certify/llama-parity-external-available`
- `target/family-certify/llama-parity-cache-deepseek`
- `target/family-certify/llama-parity-dense-tranche-1`
- `target/family-certify/llama-parity-dense-tranche-2`
- `target/family-certify/llama-parity-dense-tranche-2-granite-fix2`
- `target/family-certify/llama-parity-dense-tranche-2-external`
- `target/family-certify/llama-parity-starcoder2-runtime-slice-1`
- `target/family-certify/llama-parity-mpt-runtime-slice-1`
- `target/family-certify/llama-parity-olmo2-runtime-slice-1`
- `target/family-certify/llama-parity-olmoe-runtime-slice-1`
- `target/family-certify/llama-parity-phimoe-runtime-slice-2`
- `target/family-certify/llama-parity-glm4-moe-runtime-slice-1`
- `target/family-certify/llama-parity-remaining-local-1`
- `target/family-certify/llama-parity-remaining-external-1`
- `target/family-certify/llama-parity-decoder-tranche-3c`
- `target/family-certify/llama-parity-decoder-tranche-3e`
- `target/family-certify/llama-parity-gemma-f32-wire-1`
- `target/family-certify/llama-parity-jamba-runtime-slice-2`
- `target/family-certify/llama-parity-lfm2-runtime-slice-2`
- `target/family-certify/llama-parity-mamba-runtime-slice-2`
- `target/family-certify/llama-parity-mamba2-runtime-slice-2`
- `target/family-certify/llama-parity-granite-hybrid-runtime-slice-2`
- `target/family-certify/llama-parity-granite-moe-runtime-slice-2`
- `target/family-certify/llama-parity-hunyuan-dense-runtime-slice-2`
- `target/family-certify/llama-parity-hunyuan-moe-runtime-slice-1`
- `target/family-certify/llama-parity-qwen2moe-runtime-slice-4`
- `target/family-certify/llama-parity-qwen3moe-runtime-slice-2`
- `target/family-certify/llama-parity-candidate-multimodal-20260506b`
- `target/family-certify/llama-parity-qwen2vl-tied-embd-fix-20260506`
- `target/family-certify/llama-parity-qwen3vl-tied-embd-fix-20260506`
- `target/family-certify/rwkv7-sideband-single-step.json`
- `target/family-certify/rwkv7-sideband-chain.json`
- `target/family-certify/rwkv7-sideband-dtype-matrix.json`
- `/tmp/skippy-cache-correctness-dense-small`
- `/tmp/skippy-cache-correctness-dense-medium`
- `/tmp/skippy-cache-correctness-dense-large`
- `/tmp/skippy-cache-correctness-dense-mistral3`
- `/tmp/skippy-cache-correctness-dense-gemma`
- `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506`
- `/Volumes/External/tmp/skippy-smoke-qwen3-coder-480b/state-handoff.json`

## Cache Correctness Evidence

The source/target native-sequence remap gate is reproducible with:

```bash
LLAMA_STAGE_BUILD_DIR=$PWD/.deps/llama-build/build-stage-abi-metal \
  python3 evals/skippy-cache-correctness-gate.py \
    --output-dir /tmp/skippy-cache-correctness-gate \
    --llama-stage-build-dir $PWD/.deps/llama-build/build-stage-abi-metal \
    --topology one-stage \
    --topology split-middle \
    --topology split-final \
    --prefix-tokens 16 \
    --suffix-token-count 3 \
    --runtime-lane-count 4 \
    --cache-hit-repeats 2 \
    --n-gpu-layers 999
```

Latest local result: `102/102` rows passed across tranche runs. Every row
restored into a different native sequence (`0 -> 1`), suffix-prefill-then-decode
matched normal prefill, and repeated hits stayed stable. `KvRecurrent` rows
carried non-zero recurrent payloads. Pure recurrent families and recurrent-only
stage ranges correctly recorded zero native KV bytes plus recurrent state.
Dense and MoE-text `ResidentKv` additions passed in:
`/tmp/skippy-cache-correctness-dense-small`,
`/tmp/skippy-cache-correctness-dense-medium`,
`/tmp/skippy-cache-correctness-dense-large`,
`/tmp/skippy-cache-correctness-dense-mistral3`, and
`/tmp/skippy-cache-correctness-dense-gemma`. An attempted unified all-in-one
rerun exited early during process startup before writing a report; the completed
tranche reports are the current evidence.

Negative policy regression tests now assert that Falcon-H1, Qwen3Next, Jamba,
Kimi Linear, LFM2, Mamba, Mamba2, RWKV6, and RWKV7 select `KvRecurrent`, never
`ResidentKv`, through mesh family policy and server-side auto-payload
inference.

| Family | Model ref | Payload | Topology | Result | Seq remap | Source -> target seq | Suffix prefill | Payload bytes | Recurrent bytes | Repeated hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Phi2 | `TheBloke/phi-2-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Phi2 | `TheBloke/phi-2-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Granite-Hybrid | `magiccodingman/Granite-4.0-H-350M-Unsloth-MXFP4-Hybrid-GGUF:MXFP4_MOE-output_q6_K-router_gate_emb_q6_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `22885052` | `22622908` | pass |
| Granite-MoE | `mradermacher/tiny-random-granite-moe-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Hunyuan-Dense | `Edge-Quant/Hunyuan-1.8B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `76923484` | `76530268` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `76661340` | `76530268` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `76661340` | `76530268` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `79430524` | `79037308` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `79168380` | `79037308` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `79168380` | `79037308` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `10134156` | `10117772` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `10117772` | `10117772` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `10125964` | `10117772` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `278796` | `82188` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `147724` | `82188` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `180492` | `82188` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
- `target/family-certify/llama-parity-rwkv6-runtime-slice-3`
- `target/family-certify/cache-smoke/reports`

## Current Blockers

- Runtime-slice expansion and cache restore now pass for `baichuan`, `bloom`,
  `command_r`, `cohere2`, `exaone`, `exaone4`, `falcon`, `gemma` with `f32`
  wire, `gpt2`, `gptneox`, `granite`, `granite_hybrid`, `granite_moe`,
  `hunyuan_dense`, `hunyuan_moe`, `internlm2`, `mistral3`, `mpt`, `olmo2`,
  `phi2`, `phi3`, `stablelm`, and `starcoder2`.
- `hunyuan_dense` required a llama.cpp stage-filter fix in the shared
  Hunyuan-Dense/VL graph and uses `ResidentKv`.
- `hunyuan_moe` required a llama.cpp stage-filter fix in the Hunyuan-MoE graph
  and uses `ResidentKv`.
- `granite_hybrid` required a llama.cpp stage-filter fix for its hybrid graph
  and uses `KvRecurrent`; the exact full-state payload is too large to move as a
  production cache value.
- `granite_moe` reuses the Granite graph. Current evidence uses a tiny random
  GGUF, so it certifies graph and tensor-layout support rather than model
  quality.
- `phi2` required a llama.cpp stage-filter fix for filtered fused-QKV tensors:
  when a merged QKV weight is skipped for a slice, the matching merged QKV bias
  must also be accounted for instead of falling back to separate Q/K/V tensors.
- `gemma3n` now has text split-serving support. Its graph needs a full AltUp
  sideband, and the safe multi-stage split keeps layers 20+ with KV-owner layers
  15..19 instead of using an even `0..10,10..20,20..30` split.
- `exaone_moe` is package-only: remote range metadata verified the public
  K-EXAONE Q4_K_M GGUF shape and architecture, but the 143.5 GB artifact is too
  large for the cheap monolithic correctness lane.
- `olmoe`, `qwen2moe`, and `qwen3moe` now pass MoE expert-stage smoke for
  one-stage, split-middle, and split-final ranges. The smoke confirms nonzero
  expert tensor ownership in each tested range, native sequence remap `0 -> 1`,
  suffix-prefill match, repeated hit stability, and nonzero resident KV bytes.
- `jamba`, `lfm2`, `mamba`, `mamba2`, `rwkv6`, and `rwkv7` now pass
  `KvRecurrent` cache smoke for one-stage, split-middle, and split-final
  restore into a different native sequence. Keep recurrent ownership sticky for
  normal staged decode; the cache proof is for exact prefix restore.
- `kimi_linear` now passes split text, chained split, f32/f16/q8 wire, and
  `KvRecurrent` restore into a different native sequence. The fixes were
  llama.cpp-side: optional filtered tensor probes are counted once, and native
  KV page export/import supports sparse K-only MLA pages alongside recurrent KDA
  state.
- `rwkv7` uses a wider activation-frame contract. Later RWKV7 layers depend on
  the layer-0 `v_first` tensor, so non-first stages receive hidden state plus a
  `v_first` activation sideband. The sampled 12-layer artifact now passes
  two-stage, three-stage, f32/f16/q8 wire checks, and `KvRecurrent` cache smoke.
- The old local `rwkv6` sample was not a GGUF artifact. Its files carry the
  legacy `fmgg` magic and fail GGUF metadata inspection, so the replacement
  candidate is `latestissue/rwkv-6-finch-1b6-gguf`. The replacement passed the
  text lane; exact state is intentionally rejected as too large.
- `gemma` is stage-correct only with `f32` activation wire for the sampled
  artifact. The earlier default-`f16` cheap run predicted token `0`, and `q8`
  predicted token `107`, while `f32` matched token `1106`.
- `llama4` does not have a cheap local certification artifact. The local
  `glogwa68/Llama-4-scout-GGUF` sample reports `general.architecture = llama`,
  not `llama4`, so official Scout Q4_K_M is certified through package
  materialization and validation instead of a monolithic full-model baseline.
- The old `mistral3` candidate was not a `mistral3` GGUF. It reports
  `general.architecture = llama`, so it cannot be used for llama.cpp family
  parity even though the run itself passed. The replacement candidate is
  `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF`.
- qwen2 full-state handoff was fixed by syncing token-count-aware imports back
  into the native session position before decode. Without that, restored KV was
  present but decode restarted at position zero.
- `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, `hunyuan_vl`/HunyuanOCR, and
  `deepseek2ocr` now pass split multimodal smoke with real projectors and an
  image fixture. The fix is not just a doc promotion: stage-0 media prefill now
  forwards each projector chunk separately, carries the M-RoPE position sideband
  across the stage protocol, samples from the final media prefill without
  re-decoding logits, and pads Qwen3-VL activation frames to the native input
  width expected by downstream llama.cpp graphs. Historical lower-precision
  results are retained as research evidence; shipping transport is raw f32.
  FullState restore remains blocked by M-RoPE native-position rules, so the
  production text-cache proof for Qwen VL remains `ResidentKv`.
- `bert` and `t5` are now downloaded and inspected, but intentionally remain
  outside the causal stage-split board. The BERT representative reports
  `general.architecture = nomic-bert`, 12 blocks, and width 768. The T5
  representative reports `general.architecture = t5`, 8 blocks, and width 512.
  They need embedding/encoder-decoder serving lanes rather than the decoder
  activation-handoff path.

## Cache Smoke Commands

The production cache gate should use `ResidentKv` or `KvRecurrent`, not
`FullState`. Dense families use this shape:

```bash
target/debug/skippy-correctness state-handoff \
  --model /path/to/model.gguf \
  --model-id org/repo \
  --layer-end 24 \
  --ctx-size 128 \
  --n-gpu-layers 999 \
  --prompt Hello \
  --stage-server-bin target/debug/skippy-server \
  --activation-width 896 \
  --source-bind-addr 127.0.0.1:19831 \
  --restore-bind-addr 127.0.0.1:19832 \
  --state-payload-kind resident-kv \
  --borrow-resident-hits \
  --cache-hit-repeats 3 \
  --prefix-token-count 64 \
  --report-out target/family-certify/cache-smoke/reports/family-resident-kv.json
```
