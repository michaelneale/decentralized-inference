# Model Family Testing Matrix

This document records which llama.cpp model families are in the internal
testing matrix (the download-and-certify corpus), which are deliberately
excluded, and why. It is the internal counterpart to the customer-facing
`docs/skippy/FAMILY_STATUS.md`: a family can be excluded here and still ship
customer support from prior certification evidence, and a family included here
has no support claim until it certifies through
`docs/skippy/NEW_MODEL_ONBOARDING.md`.

Decision authority: James approved the Tier 1 removal and the Tier 3 removal
on 2026-09-07 (see skippy channel: "lets drop all tier 1 from testing and from
the download script"; "tier 3 - remove them. we will do them in
https://github.com/Mesh-LLM/mesh-llm/issues/1677"). The corpus was derived
from the verified one-GGUF-per-architecture manifest embedded in the download
script.

Corpus state after both removals: **85 architectures, 99 files, 592 GB**,
which fits the 837 GB external scratch volume in one pass.

Download tool: `~/bin/download-all-families.py` (one verified GGUF per arch,
`general.architecture` confirmed via the HF parsed metadata API, resumable,
sha256-checked). The removed arch list is also recorded in that script's
header.

## Policy

The inclusion rule is **graph coverage, not recency and not completeness**. A
family earns a testing slot if it exercises a graph or state path that no other
included family covers. Age alone does not exclude a family; redundancy does.

- A family is a candidate for removal when a newer sibling exercises the same
  or a stricter code path (same attention variant, same normalization scheme,
  same state machinery) and the old artifact adds no unique coverage.
- Families whose only distinguishing feature is vintage but which carry a
  unique graph path stay (see the keep list below).
- Non-text-generation modalities (encoders/embeddings, TTS, vision/OCR,
  diffusion LMs, standalone speculators) are out of the corpus: the staged
  runtime and its OpenAI-compatible frontend are scoped to autoregressive
  text generation. Support for those classes is tracked in
  <https://github.com/Mesh-LLM/mesh-llm/issues/1677>; the download script
  header lists the exact removed archs.

## Removed — Tier 1: old and graph-superseded (2026-09-07)

These 28 architectures were removed from the testing matrix and from the
download script. Each maps to a retained family covering the same graph path.

| Arch | Superseded by |
| --- | --- |
| dbrx | MoE coverage via granitemoe / olmoe / qwen3moe |
| phimoe | phi3 plus modern MoE lanes |
| rwkv6qwen2 | rwkv6 and rwkv7 both retained |
| command-r | cohere2 |
| qwen | qwen2 lineage retained upstream of qwen3/qwen35 |
| qwen2 | qwen3 / qwen35 (largest real-world deployment arch; removal is explicit and reversible) |
| qwen2moe | qwen3moe |
| starcoder | starcoder2 |
| orion | llama-like, no unique path |
| falcon | parallel-residual covered by gptneox; falcon-h1 retained |
| bloom | ALiBi covered by mpt |
| codeshell | code lanes via starcoder2 / maincoder |
| chatglm | glm4 / glm4moe |
| baichuan | llama-like |
| exaone | exaone4 |
| minicpm3 | minicpm |
| deci | llama-like depth-scaled |
| gptj | parallel residual covered by gptneox |
| gpt2 | dead-end lineage, no unique path |
| phi2 | phi3 |
| xverse | llama clone |
| plamo | plamo2 / plamo3 |
| internlm2 | llama-like |
| stablelm | olmo2 and peers |
| refact | starcoder2 |
| openelm | llama-ish depth-scaled |
| olmo | olmo2 |
| nemotron | nemotron_h |

Corpus effect: 141 -> 113 archs, ~966 -> 710 GB.

## Removed — Tier 3: non-text modalities (2026-09-07, deferred to #1677)

Removed the same day, per the same decision thread. The staged runtime's scope
is autoregressive text generation, so these 28 architectures left the corpus
until issue #1677 lands embedding, OCR, TTS, and encoder-only support:

- Encoders / embeddings: bert, modern-bert, neo-bert, nomic-bert,
  nomic-bert-moe, jina-bert-v2, jina-bert-v3, eurobert, gemma-embedding,
  llama-embed, t5, t5encoder.
- Audio / TTS: pockettts, qwen3tts, wavtokenizer-dec.
- Vision / OCR: cogvlm, qwen2vl, qwen3vl, qwen3vlmoe, deepseek2-ocr,
  hunyuan_vl, paddleocr.
- Diffusion LMs: llada, llada-moe, dream, rnd1.
- Draft/speculative-only: eagle3, dflash.

Kept despite the modality question because their profiles are planner targets:
step35 and qwen4exp (MTP/speculative).

Corpus effect: 113 -> 85 archs, 710 -> 592 GB.

Note: qwen2vl and qwen3vl are currently *Supported* (split multimodal) in
`docs/skippy/FAMILY_STATUS.md`. Their removal here stops active
re-certification only; the existing support claims stand until #1677 changes
the picture.

Note: historical certification evidence for these families in
`docs/skippy/FAMILY_STATUS.md` (Bloom, GPT2, ChatGLM, CodeShell, Deci, XVerse,
StableLM, OpenELM, OLMo, MiniCPM3, InternLM2, Refact, Falcon, and others) is
**retained as-is**. Removing a family from the testing corpus does not revoke
existing support claims; it only stops active re-certification.

## Retained — old but graph-unique

Kept despite age because nothing newer covers their path:

- **gptneox** — parallel residual with unshared layer norm; oldest surviving
  distinct topology in the matrix.
- **mpt** — only ALiBi representative after the bloom removal.
- **rwkv6** — pre-v7 recurrent gating; rwkv7 is a different mechanism, not a
  superset.
- **jamba** — attention+Mamba interleave; granitehybrid and nemotron_h are
  different hybrid layouts.
- **deepseek2** — MLA baseline; deepseek4 is a rewrite, so the baseline stays.
- **grok** — its own arch with a single public sample.

## Open Questions

1. Keep qwen2 in the corpus after all? It was removed under Tier 1, but it
   remains the largest real-world deployment architecture.
2. The 7 architectures with no public GGUF anywhere on Hugging Face
   (arctic, bailingmoe, bailingmoe2, deepseek, exaone-moe, minimax-01,
   pangu-embedded) need upstream conversion before they can enter the matrix
   at all; conversion work is not currently scheduled.
