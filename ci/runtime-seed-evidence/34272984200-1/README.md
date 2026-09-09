# Existing CPU seed qualification

Retained artifact evidence from [run 34272984200, attempt 1](https://github.com/Mesh-LLM/mesh-llm/actions/runs/34272984200), source `1f4545616e98db715e37c57e1196cbdc975a010e`. `sha256.json` hashes original retained file bytes.

All six real runtime package actions verified. All three warm samples restored exact main cache 7456497330 and observed zero hits: each had 603 C/C++, 139 assembler and 304 Rust misses. Cache read/write errors and timeouts were zero. Warm 1% floor failures are expected negative qualification outcomes. Whole-action counters include packaging dependencies; no exact cause of misses is established.

Pair 1 and 2 hardware differs; the full cohort cannot establish comparable timing or improvement. Pair 3 hardware matches but does not supply three comparable pairs. Runtime restore is deliberately excluded for this seed/image/workload. Other workload coverage remains unknown.
