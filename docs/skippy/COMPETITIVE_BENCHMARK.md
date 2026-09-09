# Mesh versus llama.cpp competitive benchmark

`evals/skippy-competitive-benchmark.py` is the reproducible entrypoint for the
cross-platform scheduler benchmark. It compares the release `skippy-server`
OpenAI surface with the repository's pinned raw llama.cpp baseline without
changing serving code during a run.

The checked-in contract covers:

- CUDA and Metal;
- Llama 3.2 1B dense, DeepSeek Coder V2 Lite MoE, Falcon-H1 recurrent,
  and Granite 4.0 H 1B hybrid;
- llama-benchy `pp=512`, `tg=8/64/256`, exact output length;
- 256 deterministic prompts derived from the Thoughtworks agentic coding
  trajectories dataset;
- offered concurrency `1/2/4/8/16/32/64/128/256`; and
- raw results, parity evidence, CSV tables, SVG charts, `REPORT.md`, and a
  SHA-256 inventory.

The Thoughtworks subset is pinned to the MIT-licensed
`swe-smith-claude-3-7-sonnet` source. The eight selected trajectory identities,
dataset commit, dataset checksum, and generated manifest checksum live in
`evals/skippy-competitive-benchmark.json`. The exact llama-benchy tokenizer
directory digest for every model is pinned there as well. Corpus, tokenizer,
and model bytes are never checked into the repository.

## Inspect the matrix

`plan` is side-effect free:

```bash
python3 evals/skippy-competitive-benchmark.py plan
```

The full required raw-versus-Mesh matrix contains 576 arm-level cells. Optional
comparison arms add cells only where their pinned model inputs are available.
Filters are repeatable:

```bash
python3 evals/skippy-competitive-benchmark.py plan \
  --platform metal \
  --model falcon-h1-recurrent \
  --workload thoughtworks
```

## Acquire pinned inputs

Run `prefetch` outside the timed benchmark window. It invokes `hf download` at
the checked-in revisions, invokes `hf cache verify`, verifies each required
file's SHA-256, regenerates the prompt manifest, then verifies its SHA-256 and
row provenance.

```bash
python3 evals/skippy-competitive-benchmark.py prefetch \
  --model-root /path/to/competitive/models \
  --dataset-root /path/to/competitive/dataset \
  --manifest /path/to/competitive/thoughtworks-256.json
```

The manifest generator requires DuckDB in the selected Python environment.
The model and dataset SHA-256 checks are authoritative for the selectively
downloaded files; `hf cache verify` additionally validates every locally
present Hub file without requiring unrelated quantizations from the same repo.
In the two-machine lab, download from Studio54 with
`HF_HOME=/Volumes/External/models/huggingface`; Micstudio reads the shared NFS
cache and must not download the same input concurrently.

The llama-benchy tokenizer directories are inputs rather than generated report
data. Put the pinned directories under `<tokenizer-root>/<model-key>`; `run`
hashes every relative file and fails before timing if a directory differs from
the checked-in digest.

`scripts/materialize-competitive-inputs.sh` materializes every pinned input —
GGUF models, tokenizer directories, per-model vLLM `config.json` files, the
Thoughtworks parquet, and the deterministic prompt manifest — from Hugging Face
into the HF cache and verifies each digest before linking a runnable input tree:

```bash
scripts/materialize-competitive-inputs.sh --root /path/to/competitive
```

The Python environment needs `transformers` v5 (tokenizer re-exports) and
`duckdb` (manifest generation). deepseek/falcon tokenizer directories are
transformers v5 re-exports of the pinned `vllm_hf_config` revisions — verified
byte-identical to the checked-in digests — and granite's directory is the
pinned snapshot minus `README.md`; the script header documents each provenance.

Granite's alternate container digest is derived only after the model-specific
value check succeeds. In a Python environment with `gguf`, `numpy`,
`safetensors`, and `torch`:

```bash
python3 evals/skippy-granite-tensor-equivalence.py \
  --gguf /path/to/granite-4.0-h-1b-bf16.gguf \
  --safetensors /path/to/model.safetensors
```

The verifier covers every source and GGUF tensor, including the official MLP
split, q/k permutation, BF16-to-F32 scalar promotion, and Mamba `A_log`
conversion, then prints the canonical BF16 tensor digest pinned in the config.

## Run one hardware platform

Build `skippy-server` in release mode before measuring. Run the same entrypoint
on each hardware host and point both at one artifact root. The runner records
the exact Git heads and SHA-256 values of both binaries and refuses to resume
into a directory created by different binaries or config.

```bash
python3 evals/skippy-competitive-benchmark.py run \
  --platform metal \
  --output /path/to/artifact \
  --model-root /path/to/competitive/models \
  --tokenizer-root /path/to/competitive/tokenizers \
  --manifest /path/to/competitive/thoughtworks-256.json \
  --mesh-root "$PWD" \
  --mesh-binary "$PWD/target/release/skippy-server" \
  --native-dir "$PWD/.deps/llama-build/build-stage-abi-static-metal" \
  --llama-root /path/to/pinned/llama.cpp \
  --llama-binary /path/to/pinned/llama.cpp/build/bin/llama-server \
  --benchy /path/to/llama-benchy
```

Use `--platform cuda` and that host's native runtime directory for the CUDA
run. `--model` and `--workload` may be repeated for a narrowed diagnosis.
Resume is on by default. A cell is skipped only when its completion marker
matches the config, manifest, and relevant binary hashes.

Synthetic cells use four active execution lanes. Thoughtworks runtime shape is
pinned per family: dense Llama and Granite use 131,072 total KV tokens and 16
live sequences, DeepSeek uses 16,384/8, and Falcon uses 16,384/2. Every backend
inside a family row receives the same context, total KV-token budget, and live
sequence cap. Offered concurrency still reaches 256: requests beyond the active
lanes exercise waiting admission, queueing, and scheduler behavior. The trace
arms alternate raw/Mesh and Mesh/raw across the concurrency ladder to reduce
time-order bias. Reports keep separate tables and charts per family; they do
not compare or aggregate absolute throughput across models.

On Linux CUDA, `--mesh-adaptive` adds a staged Mesh arm whose committed active
generation permit count starts at one. Under sustained queued load it tentatively
probes one additional lane, keeps it only when useful-token retirement improves
without hardware service-time or p95 latency pressure, and otherwise drains back
to the last committed limit. Failed generations force immediate rollback or
backoff, and a cooldown periodically re-probes after workload changes. Use
`--comparison-backend vllm` and/or `--comparison-backend sglang` for optional
external comparisons. vLLM serves the pinned GGUF with the pinned tokenizer by
default and requires `vllm-gguf-plugin` in the vLLM virtual environment plus
the pinned per-model configs under `--vllm-hf-config-root`. A native alternate
container may be supplied under `--vllm-model-root`.
SGLang likewise defaults to the pinned GGUF; an alternate per-model input may
be supplied with `--sglang-model-root`. Alternate containers are accepted only
when their directory digest, source revision, and canonical tensor-equivalence
digest are pinned in the model config. Granite uses the official BF16 GGUF for
raw/Mesh and its value-equivalent official BF16 safetensors for vLLM/SGLang.
Capacity matching accounts for model-specific vLLM cache page geometry:
Granite pins the observed vLLM 0.27.1 ratio of 2,560 cache blocks per 131,072
total tokens, rather than treating its internally enlarged Mamba page as a
generic 16-token attention block.
Missing optional runtimes or model inputs are recorded in
`comparisons/<platform>/availability.json` and skipped without weakening the
required raw llama.cpp/fixed Mesh matrix.
Pass `--require-comparison-backends` for a controlled comparison that must
produce every requested external-backend number; preflight then fails before
timing if either runtime or any model input is missing. The next controlled
CUDA rerun uses this flag with both vLLM and SGLang.

Mesh arms pin `--generation-queue-capacity 256` and
`--generation-admission-timeout-secs 600`; these are benchmark overrides, not
production defaults. Active `--generation-concurrency` remains equal to the
manifest's KV-backed lane count, so waiting requests do not create native/KV
lanes.

Do not time benchmarks while model downloads, builds, or unrelated GPU work are
active. Record the host-isolation decision with the artifact when the hardware
is shared.

## Produce the artifact

After both platforms finish:

```bash
python3 evals/skippy-competitive-benchmark.py report \
  --artifact /path/to/artifact
```

The report phase does not start a server. It writes:

```text
artifact/
  benchmark-config.json
  provenance/<platform>.json
  plans/<platform>.json
  data/<platform>/<model>/<arm>/...
  trace/<platform>/<model>/c-<concurrency>/<arm>/...
  summary/
    synthetic.csv
    thoughtworks.csv
    parity.csv
    charts/*.svg
    REPORT.md
  artifact-sha256.txt
```

Throughput is marked competitive only when the paired c1 deterministic
continuation matches exactly and both arms completed every request in the
cell. Failed parity, HTTP overload, timeouts, or incomplete waves remain in the
artifact but are labeled diagnostic.
