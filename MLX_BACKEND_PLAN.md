# MLX Backend Plan

## Goal

Make MLX the future default backend for `mesh-llm`, with tensor-splitting parity comparable to exo on Apple Silicon.

This does not mean immediate parity with every `llama.cpp` feature. It means we invest until MLX reaches a replacement threshold.

## Non-Goals

- Keep `vMLX` as a long-term runtime
- Preserve Python-sidecar packaging as the strategic direction
- Pretend every MLX family will have tensor support on day one

## Architecture Call

`mesh-llm` remains the Rust control plane.

A dedicated MLX execution worker becomes the compute plane:

- process name: `mesh-llm-mlx`
- owns model load, tokenizer, prompt formatting, generation
- owns `mx.distributed` init and collectives
- owns pipeline/tensor execution details

`mesh-llm` continues to own:

- mesh discovery
- host election
- topology scoring
- API and console surfaces
- routing and failover
- lifecycle orchestration

## Parity Target

The practical target is parity with exo for MLX tensor splitting:

- `ring` and `jaccl` execution modes
- topology-aware placement
- pipeline parallel fallback
- family-specific tensor sharding rules
- overlapping prefill / distributed generation behavior

Relevant exo sources:

- [README](https://github.com/exo-explore/exo/blob/main/README.md)
- [placement_utils.py](https://github.com/exo-explore/exo/blob/main/src/exo/master/placement_utils.py)
- [auto_parallel.py](https://github.com/exo-explore/exo/blob/main/src/exo/worker/engines/mlx/auto_parallel.py)
- [generate.py](https://github.com/exo-explore/exo/blob/main/src/exo/worker/engines/mlx/generator/generate.py)

## Family Strategy

We should separate three support levels:

1. `single_host`
2. `pipeline_distributed`
3. `tensor_distributed`

“All families” should mean:

- broad single-host support through MLX/`mlx-lm`
- pipeline support for many decoder families
- tensor support only where we have explicit sharding strategies

Initial tensor shortlist:

- Llama
- Qwen
- Mistral-family if feasible
- DeepSeek after the base path is stable

Later:

- GLM variants
- GPT-OSS
- MiniMax
- NemotronH
- vision and multimodal families

## Milestones

### Phase 0: Remove vMLX

- delete `vMLX` runtime and bundle logic
- stop treating Python sidecars as the backend design
- keep only our own worker in the MLX path

### Phase 1: Single-Host MLX Baseline

- stabilize `--backend mlx` against the native worker
- clean model-family adapter layer
- reliable chat/completions behavior
- release-grade bundle includes `mesh-llm-mlx`

Exit criteria:

- local serving works through `:9337`
- remote mesh consumers can reach an MLX host through the current tunnel path
- smoke coverage for Llama and Qwen

### Phase 2: Worker Contract

Define the backend contract between `mesh-llm` and the MLX worker:

- startup args
- health check
- reported model id
- capability report
- error surface
- shutdown behavior

Add capability flags such as:

- `single_host`
- `pipeline`
- `tensor`
- `vision`
- `moe`
- `jaccl`

Exit criteria:

- no MLX-family conditionals in the `mesh-llm` request path
- worker can advertise what it actually supports

### Phase 3: Distributed Pipeline Backend

Build `mx.distributed` pipeline serving first.

Scope:

- `ring` first
- `jaccl` when topology supports it
- placement computed in Rust, execution in the worker
- layer-range assignment per node

Likely code ownership:

- [election.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/election.rs): placement decisions
- [launch.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/launch.rs): rank launch
- worker: layer loading and distributed forward pass

Borrow from exo:

- pipeline shard metadata shape
- pipeline prefill scheduling ideas
- communication-layer wrapping pattern

Exit criteria:

- multi-node MLX pipeline serving works over the mesh
- unsupported tensor families can still run distributed via pipeline

### Phase 4: Tensor Parallel Backend

Add tensor sharding family by family.

Borrow from exo selectively:

- family-specific sharding strategies
- head-count adjustment logic
- sharded/all-gather layer patterns
- model-specific exceptions

Do not port exo wholesale. Port only the model-specific graph surgery we need.

Exit criteria:

- Llama tensor path works across multiple Macs
- Qwen tensor path works across multiple Macs
- `ring` path works
- `jaccl` path works when topology qualifies

### Phase 5: Advanced Feature Parity

- MoE support
- better cache/prefill behavior
- throughput tuning
- recovery after worker/rank failure
- vision family support
- richer placement heuristics

## Backend Selection Policy

Near term:

- `llama` remains the production default
- `mlx` is the strategic backend under active buildout

Default-switch criteria:

- stable single-host path
- stable distributed pipeline path
- tensor parity for at least Llama and Qwen
- bundle/install story is release-grade
- acceptance tests pass on real multi-Mac clusters

## Packaging Direction

Do not optimize for “single binary” right now.

Optimize for:

- one release artifact
- `mesh-llm`
- `rpc-server`
- `llama-server`
- `mesh-llm-mlx`

Once MLX becomes strong enough, reassess whether the worker should remain separate or fold into a tighter packaging model.

## Code Changes To Expect

Core:

- [main.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/main.rs)
- [launch.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/launch.rs)
- [election.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/election.rs)
- [proxy.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/proxy.rs)

Worker:

- [mesh-llm/src/plugins/mlx-native/src/main.rs](/Users/jdumay/.codex/worktrees/3d01/mesh-llm/mesh-llm/src/plugins/mlx-native/src/main.rs)

Potential refactor:

- keep the worker as a separate binary, `mesh-llm-mlx`

## Immediate Next Steps

1. Make the current native worker path the only MLX path in the tree.
2. Refactor the worker into explicit family adapters instead of inline family checks.
3. Define and document worker capabilities returned to `mesh-llm`.
4. Implement distributed pipeline execution with `mx.distributed`.
5. Port Llama and Qwen tensor sharding from exo-style strategies.
