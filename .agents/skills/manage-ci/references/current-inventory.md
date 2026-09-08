# MeshLLM CI inventory

This file records checked-in CI facts and selected controlled probe evidence.
It is not a complete historical run log or live GitHub/Depot administration.
Read it with `../SKILL.md` and `ci/ci.md` before editing CI.

## Entry workflows

| Workflow | Trigger | Ownership |
| --- | --- | --- |
| `pr_quality.yml` (`PR · Quality`) | PR lifecycle | Canonical PR planning plus the protected reusable Quality lane |
| `pr_website.yml` (`PR · Website`) | PR lifecycle | Canonical PR planning plus the protected reusable Website lane |
| `pr_linux.yml` (`PR · Linux`) | PR lifecycle | Canonical PR planning plus the protected reusable Linux lane |
| `pr_macos.yml` (`PR · macOS`) | PR lifecycle | Canonical PR planning plus the protected reusable macOS lane |
| `pr_windows.yml` (`PR · Windows`) | PR lifecycle | Canonical PR planning plus the protected reusable Windows lane |
| `pr-cancel-sibling-runs.yml` (`PR · Cancel sibling lanes`) | protected `workflow_run` on `PR · Quality` entering progress | No-PR-checkout monitor that cancels other exact-revision PR validation lanes after the first definitive job failure |
| `main_quality.yml` (`Main · Quality`) | push to `main` | Exhaustive main planning plus the same-commit reusable Quality lane |
| `main_website.yml` (`Main · Website`) | push to `main` | Exhaustive main planning plus the same-commit reusable Website lane |
| `main_linux.yml` (`Main · Linux`) | push to `main` | Exhaustive main planning plus the same-commit reusable Linux lane |
| `main_macos.yml` (`Main · macOS`) | push to `main` | Exhaustive main planning plus the same-commit reusable macOS lane |
| `main_windows.yml` (`Main · Windows`) | push to `main` | Exhaustive main planning plus the same-commit reusable Windows lane |
| `ci.yml` | `workflow_call` only | Temporary inert shim for the former main ingress filename; pending protected-main runner-contract update; no push trigger or dispatch |
| `ci-control.yml` (`CI · Manual Full`) | dispatch on default branch | Explicit operator-only full plan, bounded lane dispatch and correlated diagnostic checks |
| `release.yml` | release tags, dispatch | Canonical version synchronization, release-only signing, assets and publication |
| `website-pages.yml` | main website paths, dispatch | Public website deployment |
| `pr_cleanup.yml` | PR close, dispatch | Positively matched cleanup only |
| `pr_auto_assign.yml` | PR lifecycle | Metadata only |
| `cache-warm-sccache.yml` (`Cache · Trusted sccache seed`) | successful Main Quality, dispatch | Sole bounded Linux compiler-seed publisher on GitHub-hosted infrastructure |

Other scheduled, deployment, Docker, package, canary and cache-warming
workflows are independent of required PR readiness.
`nightly-competitive-benchmark.yml` is an opt-in daily or explicit manual
trusted-main benchmark on the persistent Linux `white` runner. Placement uses
the fixed `[self-hosted, Linux, X64, cuda]` label set and then fails closed
unless `RUNNER_NAME` is exactly `white`. It has read-only permissions, never
accepts a ref or runner label, installs no tools, consumes pre-baked inputs
selected by `MESH_NIGHTLY_COMPETITIVE_*` repository variables, and retains
complete or partial benchmark/report evidence for 30 days. The pre-baked
Hugging Face CLI is selected explicitly by
`MESH_NIGHTLY_COMPETITIVE_HF_CLI`; it is required only when history is enabled.
The workflow captures stable GPU/driver identity plus observed clocks and
temperature.
When `MESH_PERFORMANCE_HISTORY_ENABLED=1`, it requires the dataset repo
variable and write-token secret, downloads prior immutable JSONL shards,
verifies the checked-in schema, emits an exact-cohort regression report, and
appends one source/run-addressed shard. Thresholds and promotion remain
report-only reviewed decisions.
`nightly-stability.yml` calls the fixed GitHub-hosted
`nightly-stability-run.yml`; the reusable run executes both the general
stability harness and the existing KV tool-loop/prefix-reuse harness, preserves
both evidence sets, and fails after both have had a chance to run.
`nightly-kv-coverage.yml` is a separate trusted-`main` GitHub-hosted schedule
for a much larger deterministic radix/blob ownership corpus. It uses the
pinned `public cpu` image, has no secrets, records exact seed/step budgets and
source SHA, and uploads the reproducible failure log.
`llama-upstream-canary.yml` runs only on its daily schedule or an explicit
manual dispatch; it is not ordinary push or PR CI. It executes trusted
default-branch content only on the persistent self-hosted `family-certify`
runner group (tools come from the runner image; no GitHub Actions model
caching). Before native compilation,
`scripts/plan-family-battery.py` validates the versioned JSON family policy,
the mandatory three-lane contract for every certified profile, and every exact
artifact revision/file in the immutable local cache. It reads only GGUF
metadata headers, requires each artifact to have at least one metadata-bearing
shard, and requires every shard that carries `*.block_count` and
`*.embedding_length` to equal the planned runtime range and activation width
before compilation; Qwen4 experimental artifacts derive their wider boundary
from `hyper_connection.count * embedding_length`. It emits
deterministic bounded GitHub matrix shards; the current one-runner topology consumes one
selected-family shard while retaining the plan as evidence. On a changed llama.cpp pin,
the canary diffs the actual old and new upstream revisions inside the prepared llama.cpp
checkout. A change limited to model implementation files named by the generated-family
map selects their certified families plus fixed architecture sentinels. Any shared
upstream source, unmapped model source, or unavailable diff fails closed to the full
bump battery; non-bump runs retain their cadence-owned cohort. The runner's `.env` exports
`HF_CACHE` pointing at a pre-warmed HF cache that lives on the lab NFS models
volume and `HF_HUB_OFFLINE=1` (NFS offers no `flock`, so `hf` on the runner is
read-only; the cache is populated by a two-stage prewarm that downloads on
local disk and moves each repo to NFS). The workflow builds its four
certification binaries before the manifest lanes; the family battery builds
them once itself unless `--skip-build` is selected, in which case it verifies
that every binary already exists. A scheduled unchanged pin selects the four
`nightly` cache-mechanism sentinels (Qwen3 dense, Falcon-H1, Qwen3Next, and
Mamba). A changed pin selects `llama-bump`; a manual dispatch may set
`force_certify` to select `manual-full`. Both latter cadences retain the full
family battery. Before any certification starts, every selected GGUF is resolved
directly by the immutable snapshot SHA checked into
`ci/llama-canary/family-certified.json`. The runtime preflight records the
revisions, verifies all shard/tensor scans and declared runtime/MTP layer
counts/model bytes, disk
headroom and certification ports. Native MTP/NextN heads remain part of the
single target model; the battery never reopens that model as a separate draft.
Those rows require native draft sidebands in staged single-step and chain
correctness, where each proposed token is verified against the target. The
general `llama-spec-bench` target/draft benchmark remains available for explicit
two-model experiments outside the family battery. Per-lane
outcomes, immutable model manifests, summaries, model scans, preflight
evidence, and logs are uploaded for 14 days even when the battery fails. Stage
readiness uses a declared per-model override or a model-size-derived deadline,
each complete certification has
a portable process-group wall-clock limit, and the workflow's outer battery
ceiling is 12 hours. On a
patch-apply failure it hands the queue to a non-interactive `opencode` agent
(`CANARY_AGENT_MODEL`, default `zai-coding-plan/glm-5.3-flash`, overridable
via the `LLAMA_CANARY_AGENT_MODEL` repository variable) which rebases
`third_party/llama.cpp/patches`, runs the supported-families certification
battery (`scripts/skippy-family-battery.sh`), and opens or reuses the repair PR
on `llama-canary/patch-queue-fix`. The deterministic wrapper writes the sole
upstream selector, `third_party/llama.cpp/upstream.txt`, to the resolved repair
target, prepares through the checked-in `pinned`
selector, and verifies the prepared-upstream stamp before any repair branch is
published or certified. The same repair loop also runs when the queue applies
but a certification lane fails (`battery` mode). After each agent turn the
repair script itself runs the battery and, on failure, loops certify -> agent fix ->
recertify up to `CANARY_REPAIR_MAX_TURNS` (default 2) turns; the script only
succeeds when the wrapper's own battery run passes. Every outcome (battery
green, queue still broken, battery exhausted) posts a status comment on the
repair PR — creating the PR (or a fallback issue) itself if the agent did
not — and an agent turn writes the PR description (key upstream changes,
patch-queue evolution, risks) with a deterministic fallback. Repair pushes and
PR operations authenticate with the `CANARY_REPAIR_TOKEN` fine-grained PAT;
the canary job itself remains `contents: read`. Any repair outcome keeps the
canary run red: the certified fix must be merged from the repair PR before
trusted main can certify. The upstream pin commit to
`main` is gated on the battery passing and writes the sole upstream pin from
the validated SHA.

For a non-canary manual dispatch, `release.yml` runs the checked-in
`scripts/release-version.sh`, creates one linear release-source commit when the
tracked version surface changes, and fast-forwards `main` before any release
build starts. `just release` is a preflight and synchronous dispatcher for that
same workflow. A tag-push release is read-only with respect to `main` and is
accepted only when the tag is already reachable from `main` and applying the
same version script produces no tracked diff. Canary dispatches never update
`main` or publish. The publish job creates only the release-specific tag commit
for generated Swift/SDK resources and enables GitHub-generated release notes.
The comparison base is the highest stable `vMAJOR.MINOR.PATCH` tag below the
target; prerelease tags are excluded so RC and final notes use the same stable
baseline.

The five PR lifecycle rows and five main push rows above are the complete
allowed routine validation entry sets. The protected sibling monitor is
metadata/control infrastructure, not a sixth validation entrypoint or required
check. Their separation and direct GitHub log visibility are contractual, not
a presentation preference. The retained `ci.yml` is reusable-only migration
scaffolding and must never regain event triggers or call the five lanes; remove
it after the protected-main runner-contract update is active.

## Reusable workflows and slices

| Workflow | Contract |
| --- | --- |
| `ci-quality-lane.yml` | Quality and runner/cache contract graph; reusable from PRs and dispatchable for main/manual |
| `ci-website-lane.yml` | Console and website graph; reusable from PRs and dispatchable for main/manual |
| `ci-linux-lane.yml` | Linux host/runtime/product/Rust/SDK/smoke graph with one platform-local UI producer |
| `ci-macos-lane.yml` | macOS host/runtime/product/platform/Swift/Metal graph with one platform-local UI producer |
| `ci-windows-lane.yml` | Windows host/runtime/product/platform graph with one platform-local UI producer |
| `ci-quality-slice.yml` | Contracts, format, Clippy and generated CLI inventory freshness; additive protected authority sentinel |
| `ci-web-slice.yml` | Console quality, console Playwright E2E, public website build, and CLI explorer browser validation |
| `ci-ui-artifact-slice.yml` | Immutable console distribution producer; release callers prepare one source/version-bound UI with complete file checksums, shared by all hosts and SDK resources |
| `static-abi-artifact.yml` | Typed static llama ABI producer with internal runner policy and an exact toolchain-epoch output |
| `ci-rust-tests-slice.yml` | Typed deterministic Cargo test batches that verify the producer-owned static ABI toolchain epoch and a pinned, digest-verified Skippy correctness fixture; related PR changes additionally compile one asserted, fully qualified runtime test and smoke an immutable SmolLM2 SafeTensors checkpoint through the complete Mesh config/resolver/server/native path to sampled prefill and decode with every supported load-time quantization |
| `ci-{linux,macos,windows}-host-slice.yml` | Platform-pure neutral host producers; no empty cross-platform jobs |
| `ci-{linux,macos,windows}-runtime-slice.yml` | Platform-pure native runtime producers |
| `ci-{linux,macos,windows}-product-slice.yml` | Platform-pure composition-only product consumers |
| `ci-platform-checks-slice.yml` | macOS portable/unit, Windows portable, and Windows log-store privacy ACL checks |
| `ci-linux-product-smoke-slice.yml`, `ci-macos-product-smoke-slice.yml` | Platform-local callers of the typed CPU/CUDA/Vulkan (`gpu-nvidia` self-hosted), conditional ROCm (`gpu-amd`), and Metal product-integration suite plus model-download. The suite stages the registry-pinned SmolLM2 Q8 and IBM Granite 4.0 H Q4 pair once, runs dense standalone/SDK/restart, then dense passive-client split routing and strict recurrent `KvRecurrent` validation. Each split phase persists strict-whitelist seed/worker node, mesh, and peer identity plus stage/model snapshots, then atomically reconciles exact two-observer, topology/run/model/package/manifest, two-stage contiguous-cut and bind-address, ready-status, and served-model agreement. A capped five-minute wall-clock deadline with parallel, bounded endpoint capture finalizes failure evidence before workflow cancellation; the status projection excludes invite tokens, nested fields, and unrelated paths. Product reconciliation independently verifies both evidence files, records their paths and SHA-256 digests in `phase-results.json`, rejects missing or modified evidence, and uploads every JSON snapshot/evidence file with logs on success or failure. ROCm skips unless `MESH_ROCM_INFERENCE_RUNNER_ENABLED` is exactly `true`; accelerator product-integration rows remain outside the checked plan pending live qualification. |
| `ci-linux-sdk-slice.yml`, `ci-macos-sdk-slice.yml` | Platform-local Rust/Kotlin/Swift smoke consumers; SDK producers are independent top-level calls and each smoke receives the lane-local immutable UI artifact |
| `ci-runner-contract-slice.yml` | Provider/cache/plan trust and main runner-image checks |
| `native-sdk-artifact.yml` | Typed native SDK producer |
| `swift-sdk-artifact.yml` | Host-only/full XCFramework producer; full mode builds the seven Apple Rust targets as a bounded matrix (maximum four concurrent macOS runners) and joins their immutable libraries in one assembly job, while host-only remains a single producer. Trusted main remains `macos-15`, while eligible same-repository PRs follow the protected Depot macOS 15 gate |
| `smoke.yml` | Artifact-based inference/OpenAI/split smoke |
| `scripted-binary-smoke.yml` | Artifact-based scripted product smoke with optional typed model context-size and recurrent-model inputs; recurrent models restore and save through a dedicated trust-scoped cache before the smoke runs |
| `sdk-smoke.yml` | Artifact-based SDK consumers; all SDK rows consume the lane's immutable console UI artifact, while Rust smoke restores the main-seeded, target/profile/image/toolchain/recipe-bound Cargo/target cache through `Swatinem/rust-cache` |
| `hf-download-smoke.yml` | Hugging Face download smoke |

All workflow calls use typed, bounded semantic inputs. Credential-bearing smoke
workflows remain fixed to GitHub-hosted runners; the PR entrypoints pass no
repository secrets. The trusted main entrypoint may pass the optional
`HF_TOKEN` for public-fixture rate-limit resilience.

## Prebuilt runner-image containerization

`ci/runner-images.json` records the checked-in image references, semantic job
bindings, native epochs, compiler-seed identity and the separate SDK Rust
toolchain identity. `scripts/runner-image-identity.py check` compares those
values with every literal workflow image binding and the actual planner rows.
Its focused tests run through the existing `just ci-validate` discovery. This
catalog adds no planner authority and changes no image pins or cache keys.
Historical receipt, provenance and workload-coverage fields are explicitly
unknown. `diagnose` reports the runtime architecture guard mismatch separately;
matching image identity does not qualify the host seed for native runtime work.

Some CI jobs run inside a `container:` pinned to a digest from the
`mesh-llm-runner-images` repo instead of installing tooling per-run with
`actions/setup-*`. There is no separate sister repo: the `public web`
backend (baked Chromium/Playwright) lives on `mesh-llm-runner-images` main
alongside every other family, added by `17283ab` (#20, the `public web`
backend) and `5ea673b` (#21, the Playwright version assert). The GHCR
package name
`ghcr.io/mesh-llm/mesh-llm-cuda-runner` is legacy: it hosts every backend
family (`public cpu`, `public cuda`, `public rocm`, `public vulkan`,
`public web`, `self-hosted`), not only CUDA. Each image bakes
`cargo cmake docker git jq just lld node ninja npm pnpm python rustc sccache`
(asserted by `verify-runner-image`, see below) plus a Python venv on `PATH`
(`VIRTUAL_ENV=/opt/mesh-llm/venv`), pinned pnpm/node (`PNPM_HOME`,
`CARGO_HOME`, `RUSTUP_HOME` baked as ENV so they resolve the same regardless
of the container's `HOME`), and, for the `public` stage only, runs as
**root** (`USER root`, never dropped back) rather than `runner` --
`self-hosted` is the only stage that ends `USER runner`.

Reusable slices/workflows with a `container:` job, and what backs it:

| Workflow | Job(s) | Image family |
| --- | --- | --- |
| `ci-{linux}-host-slice.yml`, `ci-linux-runtime-slice.yml`, `ci-linux-product-slice.yml`, `ci-rust-tests-slice.yml`, `ci-quality-slice.yml` (Clippy batches) | matrix-selected | `public cpu` (pre-existing, predates this containerization pass) |
| `native-sdk-artifact.yml`, `node-sdk-addon-artifact.yml`, `static-abi-artifact.yml`, `swift-sdk-artifact.yml` | producer job | `public cpu` (pre-existing) |
| `hf-download-smoke.yml`, `scripted-binary-smoke.yml` | their single job | `public cpu`, sha256:8d93de6b... -- unconditional, no bare-metal row |
| `smoke.yml` | `smoke_tests` | `public cpu` when `inputs.runner != 'gpu-nvidia'`, else uncontainerized (see opt-out below) |
| `sdk-smoke.yml` | its job | `public cpu` when `inputs.sdk_kind != 'swift'`, else uncontainerized |
| `ci-ui-artifact-slice.yml` | `ui_artifact` | `public web`, sha256:1c73f0f2... |
| `ci-web-slice.yml` | `ui_quality`, `ui_e2e`, `website` | `public web` |
| `website-pages.yml` | `build` | `public web` |
| `nightly-stability-run.yml` | `stability` | `public web` (bakes node/pnpm the CLI-smoke step needs) |
| `nightly-kv-coverage.yml` | `ownership-state-machines` | `public cpu`, sha256:8d93de6b... |
| `release.yml` (CUDA/ROCm/Vulkan compiler rows) | per-backend `public` digests | Native compiler/toolkit images remain backend-specific; amd64 CUDA preserves its intentional ARC self-hosted placement |
| `release.yml` (Linux CUDA/ROCm/Vulkan composition rows) | `public cpu`, sha256:8d93de6b... | Verify, compose, readiness-test and archive exact producer bytes without downloading a compiler toolkit |

`public cpu` and `public web` are separate image builds (the latter adds
`PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright`,
`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`, and a stamped
`/etc/mesh-runner-playwright-version`); do not assume one digest covers both.

### `image: ''` opt-out rows

`smoke.yml`'s `gpu-nvidia` row (the approved uncredentialed self-hosted CUDA
smoke exception) and `sdk-smoke.yml`'s `swift` row (host-only macOS SDK
build, `macos-15`, never container-capable) opt out per-run with
`image: ''` rather than being a separate job, so the rest of the job body
(steps, `if: job.container.id == ''` gates) stays shared. This is proven to
actually opt a job out of containerization by two runs on the temporary
branch-head harness used to validate #1380 (run 32349670919, jobs
96370649138 `CUDA inference smoke` and 96375145155 `swift SDK Smoke`): no
`Initialize containers` log group, no `docker create`, and the gated
`actions/setup-python`/`pnpm/action-setup` steps ran. There is no other
empty-image job anywhere in this repo's workflow history.

The ternary that selects `image: ''` must put the **non-empty** value in the
`&&` branch: `cond && url || ''`, never `cond && '' || url`. GitHub Actions
expressions are JS-style short-circuit and `''` is falsy, so
`cond && '' || url` always evaluates to `url` regardless of `cond` -- the
opt-out branch becomes unreachable. `scripts/tests/test_ci_workflow_ternary_contract.py`
fails any `${{ }}` ternary whose `&&` branch is a falsy literal (`''`, `""`,
`0`, `false`) across every workflow; it exists specifically because this bug
class is invisible to `actionlint`.

### `job.container.id == ''` gating

When a job has both a containerized and a bare-metal row, setup actions needed
only by the bare-metal row are gated `if: job.container.id == ''` rather than
allowed to shadow tools from the image. `sdk-smoke.yml`'s bare-metal Swift row
consumes the immutable UI artifact with `--skip-build`, so it needs neither
Node nor pnpm; those setup actions and their unused package cache are absent.
Its smoke script temporarily creates the legacy protected-main workflow's
computed pnpm store when that older workflow installed pnpm, preventing the
setup-node post-job cache hook from failing before this workflow change lands
on `main`. **Deliberate exception: `actions/setup-java` in
`sdk-smoke.yml` is never gated.** `verify-runner-image`'s asserted tool list
has no JDK, so the image provides nothing for it to shadow; gating it would
break the Kotlin SDK smoke on the containerized row instead of protecting it.
Jobs with no bare-metal row at all (the `ci-web-slice.yml` / `website-pages.yml`
/ `ci-ui-artifact-slice.yml` / `nightly-stability-run.yml` set) delete the
now-redundant setup actions outright instead of gating them -- there is
nothing for the `if:` to select between.

`npm install --global openai` in `smoke.yml` is **not** gated on
`job.container.id`, even though `install-core-tools.sh:83` bakes an
exact-pinned `openai` into the image on `mesh-llm-runner-images` main. The
`public cpu` digest pinned in `ci/slices.yml` predates that bake (see
"A pinned digest is a frozen artifact" below), so the containerized row needs
the install too, and the step runs unconditionally for both it and the
bare-metal `gpu-nvidia` row. Re-gate it only once the CPU digest is promoted
past `mesh-llm-runner-images` #20 and that is confirmed from a green run.

Both call sites install `openai@7.5.0`, not floating `openai`. The step runs
with the full job environment (`HF_TOKEN` included) and npm lifecycle scripts
inherit it, so an unreviewed upstream release must not be able to execute
there; `zizmor`'s `adhoc-packages` rule flags the floating form. The version
deliberately tracks the image's own `ARG OPENAI_NPM_VERSION`
(`mesh-llm-runner-images` `Dockerfile:25`) so that re-gating the step later is
a no-op rather than a version swap -- bump both sides together.

### Container jobs default `run:` to `sh`, not `bash`

Jobs with a `container:` block resolve the default `run:` shell to
`sh -e {0}`, not `bash -e {0}` (bare-metal Linux/macOS runners default to
bash; this only changes inside a container). Composite actions are
unaffected -- they declare their own shell. Any `run:` step in a
containerized job that uses a bashism (`<<<`, `set -o pipefail`, `[[`,
array assignment, `${v//}`/`${v^^}`/`${v,,}`, `&>`, `source`, `+=(`, ...)
must declare `shell: bash` explicitly or it fails at runtime with a
`dash`/`sh` syntax error that `actionlint` cannot catch -- its shellcheck
integration assumes bash. Two sites hit this in the same PR:
`ci-web-slice.yml`'s `ui_e2e` preflight (`<<<`) and
`website-pages.yml`'s `Stage Pages artifact` (`set -euo pipefail`); both now
declare `shell: bash`.

`$(( ))` arithmetic expansion is **not** on that list and must not be added.
It is POSIX (Shell Command Language 2.6.4) and `dash` evaluates it correctly;
flagging it would reject valid `sh` steps and force a spurious `shell: bash`.
`scripts/tests/test_ci_workflow_container_shell_contract.py` carries the
pattern list and an inline note saying so.

### Reusable-workflow permission chain

A called reusable workflow may not request a permission scope its caller job
does not grant; GitHub rejects at run creation with a **zero-job
`startup_failure`** -- no jobs, no logs, no check run on the commit, and
`actionlint` cannot see it. Containerizing surfaced this because
`packages: read` (needed to pull the private GHCR runner images) has to be
granted at *every* hop, and
`ci-linux-product-smoke-slice.yml` / `ci-macos-product-smoke-slice.yml` sat at
`contents: read` between granted parents and requesting children.
`scripts/tests/test_ci_workflow_permission_contract.py` walks every local
`uses: ./.github/workflows/X.yml` edge and asserts the caller's effective
permissions (job-level, else workflow-level) cover what `X.yml` requests.

Two properties make that assertion real rather than decorative, and both were
absent when the test was first written:

1. **The callee's requested set is the workflow-level block merged with every
   explicit job-level block.** Five reusable workflows here
   (`native-sdk-artifact.yml`, `node-sdk-addon-artifact.yml`, `sdk-smoke.yml`,
   `static-abi-artifact.yml`, `swift-sdk-artifact.yml`) declare permissions
   only at job level, so reading the workflow-level block alone returns `None`
   for them and skips their caller edges entirely -- including the
   `packages: read` edges this test exists to cover.
2. **Scope levels are compared, not scope names.** `contents: read` does not
   satisfy a callee's `contents: write`; GitHub rejects that downgrade at run
   creation exactly like a missing scope. The comparison ranks
   `none < read < write`, and where a scope is declared in more than one block
   the strictest level wins. A name-only set comparison silently passes the
   downgrade.

3. **`read-all`/`write-all` are modelled on the granting side, not skipped.**
   As a *grant* they are perfectly enumerable -- `write-all` satisfies any
   request, `read-all` satisfies a `read` request but not a `write` one -- so
   returning "unknown" and skipping the edge would hide the same
   run-creation failure. As a *request* they stay opaque: a callee asking
   `write-all` names no scopes to hold its caller to, and asserting there
   would be invention rather than checking.
4. **Both workflow extensions are read.** Globbing `*.yml` alone would skip a
   `*.yaml` callee entirely; the repo has none today, which is exactly when
   that gap is cheapest to close.

None of these were breakages -- the repo satisfies the contract at every edge
under the strict check, and it has no `.yaml` workflows or all-scope grants at
all. That is the point: a permission test that under-reads its inputs reports
green for edges it never examined, and each of these was found by tightening
the test rather than by anything failing.

### `verify-runner-image` preflight

Containerized jobs run `verify-runner-image <environment> <backend> ...`
(positional args: environment, backend, mesh-llm revision, CUDA series, ROCm
version, runner-images revision, and -- `public web` only -- expected
Playwright version, added in `mesh-llm-runner-images`#21) before doing real
work, asserting `/etc/mesh-runner-*` files match what the job expects rather
than trusting the digest pin alone. `ci-web-slice.yml`'s `ui_e2e` job
resolves the installed `@playwright/test` version with
`pnpm exec playwright --version | head -n1 | awk '{print $NF}'` (guarded by a
`^[0-9]+\.[0-9]+\.[0-9]+$` shape assertion -- `playwright --version` can share
stdout with an npm warning) and passes it as the seventh argument; a mismatch
against the image's own build-time `playwright --version` fails fast instead
of surfacing as a confusing Playwright/Chromium error deep in the E2E run.

`crates/mesh-llm-ui/package.json`'s `@playwright/test` and
`mesh-llm-runner-images`' `config/playwright-pin.txt` are now a matched pair
(both `1.62.1` as of 2026-08-20; re-check the two sources rather than
trusting this line). Bumping the mesh-llm side alone fails `ui_e2e` on
**every** PR at this preflight, not just locally. The bump is a four-step
cross-repo sequence, in order: bump `config/playwright-pin.txt` in
`mesh-llm-runner-images`, rebuild and promote the `public web` image, re-pin
the new digest in `ci-web-slice.yml` (and `ci-ui-artifact-slice.yml` /
`website-pages.yml` / `nightly-stability-run.yml`, which share it), then
bump `@playwright/test` in `crates/mesh-llm-ui/package.json`.

### `setup-macos-lld` composite

`.github/actions/setup-macos-lld` replaces per-callsite
`brew install lld` plus a hand-rolled `PATH`/`RUSTFLAGS` export with one
composite: install lld via brew, resolve `$(brew --prefix lld)/bin`, link a
real `edition = "2024"` probe binary with `-Clink-arg=-fuse-ld=lld`, then
export `CARGO_ENCODED_RUSTFLAGS=-Clink-arg=-fuse-ld=lld` and the resolved bin
directory. `CARGO_ENCODED_RUSTFLAGS` **replaces** any
`target.<triple>.rustflags` from a checked-in `.cargo/config.toml` rather
than merging with it -- confirmed safe here because every call site is
macOS-gated and no call site touches an `android` target (the repo's
`.cargo/config.toml` android entries carry a `-Wl,-z,max-page-size=16384`
flag that would otherwise silently stop applying). Seven call sites:
`ci-platform-checks-slice.yml`, `ci-macos-host-slice.yml`,
`swift-sdk-artifact.yml`, `native-sdk-artifact.yml`,
`node-sdk-addon-artifact.yml`, and two in `release.yml`. Only the first
three were reachable by the temporary branch-head harness used to validate
PR #1380 (no macOS row in
`native-sdk-artifact.yml`/`node-sdk-addon-artifact.yml` ran there, and
`release.yml` only runs on an actual release cut) -- the other four are
statically cleared (macOS-gated, no android target on any of them) rather
than proven by a real run. `node-sdk-addon-artifact.yml`'s own
`Validate macOS x64 cross-linker` step is a deliberate near-duplicate of the
composite's probe, not dead code: it passes `--target x86_64-apple-darwin`
where the composite only probes the host target.

### A pinned digest is a frozen artifact

`mesh-llm-runner-images` HEAD says nothing about what is inside the digest a
workflow pins -- the `public cpu` digest pinned in `ci/slices.yml` was built
2026-07-22 and does not contain changes merged to that repo afterwards
(the `smoke.yml` openai bake landed a week later, in #20). Before deleting or
gating a dependency install on the grounds that "the image bakes it,"
confirm the capability exists **in the pinned digest**, and confirm it from a
green run of the job that needs it. `verify-runner-image`'s JSON is the
cheap probe: `mesh_llm_revision` dates the build, and missing keys (added to
the asserted object in later `mesh-llm-runner-images` commits) date the
baked verify script itself.

### Digest promotion

`build-and-push.yml` (in `mesh-llm-runner-images`) runs `stage_families` for
both `operation=stage` and `operation=promote`; `promote_versioned` reads the
candidate descriptor artifact from that **same run**, not from an earlier
stage run. A `promote` dispatch therefore re-stages and promotes its own
build. Read the digest to pin from the promote job's own `digest=` output
(e.g. `promoted ghcr.io/... -> sha256:...` in its log) -- never carry forward
a digest observed from an earlier stage-only run, even one at the same
source commit.

## Planner contract

- `scripts/plan-ci.py` is the only routing implementation.
- `ci/ownership.yml` maps paths and direct crates to semantic domains; unknown
  paths fail closed.
- `ci/slices.yml` defines profiles, slice dependencies, rows, runner roles,
  cache modes and worker budgets.
- `ci/ci-plan.schema.json` versions the machine-readable output.
- `compute-changes` supplies the complete event diff and affected Cargo
  closure; the planner owns signals and final matrix selection.
- Each `pr_*.yml` workflow checks out the default branch for canonical planning,
  projects one bounded lane, and calls its matching default-branch lane as a
  nested reusable workflow. The protected planner action extracts only
  `ci/ownership.yml` and `ci/slices.yml` from the validated immutable PR source
  SHA into a unique runner-temp directory. It treats those manifests as data;
  both source catalogs must match the protected catalogs before use, so PRs
  cannot alter ownership or expand protected matrix or worker ceilings.
  planner code, Cargo workspace discovery, and affected-crate operations remain
  rooted in the protected checkout. Missing or non-regular source manifests
  fail planning. Jobs and logs remain attached to five focused PR runs rather
  than one monolithic graph.
- Catalog evolution is a sequenced maintainer merge. A branch that needs a new
  `ci/ownership.yml` or `ci/slices.yml` entry cannot pass its own Plan gate,
  because the byte-identical compare is the boundary keeping PR-controlled
  routing out of the protected planner. Land a catalog-only commit on the
  default branch first, then rebase the dependent branch onto it. Do not relax
  the compare, add a label-gated bypass, or special-case catalog paths.
- Each `main_*.yml` workflow plans the exhaustive main profile at the pushed
  SHA, projects one bounded lane, and calls its matching same-commit lane as a
  nested reusable workflow. Routine main jobs and logs therefore remain
  attached to five focused main runs.
- `ci-control.yml` is manual-full only. It calls the planner once and dispatches
  bounded JSON lane projections as native inputs for explicit operator
  diagnostics; it cannot receive a push, PR, or workflow-run event.

Main/manual profiles enumerate every workspace crate exactly once and all
supported product/SDK rows. Ready PR profiles select affected or directly owned
rows from that same catalog. Draft PR profiles select no build rows unless the
changed paths require the documented CI-control or runner-infrastructure
fail-open policy.

## Artifact and cache owners

- `restore-release-ui` / `scripts/ui-distribution.py`: verify the shared release
  console's source SHA, version, complete file hashes and built JavaScript entry
  before platform-specific Rust compilation or SDK resource packaging. The
  UI producer's version step trusts only `GITHUB_WORKSPACE` in the container's
  active Git configuration before verifying the exact checkout SHA.
  The `prepared-release-ui-*` artifact retains for 90 days and is excluded from the
  GitHub release asset glob. Swift release resource assembly skips pnpm and the
  console build when this artifact is supplied; ordinary PR/main behavior is
  unchanged.

- `prepare-host-input` / `prepare-windows-host-input`: neutral host bytes,
  import report and checksum.
- `prepare-native-runtime-input`: one verified native runtime archive and
  manifest. Non-Windows artifacts include the checksum-bound
  `skippy-model-package` tool used by split-serving consumers to prepare
  package-v2 fixtures; Windows artifacts remain DLL-only until the producer has
  a reliable import-library path for the tool.
- `prepare-static-abi-input`: portable static ABI archive.
- `compose-product-input`: exact host/runtime verification and composition.
- `ci/model-artifacts/registry.json`: canonical immutable model identities,
  integrity, family capability tags, and allowed suite/cadence membership.
  `scripts/generate-test-model-manifests.py` owns the family battery and
  suite-specific projections; CI contract tests reject stale projections.
- `restore-smoke-inputs`: product/model extraction for consumers. Model
  restores resolve generated suite manifests, use exact digest-bearing cache
  keys, and stream-verify size and SHA-256 before use.
- `select-ci-runners`: provider labels, cache permissions, and the
  provider-derived `allow_native_github_cache` / `allow_depot_remote_cache`
  outputs. Depot selections disable both cache paths by default. During the
  bounded approved exception, eligible same-repository PR and trusted-main
  Depot jobs enable the GitHub Actions cache API while direct Depot remote
  cache remains disabled. Hosted PR, release, and cache-warmer selections
  retain native GitHub cache behavior.
- `configure-sccache-gha`: event/provider-derived compiler-cache setup.
- `restore-sccache-seed`: exact-key restore of the trusted 2 GiB Linux seed;
  central runner policy permits it only for GitHub-hosted selections, and
  runtime rows must match the seed's container image and toolchain epoch.
- `capture-sccache-stats`: machine-readable cache evidence. Warm consumers with
  a positive floor fail when no cache requests are observable; the zero-floor
  SafeTensors observation remains non-failing and emits a wiring warning.

Rust-test batches that contain `skippy-runtime` or `skippy-model-package`
resolve the generated Skippy correctness manifest, then restore the pinned Qwen
fixture from one exact GitHub Actions cache key containing its file SHA-256 and
`.github/cache-version.txt`. Every use is verified against the pinned size and
digest before tests. The SafeTensors smoke resolves its repository, revision,
complete file set, per-file integrity, and quantization sweep from the same
registry. Cache publication is limited
to the exhaustive trusted-main batch containing `skippy-runtime`; PR jobs are
restore-only and jobs for which central runner policy denies native GitHub
cache access download and verify the immutable revision without publishing.

`scripts/collect-ci-metrics.py` is the read-only timing evidence collector. Its
schema-v3 report keeps workflow wall/queue, job runner queue, measured
dependency wait, job execution, runner-minutes, cancelled runner-minutes and
peak workers separate. It groups observations by provider, operating system,
architecture, semantic runner role and Depot size, and emits deterministic
queue/capacity heuristics plus an optional provider-cohort comparison. Raw
inputs and dated reports belong under `/tmp` or a tracking issue/artifact, not
under `ci/` or this inventory.

Artifacts are correctness boundaries; caches only accelerate regeneration.
PR artifacts generally retain for one day. Fork lanes cannot publish shared
trusted-main caches. Same-repository PRs normally use GitHub's ref-scoped cache;
eligible PR jobs may temporarily use Depot's shared cross-branch
namespace under `ci/DEPOT_PR_RISK_EXCEPTION.md`. That namespace is treated as
untrusted input, not an authority or correctness boundary. Linux Clippy,
Rust-test, host, and runtime jobs restore one bounded trusted sccache seed
instead of per-row Cargo target archives. The protected warmer's
`just ci-sccache-seed-build` recipe covers the dominant `mesh-llm` Clippy graph
and the isolated `mesh-llm-cli` test graph used by the Rust-test matrix. Depot
selections cannot restore that
seed through their cross-trust cache proxy. Its exact key fingerprints the
warmer image and toolchain epoch, so mismatched native-runtime rows are cold and
do not restore it. These four high-fanout families disable per-object GHA
publication on every provider. Exact Linux static ABI, Swift ABI, macOS Metal unit ABI,
and Windows native ABI caches may publish into GitHub's isolated PR merge-ref
scope for same-PR reruns. UI installs (`ui_quality`, `ui_e2e`, `ui_artifact`) point pnpm at the runner
image's baked store instead of an Actions cache — there is no shared pnpm
key or publisher to race. Trusted main owns shared publication.

PR Rust-test, host, native-runtime, product, platform-check, and full Swift
target matrices receive `fail_fast: true`; main/manual and release pass
`false`. Quality matrices remain
non-fail-fast and failed producers suppress impossible consumers through
`needs`. One protected, default-branch `workflow_run` monitor starts with
`PR · Quality`, polls the five exact-PR/exact-SHA validation runs, preserves the
run containing the first definitive failed job, and cancels its queued or
in-progress siblings. It checks out only the default branch, owns the sole
`actions: write` token for this operation, and never targets main, manual,
release, deployment, cleanup, cache-warming, another PR event epoch, or a newer
revision. PR-controlled workflows and executor jobs retain no Actions-write
permission.

## Providers and variables

GitHub-hosted labels are `ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15`, and
`windows-2022`. Depot labels are selected only by `select-ci-runners`; no
workflow accepts a raw provider label. Trusted main Linux requires
`DEPOT_RUNNERS_ENABLED=true`. Eligible same-repository PR jobs may use the
time-bounded exception when `DEPOT_PR_RUNNERS_ENABLED=true`; it expires on
2026-09-14 UTC. Forks remain hosted. The intended permanent gate
may cover eligible build/test rows across Linux, Depot macOS 15 and Windows
2022 when equivalent images/architectures exist; planning/required summaries,
credential-bearing smokes, `gpu-nvidia` hardware and uncertified Intel macOS
rows remain exceptions. The documented `gpu-nvidia` ephemeral scale set is
the sole currently verified uncredentialed, hardware-qualified same-repository
PR exception. The typed ROCm job remains skipped unless
`MESH_ROCM_INFERENCE_RUNNER_ENABLED` explicitly enables the repository-scoped
`gpu-amd` role.

The permanent Depot PR gate is documented in `ci/DEPOT_MIGRATION.md`; the
accepted temporary findings and risks are in
`ci/DEPOT_PR_RISK_EXCEPTION.md`. Permanent activation requires cache
isolation, no PR cache/registry tokens, exact protected workflow refs,
ephemeral runners, a successful sentinel, and a tested GitHub rollback.

External administrative posture is now verified as follows: automatic Depot
Cache connectivity is disabled, automatic Registry Actions authentication is
disabled, and the Depot runner group is restricted to `Mesh-LLM/mesh-llm` and
the exact protected workflow refs. The repository token cannot independently
inspect organization runner-group settings through the API (403), so these
remain external facts rather than checked-in evidence. The two switches remove
Depot's direct `DEPOT_CACHE_TOKEN`/WebDAV build-tool preconfiguration and
Registry Actions authentication on fresh runners; they do not document or
enforce a per-connection/job/ref disable or ACL for the GitHub Actions cache
proxy/runtime-token path. The controlled
trusted-main seed [run 31816775585](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816775585)
succeeded at `main` commit `9e977e246`; the same-repository PR authority
sentinel [run 31816869128 / job 94821057215](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816869128/job/94821057215)
read and exactly validated the trusted seed, saved/cleared/restored and
exactly validated the poison, then failed its intended seed-isolation gate;
the enclosing PR run was later cancelled during cleanup. Trusted-main verify
[run 31817111471 / job 94821343605](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31817111471/job/94821343605)
restored and exactly validated that poison, then failed its intended expected-
miss gate. This proves unsafe repository-scoped cross-trust authority, so it is
not a successful isolation result. The bounded exception knowingly accepts
that risk for eligible same-repository PRs to gain CI
iteration speed; it is not permanent-isolation evidence. The exact-SHA
five-lane candidate, provider comparison, and identical-SHA hosted rollback
are recorded in `.omo/specs/depot-pr-rollout-evidence.md`; Quality and Linux
had favorable queue observations but remain unclassified because execution
was cache-confounded, Website had insufficient samples, and macOS/Windows hit
the capacity rollback threshold. Fork PR validation and namespace purge/expiry
confirmation remain pending. Fork PR validation remains hosted and is the
no-Depot-authority half of the sentinel acceptance evidence; only the exact
same-repository sentinel ref may exercise the diagnostic Depot job. All three
sentinel cache phases attest the
provider-injected `ACTIONS_CACHE_URL`/`ACTIONS_RESULTS_URL` structure before
invoking pinned `actions/cache` restore/save actions. The shell attestation
does not require ambient `ACTIONS_RUNTIME_TOKEN`: GitHub's
`NodeScriptActionHandler` injects that credential into the cache actions, while
the shell `ScriptHandler` does not. Successful full restore/save is the
credential/token proof. The non-loopback check includes all IPv4 `127/8`
and IPv4-mapped IPv6 loopback spellings.
The protected PR probe clears and fully restores its saved poison key, requires
a cache hit and exact marker bytes before the trusted-seed gate, and thereby
proves the same-job Node token/write path; main verify's poison miss remains
the cross-scope proof.

The provider contract required before permanent PR placement is enabled is a documented,
server-enforced per-connection/job/ref control for the GitHub Actions cache
path. It must either leave PR jobs on GitHub-native branch-scoped
`ACTIONS_CACHE_URL`/`ACTIONS_RESULTS_URL` and runtime-token semantics with no
Depot proxy or direct cache token, or issue a PR-isolated namespace/token whose
ACL permits reads and writes only within that PR, denying reads and writes from
trusted main/release and every other PR namespace, without exposing
`DEPOT_CACHE_TOKEN`.
Key prefixes, loopback proxies,
ephemeral runners and the org switches are not equivalent controls. A fresh
same-repository PR, fork PR and trusted-main seed/verify sentinel must prove
the selected behavior before the temporary exception is removed.
Bracketed IPv6 authorities use the fixed runner's Python 3.8+ stdlib
`ipaddress` classifier; parser absence/version/invalidity fails closed.
Attestation reports only value-free variable/reason classes and fails closed
on malformed or missing backend data.

Relevant repository variable names include `DEPOT_RUNNERS_ENABLED`,
`DEPOT_PR_RUNNERS_ENABLED` (global temporary exception gate),
`DEPOT_PR_CANARY_REF` (absent by default; one exact
`refs/pull/<number>/merge` ref only), `DEPOT_PR_SENTINEL_REF` (absent by
default; one exact same-repository merge ref used only by the protected
no-checkout authority diagnostic), and `DEPOT_PR_SENTINEL_ID` (absent by
default; exactly 32 lowercase hexadecimal characters when the diagnostic is
deliberately armed). The canary and sentinel variables are bounded selectors,
not cache-isolation proofs or replacements for the global PR gate. The normal
Quality runner policy continues to use `DEPOT_PR_CANARY_REF`; the sentinel
uses a separate selector output and cannot move the normal build jobs.
The eligible five-lane Depot graph disables every native GitHub cache consumer
when `allow_native_github_cache=false`. During the bounded exception eligible
same-repository PR and trusted-main Depot jobs set that output true for
cross-branch Depot Actions-cache reuse; direct Depot remote cache remains
false. This checked-in mode does not
prove the absence of ambient Depot/WebDAV authority, so the runtime sentinel
has recorded unsafe repository-scoped cross-trust authority and must be
redesigned and repeated successfully; no-secret/no-token, fork and provider-
parity canaries remain required. Other variables include `CUDA_VERSION`,
`VULKAN_SDK_VERSION`, smoke configuration variables, and release/deployment
variables. Secret values never belong in this inventory;
known names include `HF_TOKEN`, release-attestation keys, `CARGO_REGISTRY_TOKEN`
and deployment tokens.

## Live inspection

Use read-only commands when live state matters:

```bash
gh workflow list --all --repo Mesh-LLM/mesh-llm
gh run list --repo Mesh-LLM/mesh-llm --limit 30
gh variable list --repo Mesh-LLM/mesh-llm
gh api repos/Mesh-LLM/mesh-llm/rulesets
gh api orgs/Mesh-LLM/actions/runner-groups
```

Organization runner-group responses of `403` are unverified administrative
state, not proof that a restriction is absent.

### Proposed retained-cohort identity evidence

Runner-images PR #23 is pending; its retained exact-attempt admission flow is
not yet the producer's merged-main behavior. After that producer flow lands,
`scripts/runner-image-identity.py bind` can prepare offline catalog proposals
from maintainer-reviewed admission anchors. The exact cohort bytes are retained
under `ci/runner-image-evidence/<sha256>.json`; ordinary commands validate hashes
and consumer relationships without executing producer code or authenticating
GitHub provenance again. See `ci/ci.md` for the explicit trust boundary and CLI.
Current image references and historical null evidence remain unchanged.

The `product-smoke` catalog role covers both the legacy `smoke.yml` job and
the typed `product-integration-smoke.yml` job. The latter uses the same pinned
CPU image only for Linux CPU; accelerator and macOS paths retain their existing
container opt-outs. The inventory has 30 roles and 31 literal workflow bindings.
