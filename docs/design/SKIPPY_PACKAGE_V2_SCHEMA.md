# Skippy Model Package v2 Schema

Status: implementation contract for `feature/skippy_graph_filter_v2`.

The authoritative Rust types and validation rules live in
`crates/skippy-package-format`. Package creation, preflight, runtime loading,
offline conversion, and test fixtures must consume that crate rather than
declaring private copies of the manifest.

## Purpose

Package v2 describes the complete model inventory without assigning tensors to
runtime stages. A graph-derived `StageSlicePlan` selects exact tensor ids from
the catalog after the normal model graph is built in metadata-only mode.

The package format has no model-family staging policy, endpoint ownership, or
cut-specific artifacts. Per-layer GGUF files remain valid physical containers,
but their placement does not determine whether a stage owns a tensor.

## Top-Level Contract

`PackageManifest` contains:

- `schema_version`: exactly `2`;
- `package_id`: canonical `sha256:<lowercase-hex>` identity;
- source and model identities;
- complete graph-construction metadata;
- an artifact catalog;
- a tensor catalog;
- optional tokenizer, projector, and generation sidecars;
- native ABI and package-generator versions;
- creation time for provenance.

Unknown fields are rejected. Schema evolution therefore requires a deliberate
schema-version change rather than silently changing the meaning of an existing
package.

## Canonical Package Identity

The package id is computed by the shared crate as follows:

1. Clone the manifest and replace `package_id` with the empty string.
2. Sort source files by path, artifacts by id, tensors by id, and sidecars by
   `(kind, name, artifact_id)`.
3. Serialize the normalized manifest with the shared Rust schema.
4. Hash the serialized bytes with SHA-256 and prefix the lowercase digest with
   `sha256:`.

Validation recomputes this identity. Changing metadata, an artifact digest, a
tensor record, ABI requirements, or generator provenance changes the package
identity. Enumeration order alone does not.

## Artifact Catalog

Each artifact has a stable id, normalized relative package path, total byte
size, and lowercase SHA-256 digest. Artifact ids and paths are unique. Absolute
paths, parent traversal, current-directory components, and backslash-separated
paths are rejected.

The artifact digest binds the complete file, including its metadata and tensor
directory. Tensor records may rely on that digest or additionally provide a
tensor-level SHA-256 digest.

## Tensor Catalog

Each logical source tensor appears once and contains:

- stable tensor id and native tensor name;
- GGML type number;
- dimensions in native GGUF order;
- optional stage-independent layer ordinal for physical grouping and
  diagnostics only;
- owned storage or an explicit alias.

Owned storage records use an absolute byte offset within the artifact, stored
byte length, power-of-two alignment, and integrity mode. Validation rejects
unknown artifacts, misalignment, integer overflow, out-of-bounds ranges, and
overlap between independently owned tensor records.

Alias records point directly to an owned tensor. Alias chains are rejected.
Alias GGML type and dimensions must match the target. This makes shared storage
explicit without allowing overlapping owned ranges to masquerade as aliases.

Tensor ids and names are both unique. Layer ordinals must be less than the
manifest layer count and dimensions must be non-zero.

## Sidecars

The schema currently admits one closed sidecar kind: `mmproj`. Unknown kinds
fail during deserialization. Sidecars reference artifact ids but imply no stage
ownership, and their semantic identity is the unique `(kind, name)` pair.
Multiple projectors therefore require stable distinct names; the package writer
uses each projector's deterministic artifact id as its name. Generation remains
a typed manifest field rather than a generic sidecar.

## Loading Rule

The runtime must validate the manifest before resolving a graph plan. It then
maps `StageSlicePlan.tensor_dependencies` to catalog entries by exact tensor id,
verifies artifact integrity, and reads only owned storage required by the
closure. Aliases resolve to their owned target before allocation.

No runtime path accepts package v1 after the atomic v2 cutover. A v1 converter
is offline tooling only and may certify completeness only against the original
source tensor directories or an independently captured source-bound inventory.

## Required Follow-Up

Before production cutover:

- the writer, preflight command, runtime reader, and converter must use the
  shared schema;
- written artifacts must be reopened and reconciled against the exact source
  inventory;
- package certification must build the metadata-only unsplit graph without
  reading tensor payloads;
- v1 manifest structs and heuristic stage-selection policy must be deleted.
