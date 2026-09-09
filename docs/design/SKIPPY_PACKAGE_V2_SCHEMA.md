# Skippy Model Package v2 Schema

Status: implementation contract for `feature/skippy_graph_filter_v2`.

The authoritative Rust types and validation rules live in
`crates/skippy-package-format`. Package creation, preflight, runtime loading,
offline conversion, and test fixtures must consume that crate rather than
declaring private copies of the manifest.

## Purpose

Package v2 describes the complete model inventory without assigning tensors to
runtime stages. A small JSON root binds the package artifacts, while a
payload-free GGUF carrier holds the complete model metadata, tensor directory,
and payload locators. A graph-derived `StageSlicePlan` selects exact tensor ids
after the normal model graph is built in metadata-only mode.

The package format has no model-family staging policy, endpoint ownership, or
cut-specific artifacts. Per-layer GGUF files remain valid physical containers,
but their placement does not determine whether a stage owns a tensor.

`source_model` records the immutable input distribution. Its digest is bound to
the declared primary source file and does not identify any generated package
artifact. `source_model.metadata_artifact_id` instead selects the generated GGUF
container used to construct the metadata-only graph. That artifact has its own
size and digest in the artifact catalog, so a packager may emit a small shared
metadata container while preserving the original source identity for
provenance and verification.

## Top-Level Contract

The serialized `model-package.json` root contains:

- `schema_version`: exactly `2`;
- `package_id`: canonical `sha256:<lowercase-hex>` identity;
- source and model identities;
- an artifact catalog;
- optional projector and generation sidecars;
- native ABI and package-generator versions;
- creation time for provenance.

`model_metadata` and `tensor_catalog` are runtime fields reconstructed from the
metadata carrier. They are rejected if they appear in the JSON root. This keeps
tokenizer arrays and tensor descriptors in their native GGUF representation and
prevents two serialized copies from disagreeing.

Unknown fields are rejected. Schema evolution therefore requires a deliberate
schema-version change rather than silently changing the meaning of an existing
package.

## Canonical Package Identity

The package id is computed by the shared crate as follows:

1. Clone the manifest and replace `package_id` with the empty string.
2. Sort source files by path, artifacts by id, and sidecars by
   `(kind, name, artifact_id)`.
3. Serialize the normalized root with the shared Rust schema.
4. Hash the serialized bytes with SHA-256 and prefix the lowercase digest with
   `sha256:`.

Validation recomputes this identity. Changing the carrier digest, a payload
artifact digest, ABI requirements, or generator provenance changes the package
identity. The carrier digest transitively binds its model metadata, tensor
directory, and payload locator plane. Enumeration order alone does not.

## Artifact Catalog

Each artifact has a stable id, normalized relative package path, total byte
size, and lowercase SHA-256 digest. Artifact ids and paths are unique. Absolute
paths, parent traversal, current-directory components, and backslash-separated
paths are rejected.

The artifact digest binds the complete file, including its metadata and tensor
directory. Tensor records may rely on that digest or additionally provide a
tensor-level SHA-256 digest.

## Metadata Carrier

`source_model.metadata_artifact_id` selects a zero-payload GGUF artifact. Its
file length equals its GGUF data offset. The carrier contains the source model's
normalized GGUF metadata and complete tensor directory plus these typed Skippy
keys:

- `skippy.package.metadata_only = true`;
- `skippy.package.part_count`: number of payload GGUF artifacts;
- `skippy.package.locator_schema = 1`;
- `skippy.package.tensor_part`: `uint32[]` payload artifact indices;
- `skippy.package.tensor_offset`: `uint64[]` absolute payload offsets;
- `skippy.package.tensor_size`: `uint64[]` stored payload lengths;
- `skippy.package.tensor_alignment`: `uint32[]` required alignments.

Each locator array has exactly one entry per carrier tensor and follows GGUF
tensor-directory order. Payload artifacts are indexed by artifact id after
excluding the metadata carrier and sidecars. The runtime rejects an unknown
locator version, wrong array type or length, invalid part index, invalid
alignment, or an extent outside the declared artifact size.

## Runtime Tensor Catalog

Carrier resolution reconstructs one runtime record for each logical source
tensor with:

- stable tensor id and native tensor name;
- GGML type number;
- dimensions in native GGUF order;
- optional stage-independent layer ordinal for physical grouping and
  diagnostics only;
- owned storage in one payload artifact.

Owned storage records use an absolute byte offset within the artifact, stored
byte length, power-of-two alignment, and integrity mode. Validation rejects
unknown artifacts, misalignment, integer overflow, out-of-bounds ranges, and
overlap between independently owned tensor records.

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

The runtime validates the JSON root, fetches and verifies its declared metadata
carrier, resolves the runtime model metadata and tensor catalog, and validates
the complete package inventory before graph planning. It then maps
`StageSlicePlan.tensor_dependencies` to catalog entries by exact tensor id,
verifies payload artifact integrity, and reads only storage required by the
closure.

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
