#!/usr/bin/env python3
"""Cheap llama.cpp parity certification helper.

The script joins the pinned llama.cpp model implementation inventory with a
small GGUF candidate manifest. It resolves already-cached Hugging Face GGUFs,
prints missing download commands, and can run the existing family-certify
wrapper for locally available candidates.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/skippy/llama-parity-candidates.json"
DEFAULT_UPSTREAM_PIN = ROOT / "third_party/llama.cpp/upstream.txt"
SHARDED_GGUF_RE = re.compile(r"-0*(\d+)-of-0*\d+\.gguf$", re.IGNORECASE)


def repo_cache_dir(repo: str) -> Path:
    cache_root = os.environ.get("HF_HUB_CACHE")
    if cache_root:
        hub = Path(cache_root)
    elif os.environ.get("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache/huggingface/hub"
    return hub / ("models--" + repo.replace("/", "--"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args: list[str], *, cwd: Path | None = None, quiet: bool = False) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        if quiet:
            return ""
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}"
        )
    return proc.stdout


def pinned_llama_models(llama_src: Path | None) -> list[str]:
    if llama_src:
        models_dir = llama_src / "src/models"
        if not models_dir.is_dir():
            raise SystemExit(f"llama source has no src/models: {llama_src}")
        return sorted(path.stem for path in models_dir.glob("*.cpp"))

    deps = ROOT / ".deps/llama.cpp/src/models"
    if deps.is_dir():
        return sorted(path.stem for path in deps.glob("*.cpp"))

    pin = DEFAULT_UPSTREAM_PIN.read_text(encoding="utf-8").strip()
    cache_root = Path(tempfile.mkdtemp(prefix="skippy-llama-pin."))
    try:
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/ggml-org/llama.cpp",
                str(cache_root),
            ],
            quiet=True,
        )
        run(["git", "fetch", "--depth", "1", "origin", pin], cwd=cache_root, quiet=True)
        listing = run(["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD", "src/models"], cwd=cache_root)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    models = []
    for line in listing.splitlines():
        path = Path(line)
        if path.suffix == ".cpp" and path.parent.as_posix() == "src/models":
            models.append(path.stem)
    return sorted(models)


def candidate_index(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for candidate in manifest.get("candidates", []):
        index.setdefault(candidate["llama_model"], []).append(candidate)
    return index


def priority_lookup(manifest: dict[str, Any]) -> dict[tuple[str, str], str]:
    priorities = manifest.get("support_priority", {})
    lookup: dict[tuple[str, str], str] = {}
    for priority in ("p0", "p1", "p2"):
        group = priorities.get(priority, {})
        for llama_model in group.get("llama_models", []):
            lookup[("llama_model", llama_model)] = priority
        for family in group.get("families", []):
            lookup[("family", family)] = priority
    return lookup


def row_priority(row: dict[str, Any], lookup: dict[tuple[str, str], str]) -> str:
    return (
        lookup.get(("family", str(row.get("family", ""))))
        or lookup.get(("llama_model", str(row.get("llama_model", ""))))
        or "p2"
    )


def filter_priority(
    rows: list[dict[str, Any]],
    priorities: list[str] | None,
) -> list[dict[str, Any]]:
    if not priorities:
        return rows
    requested = {priority.lower() for priority in priorities}
    return [row for row in rows if str(row.get("priority", "p2")).lower() in requested]


def candidate_file_rank(path: Path) -> int:
    name = path.name.lower()
    if "mmproj" in name:
        return 3
    shard = SHARDED_GGUF_RE.search(name)
    if shard:
        return 0 if int(shard.group(1)) == 1 else 2
    return 1


def resolve_candidate_file(candidate: dict[str, Any]) -> Path | None:
    repo = candidate.get("repo")
    include = candidate.get("include", "*.gguf")
    if not repo:
        return None
    includes = include if isinstance(include, list) else [include]
    base = repo_cache_dir(repo) / "snapshots"
    matches: list[Path] = []
    for pattern in includes:
        matches.extend(Path(path) for path in glob.glob(str(base / "*" / pattern), recursive=True))
    matches = [path for path in matches if path.exists()]
    if not matches:
        return None
    matches.sort(
        key=lambda path: (
            candidate_file_rank(path),
            path.stat().st_size,
            str(path),
        )
    )
    return matches[0]


def download_command(candidate: dict[str, Any]) -> str:
    repo = candidate.get("repo")
    include = candidate.get("include", "*.gguf")
    if not repo:
        return ""
    if isinstance(include, list):
        includes = " ".join(f"--include '{item}'" for item in include)
    else:
        includes = f"--include '{include}'"
    return f"hf download {repo} {includes}"


class GgufReader:
    def __init__(self, path: Path):
        self.handle = path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def read(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise EOFError("short GGUF read")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, typ: int) -> Any:
        if typ == 0:
            return self.read(1)[0]
        if typ == 1:
            return struct.unpack("<b", self.read(1))[0]
        if typ == 2:
            return struct.unpack("<H", self.read(2))[0]
        if typ == 3:
            return struct.unpack("<h", self.read(2))[0]
        if typ == 4:
            return self.u32()
        if typ == 5:
            return self.i32()
        if typ == 6:
            return self.f32()
        if typ == 7:
            return bool(self.read(1)[0])
        if typ == 8:
            return self.string()
        if typ == 9:
            item_type = self.u32()
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        if typ == 10:
            return self.u64()
        if typ == 11:
            return self.i64()
        if typ == 12:
            return self.f64()
        raise ValueError(f"unsupported GGUF value type {typ}")


def gguf_metadata(path: Path) -> dict[str, Any]:
    reader = GgufReader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = reader.u32()
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}: {path}")
        tensor_count = reader.u64()
        kv_count = reader.u64()
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            typ = reader.u32()
            metadata[key] = reader.value(typ)
        metadata["_tensor_count"] = tensor_count
        return metadata
    finally:
        reader.close()


def infer_model_shape(path: Path) -> tuple[int, int, str | None]:
    metadata = gguf_metadata(path)
    arch = metadata.get("general.architecture")
    layer_count = None
    activation_width = None
    for key, value in metadata.items():
        if key.endswith(".block_count") and isinstance(value, int):
            layer_count = value
        if key.endswith(".embedding_length") and isinstance(value, int):
            activation_width = value
    if layer_count is None:
        raise ValueError(f"could not infer layer count from {path}")
    if activation_width is None:
        raise ValueError(f"could not infer embedding length from {path}")
    return layer_count, activation_width, arch if isinstance(arch, str) else None


def split_args(layer_count: int) -> tuple[int, str]:
    first = max(1, layer_count // 3)
    second = max(first + 1, (2 * layer_count) // 3)
    if second >= layer_count:
        second = layer_count - 1
    split_layer = max(1, layer_count // 2)
    if split_layer >= layer_count:
        split_layer = layer_count - 1
    return split_layer, f"{first},{second}"


def default_stage_build_dir() -> str | None:
    if os.environ.get("LLAMA_STAGE_BUILD_DIR"):
        return os.environ["LLAMA_STAGE_BUILD_DIR"]
    llama_build_roots = (
        ROOT / ".deps/llama-build",
        ROOT / ".deps/llama.cpp",
    )
    for name in (
        "build-stage-abi-metal",
        "build-stage-abi-static",
        "build-stage-abi-cuda",
        "build-stage-abi-vulkan",
        "build-stage-abi-rocm",
    ):
        for llama_root in llama_build_roots:
            candidate = llama_root / name
            if candidate.is_dir():
                return str(candidate)
    return None


def inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_json(args.manifest)
    candidates = candidate_index(manifest)
    priorities = priority_lookup(manifest)
    rows = []
    for model in pinned_llama_models(args.llama_src):
        entries = candidates.get(model, [])
        if not entries:
            row = {
                "llama_model": model,
                "family": model.replace("-", "_"),
                "status": "missing_candidate",
                "repo": None,
                "local_path": None,
                "download": "",
            }
            row["priority"] = row_priority(row, priorities)
            rows.append(row)
            continue
        for entry in entries:
            path = resolve_candidate_file(entry)
            row = {
                "llama_model": model,
                "family": entry.get("family", model.replace("-", "_")),
                "status": entry.get("status", "candidate"),
                "repo": entry.get("repo"),
                "include": entry.get("include"),
                "local_path": str(path) if path else None,
                "download": "" if path else download_command(entry),
                "notes": entry.get("notes", ""),
            }
            row["priority"] = row_priority(row, priorities)
            if path:
                try:
                    if path.suffix == ".gguf":
                        layer_count, activation_width, arch = infer_model_shape(path)
                        split_layer, splits = split_args(layer_count)
                        row.update(
                            {
                                "gguf_arch": arch,
                                "layer_end": entry.get("layer_end", layer_count),
                                "activation_width": activation_width,
                                "split_layer": entry.get("split_layer", split_layer),
                                "splits": entry.get("splits", splits),
                            }
                        )
                    else:
                        row["package_manifest"] = str(path)
                except Exception as exc:  # keep inventory useful for bad/corrupt downloads
                    row["inspect_error"] = str(exc)
            rows.append(row)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| priority | llama model | family | status | local | candidate/download |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        local = "yes" if row.get("local_path") else "no"
        target = row.get("repo") or row.get("download") or ""
        if row.get("download"):
            target = f"`{row['download']}`"
        print(
            f"| `{row.get('priority', 'p2')}` | `{row['llama_model']}` | `{row['family']}` | {row['status']} | {local} | {target} |"
        )


def validate_inventory(
    rows: list[dict[str, Any]], llama_src: Path | None = None
) -> int:
    failures = 0
    missing = [row for row in rows if row.get("status") == "missing_candidate"]
    if missing:
        failures += len(missing)
        print("Missing parity manifest rows:", file=sys.stderr)
        for row in missing:
            print(f"  - {row['llama_model']}", file=sys.stderr)

    allowed_statuses = {
        "candidate",
        "candidate_stateful",
        "candidate_multimodal",
        "certified",
        "certified_package_only",
        "implementation_base",
        "needs_boundary_registration",
        "needs_candidate",
        "needs_runtime_slice_support",
        "no_public_gguf_candidate",
        "non_causal_aux",
        "package_or_remote_only",
    }
    unknown_statuses = [
        row
        for row in rows
        if row.get("status") not in allowed_statuses
        and row.get("status") != "missing_candidate"
    ]
    if unknown_statuses:
        failures += len(unknown_statuses)
        print("Unknown parity statuses:", file=sys.stderr)
        for row in unknown_statuses:
            print(
                f"  - {row['llama_model']}: {row.get('status')}",
                file=sys.stderr,
            )

    failures += validate_runtime_slice_admission()
    failures += validate_boundary_registration(
        rows, boundary_registered_models(llama_src)
    )
    failures += validate_model_pins(rows)
    failures += validate_pin_manifest_join(rows)

    return failures


def validate_pin_manifest_join(rows: list[dict[str, Any]]) -> int:
    """A parity model_pin must join an identical family-certified artifact.

    The family-certified manifest's `file_integrity` is the certification
    source of truth; a parity row that pins a model the certification
    manifest does not know about (or disagrees with on repo, revision,
    file, size, or blob sha256) fails closed. This keeps the two manifests
    from drifting apart on the same immutable model.
    """
    certified_path = ROOT / "ci/llama-canary/family-certified.json"
    try:
        certified = json.loads(certified_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"cannot read {certified_path}: {error}", file=sys.stderr)
        return 1
    # (repo, revision, file) -> (size_bytes, blob_sha256)
    certified_index: dict[tuple[str, str, str], tuple[int, str]] = {}
    for model in certified.get("models", []):
        artifact = model.get("artifact") or {}
        integrity = artifact.get("file_integrity") or {}
        for file_name, record in integrity.items():
            key = (artifact.get("repo", ""), artifact.get("revision", ""), file_name)
            certified_index[key] = (
                record.get("size_bytes", -1),
                record.get("blob_id", ""),
            )

    failures = 0
    for row in rows:
        pin = row.get("model_pin")
        if pin is None:
            continue
        key = (pin.get("repo", ""), pin.get("revision", ""), pin.get("file", ""))
        record = certified_index.get(key)
        if record is None:
            failures += 1
            print(
                f"model_pin for {row['llama_model']} does not join any "
                f"family-certified.json artifact: {key[0]}@{key[1][:12]} {key[2]}",
                file=sys.stderr,
            )
            continue
        size, blob = record
        if pin.get("size_bytes") != size or pin.get("blob_sha256") != blob:
            failures += 1
            print(
                f"model_pin for {row['llama_model']} disagrees with "
                f"family-certified.json integrity for {key[2]}",
                file=sys.stderr,
            )
    return failures


def boundary_registered_models(llama_src: Path | None) -> set[str]:
    """Model implementations that register stage block boundaries.

    A model file counts as registered only when it calls both `begin_block`
    and `end_block` (the per-layer boundary pair is always added in the same
    edit per the llama-patch-changes skill). A file with only one half of
    the pair cannot certify.
    """
    source = llama_src or ROOT / ".deps/llama.cpp"
    models_dir = source / "src/models"
    if not models_dir.is_dir():
        return set()
    registered: set[str] = set()
    for path in models_dir.glob("*.cpp"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "begin_block" in text and "end_block" in text:
            registered.add(path.stem)
    return registered


def validate_model_pins(rows: list[dict[str, Any]]) -> int:
    """Pinned model rows must be immutable — no floating refs.

    A row that carries a `model_pin` (added when the repair lane registers a
    new family's smallest runnable GGUF) must pin repo, 40-hex revision,
    file name, byte size, and a 64-hex blob sha256, mirroring the
    `file_integrity` schema of `ci/llama-canary/family-certified.json`.
    """
    failures = 0
    for row in rows:
        pin = row.get("model_pin")
        if pin is None:
            continue
        repo = pin.get("repo")
        revision = pin.get("revision")
        file_name = pin.get("file")
        size = pin.get("size_bytes")
        blob = pin.get("blob_sha256")
        problems = []
        if not isinstance(repo, str) or "/" not in repo:
            problems.append("repo must be an org/name string")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            problems.append("revision must be a 40-hex commit sha")
        if not isinstance(file_name, str) or not file_name.endswith(".gguf"):
            problems.append("file must name a .gguf")
        if not isinstance(size, int) or size <= 0:
            problems.append("size_bytes must be a positive integer")
        if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{64}", blob):
            problems.append("blob_sha256 must be 64-hex")
        if problems:
            failures += 1
            print(f"invalid model_pin for {row['llama_model']}: {problems}", file=sys.stderr)
    return failures


def validate_boundary_registration(
    rows: list[dict[str, Any]],
    registered: set[str],
) -> int:
    """Fail closed when a runnable family lacks boundary registration.

    Rows whose llama model does not register `begin_block`/`end_block` must
    carry an explicit `unsupported_reason` so the gap is classified rather
    than silently certified. Only a non-runnable classification may carry a
    reason: a `certified` row with `unsupported_reason` is a manifest error
    regardless of hook state.
    """
    failures = 0
    for row in rows:
        status = row.get("status")
        if status not in {"certified", "candidate", "candidate_stateful"}:
            continue
        if status == "certified" and row.get("unsupported_reason"):
            failures += 1
            print(
                f"certified row carries unsupported_reason: {row['llama_model']}",
                file=sys.stderr,
            )
            continue
        if row["llama_model"] in registered:
            continue
        reason = row.get("unsupported_reason")
        if not reason:
            failures += 1
            print(
                f"runnable family lacks boundary registration and no unsupported_reason: "
                f"{row['llama_model']} ({status})",
                file=sys.stderr,
            )
    return failures


def needs_boundary_registration_rows(
    rows: list[dict[str, Any]], registered: set[str]
) -> list[str]:
    """Models whose rows must move to `needs_boundary_registration`."""
    return sorted(
        row["llama_model"]
        for row in rows
        if row.get("status") in {"certified", "candidate", "candidate_stateful"}
        and row["llama_model"] not in registered
        and not row.get("unsupported_reason")
    )


def executable_cpp(
    source: str, preserved_string_literals: tuple[str, ...] = ()
) -> str:
    masked = list(source)
    state = "code"
    index = 0
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "line-comment":
            if char == "\n":
                state = "code"
            else:
                masked[index] = " "
        elif state == "block-comment":
            masked[index] = "\n" if char == "\n" else " "
            if char == "*" and next_char == "/":
                masked[index + 1] = " "
                state = "code"
                index += 1
        elif char == "/" and next_char == "/":
            masked[index] = masked[index + 1] = " "
            state = "line-comment"
            index += 1
        elif char == "/" and next_char == "*":
            masked[index] = masked[index + 1] = " "
            state = "block-comment"
            index += 1
        else:
            raw_match = None
            if index == 0 or not (source[index - 1].isalnum() or source[index - 1] == "_"):
                raw_match = re.match(
                    r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(', source[index:]
                )
            if raw_match is not None:
                delimiter = raw_match.group(1)
                close = f'){delimiter}"'
                end = source.find(close, index + raw_match.end())
                end = len(source) if end < 0 else end + len(close)
                for mask_index in range(index, end):
                    if source[mask_index] != "\n":
                        masked[mask_index] = " "
                index = end - 1
            elif char in {"'", '"'}:
                quote = char
                end = index + 1
                while end < len(source):
                    if source[end] == "\\":
                        end += 2
                        continue
                    end += 1
                    if source[end - 1] == quote:
                        break
                literal = source[index:end]
                preserve = quote == '"' and literal in preserved_string_literals
                if not preserve:
                    for mask_index in range(index, min(end, len(source))):
                        if source[mask_index] != "\n":
                            masked[mask_index] = " "
                index = end - 1
        index += 1
    return "".join(masked)


def _skip_balanced(text: str, index: int, open_char: str, close_char: str) -> int:
    depth = 0
    while index < len(text):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return index


def has_unbraced_control_statement(executable_source: str) -> bool:
    """Detect a depth-0 control statement whose body is not a braced block.

    Braced control bodies are opaque to the failure-path checks once nested
    blocks are masked, so they cannot conditionally expose a failure path (the
    production guards emit failure events through exactly such a block). An
    unbraced control statement leaves its single statement visible to the
    ordered failure-path match while keeping it conditionally executed, so a
    guard like `if (false) llama_model_free(model); ...` must be rejected.
    """
    keyword_pattern = re.compile(r"\b(?:if|for|while|switch|catch|do|try|else)\b")
    index = 0
    while True:
        match = keyword_pattern.search(executable_source, index)
        if match is None:
            return False
        cursor = match.end()
        if match.group(0) == "else" and executable_source[cursor:].lstrip().startswith(
            "if"
        ):
            # `else if` is validated through the nested `if` match.
            index = cursor
            continue
        if match.group(0) not in {"do", "try", "else"}:
            while cursor < len(executable_source) and executable_source[cursor].isspace():
                cursor += 1
            if cursor < len(executable_source) and executable_source[cursor] == "(":
                cursor = _skip_balanced(executable_source, cursor, "(", ")")
        while cursor < len(executable_source) and executable_source[cursor].isspace():
            cursor += 1
        if cursor >= len(executable_source) or executable_source[cursor] != "{":
            return True
        index = _skip_balanced(executable_source, cursor, "{", "}")


def mask_nested_blocks(source: str) -> str:
    masked = list(source)
    depth = 0
    state = "code"
    index = 0
    while index < len(source):
        char = source[index]
        if state in {"single-quote", "double-quote"}:
            if depth > 0 and char != "\n":
                masked[index] = " "
            if char == "\\":
                index += 1
                if index < len(source) and depth > 0 and source[index] != "\n":
                    masked[index] = " "
            elif (state == "single-quote" and char == "'") or (
                state == "double-quote" and char == '"'
            ):
                state = "code"
        elif char == "'":
            state = "single-quote"
            if depth > 0:
                masked[index] = " "
        elif char == '"':
            state = "double-quote"
            if depth > 0:
                masked[index] = " "
        elif char == "{":
            depth += 1
            masked[index] = " "
        elif char == "}":
            masked[index] = " "
            depth = max(0, depth - 1)
        elif depth > 0 and char != "\n":
            masked[index] = " "
        index += 1
    return "".join(masked)


def extract_braced_block(source: str, header_pattern: str) -> str | None:
    executable_source = executable_cpp(source)
    match = re.search(header_pattern + r"\s*\{", executable_source)
    if match is None:
        return None
    open_brace = executable_source.find("{", match.start(), match.end())
    depth = 0
    index = open_brace
    while index < len(source):
        char = executable_source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
        index += 1
    return None


def validate_runtime_slice_admission(llama_root: Path | None = None) -> int:
    llama_root = llama_root or ROOT / ".deps/llama.cpp"
    model_loading = llama_root / "src/skippy/model_loading.cpp"
    if not model_loading.exists():
        return 0

    source = model_loading.read_text(encoding="utf-8")
    executable_source = executable_cpp(source)
    # Match the admission function with or without internal-linkage
    # qualifiers: upstream revisions have carried both spellings.
    function_start = -1
    for needle in (
        "static enum skippy_status skippy_finish_model_open(",
        "enum skippy_status skippy_finish_model_open(",
    ):
        function_start = executable_source.find(needle)
        if function_start >= 0:
            break
    function_end = executable_source.find(
        "enum skippy_status skippy_model_open_impl(", function_start
    )
    if function_start < 0 or function_end < 0:
        print("Cannot locate Skippy runtime-slice admission function", file=sys.stderr)
        return 1

    admission = source[function_start:function_end]
    if re.search(
        r"(?m)^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b",
        executable_cpp(admission),
    ):
        print(
            "Runtime-slice admission contract must not use preprocessor branches",
            file=sys.stderr,
        )
        return 1
    validation_end = admission.find("skippy_model * stage_model")
    if validation_end < 0:
        print("Cannot locate Skippy runtime-slice validation boundary", file=sys.stderr)
        return 1
    validation = admission[:validation_end]
    executable_validation = executable_cpp(validation)
    architecture_guards = sorted(
        set(
            re.findall(
                r"model->arch\s*[!=]=\s*LLM_ARCH_([A-Z0-9_]+)",
                executable_validation,
            )
        )
    )
    failures = 0
    if architecture_guards:
        failures += len(architecture_guards)
        print(
            "Runtime-slice admission must not depend on a model-architecture allowlist:",
            file=sys.stderr,
        )
        for architecture in architecture_guards:
            print(f"  - {architecture}", file=sys.stderr)

    invalid_argument_contracts = (
        (
            "layer_end range",
            r"if\s*\(\s*config->layer_end\s*>\s*n_layer\s*\)",
            "layer_end exceeds model layer count",
        ),
        (
            "embedding ownership",
            r"if\s*\(\s*config->include_embeddings\s*&&\s*"
            r"config->layer_start\s*!=\s*0\s*&&\s*!config->include_output\s*\)",
            "only the first runtime slice may include token embeddings",
        ),
        (
            "first-slice embeddings",
            r"if\s*\(\s*config->layer_start\s*==\s*0\s*&&\s*"
            r"!config->include_embeddings\s*\)",
            "the first runtime slice must include token embeddings",
        ),
        (
            "output ownership",
            r"if\s*\(\s*config->include_output\s*&&\s*"
            r"config->layer_end\s*!=\s*n_layer\s*\)",
            "only the final runtime slice may include output tensors",
        ),
    )
    missing_checks = []
    for name, guard, message in invalid_argument_contracts:
        body = extract_braced_block(admission, guard)
        executable_body = (
            executable_cpp(body, (f'"{message}"',)) if body is not None else ""
        )
        masked_body = mask_nested_blocks(executable_body) if body is not None else ""
        required_failure_path = (
            r"llama_model_free\s*\(\s*model\s*\)\s*;"
            r"[\s\S]*?const\s+char\s*\*\s*message\s*=\s*"
            + re.escape(f'"{message}"')
            + r"\s*;[\s\S]*?skippy_set_error\s*\(\s*out_error\s*,\s*"
            r"SKIPPY_STATUS_INVALID_ARGUMENT\s*,\s*message\s*\)\s*;"
            r"[\s\S]*?return\s+SKIPPY_STATUS_INVALID_ARGUMENT\s*;"
        )
        failure_match = re.search(required_failure_path, masked_body)
        first_return = re.search(r"\breturn\b", masked_body)
        expected_return = re.search(
            r"return\s+SKIPPY_STATUS_INVALID_ARGUMENT\s*;", masked_body
        )
        if (
            body is None
            or failure_match is None
            or first_return is None
            or expected_return is None
            or first_return.start() != expected_return.start()
            or has_unbraced_control_statement(executable_body)
        ):
            missing_checks.append(name)

    boundary_contracts = (
        (
            "output activation boundary",
            r"if\s*\(\s*!stage_model->ctx->get_activation_boundary\s*\([^)]*\)\s*\)",
            "stage graph did not expose a stable output activation boundary",
        ),
        (
            "input activation boundary",
            r"if\s*\(\s*!stage_model->ctx->get_input_activation_boundary\s*\([^)]*\)\s*\)",
            "stage graph did not expose a stable input activation boundary",
        ),
    )
    for name, guard, message in boundary_contracts:
        body = extract_braced_block(admission, guard)
        executable_body = (
            executable_cpp(body, (f'"{message}"',)) if body is not None else ""
        )
        masked_body = mask_nested_blocks(executable_body) if body is not None else ""
        failure_path = (
            r"return\s+fail_boundary_load\s*\(\s*"
            + re.escape(f'"{message}"')
            + r"\s*\)\s*;"
        )
        failure_match = re.search(failure_path, masked_body)
        first_return = re.search(r"\breturn\b", masked_body)
        if (
            body is None
            or failure_match is None
            or first_return is None
            or first_return.start() != failure_match.start()
            or has_unbraced_control_statement(executable_body)
        ):
            missing_checks.append(name)
    if missing_checks:
        failures += len(missing_checks)
        print(
            "Runtime-slice admission is missing realized-contract checks:",
            file=sys.stderr,
        )
        for check in missing_checks:
            print(f"  - {check}", file=sys.stderr)

    return failures


def run_certifications(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    defaults = load_json(args.manifest).get("defaults", {})
    statuses = set(args.status) if args.status else {
        "candidate",
        "candidate_stateful",
        "candidate_multimodal",
        "certified",
    }
    failures = 0
    selected = [
        row
        for row in rows
        if row.get("local_path")
        and row.get("layer_end")
        and row.get("activation_width")
        and row.get("status") in statuses
        and (not args.family or row.get("family") in args.family)
        and (not args.llama_model or row.get("llama_model") in args.llama_model)
    ]
    selected = filter_priority(selected, args.priority)
    if args.limit:
        selected = selected[: args.limit]
    for row in selected:
        stage_build_dir = default_stage_build_dir()
        ctx_size = args.ctx_size or defaults.get("ctx_size", 128)
        if args.prefix_token_count:
            # ResidentKv checks keep the live source lane and a resident cache
            # sequence in the same context before verifying suffix prefill.
            ctx_size = max(ctx_size, args.prefix_token_count * 2 + 32)

        cmd = [
            str(ROOT / "scripts/family-certify.sh"),
            "--family",
            row["family"],
            "--target-model",
            row["local_path"],
            "--model-id",
            row.get("repo") or row["family"],
            "--layer-end",
            str(row["layer_end"]),
            "--split-layer",
            str(row["split_layer"]),
            "--splits",
            str(row["splits"]),
            "--activation-width",
            str(row["activation_width"]),
            "--ctx-size",
            str(ctx_size),
            "--n-gpu-layers",
            str(args.n_gpu_layers if args.n_gpu_layers is not None else defaults.get("n_gpu_layers", 999)),
            "--prompt",
            str(defaults.get("prompt", "Hello")),
            "--run-id",
            args.run_id,
        ]
        if args.startup_timeout_secs is not None:
            cmd.extend(["--startup-timeout-secs", str(args.startup_timeout_secs)])
        entry = next(
            (
                candidate
                for candidate in load_json(args.manifest).get("candidates", [])
                if candidate.get("family") == row["family"]
                and candidate.get("llama_model") == row["llama_model"]
            ),
            {},
        )
        recurrent_ranges = entry.get("recurrent_ranges")
        has_recurrent_state = entry.get("recurrent") == "all" or bool(recurrent_ranges)
        if entry.get("recurrent") == "all":
            cmd.append("--recurrent-all")
        elif recurrent_ranges:
            if isinstance(recurrent_ranges, list):
                recurrent_ranges = ",".join(str(item) for item in recurrent_ranges)
            cmd.extend(["--recurrent-ranges", str(recurrent_ranges)])
        if args.skip_build:
            cmd.append("--skip-build")
        if args.skip_state:
            cmd.append("--skip-state")
        state_payload_kind = args.state_payload_kind
        if not state_payload_kind and args.prefix_token_count:
            state_payload_kind = (
                "kv-recurrent"
                if has_recurrent_state
                else defaults.get("state_payload_kind", "resident-kv")
            )
        if state_payload_kind:
            cmd.extend(["--state-payload-kind", state_payload_kind])
        if args.prefix_token_count:
            cmd.extend(["--prefix-token-count", str(args.prefix_token_count)])
        if args.cache_hit_repeats:
            cmd.extend(["--cache-hit-repeats", str(args.cache_hit_repeats)])
        if args.borrow_resident_hits:
            cmd.append("--borrow-resident-hits")
        if args.cache_decoded_result_hits:
            cmd.append("--cache-decoded-result-hits")
        if args.dry_run:
            prefix = f"LLAMA_STAGE_BUILD_DIR={stage_build_dir} " if stage_build_dir else ""
            print(prefix + " ".join(cmd))
            continue
        print(f"==> certifying {row['family']} ({row['llama_model']})")
        env = os.environ.copy()
        if stage_build_dir:
            env["LLAMA_STAGE_BUILD_DIR"] = stage_build_dir
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if proc.returncode != 0:
            failures += 1
            if args.stop_on_failure:
                return failures
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--llama-src", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="print joined llama/candidate inventory")
    inv.add_argument("--json", action="store_true")
    inv.add_argument("--missing-only", action="store_true")
    inv.add_argument("--local-only", action="store_true")
    inv.add_argument("--priority", action="append", help="filter by p0/p1/p2 support priority")

    commands = sub.add_parser("download-commands", help="print hf download commands for missing candidates")
    commands.add_argument("--all", action="store_true", help="include non-candidate/package-only rows")
    commands.add_argument("--priority", action="append", help="filter by p0/p1/p2 support priority")

    sub.add_parser("validate", help="fail if the pinned llama.cpp inventory is not fully classified")

    classify = sub.add_parser(
        "classify-boundaries",
        help="list runnable families lacking boundary registration (repair queue)",
    )
    classify.add_argument("--json", action="store_true")

    run_parser = sub.add_parser("run", help="run family-certify for local candidates")
    run_parser.add_argument("--status", action="append")
    run_parser.add_argument("--family", action="append")
    run_parser.add_argument("--llama-model", action="append")
    run_parser.add_argument("--priority", action="append")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--skip-build", action="store_true")
    run_parser.add_argument("--skip-state", action="store_true")
    run_parser.add_argument("--state-payload-kind")
    run_parser.add_argument("--prefix-token-count", type=int)
    run_parser.add_argument("--cache-hit-repeats", type=int)
    run_parser.add_argument("--borrow-resident-hits", action="store_true")
    run_parser.add_argument("--cache-decoded-result-hits", action="store_true")
    run_parser.add_argument("--stop-on-failure", action="store_true")
    run_parser.add_argument("--ctx-size", type=int)
    run_parser.add_argument("--n-gpu-layers", type=int)
    run_parser.add_argument("--startup-timeout-secs", type=int)
    run_parser.add_argument("--run-id", default="llama-parity-cheap")

    args = parser.parse_args()
    rows = inventory(args)
    if args.command == "inventory":
        rows = filter_priority(rows, args.priority)
        if args.missing_only:
            rows = [row for row in rows if not row.get("local_path")]
        if args.local_only:
            rows = [row for row in rows if row.get("local_path")]
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            print_table(rows)
        return 0
    if args.command == "download-commands":
        rows = filter_priority(rows, args.priority)
        statuses = {
            "candidate",
            "candidate_stateful",
            "candidate_multimodal",
            "certified",
            "needs_candidate",
            "package_or_remote_only",
        }
        seen = set()
        for row in rows:
            command = row.get("download")
            if not command or command in seen:
                continue
            if not args.all and row.get("status") not in statuses:
                continue
            seen.add(command)
            print(command)
        return 0
    if args.command == "validate":
        return 1 if validate_inventory(rows, args.llama_src) else 0
    if args.command == "classify-boundaries":
        pending = needs_boundary_registration_rows(
            rows, boundary_registered_models(args.llama_src)
        )
        if args.json:
            print(json.dumps(pending, indent=2))
        else:
            for model in pending:
                print(model)
        return 0
    if args.command == "run":
        return 1 if run_certifications(args, rows) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
