"""Offline bindings to maintainer-reviewed producer admission; no authentication."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 32 * 1024 * 1024
MAX_SAFE = 9007199254740991
IMAGE = "ghcr.io/mesh-llm/mesh-llm-cuda-runner"
SHA = re.compile(r"[a-f0-9]{40}\Z")
DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError("runner evidence: " + message)


def fields(value, names):
    require(isinstance(value, dict) and set(value) == set(names.split()), "invalid fields: " + names)


def text(value):
    require(isinstance(value, str) and 0 < len(value) <= 256 and not any(ord(c) < 32 or ord(c) == 127 for c in value), "invalid bounded string")


def digest(value):
    require(isinstance(value, str) and DIGEST.fullmatch(value), "invalid SHA256")


def revision(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid revision")


def hash_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    require(len(raw) <= MAX_BYTES, "JSON exceeds 32 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: require(False, "nonfinite JSON"))
    except RecursionError as error:
        raise ValueError("runner evidence: JSON nesting exceeds parser limit") from error
    def bounded(node, depth=0):
        require(depth <= 64, "JSON nesting exceeds 64 levels")
        if isinstance(node, dict):
            for key, child in node.items():
                text(key)
                bounded(child, depth + 1)
        elif isinstance(node, list):
            require(len(node) <= 4096, "array exceeds 4096 entries")
            for child in node:
                bounded(child, depth + 1)
        elif type(node) in (int, float):
            require(type(node) is int and abs(node) <= MAX_SAFE, "expected safe JSON integer")
    bounded(value)
    return value


def read_bytes(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES, "expected bounded regular file")
    with path.open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "file exceeds 32 MiB")
    return raw


def evidence_path(root, sha):
    digest(sha)
    require(root.is_dir() and not root.is_symlink(), "invalid evidence root")
    root = root.resolve()
    path = root
    for part in ("ci", "runner-image-evidence", sha[7:] + ".json"):
        path = path / part
        require(not path.is_symlink(), "symlink evidence path")
    return path


def origin(value):
    fields(value, "repository repository_id workflow_id workflow_path run_id run_attempt event runner_images_revision mesh_revision timestamp")
    require(value["repository"] == "Mesh-LLM/mesh-llm-runner-images" and value["workflow_path"] == ".github/workflows/build-and-push.yml", "unexpected producer")
    require(value["event"] in ("push", "schedule", "workflow_dispatch"), "untrusted producer event")
    for key in ("repository_id", "workflow_id", "run_id", "run_attempt"):
        require(type(value[key]) is int and 0 < value[key] <= MAX_SAFE, "invalid origin integer")
    for key in ("runner_images_revision", "mesh_revision"):
        revision(value[key])
    require(isinstance(value["timestamp"], str) and re.fullmatch(r"[0-9]{14}", value["timestamp"]), "invalid timestamp")
    datetime.strptime(value["timestamp"], "%Y%m%d%H%M%S")


def provenance(value):
    fields(value, "schema scope validation cohort_sha256 origin admission_validator_revision")
    require(type(value["schema"]) is int and value["schema"] == 1, "invalid provenance schema")
    require(value["scope"] == "reviewed_producer_admission" and value["validation"] == "offline_binding_only", "unsupported trust claim")
    digest(value["cohort_sha256"])
    origin(value["origin"])
    revision(value["admission_validator_revision"])


def validate_binding(image, raw):
    """Check membership/identity, not the producer's tool policy or admission again."""
    ref, proof = image["receipt"], image["provenance"]
    fields(ref, "schema cohort_sha256 index_candidate_key")
    require(type(ref["schema"]) is int and ref["schema"] == 1, "invalid receipt schema")
    provenance(proof)
    digest(ref["cohort_sha256"])
    require(ref["cohort_sha256"] == proof["cohort_sha256"] == hash_bytes(raw), "cohort bytes differ from reviewed anchor")
    text(ref["index_candidate_key"])
    cohort = decode(raw)
    fields(cohort, "schema type image origin catalog_sha256 candidates platforms")
    require(type(cohort["schema"]) is int and cohort["schema"] == 1 and cohort["type"] == "mesh-llm-runner-staged-cohort" and cohort["image"] == IMAGE, "invalid cohort")
    require(cohort["origin"] == proof["origin"], "origin differs from reviewed anchor")
    digest(cohort["catalog_sha256"])
    require(isinstance(cohort["candidates"], dict) and 0 < len(cohort["candidates"]) <= 256 and isinstance(cohort["platforms"], dict) and 0 < len(cohort["platforms"]) <= 256, "invalid cohort maps")
    candidate = cohort["candidates"].get(ref["index_candidate_key"])
    fields(candidate, "schema type image environment backend mesh_revision runner_images_revision digest children")
    require(type(candidate["schema"]) is int and candidate["schema"] == 1 and candidate["type"] == "mesh-llm-runner-image-candidate", "invalid index candidate")
    require(candidate["image"] == IMAGE and candidate["environment"] == image["environment"] and image["reference"] == IMAGE + "@" + candidate["digest"], "image/index binding mismatch")
    digest(candidate["digest"])
    family = candidate["backend"]
    fields(family, "id name cuda_series rocm_version")
    for key in ("id", "name"):
        text(family[key])
    require(family["id"] == image["backend"] or image["backend"] == "rocm" and family["name"] == "rocm" and re.fullmatch(r"rocm[0-9]+", family["id"]), "family binding mismatch")
    require(ref["index_candidate_key"] == f"candidate-index-{candidate['environment']}-{family['id']}", "index candidate key/family mismatch")
    expected_backend = "cuda" if image["backend"].startswith("cuda") else image["backend"]
    require(family["name"] == expected_backend, "normalized family mismatch")
    for key in ("cuda_series", "rocm_version"):
        if family[key] is not None:
            text(family[key])
    require((family["cuda_series"] is not None) == (expected_backend == "cuda") and (family["rocm_version"] is not None) == (expected_backend == "rocm"), "toolkit family mismatch")
    for key in ("mesh_revision", "runner_images_revision"):
        require(candidate[key] == proof["origin"][key], "candidate source mismatch")
    children = candidate["children"]
    require(isinstance(children, list) and 0 < len(children) <= 2, "invalid index children")
    seen = set()
    for child in children:
        fields(child, "os architecture digest")
        require(child["os"] == "linux" and child["architecture"] in ("amd64", "arm64") and child["architecture"] not in seen, "invalid or duplicate child platform")
        seen.add(child["architecture"])
        digest(child["digest"])
        key = f"{candidate['environment']}-{family['id']}-{child['architecture']}"
        entry = cohort["platforms"].get(key)
        fields(entry, "candidate receipt index_base64 manifest_base64")
        platform = {"os": "linux", "architecture": child["architecture"]}
        p = entry["candidate"]
        fields(p, "schema type image environment backend mesh_revision runner_images_revision platform digest child_digest")
        require(type(p.get("schema")) is int and p.get("schema") == 1 and p.get("type") == "mesh-llm-runner-image-platform-candidate", "invalid platform candidate")
        require(p.get("platform") == platform and p.get("backend") == family and p.get("environment") == candidate["environment"] and p.get("image") == IMAGE and p.get("child_digest") == child["digest"], "platform candidate mismatch")
        r = entry["receipt"]
        fields(r, "schema type image platform backend_id oci runtime layers scope")
        require(type(r["schema"]) is int and r["schema"] == 1 and r["type"] == "mesh-llm-runner-image-identity" and r["image"] == IMAGE and r["platform"] == platform and r["backend_id"] == family["id"], "platform receipt mismatch")
        require(r["oci"]["root"]["digest"] == p.get("digest") and r["oci"]["manifest"]["digest"] == child["digest"], "OCI digest binding mismatch")
        runtime = r["runtime"]
        fields(runtime, "schema type platform family source verification expected_tools tools dependencies cache")
        require(type(runtime["schema"]) is int and runtime["schema"] == 1 and runtime["type"] == "mesh-llm-runner-runtime-identity" and runtime["platform"] == platform, "runtime platform mismatch")
        require(runtime["family"] == {"environment": candidate["environment"], "backend": family["name"], "cuda_series": family["cuda_series"], "rocm_version": family["rocm_version"]}, "runtime family mismatch")
        expected_source = {key: proof["origin"][key] for key in ("mesh_revision", "runner_images_revision")}
        require(runtime["source"] == expected_source and all(p.get(key) == value for key, value in expected_source.items()), "runtime source mismatch")
        require(runtime["verification"]["verifier_revision"] == expected_source["runner_images_revision"], "verifier revision mismatch")
        for key in ("tool_pins_sha256", "cache_policy_sha256"):
            digest(runtime["verification"][key])
        for key in ("expected_tools", "tools", "dependencies", "cache"):
            require(isinstance(runtime[key], dict), "missing producer tool/cache observations")
    prefix = f"{candidate['environment']}-{family['id']}-"
    require({key for key in cohort["platforms"] if key.startswith(prefix)} == {prefix + arch for arch in seen}, "index child set differs from cohort family platforms")
    require(len({child["digest"] for child in children}) == len(children), "duplicate child digest")
    return cohort


def validate_catalog(catalog, root):
    platforms = {}
    for image_id, image in catalog["images"].items():
        if image["receipt"] is None and image["provenance"] is None:
            continue
        require(image["receipt"] is not None and image["provenance"] is not None, "receipt/provenance must be paired")
        provenance(image["provenance"])
        cohort = validate_binding(image, read_bytes(evidence_path(root, image["provenance"]["cohort_sha256"])))
        candidate = cohort["candidates"][image["receipt"]["index_candidate_key"]]
        platforms[image_id] = {child["architecture"] for child in candidate["children"]}
    require_platforms(catalog, platforms)
    return platforms


def require_platforms(catalog, platforms):
    for row_id, row in catalog["runtime_rows"].items():
        if row["image_id"] in platforms:
            require(row["architecture"] in platforms[row["image_id"]],
                    f"{row_id}: qualified image lacks runtime row platform")
    seed = catalog["compiler_seed"]
    if seed["image_id"] in platforms:
        require(seed["architecture"] in platforms[seed["image_id"]],
                "qualified image lacks compiler seed platform")


def bind(catalog, image_id, cohort_path, anchor_path, output, root=None):
    """Emit a proposal directory, never adopt evidence into the input catalog."""
    require(image_id in catalog["images"], "unknown image ID")
    anchor = decode(read_bytes(anchor_path))
    fields(anchor, "receipt provenance")
    result = copy.deepcopy(catalog)
    current = result["images"][image_id]
    require(current["receipt"] is None and current["provenance"] is None or current["receipt"] == anchor["receipt"] and current["provenance"] == anchor["provenance"], "immutable binding conflict")
    current.update(anchor)
    raw = read_bytes(cohort_path)
    validate_binding(current, raw)
    evidence = {current["provenance"]["cohort_sha256"]: raw}
    qualified_platforms = {}
    for qualified_id, image in result["images"].items():
        if image["provenance"] is None:
            continue
        sha = image["provenance"]["cohort_sha256"]
        if sha not in evidence:
            require(root is not None, "existing evidence requires catalog root")
            evidence[sha] = read_bytes(evidence_path(root, sha))
        cohort = validate_binding(image, evidence[sha])
        candidate = cohort["candidates"][image["receipt"]["index_candidate_key"]]
        qualified_platforms[qualified_id] = {child["architecture"] for child in candidate["children"]}
    require_platforms(result, qualified_platforms)
    require(not output.exists() and not output.is_symlink(), "proposal output must not exist")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "invalid proposal parent")
    output.mkdir()
    (output / "ci/runner-image-evidence").mkdir(parents=True)
    for sha, payload in evidence.items():
        evidence_path(output, sha).write_bytes(payload)
    (output / "ci/runner-images.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"proposal": str(output), "scope": "reviewed_producer_admission", "validation": "offline_binding_only"}
