#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pinned}"

LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
PIN_FILE="${LLAMA_PIN_FILE:-$ROOT/third_party/llama.cpp/upstream.txt}"
PATCH_DIR="${LLAMA_PATCH_DIR:-$ROOT/third_party/llama.cpp/patches}"
PREPARE_SCHEMA=4

if [[ ! -f "$PIN_FILE" ]]; then
  echo "missing llama upstream pin: $PIN_FILE" >&2
  exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "missing llama patch directory: $PATCH_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$LLAMA_WORKDIR")"

# All git operations below run as `git -C "$LLAMA_WORKDIR"`. `git -C` only
# changes directory; it does NOT pin git to that directory. If $LLAMA_WORKDIR
# has no valid .git (e.g. an interrupted/partial clone), git's normal upward
# repository discovery walks PAST it to the nearest enclosing .git. When
# mesh-llm is vendored inside another git repo's working tree (cargo/hermit git
# checkouts do exactly this), that enclosing repo gets mutated instead -- its
# origin rewritten by `remote set-url`, its HEAD clobbered by `git am`. Pin the
# ceiling to the workdir's parent so discovery can never escape upward: a
# missing .git then errors loudly here instead of corrupting the parent repo.
# This is transparent to a normal build, where $LLAMA_WORKDIR has its own .git
# at or below this ceiling.
#
# Resolve to a physical path (pwd -P): git compares GIT_CEILING_DIRECTORIES
# against symlink-resolved paths, so a logical path could silently fail to
# match and let discovery escape. Fail loudly if resolution fails rather than
# exporting an empty ceiling, which would disable the guard entirely.
LLAMA_WORKDIR_PARENT="$(cd "$(dirname "$LLAMA_WORKDIR")" && pwd -P)" || {
  echo "error: cannot resolve parent directory of LLAMA_WORKDIR ($LLAMA_WORKDIR)" >&2
  exit 1
}
export GIT_CEILING_DIRECTORIES="$LLAMA_WORKDIR_PARENT"

git_retry() {
  local attempt=1
  local max_attempts="${LLAMA_GIT_MAX_ATTEMPTS:-4}"
  local delay="${LLAMA_GIT_RETRY_DELAY_SECONDS:-10}"
  local status=0

  while (( attempt <= max_attempts )); do
    if "$@"; then
      return 0
    else
      status=$?
    fi

    if (( attempt == max_attempts )); then
      return "$status"
    fi

    echo "git command failed (attempt $attempt/$max_attempts): $*" >&2
    echo "retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

clone_llama_workdir() {
  local attempt=1
  local max_attempts="${LLAMA_GIT_MAX_ATTEMPTS:-4}"
  local delay="${LLAMA_GIT_RETRY_DELAY_SECONDS:-10}"
  local status=0

  while (( attempt <= max_attempts )); do
    rm -rf "$LLAMA_WORKDIR"
    if git clone "$LLAMA_UPSTREAM_URL" "$LLAMA_WORKDIR"; then
      return 0
    else
      status=$?
    fi

    if (( attempt == max_attempts )); then
      return "$status"
    fi

    echo "llama.cpp clone failed (attempt $attempt/$max_attempts)" >&2
    echo "retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

is_partial_llama_workdir() {
  local promisor
  promisor="$(git -C "$LLAMA_WORKDIR" config --bool --get remote.origin.promisor 2>/dev/null || true)"
  [[ "$promisor" == "true" ]] ||
    [[ -n "$(git -C "$LLAMA_WORKDIR" config --get extensions.partialClone 2>/dev/null || true)" ]]
}

# Re-clone unless $LLAMA_WORKDIR is genuinely its own complete git repository.
# A bare `[[ ! -d "$LLAMA_WORKDIR/.git" ]]` check passes a partial/corrupt
# checkout (dir present, .git missing or incomplete) straight through to the
# `git -C` operations below; combined with discovery walking upward, that is
# what lets this script escape into an enclosing repo. A blobless checkout is
# also unsuitable: `git am --3way` must read both older upstream preimages and
# patch-result object IDs, which a promisor remote can incorrectly try to
# fetch from upstream. `rev-parse --git-dir` only succeeds for a real repo
# rooted at the workdir (discovery cannot escape past GIT_CEILING_DIRECTORIES
# set above). clone_llama_workdir rm -rf's first, so re-cloning this generated
# dependency worktree is safe.
if ! git -C "$LLAMA_WORKDIR" rev-parse --git-dir >/dev/null 2>&1 ||
    is_partial_llama_workdir; then
  clone_llama_workdir
fi

case "$MODE" in
  pinned)
    TARGET_SHA="$(tr -d '[:space:]' < "$PIN_FILE")"
    ;;
  latest)
    git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
    git_retry git -C "$LLAMA_WORKDIR" fetch origin master --tags
    TARGET_SHA="$(git -C "$LLAMA_WORKDIR" rev-parse origin/master)"
    ;;
  *)
    TARGET_SHA="$MODE"
    ;;
esac

PATCHES=()
while IFS= read -r patch; do
  PATCHES+=("$patch")
done < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | sort)

validate_patch_sequence() {
  local expected=1
  local expected_prefix filename patch prefix sequence

  for patch in "${PATCHES[@]}"; do
    filename="${patch##*/}"
    if [[ ! "$filename" =~ ^([0-9]{4})-.+\.patch$ ]]; then
      echo "invalid llama patch filename: $filename" >&2
      echo "expected a four-digit sequence prefix such as 0001-description.patch" >&2
      return 1
    fi

    prefix="${BASH_REMATCH[1]}"
    sequence=$((10#$prefix))
    printf -v expected_prefix '%04d' "$expected"
    if (( sequence != expected )); then
      echo "invalid llama patch sequence: expected $expected_prefix, found $prefix ($filename)" >&2
      echo "patch numbers must be unique and contiguous" >&2
      return 1
    fi
    expected=$((expected + 1))
  done
}

validate_patch_sequence

GENERATED_PATCH_DIR="$PATCH_DIR/generated"
GENERATED_SERIES="$GENERATED_PATCH_DIR/series"
if [[ -e "$GENERATED_PATCH_DIR" && ! -f "$GENERATED_SERIES" ]]; then
  echo "generated family patch directory is missing its series file: $GENERATED_PATCH_DIR" >&2
  exit 1
fi
if [[ -f "$GENERATED_SERIES" ]]; then
  generated_expected=1
  generated_count=0
  while IFS= read -r filename || [[ -n "$filename" ]]; do
    if [[ ! "$filename" =~ ^([0-9]{4})-family-[a-z0-9.-]+(--[a-z0-9.-]+)*\.patch$ ]]; then
      echo "invalid generated family patch filename in series: $filename" >&2
      exit 1
    fi
    sequence=$((10#${BASH_REMATCH[1]}))
    if (( sequence != generated_expected )); then
      printf -v expected_prefix '%04d' "$generated_expected"
      echo "invalid generated family patch sequence: expected $expected_prefix, found ${BASH_REMATCH[1]} ($filename)" >&2
      exit 1
    fi
    patch="$GENERATED_PATCH_DIR/$filename"
    if [[ ! -f "$patch" ]]; then
      echo "generated family patch listed by series is missing: $patch" >&2
      exit 1
    fi
    PATCHES+=("$patch")
    generated_expected=$((generated_expected + 1))
    generated_count=$((generated_count + 1))
  done < "$GENERATED_SERIES"
  actual_generated_count="$(find "$GENERATED_PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | wc -l | tr -d '[:space:]')"
  if (( generated_count == 0 || actual_generated_count != generated_count )); then
    echo "generated family patch series does not exactly cover its patch directory" >&2
    exit 1
  fi
fi

python_bin() {
  local candidate
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    local python
    python="$(python_bin)" || {
      echo "shasum, sha256sum, or python is required" >&2
      return 1
    }
    "$python" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
  fi
}

sha256_stream() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    local python
    python="$(python_bin)" || {
      echo "shasum, sha256sum, or python is required" >&2
      return 1
    }
    "$python" -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
  fi
}

compute_patch_digest() {
  (
    for patch in "${PATCHES[@]}"; do
      rel="${patch#"$PATCH_DIR"/}"
      checksum="$(sha256_file "$patch")"
      printf '%s\n' "$rel"
      printf '%s\n' "$checksum"
    done
  ) | sha256_stream
}

PATCH_DIGEST="$(compute_patch_digest)"

if [[ -f "$LLAMA_WORKDIR/.mesh-llm-upstream-sha" &&
      -f "$LLAMA_WORKDIR/.mesh-llm-patched-sha" &&
      -f "$LLAMA_WORKDIR/.mesh-llm-patch-digest" &&
      -f "$LLAMA_WORKDIR/.mesh-llm-prepare-schema" ]]; then
  PREPARED_UPSTREAM="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-upstream-sha")"
  PREPARED_PATCHED="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
  PREPARED_DIGEST="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patch-digest")"
  PREPARED_SCHEMA="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-prepare-schema")"
  CURRENT_HEAD="$(git -C "$LLAMA_WORKDIR" rev-parse HEAD 2>/dev/null || true)"

  if [[ "$PREPARED_SCHEMA" == "$PREPARE_SCHEMA" &&
        "$PREPARED_UPSTREAM" == "$TARGET_SHA" &&
        "$PREPARED_PATCHED" == "$CURRENT_HEAD" &&
        "$PREPARED_DIGEST" == "$PATCH_DIGEST" &&
        ! -d "$LLAMA_WORKDIR/.git/rebase-apply" ]] &&
     git -C "$LLAMA_WORKDIR" diff-index --quiet HEAD --; then
    echo "llama.cpp already prepared"
    echo "  upstream: $TARGET_SHA"
    echo "  patched:  $PREPARED_PATCHED"
    echo "  workdir:  $LLAMA_WORKDIR"
    exit 0
  fi
fi

git -C "$LLAMA_WORKDIR" am --abort >/dev/null 2>&1 || true
git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
if [[ "$MODE" != "latest" ]]; then
  git_retry git -C "$LLAMA_WORKDIR" fetch origin master --tags
fi
# The patched checkout is an artifact identity shared across CI jobs. Keep the
# synthetic committer and timestamp deterministic so applying the same ordered
# patch queue to the same upstream pin always produces the same HEAD.
git -C "$LLAMA_WORKDIR" config user.name "Mesh-LLM CI"
git -C "$LLAMA_WORKDIR" config user.email "ci@mesh-llm.local"

# The llama.cpp checkout is a generated dependency worktree. Local edits there
# should live in third_party/llama.cpp/patches, so reset before switching pins.
git -C "$LLAMA_WORKDIR" reset --hard HEAD
git -C "$LLAMA_WORKDIR" clean -fdx
git -C "$LLAMA_WORKDIR" checkout --force --detach "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" reset --hard "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" clean -fdx

printf '%s\n' "$TARGET_SHA" > "$LLAMA_WORKDIR/.mesh-llm-upstream-sha"

if (( ${#PATCHES[@]} > 0 )); then
  (
    # Do not let ambient Git identity/date overrides make the generated
    # patched commit graph job-specific.
    unset \
      GIT_AUTHOR_DATE \
      GIT_AUTHOR_EMAIL \
      GIT_AUTHOR_NAME \
      GIT_COMMITTER_DATE \
      GIT_COMMITTER_EMAIL \
      GIT_COMMITTER_NAME
    # `git am --no-verify` was added after the Git shipped by Debian Bookworm.
    # Disable hooks through configuration instead so container and local Git
    # versions produce the same non-interactive patch application.
    git -c core.hooksPath=/dev/null -C "$LLAMA_WORKDIR" am \
      --3way \
      --committer-date-is-author-date \
      --no-gpg-sign \
      "${PATCHES[@]}"
  )
fi

git -C "$LLAMA_WORKDIR" rev-parse HEAD > "$LLAMA_WORKDIR/.mesh-llm-patched-sha"
printf '%s\n' "$PATCH_DIGEST" > "$LLAMA_WORKDIR/.mesh-llm-patch-digest"
printf '%s\n' "$PREPARE_SCHEMA" > "$LLAMA_WORKDIR/.mesh-llm-prepare-schema"

echo "prepared llama.cpp"
echo "  upstream: $TARGET_SHA"
echo "  patched:  $(cat "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
echo "  workdir:  $LLAMA_WORKDIR"
