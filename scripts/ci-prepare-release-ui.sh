#!/usr/bin/env bash
# The same release preparation runs in every UI producer's pinned container.
# Ordinary CI rehearses in a disposable clone; its product source is untouched.
set -euo pipefail
source_sha="${1:?expected immutable source SHA}"
release_tag="${2:-}"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid UI source SHA" >&2; exit 1; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Checkout's temporary HOME does not survive into subsequent container steps.
# Scope Git trust to this checkout, never a wildcard or a persistent runner HOME.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$root"
test "$(git -C "$root" rev-parse HEAD)" = "$source_sha"

if [[ -z "$release_tag" ]]; then
    rehearsal="$(mktemp -d "${RUNNER_TEMP:-/tmp}/mesh-release-ui.XXXXXX")"
    trap 'rm -rf -- "$rehearsal"' EXIT
    git clone --quiet --shared --no-checkout "$root" "$rehearsal/source"
    git -c safe.directory="$rehearsal/source" -C "$rehearsal/source" checkout --quiet --detach "$source_sha"
    # Invoke the exact release branch, with no publishing token or remote writes.
    GITHUB_ENV="$rehearsal/github-env" \
        "$rehearsal/source/scripts/ci-prepare-release-ui.sh" "$source_sha" v99.99.99-ci-rehearsal
    exit 0
fi

cd "$root"
scripts/release-version.sh "$release_tag"
echo "VITE_MESH_LLM_DEBUG_UI=false" >> "${GITHUB_ENV:?expected GitHub environment file}"
