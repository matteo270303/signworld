#!/usr/bin/env bash
# HTCondor job: run one signworld analysis command (manifests, checks, embeddings).
# Usage: analysis.sh <signworld arguments...>
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
signworld="${repo_dir}/.venv/bin/signworld"

if ! "${repo_dir}/.venv/bin/python" -c '' 2>/dev/null; then
  echo "error: no working environment in ${repo_dir}/.venv; run 'uv sync' there first" >&2
  exit 2
fi

# Gated models (EmbeddingGemma) need the token; it never lives in the repository.
if [[ -r "${HOME}/.secrets/hf_token" ]]; then
  HF_TOKEN="$(cat "${HOME}/.secrets/hf_token")"
  export HF_TOKEN
fi
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"

echo "$(date -Is) host=$(hostname) command: $*"
cd "${repo_dir}"
exec "${signworld}" "$@"
