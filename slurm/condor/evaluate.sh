#!/usr/bin/env bash
# HTCondor job: the test of a trained WorldSign run (§4.12) on one GPU.
# Usage: evaluate.sh <run directory> <index> <split> <checkpoint> <config.yaml> [overlay.yaml ...]
# The report is written to <run directory>/<split>_<checkpoint>.json.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run="$1"; index="$2"; split="$3"; checkpoint="$4"; shift 4
configs=()
for file in "$@"; do configs+=(-c "${repo_dir}/${file}"); done

# Compute nodes are offline: models come from the local caches only.
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "$(date -Is) host=$(hostname) run=${run} split=${split} checkpoint=${checkpoint} configs=$*"
nvidia-smi -L || true
cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/signworld" train evaluate "${configs[@]}" \
  --run "${run}" --index "${index}" --split "${split}" --checkpoint "${checkpoint}" \
  --output "${run}/${split}_${checkpoint}.json" --device cuda
