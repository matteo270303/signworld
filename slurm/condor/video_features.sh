#!/usr/bin/env bash
# HTCondor job: one shard of one frozen-video feature run (PC2, PC3, PC4, frame collaudo).
# Usage: video_features.sh <source> <run> <shard> <num_shards>
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Compute nodes are offline: models come from the local caches only.
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
echo "$(date -Is) host=$(hostname) run=$2 shard=$3/$4"
cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/signworld" experiment video-features "$1" --run "$2" \
  --shard "$3" --num-shards "$4" --device cuda -c "${repo_dir}/parameters/analysis/default.yaml"
