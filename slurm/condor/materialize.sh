#!/usr/bin/env bash
# HTCondor job: one shard of stage 0 on the whole corpus (crop, 64 frames, pose; §3.6, §4.10).
# Usage: materialize.sh <source> <output root> <shard> <num_shards>
# A shard resumes where it stopped: clips already in its index are skipped.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
echo "$(date -Is) host=$(hostname) source=$1 shard=$3/$4"
cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/signworld" train materialize "$1" --output "$2" --device cuda \
  --shard "$3" --num-shards "$4" -c "${repo_dir}/parameters/analysis/default.yaml"
