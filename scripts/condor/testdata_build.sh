#!/usr/bin/env bash
# HTCondor job: one shard of the test clips (cut, crop, frame selection, pose) under datiTest.
# Usage: testdata_build.sh <source> <shard> <num_shards>
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/signworld" testdata build "$1" --device cuda \
  --shard "$2" --num-shards "$3" -c "${repo_dir}/configs/analysis.yaml"
