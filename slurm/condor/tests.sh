#!/usr/bin/env bash
# HTCondor job: the whole test suite, the slow tests included (minutes on CPU each: a short
# run of the trainer through every stage, the DDP run, P13), on one GPU node.
# Usage: tests.sh <report directory>
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
report="$1"
mkdir -p "${report}"
export SIGNWORLD_SLOW_TESTS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
echo "$(date -Is) host=$(hostname)"
nvidia-smi -L || true
cd "${repo_dir}"
"${repo_dir}/.venv/bin/python" -m pytest -q -p no:cacheprovider \
  --junitxml="${report}/tests.xml" --durations=25 tests
