#!/usr/bin/env bash
# HTCondor job: the pose checks of the collaudo on the test clips of one dataset, which
# testdata_build.sub must have produced (datiTest).
#   1. A3: frame selection reproduced, then poses re-estimated on the decoded frames
#   2. A4: shoulders, boxes and keypoints outside the frame
# Usage: pose_collaudo.sh <source>
set -uo pipefail

source_name="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
signworld="${repo_dir}/.venv/bin/signworld"
config="${repo_dir}/parameters/analysis/default.yaml"
cd "${repo_dir}"

run() {
  echo "$(date -Is) signworld $*"
  "${signworld}" "$@" -c "${config}" || echo "$(date -Is) FAILED ($?): signworld $*"
}

run check frame-selection "${source_name}"
run check pose-alignment "${source_name}" --device cuda
run check pose-quality "${source_name}"
