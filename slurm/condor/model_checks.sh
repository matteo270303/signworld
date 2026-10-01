#!/usr/bin/env bash
# HTCondor job: the model checks of the collaudo on one GPU, each independent of the others.
#   A1 video: weights load whole and our input path reproduces the official tokens
#   A1 pose:  S-JEPA reloads whole and reproduces its reference latents
#   PC6:      V-JEPA 2.1 predictor contents and the zero-initialised multi-level input
# Usage: model_checks.sh <source>
set -uo pipefail

source_name="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
signworld="${repo_dir}/.venv/bin/signworld"
config="${repo_dir}/parameters/analysis/default.yaml"
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
cd "${repo_dir}"

status=0
run() {
  echo "$(date -Is) signworld $*"
  "${signworld}" "$@" -c "${config}" || { echo "$(date -Is) FAILED ($?): signworld $*"; status=1; }
}

run check video-reproduction "${source_name}" --encoder vjepa2_1_vitl --device cuda
run check video-reproduction "${source_name}" --encoder vjepa2_vitl --device cuda
run check pose-reproduction "${source_name}" --device cuda
run check predictor "${source_name}" --encoder vjepa2_1_vitl --device cuda
run check predictor "${source_name}" --encoder vjepa2_1_vitb --device cuda
exit "${status}"
