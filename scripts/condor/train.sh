#!/usr/bin/env bash
# HTCondor job: one WorldSign training run on the GPUs of one node, one process per GPU.
# Usage: train.sh <run directory> <gpus> <config.yaml> [overlay.yaml ...]
# A run directory that already has checkpoints/latest.pt is resumed, so the same submission
# can be repeated after a pre-emption.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="$1"; gpus="$2"; shift 2
configs=()
for file in "$@"; do configs+=(-c "${repo_dir}/${file}"); done

# Compute nodes are offline: models come from the local caches only.
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
# Data-loader processes do the decoding; keep each process's own threads few.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Less fragmentation of the GPU memory: PyTorch's own advice when the trial ran out of it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "$(date -Is) host=$(hostname) gpus=${gpus} output=${output} configs=$*"
nvidia-smi -L || true
cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${gpus}" \
  --no-python "${repo_dir}/.venv/bin/signworld" train run "${configs[@]}" --output "${output}"
