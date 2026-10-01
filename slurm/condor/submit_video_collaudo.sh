#!/usr/bin/env bash
# Submit, all in parallel, every GPU job of the video collaudo on one dataset:
# the four feature runs (sharded) and the model checks. The read-outs (CPU) follow once the
# feature jobs have finished; the command is printed at the end.
# Usage: slurm/condor/submit_video_collaudo.sh [dataset] [shards]
set -euo pipefail

dataset="${1:-youtube_sl25}"
shards="${2:-4}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/condor

for run in vjepa2_1_vitl-256 vjepa2_vitl-256 vjepa2_1_vitl-384; do
  condor_submit -name ettore -a "dataset=${dataset}" -a "run=${run}" -a "shards=${shards}" \
    slurm/condor/video_features.sub
done
# 1 000 clips only: one shard is enough.
condor_submit -name ettore -a "dataset=${dataset}" -a "run=vjepa2_1_vitl-256-contiguous" \
  -a "shards=1" slurm/condor/video_features.sub
condor_submit -name ettore -a "arguments = ${dataset}" slurm/condor/model_checks.sub

cat <<NEXT

When every features_* job has closed with a shard-*.npz path in its .out:
  condor_submit -name ettore \\
    -a 'arguments = experiment video-probes ${dataset} -c parameters/analysis/default.yaml' \\
    -a 'job=video_probes' -a 'cpus=16' -a 'memory=64 GB' slurm/condor/analysis.sub
NEXT
