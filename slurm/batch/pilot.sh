#!/bin/bash
# Pilota di costo e di gate (parameters/model/pilot.yaml): braccio A, 1 nodo, 4 GPU, 1 h 45, QOS normal.
# Dalla radice del repo: slurm/batch/pilot.sh        (altro account: ACCOUNT=... slurm/batch/pilot.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT=${ACCOUNT:-IscrC_BRIDGEV}
mkdir -p out runs
sbatch --account="$ACCOUNT" --qos=normal --time=01:45:00 --nodes=1 --job-name=sw-pilot \
  --output=out/sw-pilot-%j.log slurm/launch_experiment train pilot
