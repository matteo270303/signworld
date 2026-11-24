#!/bin/bash
# Debug del carico su boost_qos_dbg: un nodo per esperimento (max 8), due segmenti da 30 min,
# il secondo riprende il primo (afterany). Poi: python slurm/tools/run_report.py debug
# Dalla radice del repo: slurm/batch/debug.sh [esp...]       (default: DEBUG_SET di experiments.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
source slurm/experiments.sh
ACCOUNT=${ACCOUNT:-IscrC_SPARTAN}
[ $# -gt 0 ] || set -- "${DEBUG_SET[@]}"
[ $# -le 8 ] || { echo "boost_qos_dbg: al massimo 8 nodi per job" >&2; exit 1; }
for name in "$@"; do experiment_files "$name" > /dev/null || exit 1; done
[ -f data/training/youtube_sl25/index.debug.parquet ] \
  || { echo "manca l'indice di debug: lancia slurm/tools/setup.sh" >&2; exit 1; }
mkdir -p runs/debug/logs  # log del job (più run): runs/debug/logs; quelli dei run in <run>/logs
COMMON=(--account="$ACCOUNT" --parsable)
[ -z "${MAIL:-}" ] || COMMON+=(--mail-type=END,FAIL --mail-user="$MAIL")
args=("${COMMON[@]}" --qos=boost_qos_dbg --time=00:30:00 --nodes="$#"
  --job-name=sw-debug --output=runs/debug/logs/debug-%j.log)
first=$(STOP_MARGIN_MIN=6 sbatch "${args[@]}" slurm/launch_experiment debug "$@")
second=$(STOP_MARGIN_MIN=6 sbatch "${args[@]}" --dependency="afterany:$first" \
  slurm/launch_experiment debug "$@")
echo "debug di $*: job $first, poi $second (riprende); esiti e log in runs/debug/<esp>-$RUN_STAMP/logs"
echo "per riprendere: RUN_STAMP=$RUN_STAMP slurm/batch/debug.sh $*"
