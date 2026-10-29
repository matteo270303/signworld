#!/bin/bash
# Prima ondata di ESP-1, in parallelo su 5 nodi: A-gate B0 B C V (slurm/experiments.sh).
# Dalla radice del repo: slurm/batch/wave1.sh        (un solo esperimento: slurm/batch/wave1.sh B)
# Variabili: ACCOUNT QOS TIME SEGMENTS STOP_MARGIN_MIN MAIL ALLOW_CODE_CHANGE (valori in slurm/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
source slurm/experiments.sh
ACCOUNT=${ACCOUNT:-IscrC_SPARTAN}
QOS=${QOS:-boost_qos_lprod}
TIME=${TIME:-4-00:00:00}
SEGMENTS=${SEGMENTS:-6}
mkdir -p out
COMMON=(--account="$ACCOUNT" --parsable)
[ -z "${MAIL:-}" ] || COMMON+=(--mail-type=END,FAIL --mail-user="$MAIL")
NAMES=("${WAVE1[@]}")
[ $# -eq 0 ] || NAMES=("$@")
for name in "${NAMES[@]}"; do experiment_files "$name" > /dev/null || exit 1; done

# Per esperimento una catena di SEGMENTS job da un nodo, legati da afterok; ognuno riprende da latest.pt
# (STOP_MARGIN_MIN minuti prima del walltime il trainer salva e il job esce 0; vedi slurm/README.md).
for name in "${NAMES[@]}"; do
  [ ! -f "$(run_dir train "$name")/FAILED" ] || { echo "$(run_dir train "$name")/FAILED: controlla il log, poi toglilo" >&2; exit 1; }
  previous=""
  for ((k = 1; k <= SEGMENTS; k++)); do
    deps=()
    [ -z "$previous" ] || deps=(--dependency="afterok:$previous" --kill-on-invalid-dep=yes)
    job=$(STOP_MARGIN_MIN=${STOP_MARGIN_MIN:-45} sbatch "${COMMON[@]}" --qos="$QOS" --time="$TIME" \
      --nodes=1 --job-name="sw-$name" --output="out/sw-train-$name-%j.log" "${deps[@]}" \
      slurm/launch_experiment train "$name")
    previous=$job
    echo "$name: segmento $k/$SEGMENTS, job $job"
  done
done
