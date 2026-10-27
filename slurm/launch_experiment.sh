#!/bin/bash
# Lancio dei run WorldSign su Leonardo (Booster: 4 A100 64 GB per nodo), dalla radice del repo.
#
#   slurm/launch_experiment.sh setup              login node: ambiente, dati, pesi, indice di debug
#   slurm/launch_experiment.sh check [esp...]     risolve le config (nessun costo)
#   slurm/launch_experiment.sh debug [esp...]     boost_qos_dbg: un nodo per esperimento (max 8),
#                                                 due segmenti da 30 min, il secondo riprende il primo
#   slurm/launch_experiment.sh report             memoria e s/passo per stage del debug: OK o RISCHIO OOM
#   slurm/launch_experiment.sh train esp...       per esperimento una catena di SEGMENTS job da un nodo;
#                                                 ognuno riprende da latest.pt
#   slurm/launch_experiment.sh status             job in coda e ultimo passo di ogni run
#   slurm/launch_experiment.sh cancel esp...      cancella i job di un esperimento
#
# Esperimenti (slurm/experiments.sh): A-gate A B0 B C V, e su θ* esp2 esp4 esp6 g1 d4.
# `wave1` sta per A-gate B0 B C V.
#
# Variabili (valori di default):
#   ACCOUNT=IscrC_SPARTAN  QOS=boost_qos_lprod  TIME=4-00:00:00  SEGMENTS=6
#   STOP_MARGIN_MIN=45     minuti prima del walltime in cui il run salva latest e si ferma
#   THETA=A                braccio sotto le ablation su θ*
#   DATA_ROOT=$SCRATCH     dove scripts/sync_dataset.sh ha copiato youtube-sl-25/ e training/
#   MAIL=                  indirizzo per le mail di fine/fallimento (vuoto: nessuna mail)
#   ALLOW_CODE_CHANGE=0    1: riprende anche se il codice è cambiato dal primo lancio (registrato)
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=experiments.sh
source slurm/experiments.sh
ACCOUNT=${ACCOUNT:-IscrC_SPARTAN}
QOS=${QOS:-boost_qos_lprod}
TIME=${TIME:-4-00:00:00}
SEGMENTS=${SEGMENTS:-6}
DATA_ROOT=${DATA_ROOT:-${SCRATCH:-}}
VJEPA_COMMIT=204698b45b3712590f06245fbfba32d3be539812
HUB=checkpoints/hub/facebookresearch_vjepa2_main
DEBUG_INDEX=data/training/youtube_sl25/index.debug.parquet

die() {
  echo "$*" >&2
  exit 1
}

select_experiments() {  # NAMES = argomenti, con wave1 espanso; si ferma su un nome sconosciuto
  local name
  NAMES=()
  for name in "$@"; do
    if [ "$name" = wave1 ]; then
      NAMES+=("${WAVE1[@]}")
    else
      experiment_files "$name" > /dev/null || exit 1
      NAMES+=("$name")
    fi
  done
}

config_args() {  # file di configurazione di un esperimento -> CONFIG=(-c f1 -c f2 ...)
  local file
  CONFIG=()
  while read -r file; do CONFIG+=(-c "$file"); done < <(experiment_files "$1")
}

sbatch_common() {
  COMMON=(--account="$ACCOUNT" --parsable)
  [ -z "${MAIL:-}" ] || COMMON+=(--mail-type=END,FAIL --mail-user="$MAIL")
}

check() {
  local name resolved
  for name in "$@"; do
    config_args "$name"
    resolved=$(uv run --no-sync signworld train config "${CONFIG[@]}" 2>&1) || die "$name: $resolved"
    printf '%-8s %-20s %s\n' "$name" "$(run_dir train "$name")" "${resolved%%$'\n'*}"
  done
}

setup() {
  command -v uv > /dev/null || die "manca uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
  uv sync --frozen
  [ -n "$DATA_ROOT" ] || die "DATA_ROOT vuoto: imposta SCRATCH o DATA_ROOT"
  for dir in youtube-sl-25 training; do
    [ -e "data/$dir" ] || ln -s "$DATA_ROOT/$dir" "data/$dir"
  done
  [ -f data/training/youtube_sl25/index.parquet ] \
    || die "manca $DATA_ROOT/training/youtube_sl25/index.parquet: copia i dati con scripts/sync_dataset.sh"
  [ -d data/youtube-sl-25/text ] || die "manca $DATA_ROOT/youtube-sl-25/text"
  if [ ! -d "$HUB" ]; then
    git clone https://github.com/facebookresearch/vjepa2 "$HUB"
    git -C "$HUB" checkout --quiet "$VJEPA_COMMIT"
  fi
  # I pesi di V-JEPA 2.1 (~5 GB) dal rilascio di Meta, con il controllo dello SHA-256 della config.
  config_args A
  uv run --no-sync signworld train fetch-models "${CONFIG[@]}"
  [ -f "$DEBUG_INDEX" ] || uv run --no-sync python slurm/make_debug_index.py
  mkdir -p out runs/debug
  check "${EXPERIMENTS[@]}"
  echo "setup finito"
}

debug() {
  local first second
  [ $# -gt 0 ] || set -- "${DEBUG_SET[@]}"
  select_experiments "$@"
  local names=("${NAMES[@]}")
  [ "${#names[@]}" -le 8 ] || die "boost_qos_dbg: al massimo 8 nodi per job"
  [ -f "$DEBUG_INDEX" ] || die "manca $DEBUG_INDEX: lancia prima setup"
  mkdir -p out
  sbatch_common
  local args=("${COMMON[@]}" --qos=boost_qos_dbg --time=00:30:00 --nodes="${#names[@]}"
    --job-name=sw-debug --output=out/sw-debug-%j.log)
  first=$(STOP_MARGIN_MIN=6 sbatch "${args[@]}" slurm/experiment_job debug "${names[@]}")
  second=$(STOP_MARGIN_MIN=6 sbatch "${args[@]}" --dependency="afterany:$first" \
    slurm/experiment_job debug "${names[@]}")
  echo "debug di ${names[*]}: job $first, poi $second (riprende); esiti in runs/debug/, poi: $0 report"
}

train() {
  local name previous job k
  [ $# -gt 0 ] || die "uso: $0 train <esp>... (o wave1)"
  select_experiments "$@"
  local names=("${NAMES[@]}")
  mkdir -p out
  sbatch_common
  for name in "${names[@]}"; do
    [ ! -f "$(run_dir train "$name")/FAILED" ] || die "$(run_dir train "$name")/FAILED: controlla il log, poi toglilo"
    previous=""
    for ((k = 1; k <= SEGMENTS; k++)); do
      local deps=()
      [ -z "$previous" ] || deps=(--dependency="afterok:$previous" --kill-on-invalid-dep=yes)
      job=$(STOP_MARGIN_MIN=${STOP_MARGIN_MIN:-45} sbatch "${COMMON[@]}" --qos="$QOS" --time="$TIME" \
        --nodes=1 --job-name="sw-$name" --output="out/sw-train-$name-%j.log" "${deps[@]}" \
        slurm/experiment_job train "$name")
      previous=$job
      echo "$name: segmento $k/$SEGMENTS, job $job"
    done
  done
}

case ${1:-} in
  setup) setup ;;
  check)
    shift
    [ $# -gt 0 ] || set -- "${EXPERIMENTS[@]}"
    select_experiments "$@"
    check "${NAMES[@]}"
    ;;
  debug)
    shift
    debug "$@"
    ;;
  report) uv run --no-sync python slurm/run_report.py debug ;;
  train)
    shift
    train "$@"
    ;;
  status)
    squeue -u "$USER" -o "%.10i %.14j %.16q %.8T %.11M %.12l %.5D %R"
    uv run --no-sync python slurm/run_report.py status
    ;;
  cancel)
    shift
    [ $# -gt 0 ] || die "uso: $0 cancel <esp>..."
    select_experiments "$@"
    for name in "${NAMES[@]}"; do scancel --name="sw-$name"; done
    ;;
  *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0" ;;
esac
