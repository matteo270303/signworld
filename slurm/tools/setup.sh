#!/bin/bash
# Login node, nessun costo: ambiente, link ai dati, codice hub e pesi di V-JEPA 2, indice di debug,
# risoluzione delle config di tutti gli esperimenti. Si può rilanciare: salta quello che c'è già.
# Dalla radice del repo: slurm/tools/setup.sh        (DATA_ROOT=... se i dati non sono in $SCRATCH)
#   slurm/tools/setup.sh check [esp...]              solo la risoluzione delle config
set -euo pipefail
cd "$(dirname "$0")/../.."
source slurm/experiments.sh
DATA_ROOT=${DATA_ROOT:-${SCRATCH:-}}
VJEPA_COMMIT=204698b45b3712590f06245fbfba32d3be539812
HUB=checkpoints/hub/facebookresearch_vjepa2_main
DEBUG_INDEX=data/training/youtube_sl25/index.debug.parquet

die() {
  echo "$*" >&2
  exit 1
}

config_args() {  # file di configurazione di un esperimento -> CONFIG=(-c f1 -c f2 ...)
  local file
  CONFIG=()
  while read -r file; do CONFIG+=(-c "$file"); done < <(experiment_files "$1")
}

check() {
  local name resolved
  for name in "$@"; do
    config_args "$name" || exit 1
    resolved=$(uv run --no-sync signworld train config "${CONFIG[@]}" 2>&1) || die "$name: $resolved"
    printf '%-8s %-20s %s\n' "$name" "$(run_dir train "$name")" "${resolved%%$'\n'*}"
  done
}

if [ "${1:-}" = check ]; then
  shift
  [ $# -gt 0 ] || set -- "${EXPERIMENTS[@]}"
  check "$@"
  exit 0
fi

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
[ -f "$DEBUG_INDEX" ] || uv run --no-sync python slurm/tools/make_debug_index.py
mkdir -p out runs/debug
check "${EXPERIMENTS[@]}"
echo "setup finito"
