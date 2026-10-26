#!/bin/bash
# YouTube-SL-25 dall'inizio alla fine: i comandi di README_youtube_sl25.md (§1-§5.1) in successione.
#
# Uso (dalla root del repo, in tmux):
#   scripts/pipeline_ytsl25.sh                                   # baseline, negatives, smoke, train (arm A)
#   ARMS="A B0 B C" scripts/pipeline_ytsl25.sh                   # piu' bracci, uno dopo l'altro
#   STEPS="download refresh baseline negatives smoke train" scripts/pipeline_ytsl25.sh   # tutto
#   STEPS=train ARMS=C scripts/pipeline_ytsl25.sh                # un solo passo
#
# Passi (in quest'ordine, qualunque sia l'ordine in STEPS):
#   download   §1   scripts/complete_ytsl25.sh download   (giorni; servono cookie freschi)
#   refresh    §2   scripts/complete_ytsl25.sh refresh    (manifest, embedding, clip e pose, indice, check)
#   baseline   §4   train ridge-baseline  -> reports/youtube_sl25/ridge_baseline.json
#   negatives  §4   text false-negatives  -> data/training/youtube_sl25/false_negatives
#   smoke      §4   train overfit + train benchmark (collaudo, PC7)
#   train      §5.1 train run + evaluate su test_channel + report, per ogni braccio in ARMS
# download e refresh sono esclusi di default: il corpus attuale (34 %) e' gia' indicizzato e refresh
# rifa' text embed per intero. baseline, negatives e smoke si saltano se il loro output esiste gia'.
#
# Invece di modificare a mano youtube_sl25.yaml, i risultati di baseline e negatives finiscono in un
# overlay generato (tmp/ytsl25_auto.yaml: diagnostics.ridge_baseline, losses.false_negatives) messo
# dopo youtube_sl25_run.yaml. Una run gia' avviata senza overlay non si riprende (ProvenanceError):
# usa un altro OUT_PREFIX o archiviala in runs/archived/.
# Variabili: GPUS (0), ARMS (A), OUT_PREFIX (runs/youtube_sl25), WORKERS (passato a complete_ytsl25.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-baseline negatives smoke train}
ARMS=${ARMS:-A}
GPUS=${GPUS:-0}
GPU0=${GPUS%%,*}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT_PREFIX=${OUT_PREFIX:-runs/youtube_sl25}
DATA=data/youtube-sl-25
TRAIN=data/training/youtube_sl25
INDEX=$TRAIN/index.parquet
RIDGE=reports/youtube_sl25/ridge_baseline.json
NEGATIVES=$TRAIN/false_negatives
AUTO=tmp/ytsl25_auto.yaml
BASE=(-c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml)
RUN=(-c parameters/model/youtube_sl25_run.yaml)
mkdir -p logs tmp reports/youtube_sl25

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

want() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo; echo "=== [$(date '+%F %T')] $* ==="; }
# Lo store di embedding usato da youtube_sl25_run.yaml (data.embeddings).
embeddings() { sed -n 's/^ *embeddings: *//p' parameters/model/youtube_sl25_run.yaml | head -1; }

# L'overlay con quello che baseline e negatives hanno prodotto finora.
write_auto() {
  {
    if [ -f "$RIDGE" ]; then
      uv run --no-sync python -c 'import json,sys; b=json.load(open(sys.argv[1]))["ridge_baseline"]; print(f"diagnostics:\n  ridge_baseline: {{t2v: {b[\"t2v\"]}, v2t: {b[\"v2t\"]}}}")' "$RIDGE"
    fi
    if [ -d "$NEGATIVES" ]; then
      printf 'losses:\n  false_negatives: %s\n' "$NEGATIVES"
    fi
  } > "$AUTO"
}

EXTRA=(-c "$AUTO")
if [ "$NGPU" = 1 ]; then  # 128 clip non entrano in una GPU: prova a 32, non e' la run dell'articolo
  EXTRA+=(-c parameters/model/local_single_gpu.yaml)
  echo "ATTENZIONE: una sola GPU, batch 32 (parameters/model/local_single_gpu.yaml)" >&2
fi
arm_cfg() { CFG=("${BASE[@]}" -c "parameters/ablation/arm_$1.yaml" "${RUN[@]}" "${EXTRA[@]}"); }

# La configurazione di ogni braccio, prima di qualunque passo. L'overlay $AUTO e' quello attuale:
# se baseline o negatives lo cambiano, train run ristampa la configurazione definitiva.
if want train; then
  write_auto
  for ARM in $ARMS; do
    log "configurazione, braccio $ARM"
    arm_cfg "$ARM"
    uv run --no-sync signworld train config "${CFG[@]}"
  done
fi

if want download; then
  log "§1 download"
  scripts/complete_ytsl25.sh download
fi

if want refresh; then
  log "§2 refresh"
  scripts/complete_ytsl25.sh refresh
  STORE=$(ls -td $DATA/text/*/ | head -1)
  if [ "${STORE%/}" != "$(embeddings)" ]; then
    echo "ERRORE: nuovo fingerprint ${STORE%/}: aggiorna data.embeddings in parameters/model/youtube_sl25_run.yaml e rilancia senza refresh" >&2
    exit 1
  fi
fi

if [ ! -f "$INDEX" ]; then echo "Manca $INDEX: lancia con STEPS=refresh" >&2; exit 1; fi

if want baseline; then
  if [ -f "$RIDGE" ]; then
    log "§4 ridge-baseline: $RIDGE esiste, salto"
  else
    log "§4 ridge-baseline (PC2)"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run --no-sync signworld train ridge-baseline \
      "${BASE[@]}" -c parameters/ablation/arm_A.yaml "${RUN[@]}" \
      --split val_channel --split test_channel --clips 5000 --device cuda \
      --output "$RIDGE" 2>&1 | tee -a logs/ridge_baseline.log
  fi
fi

if want negatives; then
  if [ -d "$NEGATIVES" ]; then
    log "§4 false-negatives: $NEGATIVES esiste, salto"
  else
    log "§4 false-negatives (arm C)"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run --no-sync signworld text false-negatives \
      --index "$INDEX" --embeddings "$(embeddings)" \
      --output "$NEGATIVES" --device cuda 2>&1 | tee -a logs/false_negatives.log
  fi
fi

write_auto

if want smoke; then
  if [ -f "$OUT_PREFIX-overfit.json" ]; then
    log "§4 overfit: $OUT_PREFIX-overfit.json esiste, salto"
  else
    log "§4 overfit (collaudo §4.13.1)"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run --no-sync signworld train overfit "${BASE[@]}" "${RUN[@]}" \
      --output "$OUT_PREFIX-overfit.json" 2>&1 | tee -a logs/overfit.log
  fi
  if [ -e "$OUT_PREFIX-benchmark" ]; then
    log "§4 benchmark: $OUT_PREFIX-benchmark esiste, salto"
  else
    log "§4 benchmark (PC7)"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run --no-sync signworld train benchmark "${BASE[@]}" "${RUN[@]}" \
      --output "$OUT_PREFIX-benchmark" 2>&1 | tee -a logs/benchmark.log
  fi
fi

if want train; then
  for ARM in $ARMS; do
    ARM_CFG=parameters/ablation/arm_$ARM.yaml
    OUT=$OUT_PREFIX-$ARM
    if [ ! -f "$ARM_CFG" ]; then echo "ERRORE: manca $ARM_CFG" >&2; exit 1; fi
    if [ "$ARM" = C ] && [ ! -d "$NEGATIVES" ]; then
      echo "ERRORE: arm C richiede $NEGATIVES (passo negatives)" >&2; exit 1
    fi
    arm_cfg "$ARM"

    log "§5.1 train run, braccio $ARM -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPUS uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" \
      --no-python .venv/bin/signworld train run "${CFG[@]}" --output "$OUT" 2>&1 | tee -a "logs/train_$ARM.log"

    log "§5.1 evaluate su test_channel, braccio $ARM"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run --no-sync signworld train evaluate "${CFG[@]}" \
      --run "$OUT" --index "$INDEX" --split test_channel --output "$OUT/evaluation_test_channel.json"

    log "§5.1 report, braccio $ARM"
    REPORT=(--run "$OUT" --evaluation "$OUT/evaluation_test_channel.json" --split test_channel)
    [ -f "$RIDGE" ] && REPORT+=(--ridge "$RIDGE")
    uv run --no-sync signworld train report "${REPORT[@]}"
  done
fi

log "fatto: $STEPS (bracci: $ARMS)"
