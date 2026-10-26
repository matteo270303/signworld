#!/bin/bash
# Run di test (arm A) su youtube_sl25: indice (se manca), training, valutazione, report.
# Uso (dalla root del repo): scripts/run_train_ytsl25.sh
# Variabili: GPUS (default 0, l'unica disponibile: batch 32 di prova; con più GPU il batch 128 si divide), ARM (default A),
#            OUT (default runs/youtube_sl25-$ARM). Riprende da solo da latest.pt se rilanciato.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0}
ARM=${ARM:-A}
OUT=${OUT:-runs/youtube_sl25-$ARM}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CFG=(-c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml -c parameters/ablation/arm_$ARM.yaml -c parameters/model/youtube_sl25_run.yaml)
if [ "$NGPU" = 1 ]; then  # 128 clip non entrano in una GPU: prova locale a 32 (non e' la run dell'articolo)
  CFG+=(-c parameters/model/local_single_gpu.yaml)
  echo "ATTENZIONE: una sola GPU, batch 32 (parameters/model/local_single_gpu.yaml)" >&2
fi
INDEX=data/training/youtube_sl25/index.parquet

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [ ! -f "$INDEX" ]; then
  echo "Manca $INDEX: esegui prima F5 di docs/ytsl25-piano-preprocessing.md" >&2; exit 1
fi

CUDA_VISIBLE_DEVICES=$GPUS uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
  --no-python .venv/bin/signworld train run "${CFG[@]}" --output "$OUT"

CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run --no-sync signworld train evaluate "${CFG[@]}" \
  --run "$OUT" --index "$INDEX" --split test_channel --output "$OUT/evaluation_test_channel.json"
uv run --no-sync signworld train report --run "$OUT" \
  --evaluation "$OUT/evaluation_test_channel.json" --split test_channel
