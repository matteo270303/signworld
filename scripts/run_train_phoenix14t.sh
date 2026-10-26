#!/bin/bash
# Run di test (arm A) su phoenix14t: indice (se manca), training, valutazione, report.
# Uso (dalla root del repo): scripts/run_train_phoenix14t.sh
# Variabili: GPUS (default 0, l'unica disponibile: batch 32 di prova; con più GPU il batch 128 si divide), ARM (default A),
#            OUT (default runs/phoenix14t-$ARM). Riprende da solo da latest.pt se rilanciato.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0}
ARM=${ARM:-A}
OUT=${OUT:-runs/phoenix14t-$ARM}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CFG=(-c parameters/model/worldsign.yaml -c parameters/ablation/arm_$ARM.yaml -c parameters/model/phoenix14t_trial.yaml)
if [ "$NGPU" = 1 ]; then  # 128 clip non entrano in una GPU: prova locale a 32 (non e' la run dell'articolo)
  CFG+=(-c parameters/model/local_single_gpu.yaml)
  echo "ATTENZIONE: una sola GPU, batch 32 (parameters/model/local_single_gpu.yaml)" >&2
fi
INDEX=data/training/phoenix14t/index.parquet

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Ricostruito anche se non ha lo split di test: un indice fatto senza l'overlay del dataset ha
# gli split per canale (train/val_video/val_channel) invece di quelli ufficiali.
if [ ! -f "$INDEX" ] || ! uv run --no-sync python -c "import sys, pyarrow.parquet as pq; sys.exit('test' not in pq.read_table(sys.argv[1], columns=['split']).column(0).to_pylist())" "$INDEX"; then
  uv run --no-sync signworld train index --manifest data/phoenix14t/manifest/clips.parquet \
    --materialized data/training/phoenix14t --embeddings data/phoenix14t/text/2fdc70f119730c4c \
    --output $INDEX "${CFG[@]}"
fi

CUDA_VISIBLE_DEVICES=$GPUS uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
  --no-python .venv/bin/signworld train run "${CFG[@]}" --output "$OUT"

CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run --no-sync signworld train evaluate "${CFG[@]}" \
  --run "$OUT" --index "$INDEX" --split test --output "$OUT/evaluation_test.json"
uv run --no-sync signworld train report --run "$OUT" \
  --evaluation "$OUT/evaluation_test.json" --split test
