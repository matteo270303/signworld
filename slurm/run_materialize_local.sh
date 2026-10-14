#!/bin/bash
# Stage 0 (clip + pose) su tutto un dataset, in locale, su UNA GPU, con N processi in parallelo.
#
# Perche' piu' processi: un processo usa ~1,4 GB di GPU e ~3 core, ma chiama RTMW un frame alla
# volta (batch 1): e' limitato dalla latenza di ogni chiamata, non dalla potenza della GPU.
# Piu' processi sovrappongono decodifica, ridimensionamento e scrittura di uno con l'inferenza
# degli altri. Ogni processo tiene 20-25 GB di RAM: il numero e' limitato anche da quella.
# Misurato il 14/10/2026 su GPU 0 (RTX PRO 6000, condivisa con altri job al 99 %), 499 GB di RAM:
#   1 processo  ~0,19 clip/s | 8 processi ~1,25 clip/s | 16 processi ~1,85 clip/s (384 GB di RAM)
# Default 12: lascia margine di RAM per tutto il giorno; 16 e' il massimo su questa macchina.
#
# Si riprende da solo: le clip gia' negli indici (di qualunque shard) vengono saltate, quindi si puo'
# rilanciare, anche cambiando WORKERS. Prima di ogni avvio toglie l'ultima riga rotta degli indici.
#
# Uso (dalla root del repo, meglio in tmux o con nohup):
#   slurm/run_materialize_local.sh
#   GPU=0 WORKERS=16 slurm/run_materialize_local.sh
#   SOURCE=youtube_sl25 OUTPUT=data/training/youtube_sl25 slurm/run_materialize_local.sh
#
# Variabili: SOURCE, OUTPUT, GPU (default 0), WORKERS (default 12), CONFIG, RAM_PER_WORKER_GB (28),
#            STAGGER_S (20, pausa tra un avvio e l'altro: evita il picco di RAM al caricamento),
#            MAX_RETRIES (3, per processo, se muore con errore).
# Ctrl+C ferma tutto e sistema gli indici.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SOURCE=${SOURCE:-youtube_sl25}
OUTPUT=${OUTPUT:-data/training/youtube_sl25}
GPU=${GPU:-0}
WORKERS=${WORKERS:-12}
CONFIG=${CONFIG:-parameters/analysis/default.yaml}
RAM_PER_WORKER_GB=${RAM_PER_WORKER_GB:-28}
STAGGER_S=${STAGGER_S:-20}
MAX_RETRIES=${MAX_RETRIES:-3}
LOGS=${LOGS:-logs}

mkdir -p "$LOGS" "$OUTPUT"

if pgrep -af "train materialize.*--output $OUTPUT" > /dev/null; then
  echo "ERRORE: c'e' gia' un 'train materialize' su $OUTPUT (due run sulla stessa clip corrompono l'indice):" >&2
  pgrep -af "train materialize.*--output $OUTPUT" >&2
  exit 1
fi

# Limite dalla RAM disponibile.
available_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
max_by_ram=$((available_gb / RAM_PER_WORKER_GB))
if ((WORKERS > max_by_ram)); then
  echo "RAM disponibile ${available_gb} GB: riduco WORKERS da $WORKERS a $max_by_ram"
  WORKERS=$max_by_ram
fi
((WORKERS >= 1)) || { echo "ERRORE: RAM insufficiente" >&2; exit 1; }

# Thread per processo: evita che N processi si contendano tutti i core.
cores=$(nproc)
THREADS=$((cores / WORKERS))
((THREADS >= 1)) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export OPENCV_FOR_THREADS_NUM=$THREADS

# Toglie l'ultima riga di un indice se manca il newline finale o non e' JSON valido
# (la clip verra' rifatta). Argomento: un file indice.
repair_index() {
  uv run --no-sync python -I - "$1" <<'EOF'
import json, sys
path = sys.argv[1]
try:
    data = open(path, "rb").read()
except FileNotFoundError:
    sys.exit(0)
if not data:
    sys.exit(0)
lines = data.split(b"\n")
tail = lines[-1]            # vuoto se il file finisce con newline
last = tail if tail else (lines[-2] if len(lines) > 1 else b"")
bad = bool(tail)
if not bad:
    try:
        json.loads(last)
    except ValueError:
        bad = True
if bad:
    keep = lines[:-1] if tail else lines[:-2]
    open(path, "wb").write(b"\n".join(keep) + (b"\n" if keep else b""))
    print(f"indice riparato: tolta l'ultima riga di {path}")
EOF
}

repair_all() {
  for f in "$OUTPUT"/index*.jsonl; do [ -e "$f" ] && repair_index "$f"; done
}

count_done() {
  cat "$OUTPUT"/index*.jsonl 2>/dev/null | wc -l
}

# Un processo: riparte da solo se muore con errore (es. memoria), fino a MAX_RETRIES.
run_worker() {
  local shard=$1 label
  label=$(printf "shard-%05d-of-%05d" "$shard" "$WORKERS")
  local attempt=0
  while :; do
    repair_index "$OUTPUT/index.$label.jsonl"
    CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync signworld train materialize "$SOURCE" \
      --output "$OUTPUT" --device cuda --shard "$shard" --num-shards "$WORKERS" -c "$CONFIG" \
      >> "$LOGS/materialize_${WORKERS}_${shard}.log" 2>&1
    local rc=$?
    ((rc == 0)) && return 0
    attempt=$((attempt + 1))
    echo "[$(date +%T)] worker $shard terminato con codice $rc (tentativo $attempt/$MAX_RETRIES)"
    ((attempt >= MAX_RETRIES)) && return "$rc"
    sleep 10
  done
}

pids=()
cleanup() {
  trap - INT TERM
  echo; echo "Interrotto: fermo i processi..."
  # Prima i cicli di retry (altrimenti rilanciano i processi appena uccisi), poi i processi.
  kill -TERM "${pids[@]}" 2>/dev/null
  pkill -TERM -f "train materialize.*--output $OUTPUT" 2>/dev/null
  wait "${pids[@]}" 2>/dev/null
  repair_all
  echo "Indici controllati. Rilancia lo stesso comando per riprendere."
  exit 130
}
trap cleanup INT TERM

repair_all
start_done=$(count_done)
start_time=$(date +%s)
echo "Sorgente $SOURCE -> $OUTPUT | GPU $GPU | $WORKERS processi, $THREADS thread ciascuno"
echo "Clip gia' fatte: $start_done | log: $LOGS/materialize_${WORKERS}_<n>.log"

for ((i = 0; i < WORKERS; i++)); do
  run_worker "$i" &
  pids+=($!)
  if ((i < WORKERS - 1)); then
    sleep "$STAGGER_S" &
    wait $!
  fi
done

# Avanzamento ogni minuto finche' qualcuno e' vivo.
while kill -0 "${pids[@]}" 2>/dev/null; do
  sleep 60 &  # in background + wait: il trap di Ctrl+C scatta subito, non dopo il sleep
  wait $!
  now=$(count_done)
  elapsed=$(($(date +%s) - start_time))
  rate=$(awk -v n="$((now - start_done))" -v t="$elapsed" 'BEGIN {r = (t > 0) ? n / t : 0; printf "%.2f", r}')
  echo "[$(date +%T)] $now clip fatte, $rate clip/s dall'avvio"
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
repair_all
echo "Finito. Clip nell'indice: $(count_done)"
grep -h "without a signer" "$LOGS"/materialize_${WORKERS}_*.log 2>/dev/null | tail -n "$WORKERS"
((status == 0)) || echo "ATTENZIONE: almeno un processo e' terminato con errore, guarda i log."
echo "Prossimo passo: uv run --no-sync signworld train index (F5 di docs/ytsl25-piano-preprocessing.md)"
exit $status
