#!/usr/bin/env bash
# Ogni INTERVAL_S secondi controlla che run_materialize_local.sh giri; se si e' fermato con errore lo
# rilancia (stessi parametri). Finisce quando il run termina senza errori ("Finito" senza "ATTENZIONE").
# Si ferma anche dopo MAX_STALLS rilanci di fila senza nuove clip nell'indice (errore che si ripete).
# Uso, dalla root del repo: GPU=0 WORKERS=16 setsid nohup scripts/watch_materialize.sh > logs/watch_materialize.log 2>&1 &
set -u
INTERVAL_S=${INTERVAL_S:-1800}
MAX_STALLS=${MAX_STALLS:-3}
OUTPUT=${OUTPUT:-data/training/youtube_sl25}
RUNLOG=logs/run_materialize.log
mkdir -p logs
count() { cat "$OUTPUT"/index*.jsonl 2>/dev/null | wc -l; }
running() { pgrep -f "scripts/run_materialize_local.sh" >/dev/null; }
last=$(count); stalls=0
echo "[$(date +%T)] monitor attivo, clip nell'indice: $last"
while :; do
  sleep "$INTERVAL_S"
  now=$(count)
  if running; then
    echo "[$(date +%T)] in corso, clip: $now (+$((now - last)))"
    last=$now; stalls=0; continue
  fi
  if grep -q "^Finito" "$RUNLOG" 2>/dev/null && ! grep -q "^ATTENZIONE" "$RUNLOG"; then
    echo "[$(date +%T)] estrazione finita senza errori, clip: $now. Prossimo passo: train index (F5)"; exit 0
  fi
  if ((now > last)); then stalls=0; else stalls=$((stalls + 1)); fi
  if ((stalls >= MAX_STALLS)); then
    echo "[$(date +%T)] fermo $stalls volte di fila senza progresso, mi fermo: guarda $RUNLOG"; exit 1
  fi
  echo "[$(date +%T)] run fermo (clip $now), lo rilancio"
  pgrep -af "train materialize" | grep -q -- "--shard" && { echo "processi residui, aspetto"; continue; }
  cp "$RUNLOG" "logs/run_materialize.$(date +%m%d_%H%M).log" 2>/dev/null
  GPU=${GPU:-0} WORKERS=${WORKERS:-16} setsid nohup scripts/run_materialize_local.sh > "$RUNLOG" 2>&1 < /dev/null &
  last=$now
done
