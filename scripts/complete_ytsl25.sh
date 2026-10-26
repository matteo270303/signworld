#!/bin/bash
# Completamento di YouTube-SL-25: i passi F1 -> F4 -> F3b -> F5 di docs/ytsl25-piano-preprocessing.md
# in un solo script, con i controlli tra un passo e l'altro. Stato e motivazioni: docs/dataset-status.md.
#
# Uso (dalla root del repo, in tmux):
#   scripts/complete_ytsl25.sh download   # F1: video mai tentati + 'failed' + 'blocked' (riprende dal ledger)
#   scripts/complete_ytsl25.sh refresh    # F4+F3b+F5: manifest, embedding, clip e pose, indice, check
#   scripts/complete_ytsl25.sh check      # solo scripts/check_datasets.py --deep
#
# 'download' gira finche' fetch-media non esce con 0 (tutto sistemato) o 4 (YouTube rifiuta: aggiorna
# i cookie in .secrets/youtube_cookies.txt, abbassa WORKERS, poi rilancia). Puo' durare giorni.
# Richiede deno (cerca anche in ~/.deno/bin) e che nessun altro fetch-media sia attivo: due download
# dello stesso video in contemporanea scrivono sullo stesso mp4 e lo troncano (24/10/2026, 2 video).
# 'refresh' richiede che nessun fetch-media / text embed / materialize sia attivo. ATTENZIONE: rifare
# l'indice con nuovi canali sposta una parte dei canali tra gli split (hash per lingua): le run fatte
# sull'indice vecchio non sono confrontabili con quelle sul nuovo. Il vecchio indice viene copiato in
# data/training/youtube_sl25/index.<data>-<ora>.parquet.
# Variabili: GPU (0), WORKERS (download: 8; materialize: 16), BUDGET_S (null = senza tetto di tempo).
set -euo pipefail
cd "$(dirname "$0")/.."

STEP=${1:?uso: $0 download|refresh|check}
GPU=${GPU:-0}
DATA=data/youtube-sl-25
OUT=data/training/youtube_sl25
CFG=(-c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml -c parameters/ablation/arm_A.yaml -c parameters/model/youtube_sl25_run.yaml)
mkdir -p logs tmp

busy() { pgrep -af "fetch-media|text embed|train materialize|manifest build" | grep -v pgrep || true; }

case "$STEP" in
download)
  # Senza deno yt-dlp non risolve le challenge di YouTube e ogni video fallisce ("The page needs to be reloaded").
  export PATH="$HOME/.deno/bin:$PATH"
  command -v deno > /dev/null || { echo "ERRORE: deno non trovato (ne' nel PATH ne' in ~/.deno/bin)" >&2; exit 1; }
  # Solo il processo vero (.venv/bin/signworld), non comandi che contengono la parola (grep, tail, cicli di attesa).
  if pgrep -f "/bin/signworld fetch-media" > /dev/null; then
    echo "ERRORE: c'e' gia' un fetch-media attivo:" >&2; pgrep -af "/bin/signworld fetch-media" >&2; exit 1
  fi
  # La configurazione di default forza IPv6, che qui non c'e'.
  sed -e 's/ip_version: 6/ip_version: 4/' -e "s/^run_budget_s: .*/run_budget_s: ${BUDGET_S:-null}/" \
    parameters/acquisition/default.yaml > tmp/acquisition_ipv4.yaml
  while true; do
    rc=0
    uv run --no-sync signworld fetch-media youtube_sl25 --shard 0 --num-shards 1 \
      -c tmp/acquisition_ipv4.yaml --workers "${WORKERS:-8}" 2>&1 | tee -a logs/fetch.log || rc=${PIPESTATUS[0]}
    uv run --no-sync signworld status youtube_sl25 -c tmp/acquisition_ipv4.yaml || true
    case $rc in
      0) echo "Download completo. Prossimo passo: $0 refresh"; exit 0 ;;
      3) echo "Restano elementi: riprendo tra 60 s"; sleep 60 ;;
      4) echo "YouTube rifiuta (bot wall): aggiorna i cookie / abbassa WORKERS e rilancia" >&2; exit 4 ;;
      *) echo "fetch-media uscito con $rc: guarda logs/fetch.log" >&2; exit "$rc" ;;
    esac
  done
  ;;
refresh)
  if [ -n "$(busy)" ]; then echo "ERRORE: processi attivi:" >&2; busy >&2; exit 1; fi
  uv run --no-sync signworld manifest build youtube_sl25 -c parameters/analysis/default.yaml
  # Non incrementale: con un manifest nuovo va rifatto per intero (scrive 4 file, non atomico).
  CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync signworld text embed youtube_sl25 --device cuda \
    -c parameters/analysis/default.yaml 2>&1 | tee -a logs/embed.log
  # Salta le clip gia' negli indici; ripara da solo l'ultima riga rotta.
  GPU=$GPU WORKERS=${WORKERS:-16} SOURCE=youtube_sl25 OUTPUT=$OUT scripts/run_materialize_local.sh
  [ -f $OUT/index.parquet ] && cp $OUT/index.parquet "$OUT/index.$(date +%Y%m%d-%H%M%S).parquet"
  STORE=$(ls -td $DATA/text/*/ | head -1)
  uv run --no-sync signworld train index --manifest $DATA/manifest/clips.parquet \
    --materialized $OUT --embeddings "${STORE%/}" --output $OUT/index.parquet "${CFG[@]}" --workers 48
  echo "ATTENZIONE: se il fingerprint non e' piu' 2fdc70f119730c4c aggiorna data.embeddings in parameters/model/youtube_sl25_run.yaml (ora: ${STORE%/})"
  uv run --no-sync python scripts/check_datasets.py --source youtube_sl25 --deep \
    --json reports/dataset_check/youtube_sl25.json
  ;;
check)
  uv run --no-sync python scripts/check_datasets.py --deep --json reports/dataset_check/deep.json
  ;;
*) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
