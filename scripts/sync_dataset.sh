#!/usr/bin/env bash
# Copia su Leonardo solo i dati di YouTube-SL-25 che servono al training:
#   <base>/youtube-sl-25/text/       embedding dei testi
#   <base>/youtube-sl-25/manifest/   manifest delle clip (preflight)
#   <base>/training/youtube_sl25/    index.parquet + clip e pose referenziate dall'indice
#
# Non copia: raw/ (video grezzi), cookie, metadati e liste di download, indici intermedi
# (shard, before-fix, datati), clip non referenziate dall'indice.
#
# Uso:
#   scripts/sync_dataset.sh                                  # copia su Leonardo (default sotto)
#   scripts/sync_dataset.sh utente@host:/percorso/base       # base di destinazione diversa
#   scripts/sync_dataset.sh -n                               # prova (nessuna scrittura)
#   scripts/sync_dataset.sh -p 8 -c                          # 8 rsync in parallelo + verifica finale
#
# Opzioni:
#   -s DIR    dataset YouTube-SL-25 sorgente (default ~/dataset/youtube-sl-25)
#   -t DIR    cartella di training sorgente (default <repo>/data/training/youtube_sl25)
#   -n        dry-run
#   -b KB/s   limite di banda (per ogni processo rsync)
#   -p N      processi rsync in parallelo (default 4; con linee veloci 4-8; 1 = uno solo)
#   -e CMD    comando ssh (default "ssh")
#   -i FILE   chiave privata SSH
#   -r N      tentativi per ogni rsync (default 10)
#   -d SEC    secondi tra i tentativi (default 3)
#   -k        disabilita il controllo della host key (meno sicuro)
#   -c        a fine copia confronta numero di file e byte con la destinazione
#
# Destinazione di default: CINECA Leonardo. Serve un certificato SSH valido (step-cli ssh login)
# e SSH_AUTH_SOCK impostato nella shell da cui si lancia lo script (il certificato dura ~12h:
# usare tmux e rilanciare lo stesso comando se scade).
#
# È riprendibile (--partial): se si interrompe, rilanciare lo stesso comando. Non cancella
# nulla in destinazione. Mostra l'avanzamento (file trasferiti ogni 15s).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/bin/python"
DEFAULT_DEST="ecolonna@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ecolonna"
SRC="$HOME/dataset/youtube-sl-25"
TRAIN_SRC="$REPO/data/training/youtube_sl25"
DRY=""; BW=""; JOBS=4; SSH="ssh"; CHECK=0
RETRIES=10; RETRY_DELAY=3; IDENTITY=""; INSECURE=0
while getopts "s:t:nb:p:e:i:r:d:kch" o; do
  case $o in
    s) SRC=$OPTARG ;; t) TRAIN_SRC=$OPTARG ;; n) DRY="--dry-run" ;; b) BW="--bwlimit=$OPTARG" ;;
    p) JOBS=$OPTARG ;; e) SSH=$OPTARG ;; c) CHECK=1 ;;
    i) IDENTITY=$OPTARG ;; r) RETRIES=$OPTARG ;; d) RETRY_DELAY=$OPTARG ;; k) INSECURE=1 ;;
    *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
DEST=${1:-$DEFAULT_DEST}
DEST=${DEST%/}
SRC=${SRC%/}
TRAIN_SRC=${TRAIN_SRC%/}

[[ $RETRIES =~ ^[0-9]+$ && $RETRIES -ge 1 ]] || { echo "-r deve essere un intero >= 1" >&2; exit 2; }
[[ $RETRY_DELAY =~ ^[0-9]+$ ]] || { echo "-d deve essere un intero >= 0" >&2; exit 2; }
[[ $JOBS =~ ^[0-9]+$ && $JOBS -ge 1 ]] || { echo "-p deve essere un intero >= 1" >&2; exit 2; }
[ -z "$IDENTITY" ] || [ -r "$IDENTITY" ] || { echo "chiave non leggibile: $IDENTITY" >&2; exit 2; }
[ -x "$PY" ] || { echo "manca $PY (serve l'ambiente del progetto: uv sync)" >&2; exit 2; }
[ -d "$SRC/text" ] && [ -d "$SRC/manifest" ] || { echo "in $SRC mancano text/ o manifest/" >&2; exit 1; }
[ -d "$TRAIN_SRC" ] || { echo "manca la cartella di training: $TRAIN_SRC" >&2; exit 1; }

# Comando ssh come array (per le chiamate dirette) e come stringa quotata (per rsync -e).
# Keepalive per non far cadere i trasferimenti lunghi.
SSHA=($SSH -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
[ "$INSECURE" -eq 0 ] || SSHA+=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
[ -z "$IDENTITY" ] || SSHA+=(-i "$IDENTITY")
SSH_E=$(printf '%q ' "${SSHA[@]}")

EXCLUDES=(--exclude='cookies*' --exclude='*.part' --exclude='*.ytdl')
# Niente -z: mp4 e vtt sono già compressi o piccoli.
RSYNC=(rsync -a --partial --human-readable -e "$SSH_E" $DRY $BW "${EXCLUDES[@]}")

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Elenco dei file da copiare, uno per riga, relativi alla cartella sorgente.
DATA_LIST="$TMP/youtube-sl-25.list"
TRAIN_LIST="$TMP/youtube_sl25.list"
(cd "$SRC" && find text manifest -type f ! -name 'cookies*' ! -name '*.part' ! -name '*.ytdl' | sort) > "$DATA_LIST"
"$PY" - "$TRAIN_SRC" "$TRAIN_LIST" <<'EOF'
# Lista del training: index.parquet e i video e le pose che l'indice referenzia.
import os, sys
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]
table = pq.read_table(os.path.join(root, "index.parquet"), columns=["video_path", "pose_path"])
files = {"index.parquet"}
for column in ("video_path", "pose_path"):
    for path in table.column(column).to_pylist():
        if "youtube_sl25/" not in path:
            sys.exit(f"percorso fuori da youtube_sl25/: {path}")
        files.add(path.split("youtube_sl25/", 1)[1])
missing = sorted(f for f in files if not os.path.isfile(os.path.join(root, f)))
if missing:
    sys.exit(f"{len(missing)} file dell'indice mancano, esempio: {missing[0]}")
with open(out, "w") as fh:
    fh.write("\n".join(sorted(files)) + "\n")
print(f"training: {len(files)} file dall'indice di {table.num_rows} clip")
EOF

# Esegue il comando riprovando se fallisce (rete, certificato scaduto: --partial riprende).
retry() {
  local n=1
  until "$@"; do
    [ "$n" -lt "$RETRIES" ] || { echo "fallito dopo $RETRIES tentativi" >&2; return 1; }
    echo "tentativo $n/$RETRIES fallito, riprovo tra ${RETRY_DELAY}s..." >&2
    n=$((n + 1)); sleep "$RETRY_DELAY"
  done
}

# Crea la cartella di destinazione (rsync non crea le cartelle genitore).
make_dest() {
  [ -z "$DRY" ] || return 0
  if [[ $1 == *:* ]]; then
    "${SSHA[@]}" "${1%%:*}" "mkdir -p '${1#*:}'"
  else
    mkdir -p "$1"
  fi
}

# Copia i file dell'elenco list (relativi a src) dentro dst, creando le cartelle intermedie.
sync_list() {
  local src=$1 dst=$2 list=$3 name total pids p part mon fail
  name=$(basename "$src")
  total=$(wc -l < "$list")
  echo "== $src -> $dst: $total file =="
  make_dest "$dst"
  if [ "$JOBS" -le 1 ]; then
    retry "${RSYNC[@]}" --info=progress2 --files-from="$list" "$src/" "$dst/"
    return 0
  fi
  # split -n richiede un file normale (non una pipe): i blocchi sono file temporanei
  split -n "l/$JOBS" "$list" "$TMP/$name.part_"
  echo "$JOBS rsync in parallelo"
  : > "$TMP/$name.done"
  pids=()
  for part in "$TMP/$name".part_*; do
    # --out-format scrive un nome per ogni file trasferito: serve a contare l'avanzamento
    retry "${RSYNC[@]}" --out-format='%n' --files-from="$part" "$src/" "$dst/" \
      >> "$TMP/$name.done" &
    pids+=($!)
  done
  # avanzamento: file trasferiti in questa esecuzione (quelli già presenti non contano)
  (while sleep 15; do
     echo "[$(date +%H:%M:%S)] $name: trasferiti $(grep -vc '/$' "$TMP/$name.done" || true) / $total file"
   done) &
  mon=$!
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  kill "$mon" 2>/dev/null || true
  [ $fail -eq 0 ] || { echo "un rsync di $name è fallito: rilanciare" >&2; return 1; }
}

# Confronta file e byte dell'elenco: sorgente contro destinazione.
verify_list() {
  local src=$1 dst=$2 list=$3 loc rem
  loc=$( (cd "$src" && xargs -d '\n' stat -c %s) < "$list" | awk '{n++; s+=$1} END {print n+0, s+0}')
  if [[ $dst == *:* ]]; then
    rem=$("${SSHA[@]}" "${dst%%:*}" "cd '${dst#*:}' && xargs -d '\n' stat -c %s | awk '{n++; s+=\$1} END {print n+0, s+0}'" < "$list")
  else
    rem=$( (cd "$dst" && xargs -d '\n' stat -c %s) < "$list" | awk '{n++; s+=$1} END {print n+0, s+0}')
  fi
  echo "$src: locale (file byte) $loc, remoto $rem"
  if [ "$loc" = "$rem" ]; then echo "OK: coincidono"; else echo "DIVERSI: rilanciare" >&2; return 1; fi
}

DATA_DEST="$DEST/youtube-sl-25"
TRAIN_DEST="$DEST/training/youtube_sl25"
echo "destinazione: $DEST"
sync_list "$SRC" "$DATA_DEST" "$DATA_LIST"
sync_list "$TRAIN_SRC" "$TRAIN_DEST" "$TRAIN_LIST"

# verifica finale: file e byte, sorgente contro destinazione
if [ "$CHECK" -eq 1 ] && [ -z "$DRY" ]; then
  verify_list "$SRC" "$DATA_DEST" "$DATA_LIST" || exit 1
  verify_list "$TRAIN_SRC" "$TRAIN_DEST" "$TRAIN_LIST" || exit 1
fi
echo "fatto."
