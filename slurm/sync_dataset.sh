#!/usr/bin/env bash
# Copia la cartella del dataset YouTube-SL-25 su un altro server (addestramento multinodo).
#
# Uso:
#   slurm/sync_dataset.sh                                       # copia su Leonardo (default sotto)
#   slurm/sync_dataset.sh utente@host:/percorso/dest            # copia
#   slurm/sync_dataset.sh -n utente@host:/percorso/dest         # prova (nessuna scrittura)
#   slurm/sync_dataset.sh -s ~/dataset/youtube-sl-25 -p 4 -c utente@host:/dest
#
# Opzioni:
#   -s DIR    sorgente (default ~/dataset/youtube-sl-25)
#   -n        dry-run
#   -b KB/s   limite di banda (per ogni processo rsync)
#   -p N      processi rsync in parallelo sui video (default 1; con linee veloci 4-8)
#   -e CMD    comando ssh (default "ssh")
#   -i FILE   chiave privata SSH
#   -r N      tentativi per ogni rsync (default 10)
#   -d SEC    secondi tra i tentativi (default 3)
#   -k        disabilita il controllo della host key (meno sicuro)
#   -c        a fine copia confronta numero di file e byte con la destinazione
#
# Destinazione di default: CINECA Leonardo. Serve un certificato SSH valido (step ssh login)
# e SSH_AUTH_SOCK impostato nella shell da cui si lancia lo script (il certificato dura ~12h:
# usare tmux e rilanciare lo stesso comando se scade).
#
# È riprendibile (--partial): se si interrompe, rilanciare lo stesso comando. Non cancella
# nulla in destinazione. Copia la cartella intera (crea <dest>/<nome cartella>) con tutto il
# contenuto, sottocartelle comprese; esclude solo cookie e file temporanei (.part, .ytdl).
# Mostra l'avanzamento (percentuale/velocità, o file copiati ogni 15s in modalità parallela).
set -euo pipefail

DEFAULT_DEST="ecolonna@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ecolonna"
SRC="$HOME/dataset/youtube-sl-25"
DRY=""; BW=""; JOBS=1; SSH="ssh"; CHECK=0
RETRIES=10; RETRY_DELAY=3; IDENTITY=""; INSECURE=0
while getopts "s:nb:p:e:i:r:d:kch" o; do
  case $o in
    s) SRC=$OPTARG ;; n) DRY="--dry-run" ;; b) BW="--bwlimit=$OPTARG" ;;
    p) JOBS=$OPTARG ;; e) SSH=$OPTARG ;; c) CHECK=1 ;;
    i) IDENTITY=$OPTARG ;; r) RETRIES=$OPTARG ;; d) RETRY_DELAY=$OPTARG ;; k) INSECURE=1 ;;
    *) sed -n '2,32p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
DEST=${1:-$DEFAULT_DEST}
SRC=${SRC%/}
DEST="${DEST%/}/$(basename "$SRC")"

[[ $RETRIES =~ ^[0-9]+$ && $RETRIES -ge 1 ]] || { echo "-r deve essere un intero >= 1" >&2; exit 2; }
[[ $RETRY_DELAY =~ ^[0-9]+$ ]] || { echo "-d deve essere un intero >= 0" >&2; exit 2; }
[ -z "$IDENTITY" ] || [ -r "$IDENTITY" ] || { echo "chiave non leggibile: $IDENTITY" >&2; exit 2; }

# Comando ssh come array (per le chiamate dirette) e come stringa quotata (per rsync -e).
# Keepalive per non far cadere i trasferimenti lunghi.
SSHA=($SSH -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
[ "$INSECURE" -eq 0 ] || SSHA+=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
[ -z "$IDENTITY" ] || SSHA+=(-i "$IDENTITY")
SSH_E=$(printf '%q ' "${SSHA[@]}")

EXCLUDES=(--exclude='cookies*' --exclude='*.part' --exclude='*.ytdl')
# Niente -z: mp4 e vtt sono già compressi o piccoli.
RSYNC=(rsync -a --partial --human-readable -e "$SSH_E" $DRY $BW "${EXCLUDES[@]}")

# Esegue il comando riprovando se fallisce (rete, certificato scaduto: --partial riprende).
retry() {
  local n=1
  until "$@"; do
    [ "$n" -lt "$RETRIES" ] || { echo "fallito dopo $RETRIES tentativi" >&2; return 1; }
    echo "tentativo $n/$RETRIES fallito, riprovo tra ${RETRY_DELAY}s..." >&2
    n=$((n + 1)); sleep "$RETRY_DELAY"
  done
}

[ -d "$SRC/raw/videos" ] || { echo "manca $SRC/raw/videos" >&2; exit 1; }
echo "sorgente: $SRC ($(du -sh "$SRC/raw/videos" | cut -f1) di video)"
echo "destinazione: $DEST"

# 1) file piccoli: metadati, info, liste
echo "== file fuori da raw/videos/ =="
retry "${RSYNC[@]}" --info=progress2 --exclude='/raw/videos/' "$SRC/" "$DEST/"

# 2) video e sottotitoli, opzionalmente in parallelo (un rsync per blocco di file)
if [ "$JOBS" -le 1 ]; then
  echo "== raw/videos/ =="
  retry "${RSYNC[@]}" --info=progress2 "$SRC/raw/videos/" "$DEST/raw/videos/"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  # split -n l/N richiede un file normale (non una pipe): prima scrivo l'elenco completo
  (cd "$SRC/raw/videos" && find . -type f -printf '%P\n' \
     | { grep -vE '\.(part|ytdl)$' || true; } | sort) > "$tmp/all"
  split -n "l/$JOBS" "$tmp/all" "$tmp/list_"
  # una prima passata crea la cartella di destinazione (la lista è vuota)
  retry "${RSYNC[@]}" --files-from=/dev/null "$SRC/raw/videos/" "$DEST/raw/videos/"
  total=$(wc -l < "$tmp/all")
  echo "== raw/videos/: $total file, $JOBS rsync in parallelo =="
  : > "$tmp/done.log"
  pids=()
  for list in "$tmp"/list_*; do
    # --out-format scrive un nome per ogni file trasferito: serve a contare l'avanzamento
    retry "${RSYNC[@]}" --out-format='%n' --files-from="$list" "$SRC/raw/videos/" "$DEST/raw/videos/" \
      >> "$tmp/done.log" &
    pids+=($!)
  done
  # avanzamento: file trasferiti in questa esecuzione (quelli già presenti non contano)
  (while sleep 15; do
     echo "[$(date +%H:%M:%S)] trasferiti $(grep -vc '/$' "$tmp/done.log" || true) / $total file"
   done) &
  mon=$!
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  kill "$mon" 2>/dev/null || true
  [ $fail -eq 0 ] || { echo "un rsync è fallito: rilanciare" >&2; exit 1; }
fi

# 3) verifica: file e byte dei video e dei sottotitoli, sorgente contro destinazione
if [ "$CHECK" -eq 1 ] && [ -z "$DRY" ]; then
  count='find raw/videos -type f \( -name "*.mp4" -o -name "*.vtt" \) -printf "%s\n" | awk "{n++; s+=\$1} END {print n, s}"'
  loc=$(cd "$SRC" && bash -c "$count")
  if [[ $DEST == *:* ]]; then
    rem=$("${SSHA[@]}" "${DEST%%:*}" "cd '${DEST#*:}' && $count")
  else
    rem=$(cd "$DEST" && bash -c "$count")
  fi
  echo "locale (file byte): $loc"
  echo "remoto (file byte): $rem"
  if [ "$loc" = "$rem" ]; then echo "OK: coincidono"; else echo "DIVERSI: rilanciare" >&2; exit 1; fi
fi
echo "fatto."
