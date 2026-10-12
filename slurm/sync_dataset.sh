#!/usr/bin/env bash
# Copia la cartella del dataset YouTube-SL-25 su un altro server (addestramento multinodo).
#
# Uso:
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
#   -c        a fine copia confronta numero di file e byte con la destinazione
#
# È riprendibile (--partial): se si interrompe, rilanciare lo stesso comando. Non cancella
# nulla in destinazione. Esclude cookie, liste di fallimenti, log e cartelle di quarantena.
set -euo pipefail

SRC="$HOME/dataset/youtube-sl-25"
DRY=""; BW=""; JOBS=1; SSH="ssh"; CHECK=0
while getopts "s:nb:p:e:ch" o; do
  case $o in
    s) SRC=$OPTARG ;; n) DRY="--dry-run" ;; b) BW="--bwlimit=$OPTARG" ;;
    p) JOBS=$OPTARG ;; e) SSH=$OPTARG ;; c) CHECK=1 ;;
    *) sed -n '2,18p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
DEST=${1:?destinazione mancante: utente@host:/percorso}
SRC=${SRC%/}

EXCLUDES=(--exclude='cookies*' --exclude='failed_*' --exclude='video_ids_missing*'
          --exclude='logs_*' --exclude='nohup*' --exclude='quarantine*'
          --exclude='*.part' --exclude='*.ytdl' --exclude='*.info.json')
# Niente -z: mp4 e vtt sono già compressi o piccoli.
RSYNC=(rsync -a --partial --human-readable -e "$SSH" $DRY $BW "${EXCLUDES[@]}")

[ -d "$SRC/videos" ] || { echo "manca $SRC/videos" >&2; exit 1; }
echo "sorgente: $SRC ($(du -sh "$SRC/videos" | cut -f1) di video)"
echo "destinazione: $DEST"

# 1) file piccoli: metadati, info, liste
"${RSYNC[@]}" --exclude='/videos/' "$SRC/" "$DEST/"

# 2) video e sottotitoli, opzionalmente in parallelo (un rsync per blocco di file)
if [ "$JOBS" -le 1 ]; then
  "${RSYNC[@]}" --info=progress2 "$SRC/videos/" "$DEST/videos/"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  (cd "$SRC/videos" && find . -maxdepth 1 -type f -printf '%f\n' \
     | grep -vE '\.(part|ytdl)$|\.f[0-9]+\.' | sort) | split -n "l/$JOBS" - "$tmp/list_"
  # una prima passata crea la cartella di destinazione (la lista è vuota)
  "${RSYNC[@]}" --files-from=/dev/null "$SRC/videos/" "$DEST/videos/"
  pids=()
  for list in "$tmp"/list_*; do
    "${RSYNC[@]}" --files-from="$list" "$SRC/videos/" "$DEST/videos/" &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "un rsync è fallito: rilanciare" >&2; exit 1; }
fi

# 3) verifica: file e byte dei video e dei sottotitoli, sorgente contro destinazione
if [ "$CHECK" -eq 1 ] && [ -z "$DRY" ]; then
  count='find videos -type f \( -name "*.mp4" -o -name "*.vtt" \) -printf "%s\n" | awk "{n++; s+=\$1} END {print n, s}"'
  loc=$(cd "$SRC" && bash -c "$count")
  if [[ $DEST == *:* ]]; then
    rem=$($SSH "${DEST%%:*}" "cd '${DEST#*:}' && $count")
  else
    rem=$(cd "$DEST" && bash -c "$count")
  fi
  echo "locale (file byte): $loc"
  echo "remoto (file byte): $rem"
  if [ "$loc" = "$rem" ]; then echo "OK: coincidono"; else echo "DIVERSI: rilanciare" >&2; exit 1; fi
fi
echo "fatto."
