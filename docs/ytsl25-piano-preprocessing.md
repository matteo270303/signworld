# YouTube-SL-25: piano di preprocessing (download, embedding, pose, indice)

Tutti i comandi si lanciano **dalla root del repo**, in locale (niente condor). Il piano dice in che
ordine, cosa controllare prima e dopo ogni passo, e cosa succede se un comando viene interrotto.
Le affermazioni sulle riprese sono state lette nel codice (`signworld/data/corpus/materialize.py`,
`signworld/data/text.py`, `signworld/data/acquisition/ledger.py`, `signworld/data/acquisition/parallel.py`).
Dove non ho provato sul campo lo dico.

## 0. Stato di partenza

| Cosa        | Valore                                                                          |
| ----------- | ------------------------------------------------------------------------------- |
| Dati        | `data/youtube-sl-25` → `~/dataset/youtube-sl-25` (video in `raw/videos/`)       |
| Ledger      | `data/youtube-sl-25/ledgers/adopted.jsonl` (13.455 `done`, 2.640 `unavailable`) |
| Manifest    | `data/youtube-sl-25/manifest/clips.parquet`, 635.807 clip, 12.850 video         |
| Embedding   | non ancora fatti                                                                |
| Clip e pose | non ancora fatte (`data/training/youtube_sl25`)                                 |
| Risorse     | **solo la GPU 0** è utilizzabile (96 GB, ~25 GB occupati da altri), 96 CPU, 499 GB RAM |

## 1. Regole d'oro (leggile prima di lanciare qualsiasi cosa)

1. **Mai due processi `train materialize` con lo stesso `--shard` e lo stesso `--num-shards`.** Scrivono
   nello stesso `index.<shard>.jsonl` e si corrompono. Prima di rilanciare: `pgrep -af "train materialize"`.
2. **Dopo ogni interruzione di `train materialize`, esegui il controllo dell'indice (§5.1) prima di
   rilanciare.** Una riga troncata nell'indice fa fallire la ripresa di _tutti_ gli shard.
3. **Ordine fisso dopo che sono arrivati nuovi video:** `manifest build` → `text embed` → `train materialize`
   → `train index`. L'indice deve usare lo stesso manifest con cui hai calcolato gli embedding.
4. **`manifest build` solo quando nessun `text embed` o `train materialize` sta partendo.** Il file è
   scritto in modo atomico, ma i comandi lo leggono una sola volta all'avvio: chi parte durante la
   scrittura vede la versione vecchia o la nuova, mai una a metà, ma poi lavora su una versione diversa
   dagli altri.
5. **Usa sempre `uv run --no-sync`** per i lavori lanciati in parallelo (evita che due `uv run` si
   contendano il venv). Prima volta: `uv run signworld --help` una sola volta, da solo.
6. **Un lavoro lungo per finestra `tmux`**, con il log su file (`logs/`), così sai sempre cosa gira.

## 2. Ordine delle fasi

```
F0 prerequisiti ──► F1 download (in background, indipendente) ──────────────┐
        │                                                                    │ (quando vuoi
        └──► F2 embedding ──► F3 pose e clip ──► F5 indice ◄── F4 refresh ◄──┘  nuovi video)
```

### F0. Prerequisiti (una volta)

```bash
mkdir -p logs tmp
uv run signworld --help                       # sincronizza il venv una sola volta
huggingface-cli login                          # EmbeddingGemma è un modello con licenza da accettare
```

- Il primo `text embed` scarica `google/embeddinggemma-300m`; il primo `train materialize` scarica i
  pesi RTMW (`rtmlib`). Non li ho trovati in cache: verifica la rete.
- Controllo: `nvidia-smi` (scegli le GPU libere), `df -h .` (spazio).

### F1. Download dei video mancanti (opzionale, in background)

Finestra tmux `dl`. Serve IPv4 (la configurazione di default forza IPv6, che qui non c'è):

```bash
sed 's/ip_version: 6/ip_version: 4/' parameters/acquisition/default.yaml > tmp/acquisition_ipv4.yaml
# in tmp/acquisition_ipv4.yaml puoi mettere  run_budget_s: null  per non fermarti dopo 40 minuti
uv run --no-sync signworld fetch-media youtube_sl25 --shard 0 --num-shards 1 \
    -c tmp/acquisition_ipv4.yaml --limit 40       # PROVA: 40 video
uv run --no-sync signworld status youtube_sl25
# poi il run completo
uv run --no-sync signworld fetch-media youtube_sl25 --shard 0 --num-shards 1 \
    -c tmp/acquisition_ipv4.yaml 2>&1 | tee -a logs/fetch.log
```

- Codici di uscita: `0` tutto finito, `3` mancano ancora elementi, `4` YouTube ha rifiutato (bot wall).
  Con `4` abbassa `--workers` (default 8) o alza `sleep_min_s`/`sleep_max_s`, poi rilancia.
- Controllo: `uv run signworld status youtube_sl25` (le righe `done` crescono).
- Il download non blocca nulla: puoi fare F2 e F3 mentre gira.

### F2. Embedding del testo (EmbeddingGemma)

Finestra tmux `emb`, su una GPU:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync signworld text embed youtube_sl25 --device cuda \
    -c parameters/analysis/default.yaml 2>&1 | tee -a logs/embed.log
```

Stampa la cartella del risultato, `data/youtube-sl-25/text/<fingerprint>`: **annotala**, serve a F5.
Controllo di coerenza (§5.2). Opzionale: `uv run signworld text verify youtube_sl25 --device cuda`.

### F3. Clip e pose (stage 0)

**3a. Prova di tempo** (1/64 delle clip, circa 10.000): rivela subito se la GPU è usata e quanto durerà.

```bash
time CUDA_VISIBLE_DEVICES=0 uv run --no-sync signworld train materialize youtube_sl25 \
    --output data/training/youtube_sl25 --device cuda --shard 0 --num-shards 64 \
    -c parameters/analysis/default.yaml
```

La prova scrive `index.shard-00000-of-00064.jsonl`: quelle clip restano valide e il run completo le
salta. Lancia il run completo solo a prova finita (regola 1: mai due run sulla stessa clip).

**3b. Run completo**: 16 processi sulla GPU 0 (l'unica usabile), con ripresa, tetto di RAM per processo e
watchdog. Un log per processo in `logs/materialize_16_<n>.log`.

```bash
GPU=0 WORKERS=16 nohup scripts/run_materialize_local.sh > logs/run_materialize.log 2>&1 &
```

Causa dei vecchi crash (codice 137, tetto di RAM superato): `decord` decodifica in avanti in un thread
di fondo (~2,5 GB/s) finché il `VideoReader` resta aperto; `materialize` ora lo libera subito dopo aver letto
i frame (picco ~2-3 GB per processo invece di 10-28 GB). Lo script imposta anche
`MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_MMAP_THRESHOLD_=131072` contro la frammentazione di
glibc, e un tetto di 8 GB per processo (`RAM_PER_WORKER_GB`): con 16 processi il caso peggiore è ~130 GB.
Si può rilanciare cambiando `WORKERS`: le clip già negli indici di qualunque shard vengono saltate.

- Avanzamento: `cat data/training/youtube_sl25/index.*.jsonl | wc -l` e `tail -n1 logs/materialize_*.log`
  (una riga ogni 500 clip).
- **Quando uno shard è finito**: l'ultima riga del suo log è `youtube_sl25 shard-0000X-of-00008: N clips;
S without a signer, F failed`. Se rilanci e `N == S + F`, non restano clip da fare (restano solo quelle
  senza segnante).
- `F failed` > 0: cerca `materialization failed` nel log; le clip che falliscono falliranno ancora.

### F4. Refresh quando il download ha portato nuovi video

Da fare **solo con F2 e F3 fermi** (nessun processo attivo):

```bash
pgrep -af "text embed|train materialize|manifest build" || echo "libero"
uv run --no-sync signworld status youtube_sl25
uv run --no-sync signworld manifest build youtube_sl25 -c parameters/analysis/default.yaml
# poi F2 (di nuovo, riscrive tutto) e F3b (gli stessi comandi: saltano le clip già fatte)
```

Il numero di shard può cambiare rispetto al run precedente (l'indice legge tutti i file `index*.jsonl`),
ma **solo se nessun run vecchio è ancora attivo** (regola 1).

### F5. Indice di training (alla fine)

```bash
uv run --no-sync signworld train index \
    --manifest data/youtube-sl-25/manifest/clips.parquet \
    --materialized data/training/youtube_sl25 \
    --embeddings data/youtube-sl-25/text/<fingerprint> \
    --output data/training/youtube_sl25/index.parquet \
    -c parameters/model/worldsign.yaml
```

Prima: i controlli §5.1 (indice) e §5.2 (embedding coerenti con il manifest) e nessun processo attivo.
Aggiungi `--benchmark <manifest.parquet>` (ripetibile) per togliere i clip contaminati.

## 3. Se interrompi: cosa succede, comando per comando

| Comando             | Interruzione sicura?     | Cosa si perde / rischio                                                                                                                                                                                                                                                           | Come riprendi                                                                                            |
| ------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `fetch-media`       | **Sì**                   | Il ledger è scritto riga per riga con `fsync` e solo dal processo principale. Gli item in corso non sono registrati: si rifanno (`yt-dlp` riprende dai `.part`). Dopo Ctrl+C controlla con `pgrep -af yt-dlp\|fetch-media` che non restino processi (non l'ho provato sul campo). | Stesso comando. Un item è `done` solo dopo che il file `.info.json` è scritto.                           |
| `text embed`        | **Sì, ma non è atomico** | Scrive 4 file di fila (`embeddings.npy`, `captions.parquet`, `clips.parquet`, `identity.json`). Se muore a metà lo store è **incoerente**.                                                                                                                                        | Rilancia `text embed` fino in fondo (riscrive tutto). Poi il controllo §5.2.                             |
| `manifest build`    | **Sì**                   | Scrive su un file `.tmp` e poi lo rinomina: il vecchio manifest resta intatto.                                                                                                                                                                                                    | Rilancia.                                                                                                |
| `train materialize` | **Sì, con un controllo** | Per ogni clip scrive prima `clips/<id>.mp4`, poi `poses/<id>.npz`, e **per ultima** la riga nell'indice. Una clip interrotta non è nell'indice e viene rifatta (i file vengono sovrascritti). Rischio: la **riga dell'indice troncata**, vedi §5.1.                               | Controllo §5.1, poi lo stesso comando (stessi `--shard` e `--num-shards`, o altri con run vecchi fermi). |
| `train index`       | Sì                       | Scrive solo alla fine.                                                                                                                                                                                                                                                            | Rilancia.                                                                                                |
| `adopt`             | Sì                       | Idempotente: salta ciò che i ledger già registrano.                                                                                                                                                                                                                               | Rilancia.                                                                                                |

## 4. Limiti noti della ripresa (non sono errori)

- **Le clip senza segnante vengono rielaborate a ogni ripresa.** `materialize` non le registra
  (restituisce `None`), quindi ogni rilancio rifà la rilevazione su 8 frame per ciascuna. Costa tempo,
  non corrompe nulla.
- **Le clip che falliscono** (`materialization failed`) falliscono di nuovo a ogni ripresa.
- **Embedding e manifest devono coincidere.** `text embed` non è incrementale: dopo un nuovo
  `manifest build` va rifatto per intero.
- **File orfani:** dopo un'interruzione possono restare `clips/*.mp4` o `poses/*.npz` senza riga
  nell'indice. Sono innocui e vengono sovrascritti.

## 5. Controlli da fare prima di riprendere

### 5.1 Indice di `materialize` (dopo ogni interruzione, prima di rilanciare)

```bash
pgrep -af "train materialize" || echo "ok: nessun processo attivo"
for f in data/training/youtube_sl25/index.*.jsonl; do
  [ "$(tail -c1 "$f" | xxd -p)" = "0a" ] || echo "MANCA IL NEWLINE FINALE: $f"
  tail -n1 "$f" | python3 -c 'import sys,json; json.loads(sys.stdin.read())' 2>/dev/null \
    || echo "ULTIMA RIGA ROTTA: $f"
done
# duplicati di clip_id (devono essere zero)
cat data/training/youtube_sl25/index.*.jsonl | grep -o '"clip_id": "[^"]*"' | sort | uniq -d | head
```

Se un file segnala un problema, **togli l'ultima riga** (la clip verrà rifatta) e ricontrolla:

```bash
sed -i '$d' data/training/youtube_sl25/index.shard-0000X-of-00008.jsonl
```

Se compaiono duplicati vuol dire che due run hanno lavorato sulla stessa clip (regola 1): non
cancellare nulla a mano, fermati e chiedi.

### 5.2 Coerenza degli embedding con il manifest

```bash
STORE=data/youtube-sl-25/text/<fingerprint>
uv run python -I - "$STORE" <<'EOF'
import sys, pathlib, numpy as np, pyarrow.parquet as pq
s = pathlib.Path(sys.argv[1])
m = pq.read_table("data/youtube-sl-25/manifest/clips.parquet")
e = np.load(s / "embeddings.npy", mmap_mode="r")
c = pq.read_table(s / "captions.parquet")
k = pq.read_table(s / "clips.parquet")
print("embedding", e.shape[0], "captions", c.num_rows, "clip", k.num_rows, "manifest", m.num_rows)
print("OK" if e.shape[0] == c.num_rows and k.num_rows == m.num_rows
      else "INCONSISTENTE: rilancia text embed")
EOF
```

### 5.3 Download

```bash
pgrep -af "fetch-media" || echo "nessun download attivo"
uv run --no-sync signworld status youtube_sl25
ls data/youtube-sl-25/raw/videos | grep -cE '\.(part|ytdl|temp)'     # residui: normali, si riprendono
```

## 6. Checklist di ripresa (copia e segui)

1. `pgrep -af "fetch-media|text embed|train materialize|manifest build"`: sai cosa sta girando?
2. Se hai interrotto `train materialize`: controllo §5.1, e correggi le righe rotte.
3. Se hai interrotto `text embed`: rilancialo fino in fondo, poi controllo §5.2.
4. Se nel frattempo sono arrivati video e vuoi usarli: F4 (con tutto fermo).
5. Rilancia il comando con **gli stessi argomenti** dell'ultima volta (stesso `--output`, `--num-shards`).
6. Dopo qualche minuto: `cat data/training/youtube_sl25/index.*.jsonl | wc -l` cresce? `tail logs/materialize_*.log` senza errori nuovi?

## 7. Layout tmux consigliato

| Finestra | Contenuto                                                   |
| -------- | ----------------------------------------------------------- |
| `dl`     | `fetch-media` (F1)                                          |
| `emb`    | `text embed` (F2), poi libera                               |
| `pose`   | il ciclo di `train materialize` (F3), con `tail -f` sui log |
| `ctl`    | i controlli §5, `nvidia-smi -l 5`, `status`                 |
