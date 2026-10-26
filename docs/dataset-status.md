# Stato dei dataset e delle run di prova (20/10/2026, youtube_sl25 aggiornato il 26/10/2026)

Il 26/10 youtube_sl25 è stato completato con `scripts/complete_ytsl25.sh` (§5): le sezioni 1, 2.1 e 5 riportano
i numeri nuovi; csl_daily, phoenix14t e le run di prova (§3) sono rimasti come al 20/10.

Controllo fatto prima del training su **youtube_sl25**, **csl_daily** e **phoenix14t**: download di video e
testo, manifest, embedding, estrazione delle pose e taglio delle clip, indice di training, e avvio dei tre
`scripts/run_train_*.sh`. Tutti i numeri vengono da `scripts/check_datasets.py` (rilanciabile) e dai log.

```bash
uv run --no-sync python scripts/check_datasets.py            # conteggi, ~15 s
uv run --no-sync python scripts/check_datasets.py --deep     # + ffprobe di ogni video e clip, + ogni posa (~40 min)
```

Risultati grezzi: `reports/dataset_check/deep.json` (20/10, tutti e tre) e `reports/dataset_check/youtube_sl25.json`
(26/10). Liste dei file con problemi: `reports/dataset_check/<source>/*.txt`.

## 1. Sintesi

| | youtube_sl25 | csl_daily | phoenix14t |
|---|---|---|---|
| Video scaricati / attesi | **24.873 / 39.197 (63 %)**: tutto ciò che YouTube rende ancora disponibile | 20.653 / 20.653 | 8.257 / 8.257 |
| Testo (didascalie) | VTT per 27.591 file; 23.990 video con didascalie usabili | label ufficiali (caratteri) | traduzioni ufficiali (tedesco) |
| Manifest (clip) | 1.202.492 | 20.653 | 8.257 |
| Embedding (EmbeddingGemma) | coerenti (1.112.278 didascalie uniche) | coerenti (7.379) | coerenti (7.956) |
| Clip + pose fatte | 1.161.579 / 1.164.179 idonee | 20.653 / 20.653 | 8.257 / 8.257 |
| Video o clip illeggibili | **0** | **0** | **0** |
| Pose non valide | **0** | **0** | **0** |
| Indice di training | 1.161.374 clip (26/10) | 20.653 (20/10) | 8.257 (20/10) |
| Run di prova: epoca 1 avviata | sì, sull'indice vecchio (§3) | **sì** | **sì** |

**In breve:** tutti e tre i dataset sono completi e puliti. youtube_sl25 ha tutti i video ancora scaricabili
(il resto è non disponibile su YouTube) e quasi il doppio delle clip del 20/10. Nessun file ha problemi di
codifica. Ho trovato e corretto tre problemi che impedivano o falsavano il training (§4); il 24/10 due
download sovrapposti hanno troncato due video, rifatti il 26/10 (§4bis).

## 2. Dettaglio per dataset

### 2.1 youtube_sl25

**Download** (ledger `data/youtube-sl-25/ledgers/`, ultimo aggiornamento 26/10/2026):

| Stato | Video | Note |
|---|---|---|
| `done` | 24.873 | mp4 + `.info.json` presenti |
| `unavailable` | 13.666 | definitivi (privati, rimossi, "Video unavailable") |
| `failed` 3+ volte | 658 | non più ritentati; all'ultimo giro quasi tutti "This video is unavailable" |
| `blocked` / da scaricare | 0 | |

- Il download è finito il 24/10: non resta niente da tentare (`signworld status youtube_sl25` → `to fetch 0`).
- Su disco: 24.963 mp4 (24.727 H.264, 236 VP9), 24.882 `.info.json`, 27.591 `.vtt`. Nessun file parziale
  (l'unico `.part`, di un video poi risultato non disponibile, è stato cancellato il 26/10) e nessun video
  illeggibile (ffprobe di ogni file).
- 973 video scaricati non compaiono nel manifest: non hanno una traccia di didascalie usabile nella loro
  lingua. Non è un errore.

**Manifest e testo:** 1.202.492 clip da 23.990 video. Lo store degli embedding
(`data/youtube-sl-25/text/2fdc70f119730c4c`, stesso fingerprint di prima: la configurazione della run non
cambia) copre esattamente le clip del manifest: 1.112.278 didascalie uniche, dimensione 768.

**Clip e pose (stage 0):**

| | Clip |
|---|---|
| idonee (video presente, durata 1-20 s) | 1.164.179 |
| fuori durata, escluse per configurazione | 38.313 |
| materializzate (clip mp4 + posa) | 1.161.579 |
| non materializzate | 2.600, **tutte "senza segnante"** (somma esatta dei riepiloghi dell'ultimo giro, `logs/materialize_16_*.log`), 0 fallite |
| escluse dall'indice | 205: posa senza spalle (filtro di `train index`) |

- Le clip nuove sono state fatte dal 24 al 26/10 con 16 processi sulla GPU 0, a ~3,1 clip/s.
- Ogni clip (1.161.579, mpeg4 256×256) si decodifica e ogni posa è valida (`check_datasets.py --deep`).
- Le 323 clip di `rXL9ff90JCU` e `MT8iMfeE4lA` erano fallite perché i due video erano troncati (§4bis):
  dopo il nuovo download 322 sono state fatte, 1 è senza segnante.
- I traceback di decord nei log del 14/10 (5.319 righe "materialization failed" in `materialize_12_*`,
  3.551 in `materialize_16_*`) sono l'errore sui VP9 multithread, poi corretto (`num_threads=1`): quelle
  clip sono state rifatte e sono nell'indice. Dal 24/10 non ce n'è nessuno.
- Il controllo delle clip più corte della didascalia (355 al 20/10, innocue) non è stato ripetuto sulle
  clip nuove.

**Indice (26/10, 16:53):** 945.688 train · 102.798 val_channel · 95.536 test_channel · 17.352 val_video.

| split | canali | video |
|---|---|---|
| train | 2.341 | 19.177 |
| val_channel | 315 | 2.362 |
| test_channel | 312 | 2.020 |
| val_video | 209 | 377 |

- 3.001 canali in tutto. Nessun canale è condiviso tra train, val_channel e test_channel.
- Splits da congelare: `data/training/youtube_sl25/index.splits.csv`, sha256 `d7188cad78f7866e69dcd8ae24c3a21624f9c3eadc5bbcae3b5494696ccf1686`.
- **Gli split sono cambiati rispetto al 20/10:** dei 1.929 canali del vecchio indice
  (`index.before-fix-20261020.parquet`), 32 hanno cambiato gruppo tra train, val_channel e test_channel
  (20.121 delle 610.802 clip vecchie), compresi i 3 canali della correzione del leak (§4.3). Le run
  sull'indice vecchio non sono confrontabili 1:1 con le nuove.
- Backup: `index.20261026.parquet` è l'indice delle 15:42 del 26/10 (identico per canali e split a quello
  attuale, senza le 322 clip recuperate). L'indice del 22/10 è andato perso: il secondo `refresh` dello
  stesso giorno ha sovrascritto il backup con lo stesso nome. Lo script ora mette data e ora nel nome.

### 2.2 csl_daily

- Fonte: frame JPG 512×512 (`data/CSLDaily` → `~/dataset/CSLDaily`), codificati in mp4 H.264 a 30 fps da
  `fetch-media` (`signworld/data/acquisition/transfer/frames.py`). 20.653/20.653 `done`; nessun file
  parziale o illeggibile.
- Manifest: 20.653 clip, split ufficiali 18.400 train / 1.077 valid / 1.176 test. Una frase senza split
  ufficiale è esclusa di proposito (warning in `CSLDailySource.sentences`).
- Clip e pose: 20.653/20.653 (materializzate con `--min-duration-s 0`: le frasi brevi restano), 0 senza
  segnante, 0 fallite. I frame di ogni clip corrispondono alla durata del manifest.
- Embedding: 7.379 didascalie uniche, coerenti con il manifest.

### 2.3 phoenix14t

- Fonte: frame PNG 210×260 (`data/PHOENIX-2014-T`), codificati in mp4 H.264 a 25 fps. 8.257/8.257 `done`.
- Manifest: 8.257 clip, split ufficiali 7.096 train / 519 valid / 642 test.
- Clip e pose: 8.257/8.257, 0 senza segnante, 0 fallite, frame coerenti.
- Embedding: 7.956 didascalie uniche, coerenti.

## 3. Run di prova (`scripts/run_train_*.sh`)

Lanciate una alla volta sulla GPU 0 con `OUT=runs/smoke/<dataset>-A`, fermate dopo i primi step:

| Run | Preflight P1-P17 | Primo step di training | Memoria | Velocità | Durata stimata (15 epoche, 1 GPU) |
|---|---|---|---|---|---|
| csl_daily | pass (P1, P2, P12, P15 skip) | step 10, epoca 1 (stage P), loss 6,20 | 48,3 GiB | 2,84 s/step, 11,3 clip/s | ~11 h (8.625 step) |
| phoenix14t | pass (P1, P2, P12, P15 skip) | step 10, epoca 1 (stage P), loss 5,49 | 48,3 GiB | 2,84 s/step | ~5,7 h (3.315 step) |
| youtube_sl25 | pass (P2, P15 skip) | step 20, epoca 1 (stage P), loss 4,52 | 48,3 GiB | 2,87 s/step | **~8 giorni** (221.880 step) |

**youtube_sl25 dopo il 26/10:** la prova usava l'indice vecchio (473.352 clip di train). Con le 945.688 clip
di train di oggi gli step per 15 epoche diventano circa il doppio (~443.000): **~15 giorni** a 2,87 s/step
su una GPU (stima proporzionale, non misurata). La prova non è stata rifatta sull'indice nuovo.

Log: `logs/smoke/*.log`; cartelle `runs/smoke/*` (si possono cancellare). Una cartella di run creata con
un'altra configurazione non si riprende (`ProvenanceError`): `runs/phoenix14t-A` dei 4 tentativi falliti del
20/10 (batch 128, nessun checkpoint) è stata spostata in `runs/archived/phoenix14t-A.failed-20261020`. Per la run vera basta lanciare lo
script senza `OUT`, ad esempio `scripts/run_train_phoenix14t.sh`.

## 4. Problemi trovati e corretti

1. **Gli indici di csl_daily e phoenix14t avevano gli split sbagliati.** Lo script costruiva l'indice solo
   con `worldsign.yaml`, quindi con `split_source: channel`. Il risultato era train/val_video/val_channel
   invece di train/valid/test. Effetti: la validazione cercava lo split `valid` (assente) e la valutazione
   finale `--split test` sarebbe stata vuota. *Correzione:* `train index` ora riceve gli stessi overlay della
   run, e lo script ricostruisce l'indice se non contiene lo split `test`. Indici ricostruiti.
2. **Su questa macchina nessuna run poteva partire.** Solo la GPU 0 è utilizzabile: le GPU 1-3 sono in
   compute mode *Prohibited*. Il batch globale 128 non entra in una GPU da 96 GB (~2 GiB per clip,
   `docs/worldsign-gpu.md`), e il preflight P16 rifiutava qualunque batch diverso da 128. *Correzione:* nuovo
   parametro `training.effective_batch` (default 128: le run dell'articolo non cambiano) e nuovo overlay
   `parameters/model/local_single_gpu.yaml` (batch 32, lo stesso carico per GPU della configurazione 4×32).
   Gli script lo aggiungono da soli quando `GPUS` indica una sola GPU, e lo segnalano con un warning.
   **Attenzione:** con 32 clip SIGReg e InfoNCE vedono un batch 4 volte più piccolo. Queste run servono a
   provare la pipeline; non sono confrontabili con quelle a 128.
3. **Il preflight P1 bloccava youtube_sl25 ("94 channels in both").** Le cause erano due:
   - P1 contava anche `val_video`, che per costruzione contiene video nuovi di canali visti in train (§4.13.4):
     92 canali. *Correzione:* P1 esclude `val_video`.
   - **Leak reale:** 3 canali multilingua erano tenuti fuori per una lingua e in train per un'altra.
     Esempio: `UC1f5xYbrZxLkklmGkow7AgA`, 1.072 clip in test_channel (svk) e 2.883 in train (asq, gsg,
     ise, cse). *Correzione:* `assign_splits` sceglie i canali per lingua ma li tiene fuori per intero; test
     aggiunto in `tests/data/test_loaders.py`. Dopo la ricostruzione nessun canale è condiviso tra train,
     val_channel e test_channel.

   **Conseguenza:** gli split di youtube_sl25 cambiano per quei 3 canali (train da 476.311 a 473.352 clip).
   La run del 19/10 (`docs/problems.md`) usava l'indice vecchio, quindi le sue metriche non sono
   confrontabili 1:1 con le nuove run. Gli indici precedenti sono salvati come
   `data/training/<dataset>/index.before-fix-20261020.parquet` (e `index.splits.before-fix-20261020.csv`).
4. Script di training: `torchrun` ora parte con `uv run --no-sync torchrun`, quindi funziona anche senza il
   venv attivo e non risincronizza l'ambiente.

## 4bis. Due video troncati da download sovrapposti (24/10, corretto il 26/10)

- Il 24/10 tra le 00:28 e le 00:30 due `fetch-media` hanno girato insieme: 19 video sono stati scaricati
  due volte in contemporanea (due righe `done` nel ledger a pochi secondi di distanza) sullo stesso mp4.
  Due sono rimasti troncati ("moov atom not found"): `rXL9ff90JCU` e `MT8iMfeE4lA`, 323 clip fallite.
  Gli altri 17 si decodificano per intero.
- *Correzione:* i due mp4 cancellati, rimessi in coda con una riga `failed` a 0 tentativi nel ledger,
  riscaricati, poi `refresh`. Il nuovo download è riuscito solo con deno nel PATH: senza, yt-dlp non
  risolve le challenge di YouTube e ogni video fallisce con "The page needs to be reloaded".
- *Prevenzione* in `scripts/complete_ytsl25.sh download`: aggiunge `~/.deno/bin` al PATH ed esce se deno
  manca; esce se c'è già un processo `/bin/signworld fetch-media` attivo.

## 5. Processo per completare youtube_sl25 (fatto, 13/10 → 26/10)

Il piano di `docs/ytsl25-piano-preprocessing.md` è in `scripts/complete_ytsl25.sh`:

```bash
scripts/complete_ytsl25.sh download                 # F1: riprende dal ledger; esce con 4 se YouTube rifiuta
GPU=0 WORKERS=16 scripts/complete_ytsl25.sh refresh # F4+F3b+F5: manifest, embedding, clip e pose, indice, check --deep
scripts/complete_ytsl25.sh check                    # solo il check profondo di tutti i dataset
```

Com'è andata:

- **Download:** 11.418 video in più (da 13.455 a 24.873). Il 35 % dei video attesi non è più su YouTube.
- **Stage 0:** circa 550.000 clip nuove in ~2 giorni (24-26/10), ~3,1 clip/s con 16 processi sulla GPU 0.
- **Embedding:** rifatti per intero (lo store non è incrementale), ~5 minuti; il fingerprint non è cambiato,
  quindi `parameters/model/youtube_sl25_run.yaml` resta com'è.
- **Check profondo:** ~20 minuti per youtube_sl25.
- Log dell'ultimo giro: `logs/refresh_20261026.log`.

Per la run dell'articolo resta da **congelare** `index.splits.csv` (sha256 sopra, §2.1). Un nuovo `refresh`
ricostruisce l'indice: se entrano canali nuovi, una parte dei canali cambia split.

## 6. Osservazioni minori (nessuna azione necessaria)

- 33 canali di youtube_sl25 (erano 24 al 20/10) sono **solo** in `val_video`: l'hash ha mandato in val_video tutti i loro video.
  Per loro val_video misura un canale mai visto, non "video nuovi di canali visti". L'effetto è piccolo.
- `tests/experiment/collaudo/test_results.py` fallisce anche senza queste modifiche (2 test), per via
  delle cartelle `collaudo/` locali. Il resto di `tests/data`, `tests/experiment` e `tests/cli` passa.
- Alla prima validazione scattano gli allarmi `modality_gap` e `noise_drop`: è normale con i pesi iniziali
  (non fermano la run).
