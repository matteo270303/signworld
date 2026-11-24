# 🤟 signworld · YouTube-SL-25

Il percorso completo di **WorldSign su YouTube-SL-25**: dal download dei video alle run dell'articolo
(CVPR 2027). È il `README.md` generale ristretto a questo dataset, con lo stato attuale e il lavoro
che resta da fare. YouTube-SL-25 è il corpus delle run dell'articolo; OpenASL serve solo come prova
della pipeline (`openasl_trial.yaml`).

Tutti i comandi si lanciano **dalla root del repo**. Per i lavori in parallelo usa sempre
`uv run --no-sync`.

Documenti collegati:

| Documento | Contenuto |
|---|---|
| `docs/dataset-status.md` | stato dei dati e delle run di prova (20/10/2026), problemi corretti |
| `docs/ytsl25-piano-preprocessing.md` | fasi F0–F5, riprese dopo un'interruzione, controlli |
| `docs/worldsign-ytsl25.md` | analisi del corpus, integrità, split per canale |
| `docs/worldsign-ablation.md` | le run concordate (ESP-1, ESP-2, ESP-6, ESP-4) e le regole comuni |
| `docs/worldsign-gpu.md` | memoria per clip e dimensionamento del batch |

## 📍 Stato attuale

| | youtube_sl25 |
|---|---|
| Video scaricati / attesi | **13.455 / 39.197 (34 %)**, download fermo dal 13/10 |
| Da scaricare | 22.786 mai tentati + 296 da ritentare (`failed`, `blocked`) |
| Manifest | 635.807 clip da 12.850 video |
| Embedding | `data/youtube-sl-25/text/2fdc70f119730c4c` (592.449 didascalie uniche) |
| Clip + pose | 610.945 / 612.601 idonee (le 1.656 mancanti sono tutte "senza segnante") |
| Indice | 610.802 clip: 473.352 train · 65.928 val_channel · 61.964 test_channel · 9.558 val_video |
| Run di prova (arm A, 1 GPU) | preflight ok, epoca 1 avviata; 2,87 s/step, ~8 giorni per 15 epoche |

Il dataset è pulito (nessun video, clip o posa illeggibile) ma **incompleto**.

## 🛠️ Prerequisiti

```bash
uv sync
uv run signworld --help        # sincronizza il venv una sola volta, da solo
mkdir -p logs tmp
```

- **YouTube**: da un IP di datacenter servono un file di cookie Netscape di un account loggato
  (`youtube.cookies_file`, default `.secrets/youtube_cookies.txt`, permessi 600), `deno` nel `PATH` e il
  server del [bgutil proof-of-origin token provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
  (`youtube.pot_server_home`). I cookie attuali sono del 12/10: **vanno rinnovati** prima di riprendere.
- **EmbeddingGemma** è un modello con licenza: accettala su Hugging Face ed esporta `HF_TOKEN`
  (o `huggingface-cli login`).
- **Questa macchina**: solo la **GPU 0** è utilizzabile (96 GB); le GPU 1–3 sono in compute mode
  *Prohibited*. La configurazione di default forza IPv6, che qui non c'è: gli script usano IPv4.

Percorsi: dati in `data/youtube-sl-25` (→ `~/dataset/youtube-sl-25`), clip e pose in
`data/training/youtube_sl25`, run in `runs/` (vedi `parameters/PATHS.md`).

## ⬇️ 1. Completare il download

```bash
# tmux, finestra "dl": riprende dal ledger, IPv4, senza tetto di tempo
WORKERS=8 scripts/complete_ytsl25.sh download
```

`fetch-media` esce con 0 quando tutto è sistemato, 3 se restano elementi (lo script riprende da solo
dopo 60 s), 4 se YouTube rifiuta (bot wall): rinnova i cookie, abbassa `WORKERS` e rilancia.
Può durare giorni.

Comandi singoli, se servono:

```bash
uv run --no-sync signworld status youtube_sl25 -c tmp/acquisition_ipv4.yaml     # settled, failing, da fare
uv run --no-sync signworld probe-youtube youtube_sl25                           # rifiuto dell'account o della rete?
uv run --no-sync signworld adopt youtube_sl25 --unavailable data/youtube-sl-25/failed_unavailable.txt
```

**Stima:** al primo giro il 16 % dei tentativi è risultato non disponibile, quindi dai 22.786 rimanenti
ci si aspetta ~19.000 video, circa 2,3 volte i dati attuali.

## 🧾 2. Refresh: manifest → embedding → clip e pose → indice

A download finito (o fermato), **con nessun `fetch-media`, `text embed`, `train materialize` o
`manifest build` attivo**:

```bash
GPU=0 WORKERS=16 scripts/complete_ytsl25.sh refresh
```

Esegue, in quest'ordine fisso (F4 → F2 → F3b → F5 del piano di preprocessing):

```bash
uv run --no-sync signworld manifest build youtube_sl25 -c parameters/analysis/default.yaml
uv run --no-sync signworld text embed youtube_sl25 --device cuda -c parameters/analysis/default.yaml
GPU=0 WORKERS=16 SOURCE=youtube_sl25 OUTPUT=data/training/youtube_sl25 scripts/run_materialize_local.sh
uv run --no-sync signworld train index --manifest data/youtube-sl-25/manifest/clips.parquet \
    --materialized data/training/youtube_sl25 --embeddings data/youtube-sl-25/text/<fingerprint> \
    --output data/training/youtube_sl25/index.parquet \
    -c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml \
    -c parameters/ablation/arm_A.yaml -c parameters/model/youtube_sl25_run.yaml --workers 48
uv run --no-sync python scripts/check_datasets.py --source youtube_sl25 --deep \
    --json reports/dataset_check/youtube_sl25.json
```

Da sapere:

- **`text embed` non è incrementale** e scrive 4 file senza atomicità: se si interrompe va rilanciato
  fino in fondo. Se cambia il fingerprint, aggiorna `data.embeddings` in
  `parameters/model/youtube_sl25_run.yaml` (lo script lo segnala).
- **`train materialize`** salta le clip già fatte. Mai due processi con lo stesso `--shard` e
  `--num-shards`; dopo ogni interruzione, controllo §5.1 del piano (righe troncate in `index.*.jsonl`).
- **Stage 0 costa**: ~2,2 clip/s con 16 processi, quindi ~900.000 clip nuove sono ~4–5 giorni sulla GPU 0.
- **Gli split cambiano**: con nuovi canali l'hash per lingua ridistribuisce una parte dei canali esistenti.
  Le run sull'indice vecchio non sono confrontabili con quelle sul nuovo. Il vecchio indice viene copiato
  in `data/training/youtube_sl25/index.<data>.parquet`.

Controlli sui dati (report JSON in `collaudo/<test>/`):

```bash
uv run --no-sync python scripts/check_datasets.py --source youtube_sl25          # conteggi, ~15 s
uv run --no-sync signworld check durations youtube_sl25
uv run --no-sync signworld check split-duplicates youtube_sl25
uv run --no-sync signworld check contamination youtube_sl25       # sovrapposizione con le clip di benchmark
uv run --no-sync signworld text verify youtube_sl25 --device cuda # A2: fingerprint e ri-codifica
uv run --no-sync signworld check text-geometry youtube_sl25       # PC1
```

## 🧊 3. Congelare gli split

**Decisione da prendere prima della run dell'articolo:** partire dal corpus attuale (34 %) o
completarlo. Poi l'indice non si tocca più: `train index` scrive anche
`data/training/youtube_sl25/index.splits.csv`, da pubblicare con l'articolo.

Split (`parameters/model/youtube_sl25.yaml`, §3.4 del progetto): per lingua, canale e video;
`val_channel` (10 % dei canali di ogni lingua) per la validazione, `test_channel` (un altro 10 %,
disgiunto) per il test dell'articolo, `val_video` (2 % di video nuovi di canali visti) per il channel gap.
Nessuna lingua tenuta fuori.

## 🔬 4. Prima delle run: baseline e soglie

Una volta per indice congelato:

```bash
# PC2 sugli split della run: la ridge da feature V-JEPA 2.1 congelate
uv run --no-sync signworld train ridge-baseline \
    -c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml \
    -c parameters/ablation/arm_A.yaml -c parameters/model/youtube_sl25_run.yaml \
    --split val_channel --split test_channel --clips 5000 --device cuda \
    --output reports/youtube_sl25/ridge_baseline.json
# → copia {t2v, v2t} di val_channel in diagnostics.ridge_baseline di youtube_sl25.yaml

# arm C: le soglie di GloFND per i falsi negativi
uv run --no-sync signworld text false-negatives \
    --index data/training/youtube_sl25/index.parquet \
    --embeddings data/youtube-sl-25/text/2fdc70f119730c4c \
    --output data/training/youtube_sl25/false_negatives --device cuda
# → metti la directory in losses.false_negatives di youtube_sl25.yaml
```

Collaudo del training (§4.13.1), se non già fatto:

```bash
CFG="-c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml -c parameters/model/youtube_sl25_run.yaml"
uv run --no-sync signworld train overfit $CFG --output runs/youtube_sl25-overfit.json    # poche clip, ogni loss
uv run --no-sync signworld train benchmark $CFG --output runs/youtube_sl25-benchmark     # PC7: step, MFU, memoria
```

PC7 fissa il costo per step e quindi `training.epochs` (oggi 15, **[Aperto]**).

## 🏋️ 5. Training

Una run è la base più gli overlay, **in quest'ordine**:

```
worldsign.yaml + youtube_sl25.yaml + ablation/arm_<X>.yaml [+ ablation/<run su θ*>.yaml]
    + youtube_sl25_run.yaml [+ local_single_gpu.yaml]
```

### 5.1 In locale (GPU 0)

```bash
scripts/run_train_ytsl25.sh                     # arm A → runs/youtube_sl25-A
ARM=C scripts/run_train_ytsl25.sh               # un altro braccio
OUT=runs/smoke/youtube_sl25-A scripts/run_train_ytsl25.sh
```

Lo script fa training (`torchrun`), valutazione su `test_channel` e report; se rilanciato riprende da
`latest.pt`. Con **una sola GPU** aggiunge da solo `parameters/model/local_single_gpu.yaml` (batch 32
invece di 128): SIGReg e InfoNCE vedono un batch 4 volte più piccolo, quindi **queste run provano la
pipeline e non sono quelle dell'articolo**. Una cartella di run creata con un'altra configurazione non si
riprende (`ProvenanceError`): usa un altro `OUT` o archiviala in `runs/archived/`.

Comandi singoli:

```bash
CFG="-c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml \
     -c parameters/ablation/arm_A.yaml -c parameters/model/youtube_sl25_run.yaml"
uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node=2 --no-python \
    .venv/bin/signworld train run $CFG --output runs/youtube_sl25-A
uv run --no-sync signworld train evaluate $CFG --run runs/youtube_sl25-A \
    --index data/training/youtube_sl25/index.parquet --split test_channel \
    --output runs/youtube_sl25-A/evaluation_test_channel.json
uv run --no-sync signworld train report --run runs/youtube_sl25-A \
    --evaluation runs/youtube_sl25-A/evaluation_test_channel.json --split test_channel \
    --ridge reports/youtube_sl25/ridge_baseline.json
```

### 5.2 Sul cluster (batch 128)

Le run dell'articolo: batch effettivo 128, ViT-L, stesse epoche per ogni braccio.

```bash
condor_submit -name ettore -a 'run=youtube_sl25-A' -a 'gpus=2' \
    -a 'configs=parameters/model/worldsign.yaml parameters/model/youtube_sl25.yaml parameters/ablation/arm_A.yaml parameters/model/youtube_sl25_run.yaml' \
    slurm/condor/train.sub
condor_submit -name ettore -a 'run=youtube_sl25-A' \
    -a 'configs=parameters/model/worldsign.yaml parameters/model/youtube_sl25.yaml parameters/ablation/arm_A.yaml parameters/model/youtube_sl25_run.yaml' \
    slurm/condor/evaluate.sub

sbatch slurm/launch_train -c parameters/model/worldsign.yaml -c parameters/model/youtube_sl25.yaml \
    -c parameters/ablation/arm_A.yaml -c parameters/model/youtube_sl25_run.yaml --output runs/youtube_sl25-A
```

I `.sub` hanno ancora un `initialdir` assoluto e gli header `#SBATCH` sono segnaposto: adattali. Dati,
indice ed embedding vanno copiati sul cluster (`scripts/sync_dataset.sh`, default Leonardo).

## 📋 6. Le run da fare

Regole (`docs/worldsign-ablation.md` §1): una variabile alla volta; θ\* = il braccio migliore di ESP-1
secondo la media di R@1 T2V e V2T su `test_channel`; intervalli bootstrap sulle query. Costo in
**L** = un training completo ViT-L, braccio A.

**Ondata 1: ESP-1, le loss (in parallelo).**

| Run | Braccio | File | Stato |
|---|---|---|---|
| `youtube_sl25-A` | A: allineamento + SIGReg (run di gate) | `arm_A.yaml` | ☐ |
| `youtube_sl25-A0` | A₀: solo allineamento | `arm_A0.yaml` **(da creare)** | ☐ |
| `youtube_sl25-B0` | B₀: allineamento + L_unif | `arm_B0.yaml` | ☐ |
| `youtube_sl25-B` | B: allineamento + L_unif + SIGReg | `arm_B.yaml` | ☐ |
| `youtube_sl25-C` | C: InfoNCE + SIGReg (serve `losses.false_negatives`) | `arm_C.yaml` | ☐ |

**Ondata 2: ablation su θ\*** (l'overlay va **dopo** `arm_<θ*>.yaml`).

| Run | Esperimento | File | Costo | Stato |
|---|---|---|---|---|
| ESP-2 | nessun livello fisico | `esp2_no_physical.yaml` | ≈ 0,85 L | ☐ concordata |
| ESP-6 | target video V-JEPA 2.1 invece della posa | `esp6_video_target.yaml` | ≈ 1 L + target | ☐ concordata |
| ESP-4 | K = 4 ipotesi, energia libera rilassata | `esp4_latent.yaml` | ≈ 1 L | ☐ facoltativa |
| ESP-8 | VICReg al posto di SIGReg | `arm_V.yaml` | ≈ 1 L | ☐ rimandata |
| D4 | secondo seed di θ\* | `d4_seed1.yaml` | ≈ 1 L | ☐ proposta |
| G1 | gerarchia «globale» | `g1_global.yaml` | ≈ 1 L | ☐ proposta |

Totale concordato ≈ 6,85 L. Con le date di CVPR (registrazione 10/11, invio 16/11/2026) e ~8 giorni per
run su una GPU, le run dell'articolo vanno fatte sul cluster.

Tabelle delle ablation (contrasti pianificati, bootstrap appaiato sulle stesse query):

```bash
uv run --no-sync signworld train compare \
    --run runs/youtube_sl25-A  --evaluation runs/youtube_sl25-A/evaluation_test_channel.json \
    --run runs/youtube_sl25-B0 --evaluation runs/youtube_sl25-B0/evaluation_test_channel.json \
    --run runs/youtube_sl25-C  --evaluation runs/youtube_sl25-C/evaluation_test_channel.json \
    --output reports/youtube_sl25/esp1
```

## ✅ Checklist

1. ☐ Cookie YouTube rinnovati in `.secrets/youtube_cookies.txt`
2. ☐ `scripts/complete_ytsl25.sh download` fino all'uscita 0 (o decisione di fermarsi al 34 %)
3. ☐ `scripts/complete_ytsl25.sh refresh`, check profondo pulito
4. ☐ `data.embeddings` in `youtube_sl25_run.yaml` allineato al nuovo fingerprint
5. ☐ Indice e `index.splits.csv` congelati
6. ☐ `ridge-baseline` → `diagnostics.ridge_baseline`; `false-negatives` → `losses.false_negatives`
7. ☐ PC7 sul cluster → `training.epochs`
8. ☐ `arm_A0.yaml` creato
9. ☐ Ondata 1 (ESP-1) → θ\*
10. ☐ Ondata 2 (ablation su θ\*) → `train compare` → tabelle dell'articolo

## 🗂️ File di YouTube-SL-25

```
parameters/model/youtube_sl25.yaml       split per canale, epoche, baseline, falsi negativi
parameters/model/youtube_sl25_run.yaml   percorsi di stage 0 (indice, embedding, manifest)
parameters/model/local_single_gpu.yaml   batch 32 per la prova su una GPU (non per l'articolo)
scripts/complete_ytsl25.sh               download | refresh | check
scripts/run_train_ytsl25.sh              training + valutazione + report in locale
scripts/run_materialize_local.sh         stage 0 a shard sulla GPU locale
scripts/check_datasets.py                controlli su download, manifest, embedding, clip, indice
data/youtube-sl-25/                      video, ledger, manifest, embedding
data/training/youtube_sl25/              clip, pose, index.parquet, index.splits.csv
reports/dataset_check/                   esiti di check_datasets.py
runs/youtube_sl25-<braccio>/             le run
```
