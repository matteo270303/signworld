# Mappa del codice

## Percorsi critici / prestazioni

- `signworld/experiment/train/trainer.py` — ciclo dei passi di addestramento (DDP tramite `distributed.py`)
- `signworld/models/worldsign/video_branch.py` — V-JEPA 2.1 ViT-L congelato + LoRA, 8.192 token per clip
- `signworld/data/loaders.py` — caricamento di clip/pose (decord)
- `signworld/data/corpus/materialize.py`, `signworld/data/pose/estimator.py` — preprocessing dello stadio 0

## Copertura dei test

`tests/` rispecchia il pacchetto: `cli/`, `data/`, `experiment/`, `loss/`, `metrics/`, `models/`. Alcuni test richiedono il codice hub di V-JEPA in `checkpoints/hub/...` (`tests/worldsign.py`). Livello di copertura ?

## Pagine / Route

N/A — solo CLI. Ingresso: `signworld.cli:app` (`signworld/cli/__init__.py`), sotto-app in `signworld/cli/{acquisition,corpus,checks,experiment,train}.py`.

## Funzioni chiave

| Funzione/File | Ruolo/Ingressi | Uscite/Effetti collaterali |
| ------------- | -------------- | -------------------------- |
| `signworld/data/acquisition/sources/*.py` | adattatore per dataset | metadati + media in `data/<source>/` |
| `signworld/data/acquisition/ledger.py` | esiti dei download | `ledgers/shard-*.jsonl` |
| `signworld/data/corpus/builders.py` | costruttori di manifest per source | `manifest/clips.parquet` |
| `signworld/data/corpus/materialize.py` | manifest → clip + pose | `data/training/<source>/` |
| `signworld/models/worldsign/model.py` | assemblaggio di WorldSign | embedding, energie |
| `signworld/loss/worldsign.py` | loss per livello | loss scalari per livello |
| `signworld/experiment/train/config.py` | fusione degli overlay YAML | configurazione della run |
| `signworld/experiment/evaluation/worldsign.py` | run + indice + split | JSON di retrieval |
| `scripts/check_datasets.py` | audit di download, manifest, embedding, clip/pose, indice (`--deep`: ffprobe + pose) | `reports/dataset_check/<source>/*.txt` |
| `scripts/complete_ytsl25.sh` | `download` / `refresh` / `check` di YouTube-SL-25 (F1→F4→F3b→F5) | ledger, manifest, embedding, clip, indice |
| `scripts/run_train_*.sh` | indice (se manca o senza split `test`) + train + evaluate + report | `runs/<source>-<arm>/` |

## Schemi dei dati

### Schema: manifest delle clip (`clips.parquet`)

| Campo | Tipo | Esempio | Proprietario |
| ----- | ---- | ------- | ------------ |
| ? | ? | ? | `signworld/data/corpus/manifest.py` |

### Schema: indice di addestramento (`index.parquet`)

| Campo | Tipo | Esempio | Proprietario |
| ----- | ---- | ------- | ------------ |
| ? | ? | ? | `signworld/cli/train.py` (`train index`) |

> Segnalare le discrepanze di schema con ⚠️ — la prima causa di bug di integrazione.
