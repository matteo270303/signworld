# 1.10 — Lingua persa nel centring: KeyError `en-xgJ5vofi3O8` a metà run

## Context

Run locale `runs/youtube_sl25-A-1gpu-b40` (23/11/2026, 1 GPU, batch 40, `out/train_1gpu_b40.log`): morta alle 13:39 a ~passo 1040 dello stadio P con `KeyError: 'en-xgJ5vofi3O8'` in `Collate` (`signworld/data/loaders.py:350`). Nessun checkpoint (`validation_every: 4000`).

Causa: `caption_statistics` deduplica per `caption_row` (`{caption_row: caption_language}`, vince l'ultima). Nel training di `data/training/youtube_sl25/index.parquet` (945.688 clip, 143 lingue) 3.607 `caption_row` sono condivise da più lingue (stesso testo → stessa riga dello store); le 5 clip di `en-xgJ5vofi3O8` hanno tutte righe condivise → la lingua sparisce: `statistics.pt` ne ha 142. `Collate` è costruito con le lingue del centring → KeyError appena una di quelle clip entra in un batch. Bug preesistente (stessa riga prima di 1.5), non legato alle modifiche locali (test + `local_single_gpu_max.yaml`).

### Context Saturation Check

- [x] **Target files/paths**: `signworld/data/loaders.py` (`caption_statistics`), `signworld/data/negatives.py`, `signworld/experiment/train/run.py` (`prepare`), `tests/data/test_loaders.py`
- [x] **Expected behavior**: ogni `caption_language` del training ha la sua media; una lingua mancante ferma la run all'avvio, non dopo ore
- [x] **Constraints**: nessun rilancio (utente); soglie GloFND (`negatives.py`) invariate bit a bit (file già calcolati per il braccio C); niente normalizzazione dei tag `en-<id>` (decisione rimandata)
- [x] **Dependencies**: nessuna

## Approaches Considered

1. **Scelto** — `caption_statistics` dedup per coppia `(caption_row, caption_language)`: un testo condiviso conta una volta per lingua nella media di ciascuna. `negatives.py` passa a `distinct_captions` (vecchia semantica, una riga per caption), perché allinea `captions[i]` a `sorted(set(caption_row))[i]`.
2. Scartato — normalizzare `en-<id>` → `en`: cambia il modello (≈100 medie in meno), decisione dell'utente, non serve a togliere il bug.
3. Scartato — fallback in `Collate` sulla lingua base: nasconde l'errore, media sbagliata in silenzio.

## Pre-mortem

- `negatives.py` riceve righe duplicate e disallinea le soglie in silenzio → funzione separata con la vecchia semantica + test che `distinct_captions` = `embeddings[rows]`.
- `statistics.pt` vecchio riusato al rilancio nella stessa cartella → controllo all'avvio in `prepare` che elenca le lingue mancanti e dice di cancellare `statistics.pt`.
- Lettura a blocchi rotta con righe ripetute → test `block=2` sui duplicati.

## Devil's Advocate

Contare ogni caption condivisa una sola volta e assegnarla a tutte le sue lingue è esattamente l'approccio scelto; l'alternativa "una lingua per riga, ma scelta deterministica" lascerebbe comunque lingue senza righe. Nessun pivot.
