## Context

- `fit_statistics` (`signworld/experiment/train/run.py:43`) lo esegue solo il rank 0, mentre gli altri aspettano alla barrier. Su 945.688 clip ci mette ~10,5 min e fa scadere il timeout NCCL (`05-decisions-and-issues.md`, job 63260872). Obiettivo: scendere a ~1 min senza cambiare i numeri.
- Profilazione sul login node (09/11/2026, script in `out/prof_stats.py`):
  - pose: 20.000 `.npz` letti in serie, 25 ms/clip → ~500 s; con thread (16-32) 1,4-1,7 ms/clip → ~30 s
  - `embeddings[ordered]` su mmap da 3,4 GB → 193 s (anche `np.load` intero 193 s: login node limitato dalla banda)
  - `read_index` ~25 s a lettura, chiamato 2 + 1 + N_split volte
  - `caption_statistics` ricalcolata in `run()` per P11 su ogni rank

### Context Saturation Check

- [x] **Target files/paths**: `signworld/data/loaders.py` (`keypoint_statistics`, `caption_statistics`, `read_index`), `signworld/models/worldsign/text_branch.py` (`LanguageCentering.fit`), `signworld/experiment/train/run.py` (`fit_statistics`, `prepare`, `run`), `tests/data/test_loaders.py`
- [x] **Expected behavior**: stessi clip, stesse statistiche (uguali bit a bit per caption e varianza; medie per lingua a meno dell'ordine di somma in float64)
- [x] **Constraints**: §4.6, solo clip di training; P5, `statistics.pt` già salvati non si ricalcolano; stessa API pubblica per `cli/train.py`, `negatives.py`, `tests/worldsign.py`
- [ ] **Dependencies**: nessuna nuova (stdlib `concurrent.futures`)

Assunzione: la caption → lingua resta il dizionario attuale (vince l'ultima occorrenza), invariato.

## Approaches Considered

- A. Thread sulle pose + lettura a blocchi contigui degli embedding + caption/indice calcolati una volta in `prepare`. Pro: locale, output identico. Contro: tiene in RAM 2,7 GB di caption per rank fino alla fine.
- B. Dividere il campione di pose tra i rank e fare all-gather. Pro: scala con i GPU. Contro: collective dentro il calcolo, il caso a GPU singola resta lento, è più fragile.
- C. Cache condivisa delle statistiche tra le run (task successivo, non adesso).

## Selected Approach & Risks

- A. Costo stimato: medio.
- Rischio: i thread cambiano l'ordine dei risultati. Mitigazione: `executor.map` preserva l'ordine, e un test lo confronta con la versione seriale.
- Rischio: la lettura a blocchi non è più veloce del fancy index su un nodo di calcolo. Mitigazione: misura nel job 63996194 prima di decidere; se non c'è guadagno, lo si tiene solo per evitare la copia in double.
- Rischio: RAM. 2,7 GB × 4 rank su 512 GB per nodo, accettabile; i worker forkati condividono le pagine (COW).

Pre-mortem: (1) i thread saturano Lustre e rallentano gli altri job → numero di worker configurabile, default 16; (2) `index_add_` in float64 cambia l'ultimo bit delle medie float32 → test con `allclose` contro la versione a maschere; (3) `prepare` usato da `cli/train.py` rompe la sua firma → `Prepared` guadagna campi, nessun argomento nuovo obbligatorio.

Devil's Advocate: invece di ottimizzare, calcolare le statistiche offline con un comando CLI prima di sottomettere. Non lo scegliamo: la run resta autosufficiente, e il punto C copre il riuso.

## Verification Plan

- `pytest tests/data/test_loaders.py tests/models/worldsign/test_text_branch.py tests/experiment/train/test_preflight.py`
- Equivalenza sui dati veri su un nodo GPU: nuove statistiche vs `runs/A-gate.failed/statistics.pt` (stesso `statistics_clips`), più il tempo totale
