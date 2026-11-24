# 1.13 — P13: verdetto unico fra le GPU, criterio robusto, non blocca per lentezza

## Context

Job 64009434: P13 (overfit di un batch, la loss deve dimezzarsi in 30 passi) è PASS su rank 0 (4.73 → 1.78) e FAIL su rank 3 (4.73 → 2.54). Il rank 3 esce, gli altri entrano nella validazione e restano fermi su una `all_reduce` per 60 min (timeout NCCL). Problemi:
1. il verdetto è per rank: `enforce` decide sul report locale;
2. il criterio guarda solo l'ultimo passo di una curva rumorosa (lr 1e-3, bf16): la soglia 0.5 diventa un lancio di moneta;
3. un FAIL per lentezza ferma la run per sempre (ogni rilancio rifà P13 da zero, run non ancora partita).

Lo scopo di P13 nel progetto (`docs/worldsign-progetto.md:1263`): "se non riesce: bug nel grafo". Un bug nel grafo vuol dire una loss che non scende, non una che scende piano.

### Context Saturation Check

- [x] **Target files/paths**: `signworld/experiment/train/preflight.py`, `signworld/experiment/train/run.py`, `tests/experiment/train/test_preflight.py`
- [x] **Expected behavior**: tutte le GPU hanno lo stesso verdetto su P13; se una GPU fallisce un controllo, si fermano tutte subito; P13 blocca solo se la loss non scende (≥ 0.9 × iniziale), avvisa se scende senza dimezzarsi, passa se si dimezza
- [x] **Constraints**: niente campi nuovi nella config (cambierebbero `config_hash` e le cartelle esistenti verrebbero rifiutate); ogni collettiva chiamata da tutti i rank nello stesso ordine
- [x] **Dependencies**: nessuna

## Approaches Considered

- Soglia più bassa (es. 0.6): resta un lancio di moneta vicino alla soglia, e continua a bloccare per lentezza.
- Flag di config `p13: warn|fail`: cambia l'hash della config, quindi serve una cartella nuova.
- **Scelto**:
  - loss di P13 mediata fra le GPU a ogni passo (`collective.mean`): una curva sola, quindi un verdetto solo;
  - il criterio usa la media degli ultimi 5 passi divisa per la loss al passo 0;
  - tre esiti: PASS ≤ 0.5, WARN fra 0.5 e 0.9 (nuovo `Status.WARN`: log ben visibile, la run parte), FAIL > 0.9 o loss non finita;
  - in `run.py`, `collective.any(not passed)` prima di `enforce`: se un rank fallisce un controllo qualsiasi, escono tutti con l'errore invece di restare bloccati.

## Pre-mortem

- Collettive non allineate: P13 gira su ogni rank con lo stesso numero di passi; `mean` viene chiamata a ogni passo da tutti. `any` viene chiamata da tutti prima di `enforce`.
- `Status.WARN` letto altrove come FAIL: `passed` controlla solo FAIL; `enforce` lo elenca come lo SKIP.
- Test esistente che cerca un solo record ERROR: WARN usa `logger.warning`, quindi il test resta valido.

## Devil's Advocate

Togliere P13 dal blocco, cioè sempre e solo WARN: perderebbe l'unico controllo che il grafo trasmetta gradienti. Con FAIL solo a > 0.9 un grafo rotto ferma comunque la run, e il rumore no.
