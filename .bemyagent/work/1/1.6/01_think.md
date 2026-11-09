# 1.6 — Debug 63997298/63997299: perché si ferma al passo 4, warning, tqdm, log P

## Context

Debug del 09/11/2026 (`out/sw-debug-A-gate-63997298.log`, `runs/debug/A-gate/`), walltime 30 min, stop a 24 min (`STOP_MARGIN_MIN=6`):

| fase | inizio → fine | durata |
|---|---|---|
| avvio + statistiche (1.5) | 17:06:19 → 17:07:32 | ~1 min |
| P1-P12 | → 17:07:50 | 18 s |
| P13 (overfit 30 passi, modello intero con LoRA video) | → 17:15:17 | 7,5 min (~15 s/passo) |
| P14-P17 | → 17:15:33 | 16 s |
| validazione passo 0 | → 17:17:06 | 1,5 min |
| **diagnostiche rare passo 0** (`monitor.rare`) | → 17:31:24 | **14,3 min** |
| passi 1-4 (stadio P, 4,1 s/passo) | → 17:31:53 | 30 s |

Stop deciso a 17:30:19, controllato solo ai passi di log → si ferma al passo 4. Il secondo segmento (63997299) esce con `ProvenanceError`: `pyproject.toml`/`uv.lock` modificati alle 17:21 (indice cu126) → hash del codice diverso dal primo lancio.

Causa dei 14 min: `Monitor.rare` e `plausibility_tests` girano **senza autocast bf16** (validazione, P13, P17, letture dei gradienti lo hanno). In fp32 la ViT-L a 384 × 64 frame fa 6 varianti × 8 maschere = 48 forward fisici + 2 semantici su 32 clip, senza flash attention.

Log P: ogni rank stampa P1-P17 (×4) e i SKIP due volte (riga + `enforce`); il testo non dice cosa si controlla né quanto dura.

Warning: `preflight.py:347` `float(total)` con grad (nostro); DDP `broadcast_buffers` deprecato in torch 2.14 (nostro); `timm.models.layers` e `sdp_kernel()` dal codice Meta in `checkpoints/hub/` (×4).

### Context Saturation Check

- [x] **Target files/paths**: `experiment/train/monitor.py`, `models/worldsign/plausibility.py`, `experiment/train/preflight.py`, `experiment/train/trainer.py`, `experiment/train/run.py`
- [x] **Expected behavior**: stesse letture (a meno della precisione bf16 nelle rare), log P una volta con descrizione e secondi, barre tqdm sulle fasi lunghe
- [x] **Constraints**: nessuna sbatch da parte mia (utente); niente collective nuovi; `broadcast_buffers=False` mantenuto (BN locali per rank)
- [x] **Dependencies**: nessuna (tqdm già usato)

Assunzione: le letture rare in bf16 sono accettabili come lo sono validazione e P13 (le energie sono già riportate in float).

## Approaches Considered

- A. autocast bf16 in `rare` + `plausibility_tests`; log preflight sul primo GPU con descrizione e durata; tqdm su P13, preflight, plausibilità, pose delle statistiche; warning: fix dei nostri, filtro mirato per quelli Meta.
- B. Saltare le rare al passo 0 / ridurre `plausibility_masks` nel debug. Cambia le diagnostiche, non la causa.

## Selected Approach & Risks

- A. Rischio: rank ≠ 0 che falliscono un'asserzione non si vedono → i FAIL si loggano su ogni rank con il numero di rank.
- Rischio: `forward_sync_buffers` cambierebbe la semantica (sync dei buffer a ogni ricostruzione DDP) → si tiene `broadcast_buffers=False` e si silenzia solo quel warning.

Pre-mortem: (1) bf16 cambia il dtype dell'output e rompe `torch.stack` → le energie sono già `.float()`; (2) tqdm su rank ≠ 0 duplica le barre → `disable=not is_main`; (3) filtro warning troppo largo nasconde altro → filtro per `message` esatto.

Devil's Advocate: togliere del tutto le rare dal passo 0. No: il passo 0 è la linea di base delle rare; il costo era il fp32, non la frequenza.

## Verification Plan

- `pytest tests/experiment/train tests/models/worldsign` + ruff/mypy sui file toccati
- Prossimo debug lanciato dall'utente: rare < 2 min, P una volta per riga con secondi
