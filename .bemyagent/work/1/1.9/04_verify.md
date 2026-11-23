# 1.9 — Verify

Verdetto: **PASS_WITH_CAVEATS**: non provato sul cluster (H100, DDP a 4 GPU).

| Criterio | Evidenza |
|---|---|
| Loss e gradienti = eager | `equiv.py`, modello vero, passo F su 2 clip. fp32: loss identiche, errore relativo mediano dei gradienti 0,0001 (p95 0,0001, 525 tensori). bf16: distanza da fp32 1,41 compilato contro 1,40 eager. `test_compiled_blocks_give_the_eager_losses_and_gradients` (slow, tiny, checkpointing + dropout, 2 maschere diverse) → pass |
| Più veloce, stessa memoria | `train profile` b40, GPU libera: P 3,32 → 2,58, F0 8,90 → 6,26, F 12,49 → 8,37 s; picco 32,9/51,5/62,5 GiB invariato |
| Run end-to-end | `train run` con compile (P13 10 passi, validazione ogni 60, rare ogni 120): P1-P17 PASS (P13 4,74 → 1,92), validazioni ai passi 0/60/120/180, rare ai passi 0 e 120, `latest` salvato, stop pulito al passo 220 (exit 0); 2,60 s/passo wall in P contro 3,34 eager; 24 ricompilazioni, limite mai raggiunto |
| Codice Meta intatto | nessun file in `checkpoints/hub/` modificato (symlink nel worktree, ignorato da git) |
| Test | `tests/models tests/experiment tests/cli tests/data/test_loaders.py` → passano tutti tranne `test_the_checksum_record_is_published_whole` (fallisce anche su HEAD, preesistente) e `test_results` (cartella `collaudo/` assente nel worktree) |
| Lint | ruff check/format e mypy strict puliti sui 6 file di `signworld/` toccati |

Caveat:
- DDP e H100 non provati: i blocchi compilati stanno dentro il modulo che DDP avvolge, nessun cambio ai collective, ma va visto al prossimo debug.
- Ogni forma nuova costa 5-30 s di compilazione la prima volta (P13, validazione, rare, cambio di stage).
- Resume di run lanciati prima di questo cambio: configurazione e codice diversi → `--allow-code-change` e `encoder.compile_blocks: false`, oppure una cartella nuova.
