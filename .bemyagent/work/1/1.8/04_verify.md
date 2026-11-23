# 1.8 — Verify

Verdetto: **PASS_WITH_CAVEATS**. Le misure su GPU sono a batch 2-4, con la GPU 0 condivisa col run dell'utente (al 100 %): i rapporti sono indicativi, i tempi assoluti vanno rifatti a GPU libera.

| Criterio | Evidenza |
|---|---|
| Test | `pytest tests/experiment/collaudo/test_profiling.py` → 1 passed |
| Lint | `ruff check` + `ruff format` sui 3 file → All checks passed; `mypy` (strict) su `profiling.py`, `cli/train.py` → no issues |
| Gira sul run vero | `train profile ... --clips 2 --stages P --steps 1 --data-clips 5 --loader-batches 2` → tabella + `profile.json`; `--no-gpu --data-clips 100` → fasi della clip |
| Figlie ≤ padre | P b4: forward 0,623 ≥ semantic 0,584 ≥ encoder_full 0,521; F b4: physical 0,343 ≥ encoder_masked 0,184 + predictor 0,150 |

Caveat: il comando va rilanciato a GPU libera, a batch 40 (vedi il draft `lentezza-stage-f.md`).
