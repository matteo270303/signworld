# 1.14 — Verify

Verdetto: **PASS_WITH_CAVEATS**. La prova vera è un job multi-GPU: `exit_on_error` con più rank non gira nei test (CPU, un processo).

| Criterio | Evidenza |
|---|---|
| Drift | `git diff --cached --stat`: solo `distributed.py` e `cli/train.py` per questo task; `evaluate` resta su `caption_language` (`git grep written_language signworld/cli/train.py` → nessun risultato) |
| Test | `pytest tests/experiment/train/` → 58 passed, 5 skipped (lenti) |
| Lint/tipi | ruff su `distributed.py` → solo E501 a riga 39, già presente prima del riporto (riga 36 su e78283e); `ruff format --check` → ok |
