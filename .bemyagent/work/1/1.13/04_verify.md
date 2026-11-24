# 1.13 — Verify

Verdetto: **PASS_WITH_CAVEATS**. La prova vera è il prossimo lancio sul cluster con 4 GPU: `collective.mean` e `collective.any` non girano in multi-processo nei test.

| Criterio | Evidenza |
|---|---|
| Drift | `git diff --stat`: solo `preflight.py`, `run.py`, `test_preflight.py`; nessun campo in `config.py`, quindi `config_hash` invariato |
| Verdetto sulle curve del cluster | `overfit_verdict`: job 64009434 rank 0 → PASS 0.45; debug 64003252 → PASS 0.21; finale del rank 3 (2.54) → WARN 0.51; loss piatta → FAIL; NaN → FAIL |
| WARN non ferma, FAIL altrove sì | `test_a_warning_does_not_stop_the_run_a_failure_elsewhere_does` → passa |
| Test veloci | `pytest tests/experiment/train/test_preflight.py -m "not slow"` → 12 passed, 1 skipped |
| Lint/tipi | `ruff check` su run.py e sui test → ok; su preflight.py resta solo PTH105 a riga 207 (preesistente); `mypy preflight.py run.py` → ok |
| Test lento P13 | `SIGNWORLD_SLOW_TESTS=1 … -k single_batch_overfit` → FAIL preesistente, ora WARN invece di FAIL (setup tiny, 40 passi). Test non modificato, annotato in 05 |

Caveat: un'eccezione (non un FAIL) su un solo rank dentro il preflight bloccherebbe ancora gli altri fino al timeout NCCL.

Riporto su e78283e (24/11/2026): `git apply -3` pulito; `pytest tests/experiment/train/` → 58 passed, 5 skipped (lenti); ruff/mypy sui file toccati → solo errori già presenti prima del riporto (E501 `distributed.py`, PTH105 `preflight.py`, type-arg `run_report.py`).
