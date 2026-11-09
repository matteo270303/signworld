# 1.6 — Verify

Verdetto: **PASS_WITH_CAVEATS** — il guadagno di tempo sul nodo GPU si vede solo al prossimo debug (niente sbatch da parte mia).

| Criterio | Evidenza |
|---|---|
| Test verdi | `pytest tests/experiment/train tests/models/worldsign tests/data/test_loaders.py tests/cli tests/experiment/evaluation` → 258 passed, 4 skipped |
| Nessun `float(` su tensori con grad | `grep -rn "float(total)" signworld` → solo `metrics/readings.py` (no grad); P13 usa `float(total.detach())` |
| Autocast in rare/plausibility | grep → `monitor.py:392`, `plausibility.py:175` |
| Log P una riga per codice, solo primo GPU, FAIL con rank | `test_the_log_has_one_line_per_assertion_with_what_it_checks` passa |
| Lint | ruff: solo PTH105 a `preflight.py:203`, preesistente in HEAD; format ok; mypy: solo errori preesistenti |
| Drift | le asserzioni misurano le stesse cose; `broadcast_buffers=False` invariato; nessun job lanciato |

Caveat:
- P13 resta ~7,5 min a run nuova (30 passi del modello intero): ora ha la barra; ridurre `preflight_overfit_steps` nel debug è una scelta dell'utente.
- Il test lento di P13 fallisce già su HEAD (annotato in 05).
