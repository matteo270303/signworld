# 1.12 — Verify

Verdetto: **PASS_WITH_CAVEATS**. La prova vera è il prossimo job sul cluster: l'append concorrente su Lustre (task srun + client srun + sbatch) non si può simulare in locale.

| Criterio | Evidenza |
|---|---|
| Sintassi | `bash -n slurm/launch_experiment slurm/batch/*.sh` → ok; `python -m py_compile slurm/tools/run_report.py` → ok |
| wave1 | sbatch finto → `--output=runs/A-gate-20261116-120000/logs/train-%j.log --open-mode=append` |
| debug | sbatch finto → `--output=runs/debug/logs/debug-%j.log`; srun finti → `runs/debug/{A-gate,B}-20261116-120000/logs/debug-999.log` |
| pilot | sbatch finto → `--output=runs/pilot-A-20261116-120000/logs/train-%j.log` |
| Errore di srun nel log del run | srun finto che esce 1 → `runs/A-gate-…/logs/train-998.log` contiene `srun: error: fake: task 0: Exited with exit code 1`, `launch_experiment` esce 1 |
| Provenance | `SOURCE_PARTS` = signworld, parameters, pyproject.toml, uv.lock: `slurm/` escluso, il resume delle run esistenti non viene rifiutato |

Nota fuori scope: `MAIL` è anche la variabile Unix della casella locale (`/var/spool/mail/<utente>`), quindi gli script di batch passano `--mail-user=/var/spool/mail/…` anche senza che nessuno l'abbia impostata.

Riporto su e78283e (24/11/2026): `git apply -3` pulito; `pytest tests/experiment/train/` → 58 passed, 5 skipped (lenti); ruff/mypy sui file toccati → solo errori già presenti prima del riporto (E501 `distributed.py`, PTH105 `preflight.py`, type-arg `run_report.py`).

Riporto verificato: `bash -n` su `slurm/batch/*.sh` e `slurm/launch_experiment` → ok; `py_compile run_report.py` → ok.
