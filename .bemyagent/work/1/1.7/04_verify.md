# 1.7 — Verify

Verdetto: **PASS** (la prova vera è il prossimo `slurm/batch/debug.sh` sul cluster).

| Criterio | Evidenza |
|---|---|
| Sintassi | `bash -n slurm/experiments.sh slurm/launch_experiment slurm/batch/*.sh` → ok |
| Stamp nuova | `source experiments.sh; run_dir debug A-gate` → `runs/debug/A-gate-20261109-181258`; esp2 → `runs/esp2-on-A-20261109-181258`; pilot → `runs/pilot-A-20261109-181258` |
| Ereditata dai job | `RUN_STAMP=20261109-120000` + `bash -c 'source …; run_dir debug B'` → `runs/debug/B-20261109-120000` |
| Cartelle vecchie | `RUN_STAMP= …; run_dir debug A-gate` → `runs/debug/A-gate` |
