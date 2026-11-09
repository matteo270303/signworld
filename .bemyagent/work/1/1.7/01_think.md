# 1.7 — Cartelle dei run con la data del lancio

## Context

`run_dir` (`slurm/experiments.sh`) dava `runs/[debug/]<esp>`: rilanciare lo stesso esperimento riusava la cartella → resume del run vecchio, `ProvenanceError` se il codice è cambiato (debug 63997299), `FAILED` che blocca il lancio. Richiesta: data dopo il nome del run.

### Context Saturation Check

- [x] **Target files/paths**: `slurm/experiments.sh`, `slurm/launch_experiment`, `slurm/batch/{debug,wave1,wave2,pilot}.sh`, `slurm/README.md`
- [x] **Expected behavior**: ogni lancio di batch → `runs/[debug/]<esp>-<YYYYmmdd-HHMMSS>`; i segmenti della stessa catena (debug a 2, wave a 6) nella stessa cartella
- [x] **Constraints**: il resume fra segmenti non deve rompersi; cartelle vecchie ancora raggiungibili
- [x] **Dependencies**: nessuna

## Approaches Considered

- Data calcolata in `launch_experiment_exe`: scartato, ogni segmento avrebbe la sua cartella e il resume si perderebbe.
- **Scelto**: `RUN_STAMP` esportata da `experiments.sh` con `${RUN_STAMP-$(date …)}` al momento del `source` nello script di batch; `sbatch` esporta l'ambiente (default `--export=ALL`), quindi tutti i job della catena la ereditano. `RUN_STAMP=<data>` riprende, `RUN_STAMP=` vuota → nome senza data.

## Pre-mortem

- sbatch senza ambiente → ogni job una data diversa: nessun `--export` negli script; `launch_experiment` fa anche `source` per fissarla una volta per job se lanciato a mano.
- `run_report.py debug` cerca `out/sw-debug-<run.name>-*.log`: il nome del log ora include la data, quindi coincide con il nome della cartella.

## Devil's Advocate

Contatore incrementale (`A-gate-2`, `-3`): leggibile ma richiede di guardare il filesystem e non dice quando; la data è quello che ha chiesto l'utente.
