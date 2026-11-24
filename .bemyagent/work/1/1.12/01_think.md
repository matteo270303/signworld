# 1.12 — Log nella cartella del run

## Context

I log degli esperimenti finivano in `out/`, separati dalla run: il log del job (`out/sw-train-A-gate-<job>.log`, solo la riga `srun: error ... exit code 1`) e quello del programma (`out/sw-train-A-gate-<data>-<job>.log`). Con il job 64009434 non era chiaro quale leggere. Richiesta: i log in `runs/<run>/logs/<file>.log`.

### Context Saturation Check

- [x] **Target files/paths**: `slurm/launch_experiment`, `slurm/batch/{debug,wave1,wave2,pilot}.sh`, `slurm/tools/run_report.py`, `slurm/README.md`
- [x] **Expected behavior**: per ogni job e run un solo file `runs/[debug/]<run>/logs/<mode>-<job>.log`, con l'output del programma più i messaggi di srun e dello script sbatch
- [x] **Constraints**: slurm non crea le cartelle di `--output` → `mkdir -p` prima dello sbatch / srun; i job di debug coprono più run in un job solo
- [x] **Dependencies**: nessuna. `slurm/` non è in `SOURCE_PARTS` della provenance, quindi il resume delle run esistenti non si rompe

## Approaches Considered

- Due file per job (`sbatch-<job>.log` + `<mode>-<job>.log`): scartato, sono gli stessi due file confusi di prima.
- **Scelto**: un solo file per job in `runs/<run>/logs/`, aperto in append da tre scrittori: l'output di sbatch (wave/pilot: un esperimento per job), il task di srun (`--output … --open-mode=append`) e lo stdout/stderr del client srun (`>>`, dove compare `srun: error: … exit code N`). Debug (più run per job): l'output di sbatch va in `runs/debug/logs/debug-<job>.log`, mentre ogni run ha il suo `logs/debug-<job>.log` con il programma e i messaggi di srun.

## Pre-mortem

- La cartella `logs/` manca al momento di sbatch → job FAILED senza log: `mkdir -p` negli script di batch prima di `sbatch`.
- `run_report.py debug` cerca i log in `out/`: aggiornato a `run/logs/*.log`.
- Scrittori concorrenti sullo stesso file: tutti in append, e i messaggi di srun e dello script arrivano prima o dopo il task, non durante.

## Devil's Advocate

Lasciare i log in `out/` e scrivere nel run un symlink: due posti da ricordare, e `out/` continua a crescere. Il file nel run è quello che l'utente ha chiesto.
