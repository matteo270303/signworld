Verdetto: PASS_WITH_CAVEATS

- Comandi sbatch invariati: `batch/debug.sh`, `wave1.sh`, `wave2.sh` con un `sbatch` finto vs il vecchio `launch_experiment.sh debug|train wave1|train esp2 esp6 g1 d4` → `diff` identico (solo `experiment_job` → `launch_experiment`).
- `bash -n` ok su tutti i `launch_*`, `setup.sh`, `experiments.sh`; `sbatch --test-only` accettato per le 6 coppie.
- Nessun riferimento rotto: grep di `launch_experiment.sh|experiment_job|experiment_exe|slurm/pilot` fuori da work/drafts → vuoto.
- Caveat: mai eseguito su Leonardo; `lrd_all_serial` con rete e `setup.sh`/`launch_analysis` offline non provati; `--mail-user=<EMAIL>` ancora nei 4 header vecchi.
