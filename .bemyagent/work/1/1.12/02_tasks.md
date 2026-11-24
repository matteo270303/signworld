Delivers: dopo `slurm/batch/wave1.sh A-gate` il log di ogni job (programma + `srun: error`) sta in `runs/A-gate-<data>/logs/train-<job>.log`; dopo `slurm/batch/debug.sh` in `runs/debug/<esp>-<data>/logs/debug-<job>.log`.

- [x] `launch_experiment`: `mkdir -p <run>/logs`, srun `--output=<run>/logs/<mode>-%j.log --open-mode=append`, client srun `>>` sullo stesso file
- [x] `wave1.sh`, `wave2.sh`, `pilot.sh`: `mkdir -p <run>/logs`, sbatch `--output=<run>/logs/train-%j.log --open-mode=append`
- [x] `debug.sh`: sbatch `--output=runs/debug/logs/debug-%j.log`
- [x] `run_report.py debug`: log da `run/logs/*.log`
- [x] README (slurm e radice), 03, 06

CDM
- ✅ Validation: `bash -n` sugli script; `python -m py_compile run_report.py`; simulazione con `sbatch`/`srun` finti che stampano le opzioni ricevute → percorsi attesi
