# Riordino di `slurm/` (proposta, non approvata)

## Stato attuale (verificato)

- 12 launcher a coppia `launch_X` (header `#SBATCH` + `srun`) / `launch_X_exe` (env + comando): `train`, `multigpu_train`, `evaluate`, `analysis`, `fetch_media` (+ `_exe`). `launch_multigpu_train` usa `launch_train_exe`.
- `launch_experiment.sh` (launcher monolitico, 166 righe) + `experiment_job` + `experiment_exe` + `experiments.sh`: il sistema vero (debug, catene `afterok`, resume a scadenza, mai provato su Leonardo). `launch_train` ha ancora `<EMAIL>`.
- `pilot/`: copia di `experiment_job` con comando fisso; non tracciata da git.
- `condor/`: 25 file HTCondor (altro cluster). `launch_fetch_media_exe` fa `source slurm/condor/youtube_env.sh`.

## Riferimenti

- Pasquale (`LabelAnything/slurm`, `DistillFSS/slurm`, `VISTA/hpc`) e `Uni-Sign-LIS/script/slurm`: coppie `launch_X` + `launch_X_exe`, incluso un `launch_experiment` / `launch_experiment_exe`.
- `leofordummies.pdf`: tabella risorse (1 GPU = 8 CPU / 123000 MB; 4 GPU = 32 CPU / 492000 MB), `srun -u`, header con `--job-name/--out/--err`, «One Node, Many Scripts» (`srun --exclusive --ntasks=1 step.sh &` ... `wait`), nodi di calcolo senza internet.
- Handscribe (`run.py` + yaml per run) scartato: l'utente preferisce le coppie senza launcher.

## Decisioni dell'utente

- Coppie `launch_X` / `launch_X_exe` definite, come sempre; NESSUN launcher monolitico, NESSUN `lib/env.sh` (l'env resta dentro ogni `_exe`, duplicato di proposito).
- Per lanciare N run insieme: piccoli `.sh` (tipo `experiment.sh`) che chiamano `sbatch` sulle coppie; uno per caso (tutti training, tutti eval, ...).
- `slurm/condor/` resta dov'è (`youtube_env.sh` incluso).

## Proposta

```
slurm/
  launch_train            + launch_train_exe             1 GPU, DDP no (doc: 8 CPU, 123000)
  launch_multigpu_train   + launch_multigpu_train_exe    4 GPU, DDP su un nodo (32 CPU, 492000)
  launch_experiment       + launch_experiment_exe        un esperimento di experiments.sh, un nodo da 4 GPU:
                                                         ex experiment_job/experiment_exe (resume, STOP_AT, FAILED, gpu.csv)
  launch_evaluate         + launch_evaluate_exe
  launch_analysis         + launch_analysis_exe
  launch_fetch_media      + launch_fetch_media_exe
  experiments.sh          registro (invariato), letto da launch_experiment_exe
  batch/                  gli sh che sottomettono N run: debug.sh, wave1.sh, wave2.sh, pilot.sh, eval_all.sh ...
  tools/                  make_debug_index.py, run_report.py, setup.sh (login node)
  condor/                 invariato
  README.md               accorciato
```

- `batch/*.sh`: ognuno elenca gli `sbatch` del caso (per il training lungo, la catena `--dependency=afterok` di SEGMENTS job; per il debug i due segmenti con `afterany`). Ogni sh è autonomo, il ciclo della catena si ripete di ~10 righe.
- `launch_experiment_exe` mantiene identici `SIGNWORLD_STOP_AT`, `FAILED`, `gpu.csv`; il job multi-esperimento (un `srun` per nodo in parallelo, `debug`) resta in `launch_experiment`.
- `setup`, `status`, `report`, `cancel` non sottomettono job: `setup.sh` e i python in `tools/`; `cancel` = `scancel --name=sw-<esp>` nel README.
- Via: `launch_experiment.sh`, `experiment_job`, `experiment_exe`, `pilot/` (`pilot.yaml` → `parameters/`).

## Rischio / verifica

Mai provato su Leonardo. Verifica sul login node: `bash -n`, `sbatch --test-only` su ogni `launch_*`, e confronto dei comandi `sbatch` dei `batch/*.sh` con quelli di `launch_experiment.sh train|debug` prima della cancellazione.

## Aperto

- Internet: i login node sì, i nodi di calcolo (boost) no (utente + doc di laboratorio). `sinfo`/`scontrol` (2026-10-29): `lrd_all_serial` = nodi `login08,login13`, 4 h, senza GPU, QOS `lrd_all_serial`. Quindi `launch_fetch_media` (e il download dei modelli di `launch_analysis`) vanno su `-p lrd_all_serial` (inferenza: la rete dei login c'è lì; da provare con un job vero). `launch_analysis` oggi chiede 1 GPU su boost: o gira offline con i modelli già scaricati in `$SCRATCH`, o si divide in download (serial) + calcolo (boost).
