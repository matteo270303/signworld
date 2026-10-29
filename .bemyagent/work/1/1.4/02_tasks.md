Delivers: `slurm/batch/wave1.sh` sottomette le catene di training con le coppie `launch_experiment`.

- [x] salvare i comandi sbatch attuali (debug, train A-gate) come riferimento
- [x] `launch_experiment`+`_exe` (da experiment_job/exe); `launch_multigpu_train_exe`
- [x] `launch_fetch_media` su `lrd_all_serial`; `launch_analysis` offline
- [x] `batch/{debug,wave1,wave2,pilot}.sh`; `tools/` (setup.sh, python); pilot.yaml → parameters/
- [x] cancellare launch_experiment.sh, experiment_job/exe, pilot/ ; README; 03-code-map, 05

CDM
- 🎯 Drift: introdurre lib/launcher o toccare condor/
- ✅ Validation: diff comandi sbatch vuoto salvo i path; `bash -n` e `sbatch --test-only` ok
- 🔄 Pivot: se la catena non si esprime nei batch, o >2 ostacoli → rivalutare 01
