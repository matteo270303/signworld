Delivers: `slurm/batch/debug.sh` lanciato due volte scrive in due cartelle diverse `runs/debug/<esp>-<data>`, e i due segmenti dello stesso lancio condividono la cartella.

- [x] `experiments.sh`: `export RUN_STAMP=${RUN_STAMP-$(date +%Y%m%d-%H%M%S)}`, `run_dir` aggiunge `-$RUN_STAMP`
- [x] `launch_experiment`: `source experiments.sh`, log `out/sw-<mode>-<esp>-<data>-<job>.log`
- [x] batch: debug stampa la cartella e il comando di resume; wave1/wave2/pilot stampano la cartella
- [x] README, 03, 05, 06

CDM
- ✅ Validation: `bash -n` sugli script; simulazione di `run_dir` con stamp nuova, ereditata da un processo figlio, vuota
