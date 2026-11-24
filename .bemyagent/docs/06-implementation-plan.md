# Piano di implementazione

> Indice di alto livello. Le tracce di esecuzione stanno in `work/X/X.Y/`.

## Milestone 1 — Correzioni dopo la prima run di addestramento

**Obiettivo**: affrontare i problemi emersi nella prima run completa di addestramento (`docs/problems.md`) | **Stato**: in-progress

| Task | Descrizione | Stato |
|---|---|---|
| 1.1 | Suddividere `docs/problems.md` in task | todo |
| 1.2 | Check dei dataset (youtube_sl25, csl_daily, phoenix14t) e avvio dei 3 `run_train_*.sh` — report `docs/dataset-status.md` | done |
| 1.3 | Lancio su Leonardo: debug del carico, catene con resume a scadenza (`slurm/batch/*.sh`, `slurm/README.md`) | in-progress — da provare sul cluster |
| 1.4 | Riordino di `slurm/`: coppie `launch_X`/`_exe` + `batch/*.sh` (`work/1/1.4/`) | done — da provare sul cluster |
| 1.5 | Statistiche di avvio più veloci: pose in parallelo, embedding a blocchi, caption e indice letti una volta (`work/1/1.5/`) | done — sul nodo GPU ~1 min (debug 63997298) |
| 1.6 | Debug 63997298: diagnostiche rare in bf16, log P una volta con descrizione e secondi, tqdm su P13/rare/statistiche, warning (`work/1/1.6/`) | done — da misurare al prossimo debug |
| 1.7 | Cartelle dei run con la data del lancio (`RUN_STAMP` in `slurm/experiments.sh`), niente più riuso fra lanci con lo stesso nome (`work/1/1.7/`) | done — da provare sul cluster |
| 1.8 | Profilo del passo per stage e sezione (`signworld train profile`): dove va il tempo di un'epoca (`work/1/1.8/`, draft `lentezza-stage-f.md`) | done — rimisurato a batch 40 in 1.9 |
| 1.9 | `torch.compile` sui blocchi dell'encoder e del predittore fisico (`encoder.compile_blocks`), -22/-30/-33 % in P/F0/F (`work/1/1.9/`) | done — da provare sul cluster (H100, DDP) |
| 1.10 | Lingua persa nel centring (`caption_statistics` dedup per riga): KeyError `en-xgJ5vofi3O8` nella run locale youtube_sl25-A-1gpu-b40 (`work/1/1.10/`) | done — da provare alla prossima run (cancellare `statistics.pt` vecchi) |
| 1.11 | Passo più veloce su 1 GPU a metodo invariato: LoRA fusa senza gradiente, RoPE di Meta in una rotazione, augmentation del batch intero; P/F0/F −17/−18/−13 % a 8 clip (`work/1/1.11/`) | done — da rimisurare a b40 a GPU libera |
| 1.12 | Log degli esperimenti in `runs/<run>/logs/` invece di `out/`: un file per job con programma e messaggi di srun; riportato da `backup/leonardo-1118` (`work/1/1.12/`) | done — da provare sul cluster |
| 1.13 | P13 con verdetto unico fra le GPU (loss mediate), WARN se la loss scende senza dimezzarsi, `collective.any` prima di `enforce`; riportato da `backup/leonardo-1118` (`work/1/1.13/`) | done — da provare sul cluster (4 GPU) |
| 1.14 | Un rank che muore ferma subito gli altri: `Distributed.exit_on_error` in `train run`/`benchmark` (job 64584821, 60 min di GPU ferme); riportato da `backup/leonardo-1118` (`work/1/1.14/`) | done — da provare sul cluster |

## Backlog (non pianificato)

- Lavoro non committato su `refactor/structure` (frame sources, configurazioni di prova e script di esecuzione per CSL-Daily / Phoenix14T) — stato ?
- Centratura per lingua scritta (colonna `written_language` nell'indice, `en-US`/`en-xgJ5vofi3O8` → `en`): fatta su `backup/leonardo-1118` (024069c), non riportata. Cambia unità della centratura, indice, `statistics.pt` e soglie GloFND — da decidere
