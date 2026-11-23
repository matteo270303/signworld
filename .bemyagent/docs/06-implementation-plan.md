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
| 1.8 | Profilo del passo per stage e sezione (`signworld train profile`): dove va il tempo di un'epoca (`work/1/1.8/`, draft `lentezza-stage-f.md`) | done — da rimisurare a GPU libera, batch 40 |

## Backlog (non pianificato)

- Lavoro non committato su `refactor/structure` (frame sources, configurazioni di prova e script di esecuzione per CSL-Daily / Phoenix14T) — stato ?
