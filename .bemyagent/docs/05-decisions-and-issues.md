# Decisioni e problemi

## Indice delle decisioni

> Le decisioni in linea stanno qui. Le scelte complesse, a livello di sistema, hanno un ADR dedicato in `decisions/`, collegato qui.

| # | Decisione | Sintesi | File |
|---|---|---|---|
| 1 | Addestramento per livello | ogni loss aggiorna solo i pesi del suo livello | `docs/worldsign-gerarchia.md` |
| 2 | Percorsi relativi al repository | tutti i percorsi in `parameters/` sono relativi alla radice | `parameters/PATHS.md` |

### Decisioni in linea

#### Overlay di configurazione

- **Problema**: molte run condividono una configurazione di base.
- **Decisione**: `parameters/model/worldsign.yaml` + braccio di ablation + overlay dei dati, fusi in ordine.
- **Compromesso**: la configurazione effettiva è visibile solo dopo la fusione.

#### Batch di prova su una sola GPU

- **Problema**: su questa macchina solo la GPU 0 è utilizzabile (1-3 in compute mode Prohibited); 128 clip su una GPU non entrano (~2 GiB/clip), e P16 bloccava qualunque batch diverso da 128.
- **Decisione**: `training.effective_batch` (default 128, P16 lo confronta) + overlay `parameters/model/local_single_gpu.yaml` (batch 32 = carico per GPU di 4×32), aggiunto dagli script quando `GPUS` ha una sola GPU.
- **Compromesso**: SIGReg/InfoNCE vedono 32 clip: run di prova della pipeline, non dell'articolo.

#### Resume a scadenza su Leonardo

- **Problema**: `latest` si salvava solo ogni `validation_every` (4.000 passi, ~20 h in stage F su A100) e a fine epoca: il kill al walltime (24 h `normal`, 4 giorni `boost_qos_lprod`) perdeva fino a ~20 h per segmento.
- **Decisione**: variabile d'ambiente `SIGNWORLD_STOP_AT` (non nella config: ogni job ha la sua, e la config deve restare uguale fra i resume), letta ai passi di log con `collective.any`; il trainer salva `latest` e ritorna. I segmenti sono job in catena `afterok` + `--kill-on-invalid-dep`; un'uscita ≠ 0 scrive `<run>/FAILED` e ferma la catena.
- **Compromesso**: margine di 45 min (validazione + diagnostiche rare devono finire prima del walltime); se il margine non basta, Slurm uccide il job (TIMEOUT) e la catena si ferma: si rilancia a mano.

#### Debug del carico a indice ridotto

- **Problema**: `train benchmark` misura solo lo stage P (`TrainingState` nuovo) e `train overfit` gira su un GPU senza DDP: nessuno dei due misura lo stage F del run vero.
- **Decisione**: `train run` vero (DDP, 4 GPU, stesso modello e batch per GPU) su `index.debug.parquet` (16 passi per epoca, una epoca per stage), due segmenti da 30 min su `boost_qos_dbg` (il secondo prova anche il resume), un nodo per esperimento.
- **Compromesso**: i secondi per passo escludono validazioni e diagnostiche; la stima di durata è un minimo.

## Lezioni di ingegneria

- **Indici dei benchmark**: `train index` legge `split_source` dalla configurazione: va lanciato con gli stessi overlay della run (`*_trial.yaml`), altrimenti csl_daily/phoenix14t ricevono split per canale e la valutazione su `test` è vuota.
- **Canali multilingua**: `assign_splits` sceglie i canali per lingua ma li tiene fuori per intero (prima 3 canali YouTube-SL-25 erano sia in train sia in val/test); P1 esclude `val_video`, che condivide i canali per costruzione. Stato dei dataset: `docs/dataset-status.md`.

- **Percorsi sul cluster**: su ReCaS le sottocartelle di `data/` vanno collegate con symlink alle vecchie radici prima di sottomettere i job (`parameters/PATHS.md`).
- **Rifiuti in acquisizione**: i rifiuti dell'host non contano mai contro un elemento; codici di uscita 0 completato / 3 elementi rimanenti / 4 rifiutato.
- **Struttura di `slurm/` (2026-10-29)**: coppie `launch_X` / `launch_X_exe`, niente launcher monolitico né `lib/env.sh` (l'ambiente è duplicato di proposito in ogni `_exe`); N run insieme = `.sh` in `slurm/batch/`. Scartati: launcher Python alla Handscribe (la griglia di parametri non serve), launcher bash unico. I nodi boost non hanno internet: `launch_fetch_media` su `lrd_all_serial` (nodi `login08/13`, deduzione dai nomi, non provato), `launch_analysis` offline con i modelli in `$SCRATCH`. Il pilota è l'esperimento `pilot` (`runs/pilot-A`, `parameters/model/pilot.yaml`).

## Problemi noti

- `tests/experiment/train/test_trainer.py::test_a_short_run_goes_through_every_stage_and_cools_down` (lento) fallisce: si aspetta gli stage `1a, 1, 2a, 2`, il codice ora li chiama `P, F0, F` (verificato il 27/10/2026, preesistente).
- Race in `p5_checksums` (`signworld/experiment/train/preflight.py:162`): al primo avvio di una `runs/` nuova un rank legge `checksums.json` mentre il rank 0 lo sta scrivendo, trova un file vuoto e muore; gli altri restano in attesa del collective di P13 fino al timeout NCCL. Pilota 62470758 (28/10/2026). Corretto con scrittura atomica (tmp + `os.replace`), test `test_the_checksum_record_is_published_whole`; pilota ri-sottomesso come 62545635. Dettagli: `docs/drafts/pilot-A-crash.md`.

I problemi di ricerca aperti dell'ultima run di addestramento (19/10/2026, passo 10.770) sono tracciati in `docs/problems.md` (13 punti: il bersaglio di posa non contiene la velocità, bersaglio anisotropo, predizione fisica debole, fuga di informazione dai box, deriva della LoRA dell'encoder, collasso delle query, R@1 basso, …).
