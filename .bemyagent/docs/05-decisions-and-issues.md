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

## Lezioni di ingegneria

- **Indici dei benchmark**: `train index` legge `split_source` dalla configurazione: va lanciato con gli stessi overlay della run (`*_trial.yaml`), altrimenti csl_daily/phoenix14t ricevono split per canale e la valutazione su `test` è vuota.
- **Canali multilingua**: `assign_splits` sceglie i canali per lingua ma li tiene fuori per intero (prima 3 canali YouTube-SL-25 erano sia in train sia in val/test); P1 esclude `val_video`, che condivide i canali per costruzione. Stato dei dataset: `docs/dataset-status.md`.

- **Percorsi sul cluster**: su ReCaS le sottocartelle di `data/` vanno collegate con symlink alle vecchie radici prima di sottomettere i job (`parameters/PATHS.md`).
- **Rifiuti in acquisizione**: i rifiuti dell'host non contano mai contro un elemento; codici di uscita 0 completato / 3 elementi rimanenti / 4 rifiutato.

## Problemi noti

I problemi di ricerca aperti dell'ultima run di addestramento (19/10/2026, passo 10.770) sono tracciati in `docs/problems.md` (13 punti: il bersaglio di posa non contiene la velocità, bersaglio anisotropo, predizione fisica debole, fuga di informazione dai box, deriva della LoRA dell'encoder, collasso delle query, R@1 basso, …).
