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
- Timeout NCCL in `prepare` (`signworld/experiment/train/run.py:96`): il rank 0 calcola da solo le statistiche (`fit_statistics`, ~10,5 min su 945.688 clip) mentre gli altri 3 aspettano il barrier; il timeout di default (600 s) scade, il watchdog scrive il dump e dopo 480 s di heartbeat abortisce tutti (SIGABRT, `FAILED`). Job `sw-train-A-gate` 63260872 (03/11/2026). Corretto con `SIGNWORLD_COLLECTIVE_TIMEOUT_MIN` (default 60 in `slurm/launch_experiment_exe`, letta in `distributed.py`). `runs/A-gate/statistics.pt` è già completo: togliere `FAILED` e ri-sottomettere. Poi (09/11/2026, task 1.5): pose lette con 16 thread (~500 s → ~60 s), caption calcolate da ogni rank prima della barrier e riusate da P11, indice letto una volta; la barrier ora aspetta solo le pose (~1 min). Statistiche identiche bit a bit a quelle di A-gate (`out/check_stats.py`).
- Venv rotto il 09/11/2026 alle 16:22: `torch 2.14.0+cu126` è installato a mano dall'indice cu126, mentre `uv.lock` ha solo i pacchetti `nvidia-*-cu13`. Un `uv sync`/`uv run` ha rimosso i `*-cu12` come estranei, e con loro file condivisi in `nvidia/cudnn` e `nvidia/nccl` (`libcudnn.so.9` mancante, `torch` non si importa). Ripristinato con `uv pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126` e reinstallando alle stesse versioni i cu13 e `opencv-python-headless`. Indice cu126 fissato in `pyproject.toml` (`[tool.uv.sources]`, lock su torch 2.14.0+cu126). Il primo `uv sync` dopo il fix toglie i `nvidia-*-cu13`, che condividono `nvidia/cudnn` e `nvidia/nccl` con i cu12 e ne cancellerebbero i file: ricreare il venv (`rm -rf .venv && uv sync`) a job fermi, non fare un sync parziale.
- Warning di deprecazione del codice Meta in `checkpoints/hub/` (`timm.models.layers`, `torch.backends.cuda.sdp_kernel()`): filtrati per messaggio in `video_encoders.py` (09/11/2026, task 1.6). DDP `broadcast_buffers` (torch 2.14): silenziato in `trainer._enter`, non migrato a `forward_sync_buffers` perché copierebbe le statistiche BN del rank 0 a ogni cambio di stadio.
- Debug 63997298 (09/11/2026) fermo al passo 4: su 24 min utili, 7,5 min P13 (30 passi del modello intero con LoRA video, ~15 s/passo) e 14,3 min di diagnostiche rare al passo 0, girate in fp32 (`Monitor.rare` e `plausibility_tests` senza autocast, 48 forward ViT-L per GPU). Corretto con autocast bf16 come la validazione (task 1.6). Il secondo segmento 63997299 è uscito con `ProvenanceError`: `pyproject.toml`/`uv.lock` modificati alle 17:21 a run in corso; riprendere con `--allow-code-change` o una cartella nuova.
- Cartella del run riusata fra lanci con lo stesso nome (`runs/debug/A-gate`): un nuovo debug riprendeva il vecchio (`ProvenanceError` se il codice è cambiato, `FAILED` che blocca). Dal 09/11/2026 (task 1.7) `run_dir` aggiunge `RUN_STAMP` (`%Y%m%d-%H%M%S`, fissata una volta per sbatch di batch ed ereditata dai segmenti della catena); resume esplicito con `RUN_STAMP=<data>`, `RUN_STAMP=` per le cartelle vecchie. Scartato: data calcolata nel job (i segmenti di una catena finirebbero in cartelle diverse).
- Tempo per epoca su YouTube-SL-25, 1 GPU, batch 40 (23/11/2026, task 1.8, `signworld train profile`). Lo stage P costa 3,34 s/passo × 23.642 passi = 21,9 h; F0 e F costano 2,1-2,5× e 2,7-3,7× (stima 60-80 h per epoca di F). Il collo di bottiglia è la GPU, non i dati: il loader dà ~1.130 clip/s contro le 12 richieste. Flash è già il backend più veloce (cuDNN non aiuta). `torch.compile` sull'encoder: 1,15×. Togliere l'activation checkpointing: -19 % in F ma ~10 GiB/clip. Dettagli e leve: `docs/drafts/lentezza-stage-f.md`.
- `torch.compile` sui blocchi (23/11/2026, task 1.9): `encoder.compile_blocks: true` in `worldsign.yaml`, default false nel codice e nei test tiny. A batch 40 su 1 GPU: P 3,32 → 2,58 s/passo, F0 8,90 → 6,26, F 12,49 → 8,37 (-22/-30/-33 %), memoria invariata. In fp32 i gradienti coincidono con l'eager (errore relativo mediano 1e-4). Scelto il compile del `forward` di ogni blocco, non dell'encoder intero né `nn.Module.compile()`, perché gli hook di `context_levels` e il checkpointing devono restare fuori dal grafo. Tre trappole, tutte necessarie (senza ciascuna: `CheckpointError` nel backward di F):
  1. dimensioni dinamiche fin dalla prima chiamata, perché i token visibili cambiano a ogni passo;
  2. una copia del codice per encoder e predittore, perché sono la stessa classe `Block` e la cache di dynamo è per oggetto codice;
  3. il ricalcolo gira nel thread di autograd, con un `num_threads` diverso: si allinea prima di ogni chiamata.

  `drop_cudnn_attention` ora usa `torch.nn.attention.sdpa_kernel` (stessi backend, cuDNN escluso): niente graph break. Per tornare all'eager basta `encoder.compile_blocks: false`. Ogni forma nuova costa ~5-30 s di compilazione nei primi passi.
- Passo più veloce a metodo invariato (24/11/2026, task 1.11). Tre lavori ridondanti tolti dal grafo:
  1. LoRA fusa (`W + s·B·A`, un prodotto invece di tre) quando nessun adattatore chiede gradiente: la LoRA senza gradiente costa quanto l'encoder senza LoRA;
  2. RoPE di Meta in una rotazione della testa intera (stesse sin/cos, bit a bit in eager), dentro `compile_blocks`;
  3. augmentation del batch intero senza `tolist()` per clip (3,5×).

  A 8 clip su GPU condivisa: P −17 %, F0 −18 %, F −13 %, memoria uguale. Scartati: `max_autotune` (−3,5 % sulla passata, autotune per forma); cache delle feature in P (cambia il metodo). Aperte e a scelta dell'utente: FP8 sulle GEMM dell'encoder congelato. Le GPU 1-3 della macchina locale sono in modalità di calcolo "Prohibited": si misura solo sulla GPU 0. Dettagli: `docs/drafts/lentezza-stage-f.md`.
- Fuori dal passo, run `-r3` (24/11/2026): la validazione ogni 4.000 passi dura ~540 s (0,27 s/clip contro 0,065 s/clip del training con backward: rifà il ramo fisico e una seconda passata semantica sul rumore), ~5 % del tempo in P. `latest` si salva solo con la validazione: il resume è ripartito da 24.020 dopo che il run era arrivato a 27.940 (~2,8 h di GPU perse). Da decidere: `latest` più frequente, validazione più leggera in P.
- In bf16 i gradienti per tensore differiscono molto da fp32 (errore relativo mediano ~1,4, sia eager sia compilato, passo F su 2 clip, 23/11/2026). È preesistente e da capire (cancellazione nei LayerNorm dell'encoder?). Non dipende da `torch.compile`.
- `tests/experiment/train/test_preflight.py::test_the_checksum_record_is_published_whole` fallisce anche sul codice di HEAD 2a5e9a5 (ordine della lista dei file; verificato il 23/11/2026, preesistente).
- `test_the_single_batch_overfit_learns_and_restores_the_model` (lento) fallisce: P13 FAIL sul setup tiny, anche su HEAD 23a6edb (verificato il 09/11/2026, preesistente).
- Run locale `runs/youtube_sl25-A-1gpu-b40` (23/11/2026) morta a ~passo 1040 con `KeyError: 'en-xgJ5vofi3O8'` in `Collate`: `caption_statistics` teneva una lingua per `caption_row` (vince l'ultima), e una lingua con tutte le caption condivise con altre (stesso testo) restava senza media (142 lingue fittate su 143). Preesistente. Corretto (task 1.10): dedup per coppia `(caption_row, lingua)`; `negatives.py` usa `distinct_captions` (vecchia semantica, soglie GloFND identiche bit a bit); `prepare` si ferma all'avvio se una lingua del training manca dal centring. Gli `statistics.pt` fittati prima vanno cancellati. Aperto: ~100 tag `xx-<id video>` (nomi di tracce YouTube) contati come lingue a sé.
- `test_the_checksum_record_is_published_whole` fallisce: confronta l'ordine di `tmp_path.iterdir()`, che non è garantito (verificato il 23/11/2026 su 2a5e9a5, preesistente).

I problemi di ricerca aperti dell'ultima run di addestramento (19/10/2026, passo 10.770) sono tracciati in `docs/problems.md` (13 punti: il bersaglio di posa non contiene la velocità, bersaglio anisotropo, predizione fisica debole, fuga di informazione dai box, deriva della LoRA dell'encoder, collasso delle query, R@1 basso, …).
