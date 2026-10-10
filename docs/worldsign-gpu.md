# WorldSign: configurazione di lancio su nodo Booster (4 × A100 64 GB)

**Metodo.** Una sola GPU (la 0, una RTX PRO 6000 Blackwell) con il tetto dell'allocatore a 63.5 GiB per emulare una A100 64 GB. Modello **completo** (V-JEPA 2.1 ViT-L + LoRA, predittore fisico, ramo pose, predittore semantico, testa testo), pesi casuali (qui non ci sono checkpoint), batch sintetico 64×256×256, passo identico al trainer: bf16, `model.loss`, backward, AdamW fused, `expandable_segments`. Script: `tmp/full_probe.py`; risultati grezzi in `tmp/sweep/`.
Parametri: 355 M totali, 29,6 M addestrabili, pesi 1.3 GiB.

**Rimisura del 10/10/2026.** Lo sweep è stato rifatto sul codice attuale (il probe è stato adattato alle API correnti: `PoseBranch.from_settings`, `parameter_groups` dello schedule, `pose=True`). I numeri sotto **sostituiscono** quelli della prima misura, che erano su una versione precedente del modello (22 M addestrabili, 17,9 s a passo con 32 clip, stage 2 oltre il tetto): oggi lo stesso passo costa meno memoria e meno tempo. Non ho isolato quale modifica del codice lo spieghi.

**Vincolo.** Il batch globale è 128 (P16, SIGReg su tutti i 128), senza accumulo di gradiente. Con 4 GPU sono **32 clip per GPU**; con 2 GPU 64; con 1 GPU 128.

## Risultati (picco allocato per GPU, tetto 63.5 GiB)

| clip/GPU | passo completo (stage 2) | solo fisico (1a, 1) | solo semantico (2a) | due backward separati |
|---|---|---|---|---|
| 8 | 13.8 GiB | 9.6 | 7.8 | |
| 16 | 25.9 | 17.5 | 14.1 | |
| 24 | 38.0 | | | |
| 28 | 44.0 | | | |
| **32** | **50.1 (53.3 riservati): entra** | 33.4 | 26.7 | 33.5 |
| 40 | OOM (59.0 allocati al momento dell'errore) | | | |
| 48 | OOM (58.8) | 49.3 | 39.2 | 49.4 |
| 64 | OOM (54.8) | | | |

Tempo a passo (s, Blackwell): passo completo 2.2 (8 clip), 4.8 (16), 7.4 (24), 8.7 (28), **10.05 (32)**; solo fisico 7.5 (32); solo semantico 2.6 (32); due backward separati 10.1 (32).

- Senza `activation_checkpointing` il modello andava in OOM già a 4 clip nella prima misura; non l'ho rifatto, ma il checkpointing resta da considerare obbligatorio.
- Il passo completo costa circa 1.5 GiB per clip (da 8 a 32 clip: +36.3 GiB su 24 clip). Il passo completo (10.05 s) è circa la somma di fisico (7.5 s) e semantico (2.6 s).
- Tempo: circa 0.31 s per clip, circa lineare, quindi 10.05 s a passo con 32 clip. Su A100 aspettati circa 1.5–2 volte, cioè 15–20 s a passo (stima, non misurata).
- Gli OOM a 40, 48 e 64 clip sono con il tetto a 63.5 GiB: il picco allocato al momento dell'errore è sotto il tetto perché la richiesta che fallisce non ci sta nel resto.

## Conclusioni

1. **4 GPU × 32 clip entra** nello stage 2 con le impostazioni di default: 50.1 GiB allocati (53.3 riservati) contro un tetto di 63.5, cioè circa 10 GiB di margine sul riservato. Entrano anche gli stage 1a, 1 e 2a. La prima misura diceva il contrario (65.8 GiB): era un'altra versione del modello.
2. **Con 1 o 2 GPU non entra nulla:** 64 clip per GPU vanno in OOM già sotto il tetto (e 128 per 1 GPU ancora di più). Serve il nodo con 4 GPU, o 8 GPU con 16 clip ciascuna (circa 26 GiB), che però richiede un lancio multinodo: `launch_train_exe` usa `torchrun --standalone` su un solo nodo.
3. **Il backward separato per ramo non serve più per entrare**: a 32 clip il picco scende da 50.1 a 33.5 GiB (17 GiB di margine in più) senza costo in tempo (10.1 s contro 10.05 s). Resta un'opzione per avere più margine (per esempio con i diagnostici del monitor, non misurati), ma va verificato che il gradiente resti equivalente: i termini SIGReg e l'InfoNCE leggono embedding raccolti da tutte le GPU, e il grafo di ciascun ramo deve restare separabile. Ho misurato memoria e tempo, **non** l'equivalenza dei gradienti.
4. **Il margine è di circa 10 GiB, non enorme**: DDP, validazione e monitor non sono inclusi (vedi Limiti). Prima di un lancio lungo conviene un job breve su 4 GPU con controllo di `nvidia-smi` nello stage 2.

## Comando consigliato

```bash
mkdir -p out
sbatch --gres=gpu:4 --cpus-per-task=32 --mem=492000 slurm/launch_multigpu_train \
  -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml \
  --output runs/worldsign-A
```

## Limiti

- Pesi casuali: la memoria non dipende dai valori, ma i checkpoint reali non sono stati caricati.
- La Blackwell non è una A100: i tempi sono indicativi, la memoria è confrontabile (tetto emulato).
- Non incluso: DDP (bucket di gradienti: circa 0.1 GiB per 30 M parametri), validazione e diagnostiche del monitor, che girano a parte e non sono state misurate.

## Consumi con i dati che abbiamo davvero (10/10/2026)

Il costo per passo è quello misurato sopra; il numero di passi dipende dai dati. Qui si usano le clip di YouTube-SL-25 effettivamente scaricate e utilizzabili (dettaglio e criteri in `docs/worldsign-ytsl25.md`): **540.290 clip, 743 ore**, di cui **433.820 nello split di train** (601 ore). È il 25 % delle 2,16 milioni di coppie del rilascio intero.

### Tempo e GPU-ore (4 GPU × 32 clip, batch 128)

| | Valore **[Nostra stima]** |
|---|---|
| Passi per epoca (433.820 / 128) | **3.389** |
| Passi in 15 epoche (`training.epochs`) | 50.838 |
| Durata di un'epoca, Blackwell, 10,05 s a passo | **9,5 ore** |
| 15 epoche, 4 × Blackwell | **142 ore (5,9 giorni)**, 568 GPU-ore |
| 15 epoche, 4 × A100 64 GB (× 1,5–2, non misurato) | 213–284 ore (8,9–11,8 giorni), 852–1.136 GPU-ore |
| Epoche con un'`early stopping` a `patience: 3` | meno di 15: è un tetto, non la durata attesa |

Vincoli e avvertenze:
- **Il 10,05 s è il passo di stage 2**, il più costoso; gli stage 1a, 1 e 2a (circa il 7 % dei passi) costano meno (7,5 s e 2,6 s misurati a 32 clip). Le cifre sono quindi un limite superiore, di poco.
- **Il passo è misurato su dati sintetici**, senza dataloader, validazione né monitor: nella run vera sarà più lungo.
- **Il dato è un quarto del previsto.** Il rilascio intero (≈ 3,3 mila ore) darebbe circa 4 volte i passi: ≈ 24 giorni su 4 Blackwell. Il tempo cresce linearmente con le clip di train.
- **Lo split è provvisorio**: se cambia il train (altri video recuperati, esclusione dei duplicati, ricodifica dei 174 AV1) i passi cambiano in proporzione.

### Energia

Stima al limite di potenza dichiarato, quindi un **tetto**: la potenza reale in training è più bassa. Non ho misurato la potenza durante un passo.

| | Potenza per GPU | Energia, 15 epoche |
|---|---|---|
| RTX PRO 6000 Blackwell (limite impostato 600 W, 4 GPU) | 600 W | ≤ **341 kWh** |
| A100 64 GB (assunto 400 W) | 400 W | ≤ 341–454 kWh |

Esclusi: CPU, memoria, rete, raffreddamento del nodo. In idle le 4 GPU di questo nodo assorbono circa 35 W ciascuna.

### Caricamento dei dati

Misura del 9/10 su 120 clip casuali di YouTube-SL-25 (64 frame da una cue, decord con 1 thread per processo, disco locale NVMe, stato della cache di sistema non controllato), **solo decodifica**, senza ritaglio sul segnante, ridimensionamento a 256 px né posa:

| Processi | Clip al secondo | Mediana per clip | 95° percentile |
|---|---|---|---|
| 1 | 3,2 | 0,29 s | 0,56 s |
| 8 | 21,8 | 0,29 s | 0,59 s |
| 32 | 50,5 | 0,40 s | 0,77 s |

Il training consuma 128 clip ogni 10,05 s, cioè **12,7 clip/s** per l'intero nodo: con gli 8 worker previsti (`data.workers: 8`, 21,8 clip/s misurati) la sola decodifica ha un margine di circa 1,7×, stretto ma non è il collo di bottiglia. Se i worker sono per GPU il margine è maggiore (non l'ho verificato). Il margine si riduce con ritaglio, ridimensionamento e lettura della posa, che non ho misurato; il controllo a 200 passi reali di §4.13 resta necessario.

- **I video AV1 (174, l'1,3 %) non si aprono con decord**: ogni clip che ci cade dà un errore. Vanno ricodificati o esclusi prima del training.
- **Non conviene pre-estrarre le clip**: 64 frame a 256² in uint8 sono 12,6 MB a clip, **6,8 TB** per 540.290 clip grezze, contro i 525 GB dei video.

### Spazio su disco

| | Valore |
|---|---|
| Video scaricati (13.377) | 525 GB, 0,49 GB per ora di video |
| Metadati e sottotitoli | meno di 1 GB |
| Rilascio intero, stima (3.299 ore) | ≈ 1,6 TB |
| Spazio libero su `/home` | 12 TB |

### Cosa manca per chiudere la stima

1. Il passo misurato su un **nodo A100 vero**: il fattore 1,5–2 è un'ipotesi.
2. La **potenza reale** a regime (`nvidia-smi --query-gpu=power.draw`) durante la dry run PC7.
3. Il costo del dataloader **completo** (ritaglio, posa), non della sola decodifica.
