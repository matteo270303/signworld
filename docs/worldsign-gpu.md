# WorldSign: configurazione di lancio su nodo Booster (4 × A100 64 GB)

**Metodo.** Una sola GPU (la 0, una RTX PRO 6000 Blackwell) con il tetto dell'allocatore a 63.5 GiB per emulare una A100 64 GB. Modello **completo** (V-JEPA 2.1 ViT-L + LoRA, predittore fisico, ramo pose, predittore semantico, testa testo), pesi casuali (qui non ci sono checkpoint), batch sintetico 64×256×256, passo identico al trainer: bf16, `model.loss`, backward, AdamW fused, `expandable_segments`. Script: `tmp/full_probe.py`; risultati grezzi in `tmp/sweep/`.
Parametri: 354 M totali, 22 M addestrabili, pesi 1.3 GiB.

**Vincolo.** Il batch globale è 128 (P16, SIGReg su tutti i 128), senza accumulo di gradiente. Con 4 GPU sono **32 clip per GPU**; con 2 GPU 64; con 1 GPU 128.

## Risultati (picco allocato per GPU, tetto 63.5 GiB)

| clip/GPU | passo completo (stage 2) | solo fisico (1a, 1) | solo semantico (2a) | due backward separati |
|---|---|---|---|---|
| 8 | 17.6 GiB | 10.4 | 10.5 | |
| 16 | 33.7 | 19.4 | 19.4 | |
| 24 | 49.8 | | | |
| 28 | 57.8 | | | |
| **32** | **65.8 (66.9 riservati): non entra** | 37.2 | 37.3 | **37.4** |
| 48 | OOM | 55.0 | 55.2 | 55.3 |
| 64 | OOM | | | |

- Senza `activation_checkpointing` (batch 4, 8, 16) il modello va in OOM già a 4 clip. Il checkpointing resta obbligatorio.
- Il passo completo costa circa 2.0 GiB per clip, quasi la somma dei due rami (1.1 e 1.15). Il fisico e il semantico tengono entrambi il loro grafo fino al backward.
- Tempo (Blackwell): circa 0.55 s per clip, lineare, quindi 17.9 s a passo con 32 clip. Su A100 aspettati circa 1.5–2 volte, cioè 27–36 s a passo (stima, non misurata).

## Conclusioni

1. **Con le impostazioni di default, 4 GPU × 32 clip non entra** nello stage 2 (65.8 GiB allocati contro 63.5 di tetto, contesto CUDA escluso). Mancano circa 3.5 GiB. Gli stage 1a, 1 e 2a (circa il 7% dei passi) entrano.
2. **Con 1 o 2 GPU non entra nulla:** 64 clip per GPU superano 100 GiB in stage 2.
3. **Il lancio sicuro oggi** richiede 8 GPU (2 nodi, 16 clip per GPU, circa 34 GiB), ma `launch_train_exe` usa `torchrun --standalone` su un solo nodo. Serve un lancio multinodo (`--nnodes`, rendezvous c10d).
4. **La soluzione migliore su un nodo con 4 GPU** è una modifica al codice: in stage 2 fare un backward per ramo (fisico, poi semantico) invece di uno solo sulla somma. Misurato: picco **37.4 GiB** a 32 clip (circa 26 GiB di margine) con lo stesso tempo a passo (17.86 s contro 17.87 s). Anche 48 clip per GPU entrerebbe, ma P16 fissa 32. Serve verificare che il gradiente resti equivalente: i termini SIGReg e l'InfoNCE leggono embedding raccolti da tutte le GPU, e il grafo di ciascun ramo deve restare separabile.

## Comando consigliato (dopo la modifica del punto 4)

```bash
mkdir -p out
sbatch --gres=gpu:4 --cpus-per-task=32 --mem=492000 slurm/launch_multigpu_train \
  -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml \
  --output runs/worldsign-A
```

Senza la modifica, lo stesso comando parte ma va in OOM all'ingresso nello stage 2 (circa all'1–2% della run). Prima di un lancio lungo, prova un job breve su 4 GPU e controlla con `nvidia-smi` la memoria nello stage 2.

## Limiti

- Pesi casuali: la memoria non dipende dai valori, ma i checkpoint reali non sono stati caricati.
- La Blackwell non è una A100: i tempi sono indicativi, la memoria è confrontabile (tetto emulato).
- Non incluso: DDP (bucket di gradienti: circa 0.1 GiB per 22 M parametri), validazione e diagnostiche del monitor, che girano a parte e non sono state misurate.

## Consumi con i dati che abbiamo davvero (10/10/2026)

Il costo per passo è quello misurato sopra; il numero di passi dipende dai dati. Qui si usano le clip di YouTube-SL-25 effettivamente scaricate e utilizzabili (dettaglio e criteri in `docs/worldsign-ytsl25.md`): **540.290 clip, 743 ore**, di cui **433.820 nello split di train** (601 ore). È il 25 % delle 2,16 milioni di coppie del rilascio intero.

### Tempo e GPU-ore (4 GPU × 32 clip, batch 128)

| | Valore **[Nostra stima]** |
|---|---|
| Passi per epoca (433.820 / 128) | **3.389** |
| Passi in 15 epoche (`training.epochs`) | 50.838 |
| Durata di un'epoca, Blackwell, 17,9 s a passo | **16,9 ore** |
| 15 epoche, 4 × Blackwell | **253 ore (10,5 giorni)**, 1.011 GPU-ore |
| 15 epoche, 4 × A100 64 GB (× 1,5–2, non misurato) | 379–506 ore (15,8–21,1 giorni), 1.517–2.022 GPU-ore |
| Epoche con un'`early stopping` a `patience: 3` | meno di 15: è un tetto, non la durata attesa |

Vincoli e avvertenze:
- **Il 17,9 s è il passo di stage 2**, il più costoso; gli stage 1a, 1 e 2a (circa il 7 % dei passi) costano meno. Le cifre sono quindi un limite superiore, di poco.
- **Vale solo dopo la modifica del punto 4 sopra** (un backward per ramo): senza, 4 GPU × 32 clip non entra in memoria nello stage 2 e il tempo di una run non è definito.
- **Il dato è un quarto del previsto.** Il rilascio intero (≈ 3,3 mila ore) darebbe circa 4 volte i passi: ≈ 40 giorni su 4 Blackwell. Il tempo cresce linearmente con le clip di train.
- **Lo split è provvisorio**: se cambia il train (altri video recuperati, esclusione dei duplicati, ricodifica dei 174 AV1) i passi cambiano in proporzione.

### Energia

Stima al limite di potenza dichiarato, quindi un **tetto**: la potenza reale in training è più bassa. Non ho misurato la potenza durante un passo.

| | Potenza per GPU | Energia, 15 epoche |
|---|---|---|
| RTX PRO 6000 Blackwell (limite impostato 600 W, 4 GPU) | 600 W | ≤ **607 kWh** |
| A100 64 GB (assunto 400 W) | 400 W | ≤ 607–809 kWh |

Esclusi: CPU, memoria, rete, raffreddamento del nodo. In idle le 4 GPU di questo nodo assorbono circa 35 W ciascuna.

### Caricamento dei dati

Misura del 9/10 su 120 clip casuali di YouTube-SL-25 (64 frame da una cue, decord con 1 thread per processo, disco locale NVMe, stato della cache di sistema non controllato), **solo decodifica**, senza ritaglio sul segnante, ridimensionamento a 256 px né posa:

| Processi | Clip al secondo | Mediana per clip | 95° percentile |
|---|---|---|---|
| 1 | 3,2 | 0,29 s | 0,56 s |
| 8 | 21,8 | 0,29 s | 0,59 s |
| 32 | 50,5 | 0,40 s | 0,77 s |

Il training consuma 128 clip ogni 17,9 s, cioè **7,2 clip/s** per l'intero nodo: con gli 8 worker previsti (`data.workers: 8`) la sola decodifica ha un margine di circa 3×, e non è il collo di bottiglia. Il margine si riduce con ritaglio, ridimensionamento e lettura della posa, che non ho misurato; il controllo a 200 passi reali di §4.13 resta necessario.

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
