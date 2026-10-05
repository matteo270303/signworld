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
