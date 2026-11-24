# 1.11 — Passo di training più veloce su una GPU, a metodo invariato

> Scritto durante l'esecuzione, non prima: il task è nato come analisi e le leve sono emerse dalle misure. Le misure stanno in `03_execute.log`.

## Context

L'utente ha rilanciato `runs/youtube_sl25-A-1gpu-b40-r3`: in P fa 2,61 s/passo (= il valore compilato di 1.9). Chiede di analizzare il codice, portare in test i pezzi del modello e del training e abbassare il tempo per passo. Per ora solo singola GPU; per il multi-GPU solo un'analisi. Le leve che cambiano il metodo (meno token, meno epoche, checkpointing selettivo) restano escluse, come in 1.9.

### Context Saturation Check

- [x] **Target files/paths**: `models/worldsign/lora.py`, `models/worldsign/backbone.py` (`compile_blocks`), `models/worldsign/meta_rope.py` (nuovo), `data/augmentation.py` (`apply_views`), test, `scripts/perf/`
- [x] **Expected behavior**: stessa funzione (stesse loss e gradienti entro l'arrotondamento bf16), passo più breve, memoria invariata
- [x] **Constraints**: codice Meta in `checkpoints/hub/` intatto; GPU 1-3 in modalità di calcolo "Prohibited" → misure solo sulla GPU 0, condivisa col run dell'utente (contano i rapporti, A/B alternati)
- [x] **Dependencies**: nessuna

## Approaches Considered

1. **Eliminare lavoro ridondante dal grafo** (scelto): (a) LoRA fusa `W + s·B·A` quando nessun adattatore chiede gradiente; (b) RoPE di Meta in una rotazione sola; (c) augmentation del batch intero senza sync per clip.
2. FP8 per le GEMM del ViT congelato: ~2× sulle GEMM (~40 % della passata), ma cambia la numerica delle feature → decisione dell'utente, non provato.
3. `max_autotune` di Inductor: −3,5 % sulla passata semantica, compilazione lunga e rischio di ricompilazioni nel checkpointing (1.9) → scartato.

## Pre-mortem

1. La LoRA fusa divergerebbe tra forward e ricalcolo del checkpointing → la scelta dipende solo da `requires_grad` degli adattatori e dal grad mode, uguali nei due passaggi.
2. Il RoPE fuso differisce da quello di Meta negli arrotondamenti bf16 → tabelle sin/cos con le stesse operazioni di Meta; test bit a bit in eager fp32 e bf16, con e senza maschera, con i 4 canali non ruotati della testa da 64.
3. L'augmentation vettorizzata cambia i pixel → test bit a bit su clip piccole; alla taglia reale ~1 byte su 3 milioni a 1 livello di grigio (ordine di somma della media per frame), dichiarato.

## Devil's Advocate

Alternativa radicale: cache delle feature dell'encoder congelato in P (la passata semantica è ~67 % di P). Scartata: l'augmentation e il campionamento per movimento cambiano la clip a ogni epoca, quindi la cache cambia il metodo.

## CDM

- 🎯 Drift: toccare il codice Meta; cambiare loss/gradienti oltre l'arrotondamento; aumentare la memoria; ottimizzare il multi-GPU
- ✅ Validation: test di equivalenza (LoRA, RoPE, augmentation) verdi; test lento compilato/eager verde; `train profile` a 8 clip più veloce in P/F0/F con memoria invariata; ruff + mypy puliti sui file toccati
- 🔄 Pivot: se il RoPE fuso rompe il ricalcolo compilato (CheckpointError come in 1.9) → `meta_rope.ENABLED = False` e si tengono solo LoRA e augmentation
