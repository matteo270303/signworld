# 1.8 — Dove va il tempo di un'epoca (profilo del passo per stage)

## Context

L'utente stima ~24 h/epoca su YouTube-SL-25 (1 GPU, batch 40, `runs/youtube_sl25-A-1gpu-b40`) e vuole sapere cosa costa di più nel passo, a parte distribuire. `metrics.jsonl` dà solo il passo intero (`seconds` dell'ultimo passo, senza l'attesa dei dati). `train benchmark` (`measure_efficiency`) misura il passo intero nello stage P, senza scomporlo.

### Context Saturation Check

- [x] **Target files/paths**: `signworld/experiment/collaudo/profiling.py` (nuovo), `signworld/cli/train.py` (`train profile`), `tests/experiment/collaudo/test_profiling.py`
- [x] **Expected behavior**: tempi start-end sincronizzati (`cuda.synchronize`) per sezione: to_gpu, augment, forward (pose, physical → encoder_masked/predictor, semantic → encoder_full/queries/text), backward, optimizer. Per gli stage P/F0/F, sullo stesso batch. In più: le fasi di `__getitem__`, il loader reale, i backend SDPA
- [x] **Constraints**: nessuna modifica a trainer e modello (solo wrapper sulle istanze, rimossi dopo). Il run dell'utente occupa la GPU 0 (64 GB): le misure su GPU si fanno solo a batch piccolo
- [x] **Dependencies**: nessuna

Assunzione: i rapporti fra le sezioni a batch 2-4 su GPU condivisa sono indicativi; i tempi assoluti no (da rifare a GPU libera).

## Pre-mortem

1. Timer annidati che contano due volte le chiamate ripetute (encoder ×2 maschere) → i secondi per passo sono la somma delle chiamate, e il warm-up si scarta per passo.
2. Wrapper lasciati sul modello fra uno stage e l'altro → rimossi da `__dict__` alla fine di ogni stage.
3. OOM del run dell'utente → batch 2-4, ≤ 25 GiB, pochi minuti. Il picco del run (60 GiB) c'è già stato al passo 0.

## Devil's Advocate

Alternativa: `torch.profiler`/`nsys` sul run vero (draft `lentezza-stage-f.md`, V2). Dà i kernel, non le passate del modello, e obbliga a rilanciare il training. Scartata come primo passo: il profiler per sezioni risponde a "quale passata", nsys resta per "quale kernel" dentro una passata.
