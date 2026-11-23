# 1.9 — `torch.compile` sui blocchi dell'encoder e del predittore fisico

## Context

Task 1.8: la leva 4 del draft `lentezza-stage-f.md` (le altre toccano il modello; l'utente sceglie solo questa). Sul forward dell'encoder senza gradiente `torch.compile` dà 1,15×, frenato dal context deprecato `torch.backends.cuda.sdp_kernel()` in ogni blocco Meta (4 grafi e 3 graph break per blocco).

### Context Saturation Check

- [x] **Target files/paths**: `models/worldsign/backbone.py` (`drop_cudnn_attention`, `compile_blocks`), `models/worldsign/physical.py`, `models/worldsign/video_branch.py`, `experiment/train/config.py` (`EncoderSettings.compile_blocks`), `parameters/model/worldsign.yaml`, test
- [x] **Expected behavior**: stesse loss e stessi gradienti dell'eager, passo più breve, memoria invariata
- [x] **Constraints**: il codice Meta in `checkpoints/hub/` non si tocca; gli hook di `context_levels` e il checkpointing restano fuori dal grafo; cuDNN resta escluso dall'attention di Meta
- [x] **Dependencies**: nessuna (torch 2.14 già presente)

## Approaches Considered

1. **compile del `forward` di ogni blocco** (scelto): hook e checkpoint restano in `__call__`, quindi invariati.
2. compile dell'encoder intero: gli hook aggiunti e tolti a ogni chiamata invalidano le guardie.
3. `nn.Module.compile()`: compila `_call_impl`, quindi gli hook finiscono dentro il grafo, come nella 2.

## Pre-mortem

1. Il ricalcolo del checkpointing sceglie un grafo diverso da quello del forward (`CheckpointError`) → dimensioni dinamiche fin dalla prima chiamata, cache separata per encoder e predittore.
2. Il dropout del predittore ridisegnato in modo diverso nel ricalcolo → `fallback_random`, poi verifica dei gradienti contro l'eager.
3. Ricompilazioni continue (il numero di token visibili cambia a ogni passo) → `maybe_mark_dynamic` e `recompile_limit` 64.

## Devil's Advocate

Alternativa: `set_multithreading_enabled(False)` globale, per far girare il backward nel thread principale. Scartata: cambia l'autograd ovunque, mentre la correzione locale del numero di thread tocca solo i blocchi compilati.

## CDM

- 🎯 Drift: modificare il codice Meta; cambiare loss o gradienti; aumentare la memoria
- ✅ Validation: fp32, compilato contro eager sul modello vero: loss identiche, errore relativo mediano dei gradienti ≤ 1e-3; in bf16 stessa distanza da fp32 dell'eager; profilo b40 più veloce in P/F0/F; `train run` end-to-end con P1-P17 PASS, validazione, diagnostiche rare e checkpoint
- 🔄 Pivot: se F resta instabile (CheckpointError) → compilare solo l'encoder senza grad (passata semantica)
