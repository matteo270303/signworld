# 1.14 — Un rank che muore ferma subito gli altri

## Context

Job 64584821 (A-gate, 18/11/2026): `KeyError` in `Collate` sul rank 2. Gli altri rank sono rimasti 60 min sulla `all_reduce` fino al timeout NCCL: il rank 2 non usciva, era fermo nel `finally: collective.shutdown()` (`destroy_process_group` aspetta gli altri rank). Lo stesso caso era rimasto scoperto in 1.13 (un'eccezione, non un FAIL, su un solo rank). Riportato da `backup/leonardo-1118` (024069c, parte "uscita rapida"); la causa del KeyError è già corretta in 1.10.

### Context Saturation Check

- [x] **Target files/paths**: `signworld/experiment/train/distributed.py`, `signworld/cli/train.py` (`run`, `benchmark`)
- [x] **Expected behavior**: con più rank, un'eccezione stampa il traceback e chiude il processo subito; torchrun vede il rank uscire e ferma gli altri in pochi secondi. Con un rank solo, l'eccezione risale come prima
- [x] **Constraints**: niente campi di config (hash); a fine run riuscita lo `shutdown` resta quello di prima
- [x] **Dependencies**: nessuna

## Approaches Considered

- Timeout NCCL più corto: il blocco dura meno, ma resta, e rischia di tagliare collettive lente legittime.
- **Scelto**: `Distributed.exit_on_error()` dentro `except BaseException`: traceback, flush di log e stream, `os._exit(1)` senza `destroy_process_group`.

## Pre-mortem

- Log persi con `os._exit`: `logging.shutdown()` e flush di stdout/stderr prima dell'uscita.
- Il caso a un rank cambia comportamento: `exit_on_error` ritorna se `not self.active`, e il chiamante rilancia l'eccezione.

## Devil's Advocate

`torch.distributed.elastic` con `--max-restarts`: riavvia, non ferma, e un errore deterministico tornerebbe uguale.
