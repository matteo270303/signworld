# Pilota A: crash in preflight (job 62470758)

**Esito**: il job è fallito in P5 su un rank e ha lasciato gli altri tre bloccati in P13 fino al timeout NCCL (600 s). Il training non è partito. Nessun checkpoint, nessun `preflight.json`, nessun `metrics.jsonl`.

## Cosa è successo

- 01:36–01:37 avvio su `lrdn3418`, 4 GPU A100, un nodo. Il log riporta `2048 training clips, 249 validation clips`.
- P1–P4 `pass` su tutti i rank. P5 (checksum): il rank 0 scrive `runs/pilot-A/checksums.json`, gli altri tre rank lo leggono, il rank 1 no.
- Rank 1: `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` in `p5_checksums` (`signworld/experiment/train/preflight.py:169`). Il file era vuoto nel momento della lettura.
- Rank 0, 2, 3: P5–P12 `pass`, poi P13 (`p13_overfit`) entra in `pose_sigreg` → `all_sum` → `all_reduce`. Il rank 1 non arriva mai a quel collective.
- 01:47:37 watchdog NCCL: `ALLREDUCE ... ran for 600 s before timing out`. 01:48:37 abort di torchrun, `exitcode -6`.

## Causa

Race in `p5_checksums`: il file di record viene aperto e creato da `write_text` prima di essere riempito. Un rank che esegue `record.is_file()` in quell'intervallo vede un file vuoto e fallisce al `json.loads`. Gli altri rank non vedono l'errore: entrano nel collective di P13 e aspettano il rank morto.

La firma `char 0` indica un file vuoto, quindi il race è l'ipotesi più solida. Non è stato riprodotto in locale.

## Perché il debug non l'ha mostrato

Il debug `A-gate` (27/10, job 62447431) ha scritto `checksums.json` alle 18:06:56 con lo stesso percorso e ha superato P5 sui 4 rank. La finestra è stretta: il race dipende dal tempo di scrittura di 331 byte e da quanto presto il rank 1 arriva alla lettura. Il pilota è il primo run in una `runs/` nuova, quindi è il primo a scrivere il record in parallelo. Il debug ha avuto fortuna.

## Cosa non è la causa

- Non c'è OOM né errore CUDA nel log.
- Il codice era sporco (`code-0.patch` da 188 KB, `dirty: true`). Non ha causato il crash, ma un eventuale resume con codice diverso verrà rifiutato (`provenance.json`).
- `checksums.json` oggi è valido e completo: il file è stato scritto correttamente, il problema è solo nel momento della lettura.

## Costo

Circa 12 min di 1 nodo (4 GPU) su 1 h 45 richiesti. Il resto è attesa in coda.

## Fix proposto (non applicato)

`p5_checksums`: scrivere su file temporaneo e fare `os.replace`, così ogni lettore vede il file intero oppure non lo vede. I rank che non trovano il record lo scrivono da soli con contenuto identico, quindi nessun conflitto. Alternativa più pulita: scrittura solo sul rank 0 seguita da `barrier`.

Test: `tests/experiment/train/` con due processi concorrenti che chiamano `p5_checksums` sullo stesso `record` inesistente.

## Da fare prima del prossimo lancio

1. Applicare il fix a `p5_checksums` e committare (`refactor/structure`).
2. Rilanciare il pilota con `slurm/batch/pilot.sh`. Il record è già valido, quindi il race non si ripresenta, ma il fix va comunque applicato.
3. Il report del pilota non esiste ancora: vedi sotto.

## Report del pilota

Il pilota non produce un report finché non c'è un checkpoint valutato. Sequenza:

```bash
signworld train run ...           # il pilota (già fatto, da rifare)
signworld train evaluate --run runs/pilot-A --index <index> --output <json> [--checkpoint best]
signworld train report --run runs/pilot-A --evaluation <json> [--ridge <json>] [--split test]
```

`slurm/tools/run_report.py debug` leggono solo `runs/debug/`. Per lo stato dei run: `python slurm/tools/run_report.py status` (legge `runs/**/metrics.jsonl`, quindi vede il pilota solo dopo il primo passo).

Eseguire `train report` adesso produrrebbe un documento quasi vuoto: manca tutto ciò che viene dal training.

Fonti: `out/sw-pilot-62470758.log`, `runs/pilot-A/{checksums,provenance}.json`, `signworld/experiment/train/preflight.py:162-175,481`, `signworld/experiment/evaluation/report.py:311`, `signworld/cli/train.py:97,159`.

---

## Secondo run (job 62545635, 28/10 ~16:30–17:09): stop F1 a step 32

**Esito**: il fix del race in P5 ha funzionato (preflight passato, training partito). Il run si è fermato di proposito al gate F1 (`gate_stops: true`) a fine fase P (step 32/240, epoca 2). Non è un crash: `RunStoppedError` su tutti i rank, poi `exitcode 1` di torchrun. Checkpoint in `runs/pilot-A/checkpoints/{stop_F1,stopped,latest}.pt`.

**Criteri F1 falliti** (da `runs/pilot-A/metrics.jsonl`, riga `stop`):
- rank di s: 22.9 contro 69.6 a step 0 (soglia > 0.5×: serviva > 34.8)
- IsoScore di s: 0.011 (soglia ≥ 0.8)
- R² posizione mano sx/dx da s: 0.879 / 0.890 (soglia ≥ 0.9)
- R@1 held-out: 0.0081 contro chance 0.0040 (soglia 5×; invariato dallo step 0)

Passati: L_anchor e SIGReg_posa in calo, gamma_sem 0.922, noise test, queries non collassate.

**Non sono errori**: `loss nan` a inizio epoca è solo la barra di progresso (nessun dato ancora); i loss reali sono finiti (5.48 → 3.51).

**Lettura (inferenza, non verificata)**: lo spazio s si sta concentrando su poche direzioni (rank 70 → 23, IsoScore ~0) mentre il LR sale ancora in warmup (1e-4 → 2e-4); R² sulle mani quasi a soglia. Il retrieval non si muove in 32 step. Il gate F1 è tarato per l'esito di una fase P completa, non per un pilota ridotto a 2 epoche (16 step/epoca): o il pilota va allungato, o F1 va giudicato a più step.
