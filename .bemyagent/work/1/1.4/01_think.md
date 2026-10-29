## Context

- Riordinare `slurm/`: coppie `launch_X` / `launch_X_exe`, niente launcher né `lib/env.sh`, `batch/*.sh` per N run. Draft: `docs/drafts/slurm-cleanup.md`.

### Context Saturation Check

- [x] Target files/paths — `slurm/*` (letti tutti)
- [x] Expected behavior — comandi `sbatch` di debug/train invariati
- [x] Constraints — condor/ intatto; env duplicato negli `_exe`
- [x] Dependencies — nessuna nuova; `sbatch --test-only` sul login node

## Approaches Considered

- Launcher bash rinominato / Python alla Handscribe / coppie + batch (scelto dall'utente).
  > Rivalutare solo se la catena `afterok` non si lascia esprimere in `batch/*.sh`.

## Selected Approach & Risks

- Coppie + `batch/*.sh`; costo medio. Rischi: regressione dei comandi sbatch (mitigata dal diff prima/poi); analysis/fetch senza rete su boost (fetch su `lrd_all_serial`, analysis offline dopo download modelli).
- Pre-mortem: (1) cambio di un flag sbatch → confronto dei comandi; (2) `$SLURM_SUBMIT_DIR`/cwd perso nei job → ogni `_exe` fa `cd`; (3) riferimenti a file rimossi → grep finale.
- Devil's advocate: lasciare tutto e solo documentare — scartato, l'utente vuole le coppie.

## Verification Plan

- `bash -n`, `sbatch --test-only`, diff dei comandi, grep di riferimenti rotti.
