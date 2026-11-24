Delivers: con le GPU che misurano curve P13 diverse la run prende un verdetto solo; una loss che scende senza dimezzarsi fa partire la run con un WARN in log e in `preflight.json`; un FAIL su qualunque GPU ferma tutte le GPU subito, non dopo 60 min.

- [x] `Status.WARN`; `p13_overfit(…, collective)`: loss mediate fra i rank, rapporto media(ultimi 5) / loss[0], PASS ≤ 0.5, WARN ≤ 0.9, FAIL altrimenti o non finita
- [x] `run_preflight`: log WARN con `logger.warning` sul rank 0; `enforce`: riporta WARN e SKIP, accetta `failed_elsewhere`
- [x] `run.py`: `collective.any(not report.passed)` → `enforce(..., failed_elsewhere=...)`
- [x] test: verdetto a tre fasce (funzione pura), `enforce` con `failed_elsewhere`
- [x] docs: CHECKS/descrizione, 05, 06

CDM
- 🎯 Drift: toccare la config (hash) o il resto del preflight
- ✅ Validation: pytest `tests/experiment/train/test_preflight.py` (i test veloci); ruff/mypy sui file toccati; il verdetto sulle curve vere dei log (rank 0 del job 64009434, debug 64003252)
- 🔄 Pivot: se i test di preflight hanno bisogno dell'hub e non girano qui, verificare con un test della funzione pura del verdetto
