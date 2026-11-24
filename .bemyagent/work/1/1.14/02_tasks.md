Delivers: un'eccezione su un solo rank di `train run` o `train benchmark` fa uscire tutto il job in pochi secondi con il traceback nel log, invece di lasciare le altre GPU ferme fino al timeout NCCL (60 min).

- [x] verified `Distributed.exit_on_error()`: con più rank traceback, `logging.shutdown`, flush, `os._exit(1)`; con un rank solo ritorna
- [x] verified `cli/train.py` `run` e `benchmark`: `except BaseException: exit_on_error(); raise`, poi `shutdown()` solo in caso di successo
- [x] verified docs: 03, 05, 06

CDM
- ✅ Validation: ruff/mypy sui file toccati senza errori nuovi; test di `tests/experiment/train/`; il comportamento multi-rank si vede solo sul cluster
