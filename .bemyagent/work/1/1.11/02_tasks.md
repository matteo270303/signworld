Delivers: con `encoder.compile_blocks: true` un passo costa ~13-18 % in meno di 1.9 (8 clip: P −17 %, F0 −18 %, F −13 %), stessa memoria, stessa funzione.

- [x] verified `lora.MERGE_WITHOUT_GRADIENT`: `LoRALinear` con un solo prodotto `W + s·B·A` (fp32, fuori dall'autocast) quando nessun adattatore chiede gradiente (passata semantica sempre; P/F0 anche contesto e predittore)
- [x] verified `meta_rope.fuse_rope` in `compile_blocks`: una rotazione della testa al posto di sei fette, forward Meta per i casi non coperti
- [x] verified `apply_views` del batch intero, senza sync per clip
- [x] verified test: `test_lora_merges_its_weight_only_where_no_adapter_needs_a_gradient`, `tests/models/worldsign/test_meta_rope.py`, `tests/data/test_augmentation.py`; test lento compilato/eager
- [x] verified benchmark in `scripts/perf/` (`bench_encoder.py`, `bench_augment.py`, `trace_step.py`)
- [x] verified `train profile` 8 clip prima/dopo
- [x] verified ruff + mypy; 03, 05, 06, draft `lentezza-stage-f.md`
- [ ] todo rimisurare a batch 40 a GPU libera (la GPU 0 era condivisa col run, GPU 1-3 "Prohibited")

CDM: vedi `01_think.md`.
