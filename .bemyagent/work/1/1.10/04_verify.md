# 1.10 — Verify

Verdetto: **PASS** (nessun rilancio, come chiesto; la prova vera è la prossima run).

| Criterio | Evidenza |
|---|---|
| Test | `PYTHONPATH=<worktree> .venv/bin/python -m pytest tests/data/test_loaders.py tests/data/test_negatives.py tests/experiment/train/test_preflight.py tests/models/worldsign/test_text_branch.py` → 21 pass; `test_the_checksum_record_is_published_whole` fallisce anche su 2a5e9a5 (ordine di `iterdir`), `test_the_effective_batch_must_be_128` va in errore solo nel worktree (manca `checkpoints/hub`), passa sul checkout |
| Soglie GloFND invariate | indice vero: vecchia `caption_statistics` e `distinct_captions` → stessa forma (883.929, 768), sha embedding `698516d81de7051b`, sha lingue `3fb1b3ba07897df4` |
| Lingua recuperata | nuova `caption_statistics` → (888.556, 768), 143 lingue, `en-xgJ5vofi3O8` presente |
| Fail fast | `statistics.pt` di `runs/youtube_sl25-A-1gpu-b40`: mancante `['en-xgJ5vofi3O8']` → `prepare` alza `ValueError`; rifit: nessuna mancante |
| Effetto sulle altre medie | norma della differenza vecchia/nuova: mediana 0, max 0,81 (lingue che perdevano caption condivise) |
| Lint | `ruff format` + `ruff check` sui 4 file → ok |
