Delivers: una run di YouTube-SL-25 non muore più con `KeyError` in `Collate` per una lingua le cui caption sono tutte condivise; con un `statistics.pt` vecchio si ferma all'avvio con un messaggio che dice cosa cancellare.

- [x] `loaders.py`: `caption_statistics` per coppia `(caption_row, caption_language)`; `distinct_captions` con la vecchia semantica; lettura a blocchi in comune
- [x] `negatives.py`: usa `distinct_captions`
- [x] `run.py` `prepare`: dopo `load_statistics`, errore se una lingua del training manca dal centring
- [x] `test_loaders.py`: test delle due funzioni (duplicati, blocchi) e del caso del bug
- [x] 05, 06

CDM
- 🎯 Drift: toccare la normalizzazione delle lingue, cambiare le soglie GloFND, rilanciare la run
- ✅ Validation: `pytest tests/data/test_loaders.py tests/experiment/train/test_preflight.py tests/data/test_negatives*` verdi; sullo `index.parquet` vero `caption_statistics` restituisce 143 lingue e `distinct_captions` le stesse righe di prima; il controllo di `prepare` rifiuta lo `statistics.pt` della run morta
- 🔄 Pivot: se `negatives.py` o P11 dipendono dalla forma in altri punti non visti (>2 ostacoli) → tenere `caption_statistics` invariata e calcolare le lingue del centring a parte
