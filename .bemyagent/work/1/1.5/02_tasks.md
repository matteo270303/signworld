Delivers: `prepare` di una run nuova calcola le statistiche in ~1 min invece di ~10,5, con gli stessi numeri.

- [ ] (cancellato: scancel --me) misurare su nodo GPU (job 63996194): pose serie/thread, mmap vs blocchi, `centering.fit`
- [x] `keypoint_statistics`: pose caricate con `ThreadPoolExecutor.map` (ordine preservato), `workers` configurabile
- [x] `caption_statistics`: righe lette dal mmap a blocchi contigui
- [x] `LanguageCentering.fit`: somme per lingua con `index_add_` a blocchi, senza la copia double intera e senza le 142 maschere
- [x] `read_index` → `select_split(table, split)`; `prepare` legge l'indice una volta sola, `run()` ne ricava gli split
- [x] `prepare` calcola le caption una volta (tutti i rank, prima della barrier), le passa a `fit_statistics` e a P11 tramite `Prepared`
- [x] test: thread = seriale, blocchi = fancy index, `fit` nuovo ≈ vecchio
- [x] verifica sui dati veri (login node) contro `runs/A-gate.failed/statistics.pt` + tempi; aggiornare 05

CDM
- 🎯 Drift: cambiare i numeri delle statistiche, cambiare il campione dei clip, aggiungere collective nel calcolo, implementare la cache (punto 4)
- ✅ Validation: test verdi; `keypoint_variance` identica e medie `allclose(atol=1e-6)` rispetto a `statistics.pt` di A-gate; `fit_statistics` < 2 min su nodo GPU
- 🔄 Pivot: se i thread non scalano su nodo GPU, o se più di 2 ostacoli → rivalutare B in 01
