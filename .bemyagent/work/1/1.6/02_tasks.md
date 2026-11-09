Delivers: nel log del debug le diagnostiche rare durano minuti e non un quarto d'ora, le righe P compaiono una volta con cosa controllano e quanto ci mettono, e le fasi lunghe hanno una barra tqdm.

- [x] `Monitor.rare` + `plausibility_tests`: autocast bf16 come la validazione
- [x] preflight: descrizione per codice, secondi per asserzione, log sul primo GPU (FAIL su ogni rank), SKIP di `enforce` solo sul primo GPU
- [x] tqdm: passi di P13 (per le altre asserzioni bastano i secondi nella riga), plausibilità (batch × varianti), pose di `keypoint_statistics`
- [x] warning: `float(total.detach())` in P13; DDP `broadcast_buffers` silenziato localmente; `timm.models.layers` e `sdp_kernel` filtrati per messaggio
- [x] `run.py`: "N training clips" solo sul primo GPU
- [x] test + lint; 05 aggiornato (03 invariato: nessun file o modulo nuovo)

CDM
- 🎯 Drift: cambiare cosa misurano le asserzioni o le rare; cambiare la semantica DDP; lanciare job
- ✅ Validation: test verdi; nessun `float(` su tensori con grad; grep mostra autocast in rare/plausibility; un test sul log del preflight (una riga per codice, solo primo GPU)
- 🔄 Pivot: se bf16 rompe i test di plausibilità per tolleranze → autocast solo nel forward fisico
