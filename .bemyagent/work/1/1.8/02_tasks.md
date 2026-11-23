Delivers: `signworld train profile <config> --output <dir>` stampa e salva (`profile.json`) quanti secondi di ogni passo vanno in dati, encoder, predittori, backward e AdamW, per gli stage P/F0/F, con le ore per epoca a quel ritmo.

- [x] verified `Clock` (sezioni annidate, secondi per passo) + `instrument` (wrapper su pose/physical/semantic, encoder intero/mascherato, predittore, query, testo)
- [x] verified `profile_stage`: passo del trainer replicato con sync per sezione, stage forzato, rate non nulli
- [x] verified `profile_clip_reads` + `loader_rate`: fasi di `__getitem__`, clip/s del loader del run
- [x] verified `profile_attention`: flash / efficient / cuDNN a 8.192 token
- [x] verified CLI `train profile` (`--clips`, `--stages`, `--no-gpu`, ...)
- [x] verified test, ruff, mypy
- [x] verified misure su youtube_sl25 → draft `lentezza-stage-f.md`, 05, 03, 06

CDM
- ✅ Validation: test verde; ruff + mypy strict puliti sui file toccati; `train profile` gira sul run vero (dati su CPU + stage P/F0/F su GPU) e la somma delle sezioni figlie non supera la sezione padre
