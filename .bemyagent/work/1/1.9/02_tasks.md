Delivers: con `encoder.compile_blocks: true` (attivo in `worldsign.yaml`) un passo di training costa il 22-33 % in meno, con le stesse loss e gli stessi gradienti.

- [x] verified `drop_cudnn_attention` su `torch.nn.attention.sdpa_kernel` (stessi backend, cuDNN escluso, 1 grafo per blocco)
- [x] verified `compile_blocks`: forward per blocco, codice proprio per gruppo, dimensioni dinamiche, thread allineati, `fallback_random`, `recompile_limit` 64
- [x] verified `EncoderSettings.compile_blocks` (default false), passato a `PhysicalPredictor`
- [x] verified `worldsign.yaml`: `compile_blocks: true`; `tiny_config` lo forza a false
- [x] verified test: binding/deepcopy; equivalenza eager/compilato (slow); smoke `train profile`
- [x] verified equivalenza sul modello vero, profilo b40, `train run` end-to-end
- [x] verified ruff + mypy; 03, 05, 06, draft

CDM: vedi `01_think.md`.
