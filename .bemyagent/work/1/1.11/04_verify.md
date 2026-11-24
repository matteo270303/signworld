# 1.11 — Verify

Verdetto: **PASS_WITH_CAVEATS**: misure a 8 clip su GPU condivisa, non a batch 40 a GPU libera. Verifica fatta dallo stesso agente che ha scritto il codice: nessun secondo agente indipendente.

| Criterio | Evidenza |
|---|---|
| LoRA fusa = LoRA | `test_lora_merges_its_weight_only_where_no_adapter_needs_a_gradient` (split 1 e 3, float64): uscita uguale senza gradiente; percorso low-rank e gradienti degli adattatori quando si addestrano; gradiente sull'input uguale con adattatori congelati → pass |
| RoPE fuso = RoPE di Meta | `test_meta_rope.py`: `torch.equal` in eager fp32 e bf16, con e senza maschera, testa da 64 (3×20 ruotati + 4); encoder tiny intero e mascherato con `atol=rtol=0` → 5 pass |
| Augmentation = per clip | `test_augmentation.py` CPU e CUDA, `torch.equal` → pass; 64×256², 20 clip: 75 byte su 251.658.240 a ±1 (`bench_augment.py`) |
| Compilato = eager | `SIGNWORLD_SLOW_TESTS=1` su models/data/experiment: 208 pass; i 4 fail (`test_results`, `test_a_short_run…`, `test_the_single_batch_overfit…`, `test_the_checksum_record…`) falliscono identici con i sorgenti di HEAD 45eb052 rimessi al loro posto → preesistenti |
| Più veloce, stessa memoria | `train profile` 8 clip, GPU 0 condivisa: P 1,143 → 0,945 s, F0 2,690 → 2,207, F 3,765 → 3,276; picco 7,8/11,7/14,1 GiB uguale; nessun `CheckpointError` in F |
| Codice Meta intatto | modifiche solo in `signworld/`, `tests/`, `scripts/perf/` (git status) |
| Lint | ruff check/format e mypy puliti su `lora.py`, `meta_rope.py`, `backbone.py`, `augmentation.py` (resta il `block_diag` preesistente in `lora.py`) |

Caveat:
- Rapporti misurati a 8 clip su GPU condivisa: a batch 40 la quota dell'encoder cambia. Va ripetuto `signworld train profile` a b40 a GPU libera.
- Il run in corso (`-r3`) col codice nuovo chiede `--allow-code-change` al resume; le loss cambiano solo nell'arrotondamento bf16.
- Non provato in DDP né su H100/A100.
