# Stack tecnologico

Fonte di verità: `pyproject.toml` (le versioni sono limiti inferiori; i pin esatti sono in `uv.lock`).

| Nome | Versione | Ruolo | Perché |
|---|---|---|---|
| Python | 3.12 (gestito da uv) | runtime | gli interpreti di sistema mancano sui nodi di calcolo |
| uv / hatchling | — | ambiente + build | |
| torch / torchvision | ≥2.6 / ≥0.21 | addestramento, DDP con torchrun | |
| V-JEPA 2.1 ViT-L (torch.hub `facebookresearch/vjepa2`) | ckpt `vjepa2_1_vitl_384.pt` | encoder video + predictor congelati | richiede `einops`, `timm` |
| sentence-transformers / transformers | ≥5.1 / ≥4.56 | didascalie con EmbeddingGemma-300M | accesso riservato su HF |
| rtmlib + onnxruntime-gpu | ≥0.0.13 / ≥1.30 | posa wholebody | |
| decord, opencv-python-headless | ≥0.6, ≥4.10 | decodifica video | headless: niente libGL su ReCaS |
| pyarrow, numpy, pydantic, pyyaml | | manifest, configurazioni | |
| typer | ≥0.15 | CLI | |
| yt-dlp + bgutil-ytdlp-pot-provider | ≥2026.8.19 / 1.3.1 | download da YouTube | token PO necessario da IP di datacenter |
| pytest, ruff, mypy | dev | qualità | |
| wandb | opzionale, non è una dipendenza | mirror offline delle run | |

## Problemi di compatibilità noti

- `opencv-python` / `opencv-contrib-python` / `onnxruntime` CPU sono esclusi in `[tool.uv]`: richiedono libGL o sovrascrivono i file di `onnxruntime-gpu`.
- `pyproject.toml` è mantenuto a mano — il proprietario si oppone esplicitamente a modifiche automatiche; proporre le modifiche, non applicarle.

## Servizi esterni

| Nome | Uso |
|---|---|
| YouTube (yt-dlp, cookie `.secrets/youtube_cookies.txt`, `deno`, server PO bgutil) | media di youtube_sl25, openasl |
| Hugging Face Hub | dataset csl_news, EmbeddingGemma |
| BBC BOBSL | bobsl (credenziali) |
| W&B (offline, `wandb sync`) | opzionale |

## Infrastruttura

- HTCondor ReCaS (`slurm/condor/*.sub`, schedd `ettore`; `initialdir` assoluto)
- Slurm (`slurm/launch_*`, intestazioni `#SBATCH` segnaposto, cache in `$SCRATCH`)
- Multi-GPU locale (2 GPU; batch effettivo 128)
