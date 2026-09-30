"""Tiny but real V-JEPA 2.1 modules, built from Meta's code with a few channels and frames."""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from signworld.worldmodel.config import load_config
from signworld.worldmodel.masking import TokenGrid

HUB = Path("/lustrehome/mvigone/cache/torch/hub/facebookresearch_vjepa2_main")
BASE = Path(__file__).resolve().parents[2] / "configs" / "model" / "worldsign.yaml"
ABLATIONS = BASE.parent / "ablations"
GRID = TokenGrid(steps=2, rows=4, columns=4)
"""4 frames of 64² in tubelets of 2 and patches of 16."""

needs_hub = pytest.mark.skipif(not HUB.is_dir(), reason="V-JEPA 2.1 hub code not available")


def meta_modules() -> tuple[Any, Any]:
    """A 12-block encoder (so its hierarchical layers exist) and a 4-block predictor."""
    if str(HUB) not in sys.path:
        sys.path.insert(0, str(HUB))
    from app.vjepa_2_1.models import predictor as vit_predictor  # noqa: PLC0415 (after sys.path)
    from app.vjepa_2_1.models import vision_transformer as vit_encoder  # noqa: PLC0415

    torch.manual_seed(0)
    encoder = vit_encoder.VisionTransformer(
        img_size=(64, 64),
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=48,
        depth=12,
        num_heads=4,
        use_rope=True,
        interpolate_rope=True,
        n_output_distillation=1,
        img_temporal_dim_size=1,
    )
    predictor = vit_predictor.vit_predictor(
        img_size=(64, 64),
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=48,
        predictor_embed_dim=24,
        depth=4,
        num_heads=2,
        num_mask_tokens=2,
        use_mask_tokens=True,
        use_rope=True,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=40,
        img_temporal_dim_size=1,
    )
    return encoder, predictor


def tiny_config(*overlays: Path, **changes: Any) -> Any:
    """The project's YAML with the sizes of the tiny modules."""
    config = load_config(BASE, *overlays)
    encoder = config.encoder.model_copy(update={"levels": (2, 5, 8, 11)})
    semantic = config.semantic.model_copy(
        update={"width": 24, "heads": 2, "dropout": 0.0, "drop_path": 0.0}
    )
    physical = config.physical.model_copy(update={"target_dim": 8})
    update = {"encoder": encoder, "semantic": semantic, "physical": physical, **changes}
    return config.model_copy(update=update)
