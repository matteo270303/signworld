"""Frozen video encoders compared by PC3 and PC4 and inspected by PC6 (§4.4.1, §4.12.1).

Two families, behind one interface that takes uint8 frames and returns a token grid:

* ``vjepa2_1``: V-JEPA 2.1 (ViT-L and ViT-B distilled from ViT-G), built with Meta's own code
  from a local copy of the torch.hub repository (the compute nodes have no internet) and loaded
  from a saved checkpoint, either the ``{encoder, predictor}`` export or the released file with
  ``ema_encoder``; every key must load.
* ``vjepa2_hf``: V-JEPA 2 ViT-L through ``transformers`` (its encoder only), from the
  Hugging Face cache.

Pixels are scaled to [0, 1] and normalised with the ImageNet mean and deviation, which is what
both releases were trained with.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn

from signworld.experiment.collaudo.analysis import VideoEncoderSettings

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)
PATCH: Final = 16
TUBELET: Final = 2


def model_input(frames: Tensor) -> Tensor:
    """(batch, T, H, W, 3) uint8 RGB to (batch, 3, T, H, W) float, ImageNet-normalised."""
    if frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError(f"expected (batch, T, H, W, 3) uint8 frames, got {tuple(frames.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frames.device).view(1, 3, 1, 1, 1)
    scaled = frames.permute(0, 4, 1, 2, 3).float() / 255.0
    return (scaled - mean) / std


def token_grid(tokens: Tensor, frames: int, height: int, width: int) -> Tensor:
    """(batch, N, D) tokens in (time, row, column) order to (batch, T/2, H/16, W/16, D)."""
    grid = (frames // TUBELET, height // PATCH, width // PATCH)
    if tokens.shape[1] != grid[0] * grid[1] * grid[2]:
        raise ValueError(f"{tokens.shape[1]} tokens do not fill a {grid} grid")
    return tokens.reshape(tokens.shape[0], *grid, tokens.shape[-1])


@dataclass(frozen=True, slots=True)
class LoadReport:
    missing: list[str]
    unexpected: list[str]


def _clean(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keys without the ``module.`` and ``backbone.`` prefixes of the training wrappers."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}


def load_vjepa2_1(
    hub_repo: Path, entrypoint: str, checkpoint: Path
) -> tuple[nn.Module, nn.Module, dict[str, LoadReport]]:
    """Meta's encoder and predictor, built offline and loaded from ``checkpoint``."""
    # The hub code imports ``src.*`` and ``app.*`` from the repository root.
    if str(hub_repo) not in sys.path:
        sys.path.insert(0, str(hub_repo))
    hub: Any = torch.hub
    encoder, predictor = hub.load(
        str(hub_repo), entrypoint, source="local", pretrained=False, trust_repo=True
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder_key = "encoder" if "encoder" in state else "ema_encoder"
    reports = {}
    for name, module, key in (
        ("encoder", encoder, encoder_key),
        ("predictor", predictor, "predictor"),
    ):
        result = module.load_state_dict(_clean(state[key]), strict=False)
        reports[name] = LoadReport(list(result.missing_keys), list(result.unexpected_keys))
    return encoder.eval(), predictor.eval(), reports


class FrozenVideoEncoder:
    """A frozen encoder that turns uint8 clips into a grid of tokens."""

    def __init__(self, name: str, module: nn.Module, dimension: int, family: str) -> None:
        self.name = name
        # Meta's modules are untyped and carry plain attributes (out_layers, embed_dim).
        self.module: Any = module.eval().requires_grad_(False)
        self.dimension = dimension
        self.family = family

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.module.parameters()).device
        return device

    def to(self, device: torch.device) -> "FrozenVideoEncoder":
        self.module.to(device)
        return self

    def forward_video(self, video: Tensor) -> Tensor:
        """(batch, 3, T, H, W) normalised video to (batch, N, D) tokens."""
        if self.family == "vjepa2_hf":
            output = self.module(pixel_values_videos=video.permute(0, 2, 1, 3, 4))
            return output.last_hidden_state  # type: ignore[no-any-return]
        return self.module(video)  # type: ignore[no-any-return]

    @torch.no_grad()
    def tokens(self, frames: Tensor, *, bf16: bool = True) -> Tensor:
        """(batch, T, H, W, 3) uint8 to (batch, T/2, H/16, W/16, D) float32."""
        video = model_input(frames.to(self.device))
        enabled = bf16 and self.device.type == "cuda"
        with torch.autocast(self.device.type, torch.bfloat16, enabled=enabled):
            out = self.forward_video(video)
        _, _, count, height, width = video.shape
        return token_grid(out.float(), count, height, width)

    @torch.no_grad()
    def levels(self, frames: Tensor, layers: Sequence[int]) -> list[Tensor]:
        """V-JEPA 2.1 only: the normalised outputs of ``layers`` (its hierarchical layers)."""
        if self.family != "vjepa2_1":
            raise ValueError(f"{self.name}: per-level outputs exist only for V-JEPA 2.1")
        video = model_input(frames.to(self.device))
        self.module.out_layers = list(layers)
        try:
            outs: list[Tensor] = self.module(video)
        finally:
            self.module.out_layers = None
        return [out.float() for out in outs]


def build_encoder(
    name: str, settings: VideoEncoderSettings, hub_repo: Path, checkpoint: Path | None
) -> tuple[FrozenVideoEncoder, Any]:
    """The encoder, and the predictor and load report for V-JEPA 2.1 (``None`` otherwise)."""
    if settings.family == "vjepa2_1":
        if settings.entrypoint is None or checkpoint is None:
            raise ValueError(f"{name}: V-JEPA 2.1 needs an entry point and a checkpoint")
        encoder, predictor, reports = load_vjepa2_1(hub_repo, settings.entrypoint, checkpoint)
        missing = [key for report in reports.values() for key in report.missing]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} weights missing, e.g. {missing[:3]}")
        width: Any = encoder.embed_dim
        frozen = FrozenVideoEncoder(name, encoder, int(width), settings.family)
        return frozen, (predictor, reports)
    from transformers import VJEPA2Model  # noqa: PLC0415 (heavy import, only when used)

    if settings.model_id is None:
        raise ValueError(f"{name}: V-JEPA 2 needs a Hugging Face model_id")
    model = VJEPA2Model.from_pretrained(settings.model_id)
    return FrozenVideoEncoder(name, model, int(model.config.hidden_size), settings.family), None


def box_pool(grid: Tensor, boxes: np.ndarray, visible: np.ndarray) -> Tensor:
    """Mean token inside each box, per step: (steps, parts, D).

    ``grid`` is (steps, rows, columns, D); ``boxes`` (steps, parts, 4) holds x0, y0, x1, y1 in
    frame fractions and ``visible`` (steps, parts) says which exist. A box covers every patch it
    touches; an invisible one gives zeros.
    """
    steps, rows, columns, dimension = grid.shape
    pooled = torch.zeros(steps, boxes.shape[1], dimension, device=grid.device)
    cells = np.clip(np.nan_to_num(boxes, nan=0.0), 0.0, 1.0)
    x0 = np.minimum(np.floor(cells[..., 0] * columns).astype(int), columns - 1)
    y0 = np.minimum(np.floor(cells[..., 1] * rows).astype(int), rows - 1)
    x1 = np.clip(np.ceil(cells[..., 2] * columns).astype(int), x0 + 1, columns)
    y1 = np.clip(np.ceil(cells[..., 3] * rows).astype(int), y0 + 1, rows)
    for step in range(steps):
        for part in range(boxes.shape[1]):
            if visible[step, part]:
                region = grid[
                    step, y0[step, part] : y1[step, part], x0[step, part] : x1[step, part]
                ]
                pooled[step, part] = region.mean(dim=(0, 1))
    return pooled
