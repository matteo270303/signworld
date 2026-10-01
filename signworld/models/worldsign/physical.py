"""The physical predictor: V-JEPA 2.1's released predictor, reused (§4.4.5, §4.5.2).

From the visible tokens of a masked clip it predicts a token for every visible and masked
position (``predict_all``). The read-out averages them per step and articulator box and maps
them to the pose latent twice, as V-JEPA 2.1 keeps its two losses apart: once over the masked
tokens in the box (``L_pred``), once over the visible ones (``L_ctx``). Changes to the released
module:

* ``predictor_embed`` gives way to the multi-level fusion (``fusion``);
* ``predictor_proj`` and ``predictor_proj_context``, which project to the ViT-G teacher's
  1,664 dimensions, give way to the read-out head towards the pose latent;
* LoRA on the 12 blocks; the rest stays frozen;
* the RoPE grid is set to the clip's: the released module fixes it at construction (24 x 24
  patches for 384²) and does not interpolate, so at 256² every token would be decoded to the
  wrong row and column;
* ``mask_index = 0``: the only mask token the distilled checkpoint trained (PC6).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from signworld.experiment.train.config import LoRASettings

from . import lora
from .fusion import _Fusion
from .masking import Mask, TokenGrid, TokenRoles
from .readout import ArticulatorReadout, box_sum


@dataclass(frozen=True, slots=True)
class PhysicalPrediction:
    """What one mask gives the physical energy, per step and articulator box."""

    masked: Tensor
    """(batch, steps, parts, C) ŝ read from the predicted masked tokens in the box."""
    visible: Tensor
    """(batch, steps, parts, C) ŝ read from the predicted visible tokens in the box."""
    masked_count: Tensor
    """(batch, steps, parts) masked tokens in the box."""
    visible_count: Tensor
    """(batch, steps, parts) visible tokens in the box."""
    visible_weight: Tensor
    """(batch, steps, parts) sum of the visible tokens' weights in the box (1, or ``1 / √d``)."""


class PhysicalPredictor(nn.Module):
    def __init__(  # noqa: PLR0913 (the released module and the four pieces that adapt it)
        self,
        predictor: Any,
        fusion: _Fusion,
        readout: ArticulatorReadout,
        adapters: LoRASettings,
        grid: TokenGrid,
        *,
        mask_index: int = 0,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        # Meta's module is untyped; its embedding and projections are replaced here.
        released: Any = predictor.requires_grad_(False)
        released.predictor_embed = nn.Identity()
        released.predictor_proj = nn.Identity()
        released.predictor_proj_context = nn.Identity()
        released.use_activation_checkpointing = activation_checkpointing
        set_rope_grid(released, grid.rows)
        blocks: nn.Module = released.predictor_blocks
        self.adapters = lora.inject(blocks, adapters.targets, adapters.rank, adapters.alpha)
        self.predictor: Any = released
        self.fusion = fusion
        self.readout = readout
        self.grid = grid
        self.mask_index = mask_index

    @property
    def width(self) -> int:
        return int(self.predictor.predictor_norm.normalized_shape[0])

    def tokens(self, levels: Sequence[Tensor], mask: Mask) -> Tensor:
        """(batch, N, width) predicted token at every grid position; zero where neither list."""
        fused = self.fusion(levels)
        predicted, context = self.predictor(
            fused, [mask.context], [mask.target], mask_index=self.mask_index
        )
        batch = fused.shape[0]
        grid = torch.zeros(
            batch, self.grid.size, self.width, device=fused.device, dtype=predicted.dtype
        )
        grid = grid.scatter(1, mask.target[..., None].expand(-1, -1, self.width), predicted)
        return grid.scatter(
            1, mask.context[..., None].expand(-1, -1, self.width), context.to(grid.dtype)
        )

    def forward(
        self, levels: Sequence[Tensor], mask: Mask, members: Tensor, roles: TokenRoles
    ) -> PhysicalPrediction:
        """``members`` (batch, steps, parts, rows, columns); ``roles`` of the same ``mask``."""
        g = self.grid
        tokens = self.tokens(levels, mask).view(-1, g.steps, g.rows, g.columns, self.width)
        masked, visible, distance = (
            role.view(-1, g.steps, g.rows, g.columns)
            for role in (roles.masked, roles.visible, roles.distance)
        )
        return PhysicalPrediction(
            masked=self.readout(tokens, members * masked[:, :, None]),
            visible=self.readout(tokens, members * visible[:, :, None]),
            masked_count=box_sum(masked, members),
            visible_count=box_sum(visible, members),
            visible_weight=box_sum(distance, members),
        )


def set_rope_grid(module: nn.Module, rows: int) -> None:
    """Point every RoPE attention under ``module`` at a ``rows`` x ``rows`` patch grid."""
    found = False
    for layer in module.modules():
        if hasattr(layer, "grid_size") and hasattr(layer, "separate_positions"):
            layer.grid_size = rows  # type: ignore[assignment]
            found = True
    if not found:
        raise ValueError(f"no RoPE attention under {type(module).__name__}")
