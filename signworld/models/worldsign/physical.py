"""The physical predictor: V-JEPA 2.1's released predictor, reused (§4.4.5, §4.5.2).

From the visible tokens of a masked clip it predicts a token for every visible and masked
position (``predict_all``). The read-out averages the predicted tokens, visible and masked,
inside each articulator box of a step and maps the four boxes together to the pose target
``ŝ_t`` (gerarchia §4); the counts of masked and visible tokens in the boxes weigh the step in
``E_fis`` as V-JEPA 2.1 weighs its two losses. Changes to the released module:

* ``predictor_embed`` gives way to the multi-level fusion (``fusion``);
* ``predictor_proj`` and ``predictor_proj_context``, which project to the ViT-G teacher's
  1,664 dimensions, give way to the read-out head towards the pose target;
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
from .readout import StepReadout, box_sum


@dataclass(frozen=True, slots=True)
class PhysicalPrediction:
    """What one mask gives the physical energy, per step."""

    state: Tensor
    """(batch, steps, C) ŝ read from every predicted token in the step's boxes."""
    masked_count: Tensor
    """(batch, steps) masked tokens in the step's boxes, summed over the boxes."""
    visible_count: Tensor
    """(batch, steps) visible tokens in the step's boxes, summed over the boxes."""
    visible_weight: Tensor
    """(batch, steps) sum of those visible tokens' weights (1, or ``1 / √d``)."""
    box_tokens: Tensor
    """(batch, steps, parts) tokens in each box; 0 where the articulator is not visible."""

    @property
    def coverage(self) -> Tensor:
        """(batch, steps) share of the boxes' tokens that the mask hides."""
        total = self.masked_count + self.visible_count
        return self.masked_count / total.clamp_min(1.0)

    def weights(self, confidence: Tensor, context_lambda: float) -> Tensor:
        """(batch, steps) ``c_t · (n^m_t + λ · w^v_t)``: the step's weight in ``E_fis``."""
        return confidence * (self.masked_count + context_lambda * self.visible_weight)


class PhysicalPredictor(nn.Module):
    def __init__(  # noqa: PLR0913 (the released module and the four pieces that adapt it)
        self,
        predictor: Any,
        fusion: _Fusion,
        readout: StepReadout,
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
        predicted = members * (masked + visible)[:, :, None]  # tokens in either list
        return PhysicalPrediction(
            state=self.readout(tokens, predicted),
            masked_count=box_sum(masked, members).sum(dim=-1),
            visible_count=box_sum(visible, members).sum(dim=-1),
            visible_weight=box_sum(distance, members).sum(dim=-1),
            box_tokens=predicted.sum(dim=(-1, -2)),
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
