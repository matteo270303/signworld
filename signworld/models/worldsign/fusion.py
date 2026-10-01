"""Multi-level input of the physical predictor (§4.4.5), in three interchangeable forms.

Blocks 6/12/18/24 of the encoder, each through its own LayerNorm, are concatenated (4 x 1,024)
and mapped to the predictor's width (384). The LayerNorms start from the encoder's
``norms_block`` (PC6: only the last is trained in the distilled checkpoints) and train here.

* ``MLPFusion``: V-JEPA 2.1's ``predictor_embed`` for four levels, Linear → GELU → Linear;
* ``LinearFusion``: one Linear starting as the released layer on the last level, zero on the
  others, so the predictor computes exactly the released one at step 0 (PC6);
* ``ResidualFusion``: the released layer on the last level plus the MLP on all four, its last
  layer at zero: exact at step 0 and with the MLP's capacity.
"""

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from signworld.experiment.train.config import FusionSettings
from signworld.models.encoders.predictor_fusion import fused_embedding


class _Fusion(nn.Module):
    def __init__(self, norms: Sequence[nn.LayerNorm]) -> None:
        super().__init__()
        self.norms = nn.ModuleList(copy.deepcopy(norm).requires_grad_(True) for norm in norms)

    def concatenated(self, levels: Sequence[Tensor]) -> Tensor:
        if len(levels) != len(self.norms):
            raise ValueError(f"expected {len(self.norms)} levels, got {len(levels)}")
        return torch.cat([norm(level) for norm, level in zip(self.norms, levels, strict=True)], -1)


class MLPFusion(_Fusion):
    def __init__(self, norms: Sequence[nn.LayerNorm], width: int, hidden: int, output: int) -> None:
        super().__init__(norms)
        self.net = nn.Sequential(
            nn.Linear(len(norms) * width, hidden), nn.GELU(), nn.Linear(hidden, output)
        )

    def forward(self, levels: Sequence[Tensor]) -> Tensor:
        fused: Tensor = self.net(self.concatenated(levels))
        return fused


class LinearFusion(_Fusion):
    def __init__(self, norms: Sequence[nn.LayerNorm], released: nn.Linear) -> None:
        super().__init__(norms)
        self.linear = fused_embedding(released, len(norms))

    def forward(self, levels: Sequence[Tensor]) -> Tensor:
        fused: Tensor = self.linear(self.concatenated(levels))
        return fused


class ResidualFusion(_Fusion):
    def __init__(self, norms: Sequence[nn.LayerNorm], released: nn.Linear, hidden: int) -> None:
        super().__init__(norms)
        self.released = copy.deepcopy(released).requires_grad_(False)
        width, output = released.in_features, released.out_features
        last = nn.Linear(hidden, output)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.branch = nn.Sequential(nn.Linear(len(norms) * width, hidden), nn.GELU(), last)

    def forward(self, levels: Sequence[Tensor]) -> Tensor:
        # The last level goes through the released layer with its own, released, norm.
        fused: Tensor = self.released(self.norms[-1](levels[-1])) + self.branch(
            self.concatenated(levels)
        )
        return fused


def build_fusion(
    settings: FusionSettings, norms: Sequence[nn.LayerNorm], released: nn.Linear
) -> _Fusion:
    """The configured fusion; ``released`` is the distilled predictor's ``predictor_embed``."""
    if settings.kind == "linear":
        return LinearFusion(norms, released)
    if settings.kind == "residual":
        return ResidualFusion(norms, released, settings.hidden)
    return MLPFusion(norms, released.in_features, settings.hidden, released.out_features)
