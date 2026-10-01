"""V-JEPA's multi-block masks and V-JEPA 2.1's distance weights (§4.5.2).

``MultiBlockMasks`` draws the blocks as ``src/masks/multiseq_multiblock3d._MaskGenerator`` of
the official repository: a random block size, ``blocks`` blocks at random positions, each
spanning every step (a tube), the union removed from the context. One draw serves every clip of
the batch on this GPU, so the mask is identical in every frame of a clip and nothing is cut.
V-JEPA draws the positions per clip and truncates every clip's lists to the shortest in the
batch; at 64 clips per GPU that would drop 43 % (short masks) and 74 % (long masks) of the
visible tokens, all from the last steps, so the clip with the most context would see it only
in its first 13 or 6 of 32 steps. Masks still change at every step, on every GPU, and between
the two kinds. ``MaskPolicy`` draws every kind at every step, as V-JEPA does with its short
(8 x 15 %) and long (2 x 70 %) masks.

``context_weights`` gives each visible token ``1 / √d``, with ``d`` its Euclidean distance on
the (step, row, column) grid to the nearest masked token: ``compute_mask_distance`` of
V-JEPA 2.1 with ``weight_distance_loss``, used in its pre-training. Its cooldown, which we
follow, weighs every visible token 1.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from signworld.experiment.train.config import MaskSpec


@dataclass(frozen=True, slots=True)
class TokenGrid:
    """Token layout of a clip: 64 frames in tubelets of 2, 256² in patches of 16."""

    steps: int = 32
    rows: int = 16
    columns: int = 16

    @property
    def size(self) -> int:
        return self.steps * self.rows * self.columns

    def positions(self, indices: Tensor) -> Tensor:
        """(..., 3) step, row and column of flat token indices."""
        per_step = self.rows * self.columns
        step, rest = indices // per_step, indices % per_step
        return torch.stack([step, rest // self.columns, rest % self.columns], dim=-1)


@dataclass(frozen=True, slots=True)
class Mask:
    """Indices, per clip, of the visible (context) and masked (target) tokens."""

    context: Tensor
    """(batch, n_context) long, sorted."""
    target: Tensor
    """(batch, n_target) long, sorted."""

    def to(self, device: torch.device) -> "Mask":
        return Mask(self.context.to(device), self.target.to(device))

    @property
    def ratio(self) -> float:
        visible, hidden = self.context.shape[1], self.target.shape[1]
        return hidden / (visible + hidden)


class MultiBlockMasks:
    """One kind of V-JEPA multi-block mask, sampled for a whole batch."""

    def __init__(self, spec: MaskSpec, grid: TokenGrid) -> None:
        self.spec = spec
        self.grid = grid

    def _block_size(self, generator: torch.Generator) -> tuple[int, int, int]:
        spec, grid = self.spec, self.grid
        uniform = torch.rand(3, generator=generator).tolist()
        steps = max(1, int(grid.steps * spec.temporal_scale))
        keep = int(grid.rows * grid.columns * spec.spatial_scale)
        low, high = spec.aspect_ratio
        ratio = low + uniform[2] * (high - low)
        rows = min(round(math.sqrt(keep * ratio)), grid.rows)
        columns = min(round(math.sqrt(keep / ratio)), grid.columns)
        return steps, rows, columns

    def _clip(self, size: tuple[int, int, int], generator: torch.Generator) -> Tensor:
        """(steps, rows, columns) with 0 where a block covers the token."""
        grid = self.grid
        steps, rows, columns = size
        keep = torch.ones(grid.steps, grid.rows, grid.columns, dtype=torch.bool)
        for _ in range(self.spec.blocks):
            top = int(torch.randint(0, grid.rows - rows + 1, (1,), generator=generator))
            left = int(torch.randint(0, grid.columns - columns + 1, (1,), generator=generator))
            start = int(torch.randint(0, grid.steps - steps + 1, (1,), generator=generator))
            keep[start : start + steps, top : top + rows, left : left + columns] = False
        return keep.flatten()

    def __call__(self, batch: int, generator: torch.Generator) -> Mask:
        """One mask for the ``batch`` clips: every token is either visible or masked."""
        size = self._block_size(generator)
        keep = self._clip(size, generator)
        while not keep.any():  # an all-masked draw is redrawn, as in V-JEPA
            keep = self._clip(size, generator)
        context = torch.nonzero(keep).squeeze(1)
        target = torch.nonzero(~keep).squeeze(1)
        return Mask(context.expand(batch, -1), target.expand(batch, -1))


class MaskPolicy:
    """Every mask kind of the settings, drawn together at each step."""

    def __init__(self, specs: tuple[MaskSpec, ...], grid: TokenGrid) -> None:
        self.generators = [MultiBlockMasks(spec, grid) for spec in specs]
        self.grid = grid

    def __call__(self, batch: int, generator: torch.Generator) -> list[Mask]:
        return [make(batch, generator) for make in self.generators]


def context_weights(mask: Mask, grid: TokenGrid) -> Tensor:
    """(batch, n_context) ``1 / √d_min``: V-JEPA 2.1's weights of the visible tokens."""
    shared = bool(
        (mask.context == mask.context[:1]).all() and (mask.target == mask.target[:1]).all()
    )
    if shared and len(mask.context) > 1:  # one mask for the batch: measure it once
        first = context_weights(Mask(mask.context[:1], mask.target[:1]), grid)
        return first.expand(len(mask.context), -1)
    context = grid.positions(mask.context).float()
    target = grid.positions(mask.target).float()
    nearest = torch.stack(
        [
            torch.cdist(c[None], t[None])[0].min(dim=1).values
            for c, t in zip(context, target, strict=True)
        ]
    )
    return nearest.clamp_min(1.0).rsqrt()


@dataclass(frozen=True, slots=True)
class TokenRoles:
    """What each token of the grid is for one mask, as (batch, N) tensors.

    A token the batch truncation left out of both lists is 0 in all three.
    """

    masked: Tensor
    """1 for a masked token: it enters V-JEPA 2.1's ``L_pred``."""
    visible: Tensor
    """1 for a visible token: it enters ``L_ctx`` (``predict_all``)."""
    distance: Tensor
    """Weight of a visible token in ``L_ctx`` (1, or ``1 / √d``), 0 elsewhere."""


def token_roles(mask: Mask, grid: TokenGrid, weight_distance: bool = False) -> TokenRoles:
    """Masked and visible indicators and the visible tokens' weights on the grid."""
    batch, device = mask.context.shape[0], mask.context.device
    masked = torch.zeros(batch, grid.size, device=device).scatter_(1, mask.target, 1.0)
    visible = torch.zeros(batch, grid.size, device=device).scatter_(1, mask.context, 1.0)
    if not weight_distance:
        return TokenRoles(masked, visible, visible.clone())
    distance = torch.zeros(batch, grid.size, device=device).scatter_(
        1, mask.context, context_weights(mask, grid)
    )
    return TokenRoles(masked, visible, distance)


@dataclass(frozen=True, slots=True)
class LambdaSchedule:
    """λ of the visible-token term: 0, then linear up to ``value``, then held (V-JEPA 2.1)."""

    value: float
    start: float
    end: float
    """``start`` and ``end`` as fractions of the run."""

    @classmethod
    def constant(cls, value: float) -> "LambdaSchedule":
        """λ from the first step, as V-JEPA 2.1's cooldown."""
        return cls(value, 0.0, 0.0)

    def at(self, step: int, total: int) -> float:
        progress = step / max(total, 1)
        if progress < self.start:
            return 0.0
        if progress >= self.end:
            return self.value
        return self.value * (progress - self.start) / (self.end - self.start)
