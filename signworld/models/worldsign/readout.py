"""Per-step read-out of a grid of tokens (gerarchia §4).

The physical level predicts a token for every position of the grid; its target is one vector
per step. ``membership`` turns the articulator boxes (frame fractions, from the keypoints) into
which patches each box touches, ``pool`` averages the tokens inside each box, in one tensor
operation for a whole batch, and ``StepReadout`` concatenates the four boxes of a step and
projects them to the target, as the pose encoder concatenates its four parts. A box covers
every patch it touches, as in the probes of the collaudo (``models.video_encoders.box_pool``).
"""

import torch
from torch import Tensor, nn


def membership(boxes: Tensor, visible: Tensor, rows: int, columns: int) -> Tensor:
    """(batch, steps, parts, rows, columns) 1 where a visible box touches the patch.

    ``boxes`` is (batch, steps, parts, 4) with x0, y0, x1, y1 in frame fractions (NaN where
    missing); ``visible`` is (batch, steps, parts).
    """
    cells = torch.nan_to_num(boxes, nan=0.0).clamp(0.0, 1.0)
    x0 = (cells[..., 0] * columns).floor().clamp(max=columns - 1)
    y0 = (cells[..., 1] * rows).floor().clamp(max=rows - 1)
    x1 = torch.maximum((cells[..., 2] * columns).ceil(), x0 + 1).clamp(max=columns)
    y1 = torch.maximum((cells[..., 3] * rows).ceil(), y0 + 1).clamp(max=rows)
    column = torch.arange(columns, device=boxes.device, dtype=boxes.dtype)
    row = torch.arange(rows, device=boxes.device, dtype=boxes.dtype)
    inside_x = (column >= x0[..., None]) & (column < x1[..., None])
    inside_y = (row >= y0[..., None]) & (row < y1[..., None])
    inside = inside_y[..., :, None] & inside_x[..., None, :]
    return (inside & visible[..., None, None]).to(boxes.dtype)


def pool(tokens: Tensor, members: Tensor) -> Tensor:
    """(batch, steps, parts, width) mean token inside each box; zero for an empty box.

    ``tokens`` is (batch, steps, rows, columns, width).
    """
    total = torch.einsum("btahw,bthwd->btad", members, tokens)
    count = members.sum(dim=(-1, -2))[..., None]
    return total / count.clamp_min(1.0)


def box_sum(values: Tensor, members: Tensor) -> Tensor:
    """(batch, steps, parts) sum of a per-token value (batch, steps, rows, columns) in each box."""
    total: Tensor = torch.einsum("btahw,bthw->bta", members, values)
    return total


class StepReadout(nn.Module):
    """``ŝ_t = head([mean of the predicted tokens in each articulator's box at step t])``.

    The four box means are concatenated in the order of ``Articulator``; an empty box gives
    zeros in its slot.
    """

    def __init__(self, width: int, parts: int, target_dim: int) -> None:
        super().__init__()
        self.parts = parts
        self.head = nn.Linear(parts * width, target_dim)

    @property
    def target_dim(self) -> int:
        return self.head.out_features

    def forward(self, tokens: Tensor, members: Tensor) -> Tensor:
        """``tokens`` (batch, steps, rows, columns, width), ``members`` (batch, steps, parts,
        rows, columns) to (batch, steps, target_dim)."""
        if members.shape[2] != self.parts:
            raise ValueError(f"expected {self.parts} boxes per step, got {members.shape[2]}")
        latent: Tensor = self.head(pool(tokens, members).flatten(-2))
        return latent
