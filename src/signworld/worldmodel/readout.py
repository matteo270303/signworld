"""Per-step, per-articulator read-out of a grid of tokens (§4.5.2).

The physical level predicts a token for every position of the grid; its target is one vector
per step and articulator. ``membership`` turns the articulator boxes (frame fractions, from the
keypoints) into which patches each box touches, and ``pool`` averages the tokens inside, in
one tensor operation for a whole batch. A box covers every patch it touches, as in the probes
of the collaudo (``models.video_encoders.box_pool``).
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


class ArticulatorReadout(nn.Module):
    """``ŝ_{t,a} = head(mean of the predicted tokens in a's box at step t)``."""

    def __init__(self, width: int, target_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(width, target_dim)

    def forward(self, tokens: Tensor, members: Tensor) -> Tensor:
        latent: Tensor = self.head(pool(tokens, members))
        return latent
