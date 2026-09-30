"""3D rotary position embedding for video tokens (§4.4.6).

The head dimension is split in three even parts, rotated by the token's step, row and column.
Tokens without a grid position (the semantic predictor's queries) are left unrotated, so they
attend to the video by content while the video tokens keep their order in time: without any
position, attention and pooling would be blind to it.
"""

import torch
from torch import Tensor

from .masking import TokenGrid


def _angles(positions: Tensor, width: int, base: float) -> Tensor:
    """(n, width / 2) rotation angles for one axis."""
    frequencies = base ** (-torch.arange(0, width, 2, device=positions.device) / width)
    angles: Tensor = positions[:, None].float() * frequencies[None]
    return angles


class RoPE3D:
    def __init__(self, head_dim: int, grid: TokenGrid, base: float = 10_000.0) -> None:
        third = 2 * ((head_dim // 3) // 2)
        self.widths = (third, third, head_dim - 2 * third)
        if any(width % 2 for width in self.widths):
            raise ValueError(f"head dimension {head_dim} does not split into even thirds")
        self.grid = grid
        self.base = base

    def rotation(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """(N, head_dim / 2) cosines and sines for every token of the grid, in grid order."""
        positions = self.grid.positions(torch.arange(self.grid.size, device=device))
        angles = torch.cat(
            [
                _angles(positions[:, axis], width, self.base)
                for axis, width in enumerate(self.widths)
            ],
            dim=1,
        )
        return angles.cos(), angles.sin()

    @staticmethod
    def apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate the first ``len(cos)`` tokens of ``x`` (batch, heads, tokens, head_dim)."""
        rotated, rest = x[..., : len(cos), :], x[..., len(cos) :, :]
        even, odd = rotated[..., 0::2], rotated[..., 1::2]
        turned = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
        return torch.cat([turned, rest], dim=-2)
