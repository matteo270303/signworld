"""SIGReg: distance of a batch of embeddings from an isotropic standard Gaussian (§4.5.6).

For random unit directions ``v``, the projections ``<v, x>`` of an isotropic N(0, I) sample
are N(0, 1). The Epps-Pulley statistic compares the empirical characteristic function of each
projection with that of N(0, 1), ``exp(-t^2 / 2)``, weighted by ``w(t) = exp(-t^2 / 2)``; SIGReg
averages it over the directions (Balestriero and LeCun, 2025).
"""

from typing import Final

import torch
from torch import Tensor

from .geometry import require_matrix

_MIN_KNOTS: Final = 2


def random_directions(
    dimension: int,
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """``count`` directions drawn uniformly on the unit sphere, one per row."""
    directions = torch.randn(count, dimension, generator=generator, device=device)
    unit: Tensor = directions / directions.norm(dim=1, keepdim=True)
    return unit


class SIGReg:
    """Sliced Epps-Pulley test against N(0, I).

    The integrand is even in ``t``, so the integral over the real line is twice the integral
    over ``[0, t_max]``, computed with the trapezoidal rule; the weight makes the tail beyond
    ``t_max = 3`` negligible. The characteristic function is split into its cosine and sine
    means, which keeps the computation real-valued and stable in low precision. It runs in
    float32 regardless of the input type, as the triage table of §4.13.6 prescribes.
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0) -> None:
        if knots < _MIN_KNOTS:
            raise ValueError("the quadrature needs at least two knots")
        self._t = torch.linspace(0.0, t_max, knots)
        self._target = torch.exp(-0.5 * self._t.pow(2))

    def __call__(self, embeddings: Tensor, directions: Tensor) -> Tensor:
        """Statistic averaged over ``directions``; ``embeddings`` is (samples, dimensions)."""
        require_matrix("embeddings", embeddings)
        require_matrix("directions", directions)
        if embeddings.shape[1] != directions.shape[1]:
            raise ValueError(
                f"embedding dimension {embeddings.shape[1]} differs from the directions' "
                f"{directions.shape[1]}"
            )
        t = self._t.to(embeddings.device)
        target = self._target.to(embeddings.device)
        projections = embeddings.float() @ directions.float().T
        phase = projections[..., None] * t
        real = phase.cos().mean(dim=0)
        imaginary = phase.sin().mean(dim=0)
        integrand = ((real - target).pow(2) + imaginary.pow(2)) * target
        per_direction = 2.0 * torch.trapezoid(integrand, t, dim=-1)
        return per_direction.mean()
