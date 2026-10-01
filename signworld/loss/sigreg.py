"""SIGReg: distance of a batch of embeddings from an isotropic standard Gaussian (§4.5.6).

For random unit directions ``v``, the projections ``<v, x>`` of an isotropic N(0, I) sample
are N(0, 1). The Epps-Pulley statistic compares the empirical characteristic function of each
projection with that of N(0, 1), ``exp(-t^2 / 2)``, weighted by ``w(t) = exp(-t^2 / 2)``, and
multiplies by the number of samples; SIGReg averages it over the directions. This is LeJEPA's
reference implementation (Balestriero and LeCun, 2025, §4.2.3): knots ``linspace(-5, 5, 17)``,
trapezoidal rule, times N.

The factor N makes the statistic of a true Gaussian sample stay near a constant (its
expectation is ``∫ (1 - exp(-t^2)) exp(-t^2 / 2) dt ≈ 1.06``) whatever the batch size, while a
non-Gaussian sample grows with N: it is what gives LeJEPA's λ = 0.05 its meaning.
"""

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor

from signworld.metrics.geometry import require_matrix

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
    """Sliced Epps-Pulley test against N(0, I), as in LeJEPA.

    The characteristic function is split into its cosine and sine means, which keeps the
    computation real-valued; ``|φ̂(t) - exp(-t²/2)|² = (cos-mean - e)² + sin-mean²``. It runs in
    float32 regardless of the input type, as the triage table of §4.13.6 prescribes.
    """

    def __init__(self, knots: int = 17, t_max: float = 5.0) -> None:
        if knots < _MIN_KNOTS:
            raise ValueError("the quadrature needs at least two knots")
        self._t = torch.linspace(-t_max, t_max, knots)
        self._target = torch.exp(-0.5 * self._t.pow(2))

    def __call__(
        self,
        embeddings: Tensor,
        directions: Tensor,
        samples: int | None = None,
        reduce: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Statistic averaged over ``directions``; ``embeddings`` is (samples, dimensions).

        ``reduce`` sums a tensor over the GPUs of a data-parallel run: the characteristic
        function is then that of the samples of every GPU and N is their total number, as in
        LeJEPA. ``samples`` overrides N.
        """
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
        sums = torch.stack([phase.cos().sum(dim=0), phase.sin().sum(dim=0)])
        count = torch.tensor(float(embeddings.shape[0]), device=embeddings.device)
        if reduce is not None:
            sums, count = reduce(sums), reduce(count)
        real, imaginary = sums / count
        integrand = ((real - target).pow(2) + imaginary.pow(2)) * target
        factor = count if samples is None else torch.tensor(float(samples))
        per_direction = torch.trapezoid(integrand, t, dim=-1) * factor
        return per_direction.mean()
