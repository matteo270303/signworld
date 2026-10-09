"""How a set of embeddings meets each regulariser of the semantic level (A, V, B₀, C).

Every arm is read on what every regulariser asks for, the ones it does not optimise included,
so the arms can be told apart by the shape of their distributions, not by retrieval alone:

* **the sphere** (B₀'s L_unif): Wang and Isola's L_unif against its value for points uniform
  on the unit sphere, ``-2t + log ₀F₁(; m/2; t²)`` (their Prop. 1 bound, reached only by a
  uniform encoder), and the cosines of random pairs against that distribution: mean 0,
  variance 1/m;
* **the second-order moments** (V's VICReg): its variance and covariance terms;
* **the whole distribution** (A's SIGReg): the KL divergence to the closest isotropic Gaussian,
  split exactly into its second-order part (closed form) and the non-Gaussianity, read on
  random one-dimensional slices (``divergence``). SIGReg itself is in the geometry readings
  (``measures._geometry``).

A finite sample is never exactly isotropic: every reading with a ``_floor`` comes with its
value on a standard Gaussian sample of the same size, the level to read it against.

A moment constraint can whiten the covariance and leave the non-Gaussianity; a distribution
constraint drives both; the pairwise uniformity acts on the directions only. These readings
make that difference visible on held-out clips.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional

from signworld.loss.sigreg import random_directions
from signworld.loss.worldsign import uniformity, vicreg_covariance, vicreg_variance

from .divergence import gaussian_divergence, sliced_negentropy


def hypergeometric_0f1(b: float, z: float, tolerance: float = 1e-16) -> float:
    """₀F₁(; b; z) = Σ_k z^k / ((b)_k k!), by its series (it converges for every z)."""
    total, term, k = 1.0, 1.0, 0
    while True:
        term *= z / ((b + k) * (k + 1))
        total += term
        k += 1
        if abs(term) < tolerance * abs(total) or k > 10_000:  # noqa: PLR2004 (a safety cap)
            return total


def uniform_sphere_uniformity(dimension: int, t: float) -> float:
    """L_unif of points uniform on the unit sphere of R^dimension (Wang and Isola)."""
    return -2 * t + math.log(hypergeometric_0f1(dimension / 2, t * t))


def pair_cosines(rows: Tensor, pairs: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Mean and ``m``·variance of the cosine of random distinct pairs (0 and 1 if uniform)."""
    unit = functional.normalize(rows.float(), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(len(unit), (pairs,), generator=generator)
    second = (first + torch.randint(1, len(unit), (pairs,), generator=generator)) % len(unit)
    cosine = (unit[first.to(unit.device)] * unit[second.to(unit.device)]).sum(-1)
    return float(cosine.mean()), float(cosine.var() * unit.shape[1])


def distribution_measures(  # noqa: PLR0913 (the rows and the settings of every reading)
    name: str,
    rows: Tensor,
    *,
    t: float,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
    slices: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """Every reading of the module on ``rows`` (n, m), keyed ``{name}_*``."""
    x = rows.float().cpu()
    dimension = x.shape[1]
    reference = uniform_sphere_uniformity(dimension, t)
    unif = float(uniformity(x, t))
    mean, spread = pair_cosines(x, seed=seed)
    divergence = gaussian_divergence(x)
    generator = torch.Generator().manual_seed(seed)
    directions = random_directions(dimension, slices, generator=generator)
    floor_sample = torch.randn(x.shape, generator=generator)
    negentropy = sliced_negentropy(x, directions)
    floor = sliced_negentropy(floor_sample, directions)
    floor_divergence = gaussian_divergence(floor_sample)
    return {
        f"{name}_uniformity": unif,
        f"{name}_uniformity_sphere": reference,
        f"{name}_uniformity_gap": unif - reference,
        f"{name}_pair_cosine_mean": mean,
        f"{name}_pair_cosine_spread": spread,
        f"{name}_vicreg_variance": float(vicreg_variance(x, gamma, epsilon)),
        f"{name}_vicreg_covariance": float(vicreg_covariance(x)),
        f"{name}_vicreg_covariance_floor": float(vicreg_covariance(floor_sample)),
        f"{name}_kl_standard": divergence.to_standard,
        f"{name}_kl_shape": divergence.to_isotropic,
        f"{name}_kl_shape_floor": floor_divergence.to_isotropic,
        f"{name}_negentropy": negentropy,
        f"{name}_negentropy_floor": floor,
        f"{name}_negentropy_excess": negentropy - floor,
    }
