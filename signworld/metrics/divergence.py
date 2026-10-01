"""How far a set of embeddings is from an isotropic Gaussian, in nats of KL divergence.

For any distribution p with mean μ and covariance Σ, and a Gaussian q, the KL divergence splits
exactly (log q is quadratic, so it only sees p's first two moments):

    KL(p ‖ q)  =  KL(p ‖ N(μ, Σ))  +  KL(N(μ, Σ) ‖ q)
                  non-Gaussianity     second-order mismatch (closed form)

The second term has a closed form: towards N(0, I) it also counts the mean and the scale;
towards the closest isotropic Gaussian N(μ, s² I), s² = tr Σ / d, only the anisotropy. The
first term, the negentropy, cannot be estimated reliably in hundreds of dimensions; it is read
on random one-dimensional projections (the slices SIGReg also uses), each with Vasicek's
m-spacing entropy estimator, which has a small positive bias at finite size: a true Gaussian
sample of the same size gives the floor to read the values against.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry import covariance_spectrum


@dataclass(frozen=True, slots=True)
class GaussianDivergence:
    to_standard: float
    """KL(N(μ, Σ) ‖ N(0, I)): mean, scale and shape."""
    to_isotropic: float
    """KL(N(μ, Σ) ‖ N(μ, s² I)) with s² = tr Σ / d: shape only, 0 when isotropic."""
    floored: bool
    """True when eigenvalues fell below the floor: the values are then lower bounds."""


def gaussian_divergence(rows: Tensor, floor: float = 1e-12) -> GaussianDivergence:
    """Closed-form KLs of the moment-matched Gaussian of ``rows`` (samples, dimension)."""
    data = rows.double()
    variances = covariance_spectrum(data)
    minimum = floor * variances[0]
    floored = bool((variances < minimum).any())
    variances = variances.clamp_min(minimum)
    dimension = len(variances)
    log_det = float(variances.log().sum())
    trace = float(variances.sum())
    squared_mean = float(data.mean(dim=0).pow(2).sum())
    to_standard = 0.5 * (trace + squared_mean - dimension - log_det)
    to_isotropic = 0.5 * (dimension * math.log(trace / dimension) - log_det)
    return GaussianDivergence(to_standard, to_isotropic, floored)


def spacing_entropy(samples: Tensor) -> Tensor:
    """Vasicek's m-spacing differential entropy of each column of (samples, columns), in nats."""
    ordered = samples.double().sort(dim=0).values
    count = ordered.shape[0]
    spacing = max(1, round(math.sqrt(count)))
    index = torch.arange(count, device=samples.device)
    upper = ordered[(index + spacing).clamp(max=count - 1)]
    lower = ordered[(index - spacing).clamp(min=0)]
    gaps = (upper - lower).clamp_min(1e-300)
    return torch.log(count / (2 * spacing) * gaps).mean(dim=0)


def sliced_negentropy(rows: Tensor, directions: Tensor) -> float:
    """Mean over ``directions`` of KL(p_v ‖ N(m_v, s_v²)), nats per projection (0 if Gaussian)."""
    projections = rows.double() @ directions.double().T
    variance = projections.var(dim=0).clamp_min(1e-300)
    gaussian = 0.5 * torch.log(2 * math.pi * math.e * variance)
    return float((gaussian - spacing_entropy(projections)).mean())
