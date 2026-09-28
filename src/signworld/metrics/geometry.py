"""Geometry of a set of embeddings: rank, isotropy and collapse (§4.12.1 PC1, §4.13.3)."""

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

MATRIX_NDIM: Final = 2


def require_matrix(name: str, tensor: Tensor) -> None:
    if tensor.ndim != MATRIX_NDIM:
        raise ValueError(
            f"{name} must be a (rows, columns) matrix, got shape {tuple(tensor.shape)}"
        )


def _as_matrix(embeddings: Tensor) -> Tensor:
    require_matrix("embeddings", embeddings)
    return embeddings.double()


def centered(embeddings: Tensor) -> Tensor:
    return embeddings - embeddings.mean(dim=0, keepdim=True)


def effective_rank(embeddings: Tensor) -> float:
    """Entropy-based effective rank of Roy and Vetterli (2007) over the singular values.

    It equals the dimension for isotropic data and 1 when all points lie on a line.
    """
    singular = torch.linalg.svdvals(_as_matrix(embeddings))
    weights = singular / singular.sum()
    weights = weights[weights > 0]
    return math.exp(-(weights * weights.log()).sum().item())


def explained_variance(embeddings: Tensor) -> Tensor:
    """Fractions of total variance along the principal axes, in decreasing order."""
    singular = torch.linalg.svdvals(centered(_as_matrix(embeddings)))
    variance = singular.pow(2)
    fractions: Tensor = variance / variance.sum()
    return fractions


def isoscore(embeddings: Tensor) -> float:
    """IsoScore of Rudman et al. (2022): 1 for isotropic data, 0 when one direction holds all.

    Unlike the mean cosine it is invariant to rotations and to the mean of the data.
    """
    matrix = centered(_as_matrix(embeddings))
    dimension = matrix.shape[1]
    variances = torch.linalg.svdvals(matrix).pow(2) / (matrix.shape[0] - 1)
    variances = torch.cat([variances, variances.new_zeros(dimension - len(variances))])
    normalized = math.sqrt(dimension) * variances / variances.norm()
    distance = (normalized - 1.0).norm() / math.sqrt(2.0 * (dimension - math.sqrt(dimension)))
    fraction = (dimension - distance.pow(2) * (dimension - math.sqrt(dimension))).pow(2)
    fraction = fraction / dimension**2
    return float((dimension * fraction - 1.0) / (dimension - 1.0))


def mean_random_pair_cosine(
    embeddings: Tensor, pairs: int = 100_000, generator: torch.Generator | None = None
) -> float:
    """Mean cosine similarity between random pairs of distinct samples."""
    matrix = _as_matrix(embeddings)
    count = matrix.shape[0]
    first = torch.randint(count, (pairs,), generator=generator, device=matrix.device)
    offset = torch.randint(1, count, (pairs,), generator=generator, device=matrix.device)
    second = (first + offset) % count
    cosine = torch.nn.functional.cosine_similarity(matrix[first], matrix[second], dim=1)
    return cosine.mean().item()


def mean_dimension_std(embeddings: Tensor) -> float:
    return _as_matrix(embeddings).std(dim=0).mean().item()


@dataclass(frozen=True, slots=True)
class CollapseReading:
    effective_rank: float
    mean_dimension_std: float

    @classmethod
    def of(cls, embeddings: Tensor) -> "CollapseReading":
        return cls(effective_rank(centered(embeddings)), mean_dimension_std(embeddings))


@dataclass(frozen=True, slots=True)
class CollapseAlarm:
    """Collapse alarm of §4.13.3: rank or spread below a fraction of the step-0 reference."""

    reference: CollapseReading
    min_ratio: float = 0.5

    def triggered_by(self, reading: CollapseReading) -> list[str]:
        """Names of the measurements that fell below ``min_ratio`` times the reference."""
        ratios = {
            "effective_rank": reading.effective_rank / self.reference.effective_rank,
            "mean_dimension_std": reading.mean_dimension_std / self.reference.mean_dimension_std,
        }
        return [name for name, ratio in ratios.items() if ratio < self.min_ratio]
