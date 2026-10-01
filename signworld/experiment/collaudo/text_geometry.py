"""PC1 (§4.12.1): geometry of the caption targets at every MRL truncation.

For each dimension the embeddings are truncated and re-normalised (Matryoshka truncation),
then centred per spoken language (§4.4.4). ``d_MRL`` is the smallest dimension at which each
caption keeps, on average, 90 % of its nearest captions at the full dimension: retrieval ranks
by neighbours, so that is what truncation must not disturb.

An earlier rule compared effective ranks across dimensions. It could only ever choose the full
dimension, since the effective rank at dimension d cannot exceed d (PC1, September 2026).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import PositiveInt

from signworld.data.acquisition.config import FrozenModel
from signworld.loss.sigreg import SIGReg, random_directions
from signworld.metrics.geometry import (
    effective_rank,
    explained_variance,
    isoscore,
    mean_random_pair_cosine,
)


class TextGeometrySettings(FrozenModel):
    dimensions: tuple[PositiveInt, ...] = (768, 512, 256, 128)
    retained_neighbours: float = 0.9
    """[Our rule] Share of the full-dimension nearest neighbours the chosen dimension keeps."""
    neighbours: PositiveInt = 10
    neighbour_queries: PositiveInt = 2000
    sigreg_samples: PositiveInt = 16_384
    sigreg_directions: PositiveInt = 256
    cosine_pairs: PositiveInt = 100_000
    spectrum_head: PositiveInt = 10
    seed: int = 0


@dataclass(frozen=True, slots=True)
class GeometryAtDimension:
    dimension: int
    effective_rank: float
    isoscore: float
    mean_random_pair_cosine: float
    mean_random_pair_cosine_before_centering: float
    sigreg: float
    """SIGReg after rescaling to unit variance per direction on average.

    Unit-norm embeddings have per-direction variance 1/d, which SIGReg would flag whatever
    their shape; the rescaling isolates anisotropy and non-Gaussianity from that scale.
    """
    sigreg_gaussian_reference: float
    """SIGReg of a standard Gaussian sample of the same size and dimension, same directions.

    It is the value a perfectly isotropic Gaussian target would reach; the finite-sample bias
    keeps it above zero, so the targets are read against it, not against zero.
    """
    language_variance_fraction: float
    """Share of the variance explained by the language means, before centring."""
    explained_variance_head: list[float]
    neighbour_recall: float = 1.0
    """Of each query's nearest captions at the full dimension, the share still nearest here."""


@dataclass(frozen=True, slots=True)
class TextGeometryReport:
    captions: int
    languages: dict[str, int]
    per_dimension: list[GeometryAtDimension]
    chosen_dimension: int


def truncate(embeddings: torch.Tensor, dimension: int) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings[:, :dimension], dim=1)


def center_by_language(embeddings: torch.Tensor, languages: Sequence[str]) -> torch.Tensor:
    labels = np.asarray(languages)
    centred = embeddings.clone()
    for language in np.unique(labels):
        rows = torch.from_numpy(np.flatnonzero(labels == language))
        centred[rows] -= embeddings[rows].mean(dim=0)
    return centred


def text_geometry(
    embeddings: np.ndarray, languages: Sequence[str], settings: TextGeometrySettings
) -> TextGeometryReport:
    full = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    if full.shape[1] < max(settings.dimensions):
        raise ValueError(f"embeddings have {full.shape[1]} dimensions, fewer than requested")
    generator = torch.Generator().manual_seed(settings.seed)
    sigreg = SIGReg()
    sample = torch.randperm(len(full), generator=generator)[: settings.sigreg_samples]

    queries = torch.randperm(len(full), generator=generator)[: settings.neighbour_queries]
    reference: torch.Tensor | None = None

    results = []
    for dimension in sorted(settings.dimensions, reverse=True):
        truncated = truncate(full, dimension)
        centred = center_by_language(truncated, languages)
        found = nearest_neighbours(centred, queries, settings.neighbours)
        if reference is None:
            reference = found
        scaled = centred[sample] * (dimension / centred[sample].pow(2).sum(dim=1).mean()).sqrt()
        directions = random_directions(dimension, settings.sigreg_directions, generator=generator)
        gaussian = torch.randn(len(sample), dimension, generator=generator)
        results.append(
            GeometryAtDimension(
                dimension=dimension,
                effective_rank=effective_rank(centred),
                isoscore=isoscore(centred),
                mean_random_pair_cosine=mean_random_pair_cosine(
                    centred, settings.cosine_pairs, generator
                ),
                mean_random_pair_cosine_before_centering=mean_random_pair_cosine(
                    truncated, settings.cosine_pairs, generator
                ),
                sigreg=sigreg(scaled, directions).item(),
                sigreg_gaussian_reference=sigreg(gaussian, directions).item(),
                language_variance_fraction=language_variance_fraction(truncated, languages),
                explained_variance_head=explained_variance(centred)[
                    : settings.spectrum_head
                ].tolist(),
                neighbour_recall=neighbour_recall(found, reference),
            )
        )

    return TextGeometryReport(
        captions=len(full),
        languages=_counts(languages),
        per_dimension=results,
        chosen_dimension=choose_dimension(results, settings.retained_neighbours),
    )


def language_variance_fraction(embeddings: torch.Tensor, languages: Sequence[str]) -> float:
    """Between-language share of the total variance: how much the language alone explains."""
    total = (embeddings - embeddings.mean(dim=0)).pow(2).sum().item()
    if total == 0:
        return 0.0
    between = (
        (embeddings - center_by_language(embeddings, languages))
        .sub(embeddings.mean(dim=0))
        .pow(2)
        .sum()
        .item()
    )
    return between / total


def _counts(languages: Sequence[str]) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(languages), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def nearest_neighbours(
    embeddings: torch.Tensor, queries: torch.Tensor, count: int, chunk: int = 512
) -> torch.Tensor:
    """(queries, count) indices of each query's most cosine-similar rows, itself excluded."""
    unit = torch.nn.functional.normalize(embeddings, dim=1)
    found = []
    for start in range(0, len(queries), chunk):
        rows = queries[start : start + chunk]
        similarity = unit[rows] @ unit.T
        similarity[torch.arange(len(rows)), rows] = -torch.inf
        found.append(similarity.topk(count, dim=1).indices)
    return torch.cat(found)


def neighbour_recall(found: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean share of each row of ``reference`` that also appears in the same row of ``found``."""
    shared = (found[:, :, None] == reference[:, None, :]).any(dim=2).sum(dim=1)
    return float(shared.float().mean() / reference.shape[1])


def choose_dimension(results: Sequence[GeometryAtDimension], retained_neighbours: float) -> int:
    """Smallest dimension keeping at least ``retained_neighbours`` of the full neighbours."""
    return min(r.dimension for r in results if r.neighbour_recall >= retained_neighbours)
