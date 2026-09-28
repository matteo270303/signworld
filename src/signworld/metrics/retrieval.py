"""Retrieval metrics computed from a query-by-gallery similarity matrix (§4.12.3, §4.13.4)."""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a two-sided bootstrap confidence interval."""

    estimate: float
    low: float
    high: float


def paired_relevance(size: int, device: torch.device | None = None) -> Tensor:
    """Relevance of the standard protocol: query ``i`` matches gallery item ``i`` only."""
    return torch.eye(size, dtype=torch.bool, device=device)


def grouped_relevance(
    query_groups: Sequence[Hashable], gallery_groups: Sequence[Hashable]
) -> Tensor:
    """Relevance where any gallery item of the query's group counts as a match.

    With groups of identical or near-identical captions this gives the duplicate-tolerant
    protocol, which does not punish retrieving an equivalent item.
    """
    codes: dict[Hashable, int] = {}
    queries = torch.tensor([codes.setdefault(group, len(codes)) for group in query_groups])
    gallery = torch.tensor([codes.setdefault(group, len(codes)) for group in gallery_groups])
    return queries[:, None] == gallery[None, :]


def match_ranks(similarity: Tensor, relevance: Tensor) -> Tensor:
    """Zero-based rank of the best relevant gallery item for every query.

    Ties count against the query: an irrelevant item scoring as high as the best relevant one
    ranks before it. Constant embeddings therefore score as badly as they deserve instead of
    reaching a perfect recall through ties.
    """
    if similarity.shape != relevance.shape:
        raise ValueError(f"similarity {similarity.shape} and relevance {relevance.shape} differ")
    if not bool(relevance.any(dim=1).all()):
        raise ValueError("every query needs at least one relevant gallery item")
    best_relevant = similarity.masked_fill(~relevance, float("-inf")).amax(dim=1, keepdim=True)
    outranking = (similarity >= best_relevant) & ~relevance
    return outranking.sum(dim=1)


def recall_at_k(similarity: Tensor, relevance: Tensor, ks: Iterable[int]) -> dict[int, float]:
    """Fraction of queries whose best relevant item is among the top ``k``, for every ``k``."""
    ranks = match_ranks(similarity, relevance)
    return {k: (ranks < k).float().mean().item() for k in ks}


def bootstrap_recall(
    similarity: Tensor,
    relevance: Tensor,
    k: int,
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    generator: torch.Generator | None = None,
) -> Interval:
    """Recall@k with a percentile interval from resampling the queries."""
    hits = (match_ranks(similarity, relevance) < k).float()
    indices = torch.randint(
        len(hits), (resamples, len(hits)), generator=generator, device=hits.device
    )
    means = hits[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    low, high = torch.quantile(means, torch.tensor([tail, 1.0 - tail], device=hits.device))
    return Interval(hits.mean().item(), low.item(), high.item())


def hubness(similarity: Tensor, k: int) -> float:
    """Skewness of how often each gallery item appears in the queries' top ``k``.

    Values that grow during training mean a few items attract most queries, the signature of
    predictions pulled towards the centre of the gallery.
    """
    top = similarity.topk(k, dim=1).indices.flatten()
    occurrences = torch.bincount(top, minlength=similarity.shape[1]).float()
    deviation = occurrences - occurrences.mean()
    spread = deviation.pow(2).mean().sqrt()
    if spread == 0:
        return 0.0
    return (deviation.pow(3).mean() / spread.pow(3)).item()
