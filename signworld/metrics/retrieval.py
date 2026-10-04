"""Retrieval metrics computed from a query-by-gallery similarity matrix (§4.12.3, §4.13.4).

A (texts, clips) matrix holds both directions: ``both_ways`` gives text to video as it is and
video to text as its transpose, and ``bidirectional_measures`` reports R@k, Precision@k,
Recall@k, MRR and MedR of each, keyed ``t2v_*`` and ``v2t_*``.
"""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from .directions import DIRECTIONS

KS: Final = (1, 5, 10)
"""The k of R@k, Precision@k and Recall@k in every report."""


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


def both_ways(*matrices: Tensor) -> dict[str, tuple[Tensor, ...]]:
    """(texts, clips) matrices in each direction: as given for text to video, transposed for
    video to text, so that rows are always the queries."""
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"the matrices of one retrieval differ in shape: {sorted(shapes)}")
    oriented = {"t2v": matrices, "v2t": tuple(matrix.T for matrix in matrices)}
    return {direction: oriented[direction] for direction in DIRECTIONS}


def ranking_measures(
    similarity: Tensor, relevance: Tensor, prefix: str, ks: Sequence[int] = KS
) -> dict[str, float]:
    """R@k, Precision@k, Recall@k, MRR and MedR of one direction, keyed ``{prefix}_*``.

    R@k is the share of queries with a match among the first k, the recall of cross-modal
    retrieval papers; Precision@k the matches among the first k over k; Recall@k, in the
    information-retrieval sense, the matches among the first k over all the query's matches
    (with one match per query it equals R@k); MRR the mean reciprocal rank of the first match
    and MedR its median rank (1 = first; with an even number of queries the mean of the two
    middle ranks, as ``np.median``, where ``Tensor.median`` would take the lower one).
    """
    ranks = match_ranks(similarity, relevance).float() + 1  # 1 = the first result matches
    out = {f"{prefix}_r{k}": float((ranks <= k).float().mean()) for k in ks}
    top = similarity.topk(min(max(ks), similarity.shape[1]), dim=1).indices
    hits = relevance.gather(1, top).float()
    matches = relevance.sum(dim=1).float()
    for k in ks:
        found = hits[:, :k].sum(dim=1)
        out[f"{prefix}_precision{k}"] = float((found / k).mean())
        out[f"{prefix}_recall{k}"] = float((found / matches).mean())
    out[f"{prefix}_mrr"] = float((1 / ranks).mean())
    out[f"{prefix}_medr"] = float(ranks.quantile(0.5))
    return out


def bidirectional_measures(
    similarity: Tensor, relevance: Tensor, ks: Sequence[int] = KS
) -> dict[str, float]:
    """``ranking_measures`` of text to video and video to text on a (texts, clips) matrix."""
    out: dict[str, float] = {}
    for direction, (scores, relevant) in both_ways(similarity, relevance).items():
        out |= ranking_measures(scores, relevant, direction, ks)
    return out


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
