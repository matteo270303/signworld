"""Retrieval measures of a set of clips: the metric that decides, and the result diagnostics.

Queries are the captions ẽ and the gallery the clips ŷ (text to video), and the other way
round (video to text); clips with the same caption count as matches. With K hypotheses (ESP-4)
a clip scores its best hypothesis, as ``F_sem = min_k E_k`` asks. The metric that decides
early stopping is the mean of R@1 text-to-video and video-to-text (§4.10).

Both directions report:

* **R@k** (k = 1, 5, 10): the share of queries with a match among the first k, the recall of
  cross-modal retrieval papers, with a bootstrap interval on the queries where asked;
* **Precision@k**: matches among the first k, over k; **Recall@k** in the information-retrieval
  sense: matches among the first k, over all the query's matches (with one match per query
  it equals R@k);
* **MRR** (mean reciprocal rank of the first match) and **MedR** (1 = first);
* the R@1 **tolerant to near-duplicate captions** (§4.13.4).

Text-to-video R@1 is also split by caption language, clip duration and caption length.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional

from signworld.metrics.retrieval import bootstrap_recall, grouped_relevance, match_ranks

KS = (1, 5, 10)
EVERY_INTERVAL = tuple(f"{d}_r{k}" for d in ("t2v", "v2t") for k in KS)
DURATION_BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, math.inf))
"""Seconds; each band holds its lower bound."""
WORD_BANDS = ((1, 6), (6, 11), (11, 21), (21, math.inf))
"""Caption words: 1-5, 6-10, 11-20, over 20."""


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    t2v: dict[int, float]
    v2t: dict[int, float]
    clips: int
    chance: float = 0.0
    """Expected R@1 of a random ranking: the mean share of the gallery that matches a query."""
    measures: dict[str, float] = field(default_factory=dict)
    """Every measure, keyed as it is logged."""

    @property
    def decision(self) -> float:
        """Mean of R@1 text-to-video and video-to-text."""
        return (self.t2v[1] + self.v2t[1]) / 2

    def as_log(self) -> dict[str, float]:
        extra = {"decision": self.decision, "clips": float(self.clips), "chance": self.chance}
        return self.measures | extra


def similarity(predicted: Tensor, text: Tensor) -> Tensor:
    """(texts, clips) cosine of each caption with each clip's best hypothesis."""
    unit_text = functional.normalize(text.float(), dim=-1)
    unit_video = functional.normalize(predicted.float(), dim=-1)
    return torch.einsum("td,vkd->tvk", unit_text, unit_video).amax(dim=-1)


def ranked(scores: Tensor, relevance: Tensor, prefix: str) -> dict[str, float]:
    """R@k, Precision@k, Recall@k, MRR and MedR of one direction."""
    ranks = match_ranks(scores, relevance).float() + 1  # 1 = the first result matches
    out = {f"{prefix}_r{k}": float((ranks <= k).float().mean()) for k in KS}
    top = scores.topk(min(max(KS), scores.shape[1]), dim=1).indices
    hits = relevance.gather(1, top).float()
    matches = relevance.sum(dim=1).float()
    for k in KS:
        found = hits[:, :k].sum(dim=1)
        out[f"{prefix}_precision{k}"] = float((found / k).mean())
        out[f"{prefix}_recall{k}"] = float((found / matches).mean())
    out[f"{prefix}_mrr"] = float((1 / ranks).mean())
    out[f"{prefix}_medr"] = float(ranks.median())
    return out


def _bands(
    values: Tensor, bands: Sequence[tuple[float, float]], hits: Tensor, name: str
) -> dict[str, float]:
    out = {}
    for low, high in bands:
        inside = (values >= low) & (values < high)
        label = f"{low:g}-{high:g}" if math.isfinite(high) else f"{low:g}+"
        if int(inside.sum()) > 0:
            out[f"t2v_r1_{name}_{label}"] = float(hits[inside].float().mean())
            out[f"clips_{name}_{label}"] = float(inside.sum())
    return out


def retrieval(  # noqa: PLR0913 (the embeddings and what splits the queries)
    predicted: Tensor,
    texts: Tensor,
    rows: Tensor,
    *,
    captions: Tensor | None = None,
    duplicate_cosine: float | None = None,
    languages: Tensor | None = None,
    names: Sequence[str] = (),
    durations: Tensor | None = None,
    words: Tensor | None = None,
    bootstrap: Sequence[str] = (),
) -> RetrievalScores:
    """Every retrieval measure of the clips; ``rows`` are their caption rows.

    ``bootstrap`` names the R@k that get a 95 % interval, as ``t2v_r1`` or ``v2t_r5``.
    """
    scores = similarity(predicted, texts)
    relevance = grouped_relevance(rows.tolist(), rows.tolist())
    measures = ranked(scores, relevance, "t2v") | ranked(scores.T, relevance.T, "v2t")
    generator = torch.Generator().manual_seed(0)
    for prefix, (s, r) in {"t2v": (scores, relevance), "v2t": (scores.T, relevance.T)}.items():
        for k in KS:
            if f"{prefix}_r{k}" in bootstrap:
                interval = bootstrap_recall(s, r, k, generator=generator)
                measures[f"{prefix}_r{k}_low"] = interval.low
                measures[f"{prefix}_r{k}_high"] = interval.high
    if captions is not None and duplicate_cosine is not None:
        unit = functional.normalize(captions.float(), dim=-1)
        tolerant = relevance | ((unit @ unit.T) >= duplicate_cosine)
        measures["tolerant_t2v_r1"] = ranked(scores, tolerant, "t")["t_r1"]
        measures["tolerant_v2t_r1"] = ranked(scores.T, tolerant.T, "v")["v_r1"]
    hits = match_ranks(scores, relevance) == 0
    if languages is not None:
        for index, name in enumerate(names):
            queries = languages == index
            if int(queries.sum()) > 0:
                measures[f"t2v_r1_{name}"] = float(hits[queries].float().mean())
    if durations is not None:
        measures |= _bands(durations.float(), DURATION_BANDS, hits, "duration")
    if words is not None:
        measures |= _bands(words.float(), WORD_BANDS, hits, "words")
    return RetrievalScores(
        t2v={k: measures[f"t2v_r{k}"] for k in KS},
        v2t={k: measures[f"v2t_r{k}"] for k in KS},
        clips=len(rows),
        chance=float(relevance.float().mean()),
        measures=measures,
    )
