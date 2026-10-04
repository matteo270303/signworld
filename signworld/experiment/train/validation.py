"""Retrieval measures of a set of clips: the metric that decides, and the result diagnostics.

Every measure is computed in both directions: the captions ẽ query the clips ŷ (text to video)
and the clips query the captions (video to text); clips with the same caption count as matches.
With K hypotheses (ESP-4) a clip scores its best hypothesis, as ``F_sem = min_k E_k`` asks. The
metric that decides early stopping is the mean of R@1 text-to-video and video-to-text (§4.10).

Both directions report:

* **R@k** (k = 1, 5, 10), **Precision@k**, **Recall@k**, **MRR** and **MedR**
  (``metrics.retrieval.ranking_measures``), with a bootstrap interval on the queries where
  asked;
* the R@1 **tolerant to near-duplicate captions** (§4.13.4);
* R@1 split by caption language, clip duration and caption length: the query of either
  direction is one clip's caption or video, so both splits use the clip's attributes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional

from signworld.metrics.directions import DIRECTIONS, Bidirectional
from signworld.metrics.retrieval import (
    KS,
    bidirectional_measures,
    bootstrap_recall,
    both_ways,
    grouped_relevance,
    match_ranks,
    recall_at_k,
)

EVERY_INTERVAL = tuple(f"{d}_r{k}" for d in DIRECTIONS for k in KS)
R1_INTERVALS = tuple(f"{d}_r1" for d in DIRECTIONS)
"""The bootstrap intervals of validation: R@1 in both directions."""
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
    def r1(self) -> Bidirectional:
        return Bidirectional(self.t2v[1], self.v2t[1])

    @property
    def decision(self) -> float:
        """Mean of R@1 text-to-video and video-to-text."""
        return self.r1.mean

    def as_log(self) -> dict[str, float]:
        extra = {"decision": self.decision, "clips": float(self.clips), "chance": self.chance}
        return self.measures | extra


def similarity(predicted: Tensor, text: Tensor) -> Tensor:
    """(texts, clips) cosine of each caption with each clip's best hypothesis."""
    unit_text = functional.normalize(text.float(), dim=-1)
    unit_video = functional.normalize(predicted.float(), dim=-1)
    return torch.einsum("td,vkd->tvk", unit_text, unit_video).amax(dim=-1)


def _bands(values: Tensor, bands: Sequence[tuple[float, float]], name: str) -> dict[str, Tensor]:
    """The clips inside each band of ``values``, keyed ``{name}_{low}-{high}``."""
    groups = {}
    for low, high in bands:
        label = f"{low:g}-{high:g}" if math.isfinite(high) else f"{low:g}+"
        groups[f"{name}_{label}"] = (values >= low) & (values < high)
    return groups


def _groups(
    languages: Tensor | None,
    names: Sequence[str],
    durations: Tensor | None,
    words: Tensor | None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """The clips of each caption language, and of each band of duration and caption length;
    groups without clips are left out."""
    by_language = (
        {name: languages == index for index, name in enumerate(names)}
        if languages is not None
        else {}
    )
    by_band: dict[str, Tensor] = {}
    if durations is not None:
        by_band |= _bands(durations.float(), DURATION_BANDS, "duration")
    if words is not None:
        by_band |= _bands(words.float(), WORD_BANDS, "words")

    def present(groups: dict[str, Tensor]) -> dict[str, Tensor]:
        return {label: inside for label, inside in groups.items() if int(inside.sum()) > 0}

    return present(by_language), present(by_band)


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
    """Every retrieval measure of the clips, both ways; ``rows`` are their caption rows.

    ``bootstrap`` names the R@k that get a 95 % interval, as ``t2v_r1`` or ``v2t_r5``.
    """
    unknown = sorted(set(bootstrap) - set(EVERY_INTERVAL))
    if unknown:
        raise ValueError(f"no R@k named {unknown}; known: {list(EVERY_INTERVAL)}")
    scores = similarity(predicted, texts)
    relevance = grouped_relevance(rows.tolist(), rows.tolist())
    directions = both_ways(scores, relevance)
    measures = bidirectional_measures(scores, relevance)
    generator = torch.Generator().manual_seed(0)
    for direction, (s, r) in directions.items():
        for k in KS:
            if f"{direction}_r{k}" in bootstrap:
                interval = bootstrap_recall(s, r, k, generator=generator)
                measures[f"{direction}_r{k}_low"] = interval.low
                measures[f"{direction}_r{k}_high"] = interval.high
    if captions is not None and duplicate_cosine is not None:
        unit = functional.normalize(captions.float(), dim=-1)
        tolerant = relevance | ((unit @ unit.T) >= duplicate_cosine)
        for direction, (s, r) in both_ways(scores, tolerant).items():
            measures[f"tolerant_{direction}_r1"] = recall_at_k(s, r, (1,))[1]
    by_language, by_band = _groups(languages, names, durations, words)
    for direction, (s, r) in directions.items():
        hits = (match_ranks(s, r) == 0).float()
        for label, inside in (by_language | by_band).items():
            measures[f"{direction}_r1_{label}"] = float(hits[inside].mean())
    measures |= {f"clips_{label}": float(inside.sum()) for label, inside in by_band.items()}
    return RetrievalScores(
        t2v={k: measures[f"t2v_r{k}"] for k in KS},
        v2t={k: measures[f"v2t_r{k}"] for k in KS},
        clips=len(rows),
        chance=float(relevance.float().mean()),
        measures=measures,
    )
