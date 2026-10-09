"""GloFND's per-caption thresholds of the InfoNCE arm, fixed once on the frozen captions.

GloFND (Balmaseda et al., ICML 2025) learns for every anchor ``i`` a threshold ``λ_i``, the
(1 - alpha)-quantile of its similarities to the dataset, and drops from the denominator of
InfoNCE the negatives above it. Its subgradient step converges to that quantile; here the
similarity is the cosine of the frozen EmbeddingGemma vectors, centred per language as the text
branch centres them, so the quantile never moves and is computed once, exactly, against a fixed
random reference set of training captions. On-line updates would need many visits per caption:
with one visit per epoch they barely move in a run of a few epochs.

The output directory holds ``thresholds.npy`` (one float per row of the embedding store; rows
of captions not in training get 2.0, which no cosine exceeds) and ``meta.json`` (alpha, the
reference set, the seed, the inputs' fingerprints and the distribution of the thresholds).
"""

import datetime
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import torch
from torch.nn import functional

from signworld.models.worldsign.text_branch import LanguageCentering

from .loaders import caption_statistics

THRESHOLDS: Final = "thresholds.npy"
META: Final = "meta.json"
UNUSED: Final = 2.0
"""Threshold of a row no training caption uses: above any cosine."""


@dataclass(frozen=True, slots=True)
class ThresholdMeta:
    alpha: float
    reference: int
    """Captions of the reference set (distinct training captions, drawn with ``seed``)."""
    seed: int
    rows: int
    """Rows of the embedding store, the length of ``thresholds.npy``."""
    anchors: int
    """Distinct training captions, the rows that got a threshold."""
    rank: int
    """The threshold is the ``rank``-th largest similarity to the reference set."""
    store: str
    """Fingerprint of the embedding store (its directory name)."""
    index_sha256: str
    quantiles: dict[str, float]
    """Distribution of the thresholds over the anchors: p1, p10, p50, p90, p99."""
    created: str


def compute_thresholds(  # noqa: PLR0913 (the inputs and the knobs of the quantile)
    embeddings: np.ndarray,
    train: pa.Table,
    alpha: float,
    *,
    reference: int = 200_000,
    seed: int = 0,
    device: str = "cpu",
    chunk: int = 4096,
) -> tuple[np.ndarray, int, int]:
    """(thresholds over every store row, the anchors, the rank of the quantile).

    ``train``: the training clips of the index (``caption_row``, ``caption_language``).
    """
    captions, languages = caption_statistics(train, embeddings)
    rows = np.asarray(sorted(set(train.column("caption_row").to_pylist())))
    centering = LanguageCentering(captions.shape[1])
    centering.fit(captions, languages)
    with torch.no_grad():
        centred = centering(captions, centering.index(languages))
    unit = functional.normalize(centred.float(), dim=-1).to(device)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(rows), min(reference, len(rows)), replace=False))
    bank = unit[torch.from_numpy(chosen).to(device)]
    rank = max(1, math.ceil(alpha * len(chosen)))
    found = torch.empty(len(rows), dtype=torch.float32)
    position = torch.from_numpy(chosen).to(device)
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        similarity = unit[start:stop] @ bank.T
        anchors = torch.arange(start, stop, device=device)
        similarity[anchors[:, None] == position[None, :]] = -math.inf  # not itself
        found[start:stop] = similarity.topk(rank, dim=1).values[:, -1].float().cpu()
    thresholds = np.full(len(embeddings), UNUSED, dtype=np.float32)
    thresholds[rows] = found.numpy()
    return thresholds, len(rows), rank


def write_thresholds(  # noqa: PLR0913 (the inputs and the knobs of the quantile)
    output: Path,
    embeddings: np.ndarray,
    train: pa.Table,
    *,
    store: str,
    index: Path,
    alpha: float,
    reference: int = 200_000,
    seed: int = 0,
    device: str = "cpu",
) -> ThresholdMeta:
    """Compute the thresholds and write them with their meta (``text false-negatives``)."""
    thresholds, anchors, rank = compute_thresholds(
        embeddings, train, alpha, reference=reference, seed=seed, device=device
    )
    used = thresholds[thresholds < UNUSED]
    meta = ThresholdMeta(
        alpha=alpha,
        reference=min(reference, anchors),
        seed=seed,
        rows=len(thresholds),
        anchors=anchors,
        rank=rank,
        store=store,
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        quantiles={f"p{q}": float(np.percentile(used, q)) for q in (1, 10, 50, 90, 99)},
        created=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / THRESHOLDS, thresholds)
    (output / META).write_text(json.dumps(asdict(meta), indent=2) + "\n")
    return meta


def load_thresholds(directory: Path, alpha: float, rows: int | None = None) -> torch.Tensor:
    """The thresholds of ``directory``, checked against the run's alpha and store size."""
    meta = json.loads((directory / META).read_text())
    if not math.isclose(meta["alpha"], alpha):
        raise ValueError(f"{directory}: thresholds for alpha {meta['alpha']}, the run asks {alpha}")
    thresholds = np.load(directory / THRESHOLDS)
    if rows is not None and len(thresholds) != rows:
        raise ValueError(f"{directory}: {len(thresholds)} thresholds for {rows} caption rows")
    return torch.from_numpy(thresholds)
