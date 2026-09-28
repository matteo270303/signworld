"""Collaudo "durate delle didascalie" (§4.13.1).

The distribution of sentence durations fixes how far apart the 64 selected frames can fall:
the uniform half of the selection leaves at most about T/32 native frames between two
consecutive frames (§3.6).
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa

QUANTILES: Final = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
FRAMES_PER_CLIP: Final = 64
UNIFORM_FRAMES: Final = 32


@dataclass(frozen=True, slots=True)
class DurationSummary:
    clips: int
    mean_s: float
    quantiles_s: dict[str, float]
    max_s: float
    invalid: int
    """Clips whose end does not follow their start."""
    max_spacing_s: dict[str, float]
    """Largest gap between consecutive selected frames, T/32, at the same quantiles."""


@dataclass(frozen=True, slots=True)
class DurationReport:
    overall: DurationSummary
    by_sign_language: dict[str, DurationSummary]
    by_caption_language: dict[str, DurationSummary]


def summarize(durations: np.ndarray) -> DurationSummary:
    valid = durations[durations > 0]
    if len(valid) == 0:
        raise ValueError("no clip has a positive duration")
    quantiles = np.quantile(valid, QUANTILES)
    labels = [f"p{round(q * 100)}" for q in QUANTILES]
    return DurationSummary(
        clips=len(durations),
        mean_s=float(valid.mean()),
        quantiles_s={label: float(value) for label, value in zip(labels, quantiles, strict=True)},
        max_s=float(valid.max()),
        invalid=int((durations <= 0).sum()),
        max_spacing_s={
            label: float(value / UNIFORM_FRAMES)
            for label, value in zip(labels, quantiles, strict=True)
        },
    )


def duration_report(manifest: pa.Table) -> DurationReport:
    durations = (
        manifest.column("end_s").to_numpy() - manifest.column("start_s").to_numpy()
    ).astype(np.float64)
    return DurationReport(
        overall=summarize(durations),
        by_sign_language=_by_group(durations, manifest.column("sign_language").to_pylist()),
        by_caption_language=_by_group(durations, manifest.column("caption_language").to_pylist()),
    )


def _by_group(durations: np.ndarray, groups: list[str | None]) -> dict[str, DurationSummary]:
    labels = np.array(["unknown" if group is None else group for group in groups])
    return {
        str(label): summarize(durations[labels == label])
        for label in np.unique(labels)
        if (durations[labels == label] > 0).any()
    }
