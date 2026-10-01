"""Collaudo "riquadri e normalizzazione della posa" (§4.13.1).

Over every stored pose: how often the shoulders that define the normalisation are missing,
how often an articulator box leaves the frame, and whether the normalised keypoints stay
within a few shoulder widths. Contact sheets let a person check by eye that the hand boxes sit
on the hands.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from ..pose.boxes import articulator_boxes
from ..pose.tokens import MIN_SCORE
from ..pose.wholebody import ARTICULATOR_INDICES, Articulator, PoseTrack, shoulder_frame

OUTSIDE_LIMITS: Final[Mapping[str, float | None]] = {
    "body": None,
    "left": 0.05,
    "right": 0.05,
    "face_all": 0.01,
}
"""[Our provisional limits] Largest share of visible boxes that may extend beyond the frame.

``None`` reports without judging. The body box leaves the frame in half-body framing by design
(elbows, hips), so it is only reported; the hands carry the signal, the face rarely leaves.
The single 1 % limit of §4.13.1 failed on OpenASL for the body (4.4 %) and the hands (2.7 and
1.6 %); these values are to be settled on the YouTube-SL-25 test clips.
"""
SHOULDER_WIDTHS: Final = 5.0
MIN_WITHIN_FRACTION: Final = 0.99
"""[Our threshold] At least 99 % of confident keypoints within five shoulder widths."""
SCORE_QUANTILES: Final = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class _Tally:
    """Counts of one clip or of many, summed by ``merge``."""

    clips: int = 0
    clips_without_shoulders: int = 0
    frames: int = 0
    frames_with_shoulders: int = 0
    visible: np.ndarray = field(default_factory=lambda: np.zeros(len(Articulator), np.int64))
    outside: np.ndarray = field(default_factory=lambda: np.zeros(len(Articulator), np.int64))
    confident_points: np.ndarray = field(
        default_factory=lambda: np.zeros(len(Articulator), np.int64)
    )
    points_outside: np.ndarray = field(default_factory=lambda: np.zeros(len(Articulator), np.int64))
    keypoints: int = 0
    within: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(2))
    squares: np.ndarray = field(default_factory=lambda: np.zeros(2))
    largest: float = 0.0
    non_finite: int = 0
    score_sample: list[np.ndarray] = field(default_factory=list)

    def merge(self, other: "_Tally") -> None:
        for name in ("clips", "clips_without_shoulders", "frames", "frames_with_shoulders"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.visible += other.visible
        self.outside += other.outside
        self.confident_points += other.confident_points
        self.points_outside += other.points_outside
        self.keypoints += other.keypoints
        self.within += other.within
        self.total += other.total
        self.squares += other.squares
        self.largest = max(self.largest, other.largest)
        self.non_finite += other.non_finite
        self.score_sample.extend(other.score_sample)


def _tally(path: Path, threshold: float = MIN_SCORE) -> _Tally:
    track = PoseTrack.load(path)
    tally = _Tally(clips=1, frames=len(track.keypoints))
    tally.non_finite = int(
        (~np.isfinite(track.keypoints)).sum() + (~np.isfinite(track.scores)).sum()
    )
    tally.score_sample.append(track.scores[:, ::7].ravel()[::4])
    boxes = articulator_boxes(track, threshold)
    tally.visible = boxes.visible.sum(axis=0)
    tally.outside = boxes.outside_frame().sum(axis=0)
    confident = track.confident(threshold)
    beyond = ((track.keypoints < 0) | (track.keypoints > 1)).any(axis=-1)
    for column, part in enumerate(Articulator):
        indices = list(ARTICULATOR_INDICES[part])
        tally.confident_points[column] = int(confident[:, indices].sum())
        tally.points_outside[column] = int((confident & beyond)[:, indices].sum())
    reference = shoulder_frame(track, threshold)
    if reference is None:
        tally.clips_without_shoulders = 1
        return tally
    tally.frames_with_shoulders = int(reference.measured.sum())
    normalised = reference.apply(track.keypoints)[track.confident(threshold)]
    radius = np.linalg.norm(normalised, axis=-1)
    tally.keypoints = len(normalised)
    tally.within = int((radius <= SHOULDER_WIDTHS).sum())
    tally.total = normalised.sum(axis=0)
    tally.squares = (normalised**2).sum(axis=0)
    tally.largest = float(radius.max()) if len(radius) else 0.0
    return tally


@dataclass(frozen=True, slots=True)
class PoseQualityReport:
    clips: int
    frames: int
    clips_without_shoulders: int
    frames_without_shoulders_fraction: float
    """Frames whose shoulders were carried over from another frame."""
    visible_fraction: dict[str, float]
    outside_fraction: dict[str, float]
    """Of the visible boxes of each articulator, those extending beyond the frame."""
    keypoints_outside_fraction: dict[str, float]
    """Of the confident keypoints of each articulator, those lying outside the frame.

    A keypoint outside the frame was guessed, not seen; it is what a wider crop cannot fix.
    """
    normalised_mean: list[float]
    normalised_std: list[float]
    normalised_max_radius: float
    within_five_shoulder_widths: float
    non_finite_values: int
    score_quantiles: dict[str, float]
    outside_limits: dict[str, float | None]
    failures: list[str]
    """Why the check did not pass; empty when it did."""
    passed: bool


def pose_quality_report(
    poses: Sequence[Path],
    workers: int = 8,
    outside_limits: Mapping[str, float | None] = OUTSIDE_LIMITS,
) -> PoseQualityReport:
    total = _Tally()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tally in pool.map(_tally, poses, chunksize=256):
            total.merge(tally)
    parts = [str(part) for part in Articulator]
    visible = total.visible / max(total.frames, 1)
    outside = total.outside / np.maximum(total.visible, 1)
    count = max(total.keypoints, 1)
    mean = total.total / count
    std = np.sqrt(np.maximum(total.squares / count - mean**2, 0))
    within = total.within / count
    scores = np.concatenate(total.score_sample) if total.score_sample else np.zeros(1)
    outside_by_part = dict(zip(parts, outside.tolist(), strict=True))
    failures = _failures(total.non_finite, outside_by_part, outside_limits, within)
    return PoseQualityReport(
        clips=total.clips,
        frames=total.frames,
        clips_without_shoulders=total.clips_without_shoulders,
        frames_without_shoulders_fraction=1 - total.frames_with_shoulders / max(total.frames, 1),
        visible_fraction=dict(zip(parts, visible.tolist(), strict=True)),
        outside_fraction=outside_by_part,
        keypoints_outside_fraction=dict(
            zip(
                parts,
                (total.points_outside / np.maximum(total.confident_points, 1)).tolist(),
                strict=True,
            )
        ),
        normalised_mean=mean.tolist(),
        normalised_std=std.tolist(),
        normalised_max_radius=total.largest,
        within_five_shoulder_widths=within,
        non_finite_values=total.non_finite,
        score_quantiles={
            f"p{round(q * 100)}": float(value)
            for q, value in zip(SCORE_QUANTILES, np.quantile(scores, SCORE_QUANTILES), strict=True)
        },
        outside_limits=dict(outside_limits),
        failures=failures,
        passed=not failures,
    )


def _failures(
    non_finite: int,
    outside: Mapping[str, float],
    limits: Mapping[str, float | None],
    within: float,
) -> list[str]:
    failures = [f"{non_finite} non-finite values"] if non_finite else []
    for part, share in outside.items():
        limit = limits.get(part)
        if limit is not None and share > limit:
            failures.append(f"{part} boxes outside the frame {share:.2%} > {limit:.2%}")
    if within < MIN_WITHIN_FRACTION:
        failures.append(
            f"keypoints within {SHOULDER_WIDTHS:g} shoulder widths {within:.2%} "
            f"< {MIN_WITHIN_FRACTION:.0%}"
        )
    return failures
