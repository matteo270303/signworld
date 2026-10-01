"""Collaudo "allineamento video-posa" (§4.13.1).

The frames the data loader decodes must be the frames the stored poses were estimated on.
Two checks: recomputing the frame selection reproduces the stored indices, and re-estimating
the pose on the decoded frames reproduces the stored keypoints. Comparing against the frames
one step before and after exposes an off-by-one shift, the classic silent failure.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from signworld.data.pose.frames import local_motion, select_frames
from signworld.data.pose.wholebody import (
    ARTICULATOR_INDICES,
    CONFIDENCE_THRESHOLD,
    Articulator,
    PoseTrack,
)
from signworld.data.video import ClipReader

MAX_HAND_ERROR: Final = 0.02
"""Pass threshold: mean hand-keypoint error below 2 % of the frame width (§4.13.1)."""
SHIFTS: Final = (-1, 0, 1)
SHIFT_PROBE_FRAMES: Final = 16
"""Frames per clip re-estimated at the neighbouring indices, to keep the cost bounded."""

Estimator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class MaterializedClip:
    clip_id: str
    video: Path
    pose: Path


@dataclass(frozen=True, slots=True)
class SelectionReport:
    clips: int
    reproduced: int
    mismatched: list[str]
    """Clip IDs whose recomputed selection differs from the stored indices."""

    @property
    def passed(self) -> bool:
        return self.reproduced == self.clips


def frame_selection_report(clips: Sequence[MaterializedClip]) -> SelectionReport:
    mismatched = []
    for clip in clips:
        track = PoseTrack.load(clip.pose)
        reader = ClipReader(clip.video)
        selected = select_frames(local_motion(reader.all_frames()))
        if len(reader) != track.total_frames or selected != track.frame_indices.tolist():
            mismatched.append(clip.clip_id)
    return SelectionReport(len(clips), len(clips) - len(mismatched), mismatched)


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    clips: int
    frames: int
    hand_error_mean: float
    """Mean distance between stored and re-estimated hand keypoints, in frame widths."""
    hand_error_p95: float
    error_by_articulator: dict[str, float]
    error_by_shift: dict[int, float]
    """Mean hand error against the frame ``shift`` steps from the stored index."""
    passed: bool


def _hand_errors(
    stored: np.ndarray, estimated: np.ndarray, threshold: float
) -> dict[Articulator, np.ndarray]:
    """Per-keypoint distances for keypoints confident in both poses, by articulator."""
    errors = {}
    for part in Articulator:
        indices = list(ARTICULATOR_INDICES[part])
        both = (stored[..., indices, 2] > threshold) & (estimated[..., indices, 2] > threshold)
        distance = np.linalg.norm(stored[..., indices, :2] - estimated[..., indices, :2], axis=-1)
        errors[part] = distance[both]
    return errors


def pose_alignment_report(
    clips: Sequence[MaterializedClip],
    estimate: Estimator,
    threshold: float = CONFIDENCE_THRESHOLD,
    seed: int = 0,
) -> AlignmentReport:
    """Re-estimate the pose on the decoded frames and compare it with the stored one."""
    rng = np.random.default_rng(seed)
    errors: dict[Articulator, list[np.ndarray]] = {part: [] for part in Articulator}
    shifted: dict[int, list[np.ndarray]] = {shift: [] for shift in SHIFTS}
    frames = 0
    hands = (Articulator.LEFT_HAND, Articulator.RIGHT_HAND)
    for clip in clips:
        track = PoseTrack.load(clip.pose)
        reader = ClipReader(clip.video)
        stored = np.concatenate([track.keypoints, track.scores[..., None]], axis=-1)
        decoded = reader.frames(track.frame_indices)
        estimated = np.stack([estimate(frame) for frame in decoded])
        for part, values in _hand_errors(stored, estimated, threshold).items():
            errors[part].append(values)
        frames += len(decoded)

        probe = rng.choice(len(track.frame_indices), SHIFT_PROBE_FRAMES, replace=False)
        for shift in SHIFTS:
            indices = track.frame_indices[probe] + shift
            if shift == 0:
                probe_estimates = estimated[probe]
            else:
                probe_estimates = np.stack([estimate(f) for f in reader.frames(indices)])
            per_part = _hand_errors(stored[probe], probe_estimates, threshold)
            shifted[shift].append(np.concatenate([per_part[part] for part in hands]))

    hand = np.concatenate([values for part in hands for values in errors[part]])
    by_shift = {shift: float(np.concatenate(values).mean()) for shift, values in shifted.items()}
    mean = float(hand.mean()) if len(hand) else float("nan")
    return AlignmentReport(
        clips=len(clips),
        frames=frames,
        hand_error_mean=mean,
        hand_error_p95=float(np.quantile(hand, 0.95)) if len(hand) else float("nan"),
        error_by_articulator={
            str(part): float(np.concatenate(values).mean()) for part, values in errors.items()
        },
        error_by_shift=by_shift,
        passed=mean < MAX_HAND_ERROR and by_shift[0] <= min(by_shift.values()),
    )
