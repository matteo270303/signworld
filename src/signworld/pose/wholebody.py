"""COCO-WholeBody poses (133 keypoints) and the four articulators of Uni-Sign (§3.6).

The indices, the confidence threshold and the normalisation reproduce Uni-Sign's
``load_part_kp`` and ``crop_scale`` (github.com/ZechengLi19/Uni-Sign, commit eed438b): its
pre-trained pose encoder only means something on inputs prepared that way (assertion P9).
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import numpy as np

WHOLEBODY_KEYPOINTS: Final = 133
LEFT_SHOULDER: Final = 5
RIGHT_SHOULDER: Final = 6
_POSE_DIMENSIONS: Final = 3
_CHANNELS: Final = 3
_MIN_BODY_POINTS: Final = 4
"""Uni-Sign's minimum of confident body keypoints to fit the normalisation box."""
CONFIDENCE_THRESHOLD: Final = 0.3
MIN_SHOULDER_SHARE: Final = 0.5
"""A frame whose shoulders are closer than half the clip's median distance is a bad detection:
dividing by that distance would send the keypoints to hundreds of shoulder units.
"""
"""Keypoints scoring at or below this are treated as missing, as in Uni-Sign."""


class Articulator(StrEnum):
    BODY = "body"
    LEFT_HAND = "left"
    RIGHT_HAND = "right"
    FACE = "face_all"


ARTICULATOR_INDICES: Final[dict[Articulator, tuple[int, ...]]] = {
    # nose, ears, shoulders, elbows, wrists
    Articulator.BODY: (0, 3, 4, 5, 6, 7, 8, 9, 10),
    Articulator.LEFT_HAND: tuple(range(91, 112)),
    Articulator.RIGHT_HAND: tuple(range(112, 133)),
    # every other jaw point, the eight mouth-contour points, then the nose tip
    Articulator.FACE: (*range(23, 40, 2), *range(83, 91), 53),
}

# Hands are expressed relative to their wrist and the face relative to its nose tip.
_ROOT_POSITION: Final[dict[Articulator, int]] = {
    Articulator.LEFT_HAND: 0,
    Articulator.RIGHT_HAND: 0,
    Articulator.FACE: -1,
}


@dataclass(frozen=True, slots=True)
class PoseTrack:
    """Keypoints of the frames of one clip that the pose estimator saw.

    Coordinates are fractions of the clip's (square, cropped) frame; scores are the raw
    confidences of the estimator, which for RTMW are not probabilities and can exceed 1.
    """

    keypoints: np.ndarray
    """(frames, 133, 2) float32, x and y in [0, 1] when inside the frame."""
    scores: np.ndarray
    """(frames, 133) float32."""
    frame_indices: np.ndarray
    """(frames,) int: index of each pose frame among the clip's decoded frames."""
    total_frames: int

    @classmethod
    def load(cls, path: Path, prefix: str = "") -> Self:
        """Read ``{prefix}pose`` (frames, 133, 3), ``{prefix}idx`` and ``total`` of an ``.npz``.

        The empty prefix gives the 64 selected frames; ``contiguous_`` gives 64 consecutive
        frames of the same clip, stored by the test materialization.
        """
        with np.load(path) as archive:
            pose = archive[f"{prefix}pose"].astype(np.float32)
            if pose.ndim != _POSE_DIMENSIONS or pose.shape[1:] != (WHOLEBODY_KEYPOINTS, _CHANNELS):
                raise ValueError(f"{path}: expected (frames, 133, 3) poses, got {pose.shape}")
            return cls(
                keypoints=pose[..., :2],
                scores=pose[..., 2],
                frame_indices=archive[f"{prefix}idx"].astype(np.int64),
                total_frames=int(archive["total"]),
            )

    def confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> np.ndarray:
        return np.asarray(self.scores > threshold)


def unisign_parts(
    track: PoseTrack, threshold: float = CONFIDENCE_THRESHOLD
) -> dict[Articulator, np.ndarray]:
    """The four articulators as Uni-Sign prepares them: (frames, keypoints, 3) in [-1, 1].

    The body is fitted into a square box around its confident keypoints over the whole clip
    and mapped to [-1, 1]; hands and face are taken relative to their root keypoint and
    divided by the same box side. Keypoints at or below ``threshold`` become all zeros, and
    a clip whose body cannot be boxed yields zeros everywhere, as in Uni-Sign.
    """
    body = _with_scores(track, Articulator.BODY)
    confident = body[..., 2] > threshold
    points = body[confident][:, :2]
    parts: dict[Articulator, np.ndarray] = {}
    if len(points) < _MIN_BODY_POINTS:
        return {part: np.zeros_like(_with_scores(track, part)) for part in Articulator}
    low, high = points.min(axis=0), points.max(axis=0)
    side = float((high - low).max())
    if side == 0:
        return {part: np.zeros_like(_with_scores(track, part)) for part in Articulator}
    corner = (low + high - side) / 2

    normalised = body.copy()
    normalised[..., :2] = np.clip(((body[..., :2] - corner) / side - 0.5) * 2, -1, 1)
    parts[Articulator.BODY] = normalised
    for part, root in _ROOT_POSITION.items():
        values = _with_scores(track, part)
        values[..., :2] = np.clip((values[..., :2] - values[:, [root], :2]) / side, -1, 1)
        parts[part] = values
    for values in parts.values():
        values[values[..., 2] <= threshold] = 0
    return {part: parts[part] for part in Articulator}


def _with_scores(track: PoseTrack, part: Articulator) -> np.ndarray:
    indices = list(ARTICULATOR_INDICES[part])
    return np.concatenate(
        [track.keypoints[:, indices], track.scores[:, indices, None]], axis=-1
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ShoulderFrame:
    """The per-frame reference of §3.6: origin between the shoulders, unit = their distance."""

    centre: np.ndarray
    """(frames, 2)."""
    scale: np.ndarray
    """(frames,)."""
    measured: np.ndarray
    """(frames,) bool: shoulders reliable in this frame, rather than carried over."""

    def apply(self, keypoints: np.ndarray) -> np.ndarray:
        """Keypoints of shape (frames, n, 2) in shoulder units."""
        normalised: np.ndarray = (keypoints - self.centre[:, None]) / self.scale[:, None, None]
        return normalised


def shoulder_frame(
    track: PoseTrack, threshold: float = CONFIDENCE_THRESHOLD
) -> ShoulderFrame | None:
    """Shoulder reference per frame, or ``None`` when no frame shows both shoulders.

    A frame without reliable shoulders takes the reference of the last frame that had them;
    frames before the first reliable one take the first reliable reference. Shoulders far
    closer than the clip's median count as unreliable, whatever the estimator's confidence.
    """
    left = track.keypoints[:, LEFT_SHOULDER]
    right = track.keypoints[:, RIGHT_SHOULDER]
    distance = np.linalg.norm(left - right, axis=1)
    measured = (
        (track.scores[:, LEFT_SHOULDER] > threshold)
        & (track.scores[:, RIGHT_SHOULDER] > threshold)
        & (distance > 0)
    )
    if not measured.any():
        return None
    measured &= distance >= MIN_SHOULDER_SHARE * np.median(distance[measured])
    source = np.where(measured, np.arange(len(measured)), -1)
    source = np.maximum.accumulate(source)
    source[source < 0] = int(np.argmax(measured))
    return ShoulderFrame(
        centre=((left + right) / 2)[source],
        scale=distance[source],
        measured=measured,
    )
