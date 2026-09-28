"""Pose sequences as the pose teachers see them: 69 joints over 32 steps of two frames.

A step covers the two frames of one video tubelet, so a teacher's output lines up with the
video tokens (T' = 32, §4.3). Coordinates are in shoulder units (origin between the shoulders,
unit their distance), which removes position and scale but keeps orientation, since palm
orientation is phonological. A keypoint counts only if RTMW scores it above ``MIN_SCORE``, it
lies inside the frame and it falls within ``MAX_SHOULDER_UNITS`` of the shoulders; otherwise it
is zero and flagged missing.
"""

from dataclasses import dataclass
from typing import Final, Self

import numpy as np

from .frames import FRAMES_PER_CLIP
from .wholebody import ARTICULATOR_INDICES, Articulator, PoseTrack, shoulder_frame

JOINTS: Final = tuple(index for part in Articulator for index in ARTICULATOR_INDICES[part])
"""The 69 keypoints of the four articulators, articulator by articulator."""
JOINT_ARTICULATOR: Final = np.array(
    [column for column, part in enumerate(Articulator) for _ in ARTICULATOR_INDICES[part]]
)
FRAMES_PER_STEP: Final = 2
STEPS: Final = FRAMES_PER_CLIP // FRAMES_PER_STEP
TOKEN_CHANNELS: Final = FRAMES_PER_STEP * 3
"""x, y and a presence flag for each frame of the step."""
MAX_SHOULDER_UNITS: Final = 5.0
"""A keypoint farther than this from the origin is not on the signer: it comes from a wrong
detection, and left in it would dominate every squared error (A4 uses the same bound).
"""
MIN_SCORE: Final = 1.0
"""[Our choice] RTMW scores are not probabilities: about 5 % of keypoints score below 1.2 and
the median is 7.6, so Uni-Sign's 0.3 keeps nearly every guess. 1.0 drops the clearly unreliable.
"""


@dataclass(frozen=True, slots=True)
class PoseSequence:
    """Shoulder-normalised joints of the 64 selected frames of one clip."""

    joints: np.ndarray
    """(64, 69, 2) float32; zero where missing."""
    present: np.ndarray
    """(64, 69) bool."""

    @classmethod
    def from_track(cls, track: PoseTrack, min_score: float = MIN_SCORE) -> Self | None:
        """``None`` when no frame shows both shoulders, so no reference exists."""
        if len(track.keypoints) != FRAMES_PER_CLIP:
            raise ValueError(f"expected {FRAMES_PER_CLIP} frames, got {len(track.keypoints)}")
        reference = shoulder_frame(track, min_score)
        if reference is None:
            return None
        raw = track.keypoints[:, JOINTS]
        inside = ((raw >= 0) & (raw <= 1)).all(axis=-1)
        joints = reference.apply(raw).astype(np.float32)
        plausible = (np.abs(joints) <= MAX_SHOULDER_UNITS).all(axis=-1)
        present = (track.scores[:, JOINTS] > min_score) & inside & plausible
        joints[~present] = 0.0
        return cls(joints, present)

    def tokens(self) -> np.ndarray:
        """(32, 69, 6): for each step and joint, x, y and presence of its two frames."""
        values = np.concatenate([self.joints, self.present[..., None]], axis=-1)
        steps = values.reshape(STEPS, FRAMES_PER_STEP, len(JOINTS), 3)
        tokens: np.ndarray = steps.transpose(0, 2, 1, 3).reshape(STEPS, len(JOINTS), -1)
        return tokens.astype(np.float32)

    def step_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """(32, 69, 2) mean position over the present frames of each step, and (32, 69) weights."""
        joints = self.joints.reshape(STEPS, FRAMES_PER_STEP, len(JOINTS), 2)
        present = self.present.reshape(STEPS, FRAMES_PER_STEP, len(JOINTS)).astype(np.float32)
        count = present.sum(axis=1)
        positions = (joints * present[..., None]).sum(axis=1) / np.maximum(count, 1)[..., None]
        return positions.astype(np.float32), (count > 0).astype(np.float32)


def articulator_columns(part: Articulator) -> np.ndarray:
    """Positions of an articulator's joints within ``JOINTS``."""
    return np.flatnonzero(list(Articulator).index(part) == JOINT_ARTICULATOR)
