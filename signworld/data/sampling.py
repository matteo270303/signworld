"""Which 64 frames of a sentence the model sees (§3.6).

``MotionGuidedSampler`` is the project's selection: half of the frames spread over the
cumulative local motion (dense where the hands move), half uniform in time (no stretch left
out). It wraps ``pose.frames``, the implementation that produced the stored indices, so the
data and the model can never disagree on it. ``UniformSampler`` is the fixed-step baseline.
"""

from typing import Protocol

import numpy as np

from signworld.data.pose.frames import local_motion, select_frames, uniform_indices
from signworld.experiment.train.config import SamplingSettings


class FrameSampler(Protocol):
    def select(self, frames: np.ndarray) -> np.ndarray:
        """Sorted indices of the frames to keep, from (T, H, W, 3) uint8 frames."""
        ...


class MotionGuidedSampler:
    def __init__(self, settings: SamplingSettings) -> None:
        self.settings = settings

    def select(self, frames: np.ndarray) -> np.ndarray:
        s = self.settings
        motion = local_motion(frames, stride=s.stride, block=s.block, top_blocks=s.top_blocks)
        return np.asarray(select_frames(motion, s.frames, s.uniform_fraction), dtype=np.int64)


class UniformSampler:
    def __init__(self, settings: SamplingSettings) -> None:
        self.settings = settings

    def select(self, frames: np.ndarray) -> np.ndarray:
        return np.asarray(uniform_indices(len(frames), self.settings.frames), dtype=np.int64)


def build_sampler(settings: SamplingSettings) -> FrameSampler:
    if settings.strategy == "motion":
        return MotionGuidedSampler(settings)
    return UniformSampler(settings)
