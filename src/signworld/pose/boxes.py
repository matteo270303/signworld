"""Articulator boxes: where each hand, the body and the face are, frame by frame (§4.5.2).

The physical level reads the predicted tokens that fall inside each articulator's box, so the
box is derived from the confident keypoints and projected onto the 16-by-16 patch grid.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from .wholebody import ARTICULATOR_INDICES, CONFIDENCE_THRESHOLD, Articulator, PoseTrack

BOX_MARGIN: Final = 0.1
"""[Our choice] Each side grows by 10 % of the box size, so fingertips stay inside."""
MIN_CONFIDENT_KEYPOINTS: Final = 3
"""Fewer confident keypoints than this and the articulator is treated as not visible."""


@dataclass(frozen=True, slots=True)
class ArticulatorBoxes:
    boxes: np.ndarray
    """(frames, 4 articulators, 4) as x0, y0, x1, y1 in frame fractions; NaN when not visible."""
    visible: np.ndarray
    """(frames, 4) bool."""

    def outside_frame(self) -> np.ndarray:
        """(frames, 4) bool: a visible box that extends beyond the frame."""
        beyond = (self.boxes[..., :2] < 0).any(axis=-1) | (self.boxes[..., 2:] > 1).any(axis=-1)
        return np.asarray(self.visible & beyond)

    def patch_ranges(self, grid: int = 16) -> np.ndarray:
        """(frames, 4, 4) integer patch ranges x0, y0, x1, y1 (inclusive), clipped to the grid."""
        clipped = np.clip(np.nan_to_num(self.boxes, nan=0.0), 0.0, 1.0)
        cells = np.floor(clipped * grid).astype(np.int64)
        return np.asarray(np.clip(cells, 0, grid - 1))


def articulator_boxes(
    track: PoseTrack,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin: float = BOX_MARGIN,
) -> ArticulatorBoxes:
    frames = len(track.keypoints)
    boxes = np.full((frames, len(Articulator), 4), np.nan, dtype=np.float32)
    visible = np.zeros((frames, len(Articulator)), dtype=bool)
    confident = track.confident(threshold)
    for column, part in enumerate(Articulator):
        indices = list(ARTICULATOR_INDICES[part])
        points = track.keypoints[:, indices]
        mask = confident[:, indices]
        for frame in range(frames):
            chosen = points[frame][mask[frame]]
            if len(chosen) < MIN_CONFIDENT_KEYPOINTS:
                continue
            low, high = chosen.min(axis=0), chosen.max(axis=0)
            pad = margin * (high - low)
            boxes[frame, column] = (*(low - pad), *(high + pad))
            visible[frame, column] = True
    return ArticulatorBoxes(boxes, visible)
