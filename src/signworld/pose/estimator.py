"""Whole-body pose estimation with RTMW (rtmlib), as used for the stored poses (§3.6)."""

from typing import Final, Literal

import numpy as np
import onnxruntime
from rtmlib import Wholebody

from .wholebody import WHOLEBODY_KEYPOINTS

Mode = Literal["performance", "balanced", "lightweight"]

STORED_POSE_MODE: Final[Mode] = "performance"
"""The RTMW configuration that produced the stored OpenASL poses."""


class WholebodyEstimator:
    """RTMW on single frames; keeps the most confident person, in frame fractions.

    ``__call__`` returns (133, 3): x and y divided by the frame width and height, then the raw
    RTMW score. Frames without a detected person give zeros, as in the stored poses.
    """

    def __init__(self, mode: Mode = STORED_POSE_MODE, device: str = "cpu") -> None:
        if device == "cuda":
            # Load the CUDA and cuDNN libraries that come with torch, so onnxruntime finds them.
            onnxruntime.preload_dlls()
        self._model = Wholebody(mode=mode, backend="onnxruntime", device=device)
        if device == "cuda":
            for tool in (self._model.det_model, self._model.pose_model):
                if "CUDAExecutionProvider" not in tool.session.get_providers():
                    raise RuntimeError("onnxruntime fell back to the CPU; check the CUDA setup")

    def people(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(people, 4) person boxes x0, y0, x1, y1 in pixels, from RTMW's own detector."""
        boxes = self._model.det_model(np.ascontiguousarray(frame_rgb[:, :, ::-1]))
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Pose of the most confident person the detector finds in the frame."""
        keypoints, scores = self._model(np.ascontiguousarray(frame_rgb[:, :, ::-1]))
        return _most_confident(keypoints, scores, frame_rgb.shape[:2])

    def in_box(self, frame_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Pose of the person inside ``box`` (x0, y0, x1, y1 pixels), without the detector.

        When the signer's box is already known, as in a crop centred on them, this skips a
        detection per frame and cannot latch onto another person in the background.
        """
        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        keypoints, scores = self._model.pose_model(bgr, bboxes=[np.asarray(box).tolist()])
        return _most_confident(keypoints, scores, frame_rgb.shape[:2])


def _most_confident(
    keypoints: np.ndarray | None, scores: np.ndarray | None, shape: tuple[int, ...]
) -> np.ndarray:
    """(133, 3) of the person with the highest mean score: x / width, y / height, score."""
    height, width = shape
    pose = np.zeros((WHOLEBODY_KEYPOINTS, 3), dtype=np.float32)
    if keypoints is not None and scores is not None and len(keypoints) > 0:
        person = int(np.argmax(scores.mean(axis=1)))
        pose[:, 0] = keypoints[person][:, 0] / max(width, 1)
        pose[:, 1] = keypoints[person][:, 1] / max(height, 1)
        pose[:, 2] = scores[person]
    return pose
