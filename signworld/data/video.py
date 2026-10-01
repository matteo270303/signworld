"""Frame access for sentence clips, with the decoder the training data loader uses."""

from pathlib import Path

import decord
import numpy as np


class ClipReader:
    """Decodes frames of a clip by index with decord, the decoder of the training loader.

    Using the same decoder in the checks and in training matters: two decoders can disagree
    on frame indices of variable-frame-rate videos, which would silently misalign poses.
    """

    def __init__(self, path: Path) -> None:
        self._reader = decord.VideoReader(str(path))

    def __len__(self) -> int:
        return len(self._reader)

    def frames(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """(len(indices), H, W, 3) uint8 RGB frames; indices are clamped to the clip."""
        clamped = np.clip(np.asarray(indices, dtype=np.int64), 0, len(self) - 1)
        batch: np.ndarray = self._reader.get_batch(clamped.tolist()).asnumpy()
        return batch

    def all_frames(self) -> np.ndarray:
        return self.frames(np.arange(len(self)))
