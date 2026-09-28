"""Selection of the 64 frames of a sentence clip, guided by local motion (§3.6).

Half of the frames are spread uniformly over the cumulative local motion, so they are dense
where the hands move; the other half are spread uniformly in time, so no stretch of the
sentence is left out. The implementation reproduces the one that produced the stored OpenASL
poses, so that recomputed indices match the saved ones exactly.
"""

from typing import Final

import numpy as np

FRAMES_PER_CLIP: Final = 64
_MIN_FRAMES_FOR_MOTION: Final = 2
UNIFORM_FRACTION: Final = 0.5


def uniform_indices(total: int, count: int) -> list[int]:
    """Centre of each of ``count`` equal segments of ``total`` frames."""
    if total <= 0:
        return [0] * count
    step = total / count
    indices = []
    for position in range(count):
        start = int(position * step)
        end = max(start, int((position + 1) * step) - 1)
        indices.append(min((start + end) // 2, total - 1))
    return indices


def local_motion(
    frames: np.ndarray, stride: int = 3, block: int = 8, top_blocks: int = 8
) -> np.ndarray:
    """Motion of each frame: mean of the ``top_blocks`` largest block differences to the last.

    Taking the most active blocks, not the whole image, keeps finger movement visible while
    the arms rest. ``frames`` is (T, H, W, 3) uint8; the result has one value per frame.
    """
    grey = frames[:, ::stride, ::stride, :].astype(np.float32).mean(axis=3)
    count, height, width = grey.shape
    rows, columns = (height // block) * block, (width // block) * block
    if count < _MIN_FRAMES_FOR_MOTION or rows < block or columns < block:
        if count < _MIN_FRAMES_FOR_MOTION:
            return np.zeros(count, dtype=np.float32)
        change = np.abs(np.diff(grey, axis=0)).reshape(count - 1, -1).mean(axis=1)
        return np.concatenate([change[:1], change])
    difference = np.abs(np.diff(grey[:, :rows, :columns], axis=0))
    blocks = difference.reshape(count - 1, rows // block, block, columns // block, block)
    per_block = blocks.mean(axis=(2, 4)).reshape(count - 1, -1)
    kept = min(top_blocks, per_block.shape[1])
    motion = np.sort(per_block, axis=1)[:, -kept:].mean(axis=1)
    return np.concatenate([motion[:1], motion])


def select_frames(
    motion: np.ndarray, count: int = FRAMES_PER_CLIP, uniform_fraction: float = UNIFORM_FRACTION
) -> list[int]:
    """Sorted indices of ``count`` frames: motion-guided and uniform halves, merged.

    Clips with at most ``count`` frames are sampled uniformly, repeating frames. When the two
    halves pick the same frame, uniform frames fill the gap.
    """
    total = len(motion)
    if total <= count:
        return uniform_indices(total, count)
    uniform_count = round(count * uniform_fraction)
    motion_count = count - uniform_count
    picks: set[int] = set()
    cumulative = np.cumsum(motion)
    if cumulative[-1] > 0 and motion_count > 0:
        levels = (np.arange(motion_count) + 0.5) / motion_count
        positions = np.searchsorted(cumulative / cumulative[-1], levels)
        picks.update(int(min(position, total - 1)) for position in positions)
    picks.update(uniform_indices(total, max(uniform_count, 1)))
    for index in uniform_indices(total, count):
        if len(picks) >= count:
            break
        picks.add(index)
    selected = sorted(picks)[:count]
    return selected + [selected[-1]] * (count - len(selected))
