"""Sentence clips cut from downloaded videos, cropped to the signer, with their poses (§3.6).

For each clip: find the signer with the person detector on a few frames, crop a square around
the union of their boxes enlarged by 15 %, resize to 256 pixels, select the 64 frames by local
motion and estimate the pose on exactly those frames. The poses of 64 consecutive frames are
estimated too, for the comparison of selected and contiguous frames (§4.13.1).

Everything is written under a separate test root, never next to the downloaded data.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import cv2
import decord
import numpy as np
import pyarrow as pa

from ..acquisition.sharding import Shard
from ..pose.frames import FRAMES_PER_CLIP, local_motion, select_frames

logger = logging.getLogger(__name__)

DECODE_CHUNK: Final = 64
"""Frames decoded at a time, so long clips never sit in memory at full resolution."""


class PoseEstimator(Protocol):
    def people(self, frame_rgb: np.ndarray) -> np.ndarray: ...

    def in_box(self, frame_rgb: np.ndarray, box: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SignerBox:
    union: np.ndarray
    """Union of the signer's detections, x0, y0, x1, y1 in source pixels."""
    crop: np.ndarray
    """The square around it, enlarged by the margin; may extend beyond the source frame."""

    def union_in_crop(self, size: int) -> np.ndarray:
        """The union box in the pixels of the ``size``-pixel crop."""
        scale = size / (self.crop[2] - self.crop[0])
        inside: np.ndarray = (self.union - np.tile(self.crop[:2], 2)) * scale
        return inside


@dataclass(frozen=True, slots=True)
class ClipCut:
    clip_id: str
    video_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class MaterializedRecord:
    clip_id: str
    video: str
    pose: str
    fps: float
    first_frame: int
    frames: int
    box: list[float]
    """Square crop x0, y0, x1, y1 in source pixels; may extend beyond the source frame."""


def sample_cuts(
    manifest: "pa.Table", count: int, min_duration_s: float, max_duration_s: float, seed: int
) -> list[ClipCut]:
    """Up to ``count`` clips within the duration bounds, one channel after another in turn.

    Taking channels in turn keeps a small sample from being one prolific channel.
    """
    columns = manifest.select(["clip_id", "video_id", "start_s", "end_s", "channel_id"])
    rows = columns.to_pylist()
    rng = np.random.default_rng(seed)
    by_channel: dict[str | None, list[dict[str, Any]]] = {}
    for position in rng.permutation(len(rows)):
        row = rows[position]
        if min_duration_s <= row["end_s"] - row["start_s"] <= max_duration_s:
            by_channel.setdefault(row["channel_id"], []).append(row)
    chosen: list[ClipCut] = []
    queues = list(by_channel.values())
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                row = queue.pop()
                chosen.append(
                    ClipCut(row["clip_id"], row["video_id"], row["start_s"], row["end_s"])
                )
    return chosen


def signer_box(boxes_per_frame: list[np.ndarray], margin: float) -> SignerBox | None:
    """Square around the union of the largest person of each frame, enlarged by ``margin``."""
    largest = [
        boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
        for boxes in boxes_per_frame
        if len(boxes)
    ]
    if not largest:
        return None
    stacked = np.stack(largest)
    low, high = stacked[:, :2].min(axis=0), stacked[:, 2:].max(axis=0)
    centre = (low + high) / 2
    half = (high - low).max() * (1 + margin) / 2
    return SignerBox(np.concatenate([low, high]), np.concatenate([centre - half, centre + half]))


def crop_square(frames: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Crop ``box`` (black outside the source frame) and resize every frame to ``size``.

    The side comes from the mean of the two extents: rounding each corner on its own would give
    a 576 x 575 crop whenever the box straddles a half pixel, and the frames would not fit.
    """
    count, height, width = frames.shape[:3]
    x0, y0 = np.round(box[:2]).astype(int)
    side = round(float(box[2] - box[0] + box[3] - box[1]) / 2)
    x1, y1 = x0 + side, y0 + side
    canvas = np.zeros((count, side, side, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    canvas[:, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frames[:, sy0:sy1, sx0:sx1]
    return np.stack(
        [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in canvas]
    )


class ClipMaterializer:
    def __init__(
        self,
        estimator: PoseEstimator,
        size: int = 256,
        margin: float = 0.15,
        detection_frames: int = 8,
        *,
        contiguous: bool = True,
    ) -> None:
        """``contiguous``: also estimate the pose of 64 consecutive frames, which only the
        collaudo reads (§4.13.1); the training corpus skips it, halving the pose cost."""
        self._estimator = estimator
        self._size = size
        self._margin = margin
        self._detection_frames = detection_frames
        self._contiguous = contiguous

    def materialize(self, source: Path, cut: ClipCut, root: Path) -> MaterializedRecord | None:
        """Write the cropped clip and its poses under ``root``; ``None`` without a signer."""
        reader = decord.VideoReader(str(source))
        fps = float(reader.get_avg_fps())
        first = round(cut.start_s * fps)
        last = min(round(cut.end_s * fps), len(reader))
        if last - first < 2:  # noqa: PLR2004 (motion needs two frames)
            return None
        probes = np.linspace(first, last - 1, self._detection_frames).round().astype(int)
        boxes = [self._estimator.people(frame) for frame in reader.get_batch(probes).asnumpy()]
        signer = signer_box(boxes, self._margin)
        if signer is None:
            return None

        crops = np.concatenate(list(self._cropped_chunks(reader, first, last, signer.crop)))
        inside = signer.union_in_crop(self._size)
        video = root / "clips" / f"{cut.clip_id}.mp4"
        pose = root / "poses" / f"{cut.clip_id}.npz"
        _write_video(video, crops, fps)

        selected = np.asarray(select_frames(local_motion(crops)))
        centre = max(0, len(crops) // 2 - FRAMES_PER_CLIP // 2)
        contiguous = np.clip(np.arange(centre, centre + FRAMES_PER_CLIP), 0, len(crops) - 1)
        pose.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "pose": np.stack([self._estimator.in_box(crops[i], inside) for i in selected]).astype(
                np.float16
            ),
            "idx": selected.astype(np.int32),
            "total": np.asarray(len(crops), dtype=np.int32),
        }
        if self._contiguous:
            arrays["contiguous_pose"] = np.stack(
                [self._estimator.in_box(crops[i], inside) for i in contiguous]
            ).astype(np.float16)
            arrays["contiguous_idx"] = contiguous.astype(np.int32)
        np.savez_compressed(pose, **arrays)  # type: ignore[arg-type]  # numpy stubs: **kwds
        return MaterializedRecord(
            cut.clip_id, str(video), str(pose), fps, first, len(crops), signer.crop.tolist()
        )

    def _cropped_chunks(
        self, reader: "decord.VideoReader", first: int, last: int, box: np.ndarray
    ) -> Iterator[np.ndarray]:
        for start in range(first, last, DECODE_CHUNK):
            indices = list(range(start, min(start + DECODE_CHUNK, last)))
            yield crop_square(reader.get_batch(indices).asnumpy(), box, self._size)


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = frames.shape[1]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (size, size))
    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    finally:
        writer.release()


class MaterializedIndex:
    """JSON-lines index of a test root, one file per shard so parallel jobs never share one."""

    def __init__(self, root: Path, shard: Shard | None = None) -> None:
        self.root = root
        name = "index.jsonl" if shard is None else f"index.{shard.label}.jsonl"
        self.path = root / name

    def records(self) -> list[MaterializedRecord]:
        """Every record of every shard."""
        records: list[MaterializedRecord] = []
        for path in sorted(self.root.glob("index*.jsonl")):
            with path.open(encoding="utf-8") as stream:
                records.extend(
                    MaterializedRecord(**json.loads(line)) for line in stream if line.strip()
                )
        return records

    def append(self, record: MaterializedRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record)) + "\n")
