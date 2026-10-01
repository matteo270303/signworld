"""The toy world of the collaudo (§4.13.1): synthetic clips that exercise the whole chain.

Four coloured shapes, one per «articulator», move with inertia and gravity and bounce losing
height. Their «pose» is points on their outlines, written as the 133 whole-body keypoints the
real pipeline reads (the shapes stand where the body, the hands and the face would be); the
captions come from fixed templates («the red circle falls and bounces, the blue square moves
left, ...»). The clips are written in the format of the materialisation (``clips/*.mp4``,
``poses/*.npz``, the index and the manifest), so they go through the real data loader, boxes,
tokens, masks, read-out, losses and evaluation.

Every combination of motions is its own «channel»: the held-out channel split is then made of
combinations never seen in training, as the collaudo asks (R@1 > 20 % on unseen combinations).
Its other criteria are read by the ordinary diagnostics: the keypoints decoded from the masked
shapes against interpolation (``keypoint_margin``), and the plausibility tests (time reversed
and skipped steps raise ``Ē_fis``, a colour change does not).
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pyarrow as pa

from signworld.data.corpus.manifest import Clip, write_manifest
from signworld.data.corpus.materialize import MaterializedIndex, MaterializedRecord
from signworld.data.pose.frames import FRAMES_PER_CLIP, local_motion, select_frames
from signworld.data.pose.wholebody import (
    ARTICULATOR_INDICES,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    WHOLEBODY_KEYPOINTS,
    Articulator,
)

FPS: Final = 25.0
SCORE: Final = 5.0
"""A confident RTMW score (the pipeline keeps keypoints above 1.0)."""
SHOULDERS: Final = ((0.6, 0.55), (0.4, 0.55))
"""Fixed invisible reference points: the shoulder frame of the normalisation (left, right)."""


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    colour: str
    bgr: tuple[int, int, int]
    part: Articulator
    kind: str


SHAPES: Final = (
    Shape("square", "blue", (200, 80, 40), Articulator.BODY, "square"),
    Shape("circle", "red", (40, 40, 220), Articulator.LEFT_HAND, "circle"),
    Shape("triangle", "green", (60, 190, 60), Articulator.RIGHT_HAND, "triangle"),
    Shape("star", "yellow", (40, 220, 230), Articulator.FACE, "star"),
)
MOTIONS: Final = ("falls and bounces", "moves left", "moves right", "rises", "stays still")
COMBINATIONS: Final = tuple(itertools.product(range(len(MOTIONS)), repeat=len(SHAPES)))
"""5⁴ = 625 combinations of one motion per shape: the «channels» of the toy world."""


def caption(motions: tuple[int, ...]) -> str:
    parts = [f"the {s.colour} {s.name} {MOTIONS[m]}" for s, m in zip(SHAPES, motions, strict=True)]
    return ", ".join(parts)


def _outline(kind: str, centre: np.ndarray, radius: float, count: int) -> np.ndarray:
    """(count, 2) points spread along the outline of a shape, in frame fractions."""
    t = np.linspace(0, 1, count, endpoint=False)
    if kind == "circle":
        angle = 2 * math.pi * t
        return centre + radius * np.stack([np.cos(angle), np.sin(angle)], 1)
    star = [
        (math.cos(a) * (1.0 if i % 2 == 0 else 0.45), math.sin(a) * (1.0 if i % 2 == 0 else 0.45))
        for i, a in enumerate(np.linspace(-math.pi / 2, 1.5 * math.pi, 10, endpoint=False))
    ]
    corners: dict[str, list[tuple[float, float]]] = {
        "square": [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        "triangle": [(0.0, -1.15), (1.0, 0.8), (-1.0, 0.8)],
        "star": star,
    }
    outline = corners[kind]
    polygon = np.asarray(outline + outline[:1], dtype=np.float64)
    lengths = np.linalg.norm(np.diff(polygon, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(lengths)]) / lengths.sum()
    points = np.stack([np.interp(t, cumulative, polygon[:, axis]) for axis in range(2)], 1)
    return centre + radius * points


def _trajectory(motion: int, frames: int, rng: np.random.Generator, band: float) -> np.ndarray:
    """(frames, 2) centres: inertia, gravity and bounces that lose height, walls that reflect."""
    position = np.array([band, rng.uniform(0.25, 0.75)])
    velocity = np.zeros(2)
    gravity = np.zeros(2)
    if MOTIONS[motion] == "falls and bounces":
        position[1] = rng.uniform(0.15, 0.3)
        gravity[1] = 0.0015
    elif MOTIONS[motion] == "moves left":
        velocity[0] = -rng.uniform(0.006, 0.01)
    elif MOTIONS[motion] == "moves right":
        velocity[0] = rng.uniform(0.006, 0.01)
    elif MOTIONS[motion] == "rises":
        position[1] = rng.uniform(0.7, 0.85)
        velocity[1] = -rng.uniform(0.006, 0.01)
    centres = np.zeros((frames, 2))
    for frame in range(frames):
        velocity = velocity + gravity
        position = position + velocity
        for axis, (low, high) in enumerate(((0.12, 0.88), (0.12, 0.88))):
            if not low <= position[axis] <= high:
                position[axis] = np.clip(position[axis], low, high)
                velocity[axis] = -velocity[axis] * (0.7 if axis == 1 else 1.0)  # floor loses height
        centres[frame] = position
    return centres


@dataclass(frozen=True, slots=True)
class ToyClip:
    clip_id: str
    motions: tuple[int, ...]
    seed: int

    @property
    def channel(self) -> str:
        return "combo" + "".join(map(str, self.motions))


def toy_clips(count: int, seed: int = 0) -> list[ToyClip]:
    """The clips of a toy world: combinations in turn, each clip with its own seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(COMBINATIONS))
    return [
        ToyClip(
            f"toy{index:06d}", COMBINATIONS[order[index % len(order)]], seed * 1_000_003 + index
        )
        for index in range(count)
    ]


def render(
    clip: ToyClip, size: int = 256, frames: int = FRAMES_PER_CLIP
) -> tuple[np.ndarray, np.ndarray]:
    """(frames, size, size, 3) uint8 RGB and (frames, 133, 3) keypoints (x, y, score)."""
    rng = np.random.default_rng(clip.seed)
    video = np.full((frames, size, size, 3), 30, dtype=np.uint8)
    pose = np.zeros((frames, WHOLEBODY_KEYPOINTS, 3), dtype=np.float32)
    for index, joint in enumerate((LEFT_SHOULDER, RIGHT_SHOULDER)):
        pose[:, joint] = (*SHOULDERS[index], SCORE)
    bands = (0.2, 0.4, 0.6, 0.8)
    for shape, motion, band in zip(SHAPES, clip.motions, bands, strict=True):
        radius = rng.uniform(0.06, 0.09)
        centres = _trajectory(motion, frames, rng, band)
        joints = [
            j for j in ARTICULATOR_INDICES[shape.part] if j not in (LEFT_SHOULDER, RIGHT_SHOULDER)
        ]
        for frame, centre in enumerate(centres):
            points = _outline(shape.kind, centre, radius, len(joints))
            pose[frame, joints, :2] = points
            pose[frame, joints, 2] = SCORE
            polygon = _outline(shape.kind, centre, radius, 64) * size
            cv2.fillPoly(video[frame], [polygon.round().astype(np.int32)], shape.bgr[::-1])
    return video, pose


def write_clip(clip: ToyClip, root: Path, size: int = 256) -> MaterializedRecord:
    """The clip in the materialisation's format: the video, and the pose of the 64 frames."""
    frames, pose = render(clip, size)
    video = root / "clips" / f"{clip.clip_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (size, size))
    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    finally:
        writer.release()
    selected = np.asarray(select_frames(local_motion(frames)))
    path = root / "poses" / f"{clip.clip_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pose=pose[selected].astype(np.float16),
        idx=selected.astype(np.int32),
        total=np.asarray(len(frames), dtype=np.int32),
    )
    return MaterializedRecord(
        clip.clip_id, str(video), str(path), FPS, 0, len(frames), [0, 0, size, size]
    )


def toy_manifest(clips: list[ToyClip]) -> list[Clip]:
    duration = FRAMES_PER_CLIP / FPS
    return [
        Clip(
            c.clip_id,
            c.clip_id,
            0.0,
            duration,
            caption(c.motions),
            "en",
            "toy",
            c.channel,
            None,
            True,
        )
        for c in clips
    ]


def build(
    root: Path, count: int, *, shard: int = 0, shards: int = 1, seed: int = 0, size: int = 256
) -> int:
    """Write this shard's clips under ``root`` (resuming) and, from shard 0, the manifest."""
    clips = toy_clips(count, seed)
    if shard == 0:
        write_manifest(toy_manifest(clips), root / "manifest" / "clips.parquet")
    from signworld.data.acquisition.sharding import Shard  # noqa: PLC0415

    part = Shard(shard, shards)
    index = MaterializedIndex(root, part if shards > 1 else None)
    done = {record.clip_id for record in index.records()}
    written = 0
    for clip in clips:
        if part.owns(clip.clip_id) and clip.clip_id not in done:
            index.append(write_clip(clip, root, size))
            written += 1
    return written


def manifest_table(root: Path) -> pa.Table:
    from signworld.data.corpus.manifest import read_manifest  # noqa: PLC0415

    return read_manifest(root / "manifest" / "clips.parquet")
