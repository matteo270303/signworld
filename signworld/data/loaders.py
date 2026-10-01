"""Clips for training: the index of stage 0, the dataset, the sampler and the batches.

Stage 0 (§4.10) materialises every clip once (``corpus.materialize``): the crop at 256
pixels, the 64 frames selected by local motion and their poses. ``build_training_index`` joins
that output with the manifest and the pre-computed caption embeddings, keeps the clips whose
pose has a shoulder reference, and assigns the splits: the sign languages held out whole
(``val_language``), then a share of the channels of every other sign language
(``val_channel``), never single clips (§3.4).

``ClipDataset`` reads one clip: its 64 frames, its pose as S-JEPA tokens, the keypoints and
boxes of every step, and its caption row. In training the same random crop moves frames,
keypoints and boxes (§3.8, P12), drawn from the seed, the epoch and the clip, so a resumed run
sees the same views. ``EpochSampler`` gives every GPU its share of an epoch's permutation and
can start mid-epoch, so a resumed run also keeps the sampler's order.
"""

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from signworld.data.corpus.materialize import MaterializedRecord
from signworld.data.pose.boxes import step_boxes
from signworld.data.pose.tokens import PoseSequence
from signworld.data.pose.wholebody import PoseTrack
from signworld.data.video import ClipReader
from signworld.experiment.train.config import AugmentationSettings, DataSettings
from signworld.models.worldsign.model import WorldSignBatch

from ..metrics.probes import channel_split, video_split
from .augmentation import ClipAugmenter

logger = logging.getLogger(__name__)

TRAIN = "train"
VALIDATION_CHANNEL = "val_channel"
VALIDATION_LANGUAGE = "val_language"
VALIDATION_VIDEO = "val_video"
"""New videos of channels seen in training (§4.13.4)."""
SPLITS = (TRAIN, VALIDATION_CHANNEL, VALIDATION_LANGUAGE, VALIDATION_VIDEO)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("sign_language", pa.string()),
        pa.field("caption_language", pa.string(), nullable=False),
        pa.field("caption_row", pa.int64(), nullable=False),
        pa.field("video_path", pa.string(), nullable=False),
        pa.field("pose_path", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
        pa.field("caption_words", pa.int32(), nullable=False),
    ]
)


def usable_pose(path: Path) -> bool:
    """A pose with a shoulder reference in some frame: S-JEPA's tokens need one."""
    return PoseSequence.from_track(PoseTrack.load(path)) is not None


def assign_splits(
    channels: np.ndarray,
    sign_languages: np.ndarray,
    settings: DataSettings,
    videos: np.ndarray | None = None,
) -> np.ndarray:
    """The split of every clip, by language, channel and video, never by clip (§3.4).

    ``val_language``: the held-out sign languages. ``val_channel``: a share of the channels of
    every other language. ``val_video``: a share of the videos of the remaining channels.
    """
    split = np.full(len(channels), TRAIN, dtype=object)
    held = np.isin(sign_languages, list(settings.held_out_languages))
    split[held] = VALIDATION_LANGUAGE
    rest = np.flatnonzero(~held)
    test = channel_split(channels[rest], sign_languages[rest], settings.validation_fraction)
    split[rest[test]] = VALIDATION_CHANNEL
    if videos is not None:
        seen = (split == TRAIN) & video_split(videos, settings.seen_video_fraction)
        split[seen] = VALIDATION_VIDEO
    return split


def build_training_index(  # noqa: PLR0913 (the inputs and three options)
    manifest: pa.Table,
    records: Sequence[MaterializedRecord],
    caption_rows: pa.Table,
    settings: DataSettings,
    *,
    check_poses: bool = True,
    excluded: frozenset[str] = frozenset(),
    workers: int = 1,
) -> pa.Table:
    """One row per usable clip: materialised, captioned, with a pose reference and a split.

    ``excluded`` are the clips that overlap or repeat a benchmark's validation and test clips
    (§3.9, ``checks.contamination``): they never enter the index. ``workers`` processes read
    the poses in parallel (one read per clip: ~20 ms, ~30 minutes for 94k clips on one).
    """
    materialised = {record.clip_id: record for record in records}
    row_of = dict(
        zip(
            caption_rows.column("clip_id").to_pylist(),
            caption_rows.column("row").to_pylist(),
            strict=True,
        )
    )
    columns = [
        "clip_id",
        "video_id",
        "channel_id",
        "sign_language",
        "caption_language",
        "split",
        "start_s",
        "end_s",
        "caption",
    ]
    candidates = [
        clip
        for clip in manifest.select(columns).to_pylist()
        if clip["clip_id"] in materialised
        and clip["clip_id"] in row_of
        and clip["caption_language"] is not None
        and clip["clip_id"] not in excluded
    ]
    if check_poses:
        paths = [Path(materialised[clip["clip_id"]].pose) for clip in candidates]
        with ProcessPoolExecutor(max(1, workers)) as pool:
            usable = list(pool.map(usable_pose, paths, chunksize=256))
        candidates = [clip for clip, ok in zip(candidates, usable, strict=True) if ok]
    rows = []
    for clip in candidates:
        record = materialised[clip["clip_id"]]
        official = clip.pop("split")
        start, end, caption = clip.pop("start_s"), clip.pop("end_s"), clip.pop("caption")
        rows.append(
            {
                **clip,
                "official_split": official,
                "channel_id": clip["channel_id"] or f"video:{clip['video_id']}",
                "caption_row": row_of[clip["clip_id"]],
                "duration_s": end - start,
                "caption_words": len(caption.split()),
                "video_path": record.video,
                "pose_path": record.pose,
            }
        )
    channels = np.array([row["channel_id"] for row in rows], dtype=object)
    languages = np.array([str(row["sign_language"]) for row in rows], dtype=object)
    videos = np.array([row["video_id"] for row in rows], dtype=object)
    if settings.split_source == "manifest":
        splits = np.array([row["official_split"] for row in rows], dtype=object)
        if any(split is None for split in splits):
            raise ValueError("split_source is manifest but some clips have no split")
    else:
        splits = assign_splits(channels, languages, settings, videos)
    for row, split in zip(rows, splits, strict=True):
        del row["official_split"]
        row["split"] = split
    logger.info("Training index: %d usable clips of %d", len(rows), manifest.num_rows)
    return pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)


def records_from_files(manifest: pa.Table, videos: Path, poses: Path) -> list[MaterializedRecord]:
    """Records of clips stored as flat folders of ``<clip>.mp4`` and ``<clip>.npz``.

    The file name is the clip ID with ``:`` written as ``_`` (as the OpenASL clips were
    stored); a clip whose video or pose is missing is left out.
    """
    records = []
    for clip_id in manifest.column("clip_id").to_pylist():
        stem = clip_id.replace(":", "_")
        video, pose = videos / f"{stem}.mp4", poses / f"{stem}.npz"
        if video.is_file() and pose.is_file():
            records.append(MaterializedRecord(clip_id, str(video), str(pose), 0.0, 0, 0, []))
    return records


def write_index(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def read_index(path: Path, split: str | None = None) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(INDEX_SCHEMA):
        raise ValueError(f"{path}: not a training index; rebuild it")
    return table if split is None else table.filter(pc.equal(table.column("split"), split))


def validation_subset(table: pa.Table, clips: int) -> pa.Table:
    """A fixed subset of ``clips`` rows, the same in every run: the first by clip ID order."""
    order = np.argsort(np.asarray(table.column("clip_id").to_pylist(), dtype=object))
    return table.take(np.sort(order[:clips]))


@dataclass(frozen=True, slots=True)
class ClipSample:
    frames: Tensor
    """(64, H, W, 3) uint8."""
    pose_tokens: Tensor
    """(32, 69, 6) float32."""
    keypoints: Tensor
    """(32, 69, 2) float32, shoulder units."""
    keypoint_weights: Tensor
    """(32, 69) float32."""
    boxes: Tensor
    """(32, 4, 4) float32, frame fractions; NaN where not visible."""
    box_visible: Tensor
    """(32, 4) bool."""
    caption: Tensor
    """(768,) float32."""
    caption_row: int
    language: str
    video_id: str
    clip_id: str = ""
    duration_s: float = 0.0
    caption_words: int = 0
    view: Tensor | None = None
    """(6,) the augmentation still to apply to ``frames`` (``augmentation.apply_views``)."""


class ClipDataset(Dataset[ClipSample]):
    """The clips of one split; keys are ``(index, epoch)`` pairs from ``EpochSampler``."""

    def __init__(
        self,
        table: pa.Table,
        embeddings: np.ndarray,
        *,
        augmentation: AugmentationSettings | None,
        box_threshold: float,
        seed: int = 0,
    ) -> None:
        # Arrow buffers, unlike Python objects, are shared by the loader's worker processes
        # without being copied: a list of 3 M dicts would be duplicated in every worker.
        self.table = table.combine_chunks()
        self.embeddings = embeddings
        self.augmenter = ClipAugmenter(augmentation) if augmentation is not None else None
        self.box_threshold = box_threshold
        self.seed = seed

    def __len__(self) -> int:
        return int(self.table.num_rows)

    def __getitem__(self, key: tuple[int, int] | int) -> ClipSample:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        row = self.table.slice(index, 1).to_pylist()[0]
        track = PoseTrack.load(Path(row["pose_path"]))
        frames = torch.from_numpy(ClipReader(Path(row["video_path"])).frames(track.frame_indices))
        view = None
        if self.augmenter is not None:
            # The keypoints (and so the boxes) move here; the frames move on the GPU with the
            # same view (``apply_views``), where their resampling costs nothing.
            state = int(np.random.SeedSequence([self.seed, epoch, index]).generate_state(1)[0])
            view = self.augmenter.sample(torch.Generator().manual_seed(state))
            moved = self.augmenter.points(torch.from_numpy(track.keypoints), view)
            track = dataclasses.replace(track, keypoints=moved.numpy())
        sequence = PoseSequence.from_track(track)
        if sequence is None:
            raise ValueError(f"{row['clip_id']}: no shoulder reference; rebuild the index")
        positions, weights = sequence.step_positions()
        boxes, visible = step_boxes(track, self.box_threshold)
        return ClipSample(
            frames=frames,
            pose_tokens=torch.from_numpy(sequence.tokens()),
            keypoints=torch.from_numpy(positions),
            keypoint_weights=torch.from_numpy(weights),
            boxes=torch.from_numpy(boxes.astype(np.float32)),
            box_visible=torch.from_numpy(visible),
            caption=torch.from_numpy(np.array(self.embeddings[row["caption_row"]], np.float32)),
            caption_row=int(row["caption_row"]),
            language=row["caption_language"],
            video_id=row["video_id"],
            clip_id=row["clip_id"],
            duration_s=float(row["duration_s"]),
            caption_words=int(row["caption_words"]),
            view=None if view is None else view.as_tensor(),
        )


class Collate:
    """Samples to a ``WorldSignBatch``; languages numbered as the text branch's centring."""

    def __init__(self, languages: Sequence[str]) -> None:
        self.number = {language: index for index, language in enumerate(languages)}

    def __call__(self, samples: Sequence[ClipSample]) -> WorldSignBatch:
        def stack(name: str) -> Tensor:
            return torch.stack([getattr(sample, name) for sample in samples])

        return WorldSignBatch(
            frames=stack("frames"),
            pose_tokens=stack("pose_tokens"),
            keypoints=stack("keypoints"),
            keypoint_weights=stack("keypoint_weights"),
            boxes=stack("boxes"),
            box_visible=stack("box_visible"),
            captions=stack("caption"),
            languages=torch.tensor([self.number[sample.language] for sample in samples]),
            videos=[sample.video_id for sample in samples],
            caption_rows=torch.tensor([sample.caption_row for sample in samples]),
            clip_ids=[sample.clip_id for sample in samples],
            durations=torch.tensor([sample.duration_s for sample in samples]),
            caption_words=torch.tensor([sample.caption_words for sample in samples]),
            views=None if samples[0].view is None else stack("view"),
        )


class EpochSampler(Sampler[tuple[int, int]]):
    """This GPU's share of every epoch, in a permutation fixed by the seed and the epoch.

    Every GPU gets the same number of clips (the remainder of the epoch is dropped), and
    ``start`` skips the clips this GPU has already seen, for a resumed run.
    """

    def __init__(
        self, size: int, rank: int, world_size: int, seed: int, shuffle: bool = True
    ) -> None:
        self.per_rank = size // world_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle = seed, shuffle
        self.size = size
        self.epoch, self.start = 0, 0

    def configure(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
            order = torch.randperm(self.size, generator=generator).tolist()
        else:
            order = list(range(self.size))
        mine = order[: self.per_rank * self.world_size][self.rank :: self.world_size]
        for index in mine[self.start :]:
            yield index, self.epoch

    def __len__(self) -> int:
        return self.per_rank - self.start


def caption_statistics(table: pa.Table, embeddings: np.ndarray) -> tuple[Tensor, list[str]]:
    """Every distinct caption of the training clips with its language, for the centring.

    As PC1 measured it: one row per distinct caption, not per clip.
    """
    rows = table.select(["caption_row", "caption_language"]).to_pylist()
    language_of = {row["caption_row"]: row["caption_language"] for row in rows}
    ordered = sorted(language_of)
    captions = torch.from_numpy(np.asarray(embeddings[ordered], dtype=np.float32))
    return captions, [language_of[row] for row in ordered]


def keypoint_statistics(table: pa.Table, clips: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Step positions and weights of ``clips`` training clips, for the scale of L_anchor."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(table.num_rows, min(clips, table.num_rows), replace=False)
    paths = table.column("pose_path").take(np.sort(chosen)).to_pylist()
    positions, weights = [], []
    for path in paths:
        sequence = PoseSequence.from_track(PoseTrack.load(Path(path)))
        if sequence is None:
            continue
        position, weight = sequence.step_positions()
        positions.append(position)
        weights.append(weight)
    return torch.from_numpy(np.stack(positions)), torch.from_numpy(np.stack(weights))
