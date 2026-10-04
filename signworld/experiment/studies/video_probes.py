"""Frozen video features and the preliminary controls that read them (§4.12.1, §4.13.1).

Features are extracted once per *run* (an encoder, a crop resolution and a choice of frames),
in shards that run on separate GPU nodes, on the read-out clips of the pose-teacher comparison
(the same seeded 8 000 clips, split by the same videos). Per clip two things are kept: the mean
token, and for each step the mean token inside each articulator's box.

The read-outs then answer four questions on the CPU:

* **PC2** - the floor to beat: random features, the clip duration alone, and a ridge regression
  from frozen features to the caption embeddings, all scored in both retrieval directions
  (R@k, Precision@k, Recall@k, MRR, MedR); the floor is the best R@1 of each direction;
* **PC3** - which encoder: hand keypoints from the hand boxes (R²), ridge to text (the mean R@1
  of the two directions), and the phonological probe, which needs isolated-sign datasets not
  acquired yet;
* **PC4** - which resolution: 384 only if the hand read-out gains more than 0.05 of R²;
* **collaudo, selected against contiguous frames** - the motion-guided selection must not read
  the hands worse than 64 consecutive frames (tolerance 0.05 of R²).
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

from signworld.data.corpus.materialize import MaterializedRecord, crop_square
from signworld.data.pose.boxes import step_boxes
from signworld.data.pose.tokens import articulator_columns
from signworld.data.pose.wholebody import Articulator, PoseTrack
from signworld.data.video import ClipReader
from signworld.experiment.collaudo.analysis import VideoRun
from signworld.metrics.directions import Bidirectional
from signworld.metrics.probes import ProbeTask, video_split
from signworld.metrics.retrieval import (
    Interval,
    bidirectional_measures,
    bootstrap_recall,
    both_ways,
    grouped_relevance,
    recall_at_k,
)
from signworld.models.encoders.video_encoders import FrozenVideoEncoder, box_pool

from .pose_teachers import LabelledClip, PoseCorpus

logger = logging.getLogger(__name__)

HANDS: Final = (Articulator.LEFT_HAND, Articulator.RIGHT_HAND)
TEXT_PENALTIES: Final = (1e-2, 1e-1, 1.0, 10.0, 100.0)
CONTIGUOUS: Final = "contiguous_"


# --------------------------------------------------------------------------- extraction


@dataclass(frozen=True, slots=True)
class ClipFeatures:
    clip_ids: list[str]
    clip: np.ndarray
    """(clips, D) mean token."""
    parts: np.ndarray
    """(clips, 32, 4, D) mean token inside each articulator's box, per step."""
    visible: np.ndarray
    """(clips, 32, 4) whether the box existed in either frame of the step."""

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.tmp.npz")
        np.savez(
            temporary,
            clip_ids=np.array(self.clip_ids),
            clip=self.clip.astype(np.float16),
            parts=self.parts.astype(np.float16),
            visible=self.visible,
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, directory: Path) -> "ClipFeatures":
        """Every shard of a run, concatenated."""
        shards = sorted(directory.glob("shard-*.npz"))
        if not shards:
            raise FileNotFoundError(f"{directory}: no feature shards; run video-features first")
        loaded = [np.load(shard) for shard in shards]
        return cls(
            [str(c) for part in loaded for c in part["clip_ids"]],
            np.concatenate([part["clip"] for part in loaded]).astype(np.float32),
            np.concatenate([part["parts"] for part in loaded]).astype(np.float32),
            np.concatenate([part["visible"] for part in loaded]),
        )

    def select(self, clip_ids: Sequence[str]) -> "ClipFeatures":
        row = {clip: position for position, clip in enumerate(self.clip_ids)}
        rows = np.array([row[clip] for clip in clip_ids], dtype=np.int64)
        return ClipFeatures(list(clip_ids), self.clip[rows], self.parts[rows], self.visible[rows])


class FrameSource:
    """The 64 frames of a clip for one run: the stored crop, or a new crop from the source."""

    def __init__(self, run: VideoRun, stored_size: int, raw_videos: Path) -> None:
        self._run = run
        self._recut = run.size != stored_size
        self._raw = raw_videos

    @property
    def prefix(self) -> str:
        return CONTIGUOUS if self._run.frames == "contiguous" else ""

    def frames(self, record: MaterializedRecord, video_id: str, track: PoseTrack) -> np.ndarray:
        indices = track.frame_indices
        if not self._recut:
            return ClipReader(Path(record.video)).frames(indices)
        import decord  # noqa: PLC0415 (only runs that re-cut need the source videos)

        reader = decord.VideoReader(str(self._raw / f"{video_id}.mp4"))
        source = np.clip(record.first_frame + indices, 0, len(reader) - 1)
        batch = reader.get_batch(source.tolist()).asnumpy()
        return crop_square(batch, np.asarray(record.box), self._run.size)


def extract(
    clips: Sequence[LabelledClip],
    records: dict[str, MaterializedRecord],
    source: FrameSource,
    encoder: FrozenVideoEncoder,
    batch_size: int,
    progress: Callable[[str], None] = logger.info,
) -> ClipFeatures:
    """Features of ``clips``, ``batch_size`` clips per forward pass."""
    ids, clip_means, parts, visible = [], [], [], []
    for start in range(0, len(clips), batch_size):
        batch = clips[start : start + batch_size]
        tracks = [PoseTrack.load(clip.pose, source.prefix) for clip in batch]
        frames = np.stack(
            [
                source.frames(records[clip.clip_id], clip.video_id, track)
                for clip, track in zip(batch, tracks, strict=True)
            ]
        )
        grids = encoder.tokens(torch.from_numpy(frames))
        for clip, track, grid in zip(batch, tracks, grids, strict=True):
            boxes, shown = step_boxes(track)
            ids.append(clip.clip_id)
            clip_means.append(grid.mean(dim=(0, 1, 2)).cpu().numpy())
            parts.append(box_pool(grid, boxes, shown).cpu().numpy())
            visible.append(shown)
        done = start + len(batch)
        if done % (25 * batch_size) < batch_size or done == len(clips):
            progress(f"{done}/{len(clips)} clips")
    return ClipFeatures(ids, np.stack(clip_means), np.stack(parts), np.stack(visible))


# --------------------------------------------------------------------------- read-outs


@dataclass(frozen=True, slots=True)
class HandScores:
    r2_position: float
    r2_velocity: float


@dataclass(frozen=True, slots=True)
class TextScores:
    t2v_r1: Interval
    """Text-to-video R@1 with a bootstrap interval over the queries."""
    v2t_r1: Interval
    """Video-to-text R@1, likewise."""
    measures: dict[str, float]
    """R@k, Precision@k, Recall@k, MRR and MedR of both directions, keyed as validation logs
    them (``t2v_r5``, ``v2t_mrr``, …)."""
    penalty: float
    gallery: int

    @property
    def r1(self) -> Bidirectional:
        return Bidirectional(self.t2v_r1.estimate, self.v2t_r1.estimate)


@dataclass(frozen=True, slots=True)
class RunScores:
    run: str
    clips: int
    hands: dict[str, HandScores]
    text: TextScores

    @property
    def hand_r2(self) -> float:
        """Mean position R² of the two hands: the number PC3 and PC4 compare."""
        return float(np.mean([score.r2_position for score in self.hands.values()]))


def hand_scores(features: ClipFeatures, corpus: PoseCorpus) -> dict[str, HandScores]:
    """Keypoints of each hand read from the tokens in its box, on held-out videos."""
    test = video_split(corpus.videos)
    scores = {}
    for part in HANDS:
        column = list(Articulator).index(part)
        columns = articulator_columns(part)
        values = features.parts[:, :, column]
        weights = corpus.weights[:, :, columns] * features.visible[:, :, column, None]
        targets = corpus.positions[:, :, columns]
        scores[str(part)] = HandScores(
            ProbeTask.of_frames(values, targets, weights, test, corpus.videos).held_out_r2(),
            ProbeTask.of_velocities(values, targets, weights, test, corpus.videos).held_out_r2(),
        )
    return scores


@dataclass(frozen=True, slots=True)
class TextTargets:
    """Caption embeddings of the read-out clips, whole (never truncated, §4.4.4), centred."""

    embeddings: np.ndarray
    """(clips, d)."""
    groups: np.ndarray
    """(clips,) row of the caption: clips with the same caption are equally relevant."""

    @classmethod
    def of(
        cls,
        rows: np.ndarray,
        table: np.ndarray,
        languages: np.ndarray,
        train: np.ndarray,
    ) -> "TextTargets":
        """Unit-length rows, each written language centred on the means of its training clips
        (PC1 centres per language too)."""
        values = table[rows].astype(np.float64)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        for language in np.unique(languages):
            own = languages == language
            fitting = own & train if (own & train).any() else own
            values[own] -= values[fitting].mean(axis=0)
        return cls(values, rows)


def _standardise(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean, scale = values[train].mean(axis=0), values[train].std(axis=0) + 1e-6
    standard: np.ndarray = (values - mean) / scale
    return standard


def _ridge(design: np.ndarray, targets: np.ndarray, penalties: Sequence[float]) -> list[np.ndarray]:
    design = np.concatenate([design, np.ones((len(design), 1))], axis=1)
    gram = design.T @ design
    right = design.T @ targets
    identity = np.eye(len(gram))
    identity[-1, -1] = 0.0
    return [np.linalg.solve(gram + p * len(design) * identity, right) for p in penalties]


def _predict(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    predicted: np.ndarray = np.concatenate([design, np.ones((len(design), 1))], axis=1) @ weights
    return predicted


def _similarity(texts: np.ndarray, videos: np.ndarray) -> torch.Tensor:
    t = torch.nn.functional.normalize(torch.from_numpy(texts).float(), dim=1)
    v = torch.nn.functional.normalize(torch.from_numpy(videos).float(), dim=1)
    return t @ v.T


def text_scores(
    features: np.ndarray, targets: TextTargets, videos: np.ndarray, seed: int = 0
) -> TextScores:
    """Ridge from clip features to caption embeddings; retrieval among the held-out clips.

    The penalty is the one with the best mean R@1 of the two directions, the metric that
    decides, on a validation share of the training videos. Text to video queries the held-out
    clips with their captions, video to text the other way round, and every clip with the same
    caption counts as a match.
    """
    test = video_split(videos)
    # The same hash orders both splits, so the validation videos are the next band after test.
    validation = video_split(videos, 0.35) & ~test
    fitting = ~test & ~validation
    design = _standardise(features.astype(np.float64), ~test)
    candidates = _ridge(design[fitting], targets.embeddings[fitting], TEXT_PENALTIES)
    groups = targets.groups
    relevance = grouped_relevance(groups[validation].tolist(), groups[validation].tolist())

    def decision(index: int) -> float:
        similarity = _similarity(
            targets.embeddings[validation], _predict(design[validation], candidates[index])
        )
        recalls = [recall_at_k(s, r, (1,))[1] for s, r in both_ways(similarity, relevance).values()]
        return float(np.mean(recalls))

    chosen = max(range(len(TEXT_PENALTIES)), key=decision)
    weights = _ridge(design[~test], targets.embeddings[~test], (TEXT_PENALTIES[chosen],))[0]
    similarity = _similarity(targets.embeddings[test], _predict(design[test], weights))
    relevance = grouped_relevance(groups[test].tolist(), groups[test].tolist())
    generator = torch.Generator().manual_seed(seed)
    intervals = {
        direction: bootstrap_recall(s, r, 1, generator=generator)
        for direction, (s, r) in both_ways(similarity, relevance).items()
    }
    return TextScores(
        t2v_r1=intervals["t2v"],
        v2t_r1=intervals["v2t"],
        measures=bidirectional_measures(similarity, relevance),
        penalty=TEXT_PENALTIES[chosen],
        gallery=int(test.sum()),
    )


def run_scores(
    name: str,
    features: ClipFeatures,
    corpus: PoseCorpus,
    targets: TextTargets,
    keypoint_clips: int,
) -> RunScores:
    """Hand read-outs on the first ``keypoint_clips`` clips, text read-out on all of them."""
    count = min(keypoint_clips, len(corpus.clips))
    rows = np.arange(count)
    hands = hand_scores(features.select(features.clip_ids[:count]), corpus.subset(rows))
    return RunScores(
        name,
        len(features.clip_ids),
        hands,
        text_scores(features.clip, targets, corpus.videos),
    )


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """PC2: retrieval, both ways, that any trained model must beat."""

    gallery: int
    chance_r1: float
    """Expected R@1 of a random ranking, ties to duplicate captions included; the same in both
    directions, which share the groups of identical captions."""
    random_features: TextScores
    duration_only: TextScores
    """[Our reading of «solo statistiche della didascalia»] The only thing a caption shares
    with an unseen clip without looking at it is length, so the video side is its duration."""
    frozen_features: dict[str, TextScores]
    floor_r1: Bidirectional
    """The best R@1 of the above in each direction: the minimum a trained model has to exceed,
    the ridge baseline of the gate and of stop F3."""


def baseline_report(
    durations: np.ndarray,
    targets: TextTargets,
    videos: np.ndarray,
    frozen: dict[str, TextScores],
    seed: int = 0,
) -> BaselineReport:
    rng = np.random.default_rng(seed)
    random_scores = text_scores(rng.standard_normal((len(videos), 64)), targets, videos, seed)
    duration = np.stack([durations, np.log(np.maximum(durations, 1e-3))], axis=1)
    duration_scores = text_scores(duration, targets, videos, seed)
    test = video_split(videos)
    groups = targets.groups[test]
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    chance = float(np.mean(counts[inverse] / len(groups)))
    every = [random_scores.r1, duration_scores.r1, *(s.r1 for s in frozen.values())]
    floor = Bidirectional(max(r.t2v for r in every), max(r.v2t for r in every))
    return BaselineReport(int(test.sum()), chance, random_scores, duration_scores, frozen, floor)


@dataclass(frozen=True, slots=True)
class EncoderChoice:
    """PC3: the encoder that wins at least two of the three probes; the default on a tie."""

    hand_r2: dict[str, float]
    text_r1: dict[str, float]
    """Mean R@1 of the two directions of each encoder's ridge to text."""
    phonology: str
    wins: dict[str, int]
    chosen: str
    rule: str


def choose_encoder(scores: Sequence[RunScores], default: str) -> EncoderChoice:
    hand = {s.run: s.hand_r2 for s in scores}
    text = {s.run: s.text.r1.mean for s in scores}
    wins = dict.fromkeys(hand, 0)
    for probe in (hand, text):
        wins[max(probe, key=lambda run: probe[run])] += 1
    # With the phonological probe missing, two of three means winning both remaining probes.
    leaders = [run for run, count in wins.items() if count >= 2]  # noqa: PLR2004
    chosen = leaders[0] if leaders else default
    return EncoderChoice(
        hand_r2=hand,
        text_r1=text,
        phonology=(
            "not computed: no isolated-sign dataset (Lett. 74-76) is acquired; with two probes "
            "a winner must take both"
        ),
        wins=wins,
        chosen=chosen,
        rule="best on 2 of 3 probes; tie -> V-JEPA 2.1-L (§4.12.1)",
    )


@dataclass(frozen=True, slots=True)
class ResolutionChoice:
    """PC4: 384 only if the hand read-out gains more than ``threshold`` of R²."""

    low: str
    high: str
    hand_r2_low: float
    hand_r2_high: float
    gain: float
    threshold: float
    chosen: int


def choose_resolution(
    low: RunScores, high: RunScores, low_size: int, high_size: int, threshold: float
) -> ResolutionChoice:
    gain = high.hand_r2 - low.hand_r2
    return ResolutionChoice(
        low.run,
        high.run,
        low.hand_r2,
        high.hand_r2,
        gain,
        threshold,
        high_size if gain > threshold else low_size,
    )


@dataclass(frozen=True, slots=True)
class FrameSelectionResult:
    """Collaudo: selected frames read the hands at least as well as contiguous ones, less
    ``tolerance``."""

    clips: int
    selected: dict[str, HandScores]
    contiguous: dict[str, HandScores]
    selected_r2: float
    contiguous_r2: float
    tolerance: float
    passed: bool


def frame_selection(
    selected: ClipFeatures,
    selected_corpus: PoseCorpus,
    contiguous: ClipFeatures,
    contiguous_corpus: PoseCorpus,
    tolerance: float,
) -> FrameSelectionResult:
    """Both sides on the same clips, each read against the poses of its own frames."""
    chosen = hand_scores(selected, selected_corpus)
    consecutive = hand_scores(contiguous, contiguous_corpus)
    mean_selected = float(np.mean([s.r2_position for s in chosen.values()]))
    mean_contiguous = float(np.mean([s.r2_position for s in consecutive.values()]))
    return FrameSelectionResult(
        len(selected.clip_ids),
        chosen,
        consecutive,
        mean_selected,
        mean_contiguous,
        tolerance,
        mean_selected >= mean_contiguous - tolerance,
    )
