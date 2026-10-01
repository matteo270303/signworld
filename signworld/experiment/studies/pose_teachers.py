"""Which pose latent should the physical level predict? (§4.4.3)

Four candidates, all producing one feature per step (two frames) and articulator:

* ``kinematic``: shoulder-normalised positions, in-step motion and presence, whitened;
  no learning, so no collapse, but no abstraction either (its position read-out is near
  perfect by construction; the other scores are what matter for it);
* ``unisign``: the released Uni-Sign pose encoder, frozen;
* ``mamp``: a transformer pre-trained here to reconstruct the motion of masked joints;
* ``sjepa``: the same transformer pre-trained here with S-JEPA.

MAMP and S-JEPA see the same training videos, for the same number of epochs. All four are
judged on held-out videos by the same linear read-outs: keypoint positions and velocities per
articulator, which should be high; the channel and the sign language from clip averages, which
should stay near chance (a target that knows the channel invites shortcuts, §3.4; H5 expects
little language information in pose); and the geometry of the latent.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from signworld.data.pose.tokens import STEPS, PoseSequence, articulator_columns
from signworld.data.pose.wholebody import Articulator, PoseTrack
from signworld.experiment.collaudo.analysis import PoseTeacherSettings
from signworld.experiment.collaudo.unisign import Batch, PosedClip, encode
from signworld.metrics.geometry import centered, effective_rank, isoscore
from signworld.metrics.probes import ProbeTask, channel_split, label_accuracy, video_split
from signworld.models.encoders.pose_teachers import (
    MaskedMotionTeacher,
    PoseTeacher,
    SJEPATeacher,
    TeacherShape,
)
from signworld.models.encoders.unisign_pose import UniSignPoseEncoder

logger = logging.getLogger(__name__)

Features = dict[Articulator, np.ndarray]
"""Per articulator, (clips, steps, dimensions)."""


def teacher_shape(settings: PoseTeacherSettings) -> TeacherShape:
    return TeacherShape(
        settings.width,
        settings.depth,
        settings.heads,
        settings.predictor_depth,
        settings.decoder_depth,
        settings.mask_ratio,
    )


@dataclass(frozen=True, slots=True)
class LabelledClip:
    clip_id: str
    video_id: str
    channel: str
    language: str
    pose: Path


@dataclass(frozen=True, slots=True)
class PoseCorpus:
    clips: list[LabelledClip]
    tokens: np.ndarray
    """(clips, 32, 69, 6) teacher inputs."""
    positions: np.ndarray
    """(clips, 32, 69, 2) per-step positions, the read-out targets."""
    weights: np.ndarray
    """(clips, 32, 69) presence of each target."""
    motion: np.ndarray
    """(clips, 32, 69, 2) displacement between the two frames of each step; zero if either
    frame misses the joint."""

    @classmethod
    def load(
        cls, clips: Sequence[LabelledClip], min_score: float, prefix: str = ""
    ) -> "PoseCorpus":
        """Clips without a shoulder reference in any frame are left out.

        ``prefix="contiguous_"`` reads the poses of 64 consecutive frames instead of the
        selected ones (§4.13.1).
        """
        kept, tokens, positions, weights, motion = [], [], [], [], []
        for clip in clips:
            sequence = PoseSequence.from_track(PoseTrack.load(clip.pose, prefix), min_score)
            if sequence is None:
                continue
            kept.append(clip)
            tokens.append(sequence.tokens())
            step_positions, step_weights = sequence.step_positions()
            positions.append(step_positions)
            weights.append(step_weights)
            frames = sequence.joints.reshape(STEPS, 2, -1, 2)
            both = sequence.present.reshape(STEPS, 2, -1).all(axis=1)
            motion.append((frames[:, 1] - frames[:, 0]) * both[..., None])
        return cls(kept, np.stack(tokens), np.stack(positions), np.stack(weights), np.stack(motion))

    def subset(self, rows: np.ndarray) -> "PoseCorpus":
        return PoseCorpus(
            [self.clips[row] for row in rows],
            self.tokens[rows],
            self.positions[rows],
            self.weights[rows],
            self.motion[rows],
        )

    @property
    def videos(self) -> np.ndarray:
        return np.array([clip.video_id for clip in self.clips])

    def labels(self, name: str) -> np.ndarray:
        return np.array([getattr(clip, name) for clip in self.clips])


def kinematic_features(corpus: PoseCorpus, train: np.ndarray) -> Features:
    """Positions, in-step motion and presence per articulator, PCA-whitened on training clips.

    Directions with no variance (a joint never missing, say) are dropped, not amplified.
    """
    features = {}
    for part in Articulator:
        columns = articulator_columns(part)
        raw = np.concatenate(
            [
                corpus.positions[:, :, columns].reshape(*corpus.positions.shape[:2], -1),
                corpus.motion[:, :, columns].reshape(*corpus.positions.shape[:2], -1),
                corpus.weights[:, :, columns],
            ],
            axis=-1,
        ).astype(np.float64)
        rows = raw[train].reshape(-1, raw.shape[-1])
        mean = rows.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(rows - mean, rowvar=False))
        kept = values > 1e-6 * values.max()
        whitening = vectors[:, kept] / np.sqrt(values[kept])
        features[part] = ((raw - mean) @ whitening).astype(np.float32)
    return features


def unisign_features(corpus: PoseCorpus, checkpoint: Path) -> Features:
    """The released Uni-Sign encoder on the same clips, two frames averaged into one step."""
    posed = [PosedClip(c.clip_id, c.video_id, c.language, c.pose) for c in corpus.clips]
    encoder = UniSignPoseEncoder.from_checkpoint(checkpoint).eval()
    per_frame = encode(encoder, Batch.load(posed))
    return {
        part: values.reshape(len(values), STEPS, 2, -1).mean(axis=2)
        for part, values in per_frame.items()
    }


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    epoch: int
    loss: float
    hand_rank: float
    """Effective rank of the right-hand latent on a fixed batch: collapse shows here first."""
    seconds: float


def train_teacher(
    teacher: PoseTeacher,
    tokens: torch.Tensor,
    settings: PoseTeacherSettings,
    device: torch.device,
    progress: Callable[[str], None],
) -> list[TrainingPoint]:
    """AdamW with warm-up and cosine decay; bf16 autocast on GPU."""
    teacher.to(device).train()
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    optimizer = torch.optim.AdamW(
        [p for p in teacher.parameters() if p.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(tokens) // settings.batch_size)
    total = settings.epochs * steps_per_epoch
    warmup = max(1, round(settings.warmup_fraction * total))
    probe = tokens[: settings.batch_size].to(device)
    history = []
    started = time.monotonic()
    step = 0
    for epoch in range(1, settings.epochs + 1):
        order = torch.randperm(len(tokens))
        running = 0.0
        for index in range(steps_per_epoch):
            rate = _learning_rate(step, total, warmup, settings)
            for group in optimizer.param_groups:
                group["lr"] = rate
            batch = tokens[order[index * settings.batch_size : (index + 1) * settings.batch_size]]
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                loss = teacher.loss(batch.to(device, non_blocking=True), generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 3.0)
            optimizer.step()
            step += 1
            teacher.after_step(step / total)
            running += loss.item()
        with torch.no_grad():
            teacher.eval()
            hand = teacher.encode(probe)[:, :, list(Articulator).index(Articulator.RIGHT_HAND)]
            teacher.train()
        point = TrainingPoint(
            epoch,
            running / steps_per_epoch,
            effective_rank(centered(hand.flatten(0, 1).float().cpu())),
            time.monotonic() - started,
        )
        history.append(point)
        progress(
            f"{type(teacher).__name__} epoch {epoch}/{settings.epochs}: loss {point.loss:.4f}, "
            f"right-hand rank {point.hand_rank:.1f}, {point.seconds / 60:.1f} min"
        )
    return history


def _learning_rate(step: int, total: int, warmup: int, settings: PoseTeacherSettings) -> float:
    if step < warmup:
        return settings.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    low, high = settings.min_learning_rate, settings.learning_rate
    return float(low + (high - low) * 0.5 * (1 + np.cos(np.pi * progress)))


@torch.no_grad()
def teacher_features(
    teacher: PoseTeacher, tokens: torch.Tensor, device: torch.device, batch_size: int
) -> Features:
    teacher.to(device).eval()
    chunks = []
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].to(device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            chunks.append(teacher.encode(batch).float().cpu())
    stacked = torch.cat(chunks).numpy()
    return {part: stacked[:, :, column] for column, part in enumerate(Articulator)}


@dataclass(frozen=True, slots=True)
class ArticulatorScores:
    r2_position: float
    r2_velocity: float
    effective_rank: float
    dimensions: int
    isoscore: float


@dataclass(frozen=True, slots=True)
class CandidateResult:
    name: str
    articulators: dict[str, ArticulatorScores]
    channel_accuracy: float
    channel_chance: float
    language_accuracy: float
    language_chance: float
    language_accuracy_unseen_channels: float
    """Language read on channels never seen in fitting; ``nan`` when no language spans two."""
    language_chance_unseen_channels: float
    training: list[TrainingPoint]


def evaluate(
    name: str,
    features: Features,
    corpus: PoseCorpus,
    test: np.ndarray,
    training: list[TrainingPoint] | None = None,
) -> CandidateResult:
    """Read-outs fitted on the training videos of ``corpus`` and scored on its ``test`` clips."""
    videos = corpus.videos
    scores = {}
    for part in Articulator:
        columns = articulator_columns(part)
        values = features[part]
        targets = corpus.positions[:, :, columns]
        weights = corpus.weights[:, :, columns]
        rows = values.reshape(-1, values.shape[-1])
        sample = torch.from_numpy(rows[:: max(1, len(rows) // 16_384)]).double()
        scores[str(part)] = ArticulatorScores(
            r2_position=ProbeTask.of_frames(values, targets, weights, test, videos).held_out_r2(),
            r2_velocity=ProbeTask.of_velocities(
                values, targets, weights, test, videos
            ).held_out_r2(),
            effective_rank=effective_rank(centered(sample)),
            dimensions=int(values.shape[-1]),
            isoscore=isoscore(sample),
        )
    clip_means = np.concatenate([features[part].mean(axis=1) for part in Articulator], axis=1)
    languages, channels = corpus.labels("language"), corpus.labels("channel")
    channel, channel_chance = label_accuracy(clip_means, channels, test)
    language, language_chance = label_accuracy(clip_means, languages, test)
    unseen, unseen_chance = label_accuracy(
        clip_means, languages, channel_split(channels, languages)
    )
    return CandidateResult(
        name,
        scores,
        channel,
        channel_chance,
        language,
        language_chance,
        unseen,
        unseen_chance,
        training or [],
    )


def probe_subset(
    corpus: PoseCorpus, test: np.ndarray, settings: PoseTeacherSettings
) -> tuple[PoseCorpus, np.ndarray, np.ndarray]:
    """The clips of the read-outs, a seeded random subset, with their split and positions."""
    rng = np.random.default_rng(settings.seed)
    chosen = np.sort(rng.permutation(len(corpus.clips))[: settings.probe_clips])
    return corpus.subset(chosen), test[chosen], chosen


@dataclass(frozen=True, slots=True)
class PoseTeacherReport:
    clips: int
    train_clips: int
    test_clips: int
    probe_clips: int
    channels: int
    languages: dict[str, int]
    channels_per_language: dict[str, int]
    """A language on one channel only cannot be read on unseen channels."""
    candidates: list[CandidateResult]


def run(  # noqa: PLR0913 (inputs, where the run happens, and two keyword options)
    clips: Sequence[LabelledClip],
    checkpoint: Path,
    settings: PoseTeacherSettings,
    device: torch.device,
    output: Path,
    *,
    progress: Callable[[str], None] = logger.info,
    reuse: bool = False,
) -> PoseTeacherReport:
    """Train the two teachers, extract every candidate's latent, and run the read-outs.

    The teachers train on every clip of the training videos; the read-outs use a random
    subset of ``probe_clips`` clips, split by the same videos, so no test video is ever seen.
    With ``reuse``, teachers already saved under ``output`` are loaded instead of trained.
    """
    torch.manual_seed(settings.seed)
    corpus = PoseCorpus.load(clips, settings.min_score)
    test = video_split(corpus.videos)
    probe, probe_test, chosen = probe_subset(corpus, test, settings)
    progress(
        f"{len(corpus.clips)} clips ({int((~test).sum())} train, {int(test.sum())} test); "
        f"read-outs on {len(chosen)}"
    )

    candidates = [
        evaluate("kinematic", kinematic_features(probe, ~probe_test), probe, probe_test),
        evaluate("unisign", unisign_features(probe, checkpoint), probe, probe_test),
    ]
    progress("kinematic and unisign: done")
    output.mkdir(parents=True, exist_ok=True)
    train_tokens = torch.from_numpy(corpus.tokens[~test])
    for name, teacher in (
        ("mamp", MaskedMotionTeacher(teacher_shape(settings))),
        (
            "sjepa",
            SJEPATeacher(
                teacher_shape(settings),
                momentum=(settings.ema_start, 1.0),
                centre_rate=settings.centre_rate,
            ),
        ),
    ):
        weights = output / f"{name}.pt"
        if reuse and weights.is_file():
            teacher.load_state_dict(torch.load(weights, map_location="cpu"))
            history: list[TrainingPoint] = []
            progress(f"{name}: reusing {weights}")
        else:
            history = train_teacher(teacher, train_tokens, settings, device, progress)
            torch.save(teacher.state_dict(), weights)
        features = teacher_features(
            teacher, torch.from_numpy(probe.tokens), device, settings.batch_size
        )
        candidates.append(evaluate(name, features, probe, probe_test, history))
        progress(f"{name}: done")

    languages, counts = np.unique(corpus.labels("language"), return_counts=True)
    return PoseTeacherReport(
        clips=len(corpus.clips),
        train_clips=int((~test).sum()),
        test_clips=int(test.sum()),
        probe_clips=len(chosen),
        channels=len(set(corpus.labels("channel"))),
        languages={str(k): int(v) for k, v in zip(languages, counts, strict=True)},
        channels_per_language={
            str(language): len(set(corpus.labels("channel")[corpus.labels("language") == language]))
            for language in languages
        },
        candidates=candidates,
    )
