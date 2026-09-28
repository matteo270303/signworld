"""PC5 (§4.12.1): is Uni-Sign's pose encoder a sound target for the physical level?

The encoder is kept frozen and probed with linear heads, as ``D_pose`` would read it:

* **content**: how much of each articulator's keypoints a linear head recovers from ``s_{t,a}``
  (the document requires R² ≥ 0.7), against a randomly initialised encoder of the same
  architecture, which separates what pre-training adds from what the architecture gives;
* **domain shift**: the same with batch-norm statistics re-estimated on our clips, since the
  released ones come from Chinese Sign Language news;
* **dynamics**: how well consecutive features recover keypoint velocities;
* **coverage**: per sign language, and left against right hand, which share all weights;
* **frame sampling**: the 64 motion-selected frames against 64 consecutive frames, since the
  temporal convolutions were trained on a different sampling;
* **geometry**: the starting point of SIGReg on each articulator.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import torch

from ..metrics.geometry import effective_rank, isoscore
from ..metrics.probes import ProbeTask, video_split
from ..metrics.sigreg import SIGReg, random_directions
from ..models.unisign_pose import UniSignPoseEncoder
from ..pose.wholebody import Articulator, PoseTrack, unisign_parts

MIN_R2: Final = 0.7
"""PC5 threshold on the linear read-out of the keypoints from the frozen representation."""
_ROOT: Final = {Articulator.LEFT_HAND: 0, Articulator.RIGHT_HAND: 0, Articulator.FACE: -1}


@dataclass(frozen=True, slots=True)
class PosedClip:
    clip_id: str
    video_id: str
    sign_language: str
    pose: Path


@dataclass(frozen=True, slots=True)
class Batch:
    """Inputs and targets of many clips: parts as (clips, frames, keypoints, 3)."""

    parts: dict[Articulator, np.ndarray]
    videos: np.ndarray
    languages: np.ndarray

    @classmethod
    def load(cls, clips: Sequence[PosedClip], prefix: str = "") -> "Batch":
        per_clip = [unisign_parts(PoseTrack.load(clip.pose, prefix)) for clip in clips]
        return cls(
            parts={part: np.stack([parts[part] for parts in per_clip]) for part in Articulator},
            videos=np.array([clip.video_id for clip in clips]),
            languages=np.array([clip.sign_language for clip in clips]),
        )

    def targets(self, part: Articulator) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (clips, frames, k, 2) without the constant root, and their 0/1 weights."""
        values = self.parts[part]
        if part in _ROOT:
            values = np.delete(values, _ROOT[part] % values.shape[2], axis=2)
        return values[..., :2], (values[..., 2] > 0).astype(np.float64)


def encode(
    encoder: UniSignPoseEncoder, batch: Batch, size: int = 64
) -> dict[Articulator, np.ndarray]:
    """Frozen features (clips, frames, 256) of every articulator."""
    outputs: dict[Articulator, list[np.ndarray]] = {part: [] for part in Articulator}
    clips = len(batch.videos)
    with torch.no_grad():
        for start in range(0, clips, size):
            inputs = {
                part: torch.from_numpy(values[start : start + size]).float()
                for part, values in batch.parts.items()
            }
            for part, features in encoder(inputs).items():
                outputs[part].append(features.numpy())
    return {part: np.concatenate(chunks) for part, chunks in outputs.items()}


def adapt_batch_norm(encoder: UniSignPoseEncoder, batch: Batch, size: int = 64) -> None:
    """Replace the released batch-norm statistics with those of ``batch`` (weights untouched)."""
    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over the whole pass
    encoder.train()
    encode(encoder, batch, size)
    encoder.eval()


def random_like(pretrained: UniSignPoseEncoder, seed: int) -> UniSignPoseEncoder:
    """Same architecture and graph adjacency, freshly initialised weights."""
    torch.manual_seed(seed)
    encoder = UniSignPoseEncoder()
    adjacency = {k: v for k, v in pretrained.state_dict().items() if k.endswith(".A")}
    encoder.load_state_dict(adjacency, strict=False)
    return encoder.eval()


def _geometry(
    features: np.ndarray, sigreg: SIGReg, generator: torch.Generator, seed: int
) -> tuple[float, float, float, float]:
    """Effective rank, IsoScore, SIGReg and its Gaussian reference of per-frame features."""
    rows = np.random.default_rng(seed).choice(len(features), min(16_384, len(features)))
    sample = torch.from_numpy(features[rows]).float()
    centred = sample - sample.mean(dim=0)
    scaled = centred * (centred.shape[1] / centred.pow(2).sum(dim=1).mean()).sqrt()
    directions = random_directions(centred.shape[1], 256, generator=generator)
    reference = torch.randn(len(scaled), scaled.shape[1], generator=generator)
    return (
        effective_rank(centred),
        isoscore(centred),
        sigreg(scaled, directions).item(),
        sigreg(reference, directions).item(),
    )


@dataclass(frozen=True, slots=True)
class ArticulatorResult:
    r2_pretrained: float
    r2_adapted_batch_norm: float
    r2_random_init: float
    r2_velocity_pretrained: float
    r2_velocity_random_init: float
    r2_contiguous_same_probe: float
    """Probe fitted on selected frames, applied to consecutive frames."""
    r2_contiguous_own_probe: float
    r2_by_sign_language: dict[str, float]
    visible_fraction: float
    effective_rank: float
    isoscore: float
    sigreg: float
    sigreg_gaussian_reference: float


@dataclass(frozen=True, slots=True)
class UniSignReport:
    clips: int
    test_clips: int
    articulators: dict[str, ArticulatorResult]
    passed: bool
    """Every articulator reaches R² ≥ 0.7 with the released weights."""


def unisign_report(
    clips: Sequence[PosedClip],
    checkpoint: Path,
    seed: int = 0,
    progress: Callable[[str], None] = lambda message: None,
) -> UniSignReport:
    selected = Batch.load(clips)
    contiguous = Batch.load(clips, prefix="contiguous_")
    frames = selected.parts[Articulator.BODY].shape[1]
    test = video_split(selected.videos)

    pretrained = UniSignPoseEncoder.from_checkpoint(checkpoint).eval()
    progress("encoding with the released weights")
    released = encode(pretrained, selected)
    released_contiguous = encode(pretrained, contiguous)
    progress("encoding with a random initialisation")
    random_features = encode(random_like(pretrained, seed), selected)
    progress("encoding with batch-norm statistics of our clips")
    adapted_encoder = UniSignPoseEncoder.from_checkpoint(checkpoint)
    adapt_batch_norm(
        adapted_encoder,
        Batch(
            {part: values[~test] for part, values in selected.parts.items()},
            selected.videos[~test],
            selected.languages[~test],
        ),
    )
    adapted = encode(adapted_encoder, selected)

    results = {}
    sigreg = SIGReg()
    generator = torch.Generator().manual_seed(seed)
    frame_languages = np.repeat(selected.languages, frames)
    for part in Articulator:
        progress(f"probing {part}")
        targets, weights = selected.targets(part)
        contiguous_targets, contiguous_weights = contiguous.targets(part)

        frame_task = partial(
            ProbeTask.of_frames,
            targets=targets,
            weights=weights,
            test_clips=test,
            videos=selected.videos,
        )
        velocity_task = partial(
            ProbeTask.of_velocities,
            targets=targets,
            weights=weights,
            test_clips=test,
            videos=selected.videos,
        )

        main = frame_task(released[part])
        probe = main.fit()
        on_contiguous = ProbeTask.of_frames(
            released_contiguous[part],
            contiguous_targets,
            contiguous_weights,
            test,
            selected.videos,
        )
        rank, iso, value, reference = _geometry(main.features, sigreg, generator, seed)
        results[str(part)] = ArticulatorResult(
            r2_pretrained=main.score(probe),
            r2_adapted_batch_norm=frame_task(adapted[part]).held_out_r2(),
            r2_random_init=frame_task(random_features[part]).held_out_r2(),
            r2_velocity_pretrained=velocity_task(released[part]).held_out_r2(),
            r2_velocity_random_init=velocity_task(random_features[part]).held_out_r2(),
            r2_contiguous_same_probe=on_contiguous.score(probe),
            r2_contiguous_own_probe=on_contiguous.held_out_r2(),
            r2_by_sign_language={
                str(language): main.score(probe, frame_languages == language)
                for language in np.unique(selected.languages)
            },
            visible_fraction=float(main.weights.mean()),
            effective_rank=rank,
            isoscore=iso,
            sigreg=value,
            sigreg_gaussian_reference=reference,
        )
    return UniSignReport(
        clips=len(clips),
        test_clips=int(test.sum()),
        articulators=results,
        passed=all(result.r2_pretrained >= MIN_R2 for result in results.values()),
    )
