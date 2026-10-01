"""How many directions the S-JEPA pose latent really uses, and its fixed whitening (§4.4.3).

The frozen teacher's latent is anisotropic (IsoScore ≈ 0.03 over 256 dimensions), while the
physical level would regress it under SIGReg, which asks for N(0, I). A fixed linear map fitted
once on the training clips removes the anisotropy without changing what the target contains,
but only over directions that carry signal: whitening the rest would amplify noise.

For each articulator and each k of a grid, the latent is projected on its first k principal
directions and whitened; linear read-outs then say how much of the keypoint positions and
velocities survive. ``k`` is the smallest size that loses at most ``tolerance`` of either R²
against the full latent [our rule].
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..analysis import PoseTeacherSettings
from ..metrics.geometry import centered, effective_rank, isoscore
from ..metrics.probes import ProbeTask, video_split
from ..models.pose_teachers import SJEPATeacher
from ..pose.tokens import articulator_columns
from ..pose.wholebody import Articulator
from .pose_teachers import (
    Features,
    LabelledClip,
    PoseCorpus,
    probe_subset,
    teacher_features,
    teacher_shape,
)

K_GRID: Final = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
R2_TOLERANCE: Final = 0.01
_ISOSCORE_ROWS: Final = 16_384
_SPECTRUM_HEAD: Final = 10


@dataclass(frozen=True, slots=True)
class Whitening:
    """``(x - mean) @ projection`` has zero mean and identity covariance on the fitting rows."""

    mean: np.ndarray
    projection: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        whitened: np.ndarray = (values - self.mean) @ self.projection
        return whitened.astype(np.float32)


@dataclass(frozen=True, slots=True)
class PrincipalAxes:
    """Principal directions of a set of rows, from which every truncation is derived."""

    mean: np.ndarray
    values: np.ndarray
    """Singular values of the centred rows, largest first."""
    directions: np.ndarray
    rows: int

    @classmethod
    def of(cls, rows: np.ndarray) -> "PrincipalAxes":
        mean = rows.mean(axis=0)
        _, values, directions = np.linalg.svd(rows - mean, full_matrices=False)
        return cls(mean, values, directions, len(rows))

    def variance_kept(self, k: int) -> float:
        variance = self.values**2
        return float(variance[:k].sum() / variance.sum())

    def whitening(self, k: int) -> Whitening:
        scale = self.values[:k] / np.sqrt(max(self.rows - 1, 1))
        return Whitening(self.mean, self.directions[:k].T / np.maximum(scale, 1e-12))


@dataclass(frozen=True, slots=True)
class Truncation:
    k: int
    variance_kept: float
    r2_position: float
    r2_velocity: float
    isoscore_whitened: float
    """IsoScore of the whitened latent on held-out clips: 1 is perfectly isotropic."""


@dataclass(frozen=True, slots=True)
class ArticulatorSpectrum:
    effective_rank: float
    isoscore: float
    directions_for_90: int
    directions_for_99: int
    explained_variance_head: list[float]
    truncations: list[Truncation]
    chosen_k: int


@dataclass(frozen=True, slots=True)
class SpectrumReport:
    clips: int
    train_clips: int
    test_clips: int
    tolerance: float
    articulators: dict[str, ArticulatorSpectrum]


def spectrum(
    features: Features,
    corpus: PoseCorpus,
    test: np.ndarray,
    grid: tuple[int, ...] = K_GRID,
    tolerance: float = R2_TOLERANCE,
) -> tuple[SpectrumReport, dict[Articulator, Whitening]]:
    """Spectrum, read-outs per truncation and the chosen whitening of every articulator."""
    articulators, whitenings = {}, {}
    for part in Articulator:
        values = features[part].astype(np.float64)
        rows = values[~test].reshape(-1, values.shape[-1])
        axes = PrincipalAxes.of(rows)
        sizes = [k for k in grid if k <= values.shape[-1]]
        truncations = [_truncation(values, axes, k, corpus, test, part) for k in sizes]
        full = truncations[-1]
        chosen = next(
            t.k
            for t in truncations
            if t.r2_position >= full.r2_position - tolerance
            and t.r2_velocity >= full.r2_velocity - tolerance
        )
        share = np.cumsum(axes.values**2) / np.sum(axes.values**2)
        sample = torch.from_numpy(rows[:: max(1, len(rows) // _ISOSCORE_ROWS)])
        articulators[str(part)] = ArticulatorSpectrum(
            effective_rank=effective_rank(centered(sample)),
            isoscore=isoscore(sample),
            directions_for_90=int(np.searchsorted(share, 0.90) + 1),
            directions_for_99=int(np.searchsorted(share, 0.99) + 1),
            explained_variance_head=np.diff(share[:_SPECTRUM_HEAD], prepend=0.0).tolist(),
            truncations=truncations,
            chosen_k=chosen,
        )
        whitenings[part] = axes.whitening(chosen)
    report = SpectrumReport(
        clips=len(corpus.clips),
        train_clips=int((~test).sum()),
        test_clips=int(test.sum()),
        tolerance=tolerance,
        articulators=articulators,
    )
    return report, whitenings


def _truncation(
    values: np.ndarray,
    axes: PrincipalAxes,
    k: int,
    corpus: PoseCorpus,
    test: np.ndarray,
    part: Articulator,
) -> Truncation:
    whitened = axes.whitening(k).apply(values)
    columns = articulator_columns(part)
    targets = corpus.positions[:, :, columns]
    weights = corpus.weights[:, :, columns]
    held_out = whitened[test].reshape(-1, k)
    sample = torch.from_numpy(held_out[:: max(1, len(held_out) // _ISOSCORE_ROWS)]).double()
    return Truncation(
        k=k,
        variance_kept=axes.variance_kept(k),
        r2_position=ProbeTask.of_frames(
            whitened, targets, weights, test, corpus.videos
        ).held_out_r2(),
        r2_velocity=ProbeTask.of_velocities(
            whitened, targets, weights, test, corpus.videos
        ).held_out_r2(),
        isoscore_whitened=isoscore(sample),
    )


def measure(
    clips: Sequence[LabelledClip],
    weights: Path,
    settings: PoseTeacherSettings,
    device: torch.device,
) -> tuple[SpectrumReport, dict[Articulator, Whitening]]:
    """The spectrum of a saved S-JEPA teacher on the read-out clips of the teacher comparison."""
    corpus = PoseCorpus.load(clips, settings.min_score)
    probe, probe_test, _ = probe_subset(corpus, video_split(corpus.videos), settings)
    teacher = SJEPATeacher(
        teacher_shape(settings),
        momentum=(settings.ema_start, 1.0),
        centre_rate=settings.centre_rate,
    )
    teacher.load_state_dict(torch.load(weights, map_location="cpu"))
    features = teacher_features(
        teacher, torch.from_numpy(probe.tokens), device, settings.batch_size
    )
    return spectrum(features, probe, probe_test)


def save_whitenings(whitenings: dict[Articulator, Whitening], path: Path) -> Path:
    """One ``<articulator>_mean`` and ``<articulator>_projection`` array per articulator."""
    arrays: dict[str, Any] = {}
    for part, whitening in whitenings.items():
        arrays[f"{part}_mean"] = whitening.mean
        arrays[f"{part}_projection"] = whitening.projection
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
