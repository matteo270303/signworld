"""Can a fixed map make the S-JEPA latent isotropic without losing what it knows? (§4.4.3)

The frozen teacher's latent is anisotropic (IsoScore ≈ 0.026). Three families of maps from
``models.gaussianize`` are fitted, per articulator, on the training videos of the pose-teacher
read-outs and measured on the held-out videos, before and after, each along its own knob:

* whitening, one matrix, by shrinkage ε;
* iterative Gaussianization, RBIG (PCA rotations) and SINF (least-Gaussian directions), by
  number of iterations;
* a RealNVP flow trained by maximum likelihood, by depth.

For every setting and articulator the test reads the geometry (effective rank, participation
ratio, condition number, stable rank, top direction's share, IsoScore, SIGReg against a true
Gaussian, skewness and kurtosis of projections, mean pair cosine), what the latent still says
(linear R² of keypoint positions and velocities as in PC5, and a k-NN R² of positions), how
distances change (nearest neighbours kept, step-to-step against random-pair distance), the
channel and language read from clip averages, and the cost (fitting and applying, parameters,
inversion error, flow likelihood).
"""

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from signworld.data.pose.tokens import articulator_columns
from signworld.data.pose.wholebody import Articulator
from signworld.experiment.collaudo.analysis import PoseIsotropySettings, PoseTeacherSettings
from signworld.loss.sigreg import SIGReg, random_directions
from signworld.metrics.divergence import gaussian_divergence, sliced_negentropy
from signworld.metrics.geometry import (
    condition_number,
    effective_rank,
    explained_variance,
    isoscore,
    mean_random_pair_cosine,
    participation_ratio,
    stable_rank,
)
from signworld.metrics.probes import ProbeTask, label_accuracy, video_split, weighted_r2
from signworld.models.encoders.pose_teachers import SJEPATeacher
from signworld.models.gaussianize import (
    FlowFit,
    FlowTransform,
    Identity,
    IterativeGaussianization,
    Prefix,
    Transform,
    Whitening,
    fit_flow,
)

from .pose_teachers import LabelledClip, PoseCorpus, probe_subset, teacher_features, teacher_shape

logger = logging.getLogger(__name__)

CHUNK = 65_536
"""Rows transformed at a time on the device."""
GAUSSIAN_DRAWS = 8
"""Gaussian samples averaged for the SIGReg reference."""


# --------------------------------------------------------------------------- measurements


@dataclass(frozen=True, slots=True)
class Diagnostics:
    effective_rank: float
    participation_ratio: float
    condition_number: float
    stable_rank: float
    top_variance_fraction: float
    isoscore: float
    sigreg_ratio: float
    """SIGReg over that of a true Gaussian of the same size, same directions: 1 is Gaussian."""
    mean_abs_skewness: float
    mean_abs_excess_kurtosis: float
    """Over the random projections: both 0 for a Gaussian."""
    mean_pair_cosine: float
    kl_to_standard: float
    """KL(N(μ, Σ) ‖ N(0, I)) in nats: mean, scale and shape of the moment-matched Gaussian."""
    kl_to_isotropic: float
    """KL(N(μ, Σ) ‖ N(μ, s² I)) in nats: anisotropy alone."""
    negentropy: float
    """KL(p ‖ N(μ, Σ)) read on random 1-D projections, nats per projection: non-Gaussianity."""
    covariance_floored: bool
    """Eigenvalues below 1e-12 of the largest: the KLs are then lower bounds."""


def diagnostics(
    rows: Tensor, directions: Tensor, gaussian_sigreg: float, generator: torch.Generator
) -> Diagnostics:
    """Geometry of ``rows`` (samples, dimension); SIGReg on rows scaled as in PC1."""
    data = rows.double().cpu()
    centred = data - data.mean(dim=0)
    dimension = data.shape[1]
    scaled = centred * (dimension / centred.pow(2).sum(dim=1).mean()).sqrt()
    projections = scaled.float() @ directions.T
    standard = (projections - projections.mean(dim=0)) / projections.std(dim=0).clamp_min(1e-12)
    divergence = gaussian_divergence(data)
    return Diagnostics(
        effective_rank=effective_rank(centred),
        participation_ratio=participation_ratio(data),
        condition_number=condition_number(data),
        stable_rank=stable_rank(data),
        top_variance_fraction=float(explained_variance(data)[0]),
        isoscore=isoscore(data),
        sigreg_ratio=float(SIGReg()(scaled.float(), directions)) / gaussian_sigreg,
        mean_abs_skewness=float(standard.pow(3).mean(dim=0).abs().mean()),
        mean_abs_excess_kurtosis=float((standard.pow(4).mean(dim=0) - 3.0).abs().mean()),
        mean_pair_cosine=mean_random_pair_cosine(centred, 100_000, generator),
        kl_to_standard=divergence.to_standard,
        kl_to_isotropic=divergence.to_isotropic,
        negentropy=sliced_negentropy(scaled, directions),
        covariance_floored=divergence.floored,
    )


def knn_r2(  # noqa: PLR0913 (fitting and query rows, each with targets and weights)
    fit_x: Tensor,
    fit_y: Tensor,
    fit_w: Tensor,
    query_x: Tensor,
    query_y: np.ndarray,
    query_w: np.ndarray,
    *,
    neighbours: int,
) -> float:
    """R² of keypoints predicted as the presence-weighted mean over the k nearest fit rows.

    Euclidean distance in the latent, the geometry the physical level's L1 energy sees. A
    keypoint missing in every neighbour falls back to its weighted mean over the fit rows.
    """
    fallback = (fit_y * fit_w[..., None]).sum(dim=0) / fit_w.sum(dim=0).clamp_min(1e-12)[:, None]
    predictions = []
    for start in range(0, len(query_x), 2048):
        distance = torch.cdist(query_x[start : start + 2048], fit_x)
        nearest = distance.topk(neighbours, dim=1, largest=False).indices
        weights = fit_w[nearest]
        mean = (fit_y[nearest] * weights[..., None]).sum(dim=1)
        total = weights.sum(dim=1)[..., None]
        predictions.append(torch.where(total > 0, mean / total.clamp_min(1e-12), fallback))
    predicted = torch.cat(predictions).cpu().numpy()
    return weighted_r2(query_y, predicted, query_w)


def nearest(pool: Tensor, queries: Tensor, count: int = 10) -> Tensor:
    """(queries, count) Euclidean nearest rows of ``pool`` for the rows ``queries`` of it."""
    distance = torch.cdist(pool[queries], pool)
    distance[torch.arange(len(queries), device=pool.device), queries] = math.inf
    return distance.topk(count, dim=1, largest=False).indices


def neighbour_recall(found: Tensor, reference: Tensor) -> float:
    shared = (found[:, :, None] == reference[:, None, :]).any(dim=2).sum(dim=1)
    return float(shared.float().mean() / reference.shape[1])


def temporal_ratio(clips: Tensor, generator: torch.Generator, pairs: int = 100_000) -> float:
    """Mean distance between consecutive steps over mean distance between random rows.

    Well below 1: the latent moves smoothly in time; near 1: steps look like random points.
    """
    step = (clips[:, 1:] - clips[:, :-1]).norm(dim=-1).mean()
    rows = clips.reshape(-1, clips.shape[-1])
    first = torch.randint(len(rows), (pairs,), generator=generator).to(rows.device)
    second = torch.randint(len(rows), (pairs,), generator=generator).to(rows.device)
    return float(step / (rows[first] - rows[second]).norm(dim=-1).mean())


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class PartResult:
    diagnostics: Diagnostics
    r2_position: float
    r2_velocity: float
    knn_r2_position: float
    neighbour_recall: float
    """Share of each held-out row's 10 nearest rows (raw latent) still nearest after the map."""
    temporal_ratio: float
    inversion_error: float
    """‖f⁻¹(f(x)) - x‖ / ‖x‖ on held-out rows."""
    passed: bool
    """IsoScore, SIGReg and both R² within the criteria of the settings."""


@dataclass(frozen=True, slots=True)
class ConfigResult:
    method: str
    setting: str
    parts: dict[str, PartResult]
    channel_accuracy: float
    channel_chance: float
    language_accuracy: float
    language_chance: float
    fit_seconds: float
    apply_seconds_per_1000_rows: float
    parameters: int
    flows: dict[str, FlowFit] | None


@dataclass(frozen=True, slots=True)
class IsotropyReport:
    clips: int
    train_clips: int
    test_clips: int
    criteria: dict[str, float]
    gaussian_reference: Diagnostics
    configs: list[ConfigResult]


# --------------------------------------------------------------------------- the run


@dataclass
class _Part:
    """Everything about one articulator that does not depend on the map."""

    features: np.ndarray
    """(clips, 32, 256) raw latent."""
    train_rows: Tensor
    fit_rows: Tensor
    validation_rows: Tensor
    diagnostic_rows: Tensor
    knn_fit: np.ndarray
    knn_query: np.ndarray
    pool: np.ndarray
    queries: Tensor
    raw_neighbours: Tensor | None = None


def _apply(transform: Transform, rows: Tensor) -> Tensor:
    return torch.cat([transform.forward(rows[i : i + CHUNK]) for i in range(0, len(rows), CHUNK)])


class IsotropyRun:
    def __init__(
        self,
        corpus: PoseCorpus,
        features: dict[Articulator, np.ndarray],
        test: np.ndarray,
        settings: PoseIsotropySettings,
        device: torch.device,
        progress: Callable[[str], None],
    ) -> None:
        self.corpus, self.test, self.settings = corpus, test, settings
        self.device, self.progress = device, progress
        self.generator = torch.Generator().manual_seed(settings.seed)
        self.directions = random_directions(
            next(iter(features.values())).shape[-1],
            settings.sigreg_directions,
            generator=self.generator,
        )
        dimension = self.directions.shape[1]
        # SIGReg of a finite Gaussian sample is noisy (±10 % at 16k rows): average 8 draws.
        self.gaussian_sigreg = float(
            np.mean(
                [
                    float(
                        SIGReg()(
                            torch.randn(
                                settings.diagnostic_rows, dimension, generator=self.generator
                            ),
                            self.directions,
                        )
                    )
                    for _ in range(GAUSSIAN_DRAWS)
                ]
            )
        )
        self.raw_r2: dict[Articulator, tuple[float, float]] = {}
        videos = corpus.videos
        validation = video_split(videos, 0.35) & ~test  # the next hash band after the test one
        rng = np.random.default_rng(settings.seed)
        self.parts = {}
        for part, values in features.items():
            train = values[~test].reshape(-1, values.shape[-1])
            test_rows = values[test].reshape(-1, values.shape[-1])
            steps = values.shape[1]
            train_index = np.flatnonzero(np.repeat(~test, steps))
            test_index = np.flatnonzero(np.repeat(test, steps))
            pool = rng.choice(len(test_rows), min(settings.neighbour_pool, len(test_rows)), False)
            self.parts[part] = _Part(
                features=values,
                train_rows=torch.from_numpy(train).to(device),
                fit_rows=torch.from_numpy(
                    values[~test & ~validation].reshape(-1, values.shape[-1])
                ).to(device),
                validation_rows=torch.from_numpy(
                    values[validation].reshape(-1, values.shape[-1])
                ).to(device),
                diagnostic_rows=torch.from_numpy(
                    test_rows[
                        rng.choice(
                            len(test_rows), min(settings.diagnostic_rows, len(test_rows)), False
                        )
                    ]
                ).to(device),
                knn_fit=rng.choice(
                    train_index, min(settings.knn_fit_rows, len(train_index)), False
                ),
                knn_query=rng.choice(
                    test_index, min(settings.knn_query_rows, len(test_index)), False
                ),
                pool=test_index[pool],
                queries=torch.arange(min(settings.neighbour_queries, len(pool))),
            )

    # ----------------------------------------------------------------------- per map

    def _part_result(
        self, part: Articulator, transform: Transform, transformed: np.ndarray
    ) -> PartResult:
        data = self.parts[part]
        settings = self.settings
        columns = articulator_columns(part)
        targets = self.corpus.positions[:, :, columns]
        weights = self.corpus.weights[:, :, columns]
        videos = self.corpus.videos
        r2_position = ProbeTask.of_frames(
            transformed, targets, weights, self.test, videos
        ).held_out_r2()
        r2_velocity = ProbeTask.of_velocities(
            transformed, targets, weights, self.test, videos
        ).held_out_r2()
        rows = transformed.reshape(-1, transformed.shape[-1])
        flat_targets = targets.reshape(-1, *targets.shape[2:])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        device = self.device
        knn = knn_r2(
            torch.from_numpy(rows[data.knn_fit]).to(device),
            torch.from_numpy(flat_targets[data.knn_fit]).to(device),
            torch.from_numpy(flat_weights[data.knn_fit]).to(device),
            torch.from_numpy(rows[data.knn_query]).to(device),
            flat_targets[data.knn_query],
            flat_weights[data.knn_query],
            neighbours=settings.knn_neighbours,
        )
        pool = torch.from_numpy(rows[data.pool]).to(device)
        found = nearest(pool, data.queries.to(device))
        if data.raw_neighbours is None:
            data.raw_neighbours = found
        recall = neighbour_recall(found, data.raw_neighbours)
        held_out = torch.from_numpy(transformed[self.test]).to(device)
        smooth = temporal_ratio(held_out, self.generator)
        sample = _apply(transform, data.diagnostic_rows)
        geometry = diagnostics(sample, self.directions, self.gaussian_sigreg, self.generator)
        restored = transform.inverse(sample)
        inversion = float(
            (restored - data.diagnostic_rows).norm() / data.diagnostic_rows.norm().clamp_min(1e-12)
        )
        raw_position, raw_velocity = self.raw_r2.setdefault(part, (r2_position, r2_velocity))
        passed = (
            geometry.isoscore >= settings.min_isoscore
            and geometry.sigreg_ratio <= settings.max_sigreg_ratio
            and r2_position >= raw_position - settings.max_r2_drop
            and r2_velocity >= raw_velocity - settings.max_r2_drop
        )
        return PartResult(
            geometry, r2_position, r2_velocity, knn, recall, smooth, inversion, passed
        )

    def evaluate(
        self,
        method: str,
        setting: str,
        transforms: dict[Articulator, Transform],
        fit_seconds: float,
        flows: dict[str, FlowFit] | None = None,
    ) -> ConfigResult:
        parts, means = {}, []
        applied_seconds, applied_rows = 0.0, 0
        for part, transform in transforms.items():
            values = self.parts[part].features
            rows = torch.from_numpy(values.reshape(-1, values.shape[-1])).to(self.device)
            started = time.monotonic()
            transformed = _apply(transform, rows)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            applied_seconds += time.monotonic() - started
            applied_rows += len(rows)
            array = transformed.float().cpu().numpy().reshape(values.shape)
            parts[str(part)] = self._part_result(part, transform, array)
            means.append(array.mean(axis=1))
        clip_means = np.concatenate(means, axis=1)
        channel, channel_chance = label_accuracy(
            clip_means, self.corpus.labels("channel"), self.test
        )
        language, language_chance = label_accuracy(
            clip_means, self.corpus.labels("language"), self.test
        )
        result = ConfigResult(
            method=method,
            setting=setting,
            parts=parts,
            channel_accuracy=channel,
            channel_chance=channel_chance,
            language_accuracy=language,
            language_chance=language_chance,
            fit_seconds=fit_seconds,
            apply_seconds_per_1000_rows=1000 * applied_seconds / max(applied_rows, 1),
            parameters=sum(t.parameters_count() for t in transforms.values()),
            flows=flows,
        )
        iso = " ".join(f"{p} {r.diagnostics.isoscore:.2f}" for p, r in parts.items())
        r2 = " ".join(f"{p} {r.r2_position:.3f}/{r.r2_velocity:.3f}" for p, r in parts.items())
        kl = " ".join(
            f"{p} {r.diagnostics.kl_to_isotropic:.1f}+{r.diagnostics.negentropy:.3f}"
            for p, r in parts.items()
        )
        self.progress(
            f"{method} {setting}: IsoScore {iso} · R² pos/vel {r2} · KL aniso+negentropy {kl}"
        )
        return result

    # ----------------------------------------------------------------------- the maps

    def run(self) -> list[ConfigResult]:
        settings = self.settings
        results = [self.evaluate("raw", "-", {part: Identity() for part in self.parts}, 0.0)]
        for shrinkage in settings.whitening_shrinkage:
            started = time.monotonic()
            maps: dict[Articulator, Transform] = {
                part: Whitening.fit(data.train_rows, shrinkage) for part, data in self.parts.items()
            }
            elapsed = time.monotonic() - started
            results.append(self.evaluate("whitening", f"eps={shrinkage:g}", maps, elapsed))
        for rotation, name in (("pca", "rbig"), ("sliced", "sinf")):
            models = {
                part: IterativeGaussianization.fit(
                    data.train_rows,
                    max(settings.iterations),
                    rotation,  # type: ignore[arg-type]
                    knots=settings.marginal_knots,
                    directions=settings.sinf_directions,
                    steps=settings.sinf_steps,
                    rows=settings.sinf_rows,
                    seed=settings.seed,
                )
                for part, data in self.parts.items()
            }
            self.progress(f"{name}: fitted {max(settings.iterations)} iterations")
            for count in settings.iterations:
                seconds = sum(model.seconds[count - 1] for model in models.values())
                prefixes: dict[Articulator, Transform] = {
                    part: Prefix(model, count) for part, model in models.items()
                }
                results.append(self.evaluate(name, f"iterations={count}", prefixes, seconds))
        for depth in settings.flow_depths:
            started = time.monotonic()
            flows, fits = {}, {}
            for part, data in self.parts.items():
                flow, fit = fit_flow(
                    data.fit_rows,
                    data.validation_rows,
                    depth,
                    settings.flow_hidden,
                    batch_size=settings.flow_batch,
                    learning_rate=settings.flow_learning_rate,
                    max_epochs=settings.flow_max_epochs,
                    patience=settings.flow_patience,
                    seed=settings.seed,
                )
                flows[part], fits[str(part)] = FlowTransform(flow), fit
                self.progress(
                    f"flow depth {depth} {part}: epoch {fit.epochs}, NLL {fit.validation_nll:.3f}"
                )
            elapsed = time.monotonic() - started
            maps = dict(flows)
            results.append(self.evaluate("flow", f"depth={depth}", maps, elapsed, fits))
        return results


def run(
    clips: Sequence[LabelledClip],
    weights: Path,
    teacher_settings: PoseTeacherSettings,
    settings: PoseIsotropySettings,
    device: torch.device,
    *,
    progress: Callable[[str], None] = logger.info,
) -> IsotropyReport:
    """The whole comparison on the read-out clips of the pose-teacher comparison (PC5)."""
    torch.manual_seed(settings.seed)
    corpus = PoseCorpus.load(clips, teacher_settings.min_score)
    probe, probe_test, _ = probe_subset(corpus, video_split(corpus.videos), teacher_settings)
    teacher = SJEPATeacher(
        teacher_shape(teacher_settings),
        momentum=(teacher_settings.ema_start, 1.0),
        centre_rate=teacher_settings.centre_rate,
    )
    teacher.load_state_dict(torch.load(weights, map_location="cpu"))
    features = teacher_features(
        teacher, torch.from_numpy(probe.tokens), device, teacher_settings.batch_size
    )
    features = {part: values.astype(np.float32) for part, values in features.items()}
    progress(
        f"{len(probe.clips)} clips ({int((~probe_test).sum())} train, {int(probe_test.sum())} test)"
    )
    job = IsotropyRun(probe, features, probe_test, settings, device, progress)
    configs = job.run()
    reference = torch.randn(
        settings.diagnostic_rows, job.directions.shape[1], generator=job.generator
    )
    criteria: dict[str, Any] = {
        "min_isoscore": settings.min_isoscore,
        "max_r2_drop": settings.max_r2_drop,
        "max_sigreg_ratio": settings.max_sigreg_ratio,
    }
    return IsotropyReport(
        clips=len(probe.clips),
        train_clips=int((~probe_test).sum()),
        test_clips=int(probe_test.sum()),
        criteria=criteria,
        gaussian_reference=diagnostics(
            reference, job.directions, job.gaussian_sigreg, job.generator
        ),
        configs=configs,
    )
