"""Linear read-outs of frozen representations, fitted and scored on held-out videos.

A representation is judged by what a ridge regression recovers from it: keypoints, their
velocities, or a label. Clips are split by video so that no clip of a test video is seen
during fitting, and missing targets carry zero weight.
"""

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np

TEST_FRACTION: Final = 0.2
RIDGE_PENALTIES: Final = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _bucket(key: str) -> float:
    """A stable position in [0, 1) for an identifier, the same on every run and machine."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest) / 2**64


def video_split(videos: np.ndarray, fraction: float = TEST_FRACTION) -> np.ndarray:
    """True for clips of held-out videos, by a stable hash of the video ID."""
    return np.array([_bucket(video) < fraction for video in videos])


def channel_split(
    channels: np.ndarray, labels: np.ndarray, fraction: float = TEST_FRACTION
) -> np.ndarray:
    """True for clips of held-out channels, chosen within each label.

    A label read on channels never seen in fitting cannot lean on who signs. Each label with at
    least two channels keeps some on both sides; a label seen on one channel only cannot be
    tested this way, and its clips all stay in fitting.
    """
    test = np.zeros(len(channels), dtype=bool)
    for label in np.unique(labels):
        own = sorted(np.unique(channels[labels == label]), key=_bucket)
        if len(own) < 2:  # noqa: PLR2004 (one channel on each side at least)
            continue
        held = own[: min(len(own) - 1, max(1, round(fraction * len(own))))]
        test |= (labels == label) & np.isin(channels, held)
    return test


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray, penalty: float
) -> np.ndarray:
    """Weighted ridge per keypoint: (d + 1, k, 2) coefficients, the last row the intercept.

    Features are standardised by the caller. Missing keypoints (weight 0) do not enter the fit
    of their own coordinate, as the confidence-weighted anchor of §4.5.3 would treat them.
    """
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    identity = np.eye(design.shape[1])
    identity[-1, -1] = 0.0
    coefficients = np.zeros((design.shape[1], targets.shape[1], 2))
    for keypoint in range(targets.shape[1]):
        weighted = design * weights[:, keypoint, None]
        gram = design.T @ weighted + penalty * len(design) * identity
        coefficients[:, keypoint] = np.linalg.solve(gram, weighted.T @ targets[:, keypoint])
    return coefficients


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    predictions: np.ndarray = np.einsum("nd,dkc->nkc", design, coefficients)
    return predictions


def weighted_r2(targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> float:
    w = np.broadcast_to(weights[..., None], targets.shape)
    if w.sum() == 0:
        return float("nan")
    mean = (targets * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
    residual = (w * (targets - predictions) ** 2).sum()
    total = (w * (targets - mean) ** 2).sum()
    return float(1 - residual / total) if total > 0 else float("nan")


@dataclass(frozen=True, slots=True)
class Probe:
    """A linear read-out fitted on training videos, penalty chosen on a validation share."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, videos: np.ndarray
    ) -> "Probe":
        mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-6
        standard = (features - mean) / scale
        validation = video_split(videos, 0.15)
        scores = {}
        for penalty in RIDGE_PENALTIES:
            coefficients = fit_ridge(
                standard[~validation], targets[~validation], weights[~validation], penalty
            )
            prediction = predict(standard[validation], coefficients)
            scores[penalty] = weighted_r2(targets[validation], prediction, weights[validation])
        best = max(scores, key=lambda penalty: np.nan_to_num(scores[penalty], nan=-np.inf))
        return cls(mean, scale, fit_ridge(standard, targets, weights, best))

    def r2(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
        prediction = predict((features - self.mean) / self.scale, self.coefficients)
        return weighted_r2(targets, prediction, weights)


def _frames(values: np.ndarray) -> np.ndarray:
    """(clips, frames, ...) to (clips * frames, ...)."""
    return values.reshape(-1, *values.shape[2:])


@dataclass(frozen=True, slots=True)
class ProbeTask:
    """Per-frame features and targets of one read-out, split by held-out videos."""

    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    test: np.ndarray
    videos: np.ndarray

    @classmethod
    def of_frames(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        test_clips: np.ndarray,
        videos: np.ndarray,
    ) -> "ProbeTask":
        """Each frame's feature reads that frame's keypoints."""
        frames = features.shape[1]
        return cls(
            _frames(features),
            _frames(targets),
            _frames(weights),
            np.repeat(test_clips, frames),
            np.repeat(videos, frames),
        )

    @classmethod
    def of_velocities(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        test_clips: np.ndarray,
        videos: np.ndarray,
    ) -> "ProbeTask":
        """Two consecutive frames' features read the keypoint displacement between them."""
        pairs = np.concatenate([features[:, 1:], features[:, :-1]], axis=-1)
        return cls.of_frames(
            pairs,
            targets[:, 1:] - targets[:, :-1],
            weights[:, 1:] * weights[:, :-1],
            test_clips,
            videos,
        )

    def fit(self) -> Probe:
        train = ~self.test
        return Probe.fit(
            self.features[train], self.targets[train], self.weights[train], self.videos[train]
        )

    def score(self, probe: Probe, mask: np.ndarray | None = None) -> float:
        """R² of ``probe`` on the held-out frames, optionally restricted by ``mask``."""
        chosen = self.test if mask is None else self.test & mask
        return probe.r2(self.features[chosen], self.targets[chosen], self.weights[chosen])

    def held_out_r2(self) -> float:
        return self.score(self.fit())


def label_accuracy(
    features: np.ndarray, labels: np.ndarray, test: np.ndarray, penalty: float = 1e-2
) -> tuple[float, float]:
    """Held-out accuracy of a ridge classifier, and the majority-class share it must beat.

    One-vs-rest least squares on standardised features, fitted on the training rows; test rows
    whose label never occurs in training are left out. ``nan`` when fewer than two classes
    remain on either side.
    """
    classes = np.unique(labels[~test])
    evaluated = test & np.isin(labels, classes)
    # Accuracy says nothing unless both sides hold at least two classes.
    if len(classes) < 2 or len(np.unique(labels[evaluated])) < 2:  # noqa: PLR2004
        return float("nan"), float("nan")
    mean, scale = features[~test].mean(axis=0), features[~test].std(axis=0) + 1e-6
    standard = (features - mean) / scale
    one_hot = (labels[~test, None] == classes[None]).astype(np.float64)
    design = np.concatenate([standard[~test], np.ones(((~test).sum(), 1))], axis=1)
    gram = design.T @ design + penalty * len(design) * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ one_hot)
    scores = np.concatenate([standard[evaluated], np.ones((evaluated.sum(), 1))], axis=1) @ weights
    accuracy = float((classes[scores.argmax(axis=1)] == labels[evaluated]).mean())
    _, counts = np.unique(labels[evaluated], return_counts=True)
    return accuracy, float(counts.max() / counts.sum())
