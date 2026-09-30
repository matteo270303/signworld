"""The isotropy test's measurements on known cases, and the whole run on a tiny corpus."""

from pathlib import Path

import numpy as np
import pytest
import torch

from signworld.analysis import PoseIsotropySettings
from signworld.experiments.pose_isotropy import (
    IsotropyRun,
    diagnostics,
    knn_r2,
    temporal_ratio,
)
from signworld.experiments.pose_teachers import PoseCorpus
from signworld.metrics.probes import video_split
from signworld.metrics.sigreg import SIGReg, random_directions
from signworld.pose.tokens import STEPS
from signworld.pose.wholebody import Articulator

from .test_pose_teachers import TINY, _clips


def test_diagnostics_of_a_gaussian_read_as_gaussian() -> None:
    generator = torch.Generator().manual_seed(0)
    directions = random_directions(16, 64, generator=generator)
    reference = float(
        np.mean(
            [
                float(SIGReg()(torch.randn(8192, 16, generator=generator), directions))
                for _ in range(8)
            ]
        )
    )

    result = diagnostics(
        torch.randn(8192, 16, generator=generator), directions, reference, generator
    )

    assert result.sigreg_ratio == pytest.approx(1.0, abs=0.5)
    assert result.isoscore > 0.9
    assert result.mean_abs_excess_kurtosis < 0.2
    assert result.participation_ratio == pytest.approx(16.0, rel=0.05)


def test_knn_recovers_targets_the_features_encode() -> None:
    rng = np.random.default_rng(0)
    targets = rng.standard_normal((3000, 3, 2)).astype(np.float32)
    features = targets.reshape(3000, -1) @ rng.standard_normal((6, 12)).astype(np.float32)
    weights = np.ones((3000, 3), np.float32)
    t = torch.from_numpy

    r2 = knn_r2(
        t(features[:2500]),
        t(targets[:2500]),
        t(weights[:2500]),
        t(features[2500:]),
        targets[2500:],
        weights[2500:],
        neighbours=5,
    )

    assert r2 > 0.8


def test_temporal_ratio_separates_smooth_from_random_sequences() -> None:
    generator = torch.Generator().manual_seed(0)
    smooth = torch.randn(50, 1, 8, generator=generator) + 0.01 * torch.randn(
        50, 32, 8, generator=generator
    ).cumsum(dim=1)
    random = torch.randn(50, 32, 8, generator=generator)

    assert temporal_ratio(smooth, generator) < 0.1
    assert temporal_ratio(random, generator) == pytest.approx(1.0, abs=0.05)


def test_every_map_is_fitted_and_measured(tmp_path: Path) -> None:
    corpus = PoseCorpus.load(_clips(tmp_path, videos=40), TINY.min_score)
    test = video_split(corpus.videos)
    rng = np.random.default_rng(0)
    scales = np.linspace(0.1, 3.0, 12).astype(np.float32)
    features = {
        part: (rng.exponential(size=(len(corpus.clips), STEPS, 12)) * scales).astype(np.float32)
        for part in Articulator
    }
    settings = PoseIsotropySettings(
        iterations=(1, 2),
        marginal_knots=64,
        sinf_directions=3,
        sinf_steps=5,
        sinf_rows=512,
        flow_depths=(2,),
        flow_hidden=16,
        flow_batch=256,
        flow_max_epochs=2,
        knn_neighbours=3,
        knn_fit_rows=500,
        knn_query_rows=200,
        diagnostic_rows=512,
        sigreg_directions=16,
        neighbour_queries=50,
        neighbour_pool=300,
    )

    results = IsotropyRun(corpus, features, test, settings, torch.device("cpu"), print).run()

    names = [(r.method, r.setting) for r in results]
    assert names[0] == ("raw", "-")
    assert {m for m, _ in names} == {"raw", "whitening", "rbig", "sinf", "flow"}
    raw, whitened = results[0], results[1]
    for part in raw.parts:
        # Whitening is linear and invertible: only the ridge penalty can move the read-out.
        assert whitened.parts[part].r2_position == pytest.approx(
            raw.parts[part].r2_position, abs=0.05
        )
        assert whitened.parts[part].diagnostics.isoscore > raw.parts[part].diagnostics.isoscore
        assert raw.parts[part].neighbour_recall == 1.0
        assert whitened.parts[part].inversion_error < 1e-4
