"""Collaudo "testare i test" (§4.13.1): geometry and collapse metrics on known cases."""

import pytest
import torch

from signworld.metrics.geometry import (
    CollapseAlarm,
    CollapseReading,
    effective_rank,
    explained_variance,
    isoscore,
    mean_random_pair_cosine,
)


def _gaussian(samples: int, dimension: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(samples, dimension, generator=torch.Generator().manual_seed(seed))


def test_effective_rank_of_orthogonal_directions_is_their_number() -> None:
    assert effective_rank(torch.eye(32)) == pytest.approx(32.0)


def test_effective_rank_of_points_on_a_line_is_one() -> None:
    line = torch.linspace(-1, 1, 100)[:, None] * torch.ones(1, 16)

    assert effective_rank(line) == pytest.approx(1.0)


def test_isoscore_separates_isotropic_from_one_dimensional_data() -> None:
    isotropic = _gaussian(20_000, 16)
    one_dimensional = isotropic[:, :1] * torch.eye(16)[0]

    assert isoscore(isotropic) > 0.95
    assert isoscore(one_dimensional) == pytest.approx(0.0, abs=1e-6)


def test_isoscore_ignores_rotation_and_mean() -> None:
    data = _gaussian(5000, 8) * torch.linspace(0.2, 2.0, 8)
    rotation, _ = torch.linalg.qr(_gaussian(8, 8, seed=1))

    moved = data @ rotation.T + 5.0

    assert isoscore(moved) == pytest.approx(isoscore(data), abs=1e-6)


def test_explained_variance_sums_to_one_and_decreases() -> None:
    fractions = explained_variance(_gaussian(1000, 8) * torch.arange(1, 9))

    assert fractions.sum().item() == pytest.approx(1.0)
    assert torch.all(fractions[:-1] >= fractions[1:])


def test_random_pair_cosine_detects_a_shared_direction() -> None:
    isotropic = _gaussian(5000, 64)
    anisotropic = isotropic + 3.0

    assert mean_random_pair_cosine(isotropic) == pytest.approx(0.0, abs=0.01)
    assert mean_random_pair_cosine(anisotropic) > 0.8


def test_collapse_alarm_fires_on_constant_vectors_only() -> None:
    reference = CollapseReading.of(_gaussian(2000, 64))
    alarm = CollapseAlarm(reference)

    healthy = CollapseReading.of(_gaussian(2000, 64, seed=1))
    collapsed = CollapseReading.of(torch.ones(2000, 64) + 1e-3 * _gaussian(2000, 64, seed=2))

    assert alarm.triggered_by(healthy) == []
    assert alarm.triggered_by(collapsed) == ["mean_dimension_std"]


def test_collapse_alarm_fires_on_rank_loss_with_preserved_spread() -> None:
    reference = CollapseReading.of(_gaussian(2000, 64))
    low_rank = _gaussian(2000, 2, seed=1) @ _gaussian(2, 64, seed=2)

    assert "effective_rank" in CollapseAlarm(reference).triggered_by(CollapseReading.of(low_rank))
