"""KL divergences from Gaussians, on cases with a closed-form answer."""

import math

import pytest
import torch

from signworld.loss.sigreg import random_directions
from signworld.metrics.divergence import gaussian_divergence, sliced_negentropy, spacing_entropy


def _normal(rows: int, dimension: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(rows, dimension, generator=torch.Generator().manual_seed(seed))


def test_a_standard_gaussian_is_at_zero_up_to_the_sampling_bias() -> None:
    result = gaussian_divergence(_normal(50_000, 16))

    assert result.to_standard == pytest.approx(0.0, abs=0.02)
    assert result.to_isotropic == pytest.approx(0.0, abs=0.02)
    assert not result.floored


def test_closed_forms_for_scale_shift_and_anisotropy() -> None:
    shifted = _normal(200_000, 1) * 2.0 + 1.0  # N(1, 4)
    stretched = _normal(200_000, 2) * torch.tensor([1.0, 2.0])  # variances 1 and 4

    assert gaussian_divergence(shifted).to_standard == pytest.approx(
        0.5 * (4 + 1 - 1 - math.log(4)), rel=0.02
    )
    assert gaussian_divergence(shifted).to_isotropic == pytest.approx(0.0, abs=1e-9)
    expected = 0.5 * (2 * math.log(2.5) - math.log(4))
    assert gaussian_divergence(stretched).to_isotropic == pytest.approx(expected, rel=0.03)


def test_a_rank_deficient_set_is_flagged() -> None:
    flat = _normal(1000, 1) * torch.tensor([1.0, 0.0, 0.0])

    assert gaussian_divergence(flat).floored


def test_spacing_entropy_of_a_standard_normal() -> None:
    entropy = spacing_entropy(_normal(20_000, 3))

    assert torch.allclose(
        entropy,
        torch.full((3,), 0.5 * math.log(2 * math.pi * math.e), dtype=entropy.dtype),
        atol=0.02,
    )


def test_sliced_negentropy_separates_gaussian_from_uniform() -> None:
    directions = random_directions(2, 32, generator=torch.Generator().manual_seed(1))
    uniform = torch.rand(20_000, 2, generator=torch.Generator().manual_seed(2))

    assert sliced_negentropy(_normal(20_000, 2), directions) < 0.01
    assert sliced_negentropy(uniform, directions) > 0.05
