"""Collaudo "testare i test" (§4.13.1): SIGReg is near zero on Gaussians and high otherwise."""

import torch

from signworld.metrics.sigreg import SIGReg, random_directions

DIMENSION = 128
SAMPLES = 4096


def _statistic(embeddings: torch.Tensor) -> float:
    directions = random_directions(DIMENSION, 256, generator=torch.Generator().manual_seed(1))
    return SIGReg()(embeddings, directions).item()


def _gaussian(seed: int = 0) -> torch.Tensor:
    return torch.randn(SAMPLES, DIMENSION, generator=torch.Generator().manual_seed(seed))


def test_standard_gaussian_is_near_zero() -> None:
    assert _statistic(_gaussian()) < 1e-3


def test_constant_vectors_score_high() -> None:
    assert _statistic(torch.ones(SAMPLES, DIMENSION) / DIMENSION**0.5) > 0.1


def test_gaussian_mixture_scores_high() -> None:
    generator = torch.Generator().manual_seed(2)
    centres = torch.randn(2, DIMENSION, generator=generator)
    centres = 2.0 * centres / centres.norm(dim=1, keepdim=True)
    labels = torch.randint(2, (SAMPLES,), generator=generator)

    mixture = 0.3 * _gaussian() + centres[labels]

    assert _statistic(mixture) > 50 * _statistic(_gaussian(seed=3))


def test_wrong_scale_scores_high() -> None:
    assert _statistic(0.1 * _gaussian()) > 50 * _statistic(_gaussian(seed=3))


def test_statistic_shrinks_with_more_samples_on_gaussians() -> None:
    directions = random_directions(DIMENSION, 256, generator=torch.Generator().manual_seed(1))
    sigreg = SIGReg()
    small = torch.randn(256, DIMENSION, generator=torch.Generator().manual_seed(4))
    large = torch.randn(16_384, DIMENSION, generator=torch.Generator().manual_seed(5))

    assert sigreg(large, directions) < sigreg(small, directions)


def test_computation_is_float32_even_for_half_precision_input() -> None:
    directions = random_directions(DIMENSION, 16)

    value = SIGReg()(_gaussian().half(), directions)

    assert value.dtype == torch.float32
    assert torch.isfinite(value)
