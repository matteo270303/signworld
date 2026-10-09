"""Collaudo "testare i test" (§4.13.1): SIGReg, as LeJEPA defines it, on known cases.

With the factor N a true Gaussian sits near a constant (≈ 1.06) whatever the batch size, and
any other distribution grows with N.
"""

import math

import pytest
import torch

from signworld.loss.sigreg import SIGReg, random_directions

DIMENSION = 128
SAMPLES = 4096
GAUSSIAN_EXPECTATION = math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)
"""∫ (1 - exp(-t²)) exp(-t²/2) dt over the real line: E[N·|φ̂ - φ|²] integrated."""


def _statistic(embeddings: torch.Tensor) -> float:
    directions = random_directions(DIMENSION, 256, generator=torch.Generator().manual_seed(1))
    return SIGReg()(embeddings, directions).item()


def _gaussian(seed: int = 0, samples: int = SAMPLES) -> torch.Tensor:
    return torch.randn(samples, DIMENSION, generator=torch.Generator().manual_seed(seed))


def test_standard_gaussian_sits_at_its_expectation() -> None:
    assert _statistic(_gaussian()) == pytest.approx(GAUSSIAN_EXPECTATION, rel=0.15)


def test_the_gaussian_value_does_not_depend_on_the_batch_size() -> None:
    small = _statistic(_gaussian(seed=4, samples=256))
    large = _statistic(_gaussian(seed=5, samples=16_384))

    assert small == pytest.approx(large, rel=0.2)


def test_a_non_gaussian_batch_grows_with_its_size() -> None:
    narrow = 0.5 * _gaussian(seed=6, samples=8192)  # the wrong scale is not N(0, I)

    assert _statistic(narrow) > 3 * _statistic(narrow[:2048])


def test_constant_vectors_score_high() -> None:
    assert _statistic(torch.ones(SAMPLES, DIMENSION) / DIMENSION**0.5) > 100 * GAUSSIAN_EXPECTATION


def test_gaussian_mixture_scores_high() -> None:
    generator = torch.Generator().manual_seed(2)
    centres = torch.randn(2, DIMENSION, generator=generator)
    centres = 2.0 * centres / centres.norm(dim=1, keepdim=True)
    labels = torch.randint(2, (SAMPLES,), generator=generator)

    mixture = 0.3 * _gaussian() + centres[labels]

    assert _statistic(mixture) > 50 * _statistic(_gaussian(seed=3))


def test_wrong_scale_scores_high() -> None:
    assert _statistic(0.1 * _gaussian()) > 50 * _statistic(_gaussian(seed=3))


def test_the_sample_count_can_be_given_for_gathered_statistics() -> None:
    directions = random_directions(DIMENSION, 16, generator=torch.Generator().manual_seed(1))
    x = _gaussian()

    assert SIGReg()(x, directions, samples=2 * SAMPLES) == pytest.approx(
        2 * SIGReg()(x, directions).item()
    )


def test_computation_is_float32_even_for_half_precision_input() -> None:
    directions = random_directions(DIMENSION, 16)

    value = SIGReg()(_gaussian().half(), directions)

    assert value.dtype == torch.float32
    assert torch.isfinite(value)
