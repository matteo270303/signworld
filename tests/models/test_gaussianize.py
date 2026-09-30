"""The isotropy maps: each is invertible and moves synthetic data towards N(0, I)."""

import torch

from signworld.metrics.sigreg import SIGReg, random_directions
from signworld.models.gaussianize import (
    IterativeGaussianization,
    MarginalGaussianization,
    Prefix,
    Whitening,
    fit_flow,
)


def _skewed(rows: int, dimension: int, seed: int = 0) -> torch.Tensor:
    """Correlated, skewed, two-cluster data: far from any Gaussian."""
    generator = torch.Generator().manual_seed(seed)
    exponential = -torch.log(torch.rand(rows, dimension, generator=generator))
    cluster = (torch.rand(rows, 1, generator=generator) < 0.3).float() * 4.0
    mixing = torch.randn(dimension, dimension, generator=generator)
    return (exponential + cluster) @ mixing


def _sigreg(x: torch.Tensor) -> float:
    directions = random_directions(x.shape[1], 64, generator=torch.Generator().manual_seed(1))
    centred = x - x.mean(dim=0)
    scaled = centred * (x.shape[1] / centred.pow(2).sum(dim=1).mean()).sqrt()
    return float(SIGReg()(scaled, directions))


def test_whitening_gives_identity_covariance_and_inverts() -> None:
    x = _skewed(20_000, 6)

    whitening = Whitening.fit(x)
    y = whitening.forward(x)

    assert torch.allclose(torch.cov(y.T), torch.eye(6), atol=1e-3)
    assert torch.allclose(whitening.inverse(y), x, atol=1e-3)


def test_marginal_gaussianization_is_normal_per_coordinate_and_invertible() -> None:
    x = _skewed(20_000, 4)

    marginal = MarginalGaussianization.fit(x, knots=256)
    y = marginal.forward(x)

    assert torch.allclose(y.mean(dim=0), torch.zeros(4), atol=0.05)
    assert torch.allclose(y.std(dim=0), torch.ones(4), atol=0.05)
    outside = x * 1.5  # beyond the fitted range: the linear extension still inverts
    assert torch.allclose(marginal.inverse(marginal.forward(outside)), outside, atol=1e-3)


def test_rbig_and_sinf_approach_a_gaussian_and_invert() -> None:
    x = _skewed(10_000, 6)
    before = _sigreg(x)

    for rotation in ("pca", "sliced"):
        model = IterativeGaussianization.fit(
            x, 10, rotation, knots=256, directions=3, steps=30, rows=4096, seed=0
        )
        y = model.forward(x)

        assert _sigreg(y) < 0.2 * before, rotation
        assert torch.allclose(model.inverse(y), x, atol=1e-2 * x.abs().max()), rotation
        assert torch.equal(Prefix(model, 3).forward(x), model.forward(x, 3))
        assert len(model.seconds) == 10


def test_flow_lowers_its_likelihood_and_inverts() -> None:
    x = _skewed(4096, 6)
    flow, fit = fit_flow(x[:3072], x[3072:], depth=2, hidden=32, batch_size=512, max_epochs=5)

    assert fit.epochs >= 1
    initial = flow.__class__(6, 2, 32)
    initial.mean.copy_(x[:3072].mean(dim=0))
    initial.scale.copy_(x[:3072].std(dim=0))
    with torch.no_grad():
        assert flow.negative_log_likelihood(x[3072:]) < initial.negative_log_likelihood(x[3072:])
    z = flow.forward_map(x[3072:])
    assert torch.allclose(flow.inverse_map(z), x[3072:], atol=1e-3 * x.abs().max())
