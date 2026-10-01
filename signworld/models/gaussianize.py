"""Maps that make a latent isotropic, or Gaussian, fitted once on training rows (§4.4.3).

Three families, from the least to the most work:

* ``Whitening``: one matrix, ``(Σ + εI)^(-1/2)`` in ZCA form. The covariance becomes the
  identity; the shape of the distribution does not become Gaussian.
* ``IterativeGaussianization``: repeated marginal Gaussianization (every coordinate through
  its own CDF and the inverse normal CDF) and rotation. With PCA rotations it is RBIG
  (Laparra, Camps-Valls and Malo, 2011); with the directions whose projections are least
  Gaussian it follows SINF (Dai and Seljak, 2021), which Gaussianizes only along those.
* ``RealNVPFlow``: an invertible network of affine couplings trained by maximum likelihood
  towards N(0, I), as BERT-flow does for sentence embeddings (Li et al., 2020).

Every map is invertible, so none removes information; what they change is how linearly
readable it stays and how distances look, which is what the isotropy test measures.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor, nn


class Transform(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...

    def inverse(self, y: Tensor) -> Tensor: ...

    def parameters_count(self) -> int: ...


# --------------------------------------------------------------------------- linear


@dataclass
class Identity:
    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, y: Tensor) -> Tensor:
        return y

    def parameters_count(self) -> int:
        return 0


@dataclass
class Standardize:
    """Per-coordinate zero mean and unit variance."""

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, x: Tensor) -> "Standardize":
        return cls(x.mean(dim=0), x.std(dim=0).clamp_min(1e-8))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.scale

    def inverse(self, y: Tensor) -> Tensor:
        return y * self.scale + self.mean

    def parameters_count(self) -> int:
        return 2 * self.mean.numel()


@dataclass
class Whitening:
    """ZCA whitening with shrinkage: ``(x - μ) V (Λ + ε·mean(Λ))^(-1/2) Vᵀ``."""

    mean: Tensor
    matrix: Tensor
    inverse_matrix: Tensor

    @classmethod
    def fit(cls, x: Tensor, shrinkage: float = 0.0) -> "Whitening":
        rows = x.double()
        mean = rows.mean(dim=0)
        covariance = torch.cov((rows - mean).T)
        values, vectors = torch.linalg.eigh(covariance)
        values = values.clamp_min(0.0) + shrinkage * values.clamp_min(0.0).mean()
        values = values.clamp_min(1e-12 * values.max())
        matrix = vectors @ torch.diag(values.rsqrt()) @ vectors.T
        inverse = vectors @ torch.diag(values.sqrt()) @ vectors.T
        return cls(mean.to(x.dtype), matrix.to(x.dtype), inverse.to(x.dtype))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) @ self.matrix

    def inverse(self, y: Tensor) -> Tensor:
        return y @ self.inverse_matrix + self.mean

    def parameters_count(self) -> int:
        return self.matrix.numel() + self.mean.numel()


# --------------------------------------------------------------------------- marginal


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Column-wise piecewise-linear map through the knots (xp, fp), extrapolated linearly.

    ``x`` is (rows, columns); ``xp`` and ``fp`` are (columns, knots), ``xp`` increasing.
    """
    values = x.T.contiguous()
    index = torch.searchsorted(xp, values).clamp(1, xp.shape[1] - 1)
    x0, x1 = xp.gather(1, index - 1), xp.gather(1, index)
    f0, f1 = fp.gather(1, index - 1), fp.gather(1, index)
    t = (values - x0) / (x1 - x0)
    return (f0 + t * (f1 - f0)).T


@dataclass
class MarginalGaussianization:
    """Each coordinate through its empirical CDF and the inverse normal CDF.

    The map is monotone and piecewise linear between ``knots`` quantiles, extended linearly
    beyond them, so it is exactly invertible, also outside the range seen in fitting.
    """

    quantiles: Tensor
    """(columns, knots) data values at the knots, strictly increasing."""
    normal: Tensor
    """(knots,) standard-normal values at the same probabilities."""

    @classmethod
    def fit(cls, x: Tensor, knots: int = 1024) -> "MarginalGaussianization":
        probabilities = (torch.arange(knots, dtype=x.dtype, device=x.device) + 0.5) / knots
        ordered = x.sort(dim=0).values
        positions = (probabilities * (len(x) - 1)).round().long()
        quantiles = ordered[positions].T.contiguous()
        # Ties would make the map flat and not invertible: nudge them into a strict order.
        spread = (quantiles[:, -1] - quantiles[:, 0]).clamp_min(1e-6)[:, None]
        step = 1e-6 * spread * torch.arange(knots, dtype=x.dtype, device=x.device)
        quantiles = torch.cummax(quantiles, dim=1).values + step
        return cls(quantiles, torch.special.ndtri(probabilities))

    def forward(self, x: Tensor) -> Tensor:
        return _interp(x, self.quantiles, self.normal.expand_as(self.quantiles).contiguous())

    def inverse(self, y: Tensor) -> Tensor:
        normal = self.normal.expand_as(self.quantiles).contiguous()
        return _interp(y, normal, self.quantiles)

    def parameters_count(self) -> int:
        return self.quantiles.numel()


def gaussian_quantiles(count: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    probabilities = (torch.arange(count, dtype=dtype, device=device) + 0.5) / count
    quantiles: Tensor = torch.special.ndtri(probabilities)
    return quantiles


def least_gaussian_directions(
    x: Tensor, count: int, steps: int, rows: int, generator: torch.Generator
) -> Tensor:
    """(dimension, count) orthonormal directions whose projections are furthest from N(0, 1).

    Maximises the summed squared 1-D Wasserstein distance between the sorted projections and
    the normal quantiles, by Adam on the matrix with a QR retraction onto orthonormal columns,
    as max-sliced Wasserstein in SINF.
    """
    sample = x[torch.randperm(len(x), generator=generator, device="cpu")[:rows].to(x.device)]
    start = torch.randn(x.shape[1], count, generator=generator).to(x)
    directions = nn.Parameter(torch.linalg.qr(start).Q)
    target = gaussian_quantiles(len(sample), x.dtype, x.device)[:, None]
    optimizer = torch.optim.Adam([directions], lr=0.02)
    for _ in range(steps):
        projected = (sample @ directions).sort(dim=0).values
        loss = -(projected - target).pow(2).mean(dim=0).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            directions.copy_(torch.linalg.qr(directions).Q)
    return directions.detach()


@dataclass
class _Layer:
    """One iteration: Gaussianize along ``directions`` (all axes when it is a rotation)."""

    marginal: MarginalGaussianization
    directions: Tensor
    """(dimension, k) orthonormal columns; k = dimension for RBIG's full rotation."""
    full: bool

    def forward(self, x: Tensor) -> Tensor:
        if self.full:
            return self.marginal.forward(x) @ self.directions
        projected = x @ self.directions
        return x + (self.marginal.forward(projected) - projected) @ self.directions.T

    def inverse(self, y: Tensor) -> Tensor:
        if self.full:
            return self.marginal.inverse(y @ self.directions.T)
        projected = y @ self.directions
        return y + (self.marginal.inverse(projected) - projected) @ self.directions.T


@dataclass
class IterativeGaussianization:
    """RBIG (``rotation="pca"``) or SINF (``rotation="sliced"``), prefix-evaluable.

    ``forward(x, iterations)`` applies only the first ``iterations`` layers, so one fit gives
    the whole curve of the test.
    """

    standardize: Standardize
    layers: list[_Layer] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    """Cumulative fitting time after each layer."""

    @classmethod
    def fit(  # noqa: PLR0913 (the data and the knobs of the two variants)
        cls,
        x: Tensor,
        iterations: int,
        rotation: Literal["pca", "sliced"] = "pca",
        *,
        knots: int = 1024,
        directions: int = 32,
        steps: int = 100,
        rows: int = 16384,
        seed: int = 0,
    ) -> "IterativeGaussianization":
        generator = torch.Generator().manual_seed(seed)
        started = time.monotonic()
        standardize = Standardize.fit(x)
        current = standardize.forward(x)
        layers, seconds = [], []
        for _ in range(iterations):
            if rotation == "pca":
                marginal = MarginalGaussianization.fit(current, knots)
                gaussian = marginal.forward(current)
                _, _, vectors = torch.linalg.svd(
                    gaussian - gaussian.mean(dim=0), full_matrices=False
                )
                layer = _Layer(marginal, vectors.T.contiguous(), full=True)
            else:
                axes = least_gaussian_directions(current, directions, steps, rows, generator)
                marginal = MarginalGaussianization.fit(current @ axes, knots)
                layer = _Layer(marginal, axes, full=False)
            current = layer.forward(current)
            layers.append(layer)
            seconds.append(time.monotonic() - started)
        return cls(standardize, layers, seconds)

    def forward(self, x: Tensor, iterations: int | None = None) -> Tensor:
        y = self.standardize.forward(x)
        for layer in self.layers[:iterations]:
            y = layer.forward(y)
        return y

    def inverse(self, y: Tensor, iterations: int | None = None) -> Tensor:
        for layer in reversed(self.layers[:iterations]):
            y = layer.inverse(y)
        return self.standardize.inverse(y)

    def parameters_count(self, iterations: int | None = None) -> int:
        layers = self.layers[:iterations]
        return self.standardize.parameters_count() + sum(
            layer.marginal.parameters_count() + layer.directions.numel() for layer in layers
        )


@dataclass
class Prefix:
    """The first ``iterations`` layers of an iterative Gaussianization, as a Transform."""

    model: IterativeGaussianization
    iterations: int

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x, self.iterations)

    def inverse(self, y: Tensor) -> Tensor:
        return self.model.inverse(y, self.iterations)

    def parameters_count(self) -> int:
        return self.model.parameters_count(self.iterations)


# --------------------------------------------------------------------------- flow


class _Coupling(nn.Module):
    """Affine coupling: the first half conditions a scale and shift of the second half."""

    permutation: Tensor
    undo: Tensor

    def __init__(self, dimension: int, hidden: int, permutation: Tensor) -> None:
        super().__init__()
        self.register_buffer("permutation", permutation)
        self.register_buffer("undo", torch.argsort(permutation))
        half = dimension // 2
        self.split = half
        last = nn.Linear(hidden, 2 * (dimension - half))
        # The last layer starts at zero: the flow begins as the identity.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), last
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x[:, self.permutation]
        first, second = x[:, : self.split], x[:, self.split :]
        log_scale, shift = self.net(first).chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)
        y = torch.cat([first, second * log_scale.exp() + shift], dim=1)
        return y, log_scale.sum(dim=1)

    def inverse(self, y: Tensor) -> Tensor:
        first, second = y[:, : self.split], y[:, self.split :]
        log_scale, shift = self.net(first).chunk(2, dim=1)
        x = torch.cat([first, (second - shift) * torch.tanh(log_scale).neg().exp()], dim=1)
        return x[:, self.undo]


class RealNVPFlow(nn.Module):
    """Standardization followed by ``depth`` affine couplings with fixed random permutations."""

    mean: Tensor
    scale: Tensor

    def __init__(self, dimension: int, depth: int, hidden: int, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.couplings = nn.ModuleList(
            _Coupling(dimension, hidden, torch.randperm(dimension, generator=generator))
            for _ in range(depth)
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Latent and log-determinant of the Jacobian (the standardization included)."""
        z = (x - self.mean) / self.scale
        log_det = -self.scale.log().sum().expand(len(x))
        for coupling in self.couplings:
            z, step = coupling(z)
            log_det = log_det + step
        return z, log_det

    def negative_log_likelihood(self, x: Tensor) -> Tensor:
        """Mean NLL per dimension, in nats, under N(0, I)."""
        z, log_det = self.encode(x)
        dimension = x.shape[1]
        log_p = -0.5 * z.pow(2).sum(dim=1) - 0.5 * dimension * math.log(2 * math.pi) + log_det
        return -(log_p.mean() / dimension)

    @torch.no_grad()
    def forward_map(self, x: Tensor) -> Tensor:
        return self.encode(x)[0]

    @torch.no_grad()
    def inverse_map(self, z: Tensor) -> Tensor:
        for coupling in reversed(list(self.couplings)):
            z = coupling.inverse(z)
        return z * self.scale + self.mean


@dataclass(frozen=True, slots=True)
class FlowFit:
    epochs: int
    train_nll: float
    validation_nll: float


def fit_flow(  # noqa: PLR0913 (the data, the network and the schedule)
    train: Tensor,
    validation: Tensor,
    depth: int,
    hidden: int,
    *,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    max_epochs: int = 60,
    patience: int = 5,
    seed: int = 0,
) -> tuple[RealNVPFlow, FlowFit]:
    """Maximum likelihood with Adam; the best epoch on ``validation`` is kept."""
    torch.manual_seed(seed)
    flow = RealNVPFlow(train.shape[1], depth, hidden, seed).to(train.device)
    flow.mean.copy_(train.mean(dim=0))
    flow.scale.copy_(train.std(dim=0).clamp_min(1e-8))
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    best, best_state, best_epoch, waited, train_nll = math.inf, None, 0, 0, math.nan
    for epoch in range(1, max_epochs + 1):
        flow.train()
        order = torch.randperm(len(train), device=train.device)
        losses = []
        for start in range(0, len(train), batch_size):
            loss = flow.negative_log_likelihood(train[order[start : start + batch_size]])
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        flow.eval()
        with torch.no_grad():
            validation_nll = float(
                torch.stack(
                    [
                        flow.negative_log_likelihood(validation[i : i + batch_size])
                        for i in range(0, len(validation), batch_size)
                    ]
                ).mean()
            )
        if validation_nll < best - 1e-4:
            best, best_epoch, waited = validation_nll, epoch, 0
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            train_nll = float(sum(losses) / len(losses))
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        flow.load_state_dict(best_state)
    return flow.eval(), FlowFit(best_epoch, train_nll, best)


@dataclass
class FlowTransform:
    flow: RealNVPFlow

    def forward(self, x: Tensor) -> Tensor:
        return self.flow.forward_map(x)

    def inverse(self, y: Tensor) -> Tensor:
        return self.flow.inverse_map(y)

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.flow.parameters())
