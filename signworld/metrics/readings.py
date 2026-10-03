"""The measurements behind the alarms of §4.13.3, as pure functions of tensors.

Each function reads quantities the training step already produces (the pose target ``s``, the
physical read-outs, ŷ, ẽ, the query outputs, the LoRA weights and gradients) or a small extra
forward on a fixed batch. The monitor (``monitor.py``) decides when to call them, compares
them with their step-0 reference and raises the alarms; the thresholds are starting points to
calibrate in PC7.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.loss.sigreg import SIGReg, random_directions
from signworld.models.worldsign import lora
from signworld.models.worldsign.physical import PhysicalPrediction

from ..metrics.geometry import centered, effective_rank, isoscore, mean_dimension_std

COVERAGE_BINS = (0.0, 0.25, 0.5, 0.75, 1.0001)
"""Bins of the mask coverage rho of a box, for the read-out R² per coverage."""


# ------------------------------------------------------------------ generic


def weighted_r2(predicted: Tensor, target: Tensor, weights: Tensor) -> float:
    """``1 - Σ w‖p - t‖² / Σ w‖t - t̄‖²`` over rows (n, d) with weights (n,)."""
    w = weights.float()
    if float(w.sum()) <= 0:
        return float("nan")
    t, p = target.float(), predicted.float()
    mean = (w[:, None] * t).sum(0) / w.sum()
    residual = (w * (p - t).pow(2).sum(-1)).sum()
    total = (w * (t - mean).pow(2).sum(-1)).sum()
    return float(1 - residual / total) if float(total) > 0 else float("nan")


def variance_ratio(predicted: Tensor, target: Tensor, weights: Tensor | None = None) -> float:
    """gamma = tr Cov(predicted) / tr Cov(target): below ~0.3 the predictor answers the mean."""
    w = torch.ones(len(target), device=target.device) if weights is None else weights.float()
    if float(w.sum()) <= 0:
        return float("nan")

    def spread(x: Tensor) -> Tensor:
        x = x.float()
        mean = (w[:, None] * x).sum(0) / w.sum()
        return (w * (x - mean).pow(2).sum(-1)).sum() / w.sum()

    total = spread(target)
    return float(spread(predicted) / total) if float(total) > 0 else float("nan")


def correlation(predicted: Tensor, target: Tensor) -> float:
    p, t = predicted.float().flatten(), target.float().flatten()
    p, t = p - p.mean(), t - t.mean()
    denominator = p.norm() * t.norm()
    return float(p @ t / denominator) if float(denominator) > 0 else float("nan")


def spearman(a: Tensor, b: Tensor) -> float:
    """Rank correlation of two vectors of the same length."""
    ranks = [x.float().argsort().argsort().float() for x in (a, b)]
    return correlation(*ranks)


def linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA of two sets of rows (n, d) and (n, d') [Lett. 91]."""
    x, y = centered(x.double()), centered(y.double())
    cross = (x.T @ y).pow(2).sum()
    denominator = (x.T @ x).norm() * (y.T @ y).norm()
    return float(cross / denominator) if float(denominator) > 0 else float("nan")


def sigreg_ratio(rows: Tensor, directions: int = 256, seed: int = 0) -> float:
    """SIGReg of the rows, as the loss computes it, over its value for N(0, I) (≈ 1.06).

    No standardisation: SIGReg also judges the mean and the scale, and subtracting the sample
    mean would lower a Gaussian's value below 1.06.
    """
    if len(rows) < 2:  # noqa: PLR2004
        return float("nan")
    x = rows.float()
    slices = random_directions(
        x.shape[1], directions, generator=torch.Generator().manual_seed(seed)
    ).to(x.device)
    return float(SIGReg()(x, slices)) / (math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3))


def ridge_r2(
    features: Tensor, targets: Tensor, weights: Tensor, fit: Tensor, penalty: float = 1e-3
) -> float:
    """R² on the rows not in ``fit`` of a weighted ridge from ``features`` to ``targets``.

    ``features`` (n, d), ``targets`` (n, k), ``weights`` (n,), ``fit`` (n,) bool. The penalty is
    relative to the mean eigenvalue of the standardised design.
    """
    x, y, w = features.double(), targets.double(), weights.double()
    mean, std = x[fit].mean(0), x[fit].std(0).clamp_min(1e-8)
    x = torch.cat([(x - mean) / std, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)
    xf, yf, wf = x[fit], y[fit], w[fit][:, None]
    gram = xf.T @ (wf * xf)
    ridge = (
        penalty * gram.diagonal()[:-1].mean() * torch.eye(len(gram), dtype=x.dtype, device=x.device)
    )
    ridge[-1, -1] = 0.0
    coefficients = torch.linalg.solve(gram + ridge, xf.T @ (wf * yf))
    test = ~fit
    return weighted_r2(x[test] @ coefficients, y[test], w[test])


@dataclass(frozen=True, slots=True)
class Spread:
    """Collapse reading of a set of vectors: rank and spread (§4.13.3)."""

    effective_rank: float
    std: float
    isoscore: float

    @classmethod
    def of(cls, rows: Tensor) -> "Spread":
        x = rows.float()
        if len(x) < 2:  # noqa: PLR2004
            return cls(float("nan"), float("nan"), float("nan"))
        return cls(effective_rank(centered(x)), mean_dimension_std(x), isoscore(x))


# ------------------------------------------------------------------ physical level


def physical_readings(
    predictions: Sequence[PhysicalPrediction], latent: Tensor, confidence: Tensor
) -> dict[str, float]:
    """The per-step read-outs ŝ_t against LN(s_t): gamma, R² and correlation (gerarchia §4).

    Steps are told apart by the share of their boxes' tokens the mask hides (the coverage):
    overall, mostly hidden (coverage ≥ 0.5, ``*_masked``), mostly visible (``r2_visible``) and
    by coverage bin. The dynamics test reads the mostly hidden steps against the baseline «mean
    of LN(s_t) over the mostly visible steps of the same clip». ``excluded_part{a}``: the share
    of steps whose articulator ``a`` has no visible box.

    ``latent`` (batch, steps, C), the pose target; ``confidence`` (batch, steps).
    """
    target = functional.layer_norm(latent.float(), (latent.shape[-1],)).flatten(0, 1)
    out: dict[str, float] = {}
    rows: dict[str, list[Tensor]] = {"p": [], "t": [], "w": [], "hidden": [], "seen": []}
    per_coverage: dict[int, list[float]] = {}
    dynamics, baseline = [], []
    for prediction in predictions:
        coverage = prediction.coverage
        occupied = (prediction.box_tokens.sum(-1) > 0).float() * confidence
        hidden = (coverage >= 0.5).float() * occupied  # noqa: PLR2004 (mostly hidden)
        seen = (coverage < 0.5).float() * occupied  # noqa: PLR2004
        state = prediction.state.float().flatten(0, 1)
        for name, value in (
            ("p", state),
            ("t", target),
            ("w", occupied.flatten()),
            ("hidden", hidden.flatten()),
            ("seen", seen.flatten()),
        ):
            rows[name].append(value)
        for index, (low, high) in enumerate(pairwise(COVERAGE_BINS)):
            inside = ((coverage >= low) & (coverage < high)).float() * occupied
            per_coverage.setdefault(index, []).append(weighted_r2(state, target, inside.flatten()))
        normalized = target.view_as(prediction.state)
        mean_seen = (seen[..., None] * normalized).sum(1, keepdim=True) / seen.sum(1, keepdim=True)[
            ..., None
        ].clamp_min(1e-6)
        has_seen = (seen.sum(1, keepdim=True) > 0).float()
        weight = (hidden * has_seen).flatten()
        dynamics.append(weighted_r2(state, target, weight))
        baseline.append(weighted_r2(mean_seen.expand_as(normalized).flatten(0, 1), target, weight))
        for part in range(prediction.box_tokens.shape[-1]):
            excluded = (prediction.box_tokens[..., part] == 0).float().mean()
            out.setdefault(f"excluded_part{part}", float(excluded))
    joined = {k: torch.cat(v) for k, v in rows.items()}
    out["gamma_masked"] = variance_ratio(joined["p"], joined["t"], joined["hidden"])
    out["r2"] = weighted_r2(joined["p"], joined["t"], joined["w"])
    out["r2_masked"] = weighted_r2(joined["p"], joined["t"], joined["hidden"])
    out["r2_visible"] = weighted_r2(joined["p"], joined["t"], joined["seen"])
    hidden_rows = joined["hidden"] > 0
    out["correlation_masked"] = correlation(joined["p"][hidden_rows], joined["t"][hidden_rows])
    out |= {f"masked_r2_coverage{i}": _mean(values) for i, values in per_coverage.items()}
    out["dynamics_r2"] = _mean(dynamics)
    out["dynamics_baseline_r2"] = _mean(baseline)
    return out


def _mean(values: Iterable[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def keypoint_errors(
    predictions: Sequence[PhysicalPrediction],
    latent: Tensor,
    decode: nn.Module,
    keypoints: Tensor,
    weights: Tensor,
) -> dict[str, float]:
    """Keypoints decoded from the read-outs of mostly hidden steps, against interpolation and
    constant velocity from the mostly visible steps.

    The read-out predicts LN(s); it is brought back to the scale of ``s`` with ``s``'s own mean
    and spread before the anchor's decoder. Errors are mean distances in shoulder units over
    the present joints of the mostly hidden steps (§4.13.3, «Errore in keypoint»).

    ``latent`` (batch, steps, C); ``keypoints`` (batch, steps, 69, 2); ``weights``
    (batch, steps, 69).
    """
    mean = latent.float().mean(-1, keepdim=True)
    std = latent.float().var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    truth = keypoints.float()
    present = weights > 0
    errors: dict[str, list[float]] = {"model": [], "interpolation": [], "constant_velocity": []}
    for prediction in predictions:
        restored = prediction.state.float() * std + mean
        with torch.no_grad():
            decoded = decode(restored).float()
        hidden_step = prediction.coverage >= 0.5  # noqa: PLR2004 (batch, steps)
        hidden = hidden_step[..., None] & present
        seen = ~hidden_step[..., None] & present
        interpolated, extrapolated = _temporal_baselines(truth, seen)
        for name, estimate in (
            ("model", decoded),
            ("interpolation", interpolated),
            ("constant_velocity", extrapolated),
        ):
            distance = (estimate - truth).norm(dim=-1)
            usable = hidden & torch.isfinite(distance)
            if bool(usable.any()):
                errors[name].append(float(distance[usable].mean()))
    return {f"keypoint_error_{name}": _mean(values) for name, values in errors.items()}


def _temporal_baselines(truth: Tensor, seen: Tensor) -> tuple[Tensor, Tensor]:
    """Per joint over steps: linear interpolation between seen steps (held flat before the
    first and after the last, as ``np.interp``), and constant velocity from the last two seen
    steps before (NaN where no estimate exists).

    ``truth`` (batch, steps, joints, 2), ``seen`` (batch, steps, joints) bool.
    """
    steps = truth.shape[1]
    time = torch.arange(steps, device=truth.device)[None, :, None]
    last = torch.where(seen, time, -1).cummax(dim=1).values  # last seen step <= t, or -1
    after = torch.where(seen, time, steps).flip(1).cummin(dim=1).values.flip(1)  # first >= t

    def at(index: Tensor) -> Tensor:
        return truth.gather(1, index.clamp(0, steps - 1)[..., None].expand_as(truth))

    before, next_ = at(last), at(after)
    span = (after - last).clamp_min(1)[..., None].float()
    share = ((time - last)[..., None].float() / span).clamp(0, 1)
    interpolated = before + (next_ - before) * share
    interpolated = torch.where((last < 0)[..., None], next_, interpolated)
    interpolated = torch.where((after >= steps)[..., None], before, interpolated)
    interpolated = torch.where(((last < 0) & (after >= steps))[..., None], torch.nan, interpolated)

    minus_one = torch.full_like(last[:, :1], -1)
    second = torch.cat([minus_one, last[:, :-1]], dim=1)  # last seen step < t
    first = torch.where(second >= 1, last.gather(1, (second - 1).clamp_min(0)), -1)
    velocity = (at(second) - at(first)) / (second - first).clamp_min(1)[..., None].float()
    extrapolated = at(second) + velocity * (time - second)[..., None].float()
    extrapolated = torch.where((first >= 0)[..., None], extrapolated, torch.nan)
    return interpolated, extrapolated


# ------------------------------------------------------------------ semantic level


def semantic_readings(
    predicted: Tensor, target: Tensor, languages: Tensor, names: Sequence[str]
) -> dict[str, float]:
    """gamma_sem = tr Cov(ŷ)/tr Cov(ẽ) and R² = 1 - mean‖ẽ - ŷ‖²/tr Cov(ẽ), overall, per language.

    With K hypotheses, each clip is read through its best one.
    """
    best = predicted[torch.arange(len(target)), _best_hypothesis(predicted, target)]
    out = {
        "gamma_sem": variance_ratio(best, target),
        "r2_sem": weighted_r2(best, target, torch.ones(len(target), device=target.device)),
    }
    for index, name in enumerate(names):
        rows = languages == index
        if int(rows.sum()) > 1:
            out[f"r2_sem_{name}"] = weighted_r2(
                best[rows], target[rows], torch.ones(int(rows.sum()), device=target.device)
            )
            out[f"e_sem_{name}"] = float(
                (1 - functional.cosine_similarity(best[rows], target[rows], dim=-1)).mean()
            )
    return out


def _best_hypothesis(predicted: Tensor, target: Tensor) -> Tensor:
    return functional.cosine_similarity(predicted, target[:, None], dim=-1).argmax(dim=1)


def query_cosine(queries: Tensor) -> float:
    """Mean cosine between the outputs of different queries of a clip: ≈ 1 means collapsed."""
    unit = functional.normalize(queries.float(), dim=-1)
    gram = unit @ unit.transpose(1, 2)
    count = queries.shape[1]
    off = gram.sum(dim=(1, 2)) - gram.diagonal(dim1=1, dim2=2).sum(-1)
    return float((off / (count * (count - 1))).mean())


def text_head_spearman(centred: Tensor, target: Tensor) -> float:
    """Rank correlation of the pairwise similarities before and after the text head."""
    upper = torch.triu_indices(len(target), len(target), offset=1)

    def similarities(x: Tensor) -> Tensor:
        unit = functional.normalize(x.float(), dim=-1)
        return (unit @ unit.T)[upper[0], upper[1]]

    return spearman(similarities(centred), similarities(target))


def modality_gap(predicted: Tensor, target: Tensor, steps: int = 200) -> float:
    """Held-out accuracy of a logistic classifier telling ŷ from ẽ (> 0.95: separate spaces)."""
    x = torch.cat([predicted.float(), target.float()]).detach()
    y = torch.cat([torch.zeros(len(predicted)), torch.ones(len(target))]).to(x.device)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0)).to(x.device)
    x, y = x[order], y[order]
    half = len(x) // 2
    mean, std = x[:half].mean(0), x[:half].std(0).clamp_min(1e-6)
    x = (x - mean) / std
    weight = torch.zeros(x.shape[1] + 1, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], max_iter=steps)
    design = torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = design[:half] @ weight
        loss = (
            functional.binary_cross_entropy_with_logits(logits, y[:half])
            + 1e-3 * weight.pow(2).sum()
        )
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        optimizer.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        predictions = (design[half:] @ weight > 0).float()
    return float((predictions == y[half:]).float().mean())


def energy_table(physical: Tensor, semantic: Tensor) -> dict[str, float]:
    """The 2x2 table: shares of clips with each combination of low/high E_fis and E_sem."""
    high_fis = physical > physical.median()
    high_sem = semantic > semantic.median()
    table = {
        "low_fis_low_sem": ~high_fis & ~high_sem,
        "low_fis_high_sem": ~high_fis & high_sem,
        "high_fis_low_sem": high_fis & ~high_sem,
        "high_fis_high_sem": high_fis & high_sem,
    }
    return {f"energy_{k}": float(v.float().mean()) for k, v in table.items()}


def attention_readings(attention: Tensor, members: Tensor) -> dict[str, float]:
    """Entropy of the queries' attention over the video (1 = uniform) and its share on boxes.

    ``attention`` (batch, heads, queries, N); ``members`` (batch, steps, parts, rows, columns).
    """
    tokens = attention.shape[-1]
    entropy = -(attention.clamp_min(1e-12).log() * attention).sum(-1) / math.log(tokens)
    inside = members.amax(dim=2).flatten(1).float()  # (batch, N): 1 inside any box
    share = torch.einsum("bhqn,bn->bhq", attention.float(), inside)
    area = inside.mean(dim=1)
    return {
        "attention_entropy": float(entropy.mean()),
        "attention_on_articulators": float(share.mean()),
        "articulator_area": float(area.mean()),
    }


# ------------------------------------------------------------------ adapters and gradients


def lora_ratios(module: nn.Module) -> dict[str, float]:
    """``‖ΔW‖ / ‖W‖`` of every LoRA layer under ``module``, keyed by its name."""
    out: dict[str, float] = {}
    for name, layer in module.named_modules():
        if isinstance(layer, lora.LoRALinear):
            delta = torch.cat([b @ a for a, b in zip(layer.down, layer.up, strict=True)], 0)
            base = layer.base.weight
            out[name] = float(layer.scale * delta.norm() / base.norm())
        elif isinstance(layer, lora.LoRAWeight):
            delta = torch.cat([b @ a for a, b in zip(layer.down, layer.up, strict=True)], 0)
            out[name] = layer.scale * float(delta.norm())
    return out


def gradient_norms(module: nn.Module) -> dict[str, float]:
    """Norm of the gradient of every LoRA layer under ``module`` (after a backward)."""
    out = {}
    for name, layer in module.named_modules():
        if isinstance(layer, lora.LoRALinear | lora.LoRAWeight):
            grads = [p.grad for p in layer.adapter_parameters() if p.grad is not None]
            if grads:
                out[name] = float(torch.stack([g.float().norm() for g in grads]).norm())
    return out


def term_gradients(
    terms: dict[str, Tensor], weights: dict[str, float], parameters: Sequence[nn.Parameter]
) -> dict[str, Tensor]:
    """The flat gradient of each weighted term with respect to ``parameters`` (no .grad)."""
    out = {}
    for name, value in terms.items():
        if not value.requires_grad:
            continue
        grads = torch.autograd.grad(
            weights[name] * value, list(parameters), retain_graph=True, allow_unused=True
        )
        flat = [
            g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, parameters, strict=True)
        ]
        out[name] = torch.cat(flat).float()
    return out


def shares_and_cosines(gradients: dict[str, Tensor], prefix: str) -> dict[str, float]:
    """Share of the total gradient norm of each term, and the cosine of every pair."""
    norms = {k: float(v.norm()) for k, v in gradients.items()}
    total = sum(norms.values())
    out = {
        f"{prefix}_share_{k}": (n / total if total > 0 else float("nan")) for k, n in norms.items()
    }
    names = sorted(gradients)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = gradients[first], gradients[second]
            denominator = a.norm() * b.norm()
            out[f"{prefix}_cos_{first}_{second}"] = (
                float(a @ b / denominator) if float(denominator) > 0 else float("nan")
            )
    return out
