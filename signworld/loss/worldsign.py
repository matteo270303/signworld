"""Energies and regularisers of the objective (§4.5.3-§4.5.9), and their per-arm composition.

``L = (1 - λ_P) · (L_inv + L_anchor) + λ_P · SIGReg_posa
    + (1 - λ) · (E_fis + L_pred_sem) + λ · SIGReg_sem``,
λ_P = 0.04 for the pose (posa §4.2) and λ = 0.04 above it (loss §8.5), with ``L_pred_sem`` and
``SIGReg_sem`` set by the arm of ESP-1:

    A₀   E_sem                                  —
    A    E_sem                                  ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    V    E_sem + 0.5·Σ v + 0.02·Σ c (VICReg)    —
    B₀   E_sem + L_unif / 3                     —
    B    E_sem + L_unif / 3                     ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    C    InfoNCE                                ½ [SIGReg({ŷ}) + SIGReg({ẽ})]

The weights inside an arm come from the closest literature (``LossSettings``): VICReg's
25/25/1, with E_sem in place of its MSE (2·E_sem at unit variance); Wang and Isola's
0.75·L_align + 0.5·L_unif at batch 128, with L_align = 2·E_sem on the sphere. InfoNCE treats
clips with the same caption as positives and drops from its denominators the clips of the
same video and the false negatives of GloFND, read on the frozen caption embeddings.

SIGReg is applied to each modality apart, as LeJEPA applies it to each view, with the same
random directions for both. On the pose it is applied to each of the four views at each step
apart, over the clips present there (``SIGReg_posa``, posa §4.2), as LeWM does: the steps of
one clip are not independent samples. ``L_inv`` and ``SIGReg_posa`` train the pose encoder with
``L_anchor``, which comes from the pose branch. The hierarchy is trained level by level
(gerarchia §3): ``E_fis`` reads the pose target with its gradient stopped, so each term reaches
only its own level and λ matters only between the terms of one level. With the physical level
off (ESP-2) every pose and physical term is absent.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.experiment.train.config import LossSettings, SemanticSettings
from signworld.experiment.train.distributed import SINGLE, Distributed
from signworld.loss.sigreg import SIGReg, random_directions
from signworld.models.worldsign.physical import PhysicalPrediction

POSE_TERMS = ("inv_posa", "anchor", "sigreg_posa")
"""The terms of level 0, the pose (posa §4.2)."""
REGULARIZERS = ("sigreg", "vicreg", "unif")
"""Prefixes of the terms that shape the distribution rather than predict (P13 turns them off)."""


def step_errors(prediction: PhysicalPrediction, target: Tensor) -> Tensor:
    """(batch, steps) L1 error of ŝ_t against ``LN(s_t)``, averaged over the channels.

    As in V-JEPA 2.1 the target is layer-normalised over its channels (no affine parameters)
    and the error is the L1 averaged over the channels (``loss_exp = 1``).
    """
    normalized = functional.layer_norm(target.float(), (target.shape[-1],))
    return (prediction.state.float() - normalized).abs().mean(dim=-1)


def physical_energy(
    predictions: Sequence[PhysicalPrediction],
    target: Tensor,
    confidence: Tensor,
    context_lambda: float,
) -> Tensor:
    """E_fis: the step errors weighted as V-JEPA 2.1 weighs its tokens, averaged over masks.

    A step weighs ``c_t · (n^m_t + λ · w^v_t)``: every masked token in its boxes counts once
    (``L_pred``), every visible one ``λ`` times its weight (``L_ctx``; the weight is 1 in
    V-JEPA 2.1's cooldown, which we follow, ``1 / √d`` in its pre-training) (gerarchia §4).

    ``target`` (batch, steps, C), detached by the caller; ``confidence`` (batch, steps).
    """
    energies = []
    for prediction in predictions:
        weights = prediction.weights(confidence, context_lambda)
        error = step_errors(prediction, target)
        energies.append((weights * error).sum() / weights.sum().clamp_min(1e-12))
    return torch.stack(energies).mean()


def physical_energy_per_clip(
    predictions: Sequence[PhysicalPrediction],
    target: Tensor,
    confidence: Tensor,
    context_lambda: float,
) -> Tensor:
    """(batch,) E_fis of every clip on its own, averaged over the masks: plausibility (§4.5.3)."""
    energies = []
    for prediction in predictions:
        weights = prediction.weights(confidence, context_lambda)
        error = step_errors(prediction, target)
        energies.append((weights * error).sum(1) / weights.sum(1).clamp_min(1e-12))
    return torch.stack(energies).mean(dim=0)


def invariance(views: Tensor, present: Tensor) -> Tensor:
    """``L_inv`` in LeJEPA's form: each view's squared distance from the views' centre,
    averaged over the channels and the views, then over the present steps (posa §4.2).

    ``views`` (V, batch, steps, C); ``present`` (batch, steps) bool. With two views it is
    ``‖s_t - s̃_t‖² / 4C``.
    """
    z = views.float()
    squared = (z - z.mean(dim=0)).pow(2).mean(dim=(0, -1))
    weights = present.float()
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


def semantic_energy(predicted: Tensor, target: Tensor) -> Tensor:
    """(batch,) E_sem = 1 - cos(ŷ, ẽ)."""
    energy: Tensor = 1.0 - functional.cosine_similarity(predicted, target, dim=-1)
    return energy


def free_energy(predicted: Tensor, target: Tensor, relaxation: float) -> Tensor:
    """(batch,) energy over K hypotheses (batch, K, d): relaxed minimum (§4.5.9).

    ``(1 - ε) · min_k E_k + ε · mean_k E_k``: losing hypotheses still get a share ε of the
    gradient, so they are not stranded when every target falls in one Voronoi cell.
    """
    energies = semantic_energy(predicted, target[:, None, :])
    return (1.0 - relaxation) * energies.min(dim=1).values + relaxation * energies.mean(dim=1)


def uniformity(x: Tensor, t: float) -> Tensor:
    """Wang and Isola's ``log E exp(-t‖f(x_i) - f(x_j)‖²)`` on the unit sphere [Lett. 40]."""
    unit = functional.normalize(x, dim=-1)
    distances = torch.pdist(unit).pow(2)
    value: Tensor = torch.logsumexp(-t * distances, dim=0) - torch.log(
        torch.tensor(float(len(distances)), device=x.device)
    )
    return value


def vicreg_variance(x: Tensor, gamma: float, epsilon: float) -> Tensor:
    """VICReg's v(Z): the mean over dimensions of ``max(0, gamma - √(Var(z_j) + ε))``."""
    std = (x.float().var(dim=0) + epsilon).sqrt()
    return functional.relu(gamma - std).mean()


def vicreg_covariance(x: Tensor) -> Tensor:
    """VICReg's c(Z): the squared off-diagonal covariances, summed and divided by d."""
    z = x.float() - x.float().mean(dim=0)
    covariance = z.T @ z / max(len(z) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.pow(2).sum() / x.shape[-1]


def _soft_cross_entropy(logits: Tensor, positive: Tensor, dropped: Tensor) -> Tensor:
    """Mean over the rows of ``-Σ_{j ∈ P_i} log softmax_ij / |P_i|``, softmax over the columns
    that are not dropped (UniCL's target, uniform over the positives)."""
    log_p = functional.log_softmax(logits.masked_fill(dropped, float("-inf")), dim=1)
    share = positive.float() / positive.sum(dim=1, keepdim=True).float()
    per_row = -(share * log_p.masked_fill(~positive, 0.0)).sum(dim=1)
    return per_row.mean()


class InfoNCE(nn.Module):
    """Symmetric InfoNCE with a learnable temperature (arm C).

    * Positives: a clip and its caption; with ``rows``, every clip with the same caption too
      (``multi_positive``), with the target spread evenly over them.
    * Dropped from the denominators: the other clips of the same video (P15) and, with
      ``excluded``, GloFND's false negatives; a positive is never dropped.
    """

    def __init__(
        self, temperature: float, min_temperature: float = 0.01, multi_positive: bool = True
    ) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.min_temperature = min_temperature
        self.multi_positive = multi_positive

    @property
    def temperature(self) -> Tensor:
        """The learnt temperature, floored as CLIP clips its logit scale at 100."""
        return self.log_temperature.exp().clamp_min(self.min_temperature)

    def pairs(
        self, videos: Sequence[str], rows: Tensor | None, excluded: Tensor | None, device: Any
    ) -> tuple[Tensor, Tensor]:
        """(batch, batch) ``positive`` and ``dropped`` masks, video along the rows."""
        size = len(videos)
        positive = torch.eye(size, dtype=torch.bool, device=device)
        if rows is not None and self.multi_positive:
            positive |= rows.to(device)[:, None] == rows.to(device)[None, :]
        same = torch.tensor([[a == b for b in videos] for a in videos], device=device)
        dropped = same
        if excluded is not None:
            dropped = dropped | excluded.to(device)
        return positive, dropped & ~positive

    def logits(self, video: Tensor, text: Tensor) -> Tensor:
        """(batch, batch) cosines over the temperature, video along the rows."""
        cosine = functional.normalize(video, dim=-1) @ functional.normalize(text, dim=-1).T
        return cosine / self.temperature

    def forward(
        self,
        video: Tensor,
        text: Tensor,
        videos: Sequence[str],
        rows: Tensor | None = None,
        excluded: Tensor | None = None,
    ) -> Tensor:
        logits = self.logits(video, text)
        positive, dropped = self.pairs(videos, rows, excluded, logits.device)
        return 0.5 * (
            _soft_cross_entropy(logits, positive, dropped)
            + _soft_cross_entropy(logits.T, positive.T, dropped.T)
        )


class FalseNegatives:
    """GloFND's false negatives, read with thresholds fixed in advance (``text.negatives``).

    Caption ``i`` treats caption ``j`` as a false negative when the cosine of their frozen,
    language-centred EmbeddingGemma vectors reaches ``λ_i``, the k-th largest of its
    similarities to the reference set with k = ceil(alpha · size), i.e. its top alpha share:
    GloFND's fixed point, computed once since the oracle is frozen.
    A pair is dropped when either caption flags the other, as worldSign symmetrised it.
    """

    def __init__(self, thresholds: Tensor) -> None:
        self.thresholds = thresholds.float()

    def mask(self, captions: Tensor, rows: Tensor) -> Tensor:
        """(batch, batch) bool for the centred caption vectors and their store rows.

        In fp32 whatever the autocast: a bf16 cosine keeps ~3 digits, too few against a
        threshold measured in fp32."""
        with torch.autocast(captions.device.type, enabled=False):
            unit = functional.normalize(captions.float(), dim=-1)
            similarity = unit @ unit.T
        thresholds = self.thresholds.to(captions.device)[rows.to(captions.device)]
        flagged = (similarity >= thresholds[:, None]) | (similarity >= thresholds[None, :])
        different = rows.to(captions.device)[:, None] != rows.to(captions.device)[None, :]
        return flagged & different


class SIGRegLoss:
    """SIGReg with fresh random directions at every call (LeJEPA) [Lett. 35].

    The views share the directions of the call; the statistic is averaged over them. Across
    GPUs each view's characteristic function is that of the samples of every GPU, and the
    ``generator`` must draw the same directions on every rank.
    """

    def __init__(self, directions: int, knots: int, collective: Distributed = SINGLE) -> None:
        self.directions = directions
        self.statistic = SIGReg(knots)
        self.collective = collective

    def __call__(self, views: Sequence[Tensor], generator: torch.Generator) -> Tensor:
        rows = [view.reshape(-1, view.shape[-1]).float() for view in views]
        slices = random_directions(rows[0].shape[1], self.directions, generator=generator)
        slices = slices.to(rows[0].device)
        statistics = [self.statistic(view, slices, reduce=self._reduce) for view in rows]
        return torch.stack(statistics).mean()

    def per_step(self, views: Tensor, present: Tensor, generator: torch.Generator) -> Tensor:
        """SIGReg of each view at each step over the clips present there, averaged (LeWM).

        ``views`` (V, batch, steps, C); ``present`` (batch, steps) bool. The steps of one clip
        are not independent: together they would read their correlation as non-Gaussianity.
        A view at a step counts once at least two clips are present there over all the GPUs;
        with none it is 0. One set of directions serves every view and step.
        """
        count, batch, steps, width = views.shape
        rows = views.float().transpose(1, 2).reshape(count * steps, batch, width)
        members = present.T.expand(count, steps, batch).reshape(count * steps, batch)
        slices = random_directions(width, self.directions, generator=generator)
        statistics, samples = self.statistic.per_group(
            rows, members, slices.to(views.device), reduce=self._reduce
        )
        counted = (samples >= 2).float()  # noqa: PLR2004 (a distribution needs two samples)
        return (statistics * counted).sum() / counted.sum().clamp_min(1.0)

    @property
    def _reduce(self) -> Callable[[Tensor], Tensor] | None:
        return self.collective.all_sum if self.collective.active else None


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Every term of one step, kept apart so each can be logged and its gradient share read."""

    total: Tensor
    parts: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor] = field(default_factory=dict)
    """Values measured without entering the total."""
    samples: dict[str, Tensor] = field(default_factory=dict)
    """(batch,) energies of every clip, without gradient: ``e_sem`` and ``e_fis``."""


class Objective(nn.Module):
    """The objective of one arm; the pose terms arrive from the pose branch."""

    def __init__(
        self,
        losses: LossSettings,
        semantic: SemanticSettings,
        physical: bool,
        collective: Distributed = SINGLE,
        false_negatives: FalseNegatives | None = None,
    ) -> None:
        super().__init__()
        self.settings = losses
        self.relaxation = semantic.relaxation if semantic.hypotheses > 1 else 0.0
        self.physical = physical
        self.collective = collective
        self.sigreg = SIGRegLoss(losses.sigreg_directions, losses.sigreg_knots, collective)
        self.infonce = (
            InfoNCE(
                losses.infonce_temperature, losses.infonce_min_temperature, losses.multi_positive
            )
            if losses.arm == "C"
            else None
        )
        self.false_negatives = false_negatives if losses.arm == "C" else None
        self.last_diagnostics: dict[str, Tensor] = {}
        """Values of the last semantic pass that do not enter the total (arm C's pairs)."""

    @property
    def uses_sigreg(self) -> bool:
        return self.settings.arm in ("A", "B", "C")

    @property
    def uses_uniformity(self) -> bool:
        return self.settings.arm in ("B0", "B")

    @property
    def uses_vicreg(self) -> bool:
        return self.settings.arm == "V"

    def semantic_terms(
        self,
        predicted: Tensor,
        text: Tensor,
        videos: Sequence[str],
        generator: torch.Generator,
        rows: Tensor | None = None,
        captions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``predicted`` (batch, K, d) and ``text`` (batch, d), this GPU's share of the batch.

        InfoNCE, L_unif and VICReg read the whole batch, gathered from every GPU. Arm C also
        reads each clip's caption ``rows`` (same caption, positive) and, for its false
        negatives, the frozen ``captions`` centred by language (batch, 768).
        """
        terms: dict[str, Tensor] = {}
        self.last_diagnostics = {}
        gather = self.collective.all_gather
        if self.infonce is not None:
            if predicted.shape[1] != 1:
                raise ValueError("arm C with several hypotheses is not defined here")
            everyone = self.collective.gather_objects(videos)
            every_row = None if rows is None else gather(rows.to(text.device))
            excluded = None
            if self.false_negatives is not None:
                if rows is None or captions is None or every_row is None:
                    raise ValueError("the false negatives need the caption rows and vectors")
                excluded = self.false_negatives.mask(gather(captions.detach()), every_row)
            terms["infonce"] = self.infonce(
                gather(predicted[:, 0]), gather(text), everyone, every_row, excluded
            )
            self.last_diagnostics = self._pair_diagnostics(everyone, every_row, excluded)
        elif predicted.shape[1] > 1:
            terms["e_sem"] = free_energy(predicted, text, self.relaxation).mean()
        else:
            terms["e_sem"] = semantic_energy(predicted[:, 0], text).mean()
        if self.uses_uniformity:
            terms["unif"] = 0.5 * (
                uniformity(gather(predicted.flatten(0, 1)), self.settings.uniformity_t)
                + uniformity(gather(text), self.settings.uniformity_t)
            )
        if self.uses_vicreg:
            modalities = (gather(predicted.flatten(0, 1)), gather(text))
            s = self.settings
            terms["vicreg_var"] = sum(
                (vicreg_variance(z, s.vicreg_gamma, s.vicreg_epsilon) for z in modalities),
                torch.zeros((), device=text.device),
            )
            terms["vicreg_cov"] = sum(
                (vicreg_covariance(z) for z in modalities), torch.zeros((), device=text.device)
            )
        if self.uses_sigreg:
            terms["sigreg_sem"] = self.sigreg([predicted.flatten(0, 1), text], generator)
        return terms

    @torch.no_grad()
    def _pair_diagnostics(
        self, videos: Sequence[str], rows: Tensor | None, excluded: Tensor | None
    ) -> dict[str, Tensor]:
        """Arm C: positives and dropped negatives per clip, and the temperature."""
        assert self.infonce is not None
        device = self.infonce.log_temperature.device
        positive, dropped = self.infonce.pairs(videos, rows, excluded, device)
        out = {
            "temperature": self.infonce.temperature.detach(),
            "positives_per_clip": positive.float().sum(dim=1).mean() - 1.0,
            "dropped_per_clip": dropped.float().sum(dim=1).mean(),
        }
        if excluded is not None:
            out["false_negatives_per_clip"] = (
                (excluded.to(device) & ~positive).float().sum(1).mean()
            )
        return out

    def pose_sigreg(self, views: Tensor, present: Tensor, generator: torch.Generator) -> Tensor:
        """``SIGReg_posa``: the mean over the views and the steps of SIGReg of
        ``{z_{v,b,t} : c_{b,t} > 0}_b`` (posa §4.2).

        ``views`` (V, batch, steps, C), the clean sequence first; ``present`` (batch, steps)
        bool. With no step holding two present clips over all the GPUs, it is 0.
        """
        return self.sigreg.per_step(views, present, generator)

    def weights(self, names: Iterable[str]) -> dict[str, float]:
        """Each term's weight in the total: ``1 - λ`` if predictive, ``λ`` if SIGReg, with the
        pose level's own λ on its terms; L_unif and VICReg's terms carry, on top, their
        weight against E_sem from the literature (``LossSettings``)."""
        s = self.settings
        invariance, variance, covariance = s.vicreg_coefficients
        relative = {
            "unif": s.uniformity_weight,
            "vicreg_var": variance / (2 * invariance),
            "vicreg_cov": covariance / (2 * invariance),
        }
        out = {}
        for name in names:
            pose = name in POSE_TERMS
            weight = s.pose_sigreg_weight if pose else s.sigreg_weight
            level = weight if name.startswith("sigreg") else 1.0 - weight
            out[name] = level * relative.get(name, 1.0)
        return out

    def combine(self, terms: dict[str, Tensor]) -> LossTerms:
        """``(1 - λ)·(predictive) + λ·(SIGReg)`` level by level, over whichever terms are
        present (``weights``)."""
        weights = self.weights(terms)
        total = torch.stack([weights[k] * v.float() for k, v in terms.items()]).sum()
        return LossTerms(total, terms)
