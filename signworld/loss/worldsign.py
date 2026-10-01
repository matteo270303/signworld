"""Energies and regularisers of the objective (§4.5.3-§4.5.9), and their per-arm composition.

``L = (1 - λ) · (E_fis + L_anchor + L_pred_sem) + λ · (SIGReg_posa + SIGReg_sem)``, λ = 0.05,
with ``L_pred_sem`` and ``SIGReg_sem`` set by the arm of ESP-1:

    A₀   E_sem                —
    A    E_sem                ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    B₀   E_sem + L_unif       —
    B    E_sem + L_unif       ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    C    InfoNCE              ½ [SIGReg({ŷ}) + SIGReg({ẽ})]

SIGReg is applied to each modality apart, as LeJEPA applies it to each view, with the same
random directions for both; on the pose it is applied to each articulator apart
(``SIGReg_posa``, §4.5.6). ``L_anchor`` comes from the pose branch. With the physical level off
(ESP-2) every physical term is absent.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.experiment.train.config import LossSettings, SemanticSettings
from signworld.experiment.train.distributed import SINGLE, Distributed
from signworld.loss.sigreg import SIGReg, random_directions
from signworld.models.worldsign.physical import PhysicalPrediction


def physical_energy(
    predictions: Sequence[PhysicalPrediction],
    target: Tensor,
    confidence: Tensor,
    context_lambda: float,
) -> Tensor:
    """E_fis: V-JEPA 2.1's ``L_pred + λ · L_ctx`` read per articulator box, averaged over masks.

    As in V-JEPA 2.1, the target is layer-normalised over its channels (no affine parameters)
    and the error is the L1 averaged over the channels (``loss_exp = 1``). Each box enters with
    its confidence ``c̄`` and with the tokens behind its read-out:

    * ``L_pred``: the box read from its masked tokens, weighted by how many they are, so that
      every masked token counts once, as in V-JEPA's mean over the masked tokens;
    * ``L_ctx``: the box read from its visible tokens, weighted by the sum of their weights and
      divided by their number: V-JEPA's mean over the visible tokens of the weighted error.
      The weight is 1 in V-JEPA 2.1's cooldown, which we follow, and ``1 / √d`` in its
      pre-training.

    ``target`` (batch, steps, parts, C); ``confidence`` (batch, steps, parts).
    """
    normalized = functional.layer_norm(target.float(), (target.shape[-1],))
    energies = []
    for prediction in predictions:
        masked_error = (prediction.masked.float() - normalized).abs().mean(dim=-1)
        visible_error = (prediction.visible.float() - normalized).abs().mean(dim=-1)
        masked = confidence * prediction.masked_count
        l_pred = (masked * masked_error).sum() / masked.sum().clamp_min(1e-12)
        visible = confidence * prediction.visible_count
        weighted = confidence * prediction.visible_weight
        l_ctx = (weighted * visible_error).sum() / visible.sum().clamp_min(1e-12)
        energies.append(l_pred + context_lambda * l_ctx)
    return torch.stack(energies).mean()


def physical_energy_per_clip(
    predictions: Sequence[PhysicalPrediction],
    target: Tensor,
    confidence: Tensor,
    context_lambda: float,
) -> Tensor:
    """(batch,) E_fis of every clip on its own, averaged over the masks: plausibility (§4.5.3)."""
    normalized = functional.layer_norm(target.float(), (target.shape[-1],))
    energies = []
    for prediction in predictions:
        masked_error = (prediction.masked.float() - normalized).abs().mean(dim=-1)
        visible_error = (prediction.visible.float() - normalized).abs().mean(dim=-1)
        masked = confidence * prediction.masked_count
        l_pred = (masked * masked_error).sum((1, 2)) / masked.sum((1, 2)).clamp_min(1e-12)
        visible = confidence * prediction.visible_count
        weighted = confidence * prediction.visible_weight
        l_ctx = (weighted * visible_error).sum((1, 2)) / visible.sum((1, 2)).clamp_min(1e-12)
        energies.append(l_pred + context_lambda * l_ctx)
    return torch.stack(energies).mean(dim=0)


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


class InfoNCE(nn.Module):
    """Symmetric InfoNCE with a learnable temperature; same-video pairs are not negatives."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    def logits(self, video: Tensor, text: Tensor, videos: Sequence[str]) -> Tensor:
        """(batch, batch) scaled cosines; pairs of different clips of one video at -inf (P15)."""
        logits = functional.normalize(video, dim=-1) @ functional.normalize(text, dim=-1).T
        logits = logits / self.log_temperature.exp()
        same = torch.tensor([[a == b for b in videos] for a in videos], device=logits.device)
        excluded = same & ~torch.eye(len(videos), dtype=torch.bool, device=logits.device)
        masked: Tensor = logits.masked_fill(excluded, float("-inf"))
        return masked

    def forward(self, video: Tensor, text: Tensor, videos: Sequence[str]) -> Tensor:
        logits = self.logits(video, text, videos)
        labels = torch.arange(len(videos), device=logits.device)
        return 0.5 * (
            functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)
        )


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
        reduce = self.collective.all_sum if self.collective.active else None
        return torch.stack([self.statistic(view, slices, reduce=reduce) for view in rows]).mean()


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Every term of one step, kept apart so each can be logged and its gradient share read."""

    total: Tensor
    parts: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor] = field(default_factory=dict)
    """Values measured without entering the total (SIGReg on a frozen pose, ESP-3)."""
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
    ) -> None:
        super().__init__()
        self.settings = losses
        self.relaxation = semantic.relaxation if semantic.hypotheses > 1 else 0.0
        self.physical = physical
        self.collective = collective
        self.sigreg = SIGRegLoss(losses.sigreg_directions, losses.sigreg_knots, collective)
        self.infonce = InfoNCE(losses.infonce_temperature) if losses.arm == "C" else None

    @property
    def uses_sigreg(self) -> bool:
        return self.settings.arm in ("A", "B", "C")

    @property
    def uses_uniformity(self) -> bool:
        return self.settings.arm in ("B0", "B")

    def semantic_terms(
        self, predicted: Tensor, text: Tensor, videos: Sequence[str], generator: torch.Generator
    ) -> dict[str, Tensor]:
        """``predicted`` (batch, K, d) and ``text`` (batch, d), this GPU's share of the batch.

        InfoNCE and L_unif compare all the pairs of the whole batch, gathered from every GPU.
        """
        terms: dict[str, Tensor] = {}
        gather = self.collective.all_gather
        if self.infonce is not None:
            if predicted.shape[1] != 1:
                raise ValueError("arm C with several hypotheses is not defined here")
            everyone = self.collective.gather_objects(videos)
            terms["infonce"] = self.infonce(gather(predicted[:, 0]), gather(text), everyone)
        elif predicted.shape[1] > 1:
            terms["e_sem"] = free_energy(predicted, text, self.relaxation).mean()
        else:
            terms["e_sem"] = semantic_energy(predicted[:, 0], text).mean()
        if self.uses_uniformity:
            terms["unif"] = 0.5 * (
                uniformity(gather(predicted.flatten(0, 1)), self.settings.uniformity_t)
                + uniformity(gather(text), self.settings.uniformity_t)
            )
        if self.uses_sigreg:
            terms["sigreg_sem"] = self.sigreg([predicted.flatten(0, 1), text], generator)
        return terms

    def pose_sigreg(self, latent: Tensor, present: Tensor, generator: torch.Generator) -> Tensor:
        """``SIGReg_posa = ¼ Σ_a SIGReg({s_{t,a}})`` over the steps where ``a`` is present.

        ``latent`` (batch, steps, 4, C); ``present`` (batch, steps, 4) bool. An articulator with
        fewer than two present steps in the whole batch is left out, on every GPU alike.
        """
        counts = self.collective.all_sum(present.sum(dim=(0, 1)).float())
        parts = [part for part in range(latent.shape[2]) if counts[part] > 1]
        if not parts:
            return latent.new_zeros(())
        return self.sigreg([latent[:, :, part][present[:, :, part]] for part in parts], generator)

    def combine(self, terms: dict[str, Tensor]) -> LossTerms:
        """``(1 - λ)·(predictive) + λ·(SIGReg)`` over whichever terms are present."""
        weight = self.settings.sigreg_weight
        predictive = [v for k, v in terms.items() if not k.startswith("sigreg")]
        regular = [v for k, v in terms.items() if k.startswith("sigreg")]
        total = (1.0 - weight) * torch.stack(predictive).sum()
        if regular:
            total = total + weight * torch.stack(regular).sum()
        return LossTerms(total, terms)
