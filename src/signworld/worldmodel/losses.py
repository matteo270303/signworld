"""Energies and regularisers of the objective (§4.5.3-§4.5.9), and their per-arm composition.

``L = (1 - λ) · (E_fis + L_anchor + L_pred_sem) + λ · (SIGReg_posa + SIGReg_sem)``, λ = 0.05,
with ``L_pred_sem`` and ``SIGReg_sem`` set by the arm of ESP-1:

    A₀   E_sem                —
    A    E_sem                ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    B₀   E_sem + L_unif       —
    B    E_sem + L_unif       ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
    C    InfoNCE              ½ [SIGReg({ŷ}) + SIGReg({ẽ})]

SIGReg is applied to each modality apart, as LeJEPA applies it to each view, with the same
random directions for both. The pose terms (``L_anchor``, ``SIGReg_posa``) belong to the pose
branch and enter as given tensors; with the physical level off (ESP-2) every physical term is
absent.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..metrics.sigreg import SIGReg, random_directions
from .config import LossSettings, SemanticSettings
from .physical import PhysicalPrediction


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
    * ``L_ctx``: the box read from its visible tokens, weighted by the sum of their ``1 / √d``
      and divided by their number: V-JEPA's mean over the visible tokens of the error times
      ``1 / √d``.

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

    def forward(self, video: Tensor, text: Tensor, videos: Sequence[str]) -> Tensor:
        logits = functional.normalize(video, dim=-1) @ functional.normalize(text, dim=-1).T
        logits = logits / self.log_temperature.exp()
        same = torch.tensor([[a == b for b in videos] for a in videos], device=logits.device)
        excluded = same & ~torch.eye(len(videos), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(excluded, float("-inf"))
        labels = torch.arange(len(videos), device=logits.device)
        return 0.5 * (
            functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)
        )


class SIGRegLoss:
    """SIGReg with fresh random directions at every call (LeJEPA) [Lett. 35].

    The views share the directions of the call; the statistic is averaged over them.
    """

    def __init__(self, directions: int, knots: int) -> None:
        self.directions = directions
        self.statistic = SIGReg(knots)

    def __call__(self, views: Sequence[Tensor], generator: torch.Generator) -> Tensor:
        rows = [view.reshape(-1, view.shape[-1]).float() for view in views]
        slices = random_directions(rows[0].shape[1], self.directions, generator=generator)
        slices = slices.to(rows[0].device)
        return torch.stack([self.statistic(view, slices) for view in rows]).mean()


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Every term of one step, kept apart so each can be logged and its gradient share read."""

    total: Tensor
    parts: dict[str, Tensor] = field(default_factory=dict)


class Objective(nn.Module):
    """The objective of one arm; the pose terms arrive from the pose branch."""

    def __init__(self, losses: LossSettings, semantic: SemanticSettings, physical: bool) -> None:
        super().__init__()
        self.settings = losses
        self.relaxation = semantic.relaxation if semantic.hypotheses > 1 else 0.0
        self.physical = physical
        self.sigreg = SIGRegLoss(losses.sigreg_directions, losses.sigreg_knots)
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
        """``predicted`` (batch, K, d) and ``text`` (batch, d)."""
        terms: dict[str, Tensor] = {}
        if self.infonce is not None:
            if predicted.shape[1] != 1:
                raise ValueError("arm C with several hypotheses is not defined here")
            terms["infonce"] = self.infonce(predicted[:, 0], text, videos)
        elif predicted.shape[1] > 1:
            terms["e_sem"] = free_energy(predicted, text, self.relaxation).mean()
        else:
            terms["e_sem"] = semantic_energy(predicted[:, 0], text).mean()
        if self.uses_uniformity:
            terms["unif"] = 0.5 * (
                uniformity(predicted.flatten(0, 1), self.settings.uniformity_t)
                + uniformity(text, self.settings.uniformity_t)
            )
        if self.uses_sigreg:
            terms["sigreg_sem"] = self.sigreg([predicted.flatten(0, 1), text], generator)
        return terms

    def combine(self, terms: dict[str, Tensor]) -> LossTerms:
        """``(1 - λ)·(predictive) + λ·(SIGReg)`` over whichever terms are present."""
        weight = self.settings.sigreg_weight
        predictive = [v for k, v in terms.items() if not k.startswith("sigreg")]
        regular = [v for k, v in terms.items() if k.startswith("sigreg")]
        total = (1.0 - weight) * torch.stack(predictive).sum()
        if regular:
            total = total + weight * torch.stack(regular).sum()
        return LossTerms(total, terms)
