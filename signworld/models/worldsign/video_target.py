"""ESP-6: the physical level predicts the released V-JEPA 2.1 encoder instead of the pose.

Everything of the reference configuration stays (encoder with LoRA, multi-level fusion, the
released predictor with its LoRA, the masks, the stages, the semantic level) except the target:

* ``VideoTarget`` is a frozen copy of the released encoder, every weight as released (its
  norms too: the ones the adapted encoder trains are not shared), made before the LoRA is
  injected. It encodes the whole clip; its last layer, layer-normalised without parameters, is
  the target, as in V-JEPA 2.1's distillation, which replaces the EMA teacher with a frozen one
  and supervises only the teacher's last layer (arXiv 2603.14482, App. B).
* ``TokenHeads`` replace the step read-out: one linear map of the predicted tokens and one of
  the context tokens to the teacher's width, as V-JEPA 2.1's ``predictor_proj`` and
  ``predictor_proj_context`` (the released ones map to the ViT-G teacher's 1,664 channels).
* ``token_energy`` is V-JEPA 2.1's loss: L1 averaged over channels and tokens, on the masked
  tokens plus ``λ`` times on the visible ones (each weighted 1 in the cooldown, ``1 / √d`` in
  pre-training), averaged over the masks of the step.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.models.encoders.video_encoders import model_input


class VideoTarget(nn.Module):
    """The frozen teacher: the released encoder, copied before any adaptation."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        # Meta's module is untyped and carries plain attributes (embed_dim, ...).
        teacher: Any = copy.deepcopy(encoder).requires_grad_(False)
        teacher.use_activation_checkpointing = False  # no backward through it
        self.encoder: Any = teacher.eval()

    def train(self, mode: bool = True) -> "VideoTarget":
        """Always in evaluation mode: the teacher never draws dropout."""
        super().train(False)
        return self

    @property
    def width(self) -> int:
        return int(self.encoder.embed_dim)

    @torch.no_grad()
    def forward(self, frames: Tensor) -> Tensor:
        """(batch, N, width): the last layer on the whole clip, layer-normalised (no affine)."""
        tokens: Tensor = self.encoder(model_input(frames))
        return functional.layer_norm(tokens.float(), (tokens.shape[-1],))


class TokenHeads(nn.Module):
    """Predicted and context tokens to the teacher's width, apart (V-JEPA 2.1's projections)."""

    def __init__(self, width: int, target_width: int) -> None:
        super().__init__()
        self.predicted = nn.Linear(width, target_width)
        self.context = nn.Linear(width, target_width)


@dataclass(frozen=True, slots=True)
class TokenPrediction:
    """What one mask gives the token energy."""

    predicted: Tensor
    """(batch, n_target, width) the masked tokens, predicted."""
    context: Tensor
    """(batch, n_context, width) the visible tokens, predicted."""
    target_index: Tensor
    """(batch, n_target) their grid positions."""
    context_index: Tensor
    context_weight: Tensor
    """(batch, n_context) each visible token's weight: 1, or ``1 / √d``."""


def _errors(prediction: TokenPrediction, target: Tensor) -> tuple[Tensor, Tensor]:
    """(batch, n_target) and (batch, n_context) L1 errors averaged over the channels."""

    def at(index: Tensor) -> Tensor:
        return target.gather(1, index[..., None].expand(-1, -1, target.shape[-1]))

    masked = (prediction.predicted.float() - at(prediction.target_index)).abs().mean(-1)
    visible = (prediction.context.float() - at(prediction.context_index)).abs().mean(-1)
    return masked, visible * prediction.context_weight


def token_energy(
    predictions: Sequence[TokenPrediction], target: Tensor, context_lambda: float
) -> Tensor:
    """V-JEPA 2.1's ``L_pred + λ·L_ctx`` against the teacher's ``target`` (batch, N, width),
    each a mean over clips, tokens and channels, averaged over the masks."""
    energies = []
    for prediction in predictions:
        masked, visible = _errors(prediction, target)
        energies.append(masked.mean() + context_lambda * visible.mean())
    return torch.stack(energies).mean()


def token_energy_per_clip(
    predictions: Sequence[TokenPrediction], target: Tensor, context_lambda: float
) -> Tensor:
    """(batch,) the same energy for every clip on its own: plausibility (§4.5.3)."""
    energies = []
    for prediction in predictions:
        masked, visible = _errors(prediction, target)
        energies.append(masked.mean(1) + context_lambda * visible.mean(1))
    return torch.stack(energies).mean(dim=0)
