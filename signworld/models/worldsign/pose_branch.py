"""The pose side of WorldSign: S-JEPA as the target of the physical level, and the anchor.

``PoseEncoder`` is the EMA encoder of the S-JEPA teacher pre-trained in PC5 (§4.4.3): token
embedding, 8 blocks at d = 256 and the final LayerNorm, pooled per articulator into
``s_{t,a}`` (C = 256, one per tubelet step). In the reference configuration it stays frozen
with LoRA r = 4 on every block (q, k and v apart, the attention output, both MLP layers) and
ends in a trainable linear map per articulator, initialised to the identity: at step 0 the
target is exactly S-JEPA's. Everything trainable in it runs at the learning rate x0.05. In
ESP-3 it is frozen and has neither.

``KeypointDecoders`` are the heads ``D_pose^a``, one linear map per articulator from
``s_{t,a}`` to the (x, y) of its joints; ``anchor_loss`` is ``L_anchor`` (§4.5.3), divided by
the keypoint variance so that predicting each joint's mean is worth 1 (§4.5.7).
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from signworld.data.pose.tokens import JOINT_ARTICULATOR
from signworld.data.pose.wholebody import ARTICULATOR_INDICES, Articulator
from signworld.experiment.train.config import PoseEncoderSettings
from signworld.models.encoders.pose_teachers import TokenEmbedding, pool_articulators, transformer

from . import lora

PARTS = len(Articulator)


class ArticulatorLinear(nn.Module):
    """One ``C -> C`` linear map per articulator, initialised to the identity."""

    def __init__(self, width: int, parts: int = PARTS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width).repeat(parts, 1, 1))
        self.bias = nn.Parameter(torch.zeros(parts, width))

    def forward(self, x: Tensor) -> Tensor:
        """(..., parts, C) to (..., parts, C)."""
        mapped: Tensor = torch.einsum("...ac,adc->...ad", x, self.weight) + self.bias
        return mapped


def adapt_blocks(encoder: nn.TransformerEncoder, rank: int, alpha: float) -> list[lora.LoRAWeight]:
    """LoRA on every block of a PyTorch transformer: q, k, v, attention output, MLP."""
    adapters: list[lora.LoRAWeight] = []
    for block in encoder.layers:
        attention: nn.MultiheadAttention = block.self_attn
        adapters += [
            lora.adapt_weight(attention, "in_proj_weight", rank, alpha, splits=3),
            lora.adapt_weight(attention.out_proj, "weight", rank, alpha),
            lora.adapt_weight(block.linear1, "weight", rank, alpha),
            lora.adapt_weight(block.linear2, "weight", rank, alpha),
        ]
    return adapters


def _load(module: nn.Module, state: dict[str, Tensor], prefix: str) -> None:
    part = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(part, strict=True)


class PoseEncoder(nn.Module):
    """S-JEPA's EMA encoder, pooled per articulator; adapted unless frozen (ESP-3)."""

    def __init__(
        self, embed: TokenEmbedding, encoder: nn.TransformerEncoder, settings: PoseEncoderSettings
    ) -> None:
        super().__init__()
        self.embed = embed.requires_grad_(False)
        self.encoder = encoder.requires_grad_(False)
        self.settings = settings
        self.adapters: list[lora.LoRAWeight] = []
        self.final: ArticulatorLinear | None = None
        if settings.trainable:
            self.adapters = adapt_blocks(encoder, settings.lora_rank, settings.lora_alpha)
            if settings.final_layer:
                self.final = ArticulatorLinear(settings.width)

    @classmethod
    def from_checkpoint(cls, settings: PoseEncoderSettings) -> "PoseEncoder":
        """The EMA encoder (``target_embed``, ``target_encoder``) of a saved ``SJEPATeacher``."""
        if settings.checkpoint is None:
            raise ValueError("pose_encoder.checkpoint is not set")
        state = torch.load(Path(settings.checkpoint), map_location="cpu", weights_only=True)
        embed = TokenEmbedding(settings.width)
        encoder = transformer(settings.width, settings.depth, settings.heads)
        _load(embed, state, "target_embed.")
        _load(encoder, state, "target_encoder.")
        return cls(embed, encoder, settings)

    @property
    def trainable(self) -> bool:
        return self.settings.trainable

    def forward(self, tokens: Tensor) -> Tensor:
        """(batch, 32, 69, 6) pose tokens to ``s``: (batch, 32, 4, C)."""
        latent: Tensor = pool_articulators(self.encoder(self.embed(tokens)))
        return latent if self.final is None else self.final(latent)


class KeypointDecoders(nn.Module):
    """``D_pose^a``: one linear map per articulator from ``s_{t,a}`` to its joints' (x, y)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Linear(width, 2 * len(ARTICULATOR_INDICES[part])) for part in Articulator
        )

    def forward(self, latent: Tensor) -> Tensor:
        """(batch, steps, 4, C) to (batch, steps, 69, 2), joints in ``JOINTS`` order."""
        joints = [
            head(latent[..., part, :]).unflatten(-1, (-1, 2))
            for part, head in enumerate(self.heads)
        ]
        return torch.cat(joints, dim=-2)


def articulator_confidence(weights: Tensor) -> Tensor:
    """(batch, steps, 4) ``c̄_{t,a}``: mean joint weight of each articulator at each step."""
    groups = torch.as_tensor(JOINT_ARTICULATOR, device=weights.device)
    return torch.stack([weights[..., groups == part].mean(dim=-1) for part in range(PARTS)], -1)


def keypoint_variance(keypoints: Tensor, weights: Tensor) -> Tensor:
    """Weighted mean squared distance of each joint from its own mean: L_anchor of the mean.

    ``keypoints`` (clips, steps, 69, 2) and ``weights`` (clips, steps, 69), training clips only.
    """
    flat_points = keypoints.double().flatten(0, 1)
    flat_weights = weights.double().flatten(0, 1)[..., None]
    total = flat_weights.sum(dim=0).clamp_min(1e-12)
    mean = (flat_points * flat_weights).sum(dim=0) / total
    squared = (flat_points - mean).pow(2).sum(dim=-1, keepdim=True)
    return ((squared * flat_weights).sum() / flat_weights.sum()).float()


def anchor_loss(predicted: Tensor, keypoints: Tensor, weights: Tensor, variance: Tensor) -> Tensor:
    """``Σ c_{t,j}·‖D(s)_j - p̂_{t,j}‖² / Σ c_{t,j}``, over the keypoint variance."""
    squared = (predicted.float() - keypoints).pow(2).sum(dim=-1)
    return (weights * squared).sum() / weights.sum().clamp_min(1e-12) / variance


class PoseBranch(nn.Module):
    """The pose encoder with the anchor's decoders and keypoint scale."""

    keypoint_variance: Tensor

    def __init__(self, encoder: PoseEncoder, decoders: KeypointDecoders) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.register_buffer("keypoint_variance", torch.tensor(float("nan")))

    @classmethod
    def from_settings(cls, settings: PoseEncoderSettings) -> "PoseBranch":
        return cls(PoseEncoder.from_checkpoint(settings), KeypointDecoders(settings.width))

    @property
    def trainable(self) -> bool:
        return self.encoder.trainable

    def fit(self, keypoints: Tensor, weights: Tensor) -> None:
        """Set the scale of L_anchor from the training clips (§4.6: nothing from validation)."""
        self.keypoint_variance.copy_(keypoint_variance(keypoints, weights))

    def targets(self, tokens: Tensor) -> Tensor:
        """(batch, 32, 4, C) ``s_{t,a}``, the target of the physical level."""
        latent: Tensor = self.encoder(tokens)
        return latent

    def anchor(self, latent: Tensor, keypoints: Tensor, weights: Tensor) -> Tensor:
        if bool(torch.isnan(self.keypoint_variance)):
            raise RuntimeError("fit the keypoint variance on the training clips first")
        return anchor_loss(self.decoders(latent), keypoints, weights, self.keypoint_variance)
