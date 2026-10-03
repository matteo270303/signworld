"""S-JEPA's networks and loss (paper §3, Fig. 2).

* ``SegmentEmbedding``: the ``l`` frames of a joint in a segment, embedded together by a linear
  map, plus learnable spatial and temporal positions (``N = T/l · V`` tokens).
* The **view encoder** ``f_θ`` sees only the visible tokens of a view; the **target encoder**
  ``f_θ̄`` is its EMA, sees the whole original sequence, and gets no gradient.
* The **predictor** ``g_φ``: mask tokens fill the hidden places of the view encoder's output, a
  linear map to the predictor's width, positions of its own (as MAMP's decoder), ``L_p`` blocks,
  and a map back to the encoder's width; its outputs at the hidden places are ``R_p``.
* **Loss**: ``R_t``, the target encoder's output at the hidden places (masked at the output, not
  the input), is centred with the batch centre ``c ← β c + (1 - β) mean(R_t)`` and sharpened;
  the cross-entropy between ``softmax((R_t - c) / τ_t)`` and ``softmax(R_p / τ_p)``.

Downstream only the target encoder is used (``represent``).
"""

import copy
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional

from .config import SJEPASettings
from .masking import motion_aware_mask

_POSITION_STD = 0.02


def blocks(width: int, depth: int, heads: int, mlp_ratio: float) -> nn.Module:
    """The vanilla transformer: pre-norm blocks with GELU and a final LayerNorm."""
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=round(mlp_ratio * width),
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def gather(x: Tensor, indices: Tensor) -> Tensor:
    """Rows ``indices`` (batch, k) of ``x`` (batch, n, d)."""
    return torch.gather(x, 1, indices[..., None].expand(-1, -1, x.shape[-1]))


class Positions(nn.Module):
    """Separate learnable spatial (V) and temporal (T/l) positions, summed per token."""

    def __init__(self, segments: int, joints: int, width: int) -> None:
        super().__init__()
        self.spatial = nn.Parameter(torch.empty(joints, width).normal_(std=_POSITION_STD))
        self.temporal = nn.Parameter(torch.empty(segments, width).normal_(std=_POSITION_STD))

    def forward(self) -> Tensor:
        """(N, width), segment-major."""
        return (self.temporal[:, None] + self.spatial[None]).flatten(0, 1)


class SegmentEmbedding(nn.Module):
    """(batch, T, V, C) skeletons to (batch, N, width) tokens with their positions."""

    def __init__(self, settings: SJEPASettings) -> None:
        super().__init__()
        self.settings = settings
        self.project = nn.Linear(settings.segment * settings.channels, settings.width)
        self.positions = Positions(settings.segments, settings.joints, settings.width)

    def forward(self, sequence: Tensor) -> Tensor:
        s = self.settings
        batch = len(sequence)
        segments = sequence.reshape(batch, s.segments, s.segment, s.joints, s.channels)
        grouped = segments.transpose(2, 3).reshape(batch, s.tokens, s.segment * s.channels)
        embedded: Tensor = self.project(grouped.float()) + self.positions()
        return embedded


class Predictor(nn.Module):
    """``g_φ``: from the view features with mask tokens to ``R_p`` at the hidden places."""

    def __init__(self, settings: SJEPASettings) -> None:
        super().__init__()
        width = settings.width
        self.tokens = settings.tokens
        self.mask_token = nn.Parameter(torch.empty(width).normal_(std=_POSITION_STD))
        self.project = nn.Linear(width, width)
        self.positions = Positions(settings.segments, settings.joints, width)
        self.blocks = blocks(width, settings.predictor_depth, settings.heads, settings.mlp_ratio)
        self.output = nn.Linear(width, width)

    def forward(self, encoded: Tensor, visible: Tensor, hidden: Tensor) -> Tensor:
        """``encoded`` (batch, n_visible, width) to (batch, n_hidden, width)."""
        batch, _, width = encoded.shape
        features = self.mask_token.to(encoded.dtype).expand(batch, self.tokens, width).clone()
        features = features.scatter(1, visible[..., None].expand(-1, -1, width), encoded)
        predicted: Tensor = self.output(self.blocks(self.project(features) + self.positions()))
        return gather(predicted, hidden)


class SJEPA(nn.Module):
    """View encoder, EMA target encoder and predictor, with the centred cross-entropy."""

    centre: Tensor

    def __init__(self, settings: SJEPASettings) -> None:
        super().__init__()
        self.settings = settings
        self.embed = SegmentEmbedding(settings)
        self.encoder = blocks(settings.width, settings.depth, settings.heads, settings.mlp_ratio)
        self.predictor = Predictor(settings)
        self.target_embed = copy.deepcopy(self.embed).requires_grad_(False)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.register_buffer("centre", torch.zeros(settings.width))

    def loss(
        self,
        sequence: Tensor,
        view: Tensor,
        generator: torch.Generator,
        mean: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """The cross-entropy of one batch: ``sequence`` the original, ``view`` its view.

        ``mean`` averages a tensor over the GPUs of a data-parallel run, so that the centre is
        that of the whole batch.
        """
        s = self.settings
        visible, hidden = motion_aware_mask(
            sequence, s.segment, s.mask_ratio, s.motion_temperature, generator
        )
        encoded = self.encoder(gather(self.embed(view), visible))
        predicted = self.predictor(encoded, visible, hidden)
        with torch.no_grad():
            features = self.target_encoder(self.target_embed(sequence))
            target = gather(features, hidden).float()
            self._update_centre(target, mean)
            probabilities = functional.softmax((target - self.centre) / s.teacher_temperature, -1)
        log_prediction = functional.log_softmax(predicted.float() / s.student_temperature, -1)
        return -(probabilities * log_prediction).sum(dim=-1).mean()

    def _update_centre(self, target: Tensor, mean: Callable[[Tensor], Tensor] | None) -> None:
        batch_centre = target.mean(dim=(0, 1))
        if mean is not None:
            batch_centre = mean(batch_centre)
        rate = self.settings.centre_rate
        self.centre.mul_(rate).add_(batch_centre, alpha=1 - rate)

    def student(self) -> Iterator[nn.Parameter]:
        """The parameters the optimiser updates: view encoder (with its embedding), predictor."""
        for module in (self.embed, self.encoder, self.predictor):
            yield from module.parameters()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """``θ̄ ← λ θ̄ + (1 - λ) θ`` after each optimiser step."""
        pairs = ((self.embed, self.target_embed), (self.encoder, self.target_encoder))
        for online, target in pairs:
            for source, destination in zip(online.parameters(), target.parameters(), strict=True):
                destination.mul_(momentum).add_(source.detach(), alpha=1 - momentum)

    @torch.no_grad()
    def represent(self, sequence: Tensor) -> Tensor:
        """(batch, N, width): the target encoder on the whole sequence, as used downstream."""
        represented: Tensor = self.target_encoder(self.target_embed(sequence))
        return represented
