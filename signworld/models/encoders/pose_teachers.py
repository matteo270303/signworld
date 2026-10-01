"""Self-supervised pose teachers compared as targets of the physical level.

Both teachers share one backbone, a transformer over joint-by-step tokens (69 joints x 32
steps), and differ only in what they learn to predict for the ~90 % of tokens they cannot see,
sampled more often where joints move (Mao et al., 2023):

* ``MaskedMotionTeacher`` reconstructs the motion of the hidden joints (MAMP, Mao et al., 2023);
* ``SJEPATeacher`` predicts the latents that a slowly moving copy of its encoder gives the
  hidden joints, with centring and sharpening against collapse (S-JEPA, Abdelfattah and Alahi,
  2024), adapted to 2D sign poses: no horizontal flip, which would swap the dominant hand.

After pre-training, a teacher's full-sequence encoding, pooled per articulator, is the latent
the video predictor would regress.
"""

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from signworld.data.pose.tokens import JOINT_ARTICULATOR, JOINTS, STEPS, TOKEN_CHANNELS


@dataclass(frozen=True, slots=True)
class TeacherShape:
    width: int = 256
    depth: int = 8
    heads: int = 8
    predictor_depth: int = 5
    decoder_depth: int = 3
    mask_ratio: float = 0.9


def transformer(width: int, depth: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        4 * width,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)


class TokenEmbedding(nn.Module):
    """Linear projection of each token plus learned joint and step embeddings."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.project = nn.Linear(TOKEN_CHANNELS, width)
        self.joint = nn.Parameter(torch.zeros(len(JOINTS), width))
        self.step = nn.Parameter(torch.zeros(STEPS, width))
        nn.init.trunc_normal_(self.joint, std=0.02)
        nn.init.trunc_normal_(self.step, std=0.02)

    def positions(self) -> Tensor:
        """(steps * joints, width) position embedding of every token."""
        grid: Tensor = self.step[:, None] + self.joint[None]
        return grid.reshape(-1, self.joint.shape[1])

    def forward(self, tokens: Tensor) -> Tensor:
        """(batch, steps, joints, channels) to (batch, steps * joints, width)."""
        projected: Tensor = self.project(tokens).flatten(1, 2)
        return projected + self.positions()


def _frames(tokens: Tensor) -> Tensor:
    """(..., 6) tokens as (..., 2 frames, 3): x, y and presence of each frame."""
    return tokens.reshape(*tokens.shape[:-1], 2, 3)


def gather(x: Tensor, indices: Tensor) -> Tensor:
    """Rows ``indices`` (batch, k) of ``x`` (batch, n, d)."""
    return torch.gather(x, 1, indices[..., None].expand(-1, -1, x.shape[-1]))


def token_motion(tokens: Tensor) -> Tensor:
    """(batch, steps, joints, 4): displacement into each frame of the step, x and y.

    The first frame is compared with the last frame of the previous step, the second with the
    first; a displacement involving a missing keypoint is zero.
    """
    frames = _frames(tokens)
    xy, present = frames[..., :2], frames[..., 2:]
    previous = torch.cat([xy[:, :1, :, 1:], xy[:, :-1, :, 1:]], dim=1)
    previous_present = torch.cat([present[:, :1, :, 1:], present[:, :-1, :, 1:]], dim=1)
    first = (xy[..., 0, :] - previous[..., 0, :]) * present[..., 0, :] * previous_present[..., 0, :]
    second = (xy[..., 1, :] - xy[..., 0, :]) * present[..., 1, :] * present[..., 0, :]
    return torch.cat([first, second], dim=-1)


def motion_aware_mask(
    tokens: Tensor, ratio: float, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Visible and hidden token indices, each (batch, k), hiding moving joints more often.

    A token is hidden with a probability that grows with its motion, drawn without replacement
    by the Gumbel top-k trick; the constant 1 keeps still joints in play.
    """
    speed = token_motion(tokens).norm(dim=-1).flatten(1)
    weight = 1.0 + speed / speed.mean(dim=1, keepdim=True).clamp_min(1e-6)
    uniform = torch.rand(weight.shape, generator=generator, device=weight.device)
    keys = weight.log() - torch.log(-torch.log(uniform.clamp(1e-9, 1 - 1e-9)))
    hidden_count = round(ratio * keys.shape[1])
    order = keys.argsort(dim=1, descending=True)
    hidden = order[:, :hidden_count].sort(dim=1).values
    visible = order[:, hidden_count:].sort(dim=1).values
    return visible, hidden


def token_presence(tokens: Tensor) -> Tensor:
    """(batch, steps * joints): 1 when the joint is present in at least one frame."""
    presence: Tensor = _frames(tokens)[..., 2].amax(dim=-1).flatten(1)
    return presence


def pool_articulators(features: Tensor) -> Tensor:
    """(batch, steps * joints, d) to (batch, steps, 4, d): mean over each articulator's joints."""
    grid = features.reshape(features.shape[0], STEPS, len(JOINTS), features.shape[-1])
    groups = torch.as_tensor(JOINT_ARTICULATOR, device=features.device)
    pooled = [grid[:, :, groups == part].mean(dim=2) for part in range(int(groups.max()) + 1)]
    return torch.stack(pooled, dim=2)


class PoseTeacher(nn.Module, ABC):
    """Common backbone; subclasses define the pre-training objective."""

    def __init__(self, shape: TeacherShape) -> None:
        super().__init__()
        self.shape = shape
        self.embed = TokenEmbedding(shape.width)
        self.encoder = transformer(shape.width, shape.depth, shape.heads)
        self.mask_token = nn.Parameter(torch.zeros(shape.width))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @abstractmethod
    def loss(self, tokens: Tensor, generator: torch.Generator | None = None) -> Tensor: ...

    def after_step(self, progress: float) -> None:
        """Hook called after each optimiser step, ``progress`` in [0, 1]."""

    def encode(self, tokens: Tensor) -> Tensor:
        """(batch, steps, 4, width): the pose latent per step and articulator."""
        encoder, embed = self.representation_modules()
        return pool_articulators(encoder(embed(tokens)))

    def representation_modules(self) -> tuple[nn.Module, TokenEmbedding]:
        return self.encoder, self.embed

    def _with_mask_tokens(self, visible_features: Tensor, visible: Tensor) -> Tensor:
        """Encoded visible tokens in place, mask tokens elsewhere, positions added to all."""
        batch, total = visible.shape[0], STEPS * len(JOINTS)
        full = self.mask_token.expand(batch, total, -1).clone()
        full.scatter_(1, visible[..., None].expand(-1, -1, full.shape[-1]), visible_features)
        return full + self.embed.positions()


class MaskedMotionTeacher(PoseTeacher):
    """MAMP: predict the frame-to-frame motion of the hidden joints."""

    def __init__(self, shape: TeacherShape) -> None:
        super().__init__(shape)
        self.decoder = transformer(shape.width, shape.decoder_depth, shape.heads)
        self.head = nn.Linear(shape.width, 4)

    def loss(self, tokens: Tensor, generator: torch.Generator | None = None) -> Tensor:
        visible, hidden = motion_aware_mask(tokens, self.shape.mask_ratio, generator)
        encoded = self.encoder(gather(self.embed(tokens), visible))
        decoded = self.decoder(self._with_mask_tokens(encoded, visible))
        prediction = self.head(gather(decoded, hidden))
        target = gather(token_motion(tokens).flatten(1, 2), hidden)
        weight = gather(token_presence(tokens)[..., None], hidden)
        squared = ((prediction.float() - target) ** 2).mean(dim=-1, keepdim=True)
        loss: Tensor = (squared * weight).sum() / weight.sum().clamp_min(1.0)
        return loss


class SJEPATeacher(PoseTeacher):
    """S-JEPA: predict an EMA encoder's latents of the hidden joints, centred and sharpened."""

    centre: Tensor

    def __init__(
        self,
        shape: TeacherShape,
        momentum: tuple[float, float] = (0.999, 1.0),
        centre_rate: float = 0.9,
        temperatures: tuple[float, float] = (0.1, 0.06),
    ) -> None:
        super().__init__(shape)
        self.predictor = transformer(shape.width, shape.predictor_depth, shape.heads)
        self.target_embed = copy.deepcopy(self.embed)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in [*self.target_embed.parameters(), *self.target_encoder.parameters()]:
            parameter.requires_grad_(False)
        self.register_buffer("centre", torch.zeros(shape.width))
        self.momentum = momentum
        self.centre_rate = centre_rate
        self.temperatures = temperatures

    def loss(self, tokens: Tensor, generator: torch.Generator | None = None) -> Tensor:
        visible, hidden = motion_aware_mask(tokens, self.shape.mask_ratio, generator)
        view = augment_view(tokens, generator)
        encoded = self.encoder(gather(self.embed(view), visible))
        predicted = gather(self.predictor(self._with_mask_tokens(encoded, visible)), hidden)
        with torch.no_grad():
            target = gather(self.target_encoder(self.target_embed(tokens)), hidden).float()
            self._update_centre(target)
            target_distribution = torch.softmax((target - self.centre) / self.temperatures[1], -1)
        log_prediction = torch.log_softmax(predicted.float() / self.temperatures[0], dim=-1)
        cross_entropy = -(target_distribution * log_prediction).sum(dim=-1)
        weight = gather(token_presence(tokens)[..., None], hidden)[..., 0]
        return (cross_entropy * weight).sum() / weight.sum().clamp_min(1.0)

    def _update_centre(self, target: Tensor) -> None:
        batch_centre = target.mean(dim=(0, 1))
        self.centre.mul_(self.centre_rate).add_(batch_centre, alpha=1 - self.centre_rate)

    @torch.no_grad()
    def after_step(self, progress: float) -> None:
        start, end = self.momentum
        rate = end - (end - start) * (math.cos(math.pi * progress) + 1) / 2
        for online, target in (
            (self.embed, self.target_embed),
            (self.encoder, self.target_encoder),
        ):
            for source, destination in zip(online.parameters(), target.parameters(), strict=True):
                destination.mul_(rate).add_(source.detach(), alpha=1 - rate)

    def representation_modules(self) -> tuple[nn.Module, TokenEmbedding]:
        """The EMA encoder, which S-JEPA uses downstream."""
        return self.target_encoder, self.target_embed


def augment_view(tokens: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Another view of the same signing: in-plane rotation, scale, shift and jitter.

    No horizontal flip: it would swap the dominant hand and the signing space (§3.8).
    """
    batch = tokens.shape[0]

    def draw(low: float, high: float) -> Tensor:
        values = torch.rand(batch, generator=generator, device=tokens.device)
        return low + (high - low) * values

    angle = draw(-math.pi / 12, math.pi / 12)
    scale = draw(0.9, 1.1)
    shift = torch.stack([draw(-0.1, 0.1), draw(-0.1, 0.1)], dim=-1)
    cos, sin = angle.cos() * scale, angle.sin() * scale
    rotation = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)

    frames = _frames(tokens)
    xy, present = frames[..., :2], frames[..., 2:]
    moved = torch.einsum("bij,bstfj->bstfi", rotation, xy) + shift[:, None, None, None]
    jitter = 0.01 * torch.randn(moved.shape, generator=generator, device=tokens.device)
    return torch.cat([(moved + jitter) * present, present], dim=-1).flatten(-2)
