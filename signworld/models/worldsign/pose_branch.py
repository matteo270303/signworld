"""The pose side of WorldSign: level 0, the target of the physical level (posa §3-§4).

``PoseBranch`` holds the pose encoder ``G_ω``, trained from scratch with the rest of the
model, the keypoint decoder of the anchor and the views of the invariance term:

* ``target`` gives ``s``, (batch, 32, C), one vector per step; the physical level reads it
  with the gradient stopped;
* ``encode_views`` gives the four views of ``L_inv`` and ``SIGReg_posa``: the clean sequence
  and three draws of nuisances (``PoseViews``), in one pass of the encoder;
* ``anchor`` is ``L_anchor``: one linear decoder from ``s_t`` to the (x, y) of the 69 joints,
  divided by the keypoint variance so that predicting each joint's mean is worth 1.

The invariance and SIGReg terms are the objective's (``loss.worldsign``).
"""

import math

import torch
from torch import Tensor, nn

from signworld.data.pose.skeleton import PARTS
from signworld.data.pose.tokens import FRAMES_PER_STEP, JOINTS, STEPS
from signworld.data.pose.wholebody import Articulator
from signworld.experiment.train.config import PoseEncoderSettings, PoseViewSettings

from .pose_encoder import PoseEncoder

_VALUES_PER_FRAME = 3


class PoseViews:
    """Views of the same signing that differ by nuisances only (posa §4.1).

    Each draw, per clip:

    * the camera: a rotation about the origin between the shoulders, uniform in
      ``±rotation_degrees``; with probability ``affine_probability`` each, a horizontal scale
      uniform in ``1 ± aspect`` and a shear ``x ← x + h·y`` with ``|h| ≤ shear``;
    * Gaussian noise on the present joints, with each articulator's standard deviation;
    * with probability ``mask_probability``, ``mask_joints`` joints of the fingers and the
      face (roots excluded: a missing root would void its whole part's local positions)
      hidden over ``mask_steps`` consecutive steps.

    Missing joints stay missing. Draws come from ``generator`` (on the CPU), so a step's views
    are reproducible.
    """

    def __init__(self, settings: PoseViewSettings) -> None:
        self.settings = settings
        spread = {
            Articulator.BODY: settings.noise_body,
            Articulator.LEFT_HAND: settings.noise_hands,
            Articulator.RIGHT_HAND: settings.noise_hands,
            Articulator.FACE: settings.noise_face,
        }
        self.noise = torch.tensor([spread[p.articulator] for p in PARTS for _ in range(p.size)])
        self.maskable = torch.tensor(
            [
                column
                for p in PARTS
                if p.root is not None
                for column in range(p.start, p.stop)
                if column != p.root
            ]
        )
        if settings.mask_joints > len(self.maskable):
            raise ValueError(f"only {len(self.maskable)} joints can be hidden")

    @property
    def count(self) -> int:
        return self.settings.count

    def __call__(self, tokens: Tensor, generator: torch.Generator) -> list[Tensor]:
        """``count`` independent views of (batch, steps, joints, 6) tokens, each that shape."""
        return [self.draw(tokens, generator) for _ in range(self.count)]

    def draw(self, tokens: Tensor, generator: torch.Generator) -> Tensor:
        """One view of (batch, steps, joints, 6) tokens."""
        frames = tokens.reshape(*tokens.shape[:-1], FRAMES_PER_STEP, _VALUES_PER_FRAME)
        position, present = frames[..., :2], frames[..., 2:]
        camera = self._camera(len(tokens), generator).to(tokens.device, tokens.dtype)
        noise = torch.randn(position.shape, generator=generator) * self.noise[:, None, None]
        moved = torch.einsum("bij,bsnfj->bsnfi", camera, position) + noise.to(position)
        hidden = self._hidden(len(tokens), tokens.shape[1], generator).to(tokens.device)
        kept = present * ~hidden[..., None, None]
        return torch.cat([moved * kept, kept], dim=-1).reshape(tokens.shape)

    def _camera(self, batch: int, generator: torch.Generator) -> Tensor:
        """(batch, 2, 2): rotation · horizontal scale · shear, one per clip."""
        s = self.settings
        uniform = torch.rand(batch, 3, generator=generator) * 2 - 1
        chosen = torch.rand(batch, 2, generator=generator) < s.affine_probability
        angle = uniform[:, 0] * math.radians(s.rotation_degrees)
        scale = 1 + uniform[:, 1] * s.aspect * chosen[:, 0]
        shear = uniform[:, 2] * s.shear * chosen[:, 1]
        cos, sin = angle.cos(), angle.sin()
        rotation = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
        one, zero = torch.ones(batch), torch.zeros(batch)
        affine = torch.stack(
            [torch.stack([scale, scale * shear], -1), torch.stack([zero, one], -1)], -2
        )
        return rotation @ affine

    def _hidden(self, batch: int, steps: int, generator: torch.Generator) -> Tensor:
        """(batch, steps, joints) bool: the joints this draw hides."""
        s = self.settings
        span = min(s.mask_steps, steps)
        on = torch.rand(batch, generator=generator) < s.mask_probability
        start = (torch.rand(batch, generator=generator) * (steps - span + 1)).long()
        order = torch.rand(batch, len(self.maskable), generator=generator).argsort(dim=1)
        chosen = torch.zeros(batch, len(JOINTS), dtype=torch.bool)
        chosen.scatter_(1, self.maskable[order[:, : s.mask_joints]], value=True)
        step = torch.arange(steps)
        during = (step >= start[:, None]) & (step < start[:, None] + span)
        hidden: Tensor = on[:, None, None] & during[:, :, None] & chosen[:, None, :]
        return hidden


class KeypointDecoder(nn.Module):
    """``D``: one linear map from ``s_t`` to the (x, y) of the 69 joints of the step."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.head = nn.Linear(width, 2 * len(JOINTS))

    def forward(self, latent: Tensor) -> Tensor:
        """(..., C) to (..., 69, 2), joints in ``JOINTS`` order."""
        decoded: Tensor = self.head(latent)
        return decoded.reshape(*decoded.shape[:-1], len(JOINTS), 2)


def step_confidence(weights: Tensor) -> Tensor:
    """(batch, steps) ``c_t``: the mean presence of the 69 joints at each step."""
    return weights.float().mean(dim=-1)


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
    """The pose encoder with the anchor's decoder and keypoint scale, and the views."""

    keypoint_variance: Tensor

    def __init__(self, encoder: PoseEncoder, decoder: KeypointDecoder, views: PoseViews) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.views = views
        self.register_buffer("keypoint_variance", torch.tensor(float("nan")))

    @classmethod
    def from_settings(cls, settings: PoseEncoderSettings, steps: int = STEPS) -> "PoseBranch":
        return cls(
            PoseEncoder(settings, steps),
            KeypointDecoder(settings.output_dim),
            PoseViews(settings.views),
        )

    def fit(self, keypoints: Tensor, weights: Tensor) -> None:
        """Set the scale of L_anchor from the training clips (§4.6: nothing from validation)."""
        self.keypoint_variance.copy_(keypoint_variance(keypoints, weights))

    def target(self, tokens: Tensor) -> Tensor:
        """(batch, steps, C) ``s``: the clean sequence encoded, the physical level's target."""
        latent: Tensor = self.encoder(tokens)
        return latent

    def encode_views(self, tokens: Tensor, generator: torch.Generator) -> Tensor:
        """(views, batch, steps, C): the clean sequence first, then the drawn views, encoded
        in one pass (the encoder has no batch statistics, so the clips do not mix)."""
        everything = torch.cat([tokens, *self.views(tokens, generator)])
        latent: Tensor = self.encoder(everything)
        return latent.reshape(1 + self.views.count, len(tokens), *latent.shape[1:])

    def anchor(self, latent: Tensor, keypoints: Tensor, weights: Tensor) -> Tensor:
        if bool(torch.isnan(self.keypoint_variance)):
            raise RuntimeError("fit the keypoint variance on the training clips first")
        return anchor_loss(self.decoder(latent), keypoints, weights, self.keypoint_variance)
