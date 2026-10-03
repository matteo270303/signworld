"""The pose side of WorldSign: level 0, the target of the physical level (posa §3-§4).

``PoseBranch`` holds the pose encoder ``G_ω``, trained from scratch with the rest of the
model, the keypoint decoder of the anchor and the views of the invariance term:

* ``target`` gives ``s``, (batch, 32, C), one vector per step; the physical level reads it
  with the gradient stopped;
* ``view`` gives the second view of ``L_inv``: an in-plane rotation and keypoint noise, the
  nuisances only (``PoseViews``);
* ``anchor`` is ``L_anchor``: one linear decoder from ``s_t`` to the (x, y) of the 69 joints,
  divided by the keypoint variance so that predicting each joint's mean is worth 1.

The invariance and SIGReg terms are the objective's (``loss.worldsign``).
"""

import math

import torch
from torch import Tensor, nn

from signworld.data.pose.tokens import FRAMES_PER_STEP, JOINTS, STEPS
from signworld.experiment.train.config import PoseEncoderSettings, PoseViewSettings

from .pose_encoder import PoseEncoder

_VALUES_PER_FRAME = 3


class PoseViews:
    """A second view of the same signing: in-plane rotation and noise on the present joints.

    The rotation turns about the origin between the shoulders, uniform in
    ``±rotation_degrees`` per clip; the noise is Gaussian with ``noise`` standard deviation in
    shoulder units. Missing joints stay missing. Draws come from ``generator`` (on the CPU),
    so a step's views are reproducible.
    """

    def __init__(self, settings: PoseViewSettings) -> None:
        self.settings = settings

    def __call__(self, tokens: Tensor, generator: torch.Generator) -> Tensor:
        """(batch, steps, joints, 6) tokens to tokens of the same shape."""
        frames = tokens.reshape(*tokens.shape[:-1], FRAMES_PER_STEP, _VALUES_PER_FRAME)
        position, present = frames[..., :2], frames[..., 2:]
        half = math.radians(self.settings.rotation_degrees)
        angle = (torch.rand(len(tokens), generator=generator) * 2 - 1) * half
        cos, sin = angle.cos(), angle.sin()
        rotation = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
        rotation = rotation.to(tokens.device, tokens.dtype)
        noise = torch.randn(position.shape, generator=generator) * self.settings.noise
        moved = torch.einsum("bij,bsnfj->bsnfi", rotation, position) + noise.to(position)
        return torch.cat([moved * present, present], dim=-1).reshape(tokens.shape)


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

    def view(self, tokens: Tensor, generator: torch.Generator) -> Tensor:
        """The second view's tokens for ``L_inv``."""
        return self.views(tokens, generator)

    def anchor(self, latent: Tensor, keypoints: Tensor, weights: Tensor) -> Tensor:
        if bool(torch.isnan(self.keypoint_variance)):
            raise RuntimeError("fit the keypoint variance on the training clips first")
        return anchor_loss(self.decoder(latent), keypoints, weights, self.keypoint_variance)
