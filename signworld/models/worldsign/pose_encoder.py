"""The pose encoder ``G_ω``: worldSign's part-based encoder on the V-JEPA tubelet (posa §3).

Each articulator has its own spatial transformer (weights not shared: the two hands are not
interchangeable, a face is not a hand) over its joints at every step, averaged into one token.
The four tokens are concatenated, a temporal transformer runs over the 32 steps at their joint
width (4 x 128 = 512), and a LayerNorm with a projection gives ``s_t`` at ``output_dim``.
"""

import torch
from torch import Tensor, nn

from signworld.data.pose.skeleton import PARTS
from signworld.data.pose.tokens import STEPS
from signworld.experiment.train.config import PoseEncoderSettings

from .pose_features import STEP_CHANNELS, JointFeatures

_POSITION_STD = 0.02


def transformer(width: int, depth: int, heads: int, mlp_ratio: float, dropout: float) -> nn.Module:
    """``depth`` pre-norm PyTorch encoder blocks, as worldSign's encoder uses them."""
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=round(mlp_ratio * width),
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class PartEncoder(nn.Module):
    """One articulator: per-joint projection and joint type, spatial blocks, mean over joints."""

    def __init__(self, joints: int, settings: PoseEncoderSettings) -> None:
        super().__init__()
        width = settings.part_width
        self.project = nn.Linear(STEP_CHANNELS, width)
        self.joint_type = nn.Parameter(torch.empty(joints, width).normal_(std=_POSITION_STD))
        self.blocks = transformer(
            width, settings.part_depth, settings.heads, settings.mlp_ratio, settings.dropout
        )

    def forward(self, features: Tensor) -> Tensor:
        """(n, joints, 18) to (n, part_width)."""
        encoded: Tensor = self.blocks(self.project(features) + self.joint_type)
        return encoded.mean(dim=1)


class PoseEncoder(nn.Module):
    """``(batch, steps, 69, 6)`` pose tokens to ``s``: ``(batch, steps, output_dim)``.

    ``steps`` is the video grid's (32 for 64 frames in tubelets of 2): the temporal positions
    are learnt for that many steps.
    """

    def __init__(self, settings: PoseEncoderSettings, steps: int = STEPS) -> None:
        super().__init__()
        self.settings = settings
        self.steps = steps
        self.features = JointFeatures()
        self.parts = nn.ModuleList(PartEncoder(part.size, settings) for part in PARTS)
        width = len(PARTS) * settings.part_width
        self.temporal_position = nn.Parameter(torch.empty(steps, width).normal_(std=_POSITION_STD))
        self.temporal = transformer(
            width, settings.depth, settings.heads, settings.mlp_ratio, settings.dropout
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, settings.output_dim))

    @property
    def output_dim(self) -> int:
        return self.settings.output_dim

    def forward(self, tokens: Tensor) -> Tensor:
        features = self.features(tokens)
        batch, steps = features.shape[:2]
        if steps != self.steps:
            raise ValueError(f"expected {self.steps} steps, got {steps}")
        rows = features.flatten(0, 1)
        parts = [
            encode(rows[:, p.start : p.stop]) for encode, p in zip(self.parts, PARTS, strict=True)
        ]
        joined = torch.cat(parts, dim=-1).view(batch, steps, -1)
        temporal: Tensor = self.temporal(joined + self.temporal_position)
        latent: Tensor = self.head(temporal)
        return latent
