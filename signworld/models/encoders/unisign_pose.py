"""The pose encoder of Uni-Sign (Li et al., 2025), the target of the physical level (§4.4.3).

An independent implementation of the architecture the paper and its checkpoints describe: per
articulator, a linear projection of (x, y, score), three spatial graph-convolution blocks and
three spatio-temporal blocks (temporal kernel 5), then the mean over the keypoints. Hands add
the feature of their wrist from the body branch, the face that of the nose; the left hand
shares every weight with the right. Parameter names follow the released checkpoints, so their
state dict loads as it is.
"""

from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Final, Self

import torch
from torch import Tensor, nn

from ..pose.wholebody import ARTICULATOR_INDICES, Articulator

PROJECTION_CHANNELS: Final = 64
SPATIAL_CHANNELS: Final = (64, 128, 256)
TEMPORAL_BLOCKS: Final = 3
TEMPORAL_KERNEL: Final = 5
OUTPUT_CHANNELS: Final = 256
ADJACENCY_PARTITIONS: Final = 2
"""Uni-Sign's 'distance' graph strategy with one hop: the node itself and its neighbours."""

# Body keypoint whose feature each branch receives: left wrist, right wrist, nose.
_BODY_ANCHOR: Final[dict[Articulator, int]] = {
    Articulator.LEFT_HAND: -2,
    Articulator.RIGHT_HAND: -1,
    Articulator.FACE: 0,
}


class GraphConvolution(nn.Module):
    """Pointwise projection into one feature map per partition, mixed by a learned adjacency."""

    def __init__(self, in_channels: int, out_channels: int, nodes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * ADJACENCY_PARTITIONS, kernel_size=1)
        self.A = nn.Parameter(torch.zeros(ADJACENCY_PARTITIONS, nodes, nodes))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """(batch, channels, time, nodes) to (batch, out_channels, time, nodes)."""
        batch, _, time, nodes = x.shape
        features = self.conv(x).view(batch, ADJACENCY_PARTITIONS, -1, time, nodes)
        mixed = torch.einsum("bkctv,kvw->bctw", features, self.A)
        activated: Tensor = torch.relu(self.bn(mixed))
        return activated


class GraphBlock(nn.Module):
    """Graph convolution, optional temporal convolution, residual connection, ReLU."""

    def __init__(self, in_channels: int, out_channels: int, nodes: int, temporal_kernel: int):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, nodes)
        self.tcn: nn.Module = nn.Identity()
        if temporal_kernel > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(temporal_kernel, 1),
                    padding=(temporal_kernel // 2, 0),
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.residual: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = torch.relu(self.tcn(self.gcn(x)) + self.residual(x))
        return output


def _chain(channels: list[int], nodes: int, temporal_kernel: int) -> nn.Sequential:
    """Blocks named ``layer{stage}_{depth}``, as in the released checkpoints."""
    chain = nn.Sequential()
    for stage, (previous, current) in enumerate(pairwise(channels)):
        chain.add_module(f"layer{stage}_0", GraphBlock(previous, current, nodes, temporal_kernel))
    return chain


def _temporal_chain(nodes: int) -> nn.Sequential:
    chain = nn.Sequential()
    for depth in range(TEMPORAL_BLOCKS):
        block = GraphBlock(OUTPUT_CHANNELS, OUTPUT_CHANNELS, nodes, TEMPORAL_KERNEL)
        chain.add_module(f"layer0_{depth}", block)
    return chain


class UniSignPoseEncoder(nn.Module):
    """Four articulator branches; returns one 256-d feature per frame and articulator."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_linear = nn.ModuleDict()
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        for part in (Articulator.BODY, Articulator.RIGHT_HAND, Articulator.FACE):
            nodes = len(ARTICULATOR_INDICES[part])
            self.proj_linear[part] = nn.Linear(3, PROJECTION_CHANNELS)
            self.gcn_modules[part] = _chain([PROJECTION_CHANNELS, *SPATIAL_CHANNELS], nodes, 1)
            self.fusion_gcn_modules[part] = _temporal_chain(nodes)
        for modules in (self.proj_linear, self.gcn_modules, self.fusion_gcn_modules):
            modules[Articulator.LEFT_HAND] = modules[Articulator.RIGHT_HAND]

    def forward(self, parts: Mapping[Articulator, Tensor]) -> dict[Articulator, Tensor]:
        """Map (batch, time, keypoints, 3) inputs to (batch, time, 256) features."""
        spatial = {
            part: self.gcn_modules[part](self.proj_linear[part](parts[part]).permute(0, 3, 1, 2))
            for part in Articulator
        }
        body = spatial[Articulator.BODY].detach()
        features = {}
        for part in Articulator:
            x = spatial[part]
            if part in _BODY_ANCHOR:
                x = x + body[..., _BODY_ANCHOR[part]].unsqueeze(-1)
            features[part] = self.fusion_gcn_modules[part](x).mean(dim=-1).transpose(1, 2)
        return features

    @classmethod
    def from_checkpoint(cls, path: Path) -> Self:
        """Load the pose-encoder weights of a Uni-Sign checkpoint, ignoring the rest."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        encoder = cls()
        prefixes = ("proj_linear.", "gcn_modules.", "fusion_gcn_modules.")
        own = {key: value for key, value in state.items() if key.startswith(prefixes)}
        missing, unexpected = encoder.load_state_dict(own, strict=False)
        if missing or unexpected:
            raise ValueError(f"{path}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return encoder
