"""The pose encoder's input: worldSign's per-joint channels on the V-JEPA tubelet (posa §2).

From the pose tokens (x, y and presence of the two frames of every step, in the clip's shoulder
units) ``JointFeatures`` rebuilds the 64 frames, re-centres every frame between its own
shoulders (worldSign's SignSpace, frame by frame), computes nine channels per joint and frame
(global position; position from the part's root and bone to the parent, both rescaled by the
part's extent in the frame; velocity; presence) and joins the two frames of each step again:
18 channels per joint and step. A missing joint is 0 in every channel but presence. A frame
without both shoulders, or with shoulders too close to be a real detection, keeps the clip's
own reference.
"""

from typing import Final

import torch
from torch import Tensor, nn

from signworld.data.pose.skeleton import PARTS, SHOULDER_COLUMNS
from signworld.data.pose.tokens import FRAMES_PER_STEP, JOINTS, TOKEN_CHANNELS
from signworld.data.pose.wholebody import MIN_SHOULDER_SHARE

FRAME_CHANNELS: Final = 9
"""Global (2), local (2), bone (2), velocity (2), presence (1)."""
STEP_CHANNELS: Final = FRAME_CHANNELS * FRAMES_PER_STEP
_MIN_EXTENT: Final = 1e-3
"""Smallest part extent a local position is divided by, in shoulder units."""
_VALUES_PER_FRAME: Final = 3
"""x, y and presence of one frame in a pose token."""


class JointFeatures(nn.Module):
    """``(batch, steps, 69, 6)`` pose tokens to ``(batch, steps, 69, 18)`` encoder inputs."""

    parents: Tensor
    roots: Tensor
    rooted: Tensor
    part_of: Tensor

    def __init__(self) -> None:
        super().__init__()
        per_joint = [(index, part) for index, part in enumerate(PARTS) for _ in range(part.size)]
        parents = [parent for part in PARTS for parent in part.parents]
        roots = [part.root if part.root is not None else 0 for _, part in per_joint]
        rooted = [part.root is not None for _, part in per_joint]
        part_of = [index for index, _ in per_joint]
        self.register_buffer("parents", torch.tensor(parents), persistent=False)
        self.register_buffer("roots", torch.tensor(roots), persistent=False)
        self.register_buffer("rooted", torch.tensor(rooted), persistent=False)
        self.register_buffer("part_of", torch.tensor(part_of), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tuple(tokens.shape[-2:]) != (len(JOINTS), TOKEN_CHANNELS):
            raise ValueError(f"expected pose tokens (..., {len(JOINTS)}, {TOKEN_CHANNELS})")
        batch, steps, joints = tokens.shape[:3]
        frames = (
            tokens.float()
            .reshape(batch, steps, joints, FRAMES_PER_STEP, _VALUES_PER_FRAME)
            .transpose(2, 3)
            .reshape(batch, steps * FRAMES_PER_STEP, joints, _VALUES_PER_FRAME)
        )
        present = frames[..., 2] > 0
        keep = present[..., None].to(frames.dtype)
        position = self._canonical(frames[..., :2], present) * keep

        rooted = self.rooted[:, None].to(position.dtype)
        root_seen = (present[:, :, self.roots] | ~self.rooted)[..., None].to(position.dtype)
        local = (position - position[:, :, self.roots] * rooted) * keep * root_seen
        parent_seen = present[:, :, self.parents][..., None].to(position.dtype)
        bone = (position - position[:, :, self.parents]) * keep * parent_seen
        extent = self._extent(local)
        velocity = torch.zeros_like(position)
        velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) * keep[:, 1:] * keep[:, :-1]

        features = torch.cat([position, local / extent, bone / extent, velocity, keep], dim=-1)
        per_step = features.reshape(batch, steps, FRAMES_PER_STEP, joints, FRAME_CHANNELS)
        return per_step.transpose(2, 3).reshape(batch, steps, joints, STEP_CHANNELS)

    @staticmethod
    def _canonical(position: Tensor, present: Tensor) -> Tensor:
        """Every frame re-centred between its shoulders, in units of their distance."""
        left, right = (position[:, :, column] for column in SHOULDER_COLUMNS)
        width = (left - right).norm(dim=-1)
        both = present[:, :, SHOULDER_COLUMNS[0]] & present[:, :, SHOULDER_COLUMNS[1]]
        usable = both & (width >= MIN_SHOULDER_SHARE)
        centre = torch.where(usable[..., None], (left + right) / 2, torch.zeros_like(left))
        scale = torch.where(usable, width, torch.ones_like(width))
        return (position - centre[:, :, None]) / scale[:, :, None, None]

    def _extent(self, local: Tensor) -> Tensor:
        """(batch, frames, joints, 1): the half-extent of each joint's part in the frame."""
        magnitude = local.abs().amax(dim=-1)
        extent = torch.stack([magnitude[..., p.start : p.stop].amax(dim=-1) for p in PARTS], -1)
        return extent.clamp_min(_MIN_EXTENT)[..., self.part_of][..., None]
