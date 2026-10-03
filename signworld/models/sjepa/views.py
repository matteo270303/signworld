"""S-JEPA's input pipeline: the trim-and-resize of a sequence and the views (paper §3-§4.2).

``trim_and_resize`` keeps a contiguous share of the valid frames (random in training, centred at
test time, at least ``MIN_FRAMES`` as MAMP's ``valid_crop_resize``) and resizes it to the fixed
length by linear interpolation in time. ``SkeletonViews`` makes the view encoder's input: a
rotation about the skeleton's own vertical axis (Rodrigues' formula, the axis from the pelvis to
the neck), a translation and a spatial flip that mirrors x and swaps left and right joints.
"""

import math
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional

from .config import SkeletonViewSettings

MIN_FRAMES: Final = 64
"""MAMP's shortest kept window (``valid_crop_resize``), or the whole sequence if shorter."""


def trim_and_resize(
    sequence: Tensor, valid: int, frames: int, share: float, offset: float | None = None
) -> Tensor:
    """(T_raw, V, C) to (frames, V, C): ``share`` of the ``valid`` first frames, resized.

    ``offset`` in [0, 1] places the window among the possible starts; None centres it.
    """
    if not 0 < valid <= len(sequence):
        raise ValueError(f"{valid} valid frames in a sequence of {len(sequence)}")
    length = min(valid, max(MIN_FRAMES, math.floor(valid * share)))
    room = valid - length
    start = room // 2 if offset is None else min(room, math.floor(offset * (room + 1)))
    window = sequence[start : start + length].float()
    joints, channels = window.shape[1:]
    series = window.reshape(length, joints * channels).T[None]  # (1, V·C, length)
    resized = functional.interpolate(series, size=frames, mode="linear", align_corners=False)
    return resized[0].T.reshape(frames, joints, channels)


class SkeletonViews:
    """A view of a batch of skeletons (batch, T, V, 3): rotation, translation, flip."""

    def __init__(self, settings: SkeletonViewSettings, joints: int) -> None:
        self.settings = settings
        order = list(range(joints))
        for left, right in settings.left_right:
            order[left], order[right] = right, left
        self.flipped = torch.tensor(order)

    def __call__(self, sequence: Tensor, generator: torch.Generator) -> Tensor:
        batch = len(sequence)
        s = self.settings
        angle = (torch.rand(batch, generator=generator) * 2 - 1) * s.rotation
        shift = (torch.rand(batch, 3, generator=generator) * 2 - 1) * s.translation
        flip = torch.rand(batch, generator=generator) < s.flip_probability
        view = self._rotate(sequence.float(), angle.to(sequence.device))
        view = view + shift.to(view)[:, None, None]
        mirrored = view[:, :, self.flipped.to(view.device)] * torch.tensor(
            [-1.0, 1.0, 1.0], device=view.device
        )
        return torch.where(flip.to(view.device)[:, None, None, None], mirrored, view)

    def _rotate(self, sequence: Tensor, angle: Tensor) -> Tensor:
        """Each skeleton turned by ``angle`` about its vertical axis, through its pelvis."""
        pelvis = sequence[:, :, self.settings.pelvis].mean(dim=1)  # (batch, 3)
        neck = sequence[:, :, self.settings.neck].mean(dim=1)
        axis = functional.normalize(neck - pelvis, dim=-1)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        cross = torch.stack(
            [
                torch.stack([zero, -z, y], -1),
                torch.stack([z, zero, -x], -1),
                torch.stack([-y, x, zero], -1),
            ],
            -2,
        )
        identity = torch.eye(3, device=sequence.device).expand_as(cross)
        sin, cos = angle.sin()[:, None, None], angle.cos()[:, None, None]
        rotation = identity + sin * cross + (1 - cos) * cross @ cross
        centred = sequence - pelvis[:, None, None]
        return torch.einsum("bij,btvj->btvi", rotation, centred) + pelvis[:, None, None]
