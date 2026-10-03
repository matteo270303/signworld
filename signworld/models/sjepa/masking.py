"""MAMP's motion-aware masking, which S-JEPA uses on its targets (MAMP §3.3-§3.4).

The motion of a sequence ``S`` (batch, T, V, C) with stride ``m = l`` is ``M_i = S_i - S_{i-m}``
for ``i ≥ m``, its first ``m`` frames padded by replicating ``M_{m:2m}``. Reshaped into the
segments of the embedding, ``I = Σ |M|`` over each segment's frames and channels is the motion
intensity of every token; ``π = softmax(I / τ)`` over the tokens, and the hidden tokens are the
top ``round(r · N)`` of ``log π + g`` with Gumbel noise ``g``: a draw without replacement that
favours moving joints.
"""

import torch
from torch import Tensor
from torch.nn import functional


def motion_intensity(sequence: Tensor, segment: int) -> Tensor:
    """(batch, T / segment · V) motion intensity of every token, segment-major."""
    batch, frames, joints, channels = sequence.shape
    if frames % segment or frames < 2 * segment:
        raise ValueError(f"{frames} frames do not fit segments of {segment}")
    difference = sequence[:, segment:] - sequence[:, :-segment]
    motion = torch.cat([difference[:, :segment], difference], dim=1)
    per_segment = motion.abs().reshape(batch, frames // segment, segment, joints, channels)
    return per_segment.sum(dim=(2, 4)).flatten(1)


def motion_aware_mask(
    sequence: Tensor, segment: int, ratio: float, temperature: float, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Visible and hidden token indices, each (batch, k) and sorted."""
    intensity = motion_intensity(sequence.float(), segment)
    log_probability = functional.log_softmax(intensity / temperature, dim=1)
    uniform = torch.rand(intensity.shape, generator=generator).clamp(1e-9, 1 - 1e-9)
    gumbel = -torch.log(-torch.log(uniform)).to(intensity.device)
    order = (log_probability + gumbel).argsort(dim=1, descending=True)
    hidden_count = round(ratio * intensity.shape[1])
    hidden = order[:, :hidden_count].sort(dim=1).values
    visible = order[:, hidden_count:].sort(dim=1).values
    return visible, hidden
