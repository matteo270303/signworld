"""Multi-level input of the physical predictor (§4.4.5, §4.5.2; PC6).

The distilled V-JEPA 2.1 predictor reads one level, the encoder's last, through
``predictor_embed = Linear(D, P)``. The physical predictor reads the fusion of the four
hierarchical levels (blocks 6/12/18/24 of the ViT-L), as the undistilled V-JEPA 2.1 does. The
fused input layer is a ``Linear(4 D, P)`` whose slice on the last level is the released layer
and whose slices on the three new levels start at zero: at initialisation the predictor computes
exactly what was released, and the new levels enter only as training moves those weights.
"""

import torch
from torch import nn


def fused_embedding(released: nn.Linear, levels: int) -> nn.Linear:
    """``Linear(levels * D, P)`` equal to ``released`` on the last level, zero elsewhere.

    The input is the concatenation of the levels along the feature axis, the last level last,
    as V-JEPA 2.1 concatenates its hierarchical outputs.
    """
    width = released.in_features
    fused = nn.Linear(levels * width, released.out_features, bias=released.bias is not None)
    with torch.no_grad():
        fused.weight.zero_()
        fused.weight[:, (levels - 1) * width :] = released.weight
        if released.bias is not None:
            fused.bias.copy_(released.bias)
    return fused.to(released.weight.device, released.weight.dtype)
