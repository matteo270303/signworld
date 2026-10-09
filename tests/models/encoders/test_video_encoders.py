"""The frozen-video interface: input layout, token grid, box pooling and the fused predictor input.

A tiny patch-embedding module stands in for V-JEPA, so nothing here loads a checkpoint.
"""

import numpy as np
import pytest
import torch
from torch import nn

from signworld.models.encoders.predictor_fusion import fused_embedding
from signworld.models.encoders.video_encoders import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FrozenVideoEncoder,
    box_pool,
    model_input,
    token_grid,
)


class TinyEncoder(nn.Module):
    """Tubelets of 2 x 16 x 16 pixels to 8 channels, tokens in (time, row, column) order."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.patch = nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.patch(video).flatten(2).transpose(1, 2)


def tiny_encoder() -> FrozenVideoEncoder:
    return FrozenVideoEncoder("tiny", TinyEncoder(), 8, "vjepa2_1")


def test_model_input_is_channel_first_and_imagenet_normalised() -> None:
    frames = torch.zeros(1, 4, 32, 32, 3, dtype=torch.uint8)
    frames[..., 0] = 255  # pure red

    video = model_input(frames)

    assert video.shape == (1, 3, 4, 32, 32)
    assert torch.allclose(video[0, 0], torch.tensor((1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]))
    assert torch.allclose(video[0, 1], torch.tensor(-IMAGENET_MEAN[1] / IMAGENET_STD[1]))


def test_model_input_refuses_float_or_channel_first_frames() -> None:
    with pytest.raises(ValueError, match="uint8"):
        model_input(torch.zeros(1, 4, 32, 32, 3))
    with pytest.raises(ValueError, match="uint8"):
        model_input(torch.zeros(1, 3, 4, 32, 32, dtype=torch.uint8))


def test_token_grid_keeps_time_row_column_order() -> None:
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 24, 1)

    grid = token_grid(tokens, frames=4, height=48, width=64)

    assert grid.shape == (1, 2, 3, 4, 1)
    assert grid[0, 1, 2, 3, 0] == 1 * 12 + 2 * 4 + 3


def test_encoder_tokens_form_the_patch_grid() -> None:
    frames = torch.randint(0, 256, (2, 4, 32, 48, 3), dtype=torch.uint8)

    grid = tiny_encoder().tokens(frames)

    assert grid.shape == (2, 2, 2, 3, 8)


def test_box_pool_averages_the_patches_a_box_touches() -> None:
    columns = torch.arange(16, dtype=torch.float32)
    grid = columns.view(1, 1, 16, 1).expand(2, 16, 16, 1).clone()
    boxes = np.array(
        [
            [[0.5, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1]],
        ]
    )
    visible = np.array([[True, True], [True, False]])

    pooled = box_pool(grid, boxes, visible)

    assert pooled[0, 0, 0] == pytest.approx(11.5)  # right half: columns 8..15
    assert pooled[0, 1, 0] == pytest.approx(7.5)  # whole frame
    assert pooled[1, 0, 0] == pytest.approx(15.0)  # a box on the border keeps its last patch
    assert pooled[1, 1, 0] == 0.0  # not visible
    assert not torch.isnan(pooled).any()


def test_fused_embedding_reproduces_the_released_layer_on_the_last_level() -> None:
    torch.manual_seed(0)
    released = nn.Linear(8, 4)
    levels = [torch.randn(3, 5, 8) for _ in range(4)]

    fused = fused_embedding(released, 4)

    assert fused.in_features == 32
    assert torch.allclose(fused(torch.cat(levels, dim=-1)), released(levels[-1]), atol=1e-6)
    assert fused.weight[:, :24].abs().sum() == 0
