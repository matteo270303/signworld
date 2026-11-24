"""The fused rotation of Meta's RoPE attention computes Meta's own forward."""

import sys
from typing import Any

import pytest
import torch

from signworld.models.worldsign.meta_rope import fuse_rope
from tests.worldsign import HUB, meta_modules, needs_hub


def _rope_attention(**kwargs: Any) -> Any:
    sys.path.insert(0, str(HUB))
    try:
        from app.vjepa_2_1.models.utils.modules import RoPEAttention  # noqa: PLC0415
    finally:
        sys.path.remove(str(HUB))
    return RoPEAttention(**kwargs)


@needs_hub
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_one_rotation_of_the_head_is_metas_three(masked: bool, dtype: torch.dtype) -> None:
    """ViT-L's head: 64 channels, 3 x 20 rotated, 4 passing; interpolated positions."""
    torch.manual_seed(0)
    attention = _rope_attention(dim=128, num_heads=2, qkv_bias=True, grid_size=4,
                                interpolate_rope=True).eval()  # fmt: skip
    tokens = 2 * 4 * 4
    x = torch.randn(3, tokens, 128)
    mask = torch.stack([torch.randperm(tokens)[:11].sort().values for _ in range(3)])
    if masked:
        x = x[:, :11]
    arguments = {"mask": mask} if masked else {}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected, _ = attention(x, **arguments)
        assert fuse_rope(attention) == 1
        fused, _ = attention(x, **arguments)

    assert attention.meta_forward is not None and expected.dtype == fused.dtype
    assert torch.equal(fused, expected)


@needs_hub
def test_the_tiny_encoder_is_unchanged_whole_and_masked() -> None:
    encoder, _ = meta_modules()
    encoder.eval()
    clip = torch.randn(2, 3, 4, 64, 64)
    context = torch.stack([torch.arange(0, 32, 3)[:9], torch.arange(1, 32, 3)[:9]])
    with torch.no_grad():
        expected = [encoder(clip), encoder(clip, masks=[context])]
        assert fuse_rope(encoder) == len(encoder.blocks)
        fused = [encoder(clip), encoder(clip, masks=[context])]

    for before, after in zip(expected, fused, strict=True):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
