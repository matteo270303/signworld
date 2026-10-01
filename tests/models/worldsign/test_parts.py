"""LoRA, masks, sampling, augmentation, read-out and RoPE, each on a case with a known answer."""

import numpy as np
import pytest
import torch
from torch import nn

from signworld.data.augmentation import ClipAugmenter, View
from signworld.data.sampling import MotionGuidedSampler, UniformSampler
from signworld.experiment.train.config import AugmentationSettings, MaskSpec, SamplingSettings
from signworld.models.encoders.video_encoders import box_pool
from signworld.models.worldsign.lora import LoRALinear, adapter_parameters, inject
from signworld.models.worldsign.masking import (
    LambdaSchedule,
    Mask,
    MaskPolicy,
    MultiBlockMasks,
    TokenGrid,
    context_weights,
    token_roles,
)
from signworld.models.worldsign.readout import box_sum, membership, pool
from signworld.models.worldsign.rope import RoPE3D

GRID = TokenGrid()


# ----------------------------------------------------------------------------- LoRA


def test_lora_starts_as_the_frozen_layer_and_counts_as_documented() -> None:
    base = nn.Linear(1024, 3072)
    layer = LoRALinear(base, rank=16, alpha=16.0, splits=3)
    x = torch.randn(5, 1024)

    assert torch.equal(layer(x), base(x))
    assert sum(p.numel() for p in layer.adapter_parameters()) == 3 * 16 * (1024 + 1024)
    assert not any(p.requires_grad for p in layer.base.parameters())


def test_inject_replaces_named_linears_and_only_adapters_train() -> None:
    block = nn.ModuleDict(
        {"qkv": nn.Linear(8, 24), "proj": nn.Linear(8, 8), "other": nn.Linear(8, 8)}
    )
    block.requires_grad_(False)

    created = inject(block, ["qkv", "proj"], rank=2, alpha=2.0)

    assert len(created) == 2 and isinstance(block["qkv"], LoRALinear)
    assert isinstance(block["other"], nn.Linear)
    trainable = {id(p) for p in block.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in adapter_parameters(block)}


# ----------------------------------------------------------------------------- masks


def test_a_mask_is_the_same_tube_in_every_frame_and_covers_every_token() -> None:
    grid = TokenGrid()  # the full 32 steps: a cut would show in the last ones
    generator = torch.Generator().manual_seed(0)
    for spec in (MaskSpec(blocks=8, spatial_scale=0.15), MaskSpec(blocks=2, spatial_scale=0.7)):
        masks = MultiBlockMasks(spec, grid)
        mask = masks(64, generator)
        for indices in (mask.context[0], mask.target[0]):
            steps, spatial = grid.positions(indices)[:, 0], grid.positions(indices)[:, 1:]
            first = {tuple(p) for p in spatial[steps == 0].tolist()}
            assert all(
                {tuple(p) for p in spatial[steps == s].tolist()} == first for s in range(grid.steps)
            )
        assert mask.context.shape[1] + mask.target.shape[1] == grid.size  # nothing cut
        assert not set(mask.context[0].tolist()) & set(mask.target[0].tolist())
        assert torch.equal(mask.context, mask.context[:1].expand(64, -1))  # one per batch
        assert not torch.equal(masks(64, generator).context[0], mask.context[0])  # new draw
        assert 0.3 < mask.ratio < 0.999


def test_the_policy_draws_every_mask_kind_each_step() -> None:
    policy = MaskPolicy(
        (MaskSpec(blocks=8, spatial_scale=0.15), MaskSpec(blocks=2, spatial_scale=0.7)), GRID
    )

    masks = policy(3, torch.Generator().manual_seed(1))

    assert len(masks) == 2


def test_visible_tokens_weigh_one_over_the_root_of_their_distance() -> None:
    grid = TokenGrid(steps=1, rows=1, columns=5)
    mask = Mask(context=torch.tensor([[0, 1, 2, 3]]), target=torch.tensor([[4]]))

    weights = context_weights(mask, grid)

    assert torch.allclose(weights[0], torch.tensor([4.0, 3.0, 2.0, 1.0]).rsqrt())
    roles = token_roles(mask, grid, weight_distance=True)
    assert roles.masked[0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert roles.visible[0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert torch.allclose(roles.distance[0, :4], weights[0]) and roles.distance[0, 4] == 0.0


def test_in_the_cooldown_every_visible_token_weighs_one() -> None:
    grid = TokenGrid(steps=1, rows=1, columns=5)
    mask = Mask(context=torch.tensor([[0, 1, 2, 3]]), target=torch.tensor([[4]]))

    roles = token_roles(mask, grid)

    assert roles.distance[0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert LambdaSchedule.constant(0.5).at(0, 100) == 0.5


def test_lambda_warms_up_then_holds() -> None:
    schedule = LambdaSchedule(0.5, 0.1, 0.2)

    assert [schedule.at(s, 100) for s in (0, 15, 20, 90)] == [0.0, pytest.approx(0.25), 0.5, 0.5]


# ----------------------------------------------------------------------------- sampling


def test_motion_sampling_is_dense_where_the_clip_moves() -> None:
    frames = np.zeros((200, 64, 64, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    frames[100:150] = rng.integers(0, 255, (50, 64, 64, 3), dtype=np.uint8)  # motion here only
    settings = SamplingSettings()

    motion = MotionGuidedSampler(settings).select(frames)
    uniform = UniformSampler(settings).select(frames)

    assert len(motion) == len(uniform) == 64
    assert np.all(np.diff(motion) >= 0)
    in_motion = ((motion >= 100) & (motion < 151)).sum()
    assert in_motion > ((uniform >= 100) & (uniform < 151)).sum() + 10


# ----------------------------------------------------------------------------- augmentation


def test_the_view_moves_keypoints_where_it_moves_pixels() -> None:
    frames = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
    frames[:, 20, 40] = 255  # a bright pixel at x = 40.5 / 64, y = 20.5 / 64
    point = torch.tensor([[40.5 / 64, 20.5 / 64]])
    view = View(scale=1.1, shift=(0.05, -0.03), brightness=1.0, contrast=1.0, saturation=1.0)

    moved = ClipAugmenter.frames(frames, view)
    where = ClipAugmenter.points(point, view)[0] * 64 - 0.5

    brightest = divmod(int(moved[0, ..., 0].float().argmax()), 64)
    assert abs(brightest[0] - where[1].item()) <= 1 and abs(brightest[1] - where[0].item()) <= 1
    assert torch.equal(ClipAugmenter.frames(frames, View.identity()), frames)


def test_sampled_views_stay_within_their_ranges() -> None:
    augmenter = ClipAugmenter(AugmentationSettings())
    views = [augmenter.sample(torch.Generator().manual_seed(i)) for i in range(50)]

    assert all(0.9 <= v.scale <= 1.1 and abs(v.shift[0]) <= 0.1 for v in views)


# ----------------------------------------------------------------------------- read-out


def test_membership_and_pool_match_the_collaudo_box_pooling() -> None:
    torch.manual_seed(0)
    tokens = torch.randn(2, 3, 16, 16, 5)
    boxes = torch.tensor([[0.1, 0.2, 0.6, 0.9], [0.5, 0.0, 1.0, 0.4]]).expand(2, 3, 2, 4).clone()
    boxes[1, 2, 1] = torch.nan
    visible = torch.ones(2, 3, 2, dtype=torch.bool)
    visible[1, 2, 1] = False

    members = membership(boxes, visible, 16, 16)
    pooled = pool(tokens, members)

    for b in range(2):
        expected = box_pool(tokens[b], boxes[b].numpy(), visible[b].numpy())
        assert torch.allclose(pooled[b], expected, atol=1e-5)
    assert torch.equal(box_sum(torch.ones(2, 3, 16, 16), members), members.sum(dim=(-1, -2)))


# ----------------------------------------------------------------------------- RoPE


def test_rope_rotates_video_tokens_only_and_keeps_norms() -> None:
    grid = TokenGrid(steps=2, rows=2, columns=2)
    rope = RoPE3D(12, grid)
    cos, sin = rope.rotation(torch.device("cpu"))
    x = torch.randn(1, 1, grid.size + 3, 12)

    y = RoPE3D.apply(x, cos, sin)

    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.equal(y[..., grid.size :, :], x[..., grid.size :, :])  # queries unrotated
    assert torch.allclose(y[..., 0, :], x[..., 0, :])  # position (0, 0, 0): identity
    assert not torch.allclose(y[..., 1, :], x[..., 1, :])
