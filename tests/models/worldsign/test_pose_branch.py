"""The pose branch: the encoder from scratch, its views, the anchor (posa §3-§4)."""

import pytest
import torch
from torch import nn

from signworld.data.pose.tokens import JOINTS, STEPS
from signworld.experiment.train.config import PoseEncoderSettings, PoseViewSettings
from signworld.models.worldsign.lora import adapt_weight
from signworld.models.worldsign.pose_branch import (
    KeypointDecoder,
    PoseBranch,
    PoseViews,
    anchor_loss,
    keypoint_variance,
    step_confidence,
)
from tests.worldsign import POSE_WIDTH, pose_tokens

SETTINGS = PoseEncoderSettings(
    part_width=8, part_depth=1, depth=1, heads=2, output_dim=POSE_WIDTH, warmup_epochs=1.0
)


def test_lora_on_a_weight_starts_at_the_pretrained_weight_and_trains_only_its_factors() -> None:
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    x = torch.randn(2, 5, 8)
    before = attention(x, x, x)[0]

    adapter = adapt_weight(attention, "in_proj_weight", rank=2, alpha=2.0, splits=3)
    after = attention(x, x, x)[0]
    after.sum().backward()

    assert torch.allclose(before, after)
    assert len(adapter.down) == 3  # q, k and v apart
    original = attention.parametrizations.in_proj_weight.original
    assert not original.requires_grad and original.grad is None
    assert all(b.grad is not None for b in adapter.up)
    with torch.no_grad():
        adapter.up[0].fill_(1.0)
    assert not torch.allclose(attention(x, x, x)[0], before)


def test_the_branch_encodes_one_vector_per_step() -> None:
    branch = PoseBranch.from_settings(SETTINGS)

    assert branch.target(pose_tokens(2)).shape == (2, STEPS, POSE_WIDTH)


def test_the_clean_sequence_is_the_first_view_and_the_target() -> None:
    branch = PoseBranch.from_settings(SETTINGS).eval()
    tokens = pose_tokens(2)

    views = branch.encode_views(tokens, torch.Generator().manual_seed(0))

    assert views.shape == (1 + SETTINGS.views.count, 2, STEPS, POSE_WIDTH)
    assert torch.allclose(views[0], branch.target(tokens), atol=1e-5)


def test_views_without_nuisances_are_the_clean_sequence() -> None:
    quiet = PoseViewSettings(
        rotation_degrees=0.0,
        affine_probability=0.0,
        noise_body=0.0,
        noise_hands=0.0,
        noise_face=0.0,
        mask_probability=0.0,
    )
    tokens = pose_tokens(2)

    views = PoseViews(quiet)(tokens, torch.Generator().manual_seed(0))

    assert all(torch.allclose(view, tokens, atol=1e-6) for view in views)


def test_views_keep_missing_joints_missing_and_repeat_with_the_seed() -> None:
    tokens = pose_tokens(2)
    views = PoseViews(PoseViewSettings())
    missing = tokens.reshape(*tokens.shape[:-1], 2, 3)[..., 2] == 0

    first = views(tokens, torch.Generator().manual_seed(3))
    again = views(tokens, torch.Generator().manual_seed(3))

    for view, repeated in zip(first, again, strict=True):
        assert torch.equal(view, repeated)
        assert (view.reshape(*view.shape[:-1], 2, 3)[..., 2][missing] == 0).all()


def test_the_decoder_gives_the_69_joints_of_a_step() -> None:
    decoder = KeypointDecoder(POSE_WIDTH)

    assert decoder(torch.randn(1, 3, POSE_WIDTH)).shape == (1, 3, len(JOINTS), 2)


def test_predicting_each_joint_mean_costs_one() -> None:
    generator = torch.Generator().manual_seed(0)
    keypoints = torch.randn(16, 32, len(JOINTS), 2, generator=generator) * 3.0 + 1.0
    weights = (torch.rand(16, 32, len(JOINTS), generator=generator) > 0.2).float()
    variance = keypoint_variance(keypoints, weights)
    total = weights.flatten(0, 1).sum(dim=0)[:, None]
    mean = (keypoints * weights[..., None]).flatten(0, 1).sum(dim=0) / total

    loss = anchor_loss(mean.expand_as(keypoints), keypoints, weights, variance)

    assert loss.item() == pytest.approx(1.0, rel=1e-4)
    assert anchor_loss(keypoints, keypoints, weights, variance).item() == 0.0


def test_step_confidence_is_the_mean_presence_of_the_joints() -> None:
    weights = torch.ones(1, 2, len(JOINTS))
    weights[0, 1, : len(JOINTS) // 2 + 1] = 0.0

    confidence = step_confidence(weights)

    assert confidence[0, 0] == 1.0 and confidence[0, 1] < 0.5


def test_the_anchor_needs_the_keypoint_scale_first() -> None:
    branch = PoseBranch.from_settings(SETTINGS)
    latent = branch.target(pose_tokens(1))
    keypoints, weights = torch.randn(1, 32, len(JOINTS), 2), torch.ones(1, 32, len(JOINTS))

    with pytest.raises(RuntimeError, match="keypoint variance"):
        branch.anchor(latent, keypoints, weights)
    branch.fit(keypoints, weights)
    assert torch.isfinite(branch.anchor(latent, keypoints, weights))
