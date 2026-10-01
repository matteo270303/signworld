"""The pose branch: S-JEPA loaded from its checkpoint, adapted, and the anchor."""

from pathlib import Path

import pytest
import torch
from torch import nn

from signworld.data.pose.tokens import JOINT_ARTICULATOR, JOINTS
from signworld.experiment.train.config import PoseEncoderSettings
from signworld.models.worldsign.lora import LoRAWeight, adapt_weight, adapter_parameters
from signworld.models.worldsign.pose_branch import (
    ArticulatorLinear,
    KeypointDecoders,
    PoseBranch,
    PoseEncoder,
    anchor_loss,
    articulator_confidence,
    keypoint_variance,
)
from tests.worldsign import POSE_SHAPE, pose_tokens, tiny_teacher


def _settings(checkpoint: Path, **changes: object) -> PoseEncoderSettings:
    shape = {"width": POSE_SHAPE.width, "depth": POSE_SHAPE.depth, "heads": POSE_SHAPE.heads}
    return PoseEncoderSettings(checkpoint=checkpoint, **shape, **changes)  # type: ignore[arg-type]


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


def test_at_step_0_the_target_is_s_jepa_ema_encoding(tmp_path: Path) -> None:
    checkpoint, teacher = tiny_teacher(tmp_path)
    tokens = pose_tokens(2)

    encoder = PoseEncoder.from_checkpoint(_settings(checkpoint))

    with torch.no_grad():
        assert torch.allclose(encoder(tokens), teacher.encode(tokens), atol=1e-5)
    assert encoder(tokens).shape == (2, 32, 4, POSE_SHAPE.width)


def test_the_adapted_encoder_trains_lora_and_final_layer_only(tmp_path: Path) -> None:
    checkpoint, _ = tiny_teacher(tmp_path)
    encoder = PoseEncoder.from_checkpoint(_settings(checkpoint))

    encoder(pose_tokens(2)).pow(2).mean().backward()

    trainable = {id(p) for p in encoder.parameters() if p.requires_grad}
    adapters = {id(p) for p in adapter_parameters(encoder)}
    assert encoder.final is not None
    assert trainable == adapters | {id(p) for p in encoder.final.parameters()}
    assert sum(isinstance(m, LoRAWeight) for m in encoder.modules()) == 4 * POSE_SHAPE.depth
    assert all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert encoder.embed.project.weight.grad is None


def test_esp3_freezes_everything_and_adds_nothing(tmp_path: Path) -> None:
    checkpoint, teacher = tiny_teacher(tmp_path)
    tokens = pose_tokens(1)

    encoder = PoseEncoder.from_checkpoint(_settings(checkpoint, trainable=False))

    assert not any(p.requires_grad for p in encoder.parameters())
    assert encoder.final is None and not adapter_parameters(encoder)
    assert torch.allclose(encoder(tokens), teacher.encode(tokens), atol=1e-6)


def test_the_final_layer_is_one_identity_per_articulator() -> None:
    layer = ArticulatorLinear(3)
    x = torch.randn(2, 5, 4, 3)

    assert torch.equal(layer(x), x)
    with torch.no_grad():
        layer.weight[1].mul_(2.0)
    assert torch.equal(layer(x)[:, :, 1], 2.0 * x[:, :, 1])
    assert torch.equal(layer(x)[:, :, 0], x[:, :, 0])


def test_decoders_write_each_articulator_on_its_own_joints() -> None:
    decoders = KeypointDecoders(4)
    with torch.no_grad():
        for part, head in enumerate(decoders.heads):
            head.weight.zero_()
            head.bias.fill_(float(part))

    joints = decoders(torch.randn(1, 2, 4, 4))

    assert joints.shape == (1, 2, len(JOINTS), 2)
    assert torch.equal(joints[0, 0, :, 0], torch.as_tensor(JOINT_ARTICULATOR).float())


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


def test_articulator_confidence_averages_each_articulator_joints() -> None:
    weights = torch.ones(1, 1, len(JOINTS))
    weights[..., torch.as_tensor(JOINT_ARTICULATOR) == 1] = 0.0  # left hand missing

    assert articulator_confidence(weights)[0, 0].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_the_anchor_needs_the_keypoint_scale_first(tmp_path: Path) -> None:
    checkpoint, _ = tiny_teacher(tmp_path)
    branch = PoseBranch.from_settings(_settings(checkpoint))
    latent = branch.targets(pose_tokens(1))
    keypoints, weights = torch.randn(1, 32, len(JOINTS), 2), torch.ones(1, 32, len(JOINTS))

    with pytest.raises(RuntimeError, match="keypoint variance"):
        branch.anchor(latent, keypoints, weights)
    branch.fit(keypoints, weights)
    assert torch.isfinite(branch.anchor(latent, keypoints, weights))
