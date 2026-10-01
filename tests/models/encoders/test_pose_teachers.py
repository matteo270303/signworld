"""Pose teachers: masking, objectives, EMA and views, on a tiny backbone."""

import math

import pytest
import torch

from signworld.models.pose_teachers import (
    MaskedMotionTeacher,
    PoseTeacher,
    SJEPATeacher,
    TeacherShape,
    augment_view,
    motion_aware_mask,
    pool_articulators,
    token_motion,
)
from signworld.pose.tokens import JOINTS, STEPS

TINY = TeacherShape(width=16, depth=1, heads=2, predictor_depth=1, decoder_depth=1)
TOKENS = STEPS * len(JOINTS)


def _tokens(batch: int = 3, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randn(batch, STEPS, len(JOINTS), 2, 2, generator=generator)
    present = torch.ones(batch, STEPS, len(JOINTS), 2, 1)
    present[:, :, :5] = 0.0
    return torch.cat([frames * present, present], dim=-1).flatten(-2)


def test_mask_partitions_the_tokens_and_hides_the_requested_share() -> None:
    visible, hidden = motion_aware_mask(_tokens(), 0.9, torch.Generator().manual_seed(0))

    assert hidden.shape[1] == round(0.9 * TOKENS)
    everything = torch.cat([visible, hidden], dim=1).sort(dim=1).values
    assert torch.equal(everything, torch.arange(TOKENS).expand_as(everything))


def test_mask_hides_moving_joints_more_often() -> None:
    tokens = torch.zeros(64, STEPS, len(JOINTS), 6)
    tokens[..., 2] = tokens[..., 5] = 1.0
    tokens[:, :, 0, 3] = torch.arange(STEPS, dtype=torch.float32)[None] * 10  # joint 0 moves
    _, hidden = motion_aware_mask(tokens, 0.5, torch.Generator().manual_seed(0))

    moving = (hidden % len(JOINTS) == 0).float().sum()
    moving_rate = moving / (64 * STEPS)
    still_rate = (hidden.numel() - moving) / (64 * STEPS * (len(JOINTS) - 1))

    assert moving_rate > 0.95 and still_rate < 0.5


def test_motion_is_zero_where_a_frame_is_missing() -> None:
    motion = token_motion(_tokens())

    assert motion.shape == (3, STEPS, len(JOINTS), 4)
    assert (motion[:, :, :5] == 0).all()


def test_pooling_gives_one_latent_per_step_and_articulator() -> None:
    features = torch.randn(2, TOKENS, 16)

    pooled = pool_articulators(features)

    assert pooled.shape == (2, STEPS, 4, 16)
    grid = features.reshape(2, STEPS, len(JOINTS), 16)
    torch.testing.assert_close(pooled[:, :, 0], grid[:, :, :9].mean(dim=2))


@pytest.mark.parametrize("kind", [MaskedMotionTeacher, SJEPATeacher])
def test_losses_are_finite_and_reach_the_encoder(kind: type[PoseTeacher]) -> None:
    torch.manual_seed(0)
    teacher = kind(TINY)

    loss = teacher.loss(_tokens(), torch.Generator().manual_seed(0))
    loss.backward()  # type: ignore[no-untyped-call]

    assert math.isfinite(loss.item())
    gradient = teacher.embed.project.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert teacher.encode(_tokens()).shape == (3, STEPS, 4, TINY.width)


def test_sjepa_targets_follow_the_online_encoder_slowly() -> None:
    torch.manual_seed(0)
    teacher = SJEPATeacher(TINY, momentum=(0.9, 1.0))
    online = next(teacher.encoder.parameters())
    target = next(teacher.target_encoder.parameters())
    assert not target.requires_grad
    with torch.no_grad():
        online.add_(1.0)
    before = target.detach().clone()

    teacher.after_step(0.0)

    torch.testing.assert_close(target, 0.9 * before + 0.1 * online.detach())
    teacher.after_step(1.0)
    torch.testing.assert_close(target, 0.9 * before + 0.1 * online.detach())


def test_sjepa_centre_moves_toward_the_targets() -> None:
    teacher = SJEPATeacher(TINY)
    teacher.loss(_tokens(), torch.Generator().manual_seed(0))

    assert teacher.centre.abs().sum() > 0


def test_views_keep_missing_joints_and_never_mirror() -> None:
    tokens = _tokens(batch=16)
    view = augment_view(tokens, torch.Generator().manual_seed(0))

    frames, moved = tokens.reshape(16, -1, 3), view.reshape(16, -1, 3)
    assert torch.equal(moved[..., 2], frames[..., 2])
    assert (moved[frames[..., 2] == 0][:, :2] == 0).all()
    for source, target in zip(frames, moved, strict=True):
        kept = source[:, 2] > 0
        x, y = source[kept, :2], target[kept, :2]
        x_c, y_c = x - x.mean(0), y - y.mean(0)
        linear = torch.linalg.lstsq(x_c, y_c).solution
        assert torch.linalg.det(linear) > 0
