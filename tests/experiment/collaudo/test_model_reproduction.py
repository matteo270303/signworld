"""The weight-reproduction checks pass on a faithful path and catch a wrong one."""

from pathlib import Path

import numpy as np
import torch

from signworld.checks.model_reproduction import (
    Agreement,
    official_input,
    pose_reproduction,
    video_reproduction,
)
from signworld.models.video_encoders import model_input

from ..models.test_video_encoders import tiny_encoder


def test_official_input_matches_ours() -> None:
    frames = np.random.default_rng(0).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)

    ours = model_input(torch.from_numpy(frames[None]))

    assert torch.allclose(ours, official_input(frames), atol=1e-5)


def test_video_reproduction_passes_and_its_channel_control_fails() -> None:
    rng = np.random.default_rng(0)
    clips = [rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8) for _ in range(3)]

    report = video_reproduction(tiny_encoder(), clips, {})

    assert report.fp32.reproduces
    assert not report.swapped_channels.reproduces
    assert report.deterministic
    assert report.passed


def test_agreement_flags_a_perturbed_output() -> None:
    reference = torch.randn(2, 10, 8)

    assert Agreement.of(reference, reference).reproduces
    assert not Agreement.of(reference + 0.1, reference).reproduces


def test_pose_reproduction_writes_then_compares(tmp_path: Path) -> None:
    torch.manual_seed(0)
    weights = torch.randn(6, 4)
    tokens = torch.randn(3, 32, 69, 6)
    reference = tmp_path / "reference.npz"

    first = pose_reproduction(lambda t: t @ weights, tokens, reference)
    second = pose_reproduction(lambda t: t @ weights, tokens, reference)
    drifted = pose_reproduction(lambda t: t @ (weights + 0.1), tokens, reference)

    assert first.created and first.agreement is None
    assert not second.created and second.passed
    assert not drifted.passed
