"""A4 judges each articulator against its own limit."""

from pathlib import Path

import numpy as np

from signworld.data.pose.wholebody import (
    ARTICULATOR_INDICES,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    Articulator,
)
from signworld.experiment.collaudo.pose_quality import pose_quality_report

FRAMES = 8
LEFT_ELBOW, RIGHT_ELBOW = 7, 8


def _pose(tmp_path: Path, name: str, move: dict[tuple[int, ...], tuple[float, float]]) -> Path:
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.35, 0.65, (FRAMES, 133, 3)).astype(np.float32)
    pose[:, LEFT_SHOULDER, :2] = (0.6, 0.5)
    pose[:, RIGHT_SHOULDER, :2] = (0.4, 0.5)
    pose[..., 2] = 5.0
    for indices, (x, y) in move.items():
        pose[:, list(indices), 0] = x
        pose[:, list(indices), 1] = y
    path = tmp_path / f"{name}.npz"
    np.savez(path, pose=pose, idx=np.arange(FRAMES), total=FRAMES)
    return path


def test_a_body_cut_by_the_frame_is_reported_but_does_not_fail(tmp_path: Path) -> None:
    half_body = _pose(tmp_path, "half", {(LEFT_ELBOW, RIGHT_ELBOW): (0.5, 1.1)})

    report = pose_quality_report([half_body], workers=1)

    assert report.outside_fraction["body"] == 1.0
    assert report.passed, report.failures


def test_hands_beyond_the_frame_fail_with_a_reason(tmp_path: Path) -> None:
    left_hand = ARTICULATOR_INDICES[Articulator.LEFT_HAND]
    cut = _pose(tmp_path, "cut", {left_hand: (1.05, 0.5)})

    report = pose_quality_report([cut], workers=1)

    assert not report.passed
    assert any(failure.startswith("left boxes outside") for failure in report.failures)


def test_limits_can_be_tightened_per_articulator(tmp_path: Path) -> None:
    half_body = _pose(tmp_path, "half", {(LEFT_ELBOW, RIGHT_ELBOW): (0.5, 1.1)})

    report = pose_quality_report([half_body], workers=1, outside_limits={"body": 0.01})

    assert not report.passed
