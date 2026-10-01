"""The pose-teacher comparison end to end, on synthetic clips and a tiny backbone."""

from pathlib import Path

import numpy as np
import pytest
import torch

from signworld.data.pose.frames import FRAMES_PER_CLIP
from signworld.data.pose.tokens import STEPS
from signworld.data.pose.wholebody import LEFT_SHOULDER, RIGHT_SHOULDER, Articulator
from signworld.experiment.collaudo.analysis import PoseTeacherSettings
from signworld.experiment.studies import pose_teachers
from signworld.experiment.studies.pose_teachers import (
    LabelledClip,
    PoseCorpus,
    kinematic_features,
    run,
)

TINY = PoseTeacherSettings(
    width=16, depth=1, heads=2, predictor_depth=1, decoder_depth=1, epochs=1, batch_size=8
)


def _clips(root: Path, videos: int = 24, per_video: int = 2) -> list[LabelledClip]:
    rng = np.random.default_rng(0)
    clips = []
    for video in range(videos):
        for index in range(per_video):
            pose = rng.uniform(0.2, 0.8, (FRAMES_PER_CLIP, 133, 3)).astype(np.float32)
            pose[:, LEFT_SHOULDER, :2] = (0.6, 0.5)
            pose[:, RIGHT_SHOULDER, :2] = (0.4, 0.5)
            pose[..., 2] = 5.0
            path = root / f"v{video}-{index}.npz"
            np.savez(path, pose=pose, idx=np.arange(FRAMES_PER_CLIP), total=FRAMES_PER_CLIP)
            clips.append(
                LabelledClip(f"v{video}-{index}", f"v{video}", f"c{video % 3}", "ase", path)
            )
    return clips


def test_kinematic_features_are_white_on_the_training_clips(tmp_path: Path) -> None:
    corpus = PoseCorpus.load(_clips(tmp_path), TINY.min_score)
    train = np.ones(len(corpus.clips), bool)

    features = kinematic_features(corpus, train)[Articulator.BODY]

    rows = features.reshape(-1, features.shape[-1]).astype(np.float64)
    np.testing.assert_allclose(np.cov(rows, rowvar=False), np.eye(rows.shape[1]), atol=1e-3)


def test_every_candidate_is_trained_and_read_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_unisign(corpus: PoseCorpus, checkpoint: Path) -> dict[Articulator, np.ndarray]:
        rng = np.random.default_rng(1)
        return {part: rng.standard_normal((len(corpus.clips), STEPS, 8)) for part in Articulator}

    monkeypatch.setattr(pose_teachers, "unisign_features", fake_unisign)
    (tmp_path / "poses").mkdir()
    clips = _clips(tmp_path / "poses")

    report = run(clips, tmp_path / "unused.pth", TINY, torch.device("cpu"), tmp_path / "out")

    assert [c.name for c in report.candidates] == ["kinematic", "unisign", "mamp", "sjepa"]
    assert report.clips == len(clips) and report.test_clips > 0
    assert (tmp_path / "out" / "mamp.pt").is_file() and (tmp_path / "out" / "sjepa.pt").is_file()
    for candidate in report.candidates:
        assert set(candidate.articulators) == {str(part) for part in Articulator}
        assert 0.0 <= candidate.channel_accuracy <= 1.0
    assert len(report.candidates[-1].training) == TINY.epochs


def test_saved_teachers_are_reused_instead_of_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_unisign(corpus: PoseCorpus, checkpoint: Path) -> dict[Articulator, np.ndarray]:
        return {part: np.ones((len(corpus.clips), STEPS, 4)) for part in Articulator}

    monkeypatch.setattr(pose_teachers, "unisign_features", fake_unisign)
    (tmp_path / "poses").mkdir()
    clips = _clips(tmp_path / "poses")
    run(clips, tmp_path / "unused.pth", TINY, torch.device("cpu"), tmp_path / "out")

    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("a saved teacher was trained again")

    monkeypatch.setattr(pose_teachers, "train_teacher", no_training)
    report = run(
        clips, tmp_path / "unused.pth", TINY, torch.device("cpu"), tmp_path / "out", reuse=True
    )

    assert [c.training for c in report.candidates[2:]] == [[], []]
    # One language only: it cannot be read on unseen channels.
    assert np.isnan(report.candidates[0].language_accuracy_unseen_channels)
