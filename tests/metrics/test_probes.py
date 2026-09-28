"""Linear read-out machinery on synthetic data with a known answer."""

import numpy as np
import pytest

from signworld.metrics.probes import (
    Probe,
    ProbeTask,
    channel_split,
    fit_ridge,
    label_accuracy,
    predict,
    video_split,
    weighted_r2,
)


def test_weighted_ridge_recovers_a_linear_map_and_ignores_missing_targets() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((2000, 8))
    truth = rng.standard_normal((8, 3, 2))
    targets = np.einsum("nd,dkc->nkc", features, truth)
    weights = np.ones((2000, 3))
    targets[:500, 0] = 99.0
    weights[:500, 0] = 0.0

    coefficients = fit_ridge(features, targets, weights, penalty=1e-6)

    prediction = predict(features, coefficients)
    assert weighted_r2(targets, prediction, weights) > 0.999


def test_r2_is_zero_for_the_mean_and_negative_for_worse() -> None:
    targets = np.arange(20, dtype=float).reshape(10, 1, 2)
    weights = np.ones((10, 1))

    assert weighted_r2(targets, np.broadcast_to(targets.mean(0), targets.shape), weights) == (
        pytest.approx(0.0)
    )
    assert weighted_r2(targets, -targets, weights) < 0


def test_video_split_is_stable_and_keeps_videos_together() -> None:
    videos = np.array([f"video{i % 50}" for i in range(1000)])

    first, second = video_split(videos), video_split(videos)

    assert np.array_equal(first, second)
    for video in np.unique(videos):
        assert len(set(first[videos == video])) == 1
    assert 0.05 < first.mean() < 0.4


def test_probe_task_on_informative_and_random_features() -> None:
    rng = np.random.default_rng(1)
    clips, frames = 120, 8
    targets = rng.standard_normal((clips, frames, 4, 2))
    informative = np.concatenate(
        [targets.reshape(clips, frames, 8), rng.standard_normal((clips, frames, 8))], -1
    )
    noise = rng.standard_normal((clips, frames, 16))
    weights = np.ones((clips, frames, 4))
    videos = np.array([f"v{i}" for i in range(clips)])
    test = video_split(videos)

    good = ProbeTask.of_frames(informative, targets, weights, test, videos).held_out_r2()
    bad = ProbeTask.of_frames(noise, targets, weights, test, videos).held_out_r2()

    assert good > 0.99
    assert bad < 0.1
    assert isinstance(ProbeTask.of_frames(noise, targets, weights, test, videos).fit(), Probe)


def test_velocity_task_pairs_consecutive_frames() -> None:
    rng = np.random.default_rng(2)
    targets = rng.standard_normal((60, 6, 2, 2))
    features = targets.reshape(60, 6, 4)
    videos = np.array([f"v{i}" for i in range(60)])

    task = ProbeTask.of_velocities(
        features, targets, np.ones((60, 6, 2)), video_split(videos), videos
    )

    assert task.features.shape == (60 * 5, 8)
    assert task.held_out_r2() > 0.99


def test_label_accuracy_reads_separable_labels_and_reports_the_majority_share() -> None:
    rng = np.random.default_rng(0)
    labels = np.array(["a"] * 300 + ["b"] * 100)
    features = rng.standard_normal((400, 4))
    features[labels == "b", 0] += 6.0
    test = rng.uniform(size=400) < 0.3

    accuracy, majority = label_accuracy(features, labels, test)

    assert accuracy > 0.95
    assert majority == pytest.approx((labels[test] == "a").mean())


def test_label_accuracy_is_undefined_with_a_single_class() -> None:
    labels = np.array(["a"] * 10)

    accuracy, _ = label_accuracy(np.ones((10, 2)), labels, np.arange(10) < 3)

    assert np.isnan(accuracy)


def test_channel_split_keeps_every_label_on_both_sides_when_it_can() -> None:
    channels = np.array(["a1", "a1", "a2", "a3", "a4", "a5", "b1", "b2", "c1"])
    labels = np.array(["ase", "ase", "ase", "ase", "ase", "ase", "bfi", "bfi", "gsg"])

    test = channel_split(channels, labels)

    for label in ("ase", "bfi"):
        own = labels == label
        assert test[own].any() and (~test[own]).any()
        assert not set(channels[own & test]) & set(channels[own & ~test])
    assert not test[labels == "gsg"].any()


def test_a_label_on_a_single_channel_leaves_the_probe_undefined() -> None:
    # As in the YouTube-SL-25 test clips: ASL from one channel, the other language from several.
    rng = np.random.default_rng(0)
    channels = np.array(["asl"] * 200 + [f"aed{index % 4}" for index in range(200)])
    labels = np.array(["ase"] * 200 + ["aed"] * 200)
    features = rng.standard_normal((400, 3))

    accuracy, chance = label_accuracy(features, labels, channel_split(channels, labels))

    assert np.isnan(accuracy) and np.isnan(chance)
