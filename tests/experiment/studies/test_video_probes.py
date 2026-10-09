"""Read-outs and decisions of PC2, PC3, PC4 and the frame collaudo, on synthetic features."""

from pathlib import Path

import numpy as np

from signworld.data.pose.boxes import step_boxes
from signworld.data.pose.wholebody import LEFT_SHOULDER, PoseTrack
from signworld.experiment.studies.video_probes import (
    ClipFeatures,
    HandScores,
    RunScores,
    TextScores,
    TextTargets,
    baseline_report,
    choose_encoder,
    choose_resolution,
    text_scores,
)
from signworld.metrics.probes import fit_ridge, ridge_path
from signworld.metrics.retrieval import Interval


def _targets(clips: int, dimension: int = 32, seed: int = 0) -> TextTargets:
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((clips, dimension))
    return TextTargets.of(np.arange(clips), table, np.array(["en"] * clips), np.ones(clips, bool))


def _videos(clips: int) -> np.ndarray:
    return np.array([f"video{i // 2}" for i in range(clips)])


def test_text_readout_retrieves_what_the_features_encode() -> None:
    clips = 400
    targets = _targets(clips)
    rng = np.random.default_rng(1)
    informative = targets.embeddings @ rng.standard_normal((32, 64))
    informative += 0.01 * rng.standard_normal(informative.shape)

    good = text_scores(informative, targets, _videos(clips))
    noise = text_scores(rng.standard_normal((clips, 64)), targets, _videos(clips))

    assert good.r1.t2v > 0.9 and good.r1.v2t > 0.9
    assert noise.r1.t2v < 0.2 and noise.r1.v2t < 0.2
    assert good.t2v_r1.low <= good.t2v_r1.estimate <= good.t2v_r1.high


def test_text_targets_are_centred_on_training_clips_per_language() -> None:
    rng = np.random.default_rng(0)
    table = rng.standard_normal((6, 4)) + 5.0
    languages = np.array(["en", "en", "en", "es", "es", "es"])
    train = np.array([True, True, False, True, True, False])

    targets = TextTargets.of(np.arange(6), table, languages, train)

    assert targets.embeddings.shape == table.shape  # the whole vector, never truncated

    for language in ("en", "es"):
        fitted = (languages == language) & train
        np.testing.assert_allclose(targets.embeddings[fitted].mean(axis=0), 0.0, atol=1e-12)


def test_pc2_floor_is_the_best_baseline() -> None:
    clips = 200
    targets = _targets(clips)
    frozen = _text(0.3)

    report = baseline_report(np.linspace(1, 10, clips), targets, _videos(clips), {"f": frozen})

    assert report.floor_r1.t2v >= 0.3 and report.floor_r1.v2t >= 0.3
    assert 0 < report.chance_r1 < 0.1


def _text(r1: float) -> TextScores:
    return TextScores(Interval(r1, r1, r1), Interval(r1, r1, r1), {}, 1.0, 100)


def _run(name: str, hand: float, r1: float) -> RunScores:
    hands = {"left": HandScores(hand, 0.0), "right": HandScores(hand, 0.0)}
    return RunScores(name, 100, hands, _text(r1))


def test_pc3_needs_both_remaining_probes_otherwise_the_default() -> None:
    both = choose_encoder([_run("a", 0.8, 0.2), _run("b", 0.7, 0.1)], default="b")
    split = choose_encoder([_run("a", 0.8, 0.1), _run("b", 0.7, 0.2)], default="b")

    assert both.chosen == "a"
    assert split.chosen == "b"


def test_pc4_takes_384_only_above_the_gain() -> None:
    low = _run("256", 0.70, 0.1)

    assert choose_resolution(low, _run("384", 0.76, 0.1), 256, 384, 0.05).chosen == 384
    assert choose_resolution(low, _run("384", 0.74, 0.1), 256, 384, 0.05).chosen == 256


def test_clip_features_round_trip_through_shards(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for shard, ids in enumerate((["a", "b"], ["c"])):
        ClipFeatures(
            ids,
            rng.standard_normal((len(ids), 8)),
            rng.standard_normal((len(ids), 32, 4, 8)),
            np.ones((len(ids), 32, 4), bool),
        ).save(tmp_path / f"shard-{shard:05d}-of-00002.npz")

    loaded = ClipFeatures.load(tmp_path)

    assert loaded.clip_ids == ["a", "b", "c"]
    assert loaded.select(["c", "a"]).parts.shape == (2, 32, 4, 8)


def test_step_boxes_join_the_two_frames_of_a_step() -> None:
    keypoints = np.full((64, 133, 2), 0.5, dtype=np.float32)
    scores = np.zeros((64, 133), dtype=np.float32)
    right_hand = list(range(112, 133))
    keypoints[0, right_hand] = np.linspace(0.1, 0.2, 21)[:, None]
    keypoints[1, right_hand] = np.linspace(0.3, 0.4, 21)[:, None]
    scores[:2, right_hand] = 5.0
    scores[:, LEFT_SHOULDER] = 5.0
    track = PoseTrack(keypoints, scores, np.arange(64), 64)

    boxes, visible = step_boxes(track)

    assert boxes.shape == (32, 4, 4)
    assert visible[0, 2] and not visible[1, 2]
    assert boxes[0, 2, 0] < 0.1 and boxes[0, 2, 2] > 0.4


def test_ridge_path_matches_one_fit_per_penalty() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((300, 5))
    targets = rng.standard_normal((300, 4, 2))
    weights = np.ones((300, 4))
    weights[:50, 3] = 0.0  # one keypoint with its own presence pattern

    path = ridge_path(features, targets, weights, (0.1, 1.0))

    for penalty, coefficients in path.items():
        naive = np.zeros_like(coefficients)
        design = np.concatenate([features, np.ones((300, 1))], axis=1)
        identity = np.eye(6)
        identity[-1, -1] = 0.0
        for k in range(4):
            weighted = design * weights[:, k, None]
            gram = design.T @ weighted + penalty * 300 * identity
            naive[:, k] = np.linalg.solve(gram, weighted.T @ targets[:, k])
        np.testing.assert_allclose(coefficients, naive, atol=1e-10)
        np.testing.assert_allclose(fit_ridge(features, targets, weights, penalty), naive)


def test_split_scores_fit_on_training_and_read_every_held_out_split() -> None:
    from signworld.experiment.studies.video_probes import split_text_scores  # noqa: PLC0415

    rng = np.random.default_rng(2)
    mixing = rng.standard_normal((32, 64))

    def split(clips: int, seed: int) -> tuple[np.ndarray, TextTargets]:
        targets = _targets(clips, seed=seed)
        features = targets.embeddings @ mixing + 0.01 * rng.standard_normal((clips, 64))
        return features, targets

    fitting = split(400, 0)
    held = {"val_channel": split(100, 1), "test_channel": split(150, 2)}
    noise = {"val_channel": (rng.standard_normal((100, 64)), held["val_channel"][1])}

    scores = split_text_scores(*fitting, held, choose_on="val_channel")
    floor = split_text_scores(*fitting, noise, choose_on="val_channel")

    assert set(scores) == {"val_channel", "test_channel"}
    assert scores["test_channel"].gallery == 150
    assert scores["test_channel"].r1.t2v > 0.9 and scores["test_channel"].r1.v2t > 0.9
    assert floor["val_channel"].r1.t2v < 0.2
