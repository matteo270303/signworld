"""Pose sequences as the teachers see them."""

import numpy as np
import pytest

from signworld.pose.frames import FRAMES_PER_CLIP
from signworld.pose.tokens import JOINTS, STEPS, PoseSequence, articulator_columns
from signworld.pose.wholebody import LEFT_SHOULDER, RIGHT_SHOULDER, Articulator, PoseTrack


def _track(score: float = 5.0) -> PoseTrack:
    rng = np.random.default_rng(0)
    keypoints = rng.uniform(0.3, 0.7, (FRAMES_PER_CLIP, 133, 2)).astype(np.float32)
    keypoints[:, LEFT_SHOULDER] = (0.6, 0.5)
    keypoints[:, RIGHT_SHOULDER] = (0.4, 0.5)
    scores = np.full((FRAMES_PER_CLIP, 133), score, np.float32)
    return PoseTrack(keypoints, scores, np.arange(FRAMES_PER_CLIP), FRAMES_PER_CLIP)


def test_tokens_hold_both_frames_of_a_step_in_shoulder_units() -> None:
    track = _track()
    sequence = PoseSequence.from_track(track)
    assert sequence is not None

    tokens = sequence.tokens()

    assert tokens.shape == (STEPS, len(JOINTS), 6)
    shoulder = JOINTS.index(LEFT_SHOULDER)
    # Origin between the shoulders, unit their distance: the left shoulder sits half a unit away.
    assert np.linalg.norm(tokens[3, shoulder, :2]) == pytest.approx(0.5)
    assert tokens[..., 2].all() and tokens[..., 5].all()


def test_unreliable_or_outside_keypoints_are_zero_and_flagged_missing() -> None:
    track = _track()
    hand = articulator_columns(Articulator.RIGHT_HAND)[0]
    track.scores[0, JOINTS[hand]] = 0.5
    track.keypoints[1, JOINTS[hand]] = (1.2, 0.5)

    sequence = PoseSequence.from_track(track)
    assert sequence is not None

    assert not sequence.present[0, hand] and not sequence.present[1, hand]
    assert (sequence.joints[:2, hand] == 0).all()
    _, weights = sequence.step_positions()
    assert weights[0, hand] == 0 and weights[1, hand] == 1


def test_step_position_averages_only_the_present_frames() -> None:
    track = _track()
    hand = articulator_columns(Articulator.LEFT_HAND)[0]
    track.scores[2, JOINTS[hand]] = 0.0
    sequence = PoseSequence.from_track(track)
    assert sequence is not None

    positions, _ = sequence.step_positions()

    np.testing.assert_allclose(positions[1, hand], sequence.joints[3, hand], rtol=1e-6)


def test_keypoints_far_from_the_signer_are_dropped() -> None:
    track = _track()
    hand = articulator_columns(Articulator.RIGHT_HAND)[0]
    # Inside the frame and confident, but nine shoulder widths away: a wrong detection.
    track.keypoints[5, JOINTS[hand]] = (0.4 + 9 * 0.2, 0.5)

    sequence = PoseSequence.from_track(track)
    assert sequence is not None

    assert not sequence.present[5, hand]
    assert (sequence.joints[5, hand] == 0).all()


def test_no_sequence_without_shoulders() -> None:
    track = _track()
    track.scores[:, LEFT_SHOULDER] = 0.0

    assert PoseSequence.from_track(track) is None


def test_articulator_columns_partition_the_joints() -> None:
    columns = np.concatenate([articulator_columns(part) for part in Articulator])

    assert sorted(columns.tolist()) == list(range(len(JOINTS)))
