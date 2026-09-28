import numpy as np
import pytest

from signworld.pose.boxes import articulator_boxes
from signworld.pose.frames import local_motion, select_frames, uniform_indices
from signworld.pose.wholebody import (
    ARTICULATOR_INDICES,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    Articulator,
    PoseTrack,
    shoulder_frame,
    unisign_parts,
)


def _track(keypoints: np.ndarray, scores: np.ndarray) -> PoseTrack:
    frames = len(keypoints)
    return PoseTrack(keypoints, scores, np.arange(frames), frames)


def _standing(frames: int = 4, score: float = 5.0) -> PoseTrack:
    rng = np.random.default_rng(0)
    keypoints = rng.uniform(0.3, 0.7, (frames, 133, 2)).astype(np.float32)
    keypoints[:, LEFT_SHOULDER] = (0.6, 0.5)
    keypoints[:, RIGHT_SHOULDER] = (0.4, 0.5)
    return _track(keypoints, np.full((frames, 133), score, np.float32))


def test_articulators_are_the_69_keypoints_of_uni_sign() -> None:
    sizes = {part: len(indices) for part, indices in ARTICULATOR_INDICES.items()}

    assert sizes == {
        Articulator.BODY: 9,
        Articulator.LEFT_HAND: 21,
        Articulator.RIGHT_HAND: 21,
        Articulator.FACE: 18,
    }
    assert ARTICULATOR_INDICES[Articulator.FACE][-1] == 53


def test_uni_sign_parts_are_root_relative_and_bounded() -> None:
    parts = unisign_parts(_standing())

    for values in parts.values():
        assert values.shape[-1] == 3
        assert np.abs(values[..., :2]).max() <= 1
    assert np.allclose(parts[Articulator.LEFT_HAND][:, 0, :2], 0)
    assert np.allclose(parts[Articulator.FACE][:, -1, :2], 0)


def test_uncertain_keypoints_are_zeroed_and_an_empty_body_zeroes_everything() -> None:
    track = _standing()
    track.scores[:, 91] = 0.1

    assert np.all(unisign_parts(track)[Articulator.LEFT_HAND][:, 0] == 0)
    silent = _standing(score=0.0)
    assert all(np.all(values == 0) for values in unisign_parts(silent).values())


def test_shoulder_frame_centres_between_shoulders_and_carries_over_missing_frames() -> None:
    track = _standing(frames=3)
    track.scores[0, LEFT_SHOULDER] = 0.0
    track.keypoints[2, LEFT_SHOULDER] = (0.8, 0.5)

    reference = shoulder_frame(track)

    assert reference is not None
    assert reference.measured.tolist() == [False, True, True]
    assert reference.scale.tolist() == pytest.approx([0.2, 0.2, 0.4])
    normalised = reference.apply(track.keypoints)
    assert normalised[1, RIGHT_SHOULDER].tolist() == pytest.approx([-0.5, 0.0])


def test_no_reliable_shoulders_gives_no_reference() -> None:
    assert shoulder_frame(_standing(score=0.0)) is None


def test_boxes_enclose_confident_keypoints_with_a_margin() -> None:
    track = _standing(frames=1)
    hand = list(ARTICULATOR_INDICES[Articulator.RIGHT_HAND])
    track.keypoints[0, hand] = np.linspace((0.2, 0.3), (0.4, 0.7), len(hand))

    boxes = articulator_boxes(track, margin=0.1)

    column = list(Articulator).index(Articulator.RIGHT_HAND)
    assert boxes.boxes[0, column].tolist() == pytest.approx([0.18, 0.26, 0.42, 0.74])
    assert not boxes.outside_frame()[0, column]


def test_boxes_are_missing_when_too_few_keypoints_are_confident() -> None:
    track = _standing(frames=1)
    track.scores[0, list(ARTICULATOR_INDICES[Articulator.LEFT_HAND])] = 0.0

    boxes = articulator_boxes(track)

    column = list(Articulator).index(Articulator.LEFT_HAND)
    assert not boxes.visible[0, column]
    assert np.isnan(boxes.boxes[0, column]).all()


def test_short_clips_are_sampled_uniformly_with_repeats() -> None:
    assert select_frames(np.ones(10), count=16) == uniform_indices(10, 16)


def test_selection_is_sorted_complete_and_denser_where_motion_is() -> None:
    motion = np.full(400, 0.01)
    motion[100:140] = 5.0

    selected = select_frames(motion)

    assert len(selected) == 64
    assert selected == sorted(selected)
    assert len(set(selected)) == 64
    assert sum(100 <= index < 140 for index in selected) > 64 * 40 / 400 * 2


def test_local_motion_follows_the_most_active_blocks() -> None:
    frames = np.zeros((3, 96, 96, 3), np.uint8)
    frames[2, :24, :24] = 255

    motion = local_motion(frames)

    assert motion.shape == (3,)
    assert motion[1] == 0
    assert motion[2] > 0
