import numpy as np
import pyarrow as pa
import pytest

from signworld.acquisition.sharding import Shard
from signworld.corpus.materialize import (
    MaterializedIndex,
    MaterializedRecord,
    crop_square,
    sample_cuts,
    signer_box,
)


def test_signer_box_is_a_square_around_the_largest_person_with_margin() -> None:
    frames = [
        np.array([[10, 20, 30, 40], [100, 100, 200, 300]], np.float32),
        np.array([[110, 90, 190, 310]], np.float32),
    ]

    signer = signer_box(frames, margin=0.1)

    assert signer is not None
    assert signer.union.tolist() == [100, 90, 200, 310]
    box = signer.crop
    width, height = box[2] - box[0], box[3] - box[1]
    assert width == pytest.approx(height)
    assert width == pytest.approx(220 * 1.1)
    assert (box[:2] + box[2:]).tolist() == pytest.approx([300.0, 400.0])


def test_no_person_gives_no_box() -> None:
    assert signer_box([np.zeros((0, 4), np.float32)], margin=0.15) is None


def test_crop_pads_outside_the_frame_with_black() -> None:
    frames = np.full((2, 100, 100, 3), 200, np.uint8)

    crops = crop_square(frames, np.array([-50.0, 0.0, 50.0, 100.0]), size=20)

    assert crops.shape == (2, 20, 20, 3)
    assert crops[:, :, :8].max() == 0
    assert crops[:, :, 12:].min() == 200


def test_crop_stays_square_when_the_box_straddles_half_pixels() -> None:
    frames = np.full((2, 400, 600, 3), 200, np.uint8)
    # Rounding each corner on its own would give 576 columns and 575 rows.
    box = np.array([-277.5, -187.25, 297.5, 387.75])

    crops = crop_square(frames, box, size=32)

    assert crops.shape == (2, 32, 32, 3)


def test_sample_takes_channels_in_turn_within_duration_bounds() -> None:
    manifest = pa.table(
        {
            "clip_id": [f"c{i}" for i in range(9)],
            "video_id": ["v"] * 9,
            "start_s": [0.0] * 9,
            "end_s": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 30.0, 2.0, 0.5],
            "channel_id": ["A", "A", "A", "A", "A", "B", "B", "C", "C"],
        }
    )

    cuts = sample_cuts(manifest, count=3, min_duration_s=1.0, max_duration_s=20.0, seed=0)

    assert sorted(cut.clip_id for cut in cuts)[1:] == ["c5", "c7"]
    assert len({cut.clip_id for cut in cuts}) == 3


def test_index_appends_and_reads_back(tmp_path) -> None:  # type: ignore[no-untyped-def]
    index = MaterializedIndex(tmp_path / "root")
    record = MaterializedRecord("c1", "v.mp4", "p.npz", 25.0, 10, 50, [0.0, 0.0, 1.0, 1.0])

    index.append(record)

    assert index.records() == [record]


def test_union_box_maps_into_crop_pixels() -> None:
    signer = signer_box([np.array([[100, 100, 200, 300]], np.float32)], margin=0.0)

    assert signer is not None
    assert signer.union_in_crop(size=100).tolist() == pytest.approx([25.0, 0.0, 75.0, 100.0])


def test_sharded_indexes_are_read_together(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first = MaterializedIndex(tmp_path, Shard(0, 2))
    second = MaterializedIndex(tmp_path, Shard(1, 2))
    record = MaterializedRecord("c1", "v.mp4", "p.npz", 25.0, 10, 50, [0.0, 0.0, 1.0, 1.0])

    first.append(record)
    second.append(MaterializedRecord("c2", "v.mp4", "p.npz", 25.0, 10, 50, [0.0, 0.0, 1.0, 1.0]))

    assert {r.clip_id for r in MaterializedIndex(tmp_path).records()} == {"c1", "c2"}
    assert first.path != second.path
