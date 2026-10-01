"""The toy world goes through the real pipeline: index, dataset, tokens and boxes (§4.13.1)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from signworld.corpus.materialize import MaterializedIndex
from signworld.worldmodel.config import DataSettings
from signworld.worldmodel.data import ClipDataset, build_training_index
from signworld.worldmodel.toyworld import (
    COMBINATIONS,
    MOTIONS,
    build,
    caption,
    manifest_table,
    render,
    toy_clips,
)


def test_captions_and_combinations() -> None:
    assert len(COMBINATIONS) == len(MOTIONS) ** 4
    assert caption((0, 1, 2, 4)) == (
        "the blue square falls and bounces, the red circle moves left, "
        "the green triangle moves right, the yellow star stays still"
    )


def test_a_falling_shape_falls_and_a_still_one_stays() -> None:
    clip = toy_clips(1)[0]
    clip = type(clip)(clip.clip_id, (0, 4, 4, 4), clip.seed)
    _, pose = render(clip, size=64)

    body = pose[:, 0, 1]  # nose index: a point of the square, the «body»
    still = pose[:, 91, :2]  # a point of the circle, the «left hand»
    assert body[20] > body[0]  # falls: y grows
    assert np.allclose(still[0], still[-1])


def test_toy_clips_go_through_index_and_dataset(tmp_path: Path) -> None:
    written = build(tmp_path, 6, size=64)
    records = MaterializedIndex(tmp_path).records()
    manifest = manifest_table(tmp_path)
    captions = sorted(set(manifest.column("caption").to_pylist()))
    rows = pa.Table.from_pylist(
        [
            {"clip_id": c, "row": captions.index(t)}
            for c, t in zip(
                manifest.column("clip_id").to_pylist(),
                manifest.column("caption").to_pylist(),
                strict=True,
            )
        ]
    )

    index = build_training_index(manifest, records, rows, DataSettings(validation_fraction=0.5))
    sample = ClipDataset(
        index, np.zeros((len(captions), 768), np.float32), augmentation=None, box_threshold=0.3
    )[0]

    assert written == 6 and index.num_rows == 6
    assert sample.frames.shape == (64, 64, 64, 3)
    assert sample.pose_tokens.shape == (32, 69, 6)
    assert bool(sample.box_visible.all())  # every shape is always on screen
    assert torch.isfinite(sample.keypoints).all()
    assert build(tmp_path, 6, size=64) == 0  # resumes: nothing left to write
