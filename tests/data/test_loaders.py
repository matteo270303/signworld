"""The training index, the dataset, the sampler and the batches, on materialised clips."""

from pathlib import Path

import numpy as np
import pytest
import torch

from signworld.data.augmentation import ClipAugmenter, View
from signworld.data.loaders import (
    INDEX_SCHEMA,
    TRAIN,
    VALIDATION_CHANNEL,
    VALIDATION_LANGUAGE,
    ClipDataset,
    Collate,
    EpochSampler,
    build_training_index,
    caption_statistics,
    keypoint_statistics,
    read_index,
    validation_subset,
    write_index,
)
from signworld.experiment.train.config import AugmentationSettings, DataSettings
from tests.worldsign import synthetic_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return synthetic_corpus(tmp_path_factory.mktemp("corpus"))


def _index(corpus, **settings: object):  # type: ignore[no-untyped-def]
    return build_training_index(
        corpus.manifest,
        corpus.records,
        corpus.caption_rows,
        DataSettings(**settings),  # type: ignore[arg-type]
    )


def test_splits_hold_out_whole_channels_and_languages(corpus, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    table = _index(corpus, validation_fraction=0.5, held_out_languages=("lsf",))
    write_index(table, tmp_path / "index.parquet")
    rows = read_index(tmp_path / "index.parquet").to_pylist()

    assert table.schema.equals(INDEX_SCHEMA) and len(rows) == 12
    assert {r["split"] for r in rows if r["sign_language"] == "lsf"} == {VALIDATION_LANGUAGE}
    by_channel: dict[str, set[str]] = {}
    for row in rows:
        by_channel.setdefault(row["channel_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_channel.values())  # never split inside
    assert {TRAIN, VALIDATION_CHANNEL} <= {r["split"] for r in rows}
    assert len(read_index(tmp_path / "index.parquet", TRAIN)) == sum(
        r["split"] == TRAIN for r in rows
    )


def test_a_clip_without_caption_or_materialisation_is_left_out(corpus) -> None:  # type: ignore[no-untyped-def]
    partial = type(corpus)(
        corpus.manifest, corpus.records[1:], corpus.caption_rows.slice(0, 11), corpus.embeddings
    )

    table = _index(partial)

    assert set(table.column("clip_id").to_pylist()) == {f"clip{i:02d}" for i in range(1, 11)}


def test_a_sample_has_every_input_of_the_model(corpus) -> None:  # type: ignore[no-untyped-def]
    dataset = ClipDataset(_index(corpus), corpus.embeddings, augmentation=None, box_threshold=0.3)

    sample = dataset[3]

    assert sample.frames.shape == (64, 64, 64, 3) and sample.frames.dtype == torch.uint8
    assert sample.pose_tokens.shape == (32, 69, 6)
    assert sample.keypoints.shape == (32, 69, 2) and sample.keypoint_weights.shape == (32, 69)
    assert sample.boxes.shape == (32, 4, 4) and bool(sample.box_visible.all())
    assert torch.equal(sample.caption, torch.from_numpy(corpus.embeddings[1]))
    band, below = sample.frames[:, 20:36].float().mean(), sample.frames[:, 40:].float().mean()
    assert band > 40 and below < 5  # the square is in the rows where it was drawn


def test_augmentation_moves_frames_and_boxes_together_and_repeats_by_epoch(corpus) -> None:  # type: ignore[no-untyped-def]
    settings = AugmentationSettings(box_jitter=0.1, brightness=0.0, contrast=0.0, saturation=0.0)
    augmented = ClipDataset(
        _index(corpus), corpus.embeddings, augmentation=settings, box_threshold=0.3, seed=1
    )
    plain = ClipDataset(_index(corpus), corpus.embeddings, augmentation=None, box_threshold=0.3)

    first, again, other = augmented[(2, 0)], augmented[(2, 0)], augmented[(2, 1)]
    reference = plain[2]

    assert first.view is not None and torch.equal(first.view, again.view)  # type: ignore[arg-type]
    assert torch.equal(first.frames, reference.frames)  # the frames move later, on the GPU
    batch = Collate(["en", "es"])([first, other]).augmented()
    moved = ClipAugmenter.frames(first.frames, View.from_tensor(first.view))
    assert torch.equal(batch.frames[0], moved) and batch.views is None
    assert torch.equal(first.boxes, again.boxes)
    assert not torch.equal(first.boxes, other.boxes)
    assert not torch.equal(first.boxes, reference.boxes)
    both = (first.keypoint_weights > 0) & (reference.keypoint_weights > 0)
    # Shoulder units ignore where the crop is and how large: the pose does not change.
    assert torch.allclose(first.keypoints[both], reference.keypoints[both], atol=1e-3)


def test_the_sampler_splits_an_epoch_between_gpus_and_resumes_in_order() -> None:
    samplers = [EpochSampler(10, rank, 2, seed=3) for rank in range(2)]
    for sampler in samplers:
        sampler.configure(epoch=1)
    shares = [[index for index, _ in sampler] for sampler in samplers]

    assert not set(shares[0]) & set(shares[1]) and len(shares[0]) == len(shares[1]) == 5
    samplers[0].configure(epoch=1, start=2)
    assert [index for index, _ in samplers[0]] == shares[0][2:]
    samplers[0].configure(epoch=2)
    assert [index for index, _ in samplers[0]] != shares[0]


def test_batches_number_languages_as_the_centring(corpus) -> None:  # type: ignore[no-untyped-def]
    dataset = ClipDataset(_index(corpus), corpus.embeddings, augmentation=None, box_threshold=0.3)

    batch = Collate(["en", "es"])([dataset[0], dataset[1]])

    assert batch.frames.shape == (2, 64, 64, 64, 3)
    assert batch.languages.tolist() == [1, 0]  # clip 0 is Spanish, clip 1 English
    assert batch.caption_rows is not None and batch.caption_rows.tolist() == [0, 0]
    assert batch.videos == ["video0", "video0"]


def test_statistics_come_from_distinct_captions_and_sampled_poses(corpus) -> None:  # type: ignore[no-untyped-def]
    table = _index(corpus)

    captions, languages = caption_statistics(table, corpus.embeddings)
    positions, weights = keypoint_statistics(table, clips=5)

    assert captions.shape == (6, 768) and len(languages) == 6
    assert positions.shape == (5, 32, 69, 2) and weights.shape == (5, 32, 69)


def test_the_validation_subset_is_fixed() -> None:
    import pyarrow as pa  # noqa: PLC0415

    table = pa.table({"clip_id": ["c", "a", "d", "b"], "value": [0, 1, 2, 3]})

    assert validation_subset(table, 2).column("clip_id").to_pylist() == ["a", "b"]
    assert np.array_equal(
        validation_subset(table.take([3, 2, 1, 0]), 2).column("value").to_numpy(), [3, 1]
    )
