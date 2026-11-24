"""The batched augmentation of the training step equals the clip-by-clip one, bit for bit."""

import pytest
import torch

from signworld.data.augmentation import ClipAugmenter, View, apply_views
from signworld.experiment.train.config import AugmentationSettings

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.mark.parametrize("device", DEVICES)
def test_apply_views_is_clip_by_clip_frames(device: str) -> None:
    augmenter = ClipAugmenter(AugmentationSettings())
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (3, 5, 32, 48, 3), dtype=torch.uint8, generator=generator)
    drawn = [augmenter.sample(generator) for _ in range(len(frames) - 1)] + [View.identity()]
    views = torch.stack([view.as_tensor() for view in drawn])
    frames = frames.to(device)

    expected = torch.stack(
        [
            ClipAugmenter.frames(clip, View.from_tensor(view))
            for clip, view in zip(frames, views, strict=True)
        ]
    )
    batched = apply_views(frames, views.to(device))

    assert batched.dtype == torch.uint8 and batched.shape == frames.shape
    assert torch.equal(batched, expected)
