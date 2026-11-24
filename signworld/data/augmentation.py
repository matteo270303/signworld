"""Views of a clip (§3.8): the same geometry on video and keypoints, colour on video only.

The box jitter moves and rescales the crop by up to ``box_jitter`` of its side; the identical
affine map is applied to the frames and to every keypoint or box, so the physical target keeps
describing what the video shows (P12). Horizontal flips are never produced: they would swap
the dominant hand.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from signworld.experiment.train.config import AugmentationSettings


@dataclass(frozen=True, slots=True)
class View:
    """One draw of the augmentation for one clip."""

    scale: float
    """Zoom of the crop: > 1 shows a smaller region, larger."""
    shift: tuple[float, float]
    """Displacement of the crop's centre, in fractions of its side (x, y)."""
    brightness: float
    contrast: float
    saturation: float

    @classmethod
    def identity(cls) -> "View":
        return cls(1.0, (0.0, 0.0), 1.0, 1.0, 1.0)

    def as_tensor(self) -> Tensor:
        """(6,) scale, shift x, shift y, brightness, contrast, saturation."""
        return torch.tensor(
            [self.scale, *self.shift, self.brightness, self.contrast, self.saturation]
        )

    @classmethod
    def from_tensor(cls, values: Tensor) -> "View":
        scale, x, y, brightness, contrast, saturation = values.tolist()
        return cls(scale, (x, y), brightness, contrast, saturation)


class ClipAugmenter:
    def __init__(self, settings: AugmentationSettings) -> None:
        self.settings = settings

    def sample(self, generator: torch.Generator) -> View:
        s = self.settings
        u = (torch.rand(6, generator=generator) * 2 - 1).tolist()
        return View(
            scale=1.0 + s.box_jitter * u[0],
            shift=(s.box_jitter * u[1], s.box_jitter * u[2]),
            brightness=1.0 + s.brightness * u[3],
            contrast=1.0 + s.contrast * u[4],
            saturation=1.0 + s.saturation * u[5],
        )

    @staticmethod
    def points(points: Tensor, view: View) -> Tensor:
        """Keypoints or box corners (..., 2) in frame fractions, moved as the frames are."""
        centre = torch.tensor([0.5 + view.shift[0], 0.5 + view.shift[1]], device=points.device)
        return (points - centre) * view.scale + 0.5

    @staticmethod
    def frames(frames: Tensor, view: View) -> Tensor:
        """(T, H, W, 3) uint8 frames resampled by the view's crop and recoloured."""
        video = frames.permute(0, 3, 1, 2).float() / 255.0
        # affine_grid maps output coordinates in [-1, 1] to input ones: the inverse of points().
        theta = torch.zeros(1, 2, 3, device=video.device)
        theta[0, 0, 0] = theta[0, 1, 1] = 1.0 / view.scale
        theta[0, 0, 2] = 2.0 * view.shift[0]
        theta[0, 1, 2] = 2.0 * view.shift[1]
        grid = functional.affine_grid(theta.expand(len(video), -1, -1), list(video.shape), False)
        video = functional.grid_sample(video, grid, align_corners=False, padding_mode="zeros")
        video = _colour(video, view)
        return (video.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)


def apply_views(frames: Tensor, views: Tensor) -> Tensor:
    """(batch, T, H, W, 3) uint8 frames, each clip resampled by its view (batch, 6).

    The same function as ``ClipAugmenter.frames``, run where the frames are (the GPU in
    training): the data loader draws the views and moves keypoints and boxes, which is cheap,
    and leaves the frames, whose resampling costs ~0.4 s per clip on a CPU. The whole batch
    is one resampling and one recolouring, the views read on the device: no clip-by-clip
    loop, no wait for the GPU: 3.5x faster (``scripts/perf/bench_augment.py``). Equal bit for
    bit on small clips (``tests/data/test_augmentation.py``); at 64 x 256² the per-frame mean
    may sum in another order, and ~1 byte in 3 million lands one grey level away.
    """
    batch, steps = frames.shape[:2]
    video = frames.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0
    per_frame = views.to(video.device).repeat_interleave(steps, dim=0)  # (batch·T, 6)
    # theta as ClipAugmenter.frames builds it: 1/scale in double precision, then float.
    theta = torch.zeros(len(video), 2, 3, device=video.device)
    theta[:, 0, 0] = theta[:, 1, 1] = (1.0 / per_frame[:, 0].double()).float()
    theta[:, 0, 2] = 2.0 * per_frame[:, 1]
    theta[:, 1, 2] = 2.0 * per_frame[:, 2]
    grid = functional.affine_grid(theta, list(video.shape), False)
    video = functional.grid_sample(video, grid, align_corners=False, padding_mode="zeros")
    factor = per_frame[:, 3:, None, None, None]  # brightness, contrast, saturation
    video = _colour_by(video, factor[:, 0], factor[:, 1], factor[:, 2])
    video = (video.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
    return video.reshape(batch, steps, *video.shape[1:])


def _colour(video: Tensor, view: View) -> Tensor:
    """Brightness, contrast and saturation, the same for every frame of the clip."""
    return _colour_by(video, view.brightness, view.contrast, view.saturation)


def _colour_by(
    video: Tensor, brightness: Tensor | float, contrast: Tensor | float, saturation: Tensor | float
) -> Tensor:
    """``_colour`` with factors that are numbers or (frames, 1, 1, 1) tensors."""
    video = video * brightness
    mean = video.mean(dim=(1, 2, 3), keepdim=True)
    video = (video - mean) * contrast + mean
    grey = (0.299 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2])[:, None]
    return (video - grey) * saturation + grey
