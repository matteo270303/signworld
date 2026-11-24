"""Seconds of the GPU augmentation of one batch: clip by clip (before) against batched (now).

PYTHONPATH=. .venv/bin/python scripts/perf/bench_augment.py --clips 40
"""

import argparse
import statistics
import time

import torch

import signworld  # noqa: F401  (thread caps)
from signworld.data.augmentation import ClipAugmenter, View, apply_views
from signworld.experiment.train.config import AugmentationSettings


def clip_by_clip(frames: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [ClipAugmenter.frames(c, View.from_tensor(v)) for c, v in zip(frames, views, strict=True)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda")
    augmenter = ClipAugmenter(AugmentationSettings())
    generator = torch.Generator().manual_seed(0)
    views = torch.stack([augmenter.sample(generator).as_tensor() for _ in range(args.clips)])
    frames = torch.randint(0, 256, (args.clips, 64, 256, 256, 3), dtype=torch.uint8, device=device)
    gpu_views = views.to(device)
    before, after = clip_by_clip(frames, gpu_views), apply_views(frames, gpu_views)
    differing = (before != after).sum().item()
    print(f"bytes differing: {differing} of {before.numel()}, max |diff| "
          f"{(before.int() - after.int()).abs().max().item()}")  # fmt: skip
    times: dict[str, list[float]] = {"clip_by_clip": [], "batched": []}
    for _ in range(args.rounds):
        for name, run in (("clip_by_clip", clip_by_clip), ("batched", apply_views)):
            torch.cuda.synchronize()
            began = time.perf_counter()
            run(frames, gpu_views)
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - began)
    for name, values in times.items():
        print(f"{name:12s} {statistics.median(values) * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
