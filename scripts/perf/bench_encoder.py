"""Seconds of the V-JEPA 2.1 ViT-L passes of a training step, variant against variant.

The real encoder (Meta's code and weights) with the run's LoRA, on random uint8 clips of
64 frames at 256² (8192 tokens). Variants are timed in alternation, round after round, so a
GPU shared with a running job slows them alike: the ratios hold, the absolute times do not.

    PYTHONPATH=. .venv/bin/python scripts/perf/bench_encoder.py --clips 8 --variants eager,compiled
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import signworld  # noqa: F401  (thread caps)
from signworld.experiment.train.config import load_config
from signworld.models.encoders.video_encoders import load_vjepa2_1
from signworld.models.worldsign import lora, meta_rope
from signworld.models.worldsign.backbone import VideoBackbone

CONFIG = [
    "parameters/model/worldsign.yaml",
    "parameters/model/youtube_sl25.yaml",
    "parameters/ablation/arm_A.yaml",
]


MERGED = ("merged", "rope", "autotune", "nolora")


def build(variant: str, device: torch.device) -> VideoBackbone:
    settings = load_config(*map(Path, CONFIG)).encoder
    compiled = variant != "eager"
    meta_rope.ENABLED = variant in ("rope", "autotune")  # read when the blocks are compiled
    torch._inductor.config.max_autotune = variant == "autotune"
    torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON"
    settings = settings.model_copy(update={"compile_blocks": compiled})
    encoder, _, _ = load_vjepa2_1(settings.hub_repo, settings.entrypoint, settings.checkpoint)
    backbone = VideoBackbone(encoder.to(device), settings, adapted=variant != "nolora")
    torch.manual_seed(0)
    for layer in backbone.adapters:  # a trained-looking B, so the update is not zero
        for up in layer.up:
            up.data.normal_(0, 0.02)
    backbone.train()  # as in the trainer
    return backbone


def frames_like(clips: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 256, (clips, 64, 256, 256, 3), dtype=torch.uint8, device=device)


def timed(run: Any, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    began = time.perf_counter()
    run()
    torch.cuda.synchronize(device)
    return time.perf_counter() - began


def semantic_pass(backbone: VideoBackbone, frames: torch.Tensor) -> Any:
    def run() -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone.tokens(frames)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--variants", default="eager,compiled,merged,nolora")
    parser.add_argument("--trace", help="variant whose kernels torch.profiler lists")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    frames = frames_like(args.clips, device)
    variants = args.variants.split(",")
    runs = {}
    for name in variants:
        backbone = build(name, device)
        lora.MERGE_WITHOUT_GRADIENT = name in MERGED
        runs[name] = (backbone, semantic_pass(backbone, frames))
        for _ in range(2):  # compilation and cuDNN/cuBLAS warm-up
            runs[name][1]()
        lora.MERGE_WITHOUT_GRADIENT = False
        print(f"{name}: ready", flush=True)
    seconds: dict[str, list[float]] = {name: [] for name in variants}
    for _ in range(args.rounds):
        for name in variants:
            lora.MERGE_WITHOUT_GRADIENT = name in MERGED
            seconds[name].append(timed(runs[name][1], device))
    lora.MERGE_WITHOUT_GRADIENT = False
    reference = statistics.median(seconds[variants[0]])
    result = {}
    for name in variants:
        median = statistics.median(seconds[name])
        result[name] = {"median_s": median, "per_clip_ms": 1000 * median / args.clips,
                        "vs_first": median / reference}  # fmt: skip
        print(f"{name:10s} {median:7.3f} s  {1000 * median / args.clips:7.1f} ms/clip  "
              f"x{median / reference:.3f}")  # fmt: skip
    if args.trace:
        lora.MERGE_WITHOUT_GRADIENT = args.trace in MERGED
        activities = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            runs[args.trace][1]()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    if args.output:
        args.output.write_text(json.dumps({"clips": args.clips, "result": result}, indent=2))


if __name__ == "__main__":
    main()
