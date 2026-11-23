"""Where the time of a training step goes: data, every pass of the model, backward, AdamW.

``measure_efficiency`` reads the step as a whole; this splits it. Every section is timed
between two ``cuda.synchronize`` (start - end), so the GPU's asynchrony does not move time
from one section to the next: the sum is slower than a real step (no overlap), the shares
are what matter. The GPU part runs on one batch held in memory (no loader), for every
stage asked (P, F0, F), with the real model and configuration of the run. The data part
times each phase of ``ClipDataset.__getitem__`` on single clips, then the real loader with
its workers. ``sdpa`` times the attention of one ViT-L block at the clip's 8192 tokens on
every backend this GPU offers (the encoder runs without cuDNN's, ``backbone``).
"""

import dataclasses
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from signworld.data.loaders import ClipDataset
from signworld.data.pose.boxes import step_boxes
from signworld.data.pose.tokens import PoseSequence
from signworld.data.pose.wholebody import PoseTrack
from signworld.data.video import ClipReader
from signworld.experiment.train.checkpoint import TrainingState
from signworld.experiment.train.curriculum import STAGES, Curriculum
from signworld.experiment.train.trainer import Trainer
from signworld.models.worldsign.model import StepRandomness, WorldSign, WorldSignBatch


class Clock:
    """Inclusive seconds per section, nested sections named by their path (``a/b``)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: dict[str, list[float]] = defaultdict(list)
        self.stack: list[str] = []

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        self.sync()
        path = "/".join([*self.stack, name])
        self.stack.append(name)
        began = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[path].append(time.perf_counter() - began)
            self.stack.pop()

    def wrap(self, owner: Any, attribute: str, name: str) -> None:
        """Time every call of ``owner.attribute`` (an instance attribute shadows the class's)."""
        original = getattr(owner, attribute)

        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.section(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, timed)

    def per_step(self, warmup: int, steps: int) -> dict[str, float]:
        """Seconds per step of every section (all its calls in the step), warm-up left out."""
        totals = {}
        for path, values in self.seconds.items():
            calls = len(values) // (warmup + steps)
            totals[path] = float(np.sum(values[warmup * calls :]) / steps)
        return totals


def instrument(model: WorldSign, clock: Clock) -> None:
    """Timers on the passes of ``WorldSign.loss`` and on the encoder inside them."""
    video = model.video
    clock.wrap(model, "pose_terms", "pose")
    clock.wrap(model, "physical_terms", "physical")
    clock.wrap(model, "semantic_terms", "semantic")
    clock.wrap(video.backbone, "context_levels", "encoder_masked")
    clock.wrap(video.backbone, "tokens", "encoder_full")
    if video.physical_predictor is not None:
        clock.wrap(video.physical_predictor, "forward", "predictor")
    clock.wrap(video.semantic_predictor, "query_outputs", "queries")
    clock.wrap(model.text, "forward", "text")


def profile_stage(
    trainer: Trainer, batch: WorldSignBatch, stage: str, steps: int, warmup: int
) -> dict[str, Any]:
    """Mean seconds of each section of a step of ``stage``, on ``batch`` already on the CPU."""
    device = trainer.device
    model = trainer.model
    clock = Clock(device)
    instrument(model, clock)
    chosen = STAGES[stage]
    Curriculum.apply(chosen, trainer.trainable)
    trainer.stage = chosen
    trainer.wrapped = model
    state = TrainingState()
    # a step past the warm-ups, so every family's rate is non-zero as in the stage itself
    state.step = trainer.curriculum.last_start + trainer.steps_per_epoch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seed = trainer.config.training.seed
    for _ in range(warmup + steps):
        trainer._set_rates(state)
        with clock.section("step"):
            with clock.section("to_gpu"):
                moved = batch.to(device)
            with clock.section("augment"):
                moved = moved.augmented()
            randomness = StepRandomness.at(seed, state.step, 0)
            autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=trainer.bf16)
            with clock.section("forward"), autocast:
                terms = model.loss(
                    moved,
                    state.step,
                    trainer.total_steps,
                    randomness,
                    pose=chosen.pose,
                    physical=chosen.physical,
                    semantic=chosen.semantic,
                )
            with clock.section("backward"):
                terms.total.backward()  # type: ignore[no-untyped-call]
            with clock.section("optimizer"):
                trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
        state.step += 1
    memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    for name in ("pose_terms", "physical_terms", "semantic_terms"):  # drop the timers
        model.__dict__.pop(name, None)
    for owner in (model.video.backbone, model.video.physical_predictor,
                  model.video.semantic_predictor, model.text):  # fmt: skip
        if owner is not None:
            for name in ("context_levels", "tokens", "forward", "query_outputs"):
                owner.__dict__.pop(name, None)
    return {
        "stage": stage,
        "clips": len(batch.videos),
        "steps": steps,
        "peak_memory_gib": memory,
        "seconds": clock.per_step(warmup, steps),
        "calls_per_step": {path: len(v) / (warmup + steps) for path, v in clock.seconds.items()},
    }


def profile_clip_reads(data: ClipDataset, clips: int, seed: int = 0) -> dict[str, float]:
    """Mean seconds of every phase of ``ClipDataset.__getitem__`` (its steps, repeated)."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(data), min(clips, len(data)), replace=False)
    times: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def timed(name: str) -> Iterator[None]:
        began = time.perf_counter()
        yield
        times[name].append(time.perf_counter() - began)

    augmenter = data.augmenter
    for index in chosen.tolist():
        began = time.perf_counter()
        with timed("index_row"):
            row = data.table.slice(index, 1).to_pylist()[0]
        with timed("pose_load"):
            track = PoseTrack.load(Path(row["pose_path"]))
        with timed("video_open"):
            reader = ClipReader(Path(row["video_path"]))
        with timed("video_decode"):
            reader.frames(track.frame_indices)
        if augmenter is not None:
            with timed("augment_points"):
                state = int(np.random.SeedSequence([data.seed, 0, index]).generate_state(1)[0])
                view = augmenter.sample(torch.Generator().manual_seed(state))
                moved = augmenter.points(torch.from_numpy(track.keypoints), view)
                track = dataclasses.replace(track, keypoints=moved.numpy())
        with timed("pose_sequence"):
            sequence = PoseSequence.from_track(track)
        if sequence is not None:
            with timed("pose_tokens"):
                sequence.tokens()
            with timed("step_positions"):
                sequence.step_positions()
        with timed("boxes"):
            step_boxes(track, data.box_threshold)
        with timed("caption"):
            np.array(data.embeddings[row["caption_row"]], np.float32)
        times["total"].append(time.perf_counter() - began)
    return {name: float(np.mean(values)) for name, values in times.items()}


def loader_rate(trainer: Trainer, batches: int) -> dict[str, float]:
    """Clips per second of the run's own loader (its workers, prefetch, collate, pinning)."""
    trainer.sampler.configure(0)
    source = iter(trainer.loader)
    began = time.perf_counter()
    next(source)  # the workers' start-up, apart
    first = time.perf_counter() - began
    began, clips = time.perf_counter(), 0
    for _ in range(batches):
        clips += len(next(source).videos)
    seconds = time.perf_counter() - began
    return {
        "workers": float(trainer.config.data.workers),
        "first_batch_seconds": first,
        "clips_per_second": clips / seconds,
        "seconds_per_batch": seconds / batches,
    }


def profile_attention(device: torch.device, batch: int, tokens: int = 8192) -> dict[str, Any]:
    """Seconds of the attention of one ViT-L block (16 heads x 64) per backend, bf16."""
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

    shape = (batch, 16, tokens, 64)
    results: dict[str, Any] = {"shape": list(shape)}
    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ):
        q, k, v = (
            torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        try:
            with sdpa_kernel(backend):
                timings: dict[str, list[float]] = {}
                for kind in ("forward", "forward_backward"):
                    for repeat in range(6):
                        torch.cuda.synchronize(device)
                        began = time.perf_counter()
                        out: Tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                        if kind == "forward_backward":
                            out.sum().backward()  # type: ignore[no-untyped-call]
                        torch.cuda.synchronize(device)
                        if repeat == 1:
                            timings[kind] = []
                        if repeat >= 1:
                            timings[kind].append(time.perf_counter() - began)
                results[backend.name] = {kind: float(np.mean(v)) for kind, v in timings.items()}
        except RuntimeError as error:
            results[backend.name] = f"unavailable: {str(error).splitlines()[0][:120]}"
        del q, k, v
    return results
