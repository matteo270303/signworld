"""Collaudo of the training system (§4.13.1): controlled overfit and compute efficiency.

* **Overfitting controllato**: a few real clips, every loss on (SIGReg included), trained
  until they are learnt: E_fis and E_sem near 0, R@1 = 100 % on those clips, R² of the visible
  read-outs > 0.95. If the full model cannot memorise 64 clips, the graph or a loss is broken.
* **Efficienza di calcolo**: clips per second of the data loader alone; seconds per training
  step; FLOPs of a step counted by PyTorch (``FlopCounterMode``) and the resulting utilisation
  of the GPUs' peak (MFU); peak memory. PC7 needs these before any run (§4.9: a previous run
  used ~10 % of the compute with the GPUs «busy» 90 % of the time).
"""

import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode

from signworld.experiment.train.curriculum import STAGES, Curriculum, families
from signworld.experiment.train.trainer import Trainer
from signworld.experiment.train.validation import similarity
from signworld.metrics.readings import physical_readings
from signworld.metrics.retrieval import grouped_relevance, recall_at_k
from signworld.models.worldsign.model import StepRandomness, WorldSign, WorldSignBatch

H100_BF16_PEAK: float = 989e12
"""Dense bf16 FLOP/s of an H100 SXM, the value torchtitan uses (§4.9, to be checked)."""


@dataclass(frozen=True, slots=True)
class OverfitResult:
    steps: int
    e_fis: float
    e_sem: float
    r1: float
    r2_visible: float
    passed: bool


def overfit(
    model: WorldSign, batch: WorldSignBatch, steps: int, learning_rate: float = 1e-3
) -> OverfitResult:
    """Train every trainable part on ``batch`` alone, with every loss, and read what it learnt."""
    Curriculum.apply(STAGES["2"], families(model))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    model.train()
    for step in range(steps):
        terms = model.loss(batch, step, steps, StepRandomness.at(0, 0))
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    record: dict[str, Any] = {}
    with torch.no_grad():
        terms = model.loss(batch, steps, steps, StepRandomness.at(0, 0), record=record)
    parts = {k: float(v) for k, v in terms.parts.items()}
    rows = (
        batch.caption_rows.tolist()
        if batch.caption_rows is not None
        else list(range(len(batch.videos)))
    )
    scores = similarity(record["predicted"].cpu(), record["target"].cpu())
    r1 = recall_at_k(scores, grouped_relevance(rows, rows), (1,))[1]
    r2 = float("nan")
    if "latent" in record:
        r2 = physical_readings(
            record["physical"].predictions, record["latent"], record["confidence"]
        )["r2_visible"]
    e_fis, e_sem = parts.get("e_fis", 0.0), parts.get("e_sem", parts.get("infonce", 0.0))
    read_out = math.isnan(r2) or r2 > 0.95  # noqa: PLR2004 (§4.13.1)
    passed = r1 == 1.0 and read_out and e_fis < 0.1 and e_sem < 0.1  # noqa: PLR2004
    return OverfitResult(steps, e_fis, e_sem, r1, r2, passed)


@dataclass(frozen=True, slots=True)
class Efficiency:
    loader_clips_per_second: float
    step_seconds: float
    flops_per_step: float
    utilisation: float
    """Achieved FLOP/s over the peak of every GPU of the run (MFU)."""
    peak_memory_gib: float
    memory_share: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _batches(trainer: Trainer) -> Iterator[WorldSignBatch]:
    epoch = 0
    while True:
        trainer.sampler.configure(epoch)
        yield from trainer.loader
        epoch += 1


def measure_efficiency(
    trainer: Trainer, steps: int = 200, warmup: int = 10, peak_flops: float = H100_BF16_PEAK
) -> Efficiency:
    """The three measurements of «Efficienza di calcolo», on the run's own data and model."""
    device = trainer.device
    source = _batches(trainer)
    began = time.perf_counter()
    loaded = 0
    for _ in range(steps):
        loaded += len(next(source).videos)
    loader_rate = loaded / (time.perf_counter() - began)

    from signworld.experiment.train.checkpoint import TrainingState  # noqa: PLC0415

    state = TrainingState()
    for _ in range(warmup):
        trainer._step(next(source), state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    counter = FlopCounterMode(display=False)
    with counter:
        trainer._step(next(source), state)
    flops = float(counter.get_total_flops())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    for _ in range(steps):
        trainer._step(next(source), state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - began) / steps
    memory, share = 0.0, 0.0
    if device.type == "cuda":
        memory = torch.cuda.max_memory_allocated(device) / 2**30
        share = (
            torch.cuda.max_memory_allocated(device)
            / torch.cuda.get_device_properties(device).total_memory
        )
    utilisation = flops / seconds / peak_flops
    return Efficiency(loader_rate, seconds, flops, utilisation, memory, share)
