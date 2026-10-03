"""S-JEPA's pre-training (paper §4.2): data, schedules and the loop.

* ``SkeletonClips``: sequences trimmed and resized to T frames, a new trim at every epoch;
* ``Schedules``: the learning rate, linear from 0 to its peak over the warm-up epochs then a
  cosine to its minimum, and the EMA momentum from its first to its last value on a cosine,
  both per iteration;
* ``Pretrainer``: AdamW on the view encoder and the predictor (no weight decay on biases, norms,
  positions and the mask token, as timm's groups in MAMP's code), the views, the loss, the step
  and the EMA update.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .config import SJEPASettings
from .model import SJEPA
from .views import SkeletonViews, trim_and_resize


class SkeletonClips(Dataset[Tensor]):
    """Skeleton sequences (n, T_max, V, C) with their valid lengths, trimmed and resized."""

    def __init__(
        self,
        sequences: np.ndarray,
        valid: np.ndarray,
        settings: SJEPASettings,
        *,
        training: bool,
        seed: int = 0,
    ) -> None:
        if len(sequences) != len(valid):
            raise ValueError(f"{len(sequences)} sequences for {len(valid)} lengths")
        self.sequences = sequences
        self.valid = valid
        self.settings = settings
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tensor:
        s = self.settings
        sequence = torch.from_numpy(np.asarray(self.sequences[index], dtype=np.float32))
        valid = int(self.valid[index])
        if not self.training:
            return trim_and_resize(sequence, valid, s.frames, s.test_trim)
        state = np.random.SeedSequence([self.seed, self.epoch, index]).generate_state(2)
        rng = np.random.default_rng(state)
        share = float(rng.uniform(*s.trim))
        return trim_and_resize(sequence, valid, s.frames, share, float(rng.uniform()))


@dataclass(frozen=True, slots=True)
class Schedules:
    settings: SJEPASettings
    iterations_per_epoch: int

    @property
    def total(self) -> int:
        return self.settings.epochs * self.iterations_per_epoch

    def learning_rate(self, iteration: int) -> float:
        s = self.settings
        warmup = s.warmup_epochs * self.iterations_per_epoch
        if iteration < warmup:
            return s.learning_rate * iteration / warmup
        progress = min(1.0, (iteration - warmup) / max(self.total - warmup, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return s.min_learning_rate + (s.learning_rate - s.min_learning_rate) * cosine

    def momentum(self, iteration: int) -> float:
        start, end = self.settings.momentum
        progress = min(1.0, iteration / max(self.total, 1))
        return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


def parameter_groups(model: SJEPA, weight_decay: float) -> list[dict[str, Any]]:
    """The student's parameters: with weight decay, and without on biases, 1-D parameters
    (norms, the mask token) and positions."""
    names = {id(p): n for n, p in model.named_parameters()}
    decay: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    for parameter in model.student():
        name = names[id(parameter)]
        quiet = parameter.ndim <= 1 or name.endswith("bias") or ".positions." in name
        (plain if quiet else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]


class Pretrainer:
    """One S-JEPA model, its optimiser, views and schedules."""

    def __init__(self, model: SJEPA, iterations_per_epoch: int, seed: int = 0) -> None:
        s = model.settings
        self.model = model
        self.schedules = Schedules(s, iterations_per_epoch)
        self.views = SkeletonViews(s.views, s.joints)
        self.optimizer = torch.optim.AdamW(
            parameter_groups(model, s.weight_decay), lr=0.0, betas=s.betas
        )
        self.seed = seed
        self.iteration = 0

    def step(self, sequence: Tensor, mean: Callable[[Tensor], Tensor] | None = None) -> float:
        """One iteration on a batch of trimmed and resized sequences; returns the loss."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.schedules.learning_rate(self.iteration)
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.iteration)
        loss = self.model.loss(sequence, self.views(sequence, generator), generator, mean)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        self.model.update_target(self.schedules.momentum(self.iteration))
        self.iteration += 1
        return float(loss.detach())

    def fit(
        self,
        batches: Callable[[int], Iterable[Tensor]],
        device: torch.device,
        report: Callable[[int, float], None] | None = None,
    ) -> None:
        """Every epoch of the settings; ``batches(epoch)`` gives that epoch's batches and
        ``report(epoch, mean loss)`` hears about each epoch."""
        self.model.to(device).train()
        for epoch in range(self.model.settings.epochs):
            losses = [self.step(batch.to(device)) for batch in batches(epoch)]
            if report is not None and losses:
                report(epoch, sum(losses) / len(losses))
