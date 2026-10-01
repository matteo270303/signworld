"""Checkpoints of a run: what was trained, the optimiser, and where the run stands.

The frozen pre-trained weights (V-JEPA 2.1, S-JEPA) are not saved again: they are rebuilt
from their own checkpoints, and a checkpoint holds only what training changes. Saving writes a
temporary file and renames it, so a job killed while saving leaves the previous one intact.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .distributed import SINGLE, Distributed


@dataclass
class TrainingState:
    """Where a run stands; enough to resume it exactly."""

    step: int = 0
    epoch: int = 0
    position: int = 0
    """Clips of the current epoch this GPU has already consumed."""
    best_metric: float = -math.inf
    best_step: int = -1
    bad_epochs: int = 0
    cooldown_start: int | None = None
    """The step where the cooldown began; ``None`` during the constant phase."""
    finished: bool = False


def frozen_names(model: nn.Module, trainable: set[str]) -> set[str]:
    """Parameters that are never trained: rebuilt from the pre-trained checkpoints."""
    return {name for name, _ in model.named_parameters() if name not in trainable}


class CheckpointStore:
    def __init__(
        self, directory: Path, trainable: set[str], collective: Distributed = SINGLE
    ) -> None:
        self.directory = directory
        self.trainable = trainable
        self.collective = collective

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.pt"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def save(
        self, name: str, model: nn.Module, optimizer: torch.optim.Optimizer, state: TrainingState
    ) -> None:
        """Written by the first GPU only; every GPU waits for it."""
        if self.collective.is_main:
            frozen = frozen_names(model, self.trainable)
            payload = {
                "model": {k: v for k, v in model.state_dict().items() if k not in frozen},
                "optimizer": optimizer.state_dict(),
                "state": dataclasses.asdict(state),
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path(name).with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(self.path(name))
        self.collective.barrier()

    def load(
        self, name: str, model: nn.Module, optimizer: torch.optim.Optimizer | None
    ) -> TrainingState:
        payload: dict[str, Any] = torch.load(
            self.path(name), map_location="cpu", weights_only=False
        )
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        stray = set(missing) - frozen_names(model, self.trainable)
        if stray or unexpected:
            raise RuntimeError(
                f"{self.path(name)} does not fit the model: missing {sorted(stray)[:5]}, "
                f"unexpected {sorted(unexpected)[:5]}"
            )
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return TrainingState(**payload["state"])
