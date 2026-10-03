"""Learning rates: one AdamW group per curriculum family, each on its own schedule (gerarchia §6.2).

A group's learning rate is its peak times ``GroupSchedule.factor`` times ``Cooldown.factor``:

* every family but the pose is 0 before it enters training, rises linearly for
  ``activation_warmup_epochs`` from its entry (LLaVA's warm-up at every stage), then stays;
* the pose rises linearly over ``pose_encoder.warmup_fraction`` of the run, then follows a
  cosine to 0 at the planned end: the target slows down while the video chases it;
* the cooldown is V-JEPA 2's: from the end of the constant phase, or from the best checkpoint
  when early stopping ends it, every rate falls linearly to 0.

No weight decay on norms, biases, scalars, positions and learned queries.
"""

import math
from dataclasses import dataclass
from typing import Any, Final

from torch import nn

from signworld.models.worldsign.model import WorldSign

from .config import WorldSignConfig
from .curriculum import POSE, Curriculum, families

_NO_DECAY: Final = ("bias", "queries", "position", "joint_type")


@dataclass(frozen=True, slots=True)
class GroupSchedule:
    """Factor of a group's peak rate: 0 until ``start``, a linear warm-up of ``warmup`` steps,
    then constant, or a cosine to 0 at ``decay_end``."""

    start: int
    warmup: int
    decay_end: int | None = None

    def __post_init__(self) -> None:
        if self.warmup < 1:
            raise ValueError("the warm-up needs at least one step")
        if self.decay_end is not None and self.decay_end <= self.start + self.warmup:
            raise ValueError("the cosine must end after the warm-up")

    def factor(self, step: int) -> float:
        if step < self.start:
            return 0.0
        done = step - self.start
        if done < self.warmup:
            return (done + 1) / self.warmup
        if self.decay_end is None:
            return 1.0
        span = self.decay_end - self.start - self.warmup
        progress = min(1.0, (done - self.warmup) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class Cooldown:
    """V-JEPA 2's cooldown: 1 until it starts, then linear to 0 over ``steps``."""

    steps: int

    def factor(self, step: int, start: int) -> float:
        if step < start:
            return 1.0
        return max(0.0, 1.0 - (step - start + 1) / max(self.steps, 1))


def group_schedules(
    config: WorldSignConfig, curriculum: Curriculum, total_steps: int
) -> dict[str, GroupSchedule]:
    """The schedule of every family that some stage trains."""
    training = config.training
    activation_warmup = max(
        1, round(training.activation_warmup_epochs * curriculum.steps_per_epoch)
    )
    schedules: dict[str, GroupSchedule] = {}
    for family, start in curriculum.entries.items():
        if family == POSE:
            warmup = max(1, round(config.pose_encoder.warmup_fraction * total_steps))
            schedules[family] = GroupSchedule(start, warmup, decay_end=total_steps)
        else:
            schedules[family] = GroupSchedule(start, activation_warmup)
    return schedules


def _decays(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim > 1 and not name.endswith(_NO_DECAY)


def parameter_groups(model: WorldSign, config: WorldSignConfig) -> list[dict[str, Any]]:
    """AdamW groups: one per family and weight decay, each with its peak rate as ``lr``.

    ``family`` names the group's schedule for the trainer. The pose family peaks at
    ``pose_encoder.learning_rate``, every other at the run's base rate.
    """
    training = config.training
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for family, parameters in families(model).items():
        peak = config.pose_encoder.learning_rate if family == POSE else training.learning_rate
        for decay in (True, False):
            chosen = [p for p in parameters if _decays(names[id(p)], p) == decay]
            if chosen:
                groups.append(
                    {
                        "name": f"{family}-{'decay' if decay else 'no_decay'}",
                        "family": family,
                        "params": chosen,
                        "lr": peak,
                        "weight_decay": training.weight_decay if decay else 0.0,
                    }
                )
    return groups
