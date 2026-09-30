"""The learning-rate schedule and the curriculum of stages (§4.10).

``LearningRateSchedule`` is V-JEPA 2's: linear warm-up, constant phase, linear cooldown to 0.
The cooldown starts where the constant phase ends: at the end of the planned steps, or earlier
from the best checkpoint when early stopping ends the constant phase. ``CosineSchedule`` is the
pose encoder's final layer's own: linear warm-up, then a cosine to 0 well before the end, after
which the layer is fixed (FreezeOut's per-layer annealing; DINO's warm-up and cosine).

``Curriculum`` decides at every step which passes run and which parameters train:

    1a   physical pass    new physical modules only (fusion, head, D_pose, the pose encoder's
                          final layer); every LoRA frozen
    1    physical pass    the whole physical level: E_fis + L_anchor + SIGReg_posa
    2a   semantic pass    new semantic modules only (semantic predictor, text head)
    2    both passes      everything

Without the physical level (ESP-2) stages 1a and 1 do not exist and 2a starts at step 0.
Freezing is done with ``requires_grad``, so a frozen part costs no backward at all: in 2a the
backward stops before the video encoder.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from .lora import adapter_parameters
from .model import WorldSign


@dataclass(frozen=True, slots=True)
class LearningRateSchedule:
    warmup_steps: int
    cooldown_steps: int

    def factor(self, step: int, cooldown_start: int) -> float:
        """Multiplier of every group's learning rate at ``step`` (0-based)."""
        if step >= cooldown_start:
            done = step - cooldown_start + 1
            return max(0.0, 1.0 - done / max(self.cooldown_steps, 1))
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0


@dataclass(frozen=True, slots=True)
class CosineSchedule:
    warmup_steps: int
    end_step: int
    """The step where the factor reaches 0 and stays."""

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step >= self.end_step:
            return 0.0
        progress = (step - self.warmup_steps) / max(self.end_step - self.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


FAMILIES = (
    "video_lora",
    "physical_lora",
    "physical_new",
    "pose_target",
    "pose_final",
    "pose_decoders",
    "semantic_new",
)


def families(model: WorldSign) -> dict[str, list[nn.Parameter]]:
    """The model's trainable parameters, by the part of the curriculum they belong to."""
    video = model.video
    backbone = video.backbone
    lora = adapter_parameters(backbone.encoder)
    lora_ids = {id(p) for p in lora}
    norms = [
        p
        for m in backbone.encoder.modules()
        if isinstance(m, nn.LayerNorm)
        for p in m.parameters()
        if p.requires_grad and id(p) not in lora_ids
    ]
    found: dict[str, list[nn.Parameter]] = {name: [] for name in FAMILIES}
    found["video_lora"] = lora + norms
    physical = video.physical_predictor
    if physical is not None:
        found["physical_lora"] = adapter_parameters(physical.predictor)
        found["physical_new"] = [*physical.fusion.parameters(), *physical.readout.parameters()]
    if model.pose is not None:
        final = model.pose.encoder.final
        final_ids = set() if final is None else {id(p) for p in final.parameters()}
        found["pose_target"] = [
            p for p in model.pose.encoder.parameters() if id(p) not in final_ids
        ]
        found["pose_final"] = [] if final is None else list(final.parameters())
        found["pose_decoders"] = list(model.pose.decoders.parameters())
    found["semantic_new"] = [
        *video.semantic_predictor.parameters(),
        *model.text.parameters(),
        *model.objective.parameters(),
    ]
    return {name: [p for p in params if p.requires_grad] for name, params in found.items()}


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    physical: bool
    semantic: bool
    trains: frozenset[str]


STAGES = {
    "1a": Stage("1a", True, False, frozenset({"physical_new", "pose_final", "pose_decoders"})),
    "1": Stage(
        "1",
        True,
        False,
        frozenset(
            {
                "video_lora",
                "physical_lora",
                "physical_new",
                "pose_target",
                "pose_final",
                "pose_decoders",
            }
        ),
    ),
    "2a": Stage("2a", False, True, frozenset({"semantic_new"})),
    "2": Stage("2", True, True, frozenset(FAMILIES)),
}


class Curriculum:
    """Stage boundaries from the fractions of the run, and the switch between stages."""

    def __init__(self, fractions: dict[str, float], total_steps: int, physical_level: bool) -> None:
        order = ["1a", "1", "2a"] if physical_level else ["2a"]
        self.bounds: list[tuple[int, Stage]] = []
        start = 0.0
        for name in order:
            self.bounds.append((round(start * total_steps), STAGES[name]))
            start += fractions[name]
        self.bounds.append((round(start * total_steps), STAGES["2"]))

    @property
    def last(self) -> Stage:
        """Stage 2: everything trains. The cooldown always runs in it."""
        return self.bounds[-1][1]

    def stage_at(self, step: int) -> Stage:
        current = self.bounds[0][1]
        for start, stage in self.bounds:
            if step >= start:
                current = stage
        return current

    @staticmethod
    def apply(stage: Stage, trainable: dict[str, list[nn.Parameter]]) -> None:
        """Let exactly the stage's families train among the ever-trainable parameters."""
        for name, parameters in trainable.items():
            _set(parameters, name in stage.trains)


def _set(parameters: Iterable[nn.Parameter], flag: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(flag)


def trainable_names(model: WorldSign) -> set[str]:
    """Names of every parameter the curriculum can train: what a checkpoint keeps."""
    names = {id(p): n for n, p in model.named_parameters()}
    return {names[id(p)] for group in families(model).values() for p in group}
