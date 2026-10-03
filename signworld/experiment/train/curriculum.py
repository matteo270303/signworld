"""The curriculum of stages: which levels run and which parameters train (gerarchia §6.1).

The trainable parameters fall in five families; a stage runs some levels and trains some
families. Stages follow the epochs:

    P    epochs 1 .. pose_epochs      pose + semantic        pose, semantic_new
    F0   the next heads_epochs        + physical             + physical_new (fusion, read-out)
    F    to the end                   every level            + physical_lora, video_lora

Without the physical level (ESP-2) there is one stage, S, where only the semantic level runs
and trains: the encoder then has no LoRA, since only the physical level changes it, unless the
«global» ablation lets the semantic level train it. Freezing is
done with ``requires_grad``, so a frozen part costs no backward. ``entries`` says where each
family enters training, which is where its learning rate starts its warm-up (``schedules``).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from torch import nn

from signworld.models.worldsign.lora import adapter_parameters
from signworld.models.worldsign.model import WorldSign

from .config import StageSettings

POSE: Final = "pose"
SEMANTIC_NEW: Final = "semantic_new"
PHYSICAL_NEW: Final = "physical_new"
PHYSICAL_LORA: Final = "physical_lora"
VIDEO_LORA: Final = "video_lora"
FAMILIES: Final = (POSE, SEMANTIC_NEW, PHYSICAL_NEW, PHYSICAL_LORA, VIDEO_LORA)


def families(model: WorldSign) -> dict[str, list[nn.Parameter]]:
    """The model's trainable parameters, by the part of the curriculum they belong to."""
    video = model.video
    encoder = video.backbone.encoder
    lora = adapter_parameters(encoder)
    lora_ids = {id(p) for p in lora}
    norms = [
        p
        for m in encoder.modules()
        if isinstance(m, nn.LayerNorm)
        for p in m.parameters()
        if p.requires_grad and id(p) not in lora_ids
    ]
    found: dict[str, list[nn.Parameter]] = {name: [] for name in FAMILIES}
    found[VIDEO_LORA] = lora + norms
    physical = video.physical_predictor
    if physical is not None:
        found[PHYSICAL_LORA] = adapter_parameters(physical.predictor)
        found[PHYSICAL_NEW] = [*physical.fusion.parameters(), *physical.readout.parameters()]
    if model.pose is not None:
        found[POSE] = list(model.pose.parameters())
    found[SEMANTIC_NEW] = [
        *video.semantic_predictor.parameters(),
        *model.text.parameters(),
        *model.objective.parameters(),
    ]
    return {name: [p for p in params if p.requires_grad] for name, params in found.items()}


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    pose: bool
    physical: bool
    semantic: bool
    trains: frozenset[str]


STAGES: Final = {
    "P": Stage(
        "P", pose=True, physical=False, semantic=True, trains=frozenset({POSE, SEMANTIC_NEW})
    ),
    "F0": Stage(
        "F0",
        pose=True,
        physical=True,
        semantic=True,
        trains=frozenset({POSE, SEMANTIC_NEW, PHYSICAL_NEW}),
    ),
    "F": Stage("F", pose=True, physical=True, semantic=True, trains=frozenset(FAMILIES)),
    "S": Stage(
        "S", pose=False, physical=False, semantic=True, trains=frozenset({SEMANTIC_NEW, VIDEO_LORA})
    ),
}


class Curriculum:
    """Stage boundaries from the epochs, and the switch between stages."""

    def __init__(self, settings: StageSettings, steps_per_epoch: int, physical_level: bool) -> None:
        if steps_per_epoch < 1:
            raise ValueError("an epoch needs at least one step")
        self.steps_per_epoch = steps_per_epoch
        if physical_level:
            heads = settings.pose_epochs * steps_per_epoch
            full = heads + settings.heads_epochs * steps_per_epoch
            self.bounds = [(0, STAGES["P"]), (heads, STAGES["F0"]), (full, STAGES["F"])]
        else:
            self.bounds = [(0, STAGES["S"])]

    @property
    def last(self) -> Stage:
        """The stage that lasts to the end: F, or S without the physical level."""
        return self.bounds[-1][1]

    @property
    def last_start(self) -> int:
        """The step where the last stage begins: the best checkpoint and the patience count
        from there."""
        return self.bounds[-1][0]

    def start_of(self, name: str) -> int | None:
        """The step where stage ``name`` begins; None when the run has no such stage."""
        return next((start for start, stage in self.bounds if stage.name == name), None)

    def stage_at(self, step: int) -> Stage:
        current = self.bounds[0][1]
        for start, stage in self.bounds:
            if step >= start:
                current = stage
        return current

    @property
    def entries(self) -> dict[str, int]:
        """The step where every family some stage trains enters training."""
        found: dict[str, int] = {}
        for start, stage in self.bounds:
            for name in sorted(stage.trains):
                found.setdefault(name, start)
        return found

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
