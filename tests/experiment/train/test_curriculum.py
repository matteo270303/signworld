"""The learning-rate schedules and the stages of the curriculum (gerarchia §6)."""

from pathlib import Path

import pytest

from signworld.experiment.train.config import StageSettings
from signworld.experiment.train.curriculum import (
    FAMILIES,
    PHYSICAL_NEW,
    POSE,
    SEMANTIC_NEW,
    VIDEO_LORA,
    Curriculum,
    families,
)
from signworld.experiment.train.schedules import Cooldown, GroupSchedule
from tests.worldsign import ABLATIONS, needs_hub, tiny_worldsign

STAGES = StageSettings(pose_epochs=2, heads_epochs=1)


def test_a_group_is_off_before_its_entry_then_warms_up_and_holds() -> None:
    schedule = GroupSchedule(start=10, warmup=4)

    assert [schedule.factor(s) for s in (0, 9, 10, 13, 50)] == [0.0, 0.0, 0.25, 1.0, 1.0]


def test_the_pose_warms_up_then_follows_a_cosine_to_zero() -> None:
    schedule = GroupSchedule(start=0, warmup=10, decay_end=50)

    assert schedule.factor(0) == pytest.approx(0.1) and schedule.factor(9) == 1.0
    assert schedule.factor(30) == pytest.approx(0.5)
    assert schedule.factor(50) == 0.0 and schedule.factor(90) == 0.0


def test_the_cooldown_falls_linearly_to_zero_from_its_start() -> None:
    cooldown = Cooldown(steps=5)

    assert cooldown.factor(99, start=100) == 1.0
    assert cooldown.factor(100, start=100) == pytest.approx(0.8)
    assert cooldown.factor(104, start=100) == 0.0
    assert cooldown.factor(60, start=60) == pytest.approx(0.8)  # early stopping


def test_stages_follow_the_epochs() -> None:
    curriculum = Curriculum(STAGES, steps_per_epoch=10, physical_level=True)

    names = [curriculum.stage_at(step).name for step in (0, 19, 20, 29, 30, 999)]

    assert names == ["P", "P", "F0", "F0", "F", "F"]
    assert not curriculum.stage_at(0).physical and curriculum.stage_at(20).physical
    assert curriculum.last_start == 30
    assert curriculum.entries == {POSE: 0, SEMANTIC_NEW: 0, PHYSICAL_NEW: 20} | {
        name: 30 for name in FAMILIES if name not in (POSE, SEMANTIC_NEW, PHYSICAL_NEW)
    }


def test_without_the_physical_level_one_stage_lasts_the_whole_run() -> None:
    curriculum = Curriculum(STAGES, steps_per_epoch=10, physical_level=False)

    assert [curriculum.stage_at(step).name for step in (0, 9, 500)] == ["S", "S", "S"]
    assert curriculum.entries == {SEMANTIC_NEW: 0, VIDEO_LORA: 0}


@needs_hub
def test_families_cover_every_trainable_parameter_once(tmp_path: Path) -> None:
    model, _ = tiny_worldsign(tmp_path, ABLATIONS / "arm_C.yaml")

    found = families(model)

    listed = [id(p) for group in found.values() for p in group]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(listed) == sorted(trainable)
    assert all(found[name] for name in FAMILIES)
