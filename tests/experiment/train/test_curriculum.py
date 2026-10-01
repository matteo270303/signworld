"""The learning-rate schedule and the stages of the curriculum (§4.10)."""

from pathlib import Path

import pytest

from signworld.worldmodel.curriculum import (
    FAMILIES,
    CosineSchedule,
    Curriculum,
    LearningRateSchedule,
    families,
)

from .conftest import ABLATIONS, needs_hub, tiny_worldsign

FRACTIONS = {"1a": 0.01, "1": 0.05, "2a": 0.01}


def test_the_schedule_warms_up_holds_and_cools_down_to_zero() -> None:
    schedule = LearningRateSchedule(warmup_steps=4, cooldown_steps=5)

    assert [schedule.factor(s, cooldown_start=100) for s in (0, 3, 50)] == [0.25, 1.0, 1.0]
    assert schedule.factor(100, cooldown_start=100) == pytest.approx(0.8)
    assert schedule.factor(104, cooldown_start=100) == 0.0
    assert schedule.factor(60, cooldown_start=60) == pytest.approx(0.8)  # early stopping


def test_the_final_layer_warms_up_then_decays_to_zero_and_stays() -> None:
    schedule = CosineSchedule(warmup_steps=10, end_step=50)

    assert schedule.factor(0) == pytest.approx(0.1) and schedule.factor(9) == 1.0
    assert schedule.factor(10) == 1.0 and schedule.factor(30) == pytest.approx(0.5)
    assert schedule.factor(50) == 0.0 and schedule.factor(90) == 0.0


def test_stages_follow_the_fractions_of_the_run() -> None:
    curriculum = Curriculum(FRACTIONS, total_steps=1000, physical_level=True)

    names = [curriculum.stage_at(step).name for step in (0, 9, 10, 59, 60, 69, 70, 999)]

    assert names == ["1a", "1a", "1", "1", "2a", "2a", "2", "2"]
    assert not curriculum.stage_at(0).semantic and not curriculum.stage_at(60).physical


def test_without_the_physical_level_the_run_starts_at_2a() -> None:
    curriculum = Curriculum(FRACTIONS, total_steps=1000, physical_level=False)

    assert [curriculum.stage_at(step).name for step in (0, 9, 10)] == ["2a", "2a", "2"]


@needs_hub
def test_families_cover_every_trainable_parameter_once(tmp_path: Path) -> None:
    model, _ = tiny_worldsign(tmp_path, ABLATIONS / "arm_C.yaml")

    found = families(model)

    listed = [id(p) for group in found.values() for p in group]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(listed) == sorted(trainable)
    assert all(found[name] for name in FAMILIES)
    Curriculum.apply(Curriculum(FRACTIONS, 1000, True).stage_at(0), found)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {
        id(p) for name in ("physical_new", "pose_final", "pose_decoders") for p in found[name]
    }  # the final layer moves from the first step
