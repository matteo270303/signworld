"""The whole model on tiny real modules, its optimiser groups, and the budget at full size."""

import dataclasses
from pathlib import Path
from typing import Any

import torch

from signworld.data.pose.tokens import JOINTS, STEPS
from signworld.experiment.train.budget import ModelBudget
from signworld.experiment.train.config import load_config
from signworld.experiment.train.schedules import parameter_groups
from signworld.models.worldsign.masking import TokenGrid
from signworld.models.worldsign.model import (
    StepRandomness,
    WorldSign,
    WorldSignBatch,
    assemble_worldsign,
)
from signworld.models.worldsign.pose_branch import PoseBranch
from signworld.models.worldsign.video_branch import assemble
from tests.worldsign import ABLATIONS, BASE, full_size_stubs, needs_hub, pose_tokens, tiny_worldsign

CAPTION_LANGUAGES = ["en", "es"]


def _model(directory: Path, *overlays: Path) -> tuple[WorldSign, Any]:
    model, config = tiny_worldsign(directory, *overlays)
    generator = torch.Generator().manual_seed(0)
    model.text.centering.fit(torch.randn(10, 768, generator=generator), CAPTION_LANGUAGES * 5)
    if model.pose is not None:
        batch = _batch()
        model.pose.fit(batch.keypoints, batch.keypoint_weights)
    return model, config


def _batch(size: int = 2) -> WorldSignBatch:
    generator = torch.Generator().manual_seed(0)
    joints = len(JOINTS)
    return WorldSignBatch(
        frames=torch.randint(0, 256, (size, 2 * STEPS, 64, 64, 3), dtype=torch.uint8),
        pose_tokens=pose_tokens(size),
        keypoints=torch.randn(size, STEPS, joints, 2, generator=generator),
        keypoint_weights=(torch.rand(size, STEPS, joints, generator=generator) > 0.1).float(),
        boxes=torch.tensor([0.1, 0.1, 0.6, 0.7]).expand(size, STEPS, 4, 4).clone(),
        box_visible=torch.ones(size, STEPS, 4, dtype=torch.bool),
        captions=torch.randn(size, 768, generator=generator),
        languages=torch.tensor([0, 1])[:size],
        videos=[f"video-{i}" for i in range(size)],
    )


def _step(model: WorldSign, **passes: bool) -> Any:
    return model.loss(_batch(), 10, 100, StepRandomness.at(0, 10), **passes)


@needs_hub
def test_one_step_of_the_reference_model_reaches_every_trainable_part(tmp_path: Path) -> None:
    model, _ = _model(tmp_path)

    terms = _step(model)
    terms.total.backward()

    assert set(terms.parts) == {
        "inv_posa",
        "anchor",
        "sigreg_posa",
        "e_fis",
        "e_sem",
        "sigreg_sem",
    }
    assert not terms.diagnostics
    pose = model.pose
    assert pose is not None
    assert all(p.grad is not None for p in pose.encoder.parameters())
    assert all(p.grad is not None for p in pose.decoder.parameters())
    assert all(p.grad is not None for p in model.text.head.parameters())
    assert model.video.semantic_predictor.queries.grad is not None
    assert model.video.backbone.adapters[0].up[0].grad is not None


@needs_hub
def test_each_level_trains_only_its_own_parameters(tmp_path: Path) -> None:
    """P17: E_fis does not reach the pose encoder, E_sem does not reach the video LoRA."""
    model, _ = _model(tmp_path)

    terms = _step(model)
    terms.parts["e_fis"].backward(retain_graph=True)

    pose = model.pose
    assert pose is not None
    assert all(p.grad is None for p in pose.encoder.parameters())
    model.zero_grad(set_to_none=True)
    terms.parts["e_sem"].backward()
    assert all(p.grad is None for a in model.video.backbone.adapters for p in a.parameters())


@needs_hub
def test_the_curriculum_can_run_either_pass_alone(tmp_path: Path) -> None:
    model, _ = _model(tmp_path)

    pose = {"inv_posa", "anchor", "sigreg_posa"}
    assert set(_step(model, semantic=False).parts) == pose | {"e_fis"}
    assert set(_step(model, physical=False).parts) == pose | {"e_sem", "sigreg_sem"}
    alone = _step(model, pose=False, physical=False)
    assert set(alone.parts) == {"e_sem", "sigreg_sem"} and not alone.diagnostics
    measured = _step(model, pose=False)
    assert set(measured.diagnostics) == pose and "e_fis" in measured.parts


@needs_hub
def test_esp2_has_no_pose_branch_and_only_the_semantic_terms(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "esp2_no_physical.yaml")

    assert model.pose is None
    assert set(_step(model).parts) == {"e_sem", "sigreg_sem"}


@needs_hub
def test_optimizer_groups_give_the_pose_its_rate_and_spare_norms_and_biases(
    tmp_path: Path,
) -> None:
    model, config = _model(tmp_path)

    groups = {g["name"]: g for g in parameter_groups(model, config)}

    grouped = [id(p) for g in groups.values() for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(grouped) == sorted(trainable)  # each trainable parameter exactly once
    base = config.training.learning_rate
    assert groups["pose-decay"]["lr"] == config.pose_encoder.learning_rate
    assert groups["semantic_new-decay"]["lr"] == base
    assert groups["semantic_new-no_decay"]["weight_decay"] == 0.0
    assert groups["semantic_new-decay"]["weight_decay"] == config.training.weight_decay
    no_decay = {id(p) for p in groups["semantic_new-no_decay"]["params"]}
    assert id(model.video.semantic_predictor.queries) in no_decay


def test_the_full_size_budget_stays_under_the_ceiling() -> None:
    """Trainable parameters of the whole model at full size, against §4.8 and P3."""
    config = load_config(BASE)
    with torch.device("meta"):
        encoder, predictor = full_size_stubs()
        video = assemble(encoder, predictor, config, TokenGrid())
        pose = PoseBranch.from_settings(config.pose_encoder)
        model = assemble_worldsign(video, pose, config)

    budget = ModelBudget.of(model)
    assert budget.pose_encoder == 8_025_408  # posa §3: 8.03 M
    assert budget.keypoint_decoder == 192 * 2 * len(JOINTS) + 2 * len(JOINTS)
    assert budget.total < 30_000_000  # the ceiling of P3


@needs_hub
def test_arm_v_regularises_with_vicreg_instead_of_sigreg(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "arm_V.yaml")

    parts = set(_step(model).parts)

    assert {"e_sem", "vicreg_var", "vicreg_cov"} <= parts and "sigreg_sem" not in parts


@needs_hub
def test_g1_lets_e_sem_reach_the_video_lora(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "g1_global.yaml")

    _step(model).parts["e_sem"].backward()

    assert model.video.backbone.adapters[0].up[0].grad is not None


@needs_hub
def test_arm_c_reads_the_caption_rows_and_reports_its_pairs(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "arm_C.yaml")
    batch = dataclasses.replace(_batch(), caption_rows=torch.tensor([3, 3]))

    terms = model.loss(batch, 10, 100, StepRandomness.at(0, 10))

    assert "infonce" in terms.parts
    assert terms.diagnostics["positives_per_clip"].item() == 1.0  # the same caption
    assert terms.diagnostics["temperature"].item() > 0
