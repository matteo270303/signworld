"""The whole model on tiny real modules, its optimiser groups, and the budget at full size."""

from pathlib import Path
from typing import Any

import pytest
import torch

from signworld.data.pose.tokens import JOINTS, STEPS
from signworld.experiment.train.budget import ModelBudget
from signworld.experiment.train.config import load_config
from signworld.models.encoders.pose_teachers import TokenEmbedding, transformer
from signworld.models.worldsign.masking import TokenGrid
from signworld.models.worldsign.model import (
    StepRandomness,
    WorldSign,
    WorldSignBatch,
    assemble_worldsign,
)
from signworld.models.worldsign.pose_branch import KeypointDecoders, PoseBranch, PoseEncoder
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

    assert set(terms.parts) == {"e_fis", "anchor", "sigreg_posa", "e_sem", "sigreg_sem"}
    assert not terms.diagnostics
    pose = model.pose
    assert pose is not None and pose.encoder.final is not None
    assert all(adapter.up[0].grad is not None for adapter in pose.encoder.adapters)
    assert pose.encoder.final.weight.grad is not None  # the target is not stopped
    assert all(head.weight.grad is not None for head in pose.decoders.heads)
    assert all(p.grad is not None for p in model.text.head.parameters())
    assert model.video.semantic_predictor.queries.grad is not None
    assert model.video.backbone.adapters[0].up[0].grad is not None


@needs_hub
def test_the_curriculum_can_run_either_pass_alone(tmp_path: Path) -> None:
    model, _ = _model(tmp_path)

    assert set(_step(model, semantic=False).parts) == {"e_fis", "anchor", "sigreg_posa"}
    assert set(_step(model, physical=False).parts) == {"e_sem", "sigreg_sem"}


@needs_hub
def test_esp3_logs_the_pose_sigreg_and_trains_only_the_decoders(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "esp3_pose_frozen.yaml")

    terms = _step(model)
    terms.total.backward()

    assert "sigreg_posa" not in terms.parts and "sigreg_posa" in terms.diagnostics
    assert model.pose is not None
    assert not any(p.requires_grad for p in model.pose.encoder.parameters())
    assert all(head.weight.grad is not None for head in model.pose.decoders.heads)


@needs_hub
def test_esp2_has_no_pose_branch_and_only_the_semantic_terms(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ABLATIONS / "esp2_no_physical.yaml")

    assert model.pose is None
    assert set(_step(model).parts) == {"e_sem", "sigreg_sem"}


@needs_hub
def test_optimizer_groups_slow_the_pose_encoder_and_spare_norms_and_biases(tmp_path: Path) -> None:
    model, config = _model(tmp_path)

    groups = {g["name"]: g for g in model.parameter_groups(config.training)}

    grouped = [id(p) for g in groups.values() for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(grouped) == sorted(trainable)  # each trainable parameter exactly once
    base = config.training.learning_rate
    assert groups["pose-decay"]["lr"] == pytest.approx(0.05 * base)
    assert groups["base-decay"]["lr"] == base
    assert groups["base-no_decay"]["weight_decay"] == 0.0
    assert groups["base-decay"]["weight_decay"] == config.training.weight_decay
    pose = model.pose
    assert pose is not None and pose.encoder.final is not None
    slow = {id(p) for n, g in groups.items() if n.split("-")[0] == "pose" for p in g["params"]}
    assert id(pose.encoder.final.weight) not in slow  # the final layer has its own group
    final = groups["pose_final-decay"]
    assert [id(p) for p in final["params"]] == [id(pose.encoder.final.weight)]
    assert final["lr"] == pytest.approx(0.5 * base) and final["schedule"] == "final_layer"
    assert groups["pose_final-no_decay"]["params"][0] is pose.encoder.final.bias
    assert {g["schedule"] for n, g in groups.items() if not n.startswith("pose_final")} == {"main"}
    assert not slow & {id(p) for p in pose.decoders.parameters()}  # D_pose: base rate
    no_decay = {id(p) for p in groups["base-no_decay"]["params"]}
    assert id(model.video.semantic_predictor.queries) in no_decay


def test_the_full_size_budget_stays_under_the_ceiling() -> None:
    """Trainable parameters of the whole model at full size, against §4.8 and P3."""
    config = load_config(BASE)
    with torch.device("meta"):
        encoder, predictor = full_size_stubs()
        video = assemble(encoder, predictor, config, TokenGrid())
        settings = config.pose_encoder
        pose_encoder = PoseEncoder(
            TokenEmbedding(settings.width),
            transformer(settings.width, settings.depth, settings.heads),
            settings,
        )
        pose = PoseBranch(pose_encoder, KeypointDecoders(settings.width))
        model = assemble_worldsign(video, pose, config)

    budget = ModelBudget.of(model)
    assert budget.pose_lora == 8 * 4 * (3 * 512 + 512 + 1280 + 1280)  # 147,456: §4.8 ~0.15 M
    assert budget.pose_final_layer == 4 * (256 * 256 + 256)  # 263,168
    assert budget.keypoint_decoders == 256 * 2 * len(JOINTS) + 2 * len(JOINTS)  # ~0.04 M
    assert budget.text_head == 656_384  # 0.66 M
    assert budget.total < 22_000_000  # the ceiling of §3.4
