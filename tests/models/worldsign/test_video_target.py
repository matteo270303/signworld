"""ESP-6: the physical level predicts the released V-JEPA 2.1 encoder, frozen, per token."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional

from signworld.models.worldsign.model import StepRandomness
from signworld.models.worldsign.plausibility import plausibility_tests
from signworld.models.worldsign.video_target import (
    TokenPrediction,
    token_energy,
    token_energy_per_clip,
)
from tests.models.worldsign.test_model import _batch, _model
from tests.worldsign import ABLATIONS, meta_modules, needs_hub, tiny_worldsign

ESP6 = ABLATIONS / "esp6_video_target.yaml"


def test_token_energy_is_v_jepa_2_1_loss() -> None:
    target = torch.zeros(1, 4, 2)  # four tokens, two channels
    prediction = TokenPrediction(
        predicted=torch.full((1, 2, 2), 1.0),  # error 1 on tokens 0 and 1
        context=torch.full((1, 2, 2), 3.0),  # error 3 on tokens 2 and 3
        target_index=torch.tensor([[0, 1]]),
        context_index=torch.tensor([[2, 3]]),
        context_weight=torch.tensor([[1.0, 0.5]]),
    )

    energy = token_energy([prediction], target, context_lambda=0.5)

    assert energy.item() == pytest.approx(1.0 + 0.5 * (3.0 * 1.0 + 3.0 * 0.5) / 2)
    assert token_energy_per_clip([prediction], target, 0.5).shape == (1,)


@needs_hub
def test_esp6_has_no_pose_branch_and_predicts_the_video(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ESP6)

    terms = model.loss(_batch(), 10, 100, StepRandomness.at(0, 10), record={})
    terms.total.backward()

    assert model.pose is None and model.has_physical_level
    assert set(terms.parts) == {"e_fis", "e_sem", "sigreg_sem"}
    assert terms.samples["e_fis"].shape == (2,)
    physical = model.video.physical_predictor
    assert physical is not None
    assert all(p.grad is not None for p in physical.readout.parameters())
    assert model.video.backbone.adapters[0].up[0].grad is not None


@needs_hub
def test_the_teacher_is_the_released_encoder_frozen(tmp_path: Path) -> None:
    model, _ = tiny_worldsign(tmp_path, ESP6)
    released, _ = meta_modules(frames=64)  # the same seeded weights the model started from
    teacher = model.video.target
    assert teacher is not None

    for (name, value), (_, original) in zip(
        teacher.encoder.state_dict().items(), released.state_dict().items(), strict=True
    ):
        assert torch.equal(value, original), name
    assert not any(p.requires_grad for p in teacher.parameters())
    model.train()
    assert not teacher.training  # never draws dropout
    with torch.no_grad():
        tokens = model.video.target_tokens(_batch().frames)
    assert torch.allclose(tokens.mean(-1), torch.zeros(()), atol=1e-4)  # layer-normalised
    assert tokens.std(-1, unbiased=False).sub(1).abs().max() < 1e-3


@needs_hub
def test_esp6_plausibility_runs_without_the_pose_control(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, ESP6)

    results = plausibility_tests(model, [_batch()], torch.device("cpu"), masks=2)

    assert [r.name for r in results] == [
        "time_reversed",
        "steps_skipped",
        "steps_frozen",
        "colour_changed",
    ]


@needs_hub
def test_the_teacher_tokens_match_a_fresh_released_encoder(tmp_path: Path) -> None:
    model, _ = tiny_worldsign(tmp_path, ESP6)
    released, _ = meta_modules(frames=64)
    frames = _batch().frames
    from signworld.models.encoders.video_encoders import model_input  # noqa: PLC0415

    with torch.no_grad():
        expected = released(model_input(frames))
        expected = functional.layer_norm(expected, (expected.shape[-1],))

    assert torch.allclose(model.video.target_tokens(frames), expected, atol=1e-5)


@needs_hub
def test_esp6_passes_the_level_separation_and_skips_the_pose_isolation(tmp_path: Path) -> None:
    from signworld.experiment.train.preflight import (  # noqa: PLC0415
        Status,
        p14_pose_isolated,
        p17_levels,
    )

    model, _ = _model(tmp_path, ESP6)

    assert p17_levels(model, _batch()).status == Status.PASS
    assert p14_pose_isolated(model, _batch(), torch.Generator()).status == Status.SKIP
