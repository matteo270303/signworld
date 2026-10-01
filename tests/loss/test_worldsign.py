"""The energies and the objective of every arm."""

import pytest
import torch
from torch.nn import functional

from signworld.metrics.sigreg import SIGReg, random_directions
from signworld.worldmodel.config import LossSettings, SemanticSettings
from signworld.worldmodel.losses import (
    InfoNCE,
    Objective,
    free_energy,
    physical_energy,
    semantic_energy,
    uniformity,
)
from signworld.worldmodel.physical import PhysicalPrediction


def _prediction(
    masked: torch.Tensor, visible: torch.Tensor, counts: tuple[list[float], ...]
) -> PhysicalPrediction:
    """``counts``: masked tokens, visible tokens and Σ 1/√d of the visible ones, per box."""
    masked_count, visible_count, visible_weight = (torch.tensor(c).view(1, 1, -1) for c in counts)
    return PhysicalPrediction(masked, visible, masked_count, visible_count, visible_weight)


def test_physical_energy_is_v_jepa_2_1_loss_read_per_box() -> None:
    pattern = torch.tensor([1.0, -1.0, 1.0, -1.0])  # zero mean, unit variance
    target = (3.0 * pattern + 5.0).expand(1, 1, 2, 4)  # its layer norm is the pattern
    masked = torch.stack([pattern, pattern + 2.0])[None, None]  # errors 0 and 2 per channel
    visible = (pattern + 1.0).expand(1, 1, 2, 4)  # error 1 in both boxes
    prediction = _prediction(masked, visible, ([1, 3], [2, 2], [1, 0.5]))

    energy = physical_energy([prediction], target, torch.ones(1, 1, 2), context_lambda=0.5)

    l_pred = (1 * 0.0 + 3 * 2.0) / 4  # every masked token counts once
    l_ctx = (1.0 * 1.0 + 0.5 * 1.0) / 4  # Σ (1/√d)·error over the visible tokens / their number
    assert energy.item() == pytest.approx(l_pred + 0.5 * l_ctx, rel=1e-4)
    assert physical_energy([prediction], target, torch.ones(1, 1, 2), 0.0).item() == (
        pytest.approx(l_pred, rel=1e-4)
    )


def test_physical_energy_ignores_boxes_without_confidence_and_averages_masks() -> None:
    target = torch.randn(1, 1, 2, 4, generator=torch.Generator().manual_seed(0))
    exact = functional.layer_norm(target, (4,))
    wrong = exact.clone()
    wrong[0, 0, 1] += 9.0  # the box with confidence 0
    good = _prediction(wrong, wrong, ([1, 1], [1, 1], [1, 1]))
    off = _prediction(exact + 1.0, exact, ([1, 1], [1, 1], [1, 1]))
    confidence = torch.tensor([[[1.0, 0.0]]])

    assert physical_energy([good], target, confidence, 0.5).item() == pytest.approx(0.0, abs=1e-5)
    assert physical_energy([good, off], target, confidence, 0.5).item() == pytest.approx(0.5)


def test_semantic_and_free_energy() -> None:
    target = torch.tensor([[1.0, 0.0]])
    hypotheses = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # one orthogonal, one exact

    assert semantic_energy(target, target).item() == pytest.approx(0.0)
    assert free_energy(hypotheses, target, relaxation=0.0).item() == pytest.approx(0.0)
    assert free_energy(hypotheses, target, relaxation=0.5).item() == pytest.approx(0.25)


def test_uniformity_is_lower_for_spread_points() -> None:
    generator = torch.Generator().manual_seed(0)
    spread = torch.randn(256, 16, generator=generator)
    clumped = torch.ones(256, 16) + 0.01 * torch.randn(256, 16, generator=generator)

    assert uniformity(spread, 2.0) < uniformity(clumped, 2.0)


def test_infonce_does_not_use_clips_of_the_same_video_as_negatives() -> None:
    loss = InfoNCE(0.07)
    video = torch.eye(3)
    text = torch.eye(3)
    text[1] = text[0]  # clip 1 has clip 0's caption, from the same video

    same_video = loss(video, text, ["a", "a", "b"])
    other_videos = loss(video, text, ["a", "c", "b"])

    assert same_video < other_videos


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("A0", {"e_sem"}),
        ("A", {"e_sem", "sigreg_sem"}),
        ("B0", {"e_sem", "unif"}),
        ("B", {"e_sem", "unif", "sigreg_sem"}),
        ("C", {"infonce", "sigreg_sem"}),
    ],
)
def test_each_arm_has_its_terms_and_the_documented_weighting(arm: str, expected: set[str]) -> None:
    objective = Objective(LossSettings(arm=arm), SemanticSettings(), physical=True)
    generator = torch.Generator().manual_seed(0)
    predicted, text = torch.randn(8, 1, 16), torch.randn(8, 16)

    terms = objective.semantic_terms(predicted, text, [str(i) for i in range(8)], generator)
    combined = objective.combine({**terms, "e_fis": torch.tensor(1.0)})

    assert set(terms) == expected
    predictive = sum(v for k, v in combined.parts.items() if not k.startswith("sigreg"))
    regular = sum(
        (v for k, v in combined.parts.items() if k.startswith("sigreg")), torch.tensor(0.0)
    )
    assert combined.total.item() == pytest.approx((0.95 * predictive + 0.05 * regular).item())


def test_sigreg_is_applied_to_each_modality_with_the_same_directions() -> None:
    objective = Objective(LossSettings(arm="A"), SemanticSettings(), physical=True)
    predicted, text = torch.randn(8, 4, 16), torch.randn(8, 16)

    terms = objective.semantic_terms(predicted, text, list("abcdefgh"), torch.Generator())

    directions = random_directions(16, 1024, generator=torch.Generator())
    each = [SIGReg()(rows, directions) for rows in (predicted.flatten(0, 1), text)]
    assert terms["sigreg_sem"].item() == pytest.approx(0.5 * (each[0] + each[1]).item())
