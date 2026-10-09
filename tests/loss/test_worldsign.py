"""The energies and the objective of every arm."""

import pytest
import torch
from torch.nn import functional

from signworld.experiment.train.config import LossSettings, SemanticSettings
from signworld.loss.sigreg import SIGReg, random_directions
from signworld.loss.worldsign import (
    FalseNegatives,
    InfoNCE,
    Objective,
    free_energy,
    physical_energy,
    semantic_energy,
    uniformity,
    vicreg_covariance,
    vicreg_variance,
)
from signworld.models.worldsign.physical import PhysicalPrediction


def _prediction(state: torch.Tensor, counts: tuple[list[float], ...]) -> PhysicalPrediction:
    """``counts``: masked tokens, visible tokens and their summed weights, per step."""
    masked, visible, weight = (torch.tensor(c).view(1, -1) for c in counts)
    boxes = (masked + visible)[..., None].expand(-1, -1, 4) / 4
    return PhysicalPrediction(state, masked, visible, weight, boxes)


def test_physical_energy_is_v_jepa_2_1_loss_read_per_step() -> None:
    pattern = torch.tensor([1.0, -1.0, 1.0, -1.0])  # zero mean, unit variance
    target = (3.0 * pattern + 5.0).expand(1, 2, 4)  # its layer norm is the pattern
    state = torch.stack([pattern, pattern + 2.0])[None]  # errors 0 and 2 per channel
    prediction = _prediction(state, ([1, 3], [2, 2], [1, 0.5]))

    energy = physical_energy([prediction], target, torch.ones(1, 2), context_lambda=0.5)

    # a step weighs n_masked + λ·w_visible: 1 + 0.5·1 and 3 + 0.5·0.5
    assert energy.item() == pytest.approx((1.5 * 0.0 + 3.25 * 2.0) / 4.75, rel=1e-4)
    assert physical_energy([prediction], target, torch.ones(1, 2), 0.0).item() == (
        pytest.approx((1 * 0.0 + 3 * 2.0) / 4, rel=1e-4)
    )


def test_physical_energy_ignores_steps_without_confidence_and_averages_masks() -> None:
    target = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(0))
    exact = functional.layer_norm(target, (4,))
    wrong = exact.clone()
    wrong[0, 1] += 9.0  # the step with confidence 0
    good = _prediction(wrong, ([1, 1], [1, 1], [1, 1]))
    off = _prediction(exact + 1.0, ([1, 1], [1, 1], [1, 1]))
    confidence = torch.tensor([[1.0, 0.0]])

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


RELATIVE = {"unif": 1 / 3, "vicreg_var": 25 / (2 * 25), "vicreg_cov": 1 / (2 * 25)}
"""Weights against E_sem from the literature: Wang and Isola's 0.75·L_align + 0.5·L_unif
with L_align = 2·E_sem; VICReg's 25/25/1 with its MSE = 2·E_sem."""


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("A0", {"e_sem"}),
        ("A", {"e_sem", "sigreg_sem"}),
        ("V", {"e_sem", "vicreg_var", "vicreg_cov"}),
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
    w = LossSettings().sigreg_weight
    total = sum(
        (w if name.startswith("sigreg") else (1 - w) * RELATIVE.get(name, 1.0)) * value
        for name, value in combined.parts.items()
    )
    assert combined.total.item() == pytest.approx(float(total), rel=1e-6)


def test_vicreg_terms_are_zero_on_whitened_points_and_large_on_collapsed_ones() -> None:
    white = torch.randn(4096, 8, generator=torch.Generator().manual_seed(0))
    collapsed = torch.ones(4096, 8) + 1e-3 * white

    assert vicreg_variance(white, 1.0, 1e-4).item() == pytest.approx(0.0, abs=0.02)
    assert vicreg_covariance(white).item() == pytest.approx(0.0, abs=0.01)
    assert vicreg_variance(collapsed, 1.0, 1e-4).item() == pytest.approx(0.99, abs=0.01)
    correlated = torch.cat([white[:, :1]] * 8, dim=1)
    assert vicreg_covariance(correlated).item() == pytest.approx(7.0, rel=0.05)  # 56 pairs / 8


def test_infonce_with_one_caption_per_clip_is_clip_s_loss() -> None:
    loss = InfoNCE(0.07)
    video, text = torch.randn(5, 8), torch.randn(5, 8)
    logits = functional.normalize(video, dim=-1) @ functional.normalize(text, dim=-1).T / 0.07
    labels = torch.arange(5)
    clip = 0.5 * (
        functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)
    )

    assert loss(video, text, list("abcde")).item() == pytest.approx(clip.item(), rel=1e-5)


def test_infonce_spreads_the_target_over_clips_with_the_same_caption() -> None:
    loss = InfoNCE(0.07)
    text = torch.randn(3, 8)
    text[1] = text[0]  # clips 0 and 1 share a caption, from different videos
    video = text + 0.1 * torch.randn(3, 8)  # each clip close to its caption
    rows = torch.tensor([0, 0, 1])
    logits = loss.logits(video, text)

    multi = loss(video, text, ["a", "b", "c"], rows)

    log_p = functional.log_softmax(logits, dim=1)
    log_q = functional.log_softmax(logits.T, dim=1)
    v2t = -(0.5 * (log_p[0, 0] + log_p[0, 1]) + 0.5 * (log_p[1, 0] + log_p[1, 1]) + log_p[2, 2])
    t2v = -(0.5 * (log_q[0, 0] + log_q[0, 1]) + 0.5 * (log_q[1, 0] + log_q[1, 1]) + log_q[2, 2])
    assert multi.item() == pytest.approx((0.5 * (v2t + t2v) / 3).item(), rel=1e-5)


def test_infonce_drops_false_negatives_but_never_a_positive() -> None:
    loss = InfoNCE(0.07)
    video, text = torch.randn(3, 8), torch.randn(3, 8)
    excluded = torch.zeros(3, 3, dtype=torch.bool)
    excluded[0, 2] = excluded[2, 0] = True
    excluded[1, 1] = True  # a positive: kept whatever the mask says

    positive, dropped = loss.pairs(["a", "b", "c"], None, excluded, "cpu")

    assert dropped[0, 2] and dropped[2, 0] and not dropped[1, 1]
    logits = loss.logits(video, text)
    log_p = functional.log_softmax(logits.masked_fill(dropped, float("-inf")), dim=1)
    assert torch.isfinite(log_p[positive]).all()


def test_the_temperature_has_clip_s_floor() -> None:
    loss = InfoNCE(0.001, min_temperature=0.01)

    assert loss.temperature.item() == pytest.approx(0.01)


def test_false_negatives_follow_each_caption_s_threshold_both_ways() -> None:
    captions = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0]])
    rows = torch.tensor([0, 1, 2, 0])
    thresholds = torch.tensor([0.99, 0.5, 0.5])  # row 0 strict, row 1 loose

    mask = FalseNegatives(thresholds).mask(captions, rows)

    assert mask[0, 1] and mask[1, 0]  # 0.99 cosine: row 1 flags row 0, so both drop
    assert not mask[0, 2] and not mask[2, 0]  # orthogonal
    assert not mask[0, 3]  # the same caption: a positive, not a false negative


def test_sigreg_is_applied_to_each_modality_with_the_same_directions() -> None:
    objective = Objective(LossSettings(arm="A"), SemanticSettings(), physical=True)
    predicted, text = torch.randn(8, 4, 16), torch.randn(8, 16)

    terms = objective.semantic_terms(predicted, text, list("abcdefgh"), torch.Generator())

    directions = random_directions(16, 1024, generator=torch.Generator())
    each = [SIGReg()(rows, directions) for rows in (predicted.flatten(0, 1), text)]
    assert terms["sigreg_sem"].item() == pytest.approx(0.5 * (each[0] + each[1]).item())
