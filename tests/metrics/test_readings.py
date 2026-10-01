"""Collaudo «Testare i test» (§4.13.1): every diagnostic reading on a case with a known answer."""

import pytest
import torch
from torch import nn

from signworld.worldmodel.lora import LoRALinear
from signworld.worldmodel.physical import PhysicalPrediction
from signworld.worldmodel.readings import (
    Spread,
    _temporal_baselines,
    attention_readings,
    energy_table,
    linear_cka,
    lora_ratios,
    modality_gap,
    physical_readings,
    query_cosine,
    shares_and_cosines,
    sigreg_ratio,
    text_head_spearman,
    variance_ratio,
    weighted_r2,
)


def _normal(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_r2_and_variance_ratio_separate_a_copy_from_the_mean() -> None:
    target = _normal(500, 8)
    mean = target.mean(0).expand_as(target)
    ones = torch.ones(500)

    assert weighted_r2(target, target, ones) == pytest.approx(1.0)
    assert weighted_r2(mean, target, ones) == pytest.approx(0.0, abs=1e-6)
    assert variance_ratio(target, target) == pytest.approx(1.0)
    assert variance_ratio(mean, target) == pytest.approx(0.0, abs=1e-6)  # predicting the mean


def test_spread_and_sigreg_tell_isotropic_from_collapsed() -> None:
    isotropic = _normal(2000, 16)
    collapsed = _normal(2000, 1) * torch.ones(1, 16) + 0.01 * _normal(2000, 16, seed=1)
    mixture = torch.cat([_normal(1000, 16) * 0.3 + 3, _normal(1000, 16, seed=2) * 0.3 - 3])

    assert Spread.of(isotropic).effective_rank > 5 * Spread.of(collapsed).effective_rank
    assert Spread.of(isotropic).isoscore > 0.9 > Spread.of(collapsed).isoscore
    assert sigreg_ratio(isotropic) == pytest.approx(1.0, rel=0.3)
    assert sigreg_ratio(mixture) > 10


def test_cka_is_one_for_the_same_set_and_low_for_unrelated_ones() -> None:
    x = _normal(400, 32)
    rotation, _ = torch.linalg.qr(_normal(32, 32, seed=3))

    assert linear_cka(x, x @ rotation) == pytest.approx(1.0)
    assert linear_cka(x, _normal(400, 32, seed=4)) < 0.2


def test_query_collapse_and_text_head_correlation() -> None:
    same = torch.ones(4, 8, 16)
    distinct = torch.eye(16)[:8].expand(4, -1, -1)
    captions = _normal(50, 16)

    assert query_cosine(same) == pytest.approx(1.0)
    assert query_cosine(distinct) == pytest.approx(0.0)
    assert text_head_spearman(captions, captions * 2.0) == pytest.approx(1.0)


def test_the_modality_classifier_finds_a_gap_and_only_a_gap() -> None:
    video = _normal(400, 16)
    text = _normal(400, 16, seed=5)

    assert modality_gap(video, text + 1.0) > 0.95
    assert modality_gap(video, text) < 0.65


def test_energy_table_and_attention_readings() -> None:
    table = energy_table(torch.arange(8.0), torch.tensor([0, 7, 1, 6, 2, 5, 3, 4.0]))
    uniform = torch.full((1, 1, 2, 16), 1 / 16)
    members = torch.zeros(1, 1, 1, 4, 4)
    members[0, 0, 0, :2, :2] = 1.0  # 4 of 16 tokens inside a box
    focused = torch.zeros(1, 1, 2, 16)
    focused[..., 0] = 1.0

    assert sum(table.values()) == pytest.approx(1.0)
    readings = attention_readings(uniform, members)
    assert readings["attention_entropy"] == pytest.approx(1.0)
    assert readings["attention_on_articulators"] == pytest.approx(0.25)
    assert attention_readings(focused, members)["attention_on_articulators"] == pytest.approx(1.0)


def test_gradient_shares_and_cosines() -> None:
    gradients = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([3.0, 0.0]),
        "c": torch.tensor([0.0, -2.0]),
    }

    readings = shares_and_cosines(gradients, "lora")

    assert readings["lora_share_b"] == pytest.approx(0.5)
    assert readings["lora_cos_a_b"] == pytest.approx(1.0)
    assert readings["lora_cos_a_c"] == pytest.approx(0.0)


def test_a_fresh_lora_has_moved_nothing() -> None:
    layer = nn.Module()
    layer.adapted = LoRALinear(nn.Linear(8, 8), rank=2, alpha=2.0)

    assert lora_ratios(layer) == {"adapted": 0.0}
    with torch.no_grad():
        layer.adapted.up[0].fill_(1.0)
    assert lora_ratios(layer)["adapted"] > 0


def test_physical_readings_of_a_perfect_predictor() -> None:
    latent = _normal(2, 8, 4, 6)
    normalized = nn.functional.layer_norm(latent, (6,))
    hidden = torch.zeros(2, 8, 4)
    hidden[:, 4:] = 5.0  # the second half of the steps is masked
    prediction = PhysicalPrediction(normalized, normalized, hidden, 5.0 - hidden, 5.0 - hidden)

    readings = physical_readings([prediction], latent, torch.ones(2, 8, 4))

    assert readings["r2_masked"] == pytest.approx(1.0)
    assert readings["r2_visible"] == pytest.approx(1.0)
    assert readings["gamma_masked"] == pytest.approx(1.0)
    assert readings["dynamics_r2"] > readings["dynamics_baseline_r2"]
    assert readings["excluded_part0"] == 0.0


def test_the_keypoint_baselines_are_exact_on_uniform_motion() -> None:
    steps = torch.arange(10.0)
    truth = torch.stack([steps, 2 * steps], dim=-1)[None, :, None]  # (1, 10, 1, 2)
    seen = torch.zeros(1, 10, 1, dtype=torch.bool)
    seen[0, [0, 1, 5, 9]] = True

    interpolated, extrapolated = _temporal_baselines(truth, seen)

    assert torch.allclose(interpolated, truth)
    assert torch.allclose(extrapolated[0, 2:5], truth[0, 2:5])  # from steps 0 and 1
