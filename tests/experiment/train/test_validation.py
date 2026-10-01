"""Retrieval measures on rankings worked out by hand, and the measures of a set of clips."""

import pytest
import torch

from signworld.experiment.train.config import WorldSignConfig
from signworld.experiment.train.validation import ranked, retrieval
from signworld.metrics.measures import Collected, split_measures


def test_every_ranking_measure_on_a_ranking_worked_out_by_hand() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.3],  # match 0 first: rank 1
            [0.8, 0.1, 0.7, 0.2],  # matches 2 and 3, first at rank 2
            [0.1, 0.2, 0.3, 0.4],  # match 0 last: rank 4
        ]
    )
    relevance = torch.tensor(
        [[True, False, False, False], [False, False, True, True], [True, False, False, False]]
    )

    out = ranked(scores, relevance, "t2v")

    assert out["t2v_r1"] == pytest.approx(1 / 3)
    assert out["t2v_r5"] == pytest.approx(1.0)
    assert out["t2v_precision1"] == pytest.approx(1 / 3)
    assert out["t2v_precision5"] == pytest.approx((1 / 5 + 2 / 5 + 1 / 5) / 3)
    assert out["t2v_recall1"] == pytest.approx(1 / 3)
    assert out["t2v_recall5"] == pytest.approx(1.0)
    assert out["t2v_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)
    assert out["t2v_medr"] == 2.0


def test_retrieval_splits_r1_by_duration_and_caption_length() -> None:
    texts = torch.eye(4, 8)
    predicted = texts[:, None].clone()
    predicted[2, 0] = texts[2] + 0.5 * texts[3]  # closer to the fourth caption than ...
    predicted[3, 0] = 0.3 * texts[0] + 0.1 * texts[3]  # ... the fourth clip: its query misses

    scores = retrieval(
        predicted,
        texts,
        torch.arange(4),
        durations=torch.tensor([1.0, 3.0, 6.0, 12.0]),
        words=torch.tensor([3, 8, 15, 30]),
        bootstrap=("t2v_r1",),
    )
    out = scores.as_log()

    assert scores.t2v[1] == pytest.approx(0.75)
    assert out["t2v_r1_duration_0-2"] == 1.0 and out["t2v_r1_duration_10+"] == 0.0
    assert out["t2v_r1_words_1-6"] == 1.0 and out["t2v_r1_words_21+"] == 0.0
    assert "t2v_r1_low" in out and "v2t_r1_low" not in out


def test_alignment_is_zero_and_geometry_is_read_when_predictions_equal_captions() -> None:
    generator = torch.Generator().manual_seed(0)
    texts = torch.randn(64, 16, generator=generator)
    collected = Collected(
        tensors={
            "predicted": texts[:, None].clone(),
            "texts": texts,
            "captions": torch.randn(64, 32, generator=generator),
            "rows": torch.arange(64),
            "languages": torch.zeros(64, dtype=torch.long),
            "e_sem": torch.zeros(64),
        },
        averages={"loss_total": 1.5, "r2_visible": 0.5},
    )

    config = WorldSignConfig.model_validate(
        {"name": "measures", "encoder": {"hub_repo": ".", "checkpoint": "."}}
    )

    scores, out = split_measures(collected, ("en",), config, physical_prefix="val_")

    assert scores.t2v[1] == 1.0
    assert out["alignment"] == pytest.approx(0.0, abs=1e-6)
    assert out["uniformity_video"] == pytest.approx(out["uniformity_text"], abs=1e-5)
    assert out["loss_total"] == 1.5 and out["val_r2_visible"] == 0.5
    assert out["val_e_sem_en"] == 0.0
    assert {"y_effective_rank", "y_isoscore", "y_condition_number", "text_sigreg"} <= set(out)
