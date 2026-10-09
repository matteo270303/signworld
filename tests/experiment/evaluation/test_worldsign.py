"""The evaluation: manipulations of §4.12.4, probes and the whole report on tiny modules."""

from pathlib import Path

import pytest
import torch

from signworld.data.loaders import Collate
from signworld.experiment.evaluation.worldsign import evaluate, language_probe
from signworld.metrics.directions import Bidirectional
from signworld.models.worldsign.model import WorldSignBatch
from signworld.models.worldsign.plausibility import (
    change_colour,
    freeze_steps,
    reverse_time,
    shift_pose,
    skip_steps,
)
from tests.worldsign import needs_hub, synthetic_corpus, tiny_training


def _batch() -> WorldSignBatch:
    """Frames and pose that both carry their step index, so a manipulation's order shows."""
    steps = torch.arange(32.0)
    frames = torch.arange(64).div(2, rounding_mode="floor").to(torch.uint8)
    frames = frames[None, :, None, None, None].expand(2, 64, 4, 4, 3).clone()
    tokens = torch.zeros(2, 32, 69, 6)
    tokens[..., 0] = steps[:, None]  # x of the first frame of the step
    tokens[..., 3] = steps[:, None] + 0.5  # x of the second frame
    keypoints = steps[None, :, None, None].expand(2, 32, 69, 2).clone()
    return WorldSignBatch(
        frames=frames,
        pose_tokens=tokens,
        keypoints=keypoints,
        keypoint_weights=torch.ones(2, 32, 69),
        boxes=steps[None, :, None, None].expand(2, 32, 4, 4).clone(),
        box_visible=torch.ones(2, 32, 4, dtype=torch.bool),
        captions=torch.zeros(2, 768),
        languages=torch.zeros(2, dtype=torch.long),
        videos=["a", "b"],
    )


def _step_of_frames(batch: WorldSignBatch) -> torch.Tensor:
    return batch.frames[0, ::2, 0, 0, 0].float()


def test_reversal_plays_video_and_pose_backwards_together() -> None:
    batch = reverse_time(_batch())

    assert torch.equal(_step_of_frames(batch), torch.arange(31.0, -1, -1))
    assert torch.equal(batch.pose_tokens[0, :, 0, 3], torch.arange(31.0, -1, -1))  # swapped
    assert torch.equal(batch.pose_tokens[0, :, 0, 0], torch.arange(31.0, -1, -1) + 0.5)
    assert torch.equal(batch.keypoints[0, :, 0, 0], torch.arange(31.0, -1, -1))


def test_skipped_and_frozen_steps_keep_video_and_pose_aligned() -> None:
    skipped, frozen = skip_steps(_batch(), 12, 8), freeze_steps(_batch(), 12, 8)

    for batch in (skipped, frozen):
        assert torch.equal(_step_of_frames(batch), batch.pose_tokens[0, :, 0, 0])
        assert batch.frames.shape == (2, 64, 4, 4, 3)
    assert skipped.pose_tokens[0, 12, 0, 0] == 20 and skipped.pose_tokens[0, -1, 0, 0] == 31
    assert torch.equal(frozen.pose_tokens[0, 12:20, 0, 0], torch.full((8,), 12.0))


def test_controls_touch_one_stream_only() -> None:
    shifted, brighter = shift_pose(_batch(), 4), change_colour(_batch(), 1.5)

    assert torch.equal(_step_of_frames(shifted), torch.arange(32.0))  # video as it was
    assert shifted.pose_tokens[0, 0, 0, 0] == 4  # pose four steps ahead
    assert torch.equal(shifted.boxes, _batch().boxes)  # boxes follow the video
    assert torch.equal(brighter.pose_tokens, _batch().pose_tokens)
    assert brighter.frames.float().mean() > _batch().frames.float().mean()


def test_the_language_probe_reads_a_planted_language() -> None:
    labels = torch.arange(200) % 2
    features = torch.randn(200, 8, generator=torch.Generator().manual_seed(0))
    features[:, 0] += 4 * labels
    videos = [f"v{i // 2}" for i in range(200)]

    assert language_probe(features, labels, videos) > 0.95
    assert language_probe(torch.randn(200, 8), labels, videos) < 0.7


@needs_hub
def test_the_whole_report_on_tiny_modules(tmp_path: Path) -> None:
    corpus = synthetic_corpus(tmp_path / "corpus")
    model, config, _, validation, _ = tiny_training(tmp_path, corpus)
    batch = Collate(model.text.centering.languages)([validation[i] for i in range(4)])

    report = evaluate(
        model,
        [batch],
        torch.device("cpu"),
        config,
        torch.tensor([0, 1, 0, 1]),
        [f"v{i}" for i in range(4)],
        masks=2,
        ridge_baseline=Bidirectional(t2v=0.01, v2t=0.01),
        gate=True,
    )
    report.write(tmp_path / "report.json")

    measures = report.measures
    assert {"t2v_r1", "v2t_r10", "v2t_r10_low", "t2v_mrr", "t2v_precision5", "v2t_recall10"} <= set(
        measures
    )
    assert {"loss_total", "loss_e_fis", "r2_visible", "keypoint_error_model", "alignment"} <= set(
        measures
    )
    assert {"uniformity_video", "y_isoscore", "text_sigreg", "noise_drop", "leak_change"} <= set(
        measures
    )
    assert {"pose_isoscore", "pose_r2_velocity_face", "order_cosine"} <= set(measures)
    assert report.gate is not None
    assert [r.name for r in report.plausibility] == [
        "time_reversed",
        "steps_skipped",
        "steps_frozen",
        "pose_shifted",
        "colour_changed",
    ]
    assert all(r.mask_variability >= 0 for r in report.plausibility)
    assert set(report.language_probes) == {"pose", "encoder", "semantic"}
    assert (tmp_path / "report.json").is_file()
    assert measures["order_cosine"] == pytest.approx(measures["order_cosine"])  # finite
