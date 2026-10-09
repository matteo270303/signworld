import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from signworld.data.corpus.manifest import SCHEMA, Clip
from signworld.experiment.collaudo.contamination import (
    contamination_report,
    duplicates_across_splits,
)
from signworld.experiment.collaudo.durations import duration_report, summarize
from signworld.experiment.collaudo.report import write_report


def _table(*clips: Clip) -> pa.Table:
    return pa.Table.from_pylist(
        [{field: getattr(clip, field) for field in SCHEMA.names} for clip in clips], schema=SCHEMA
    )


def _clip(
    clip_id: str,
    video_id: str,
    start: float,
    end: float,
    caption: str = "caption",
    *,
    split: str | None = None,
    sign_language: str | None = "ase",
) -> Clip:
    return Clip(clip_id, video_id, start, end, caption, "en", sign_language, None, split, True)


def test_duration_summary_reports_quantiles_spacing_and_invalid_clips() -> None:
    summary = summarize(np.array([1.0, 2.0, 3.0, 4.0, 0.0]))

    assert summary.clips == 5
    assert summary.invalid == 1
    assert summary.quantiles_s["p50"] == pytest.approx(2.5)
    assert summary.max_spacing_s["p50"] == pytest.approx(2.5 / 32)


def test_duration_report_groups_by_language() -> None:
    manifest = _table(
        _clip("a", "v", 0, 2, sign_language="ase"),
        _clip("b", "v", 0, 4, sign_language="ase"),
        _clip("c", "w", 0, 8, sign_language=None),
    )

    report = duration_report(manifest)

    assert set(report.by_sign_language) == {"ase", "unknown"}
    assert report.by_sign_language["unknown"].max_s == 8


def test_contamination_excludes_overlaps_within_the_margin_and_repeated_captions() -> None:
    benchmark = _table(
        _clip("eval", "shared", 10.0, 12.0, "The storm arrives tonight.", split="test"),
        _clip("train", "shared", 40.0, 42.0, "Unrelated training sentence.", split="train"),
    )
    corpus = _table(
        _clip("inside-margin", "shared", 13.5, 15.0, "Something else entirely here"),
        _clip("outside-margin", "shared", 14.5, 16.0, "Another unrelated caption"),
        _clip("same-caption", "shared", 100.0, 102.0, "the storm arrives tonight"),
        _clip("training-overlap", "shared", 40.0, 42.0, "Unrelated training sentence."),
        _clip("other-video", "elsewhere", 10.0, 12.0, "The storm arrives tonight."),
    )

    report = contamination_report(corpus, benchmark, margin_s=2.0)

    assert report.shared_videos == 1
    assert report.excluded == {"inside-margin", "same-caption"}


def test_duplicates_across_splits_need_same_caption_and_duration() -> None:
    manifest = _table(
        _clip("a", "v1", 0.0, 2.00, "Hello there!", split="train"),
        _clip("b", "v2", 5.0, 7.02, "hello there", split="test"),
        _clip("c", "v3", 0.0, 3.00, "Hello there!", split="valid"),
        _clip("d", "v4", 0.0, 2.00, "Hello there!", split="train"),
    )

    groups = duplicates_across_splits(manifest, duration_tolerance_s=0.1)

    assert [(group.clip_ids, group.splits) for group in groups] == [
        (["a", "b", "d"], ["test", "train"])
    ]


def test_reports_serialise_dataclasses_and_sets(tmp_path: Path) -> None:
    report = contamination_report(_table(), _table())

    path = write_report(tmp_path / "report.json", "contamination", report)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["check"] == "contamination"
    assert payload["result"]["excluded"] == []
