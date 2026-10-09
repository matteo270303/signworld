"""The report of a run and the ablation tables, on run directories written by hand."""

import json
from pathlib import Path
from typing import Any

import pytest

from signworld.experiment.evaluation.compare import (
    compare_runs,
    markdown,
    paired_bootstrap,
    place,
    relation,
    verdict,
)
from signworld.experiment.evaluation.report import analyse_run, triage, write_report
from signworld.experiment.train.config import load_config
from signworld.experiment.train.provenance import record_launch
from tests.worldsign import ABLATIONS, BASE


def _run(directory: Path, *overlays: str, decision: float = 0.1) -> Path:
    config = load_config(BASE, *(ABLATIONS / name for name in overlays))
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(config.model_dump_json(indent=2))
    record_launch(directory, config, resumed=False, world_size=2, clips_per_gpu=64)
    records: list[dict[str, Any]] = [
        {"kind": "stage", "step": 0, "stage": "P"},
        *(
            {"kind": "step", "step": s, "loss": 2.0 - s / 100, "e_sem": 1.0 - s / 200}
            for s in range(100)
        ),
        {"kind": "validation", "step": 50, "decision": decision / 2, "t2v_r1": 0.05},
        {"kind": "validation", "step": 99, "decision": decision, "t2v_r1": 0.1},
        {
            "kind": "alarm",
            "step": 40,
            "name": "hubness_t2v",
            "value": 2.0,
            "threshold": 1.5,
            "meaning": "a few items attract the queries",
            "stop": False,
        },
        {
            "kind": "stop",
            "step": 20,
            "name": "F1",
            "passed": False,
            "criteria": {"R@1 > 5x chance": {"ok": False, "detail": "0.001"}},
        },
    ]
    (directory / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return directory


def _evaluation(path: Path, t2v: list[int], v2t: list[int]) -> Path:
    payload = {
        "measures": {"t2v_r1": 0.2, "y_dist_uniformity_gap": 0.01, "keypoint_error_model": 0.3},
        "gate": None,
        "plausibility": [],
        "language_probes": {"semantic": 0.5},
        "rows": list(range(len(t2v))),
        "ranks": {"t2v": t2v, "v2t": v2t},
    }
    path.write_text(json.dumps(payload))
    return path


def test_the_report_reads_identity_training_alarms_and_evaluation(tmp_path: Path) -> None:
    run = _run(tmp_path / "A", "arm_A.yaml")
    evaluation = _evaluation(tmp_path / "A.json", [0, 1, 0, 5], [0, 0, 2, 0])

    report = analyse_run(run, evaluation, split="test_channel")
    paths = write_report(report, run)

    identity = report["identity"]
    assert identity["arm"] == "A" and identity["physical_target"] == "pose"
    assert len(identity["launches"]) == 1 and identity["launches"][0]["torch"]
    assert (
        report["training"]["terms"]["e_sem"]["first"] > report["training"]["terms"]["e_sem"]["last"]
    )
    assert report["validation"]["best"]["step"] == 99
    alarm = report["alarms"][0]
    assert alarm["name"] == "hubness_t2v" and "ESP-4" in alarm["adjust"]  # from the triage
    assert report["stops"][0]["failed"] == ["R@1 > 5x chance"]
    assert report["evaluation"]["distribution"] == {"y_dist_uniformity_gap": 0.01}
    assert all(p.is_file() for p in paths)
    assert "## Diagnosis" in paths[1].read_text()


def test_triage_takes_the_longest_prefix() -> None:
    found = triage("excluded_part2")

    assert found is not None and "box threshold" in found.adjust
    assert triage("no_such_alarm") is None


def test_the_place_of_a_run_comes_from_its_configuration() -> None:
    def of(*overlays: str) -> str:
        config = load_config(BASE, *(ABLATIONS / name for name in overlays))
        return place(json.loads(config.model_dump_json()))

    assert of("arm_V.yaml") == "V"
    assert of("arm_A.yaml", "esp6_video_target.yaml") == "ESP-6"
    assert of("arm_A.yaml", "esp2_no_physical.yaml") == "ESP-2"
    assert of("arm_A.yaml", "g1_global.yaml") == "G1"
    assert of("arm_A.yaml", "d4_seed1.yaml") == "D4"


def test_the_paired_bootstrap_cancels_the_clips() -> None:
    better = {"t2v": [0] * 50 + [9] * 50, "v2t": [0] * 50 + [9] * 50}
    worse = {"t2v": [0] * 40 + [9] * 60, "v2t": [0] * 40 + [9] * 60}

    delta, low, high = paired_bootstrap(better, worse)

    assert delta == pytest.approx(0.1)
    assert 0 < low <= delta <= high
    assert relation(delta, low, high, 0.4) == ">"
    assert relation(0.3, 0.2, 0.4, 0.4) == "≫"
    assert relation(0.01, -0.02, 0.04, 0.4) == "≈"
    assert verdict(">", "≫") == "as predicted" and verdict("≈", ">") == "against the prediction"
    assert verdict(None, "≈") == "no prediction fixed in advance"


def test_theta_is_chosen_on_validation_and_the_plan_is_read(tmp_path: Path) -> None:
    a = _run(tmp_path / "A", "arm_A.yaml", decision=0.3)
    v = _run(tmp_path / "V", "arm_V.yaml", decision=0.2)
    esp2 = _run(tmp_path / "ESP2", "arm_A.yaml", "esp2_no_physical.yaml", decision=0.1)
    good = [0] * 30 + [9] * 10
    poor = [0] * 10 + [9] * 30
    runs = [
        (a, _evaluation(tmp_path / "a.json", good, good)),
        (v, _evaluation(tmp_path / "v.json", poor, poor)),
        (esp2, _evaluation(tmp_path / "e.json", poor, good)),
    ]

    comparison = compare_runs(runs)

    assert comparison["theta"] == "A"
    found = {(r["first"], r["second"]): r for r in comparison["results"]}
    assert found[("A", "V")]["relation"] == "≫"
    assert found[("θ*", "ESP-2")]["verdict"] == "as predicted"
    assert "θ* = A" in markdown(comparison)


def test_runs_on_different_clips_are_refused(tmp_path: Path) -> None:
    a = _run(tmp_path / "A", "arm_A.yaml")
    v = _run(tmp_path / "V", "arm_V.yaml")
    first = _evaluation(tmp_path / "a.json", [0, 1], [0, 1])
    second = _evaluation(tmp_path / "v.json", [0, 1, 2], [0, 1, 2])

    with pytest.raises(ValueError, match="same clips"):
        compare_runs([(a, first), (v, second)])
