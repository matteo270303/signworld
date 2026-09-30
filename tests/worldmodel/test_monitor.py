"""The alarms and stops of the monitor, on readings with a known verdict (§4.13.3, §4.13.5)."""

from pathlib import Path
from typing import Any

import pytest

from signworld.worldmodel.config import load_config
from signworld.worldmodel.monitor import Monitor, RunStoppedError

from .conftest import BASE


class _Log:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, kind: str, **values: Any) -> None:
        self.records.append({"kind": kind, **values})


def _monitor() -> tuple[Monitor, _Log]:
    log = _Log()
    return Monitor(None, load_config(BASE), [], log), log  # type: ignore[arg-type]


def test_a_collapse_is_read_against_step_0() -> None:
    monitor, log = _monitor()
    monitor._record("frequent", 0, {"s_rank_left": 40.0, "gamma_sem": 0.9})
    monitor._record("frequent", 800, {"s_rank_left": 30.0, "gamma_sem": 0.9})
    assert not [r for r in log.records if r["kind"] == "alarm"]

    monitor._record("frequent", 1600, {"s_rank_left": 15.0, "gamma_sem": 0.2})

    alarms = {r["name"] for r in log.records if r["kind"] == "alarm"}
    assert alarms == {"s_rank_left", "gamma_sem"}


def test_a_conflict_must_last_before_it_raises_an_alarm() -> None:
    monitor, log = _monitor()
    for step in range(4):
        monitor._record("frequent", step * 800, {"video_cos_physical_semantic": -0.5})
    assert not [r for r in log.records if r["kind"] == "alarm"]

    monitor._record("frequent", 3200, {"video_cos_physical_semantic": -0.5})

    assert [r["name"] for r in log.records if r["kind"] == "alarm"] == [
        "video_cos_physical_semantic"
    ]


def test_a_leak_stops_the_run() -> None:
    monitor, _ = _monitor()

    with pytest.raises(RunStoppedError, match="leak"):
        monitor._record("validation", 4000, {"leak_change": 0.3})


def test_stop_f2_reads_the_latest_values_and_f3_extrapolates() -> None:
    monitor, log = _monitor()
    monitor.total_steps = 100_000
    for step, value in ((4000, 0.05), (8000, 0.08), (16000, 0.11)):
        monitor._record("validation", step, {"decision": value, "chance": 0.001, "noise_drop": 0.9})
    monitor._record("frequent", 16000, {"gamma_sem": 0.6, "query_cosine": 0.4})

    report = monitor.stop_point("F2", 16000)
    extrapolated = monitor.extrapolate()

    assert report.passed, report.criteria
    assert extrapolated is not None and extrapolated > 0.11  # still rising, log-linearly
    assert [r["name"] for r in log.records if r["kind"] == "stop"] == ["F2"]


def test_the_gate_overlay_turns_the_stops_on(tmp_path: Path) -> None:
    config = load_config(BASE, BASE.parent / "gate.yaml")

    assert config.diagnostics.gate_stops and not load_config(BASE).diagnostics.gate_stops
