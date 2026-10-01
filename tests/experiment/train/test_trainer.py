"""A short run of the whole training loop on tiny real modules and synthetic clips."""

import csv
import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from signworld.worldmodel.checkpoint import CheckpointStore, TrainingState
from signworld.worldmodel.data import Collate
from signworld.worldmodel.model import StepRandomness, WorldSign
from signworld.worldmodel.reporting import COLUMNS, RunReport
from signworld.worldmodel.trainer import Progress, Trainer
from signworld.worldmodel.validation import RetrievalScores

from .conftest import needs_hub, slow, synthetic_corpus, tiny_training


def _setup(directory: Path, teacher: Path | None = None) -> tuple[Any, ...]:
    corpus_directory = directory / "corpus"
    corpus = synthetic_corpus(corpus_directory)
    return tiny_training(directory, corpus, teacher=teacher)


def _records(output: Path) -> list[dict[str, Any]]:
    lines = (output / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@slow
@needs_hub
def test_a_short_run_goes_through_every_stage_and_cools_down(tmp_path: Path) -> None:
    model, config, train, validation, probe = _setup(tmp_path)
    output = tmp_path / "run"

    state = Trainer(
        model, config, train, validation, output, probe_data=probe, train_subset_data=probe
    ).fit()

    assert state.finished and state.cooldown_start is not None
    records = _records(output)
    stages = [r["stage"] for r in records if r["kind"] == "stage"]
    assert stages == ["1a", "1", "2a", "2"]
    assert any(r["kind"] == "cooldown" for r in records)
    steps = [r for r in records if r["kind"] == "step"]
    assert all(torch.isfinite(torch.tensor(r["lr_factor"])) for r in steps)
    assert {"e_fis", "anchor", "sigreg_posa", "e_sem", "sigreg_sem"} <= set(steps[-1])
    for name in ("best", "latest", "final"):
        assert (output / "checkpoints" / f"{name}.pt").is_file()

    # The fail-fast system read everything, from step 0 (§4.13.3-§4.13.5).
    kinds = {r["kind"] for r in records}
    assert {"frequent", "validation", "rare", "stop"} <= kinds
    frequent = [r for r in records if r["kind"] == "frequent"]
    assert frequent[0]["step"] == 0
    assert {"r2_masked", "dynamics_margin", "s_rank_left", "video_share_max"} <= set().union(
        *frequent
    )
    validation_reads = [r for r in records if r["kind"] == "validation"]
    assert {"noise_drop", "hubness", "modality_gap", "leak_change", "cka_left"} <= set(
        validation_reads[-1]
    )
    assert validation_reads[-1]["leak_change"] < 1e-4  # masked pixels never reach the encoder
    assert {"loss_total", "t2v_mrr", "v2t_precision10", "t2v_recall5", "t2v_medr"} <= set(
        validation_reads[-1]
    )
    assert {"alignment", "uniformity_text", "y_effective_rank", "text_condition_number"} <= set(
        validation_reads[-1]
    )
    assert {"val_r2_visible", "val_keypoint_margin", "train_t2v_r1", "gap_v2t_r1"} <= set(
        validation_reads[-1]
    )
    rare = [r for r in records if r["kind"] == "rare"]
    assert "plausibility_time_reversed_increase" in set().union(*rare)
    energies = sorted((output / "checkpoints" / "energies").glob("*.parquet"))
    assert energies  # every training clip's energies, written with each latest checkpoint
    assert {r["name"] for r in records if r["kind"] == "stop"} == {"F1", "F2", "F3"}

    # Killed after the cooldown began: the same command resumes and finishes the same way.
    model, config, train, validation, probe = _setup(tmp_path, teacher=tmp_path / "sjepa.pt")
    resumed = Trainer(
        model, config, train, validation, output, probe_data=probe, train_subset_data=probe
    ).fit()
    assert resumed.finished and resumed.step == state.step


@needs_hub
def test_a_checkpoint_reloads_to_the_same_loss(tmp_path: Path) -> None:
    model, _, train, _, _ = _setup(tmp_path)
    collate = Collate(model.text.centering.languages)
    batch = collate([train[(0, 0)], train[(1, 0)]])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)  # move away from the pre-trained start
    store = CheckpointStore(tmp_path / "checkpoints", trainable)
    store.save("latest", model, optimizer, TrainingState(step=7))

    fresh: WorldSign = _setup(tmp_path, teacher=tmp_path / "sjepa.pt")[0]
    state = store.load("latest", fresh, None)

    def loss(m: WorldSign) -> float:
        m.eval()
        with torch.no_grad():
            return float(m.loss(batch, 3, 10, StepRandomness.at(0, 3)).total)

    assert state.step == 7
    assert loss(fresh) == loss(model)


def test_progress_times_the_steps_apart_from_the_evaluations() -> None:
    progress = Progress()
    progress.advance(0, now=100.0)
    progress.paused = 30.0  # a validation inside the window
    progress.advance(10, now=150.0)
    progress.validations.append(60.0)
    progress.rares.append(120.0)

    assert progress.per_step == pytest.approx(2.0)
    assert progress.eta(steps=100, validations=2, rares=1) == pytest.approx(200 + 120 + 120)


@needs_hub
def test_the_progress_bar_shows_stage_loss_recall_and_the_end(tmp_path: Path) -> None:
    model, config, train, validation, probe = _setup(tmp_path)
    trainer = Trainer(model, config, train, validation, tmp_path / "run", probe_data=probe)
    state = TrainingState(step=2, position=2 * trainer.per_gpu)
    trainer.progress.advance(0, time.perf_counter() - 20.0)
    trainer.progress.advance(2, time.perf_counter())
    recall = {1: 0.25, 5: 0.5, 10: 0.75}
    trainer.latest = (0, RetrievalScores(recall, recall | {1: 0.125}, clips=8))
    trainer.epoch_sums["loss"], trainer.epoch_steps = 3.0, 2

    tail = trainer._postfix(state)

    assert "loss 1.5000" in tail and "R@1 T2V/V2T 0.2500/0.1250 (step 0)" in tail
    assert "end ≈ ?" not in tail  # the pace is known: an end is expected


def _epoch_cadence(config: Any) -> Any:
    diagnostics = config.diagnostics.model_copy(update={"cadence": "epoch"})
    return config.model_copy(update={"diagnostics": diagnostics})


@needs_hub
def test_with_the_epoch_cadence_the_readings_wait_for_the_end_of_each_epoch(
    tmp_path: Path,
) -> None:
    model, config, train, validation, probe = _setup(tmp_path)
    trainer = Trainer(
        model, _epoch_cadence(config), train, validation, tmp_path / "run", probe_data=probe
    )
    per_epoch, per_gpu = trainer.steps_per_epoch, trainer.per_gpu
    last = TrainingState(step=per_epoch - 1, position=(per_epoch - 1) * per_gpu)
    first_of_next = TrainingState(step=per_epoch, epoch=1, position=0)
    ends = sum(0 < e * per_epoch <= trainer.constant_end for e in range(1, 4))

    assert trainer._frequent_due(TrainingState())  # step 0, the reference
    assert trainer._frequent_due(last)
    assert per_epoch == 1 or not trainer._frequent_due(first_of_next)
    # Epoch ends, end of the constant phase and final; rare: epoch ends and final.
    assert trainer._evaluations_left(TrainingState()) == (ends + 2, ends + 1)


@slow
@needs_hub
def test_a_short_run_with_the_epoch_cadence_reads_only_at_the_epoch_ends(tmp_path: Path) -> None:
    model, config, train, validation, probe = _setup(tmp_path)
    output = tmp_path / "run"
    trainer = Trainer(
        model,
        _epoch_cadence(config),
        train,
        validation,
        output,
        probe_data=probe,
        train_subset_data=probe,
    )

    state = trainer.fit()

    records = _records(output)
    per_epoch = trainer.steps_per_epoch
    ends = {epoch * per_epoch for epoch in range(1, 4)}
    allowed = {0, trainer.constant_end, state.step} | ends
    for kind in ("validation", "rare"):
        steps = {r["step"] for r in records if r["kind"] == kind}
        assert 0 in steps and steps <= allowed, (kind, steps)
    frequent = {r["step"] for r in records if r["kind"] == "frequent"}
    assert frequent and all(s == 0 or (s + 1) % per_epoch == 0 for s in frequent), frequent
    assert all(r["step"] in ends for r in records if r["kind"] == "stop")
    assert (output / "checkpoints" / "final.pt").is_file()
    epochs = _table(output / "metrics.csv")
    others = _table(output / "val_steps.csv")
    assert {row["reason"] for row in epochs} <= {"epoch", "final"} and epochs
    assert "step 0" in {row["reason"] for row in others}
    assert all(row["T2V_R@1"] != "" and row["val_loss"] != "" for row in epochs + others)


def _table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with path.open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == COLUMNS
    return rows


def test_the_report_writes_one_row_per_validation_in_fixed_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = RunReport(tmp_path, enabled=True)
    readings = {"loss_total": 2.5, "t2v_r1": 0.25, "v2t_r1": 0.125, "t2v_medr": 3.0}
    common = {"epochs": 3, "total_steps": 30, "stage": "2", "lr": 1e-4, "eta": 600.0}

    report.validation(reason="step 0", epoch=0.0, step=0, train={}, readings=readings, **common)
    report.validation(
        reason="epoch",
        epoch=1.0,
        step=10,
        train={"loss": 3.0, "e_sem": 0.5},
        readings=readings,
        **common,
    )

    epochs, others = _table(tmp_path / "metrics.csv"), _table(tmp_path / "val_steps.csv")
    assert [row["reason"] for row in epochs] == ["epoch"]
    assert [row["reason"] for row in others] == ["step 0"]
    assert epochs[0]["train_loss"] == "3.0" and epochs[0]["train_e_sem"] == "0.5"
    assert epochs[0]["T2V_R@1"] == "0.25" and epochs[0]["V2T_R@1"] == "0.125"
    assert epochs[0]["val_loss"] == "2.5" and epochs[0]["V2T_MedR"] == ""
    printed = capsys.readouterr().out
    assert "[stage 2] epoch 1/3" in printed and "R@1 T2V/V2T=0.2500/0.1250" in printed


def test_a_resumed_table_keeps_the_columns_it_has(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,T2V_R@1\n1,10,0.1\n")

    RunReport(tmp_path, enabled=True).validation(
        reason="epoch",
        epoch=2.0,
        epochs=3,
        step=20,
        total_steps=30,
        stage="2",
        lr=1e-4,
        train={},
        readings={"t2v_r1": 0.2},
        eta=None,
    )

    assert path.read_text().splitlines()[-1] == "2.0,20,0.2"
