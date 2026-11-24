"""The assertions before launching (§4.13.2) on tiny real modules and synthetic clips."""

import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch

from signworld.data.loaders import Collate, caption_statistics
from signworld.experiment.train.distributed import SINGLE
from signworld.experiment.train.preflight import (
    CHECKS,
    PreflightError,
    PreflightReport,
    Status,
    enforce,
    overfit_verdict,
    p5_checksums,
    p16_batch,
    run_preflight,
)
from tests.worldsign import needs_hub, slow, synthetic_corpus, tiny_training


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    directory = tmp_path_factory.mktemp("preflight")
    corpus = synthetic_corpus(directory / "corpus")
    model, config, train, validation, _ = tiny_training(directory, corpus)
    encoder = config.encoder.model_copy(update={"checkpoint_sha256": None})
    config = config.model_copy(update={"encoder": encoder})
    batch = Collate(model.text.centering.languages)([train[(i, 0)] for i in range(4)])
    splits = {"train": train.table, "val_channel": validation.table}
    return directory, corpus, model, config, batch, splits


def _report(setup, overfit_steps: int):  # type: ignore[no-untyped-def]
    directory, corpus, model, config, batch, splits = setup
    return run_preflight(
        model,
        config,
        splits,
        batch,
        caption_statistics(splits["train"], corpus.embeddings),
        directory / "checksums.json",
        overfit_steps=overfit_steps,
    )


@needs_hub
def test_every_assertion_runs_and_the_tiny_setup_passes_but_the_batch(setup) -> None:  # type: ignore[no-untyped-def]
    report = _report(setup, overfit_steps=0)

    status = {a.code: a.status for a in report.assertions}
    assert list(status) == [f"P{i}" for i in range(1, 18)]
    assert status["P2"] == Status.SKIP and status["P15"] == Status.SKIP  # no benchmark; arm A
    assert status["P13"] == Status.SKIP  # run by the slow test below
    failed = {a.code: a.detail for a in report.failures()}
    assert set(failed) == {"P16"}, failed  # 4 clips, not the effective batch of 128
    with pytest.raises(PreflightError, match="P16"):
        enforce(report)


@needs_hub
def test_the_log_has_one_line_per_assertion_with_what_it_checks(setup, caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.INFO, logger="signworld.experiment.train.preflight"):
        report = _report(setup, overfit_steps=0)

    lines = [r.getMessage().removeprefix("rank 0: ") for r in caplog.records]
    assert [line.split()[0] for line in lines] == [a.code for a in report.assertions]
    for line, assertion in zip(lines, report.assertions, strict=True):
        assert CHECKS[assertion.code] in line and f"{assertion.seconds:.1f} s" in line
    failed = [r for r in caplog.records if "FAIL" in r.getMessage()]
    assert [r.levelno for r in failed] == [logging.ERROR]  # P16, with the rank
    assert failed[0].getMessage().startswith("rank 0: P16")


@slow
@needs_hub
def test_the_single_batch_overfit_learns_and_restores_the_model(setup) -> None:  # type: ignore[no-untyped-def]
    model = setup[2]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    report = _report(setup, overfit_steps=40)

    assert next(a for a in report.assertions if a.code == "P13").status == Status.PASS
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_checksums_catch_a_file_that_changed(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"first")
    record = tmp_path / "checksums.json"

    assert p5_checksums({"w": weights}, {}, record).status == Status.PASS
    weights.write_bytes(b"second")
    assert p5_checksums({"w": weights}, {}, record).status == Status.FAIL
    assert p5_checksums({"w": weights}, {"w": "0" * 64}, tmp_path / "other.json").status == (
        Status.FAIL
    )


def test_the_checksum_record_is_published_whole(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"first")
    record = tmp_path / "checksums.json"

    assert p5_checksums({"w": weights}, {}, record).status == Status.PASS
    assert json.loads(record.read_text()) == {"w": hashlib.sha256(b"first").hexdigest()}
    assert [p.name for p in tmp_path.iterdir()] == ["weights.pt", "checksums.json"]


def test_the_effective_batch_must_be_128(setup) -> None:  # type: ignore[no-untyped-def]
    config = setup[3]
    full = config.model_copy(
        update={"training": config.training.model_copy(update={"batch_size": 128})}
    )

    assert p16_batch(full, SINGLE, per_gpu=128).status == Status.PASS
    assert p16_batch(full, SINGLE, per_gpu=64).status == Status.FAIL


# Loss curves from the cluster: job 64009434 (rank 0, GPU mean not logged) and debug 64003252.
CLUSTER_64009434 = [
    4.7295,
    6.3409,
    3.5393,
    2.9891,
    3.1889,
    2.5741,
    2.3490,
    2.1326,
    2.0901,
    2.2841,
    2.0137,
    2.2813,
    1.9046,
    2.3729,
    1.8561,
    2.0191,
    1.8679,
    2.4430,
    2.6783,
    2.4962,
    2.7607,
    2.3851,
    2.2850,
    2.3788,
    2.3608,
    2.3837,
    2.3879,
    2.1979,
    1.8688,
    1.7822,
]
DEBUG_64003252 = [
    5.4736,
    7.0204,
    4.0831,
    3.4142,
    3.4628,
    2.9534,
    2.7312,
    2.5119,
    2.4424,
    2.2919,
    2.1384,
    2.0434,
    2.0154,
    1.9499,
    1.9153,
    1.8616,
    1.7764,
    1.6563,
    1.5578,
    1.5276,
    1.4422,
    1.3484,
    1.3765,
    1.4348,
    1.4328,
    1.2883,
    1.1669,
    1.1478,
    1.0740,
    1.1050,
]


@pytest.mark.parametrize(
    ("losses", "status"),
    [
        (DEBUG_64003252, Status.PASS),  # halves comfortably
        (CLUSTER_64009434, Status.PASS),  # last 5 steps 2.12 of 4.73: 0.45
        ([*CLUSTER_64009434[:-2], 2.54, 2.54], Status.WARN),  # rank 3's ending: falls, no half
        ([4.0] * 30, Status.FAIL),  # a broken graph: the loss does not move
        ([4.0, 3.9, 3.8, 3.7, 3.7, 3.7], Status.FAIL),  # 0.93: barely moves
        ([4.0, *([float("nan")] * 5)], Status.FAIL),
    ],
)
def test_the_overfit_verdict_fails_only_a_loss_that_does_not_fall(losses, status) -> None:  # type: ignore[no-untyped-def]
    assert overfit_verdict(losses).status == status


def test_a_warning_does_not_stop_the_run_a_failure_elsewhere_does(caplog) -> None:  # type: ignore[no-untyped-def]
    report = PreflightReport([overfit_verdict([*CLUSTER_64009434[:-2], 2.54, 2.54])])
    assert report.passed
    with caplog.at_level(logging.WARNING, logger="signworld.experiment.train.preflight"):
        enforce(report)
    assert "P13 warn" in caplog.text

    with pytest.raises(PreflightError, match="another GPU"):
        enforce(PreflightReport([overfit_verdict(DEBUG_64003252)]), failed_elsewhere=True)
