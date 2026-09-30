"""The assertions before launching (§4.13.2) on tiny real modules and synthetic clips."""

from pathlib import Path

import pytest
import torch

from signworld.worldmodel.data import Collate, caption_statistics
from signworld.worldmodel.distributed import SINGLE
from signworld.worldmodel.preflight import (
    PreflightError,
    Status,
    enforce,
    p5_checksums,
    p16_batch,
    run_preflight,
)

from .conftest import needs_hub, slow, synthetic_corpus, tiny_training


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    directory = tmp_path_factory.mktemp("preflight")
    corpus = synthetic_corpus(directory / "corpus")
    model, config, train, validation = tiny_training(directory, corpus)
    encoder = config.encoder.model_copy(update={"checkpoint_sha256": None})
    pose = config.pose_encoder.model_copy(update={"checkpoint_sha256": None})
    config = config.model_copy(update={"encoder": encoder, "pose_encoder": pose})
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
        checksum_files={"pose_encoder": config.pose_encoder.checkpoint},
    )


@needs_hub
def test_every_assertion_runs_and_the_tiny_setup_passes_but_the_batch(setup) -> None:  # type: ignore[no-untyped-def]
    report = _report(setup, overfit_steps=0)

    status = {a.code: a.status for a in report.assertions}
    assert list(status) == [f"P{i}" for i in range(1, 17)]
    assert status["P2"] == Status.SKIP and status["P15"] == Status.SKIP  # no benchmark; arm A
    assert status["P13"] == Status.SKIP  # run by the slow test below
    failed = {a.code: a.detail for a in report.failures()}
    assert set(failed) == {"P16"}, failed  # 4 clips, not the effective batch of 128
    with pytest.raises(PreflightError, match="P16"):
        enforce(report)


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


def test_the_effective_batch_must_be_128(setup) -> None:  # type: ignore[no-untyped-def]
    config = setup[3]
    full = config.model_copy(
        update={"training": config.training.model_copy(update={"batch_size": 128})}
    )

    assert p16_batch(full, SINGLE, per_gpu=128).status == Status.PASS
    assert p16_batch(full, SINGLE, per_gpu=64).status == Status.FAIL
