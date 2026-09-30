"""A short run of the whole training loop on tiny real modules and synthetic clips."""

import json
from pathlib import Path
from typing import Any

import torch

from signworld.worldmodel.checkpoint import CheckpointStore, TrainingState
from signworld.worldmodel.data import Collate
from signworld.worldmodel.model import StepRandomness, WorldSign
from signworld.worldmodel.trainer import Trainer

from .conftest import needs_hub, synthetic_corpus, tiny_training


def _setup(directory: Path, teacher: Path | None = None) -> tuple[Any, ...]:
    corpus_directory = directory / "corpus"
    corpus = synthetic_corpus(corpus_directory)
    return tiny_training(directory, corpus, teacher=teacher)


def _records(output: Path) -> list[dict[str, Any]]:
    lines = (output / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@needs_hub
def test_a_short_run_goes_through_every_stage_and_cools_down(tmp_path: Path) -> None:
    model, config, train, validation = _setup(tmp_path)
    output = tmp_path / "run"

    state = Trainer(model, config, train, validation, output).fit()

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

    # Killed after the cooldown began: the same command resumes and finishes the same way.
    model, config, train, validation = _setup(tmp_path, teacher=tmp_path / "sjepa.pt")
    resumed = Trainer(model, config, train, validation, output).fit()
    assert resumed.finished and resumed.step == state.step


@needs_hub
def test_a_checkpoint_reloads_to_the_same_loss(tmp_path: Path) -> None:
    model, _, train, _ = _setup(tmp_path)
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
