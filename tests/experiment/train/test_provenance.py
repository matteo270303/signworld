"""Seeds and provenance: the same seed draws the same, a resume cannot mix two runs."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from signworld.experiment.train.config import load_config
from signworld.experiment.train.provenance import (
    PROVENANCE,
    Provenance,
    ProvenanceError,
    config_hash,
    record_launch,
    seed_everything,
    seed_step,
    source_hash,
)
from tests.worldsign import ABLATIONS, BASE


def _weights(seed: int) -> torch.Tensor:
    seed_everything(seed)
    return nn.Linear(8, 8).weight.detach().clone()


def test_the_seed_fixes_the_initial_weights() -> None:
    assert torch.equal(_weights(0), _weights(0))
    assert not torch.equal(_weights(0), _weights(1))


def _dropout(seed: int, step: int, rank: int) -> torch.Tensor:
    torch.manual_seed(12345)  # whatever drew before: diagnostics, a previous step
    torch.rand(7)
    seed_step(seed, step, rank)
    return nn.functional.dropout(torch.ones(64), 0.5)


def test_a_step_draws_the_same_dropout_whatever_came_before() -> None:
    assert torch.equal(_dropout(0, 5, 0), _dropout(0, 5, 0))
    assert not torch.equal(_dropout(0, 5, 0), _dropout(0, 6, 0))
    assert not torch.equal(_dropout(0, 5, 0), _dropout(0, 5, 1))  # every GPU its own
    assert not torch.equal(_dropout(0, 5, 0), _dropout(1, 5, 0))


def _repository(root: Path, text: str = "x = 1\n") -> Path:
    (root / "signworld").mkdir(parents=True, exist_ok=True)
    (root / "signworld" / "module.py").write_text(text)
    (root / "uv.lock").write_text("lock\n")
    return root


def _launch(run: Path, root: Path, config, *, resumed: bool, allow: bool = False) -> Provenance:  # type: ignore[no-untyped-def]
    return record_launch(
        run,
        config,
        root=root,
        resumed=resumed,
        world_size=2,
        clips_per_gpu=64,
        allow_code_change=allow,
    )


def test_every_launch_adds_a_segment_with_code_and_environment(tmp_path: Path) -> None:
    root, run = _repository(tmp_path / "repo"), tmp_path / "run"
    config = load_config(BASE)

    _launch(run, root, config, resumed=False)
    provenance = _launch(run, root, config, resumed=True)

    saved = json.loads((run / PROVENANCE).read_text())
    assert saved["config_sha256"] == config_hash(config) and saved["seed"] == 0
    assert [s["resumed"] for s in provenance.segments] == [False, True]
    first = provenance.segments[0]
    assert first["code"]["source_sha256"] == source_hash(root)
    assert first["environment"]["torch"] == torch.__version__
    assert first["world_size"] == 2 and first["clips_per_gpu"] == 64


def test_a_resume_with_another_configuration_is_refused(tmp_path: Path) -> None:
    root, run = _repository(tmp_path / "repo"), tmp_path / "run"
    _launch(run, root, load_config(BASE), resumed=False)

    with pytest.raises(ProvenanceError, match="configuration"):
        _launch(run, root, load_config(BASE, ABLATIONS / "arm_C.yaml"), resumed=True)


def test_other_code_resumes_only_when_allowed_and_is_recorded(tmp_path: Path) -> None:
    root, run = _repository(tmp_path / "repo"), tmp_path / "run"
    config = load_config(BASE)
    _launch(run, root, config, resumed=False)
    _repository(root, "x = 2\n")

    with pytest.raises(ProvenanceError, match="code differs"):
        _launch(run, root, config, resumed=True)
    provenance = _launch(run, root, config, resumed=True, allow=True)

    assert provenance.segments[-1]["code_change_allowed"]
    assert (
        provenance.segments[0]["code"]["source_sha256"]
        != (provenance.segments[-1]["code"]["source_sha256"])
    )


def test_a_checkpoint_without_provenance_cannot_resume(tmp_path: Path) -> None:
    root = _repository(tmp_path / "repo")

    with pytest.raises(ProvenanceError, match="without provenance"):
        _launch(tmp_path / "run", root, load_config(BASE), resumed=True)
