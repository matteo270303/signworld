from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from signworld.checks.checkpoint import CheckpointSource, inspect_checkpoint, split_sections


def test_sections_are_found_at_any_depth_next_to_metadata() -> None:
    checkpoint = {
        "encoder": {"blocks.0.norm1.weight": torch.ones(4), "blocks.0.attn.qkv": torch.ones(4, 4)},
        "state": {"predictor": {"proj.weight": torch.ones(2, 4)}, "epoch": 3},
        "meta": {"img_size": 384},
    }

    sections, metadata = split_sections(checkpoint)

    assert set(sections) == {"encoder", "state.predictor"}
    assert metadata == {"state.epoch": 3, "meta.img_size": 384}


def test_a_bare_state_dict_is_one_section() -> None:
    sections, metadata = split_sections({"weight": torch.ones(3)})

    assert list(sections) == ["checkpoint"]
    assert metadata == {}


def test_report_counts_parameters_and_normalisations(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "encoder": {
                "blocks.0.norm1.weight": torch.ones(8),
                "blocks.0.mlp.fc1.weight": torch.ones(16, 8),
                "blocks.1.mlp.fc1.weight": torch.ones(16, 8),
            }
        },
        path,
    )

    report = inspect_checkpoint(path)

    encoder = report.sections["encoder"]
    assert encoder.parameters == 8 + 2 * 16 * 8
    assert encoder.normalization_keys == ["blocks.0.norm1.weight"]
    assert [(m.prefix, m.parameters) for m in encoder.modules] == [
        ("blocks.0", 8 + 128),
        ("blocks.1", 128),
    ]
    assert encoder.shapes["blocks.0.mlp.fc1.weight"] == [16, 8]


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"path": "a.pt", "repo_id": "org/model", "revision": "abc", "filename": "a.pt"},
        {"path": "a.pt", "url": "https://example.org/a.pt"},
        {"repo_id": "org/model", "filename": "a.pt"},
    ],
)
def test_checkpoint_source_needs_exactly_one_complete_location(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        CheckpointSource.model_validate(fields)


def test_checkpoint_source_accepts_a_path_or_a_pinned_remote_file() -> None:
    CheckpointSource(path=Path("a.pt"))
    CheckpointSource(url="https://example.org/a.pt", size_bytes=10)
    CheckpointSource(repo_id="org/model", revision="abc", filename="a.pt")
