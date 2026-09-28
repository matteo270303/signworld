"""PC5 and PC6 (§4.12.1): what a pre-trained checkpoint actually contains.

Before building on a checkpoint the project needs its sections (encoder, predictor, ...), the
parameters of each, the shape of every tensor, the input projections and whether per-layer
normalisations exist. The inspection reads tensors only (``weights_only``), never code.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Self
from urllib.parse import urlparse

import torch
from pydantic import PositiveInt, model_validator

from ..acquisition.config import FrozenModel, HttpConfig
from ..acquisition.provenance import sha256sum
from ..acquisition.transfer.http import HttpDownloader
from ..acquisition.transfer.hub import HubDataset

_NORM_MARKERS: Final = ("norm", "ln_", "layernorm")


class CheckpointSource(FrozenModel):
    """A local file, a file behind a URL, or a file pinned in a Hugging Face model repository."""

    path: Path | None = None
    url: str | None = None
    size_bytes: PositiveInt | None = None
    """Expected size of the file behind ``url``, checked after the download."""
    repo_id: str | None = None
    revision: str | None = None
    """A commit, so the inspected file cannot change under the same name."""
    filename: str | None = None

    @model_validator(mode="after")
    def _one_location(self) -> Self:
        hub = [value is not None for value in (self.repo_id, self.revision, self.filename)]
        locations = [self.path is not None, self.url is not None, all(hub)]
        if any(hub) != all(hub) or sum(locations) != 1:
            raise ValueError("give exactly one of path, url, or repo_id with revision and filename")
        return self

    def materialize(self, directory: Path, http: HttpConfig) -> Path:
        """The checkpoint on local storage, downloaded into ``directory`` when remote."""
        if self.path is not None:
            return self.path
        if self.url is not None:
            name = PurePosixPath(urlparse(self.url).path).name
            return HttpDownloader(http).fetch(
                self.url, directory / name, expected_bytes=self.size_bytes
            )
        assert self.repo_id and self.revision and self.filename  # enforced by the validator
        hub = HubDataset(self.repo_id, self.revision, repo_type="model")
        return hub.fetch(self.filename, directory / self.repo_id)


@dataclass(frozen=True, slots=True)
class ModuleSummary:
    prefix: str
    tensors: int
    parameters: int


@dataclass(frozen=True, slots=True)
class SectionSummary:
    tensors: int
    parameters: int
    modules: list[ModuleSummary]
    normalization_keys: list[str]
    shapes: dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class CheckpointReport:
    path: str
    sha256: str
    sections: dict[str, SectionSummary]
    metadata: dict[str, Any]
    """Non-tensor entries, such as training arguments or the originating entry point."""


def _is_state_dict(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def split_sections(
    checkpoint: object, name: str = "checkpoint"
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """State dicts found at any depth, keyed by their path, and the remaining plain entries."""
    sections: dict[str, dict[str, torch.Tensor]] = {}
    metadata: dict[str, Any] = {}
    if _is_state_dict(checkpoint):
        sections[name] = checkpoint  # type: ignore[assignment]
    elif isinstance(checkpoint, dict):
        tensors = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if tensors:
            sections[name] = tensors
        for key, value in checkpoint.items():
            if torch.is_tensor(value):
                continue
            path = str(key) if name == "checkpoint" else f"{name}.{key}"
            if isinstance(value, dict):
                nested_sections, nested_metadata = split_sections(value, path)
                sections |= nested_sections
                metadata |= nested_metadata
            else:
                metadata[path] = (
                    value if isinstance(value, int | float | str | bool) else repr(value)
                )
    else:
        metadata[name] = repr(checkpoint)
    return sections, metadata


def summarize_section(state: dict[str, torch.Tensor], depth: int = 2) -> SectionSummary:
    """Parameter counts grouped by the first ``depth`` components of each key."""
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for key, tensor in state.items():
        grouped[".".join(key.split(".")[:depth])].append(tensor)
    return SectionSummary(
        tensors=len(state),
        parameters=sum(tensor.numel() for tensor in state.values()),
        modules=[
            ModuleSummary(prefix, len(tensors), sum(tensor.numel() for tensor in tensors))
            for prefix, tensors in grouped.items()
        ],
        normalization_keys=[
            key for key in state if any(marker in key.lower() for marker in _NORM_MARKERS)
        ],
        shapes={key: list(tensor.shape) for key, tensor in state.items()},
    )


def inspect_checkpoint(path: Path, depth: int = 2) -> CheckpointReport:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    sections, metadata = split_sections(checkpoint)
    return CheckpointReport(
        path=str(path),
        sha256=sha256sum(path),
        sections={name: summarize_section(state, depth) for name, state in sections.items()},
        metadata=metadata,
    )
