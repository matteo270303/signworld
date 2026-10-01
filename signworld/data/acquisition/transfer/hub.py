"""Revision-pinned downloads from a repository on the Hugging Face Hub."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size_bytes: int


class HubDataset:
    """A Hub repository frozen at one commit, so every download is reproducible."""

    def __init__(
        self,
        repo_id: str,
        revision: str,
        api: HfApi | None = None,
        repo_type: Literal["dataset", "model"] = "dataset",
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self._api = api or HfApi()

    def list_files(self, pattern: re.Pattern[str]) -> list[RemoteFile]:
        entries = self._api.list_repo_tree(
            self.repo_id, repo_type=self.repo_type, revision=self.revision, recursive=True
        )
        files = [
            RemoteFile(entry.path, entry.size)
            for entry in entries
            if isinstance(entry, RepoFile) and pattern.fullmatch(entry.path)
        ]
        return sorted(files, key=lambda file: file.path)

    def fetch(self, path: str, local_dir: Path) -> Path:
        """Download ``path`` to ``local_dir/path``; complete files are not downloaded again."""
        return Path(
            hf_hub_download(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                revision=self.revision,
                local_dir=local_dir,
            )
        )

    def origin(self, path: str) -> str:
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"hf://{prefix}{self.repo_id}@{self.revision}/{path}"
