"""CSL-News (Li et al., 2025): 1,985 hours of Chinese Sign Language news on the Hugging Face Hub."""

import re
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from ..config import AcquisitionConfig, FrozenModel
from ..outcome import Outcome, Status
from ..transfer.archive import extract_zip_flat
from ..transfer.hub import HubDataset, RemoteFile
from .base import Access, DatasetSource

_ARCHIVE_INDEX: Final = TypeAdapter(list[RemoteFile])


class CSLNewsSettings(FrozenModel):
    repo_id: str = "ZechengLi19/CSL-News"
    revision: str = "3a0601210333fe760efd09b5d9e2ae5f341ce339"
    labels_path: str = "data/train/CSL_News_Labels.json"
    archive_pattern: str = r"archive_\d{3}\.zip"
    keep_archives: bool = True


class CSLNewsSource(DatasetSource[CSLNewsSettings]):
    """Pre-cut sentence clips shipped in zip archives, extracted flat into one folder."""

    name = "csl_news"
    homepage = "https://huggingface.co/datasets/ZechengLi19/CSL-News"
    terms = "CC BY-NC 4.0"
    access = Access.PUBLIC
    settings_model = CSLNewsSettings

    ARCHIVE_INDEX_FILE: Final = "archives.json"

    def __init__(
        self,
        settings: CSLNewsSettings,
        config: AcquisitionConfig,
        hub: HubDataset | None = None,
    ) -> None:
        super().__init__(settings, config)
        self._hub = hub or HubDataset(settings.repo_id, settings.revision)

    @property
    def clips_dir(self) -> Path:
        return self.layout.extracted / "rgb_format"

    @property
    def archive_index(self) -> Path:
        return self.layout.metadata / self.ARCHIVE_INDEX_FILE

    def fetch_metadata(self) -> None:
        labels = self._hub.fetch(self.settings.labels_path, self.layout.metadata)
        self.provenance.register(labels, origin=self._hub.origin(self.settings.labels_path))
        archives = self._hub.list_files(re.compile(self.settings.archive_pattern))
        self.archive_index.parent.mkdir(parents=True, exist_ok=True)
        self.archive_index.write_bytes(_ARCHIVE_INDEX.dump_json(archives, indent=2))
        self.provenance.register(
            self.archive_index, origin=self._hub.origin(self.settings.archive_pattern)
        )

    def media_keys(self) -> list[str]:
        archives = _ARCHIVE_INDEX.validate_json(self.archive_index.read_bytes())
        return [archive.path for archive in archives]

    def fetch_item(self, key: str) -> Outcome:
        archive = self._hub.fetch(key, self.layout.raw)
        written = extract_zip_flat(archive, self.clips_dir)
        if not self.settings.keep_archives:
            archive.unlink()
        return Outcome(key, Status.DONE, f"{written} clips extracted")
