"""RWTH-PHOENIX-Weather 2014 T (Camgöz et al., 2018): German weather forecasts in DGS."""

import csv
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlparse

from pydantic import PositiveFloat, PositiveInt

from ..config import FrozenModel
from ..outcome import Outcome, Status
from ..transfer.archive import extract_tar
from ..transfer.frames import encode_frames, frame_count
from .base import Access, DatasetSource

logger = logging.getLogger(__name__)

SPLITS: Final = {"train": "train", "dev": "valid", "test": "test"}
"""The release's split folders, and the names the pipeline uses."""


class Phoenix14TSettings(FrozenModel):
    url: str = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
    size_bytes: PositiveInt = 41_699_758_035
    local_root: Path | None = None
    """An already extracted copy: nothing is downloaded, fetching encodes its frames to video."""
    annotation_dir: str = "annotations/manual"
    frames_dir: str = "features/fullFrame-210x260px"
    fps: PositiveFloat = 25.0


@dataclass(frozen=True, slots=True)
class PhoenixSentence:
    name: str
    split: str
    speaker: str
    translation: str
    """The German sentence."""
    official_split: str
    """The release's name of the split (``dev``, not ``valid``): the folder of the frames."""


class Phoenix14TSource(DatasetSource[Phoenix14TSettings]):
    """One archive holding the frames together with gloss and translation annotations.

    With ``local_root`` the archive is already extracted; the frame folders become
    ``raw/videos/<sentence>.mp4``, the form the pipeline reads.
    """

    name = "phoenix14t"
    homepage = "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
    terms = "See the homepage for the terms of use."
    access = Access.PUBLIC
    settings_model = Phoenix14TSettings

    def annotation_path(self, official_split: str) -> Path:
        assert self.settings.local_root is not None  # noqa: S101 (local copies only)
        return (
            self.settings.local_root
            / self.settings.annotation_dir
            / f"PHOENIX-2014-T.{official_split}.corpus.csv"
        )

    def fetch_metadata(self) -> None:
        if self.settings.local_root is None:
            logger.info("%s: annotations ship inside the archive; nothing to fetch", self.name)
            return
        for official in SPLITS:
            self.provenance.register(
                self.annotation_path(official), origin=f"local copy of {self.homepage}"
            )

    @cached_property
    def sentences(self) -> list[PhoenixSentence]:
        """Every sentence of the local copy, in the release's order."""
        sentences = []
        for official, split in SPLITS.items():
            with self.annotation_path(official).open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="|"):
                    sentences.append(
                        PhoenixSentence(
                            row["name"],
                            split,
                            row["speaker"],
                            row["translation"].strip(),
                            official,
                        )
                    )
        return sentences

    @cached_property
    def _by_name(self) -> dict[str, PhoenixSentence]:
        return {sentence.name: sentence for sentence in self.sentences}

    def frames_folder(self, sentence: PhoenixSentence) -> Path:
        assert self.settings.local_root is not None  # noqa: S101 (local copies only)
        return (
            self.settings.local_root
            / self.settings.frames_dir
            / sentence.official_split
            / sentence.name
        )

    def frame_count(self, sentence: PhoenixSentence) -> int:
        return frame_count(self.frames_folder(sentence))

    def media_keys(self) -> list[str]:
        if self.settings.local_root is None:
            return [PurePosixPath(urlparse(self.settings.url).path).name]
        return [sentence.name for sentence in self.sentences]

    def fetch_item(self, key: str) -> Outcome:
        if self.settings.local_root is None:
            archive = self.http.fetch(
                self.settings.url, self.layout.raw / key, expected_bytes=self.settings.size_bytes
            )
            extract_tar(archive, self.layout.extracted)
            return Outcome(key, Status.DONE)
        target = self.layout.raw / "videos" / f"{key}.mp4"
        if target.is_file():
            return Outcome(key, Status.DONE, str(target))
        frames = self.frames_folder(self._by_name[key])
        if not frames.is_dir():
            return Outcome(key, Status.UNAVAILABLE, "no frame folder in the local copy")
        encode_frames(frames, self.settings.fps, target, ".png")
        return Outcome(key, Status.DONE, str(target))
