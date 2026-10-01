"""OpenASL (Shi et al., 2022): 98,417 ASL sentence clips from 2,043 YouTube videos.

The release was acquired before this pipeline existed. This source downloads nothing: it
verifies the local copy against the pinned annotation files and records which videos it holds.
"""

import csv
from pathlib import Path
from typing import Final

from ..config import FrozenModel
from ..outcome import Outcome, Status
from ..provenance import verify_sha256
from .base import Access, DatasetSource

COLUMNS: Final = ("vid", "yid", "start", "end", "raw-text", "tokenized-text", "gloss", "split")
RELEASE: Final = (
    "https://github.com/chevalierNoir/OpenASL/tree/c7d2350b22f344c5a6669ad37518b493c8f78822"
)


def read_video_ids(path: Path) -> list[str]:
    """Unique YouTube IDs of the annotation TSV, in order of first appearance."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = tuple(next(reader, ()))
        if header != COLUMNS:
            raise ValueError(f"{path}: unexpected header {header!r}; expected {COLUMNS!r}")
        youtube_id = COLUMNS.index("yid")
        return list(dict.fromkeys(row[youtube_id] for row in reader))


class OpenASLSettings(FrozenModel):
    local_root: Path
    annotation_file: str = "data/openasl-v1.0.tsv"
    annotation_sha256: str = "e7e1559bcef5ac77d2c14c2ccfc9db54516768e296235f266b3f4f96de459a40"
    bbox_file: str = "data/bbox-v1.0.json"
    bbox_sha256: str = "a79b5327956670db0988bacd96aebb9229ecd5ea948b8de887cb530669152a44"
    video_dir: str = "raw-video"


class OpenASLSource(DatasetSource[OpenASLSettings]):
    """A local copy checked in place: annotation checksums, and which source videos it holds."""

    name = "openasl"
    homepage = "https://github.com/chevalierNoir/OpenASL"
    terms = "CC BY-NC-ND 4.0"
    access = Access.LOCAL
    settings_model = OpenASLSettings

    @property
    def annotation_path(self) -> Path:
        return self.settings.local_root / self.settings.annotation_file

    def fetch_metadata(self) -> None:
        pinned = (
            (self.settings.annotation_file, self.settings.annotation_sha256),
            (self.settings.bbox_file, self.settings.bbox_sha256),
        )
        for relative, sha256 in pinned:
            path = self.settings.local_root / relative
            verify_sha256(path, sha256)
            self.provenance.register(path, origin=f"{RELEASE}/{relative}")

    def media_keys(self) -> list[str]:
        return read_video_ids(self.annotation_path)

    def fetch_item(self, key: str) -> Outcome:
        video = self.settings.local_root / self.settings.video_dir / f"{key}.mp4"
        if video.is_file():
            return Outcome(key, Status.DONE, str(video))
        return Outcome(key, Status.UNAVAILABLE, "not in the local copy")
