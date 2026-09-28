"""YouTube-SL-25 (Tanzer & Zhang, 2024): 39,197 captioned YouTube videos, 25+ sign languages."""

import csv
import random
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..config import AcquisitionConfig, FrozenModel
from ..outcome import Outcome
from ..provenance import verify_sha256
from ..transfer.youtube import YouTubeDownloader
from .base import Access, DatasetSource

_UNKNOWN_LANGUAGE: Final = "???"


@dataclass(frozen=True, slots=True)
class VideoEntry:
    video_id: str
    language: str | None


def read_metadata(path: Path) -> list[VideoEntry]:
    """Parse the release CSV of ``video_id,language`` rows, which has no header.

    Languages are published as sign-language codes (mostly ISO 639-3); ``???`` becomes ``None``.
    """
    entries: list[VideoEntry] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), start=1):
            try:
                video_id, language = row
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line}: expected 'video_id,language', got {row!r}"
                ) from error
            entries.append(
                VideoEntry(video_id, None if language == _UNKNOWN_LANGUAGE else language)
            )
    return entries


def interleave_by_language(entries: list[VideoEntry]) -> list[str]:
    """Unique video IDs, one per language in turn, each language keeping its release order."""
    queues: dict[str | None, list[str]] = {}
    seen: set[str] = set()
    for entry in entries:
        if entry.video_id not in seen:
            seen.add(entry.video_id)
            queues.setdefault(entry.language, []).append(entry.video_id)
    ordered: list[str] = []
    for position in range(max((len(queue) for queue in queues.values()), default=0)):
        ordered.extend(queue[position] for queue in queues.values() if position < len(queue))
    return ordered


def download_order(
    entries: list[VideoEntry], first: Sequence[str] = (), seed: int = 0
) -> list[str]:
    """The videos of the ``first`` languages ahead of all others, each group interleaved.

    The release lists each channel's videos together (its first 684 ASL videos all come from
    one channel), so every language is shuffled with a fixed seed first: any prefix of the
    order then spreads over many channels, which channel-disjoint splits need.
    """
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    leading = interleave_by_language([entry for entry in shuffled if entry.language in first])
    seen = set(leading)
    rest = interleave_by_language([entry for entry in shuffled if entry.language not in first])
    return leading + [video_id for video_id in rest if video_id not in seen]


class YouTubeSL25Settings(FrozenModel):
    metadata_url: str = (
        "https://storage.googleapis.com/gresearch/youtube-sl-25/youtube-sl-25-metadata.csv"
    )
    metadata_sha256: str | None = None
    first_languages: tuple[str, ...] = ()
    """Sign languages fetched before all others, e.g. those the first experiments need."""
    order_seed: int = 0
    """Seed of the shuffle that spreads the download order over channels."""


class YouTubeSL25Source(DatasetSource[YouTubeSL25Settings]):
    """Video streams with their manual subtitles, which carry the sentence timing."""

    name = "youtube_sl25"
    homepage = "https://github.com/google-research/google-research/tree/master/youtube_sl_25"
    terms = "Released as video IDs; each video remains under the YouTube Terms of Service."
    access = Access.PUBLIC
    settings_model = YouTubeSL25Settings

    METADATA_FILE: Final = "youtube-sl-25-metadata.csv"

    def __init__(
        self,
        settings: YouTubeSL25Settings,
        config: AcquisitionConfig,
        downloader: YouTubeDownloader | None = None,
    ) -> None:
        super().__init__(settings, config)
        self._downloader = downloader or YouTubeDownloader(
            config.youtube, self.layout.raw / "videos", with_subtitles=True
        )

    def fetch_metadata(self) -> None:
        target = self.http.fetch(
            self.settings.metadata_url, self.layout.metadata / self.METADATA_FILE
        )
        if self.settings.metadata_sha256 is not None:
            verify_sha256(target, self.settings.metadata_sha256)
        self.provenance.register(target, origin=self.settings.metadata_url)

    def media_keys(self) -> list[str]:
        """Video IDs of ``first_languages`` first, then one per sign language in turn, shuffled.

        Downloads can stop long before the end; interleaving keeps a partial corpus as
        multilingual as the release. The set of keys, and therefore the sharding, is unchanged.
        """
        entries = read_metadata(self.layout.metadata / self.METADATA_FILE)
        return download_order(entries, self.settings.first_languages, self.settings.order_seed)

    def session(self) -> AbstractContextManager[object]:
        return self._downloader

    def fetch_item(self, key: str) -> Outcome:
        return self._downloader.download(key)
