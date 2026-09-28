"""Manifest builders: turn each dataset's own annotation format into clip rows."""

import csv
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Final, Self, cast

from ..acquisition.config import AcquisitionConfig
from ..acquisition.ledger import read_ledgers
from ..acquisition.outcome import Status
from ..acquisition.sources import UnknownSourceError, build_source
from ..acquisition.sources.base import DatasetSource
from ..acquisition.sources.openasl import COLUMNS, OpenASLSource
from ..acquisition.sources.youtube_sl25 import YouTubeSL25Source, read_metadata
from .languages import own_language_track
from .manifest import Clip, write_manifest
from .vtt import parse_timestamp, read_vtt, track_language

logger = logging.getLogger(__name__)


class ManifestBuilder[SourceT: DatasetSource[Any]](ABC):
    """Builds the manifest of one dataset source and writes it to the source's layout."""

    source_type: ClassVar[type[DatasetSource[Any]]]

    def __init__(self, source: SourceT) -> None:
        self.source = source

    @classmethod
    def from_config(cls, config: AcquisitionConfig) -> Self:
        source = build_source(cls.source_type.name, config)
        if not isinstance(source, cls.source_type):
            raise TypeError(f"{cls.__name__} needs a {cls.source_type.__name__}")
        return cls(cast(Any, source))

    @abstractmethod
    def clips(self) -> Iterator[Clip]: ...

    def build(self) -> Path:
        path = self.source.layout.manifest
        count = write_manifest(self.clips(), path)
        logger.info("%s: wrote %d clips to %s", self.source.name, count, path)
        return path


class OpenASLManifest(ManifestBuilder[OpenASLSource]):
    """Every row of the OpenASL release, with its official split."""

    source_type = OpenASLSource
    CAPTION_LANGUAGE: Final = "en"
    SIGN_LANGUAGE: Final = "ase"

    def clips(self) -> Iterator[Clip]:
        video_dir = self.source.settings.local_root / self.source.settings.video_dir
        available: dict[str, bool] = {}
        with self.source.annotation_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            if tuple(next(reader, ())) != COLUMNS:
                raise ValueError(f"{self.source.annotation_path}: unexpected header")
            for row in reader:
                fields = dict(zip(COLUMNS, row, strict=True))
                video_id = fields["yid"]
                if video_id not in available:
                    available[video_id] = (video_dir / f"{video_id}.mp4").is_file()
                yield Clip(
                    clip_id=fields["vid"],
                    video_id=video_id,
                    start_s=parse_timestamp(fields["start"]),
                    end_s=parse_timestamp(fields["end"]),
                    caption=fields["raw-text"],
                    caption_language=self.CAPTION_LANGUAGE,
                    sign_language=self.SIGN_LANGUAGE,
                    channel_id=None,
                    split=fields["split"],
                    video_available=available[video_id],
                )


class YouTubeSL25Manifest(ManifestBuilder[YouTubeSL25Source]):
    """One clip per subtitle cue, from the single track written in the video's own language.

    Videos carry manual subtitles in several spoken languages, mostly translations. Only the
    track of the video's own language is kept (§3.8 excludes captions in other spoken
    languages); videos whose sign language the release leaves unknown, and videos without such
    a track, contribute no captioned clip and stay available for the physical level.
    """

    source_type = YouTubeSL25Source

    def clips(self) -> Iterator[Clip]:
        layout = self.source.layout
        languages = {
            entry.video_id: entry.language
            for entry in read_metadata(layout.metadata / self.source.METADATA_FILE)
        }
        videos = layout.raw / "videos"
        done = sorted(
            key
            for key, entry in read_ledgers(layout.ledgers).items()
            if entry.status is Status.DONE
        )
        for video_id in done:
            info_path = videos / f"{video_id}.info.json"
            if not info_path.is_file():
                logger.warning("%s: marked done but %s is missing; skipped", video_id, info_path)
                continue
            sign_language = languages.get(video_id)
            tracks = {
                track_language(path, video_id): path
                for path in sorted(videos.glob(f"{video_id}.*.vtt"))
            }
            caption_language = own_language_track(sign_language, tracks)
            if caption_language is None:
                logger.info(
                    "%s (%s): no subtitles in the video's own language among %s; no captions kept",
                    video_id,
                    sign_language or "unknown sign language",
                    sorted(tracks) or ["none"],
                )
                continue
            channel_id = json.loads(info_path.read_text(encoding="utf-8")).get("channel_id")
            for index, cue in enumerate(read_vtt(tracks[caption_language])):
                yield Clip(
                    clip_id=f"{video_id}.{caption_language}.{index:05d}",
                    video_id=video_id,
                    start_s=cue.start_s,
                    end_s=cue.end_s,
                    caption=cue.text,
                    caption_language=caption_language,
                    sign_language=sign_language,
                    channel_id=channel_id,
                    split=None,
                    video_available=True,
                )


BUILDERS: Final[dict[str, type[ManifestBuilder[Any]]]] = {
    builder.source_type.name: builder for builder in (OpenASLManifest, YouTubeSL25Manifest)
}


def build_manifest(name: str, config: AcquisitionConfig) -> Path:
    try:
        builder = BUILDERS[name]
    except KeyError as error:
        available = ", ".join(sorted(BUILDERS))
        raise UnknownSourceError(
            f"no manifest builder for {name!r}; available: {available}"
        ) from error
    return builder.from_config(config).build()
