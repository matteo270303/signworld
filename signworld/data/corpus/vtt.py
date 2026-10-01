"""Parser for the WebVTT subtitle files YouTube serves for manual captions."""

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_TIMING: Final = re.compile(
    r"^(?P<start>(?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>(?:\d+:)?\d{2}:\d{2}\.\d{3})"
)
_TAG: Final = re.compile(r"<[^>]*>")


@dataclass(frozen=True, slots=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def parse_timestamp(value: str) -> float:
    """Seconds of a ``[hh:]mm:ss.mmm`` timestamp."""
    seconds = 0.0
    for part in value.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_vtt(text: str) -> list[Cue]:
    """Cues with their text on one line; markup is stripped and empty cues are dropped."""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        for position, line in enumerate(lines):
            timing = _TIMING.match(line.strip())
            if timing is None:
                continue
            body = " ".join(
                html.unescape(_TAG.sub("", part)).strip() for part in lines[position + 1 :]
            )
            body = " ".join(body.split())
            if body:
                cues.append(
                    Cue(parse_timestamp(timing["start"]), parse_timestamp(timing["end"]), body)
                )
            break
    return cues


def read_vtt(path: Path) -> list[Cue]:
    return parse_vtt(path.read_text(encoding="utf-8"))


def track_language(path: Path, video_id: str) -> str:
    """Language code of a ``<video_id>.<language>.vtt`` file written by yt-dlp."""
    name = path.name
    prefix, suffix = f"{video_id}.", ".vtt"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"{path}: not a subtitle track of {video_id}")
    return name[len(prefix) : -len(suffix)]
