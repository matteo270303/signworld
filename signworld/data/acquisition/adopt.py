"""Record in the ledgers the videos that were downloaded outside ``fetch-media``.

``status``, ``fetch-media`` and the manifest read the ledgers, not the disk: a video fetched
by another tool is invisible to them until a ledger says so. Adoption writes that ledger.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from .ledger import Ledger, read_ledgers
from .outcome import Outcome, Status
from .sources.base import DatasetSource
from .sources.youtube_sl25 import YouTubeSL25Source

logger = logging.getLogger(__name__)

LEDGER_NAME = "adopted.jsonl"
DETAIL = "adopted from files already on disk"


@dataclass(frozen=True, slots=True)
class Adoption:
    done: int
    unavailable: int
    without_info: int
    """Videos on disk without ``<id>.info.json``: not adopted, ``fetch-media`` completes them."""
    already_recorded: int


def adopt_downloads(dataset: DatasetSource[object], unavailable: Path | None = None) -> Adoption:
    """Mark ``done`` every video with its file and ``<id>.info.json``, and ``unavailable`` the
    IDs listed (one per line) in ``unavailable``. IDs the ledgers already settle are left alone,
    so the command can be repeated."""
    if not isinstance(dataset, YouTubeSL25Source):
        raise ValueError(f"{dataset.name}: adoption is only defined for YouTube downloads")
    videos = dataset.layout.raw / "videos"
    if not videos.is_dir():
        raise FileNotFoundError(f"{videos}: no downloaded videos to adopt")
    keys = set(dataset.media_keys())
    history = read_ledgers(dataset.layout.ledgers)
    ledger = Ledger(dataset.layout.ledgers / LEDGER_NAME, history=history)
    max_attempts = dataset.config.max_attempts

    def settled(key: str) -> bool:
        return key in history and history[key].is_settled(max_attempts)

    done = without_info = already = 0
    for path in sorted(videos.glob("*.mp4")):
        key = path.stem
        if key not in keys:
            continue
        if settled(key):
            already += 1
        elif (videos / f"{key}.info.json").is_file():
            ledger.record(Outcome(key, Status.DONE, DETAIL))
            done += 1
        else:
            without_info += 1
    gone = 0
    if unavailable is not None:
        for key in sorted(set(unavailable.read_text(encoding="utf-8").split()) & keys):
            if settled(key) or (videos / f"{key}.mp4").is_file():
                continue
            ledger.record(Outcome(key, Status.UNAVAILABLE, DETAIL))
            gone += 1
    logger.info("adopted %d done, %d unavailable", done, gone)
    return Adoption(done, gone, without_info, already)
