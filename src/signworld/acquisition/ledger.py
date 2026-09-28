"""Append-only, per-shard record of media fetch outcomes."""

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from .outcome import Outcome, Status
from .sharding import Shard

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    key: str
    status: Status
    detail: str
    attempts: int
    """Outcomes that reached the item; refusals by the host are not counted."""
    recorded_at: str

    def is_settled(self, max_attempts: int) -> bool:
        """Whether the item is finished: terminal, or failed too often to try again."""
        return self.status.is_terminal or (
            self.status is Status.FAILED and self.attempts >= max_attempts
        )


class Ledger:
    """JSON-lines log of one shard's outcomes; the latest line of a key wins.

    Each shard owns its file, so parallel jobs never share a writer. Every line is flushed
    and synced to disk, so a killed job loses at most the item in flight.
    """

    def __init__(self, path: Path, history: Mapping[str, LedgerEntry] | None = None) -> None:
        self._path = path
        self._entries = dict(history) if history is not None else _read_entries(path)

    @classmethod
    def for_shard(cls, directory: Path, shard: Shard) -> Self:
        """The shard's ledger, aware of what every shard recorded before, whatever the count."""
        return cls(directory / f"{shard.label}.jsonl", history=read_ledgers(directory))

    def latest(self) -> dict[str, LedgerEntry]:
        return dict(self._entries)

    def record(self, outcome: Outcome) -> LedgerEntry:
        previous = self._entries.get(outcome.key)
        entry = LedgerEntry(
            key=outcome.key,
            status=outcome.status,
            detail=outcome.detail,
            attempts=(previous.attempts if previous else 0)
            + (outcome.status is not Status.BLOCKED),
            recorded_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._entries[entry.key] = entry
        return entry


def read_ledgers(directory: Path) -> dict[str, LedgerEntry]:
    """Latest entry of every key across all shard ledgers in ``directory``."""
    merged: dict[str, LedgerEntry] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for key, entry in _read_entries(path).items():
            current = merged.get(key)
            if current is None or entry.recorded_at >= current.recorded_at:
                merged[key] = entry
    return merged


def _read_entries(path: Path) -> dict[str, LedgerEntry]:
    entries: dict[str, LedgerEntry] = {}
    if not path.exists():
        return entries
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
                entry = LedgerEntry(
                    key=raw["key"],
                    status=Status(raw["status"]),
                    detail=raw["detail"],
                    attempts=int(raw["attempts"]),
                    recorded_at=raw["recorded_at"],
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(
                    "Skipping unreadable line %d of %s (interrupted write?)", number, path
                )
                continue
            entries[entry.key] = entry
    return entries
