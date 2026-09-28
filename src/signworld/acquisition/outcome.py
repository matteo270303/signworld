"""Result of fetching one media item."""

from dataclasses import dataclass
from enum import StrEnum


class Status(StrEnum):
    """How fetching an item ended.

    ``FAILED`` is transient and retried on later runs, up to a maximum number of attempts.
    ``UNAVAILABLE`` is permanent (removed or private content). ``BLOCKED`` means the host
    refuses us: it says nothing about the item, so it is retried without limit.
    """

    DONE = "done"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    BLOCKED = "blocked"

    @property
    def is_terminal(self) -> bool:
        """Terminal items are never attempted again."""
        return self in {Status.DONE, Status.UNAVAILABLE}


@dataclass(frozen=True, slots=True)
class Outcome:
    key: str
    status: Status
    detail: str = ""
