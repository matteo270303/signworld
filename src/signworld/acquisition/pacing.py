"""Time limits and pauses that keep long-running fetch jobs alive and polite."""

import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

from .config import RefusalPolicy
from .outcome import Status


class DeadlineExceededError(BaseException):
    """Raised inside a ``deadline`` block when time runs out.

    It derives from ``BaseException`` so that third-party code catching ``Exception``
    (retry loops, cleanup handlers) cannot swallow it and keep a stalled transfer alive.
    """


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Interrupt the block after ``seconds`` of wall-clock time.

    Uses ``SIGALRM``, so it works only in the main thread of a Unix process.
    """

    def expire(signum: int, frame: FrameType | None) -> None:
        raise DeadlineExceededError(f"no result within {seconds:.0f} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class RefusalBackoff:
    """Pauses after consecutive refusals, doubling each pause, and says when to give up."""

    def __init__(self, policy: RefusalPolicy, sleep: Callable[[float], None] = time.sleep) -> None:
        self._policy = policy
        self._sleep = sleep
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def should_continue(self, status: Status) -> bool:
        """Account for one outcome; ``False`` once the host has refused too often in a row."""
        if status is not Status.BLOCKED:
            self._streak = 0
            return True
        self._streak += 1
        if self._streak >= self._policy.max_consecutive:
            return False
        self._sleep(self._policy.cooldown_s * 2 ** (self._streak - 1))
        return True
