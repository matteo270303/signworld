"""Fetching items in several worker processes at once.

One process per worker, not one thread: ``deadline`` needs the main thread of its process,
and each worker owns its yt-dlp client and its private copy of the cookies. The parent alone
writes the ledger, so shards keep a single writer.
"""

import logging
import multiprocessing
import multiprocessing.util
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack
from types import TracebackType
from typing import Any, Self

from .outcome import Outcome
from .sources.base import DatasetSource

logger = logging.getLogger(__name__)

_WORKER_SOURCE: DatasetSource[Any] | None = None


def _start_worker(source: DatasetSource[Any]) -> None:
    """Runs in each worker: enter the source's session, and leave it when the worker exits."""
    global _WORKER_SOURCE  # noqa: PLW0603 (one source per worker process)
    _WORKER_SOURCE = source
    session = ExitStack()
    session.enter_context(source.session())
    multiprocessing.util.Finalize(None, session.close, exitpriority=10)


def _fetch(key: str) -> Outcome:
    assert _WORKER_SOURCE is not None
    return _WORKER_SOURCE.fetch_isolated(key)


class WorkerPool:
    """Outcomes of ``workers`` processes fetching keys, in the order they complete.

    Keys are handed out lazily, a few more than there are workers, so a caller that stops
    consuming stops the run within one item per worker; ``drain`` then collects what those
    items produced, which the caller must record: the files are already on disk.
    """

    def __init__(self, source: DatasetSource[Any], workers: int) -> None:
        self._workers = workers
        # fork: the source is inherited, not pickled, and its session opens in the child.
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_start_worker,
            initargs=(source,),
        )
        self._pending: deque[Future[Outcome]] = deque()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def outcomes(self, keys: Iterable[str]) -> Iterator[Outcome]:
        queue = iter(keys)
        exhausted = False
        while True:
            while not exhausted and len(self._pending) < 2 * self._workers:
                key = next(queue, None)
                if key is None:
                    exhausted = True
                else:
                    self._pending.append(self._executor.submit(_fetch, key))
            if not self._pending:
                return
            done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
            finished = next(future for future in self._pending if future in done)
            self._pending.remove(finished)
            yield finished.result()

    def drain(self) -> list[Outcome]:
        """Wait for the items already handed to the workers, and return their outcomes."""
        pending, self._pending = list(self._pending), deque()
        results = []
        for future in pending:
            if not future.cancel():
                results.append(future.result())
        return results
