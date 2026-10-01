"""The contract of a dataset source, and the sharded fetch loop every source inherits."""

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from time import monotonic
from typing import ClassVar

from pydantic import BaseModel

from ..config import AcquisitionConfig, FrozenModel
from ..ledger import Ledger
from ..outcome import Outcome, Status
from ..pacing import RefusalBackoff
from ..provenance import ProvenanceLog
from ..sharding import Shard
from ..transfer.http import HttpDownloader

logger = logging.getLogger(__name__)


class Access(StrEnum):
    """How a release is obtained; ``LOCAL`` ones already exist and are checked in place."""

    PUBLIC = "public"
    CREDENTIALS = "credentials"
    MANUAL = "manual"
    LOCAL = "local"


class AccessRequiredError(RuntimeError):
    """Fetching cannot start until the user obtains access to the dataset."""


class NoSettings(FrozenModel):
    """Settings of a source with nothing to configure."""


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Where one dataset lives under the data root."""

    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def extracted(self) -> Path:
        return self.root / "extracted"

    @property
    def ledgers(self) -> Path:
        return self.root / "ledgers"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest" / "clips.parquet"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def text_embeddings(self) -> Path:
        return self.root / "text"


@dataclass(frozen=True, slots=True)
class ShardReport:
    source: str
    shard: Shard
    counts: dict[Status, int]
    remaining: int
    """Items of the shard that are still unsettled after the run."""
    stopped_early: bool
    """The run ended before its items were exhausted: refusals, or the time budget."""
    refused: bool = False
    """The run ended on consecutive refusals: the host needs a long rest, not a short one."""

    @property
    def complete(self) -> bool:
        return self.remaining == 0


class DatasetSource[SettingsT: BaseModel](ABC):
    """A dataset release: its metadata files, its media items, and how to fetch one item.

    Subclasses say *what* to fetch. This class owns *how* media are fetched at scale:
    deterministic sharding, resuming from the ledgers, isolating per-item failures, and
    stopping a shard when the host keeps refusing requests.
    """

    name: ClassVar[str]
    homepage: ClassVar[str]
    terms: ClassVar[str]
    access: ClassVar[Access]
    settings_model: ClassVar[type[BaseModel]]

    def __init__(self, settings: SettingsT, config: AcquisitionConfig) -> None:
        self.settings = settings
        self.config = config
        self.layout = DatasetLayout(config.data_root / self.name)

    @abstractmethod
    def fetch_metadata(self) -> None:
        """Download the annotation files and record their provenance."""

    @abstractmethod
    def media_keys(self) -> list[str]:
        """Stable identifiers of every media item, read from the fetched metadata."""

    @abstractmethod
    def fetch_item(self, key: str) -> Outcome:
        """Fetch one media item, reporting expected failures instead of raising them."""

    def check_access(self) -> None:  # noqa: B027 (optional hook: public releases need no check)
        """Raise :class:`AccessRequiredError` if fetching cannot start."""

    @cached_property
    def http(self) -> HttpDownloader:
        return HttpDownloader(self.config.http)

    @property
    def provenance(self) -> ProvenanceLog:
        return ProvenanceLog(self.layout.metadata)

    def session(self) -> AbstractContextManager[object]:
        """Resources shared by every item of a run, such as one authenticated client."""
        return nullcontext()

    def fetch_media(
        self,
        shard: Shard,
        *,
        limit: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ShardReport:
        """Fetch the unsettled items of ``shard``, at most ``limit`` of them."""
        self.check_access()
        max_attempts = self.config.max_attempts
        ledger = Ledger.for_shard(self.layout.ledgers, shard)
        history = ledger.latest()
        pending = [
            key
            for key in shard.select(self.media_keys())
            if key not in history or not history[key].is_settled(max_attempts)
        ]
        batch = pending[:limit]
        logger.info(
            "%s %s: %d items pending, %d in this run",
            self.name,
            shard.label,
            len(pending),
            len(batch),
        )

        backoff = RefusalBackoff(self.config.refusals, sleep)
        budget = self.config.run_budget_s
        started = monotonic()
        counts: Counter[Status] = Counter()
        settled = 0
        stopped_early = refused = False
        with self.session():
            for position, key in enumerate(batch, start=1):
                if budget is not None and monotonic() - started > budget:
                    logger.info(
                        "%s %s: run budget of %.0f s spent; resting before the next run",
                        self.name,
                        shard.label,
                        budget,
                    )
                    stopped_early = True
                    break
                entry = ledger.record(self._fetch_isolated(key))
                counts[entry.status] += 1
                settled += entry.is_settled(max_attempts)
                logger.info(
                    "[%d/%d] %s %s %s", position, len(batch), key, entry.status, entry.detail
                )
                if not backoff.should_continue(entry.status):
                    logger.error(
                        "%s %s: stopping after %d consecutive refusals",
                        self.name,
                        shard.label,
                        backoff.streak,
                    )
                    stopped_early = refused = True
                    break
        return ShardReport(
            self.name,
            shard,
            dict(counts),
            len(pending) - settled,
            stopped_early=stopped_early,
            refused=refused,
        )

    def _fetch_isolated(self, key: str) -> Outcome:
        try:
            return self.fetch_item(key)
        except Exception as error:  # one broken item must not abort the rest of the shard
            logger.exception("%s: unexpected error while fetching %s", self.name, key)
            return Outcome(key, Status.FAILED, f"{type(error).__name__}: {error}")


class ManualDatasetSource[SettingsT: BaseModel](DatasetSource[SettingsT]):
    """A release that cannot be fetched programmatically; fetching explains how to get it."""

    instructions: ClassVar[str]

    def check_access(self) -> None:
        raise AccessRequiredError(f"{self.name}: {self.instructions}")

    def fetch_metadata(self) -> None:
        self.check_access()

    def media_keys(self) -> list[str]:
        return []

    def fetch_item(self, key: str) -> Outcome:
        raise AccessRequiredError(f"{self.name}: {self.instructions}")
