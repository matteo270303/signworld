"""Deterministic partition of item keys across parallel jobs."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self


def stable_bucket(key: str, buckets: int) -> int:
    """Bucket of ``key``, identical on every machine and Python process."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


@dataclass(frozen=True, slots=True)
class Shard:
    """One of ``count`` disjoint parts of a key set.

    Membership depends only on the key, never on list order, so every job agrees on the
    partition without coordination.
    """

    index: int
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"shard count must be positive, got {self.count}")
        if not 0 <= self.index < self.count:
            raise ValueError(f"shard index {self.index} is outside [0, {self.count})")

    @classmethod
    def whole(cls) -> Self:
        return cls(index=0, count=1)

    @property
    def label(self) -> str:
        return f"shard-{self.index:05d}-of-{self.count:05d}"

    def owns(self, key: str) -> bool:
        return stable_bucket(key, self.count) == self.index

    def select(self, keys: Iterable[str]) -> list[str]:
        return [key for key in keys if self.owns(key)]
