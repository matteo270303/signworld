"""The two directions of cross-modal retrieval, without torch so that settings can use them.

Text to video (``t2v``) ranks the clips for each caption; video to text (``v2t``) ranks the
captions for each clip. Every retrieval measure is reported in both, and the metric that decides
is the mean of the two R@1 (§4.10).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

DIRECTIONS: Final = ("t2v", "v2t")


@dataclass(frozen=True, slots=True)
class Bidirectional:
    """One value per direction, such as an R@1 or a threshold on it."""

    t2v: float
    v2t: float

    @property
    def mean(self) -> float:
        """The mean of the two directions: the metric that decides when the values are R@1."""
        return (self.t2v + self.v2t) / 2

    def scaled(self, factor: float) -> "Bidirectional":
        """Both values times ``factor``, as fractions to percent."""
        return Bidirectional(self.t2v * factor, self.v2t * factor)

    def items(self) -> Iterator[tuple[str, float]]:
        """``(direction, value)`` in the order of ``DIRECTIONS``."""
        yield "t2v", self.t2v
        yield "v2t", self.v2t
