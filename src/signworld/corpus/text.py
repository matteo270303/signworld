"""Caption normalisation and near-duplicate detection."""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Final

_NON_WORD: Final = re.compile(r"[^\w]+")

NEAR_DUPLICATE_RATIO: Final = 0.9
"""Character-level similarity from which two normalised captions count as the same sentence.

[Our proposal] It absorbs punctuation, casing and small transcription edits between two
subtitle versions of one sentence, and stays below the similarity of distinct sentences.
"""


def normalize_caption(text: str) -> str:
    """Case-folded words of ``text`` separated by single spaces, punctuation removed."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_NON_WORD.sub(" ", folded).split())


def are_near_duplicates(first: str, second: str, ratio: float = NEAR_DUPLICATE_RATIO) -> bool:
    """Whether two captions are identical or nearly identical after normalisation."""
    a, b = normalize_caption(first), normalize_caption(second)
    if a == b:
        return True
    if not a or not b:
        return False
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.real_quick_ratio() >= ratio and matcher.ratio() >= ratio
