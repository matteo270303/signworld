"""Which subtitle track of a YouTube-SL-25 video carries the video's own language.

A video can hold manual subtitles in several spoken languages, often translations. Only the
track written in the language of the video's sign-language community is kept: using the others
as extra captions is excluded by §3.8. The sign language comes from the release metadata
(ISO 639-3 codes, plus two country names used by the release).

Uploaders sometimes tag the caption track with the sign-language code itself (for example
``ase`` for English captions of ASL videos); such a track is the video's own language by
construction and is preferred.

[Our mapping] Where a community writes several languages, they are listed in order of
preference and the first present track wins.
"""

from collections.abc import Iterable
from typing import Final

RELEASE_SIGN_LANGUAGES: Final = (
    "aed",
    "ase",
    "asf",
    "asq",
    "bfi",
    "bzs",
    "csc",
    "cse",
    "csg",
    "csn",
    "csq",
    "dse",
    "dsl",
    "eso",
    "fcs",
    "fse",
    "fsl",
    "fss",
    "gsg",
    "gss",
    "hks",
    "hsh",
    "icl",
    "ils",
    "inl",
    "ins",
    "ise",
    "isg",
    "isr",
    "jos",
    "jsl",
    "kvk",
    "lls",
    "mfs",
    "nsl",
    "nzs",
    "pks",
    "prl",
    "pso",
    "psp",
    "rsl",
    "sfb",
    "sgg",
    "slf",
    "slovenia",
    "sls",
    "ssp",
    "ssr",
    "svk",
    "swl",
    "tsm",
    "tsq",
    "tss",
    "vgt",
    "vietnam",
)
"""Every sign-language code of the pinned YouTube-SL-25 metadata, apart from the unknown ``???``."""


WRITTEN_LANGUAGES: Final[dict[str, tuple[str, ...]]] = {
    "aed": ("es",),  # Argentine
    "ase": ("en",),  # American
    "asf": ("en",),  # Australian (Auslan)
    "asq": ("de",),  # Austrian
    "bfi": ("en",),  # British
    "bzs": ("pt",),  # Brazilian
    "csc": ("ca", "es"),  # Catalan
    "cse": ("cs",),  # Czech
    "csg": ("es",),  # Chilean
    "csn": ("es",),  # Colombian
    "csq": ("hr",),  # Croatian
    "dse": ("nl",),  # Dutch
    "dsl": ("da",),  # Danish
    "eso": ("es",),  # Salvadoran
    "fcs": ("fr",),  # Quebec
    "fse": ("fi",),  # Finnish
    "fsl": ("fr",),  # French
    "fss": ("sv",),  # Finland-Swedish
    "gsg": ("de",),  # German
    "gss": ("el",),  # Greek
    "hks": ("zh", "yue"),  # Hong Kong
    "hsh": ("hu",),  # Hungarian
    "icl": ("is",),  # Icelandic
    "ils": ("en",),  # International Sign: English is its usual caption language
    "inl": ("id",),  # Indonesian
    "ins": ("hi", "en"),  # Indian
    "ise": ("it",),  # Italian
    "isg": ("en", "ga"),  # Irish
    "isr": ("he",),  # Israeli
    "jos": ("ar",),  # Jordanian
    "jsl": ("ja",),  # Japanese
    "kvk": ("ko",),  # Korean
    "lls": ("lt",),  # Lithuanian
    "mfs": ("es",),  # Mexican
    "nsl": ("no", "nb", "nn"),  # Norwegian
    "nzs": ("en",),  # New Zealand
    "pks": ("ur", "en"),  # Pakistan
    "prl": ("es",),  # Peruvian
    "pso": ("pl",),  # Polish
    "psp": ("fil", "tl", "en"),  # Philippine
    "rsl": ("ru",),  # Russian
    "sfb": ("fr",),  # French Belgian
    "sgg": ("de",),  # Swiss German
    "slf": ("it",),  # Swiss Italian
    "slovenia": ("sl",),  # Slovenian
    "sls": ("en",),  # Singapore
    "ssp": ("es",),  # Spanish
    "ssr": ("fr",),  # Swiss French
    "svk": ("sk",),  # Slovak
    "swl": ("sv",),  # Swedish
    "tsm": ("tr",),  # Turkish
    "tsq": ("th",),  # Thai
    "tss": ("zh",),  # Taiwan
    "vgt": ("nl",),  # Flemish
    "vietnam": ("vi",),  # Vietnamese
}


def _base(code: str) -> str:
    """``es-419`` and ``en-US`` are written in ``es`` and ``en``."""
    return code.split("-", maxsplit=1)[0].lower()


def written_language(track: str) -> str:
    """Written language of a caption track, the unit of the per-language centring (§4.4.4).

    Regional variants collapse (``es-419`` is ``es``); a track tagged with a sign-language code
    is written in the first language of that community (``ase`` is ``en``).
    """
    if track in WRITTEN_LANGUAGES:
        return WRITTEN_LANGUAGES[track][0]
    return _base(track)


def own_language_track(sign_language: str | None, tracks: Iterable[str]) -> str | None:
    """The track written in the video's own language, or ``None`` when there is none.

    Videos whose sign language is unknown in the release have no own language to match.
    """
    if sign_language is None:
        return None
    available = sorted(tracks)
    if sign_language in available:
        return sign_language
    for language in WRITTEN_LANGUAGES.get(sign_language, ()):
        matches = [track for track in available if _base(track) == language]
        if matches:
            return min(matches, key=lambda track: (track != language, track))
    return None
