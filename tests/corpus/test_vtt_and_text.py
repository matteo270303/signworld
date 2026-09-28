from pathlib import Path

import pytest

from signworld.corpus.languages import (
    RELEASE_SIGN_LANGUAGES,
    WRITTEN_LANGUAGES,
    own_language_track,
    written_language,
)
from signworld.corpus.text import are_near_duplicates, normalize_caption
from signworld.corpus.vtt import Cue, parse_timestamp, parse_vtt, track_language

YOUTUBE_VTT = """WEBVTT
Kind: captions
Language: es-419

00:00:03.039 --> 00:00:07.038
En esta sala se encuentra
la escultura &amp; <c>su</c> figura.

1
01:02:03.500 --> 01:02:04.000 align:start position:0%
Segunda frase

00:00:43.023 --> 00:00:44.000

"""


def test_cues_join_lines_strip_markup_and_skip_empty_text() -> None:
    assert parse_vtt(YOUTUBE_VTT) == [
        Cue(3.039, 7.038, "En esta sala se encuentra la escultura & su figura."),
        Cue(3723.5, 3724.0, "Segunda frase"),
    ]


@pytest.mark.parametrize(
    ("value", "seconds"), [("00:06.589", 6.589), ("00:00:06.589", 6.589), ("1:00:00.000", 3600.0)]
)
def test_timestamps_with_and_without_hours(value: str, seconds: float) -> None:
    assert parse_timestamp(value) == pytest.approx(seconds)


def test_track_language_comes_from_the_file_name() -> None:
    assert track_language(Path("abc-_1.es-419.vtt"), "abc-_1") == "es-419"
    with pytest.raises(ValueError, match="not a subtitle track"):
        track_language(Path("other.en.vtt"), "abc-_1")


def test_normalisation_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_caption("  Hello,   WORLD! ") == normalize_caption("hello world")


def test_near_duplicates_tolerate_small_edits_only() -> None:
    assert are_near_duplicates("Then it was time to go home.", "then it was time to go home")
    assert are_near_duplicates("The weather is sunny today", "The weather is sunny toda")
    assert not are_near_duplicates("The weather is sunny today", "The meeting starts at noon")
    assert not are_near_duplicates("", "Hello")


@pytest.mark.parametrize(
    ("sign_language", "tracks", "expected"),
    [
        ("ase", ["ase", "en-US"], "ase"),
        ("ase", ["en", "fr"], "en"),
        ("ase", ["en-US", "fr"], "en-US"),
        ("aed", ["en", "es-419"], "es-419"),
        ("ins", ["en", "hi"], "hi"),
        ("dsl", ["da", "fi", "is", "no", "sv"], "da"),
        ("dsl", ["fi", "sv"], None),
        (None, ["da", "sv"], None),
        ("ase", [], None),
    ],
)
def test_only_the_track_in_the_videos_own_language_is_chosen(
    sign_language: str | None, tracks: list[str], expected: str | None
) -> None:
    assert own_language_track(sign_language, tracks) == expected


def test_every_sign_language_of_the_release_has_a_written_language() -> None:
    assert set(RELEASE_SIGN_LANGUAGES) <= set(WRITTEN_LANGUAGES)


@pytest.mark.parametrize(
    ("track", "language"), [("ase", "en"), ("en-US", "en"), ("es-419", "es"), ("da", "da")]
)
def test_tracks_map_to_their_written_language(track: str, language: str) -> None:
    assert written_language(track) == language
