"""Every test of the collaudo has its folder and README in ``collaudo/``, and nothing else."""

import pytest

from signworld.experiment.collaudo.results import ROOT, TESTS, result_path


def test_every_test_has_a_folder_with_a_readme() -> None:
    for test in TESTS:
        readme = ROOT / test / "README.md"
        assert readme.is_file(), f"{readme} is missing"
        text = readme.read_text(encoding="utf-8")
        for heading in ("## Perché", "## Cosa testa", "## Come"):
            assert heading in text, f"{readme}: no {heading!r} section"


def test_every_folder_is_a_known_test() -> None:
    folders = {path.name for path in ROOT.iterdir() if path.is_dir()}
    assert folders <= set(TESTS), (
        f"folders not in signworld.experiment.collaudo.results.TESTS: {folders - set(TESTS)}"
    )


def test_result_path_refuses_an_unknown_test() -> None:
    assert result_path("pc1-geometria-testo", "x.json") == ROOT / "pc1-geometria-testo" / "x.json"
    with pytest.raises(KeyError):
        result_path("pc9", "x.json")
