"""Every command of ``signworld`` starts: no GPU, network or real data.

Each command runs once with stand-in configuration and data. It must reach its own logic:
finish, stop with one of its documented exit codes, or report a missing input as a one-line
user error, never with a traceback. Models, downloads and the collaudo folder are replaced.
"""

import functools
import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

import signworld.cli.acquisition
import signworld.cli.checks
import signworld.cli.experiment
import signworld.data.text
from signworld.cli import app
from signworld.experiment.collaudo.results import result_path
from signworld.experiment.train.config import load_config

REPOSITORY = Path(__file__).resolve().parents[2]
PARAMETERS = REPOSITORY / "parameters"
MODEL = PARAMETERS / "model"
EXIT_USER_ERROR = 2


def commands(group: TyperGroup, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Every command and group under ``group``, as paths of names."""
    for name, command in sorted(group.commands.items()):
        path = (*prefix, name)
        yield path
        if isinstance(command, TyperGroup):
            yield from commands(command, path)


ROOT = get_command(app)
assert isinstance(ROOT, TyperGroup)
REGISTERED = sorted(commands(ROOT))


@dataclass(frozen=True)
class Call:
    """How to start one command: its arguments and the exit codes that count as a start."""

    args: tuple[str, ...]
    exits: frozenset[int] = frozenset({0})


def user_error(*args: str) -> Call:
    return Call(args, frozenset({EXIT_USER_ERROR}))


# {acq}, {ana}, {tmp}, {toy}: the stand-in acquisition and analysis settings, a scratch
# folder, and a toy world built by the fixture.
CALLS: dict[tuple[str, ...], Call] = {
    ("sources",): Call(()),
    ("fetch-metadata",): user_error("csl_daily", "-c", "{acq}"),
    ("fetch-media",): user_error("csl_daily", "-c", "{acq}"),
    ("probe-youtube",): Call(("youtube_sl25", "-c", "{acq}", "--video", "dQw4w9WgXcQ")),
    ("status",): user_error("youtube_sl25", "-c", "{acq}"),
    ("manifest", "build"): user_error("youtube_sl25", "-c", "{ana}"),
    **{
        ("check", name): user_error("youtube_sl25", "-c", "{ana}")
        for name in (
            "durations",
            "contamination",
            "split-duplicates",
            "text-geometry",
            "unisign",
            "frame-selection",
            "pose-alignment",
            "pose-quality",
            "pose-reproduction",
        )
    },
    ("check", "checkpoint"): user_error("not-a-checkpoint", "-c", "{ana}"),
    ("check", "video-reproduction"): user_error(
        "youtube_sl25", "-c", "{ana}", "--encoder", "vjepa2_1_vitl", "--device", "cpu"
    ),
    ("check", "predictor"): user_error(
        "youtube_sl25", "-c", "{ana}", "--encoder", "vjepa2_1_vitl", "--device", "cpu"
    ),
    **{
        ("text", name): user_error("youtube_sl25", "-c", "{ana}", "--device", "cpu")
        for name in ("embed", "verify", "collaudo")
    },
    ("testdata", "build"): user_error("youtube_sl25", "-c", "{ana}"),
    **{
        ("experiment", name): user_error("youtube_sl25", "-c", "{ana}", "--device", "cpu")
        for name in ("pose-teachers", "pose-spectrum", "pose-isotropy")
    },
    ("experiment", "video-features"): user_error(
        "youtube_sl25", "-c", "{ana}", "--run", "vjepa2_1_vitl-256", "--device", "cpu"
    ),
    ("experiment", "video-probes"): user_error("youtube_sl25", "-c", "{ana}"),
    ("train", "toyworld"): Call(("--output", "{tmp}/toy", "--clips", "4")),
    ("train", "toyworld-embed"): Call(("--output", "{toy}", "--device", "cpu")),
    ("train", "index"): Call(
        (
            "--manifest",
            "{toy}/manifest/clips.parquet",
            "--materialized",
            "{toy}",
            "--embeddings",
            "{embeddings}",
            "--output",
            "{tmp}/index.parquet",
            "-c",
            str(MODEL / "worldsign.yaml"),
            "-c",
            str(MODEL / "toyworld.yaml"),
            "--workers",
            "1",
        )
    ),
    ("train", "materialize"): user_error(
        "youtube_sl25", "--output", "{tmp}/clips", "-c", "{ana}", "--device", "cpu"
    ),
    ("text", "false-negatives"): Call(
        (
            "--index",
            "{toy}/index.parquet",
            "--embeddings",
            "{embeddings}",
            "--output",
            "{tmp}/negatives",
            "--reference",
            "8",
        )
    ),
}

# They build WorldSign on the V-JEPA 2.1 weights, which need the hub code and a GPU (or, for
# fetch-models, the network): only their help, their module and their default settings are
# checked.
NEEDS_MODEL = {
    ("train", "fetch-models"): "signworld.models.encoders.weights",
    ("train", "ridge-baseline"): "signworld.experiment.evaluation.ridge",
    ("train", "report"): "signworld.experiment.evaluation.report",
    ("train", "compare"): "signworld.experiment.evaluation.compare",
    ("train", "run"): "signworld.experiment.train.run",
    ("train", "evaluate"): "signworld.experiment.evaluation.worldsign",
    ("train", "overfit"): "signworld.experiment.collaudo.worldsign",
    ("train", "benchmark"): "signworld.experiment.collaudo.worldsign",
}


def _resolve(path: tuple[str, ...]) -> Any:
    command: Any = ROOT
    for name in path:
        command = command.commands[name]
    return command


COMMANDS = [path for path in REGISTERED if not isinstance(_resolve(path), TyperGroup)]


def test_every_command_is_started() -> None:
    assert set(COMMANDS) == set(CALLS) | set(NEEDS_MODEL)
    assert not set(CALLS) & set(NEEDS_MODEL)


@pytest.mark.parametrize("path", REGISTERED, ids=" ".join)
def test_help(path: tuple[str, ...]) -> None:
    result = CliRunner().invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


class FakeEncoder:
    """Stands in for EmbeddingGemma: same identity type, random unit rows, no download."""

    def __init__(self, settings: Any, device: str | None = None) -> None:
        self.identity = signworld.data.text.EmbeddingIdentity(
            settings.model_id, "0" * 40, settings.prompt
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.random.default_rng(len(texts)).standard_normal((len(texts), 768))
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def fake_probe(config: Any, videos: list[str]) -> list[Any]:
    return []


@pytest.fixture(scope="module")
def toy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four synthetic clips with their (stand-in) caption embeddings."""
    root = tmp_path_factory.mktemp("toyworld")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(signworld.data.text, "SentenceTransformerEncoder", FakeEncoder)
        for args in (
            ["toyworld", "--output", str(root), "--clips", "4"],
            ["toyworld-embed", "--output", str(root), "--device", "cpu"],
        ):
            result = CliRunner().invoke(app, ["train", *args])
            assert result.exit_code == 0, result.output
    embeddings = next((root / "text").iterdir())
    index = [
        "index",
        *("--manifest", str(root / "manifest" / "clips.parquet"), "--materialized", str(root)),
        *("--embeddings", str(embeddings), "--output", str(root / "index.parquet")),
        *("-c", str(MODEL / "worldsign.yaml"), "-c", str(MODEL / "toyworld.yaml")),
        *("--workers", "1"),
    ]
    result = CliRunner().invoke(app, ["train", *index])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def settings(tmp_path: Path) -> dict[str, str]:
    """The repository's settings, with every folder moved under ``tmp_path``."""
    acquisition = yaml.safe_load((PARAMETERS / "acquisition" / "default.yaml").read_text("utf-8"))
    acquisition["data_root"] = str(tmp_path / "data")
    acquisition["youtube"]["cookies_file"] = str(tmp_path / "cookies.txt")
    acquisition["youtube"]["pot_server_home"] = str(tmp_path / "pot")
    acquisition_path = tmp_path / "acquisition.yaml"
    acquisition_path.write_text(yaml.safe_dump(acquisition), "utf-8")
    analysis = yaml.safe_load((PARAMETERS / "analysis" / "default.yaml").read_text("utf-8"))
    analysis["acquisition_config"] = str(acquisition_path)
    analysis["models_root"] = str(tmp_path / "models")
    analysis["test_data"]["root"] = str(tmp_path / "test-data")
    analysis_path = tmp_path / "analysis.yaml"
    analysis_path.write_text(yaml.safe_dump(analysis), "utf-8")
    return {"acq": str(acquisition_path), "ana": str(analysis_path), "tmp": str(tmp_path)}


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reports go to ``tmp_path``; models and network calls are stand-ins."""
    collaudo = functools.partial(result_path, root=tmp_path / "collaudo")
    for module in (signworld.cli.acquisition, signworld.cli.checks, signworld.cli.experiment):
        monkeypatch.setattr(module, "result_path", collaudo)
    monkeypatch.setattr(signworld.cli.acquisition, "probe", fake_probe)
    monkeypatch.setattr(signworld.data.text, "SentenceTransformerEncoder", FakeEncoder)


@pytest.mark.usefixtures("isolated")
@pytest.mark.parametrize("path", sorted(CALLS), ids=" ".join)
def test_command_starts(path: tuple[str, ...], settings: dict[str, str], toy_world: Path) -> None:
    embeddings = next((toy_world / "text").iterdir())
    values = {**settings, "toy": str(toy_world), "embeddings": str(embeddings)}
    call = CALLS[path]
    result = CliRunner().invoke(app, [*path, *(arg.format(**values) for arg in call.args)])
    assert result.exit_code in call.exits, result.output
    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), result.exception
    if result.exit_code == EXIT_USER_ERROR:  # third-party warnings may come first
        assert any(line.startswith("error: ") for line in result.output.splitlines()), result.output


@pytest.mark.parametrize("path", sorted(NEEDS_MODEL), ids=" ".join)
def test_model_command_imports_and_has_settings(path: tuple[str, ...]) -> None:
    importlib.import_module(NEEDS_MODEL[path])
    assert load_config(MODEL / "worldsign.yaml").name
