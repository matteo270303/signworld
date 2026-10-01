"""Pieces shared by every command group of the command-line interface.

The analysis helpers import their dependencies inside the function, like the commands.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

import typer

from signworld.data.acquisition.sources import UnknownSourceError
from signworld.data.acquisition.sources.base import AccessRequiredError

if TYPE_CHECKING:
    from signworld.data.acquisition.sources.base import DatasetSource
    from signworld.experiment.collaudo.analysis import AnalysisConfig

EXIT_USER_ERROR: Final = 2

SourceArgument = Annotated[str, typer.Argument(help="Dataset source; see `signworld sources`.")]
ConfigFiles = Annotated[
    list[Path],
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Model YAML; repeat to lay overlays (arm, ablation) over the base, in order.",
    ),
]


def reports_user_errors[**P, R](command: Callable[P, R]) -> Callable[P, R]:
    """Turn errors the user can fix into a one-line message and a non-zero exit code."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except (AccessRequiredError, UnknownSourceError, FileNotFoundError) as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(EXIT_USER_ERROR) from error

    return wrapper


# --------------------------------------------------------------------------- analysis

ANALYSIS_CONFIG: Final = Path("configs/analysis.yaml")

AnalysisConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        envvar="SIGNWORLD_ANALYSIS_CONFIG",
        exists=True,
        dir_okay=False,
        help="Analysis settings (YAML).",
    ),
]
DeviceOption = Annotated[str | None, typer.Option(help="Torch device, e.g. cuda.")]


def load_analysis(config: Path) -> "AnalysisConfig":
    import torch

    from signworld.experiment.collaudo.analysis import AnalysisConfig

    settings = AnalysisConfig.from_yaml(config)
    torch.set_num_threads(settings.threads)
    return settings


def open_source(name: str, settings: "AnalysisConfig") -> "DatasetSource[Any]":
    from signworld.data.acquisition.sources import build_source

    return build_source(name, settings.acquisition())


def read_source_manifest(source: "DatasetSource[Any]") -> Any:
    from signworld.data.corpus.manifest import read_manifest

    path = source.layout.manifest
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run `signworld manifest build {source.name}` first")
    return read_manifest(path)


def testdata_root(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> Path:
    return settings.test_data.root / dataset.name


def materialized_clips(
    dataset: "DatasetSource[Any]", settings: "AnalysisConfig", count: int | None = None
) -> list[Any]:
    """A reproducible sample of the clips materialized under the test root."""
    import random

    from signworld.data.corpus.materialize import MaterializedIndex
    from signworld.experiment.collaudo.pose_alignment import MaterializedClip

    root = testdata_root(dataset, settings)
    records = MaterializedIndex(root).records()
    if not records:
        raise FileNotFoundError(f"{root}: run `signworld testdata build {dataset.name}` first")
    clips = [MaterializedClip(r.clip_id, Path(r.video), Path(r.pose)) for r in records]
    random.Random(settings.pose_checks.seed).shuffle(clips)
    return clips if count is None else clips[:count]


def labelled_clips(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> list[Any]:
    """The materialized test clips with the video, channel and language the probes read."""
    from signworld.experiment.studies.pose_teachers import LabelledClip

    columns = ["clip_id", "video_id", "channel_id", "sign_language"]
    manifest = read_source_manifest(dataset).select(columns).to_pydict()
    about = {
        clip: (video, channel or "unknown", language or "unknown")
        for clip, video, channel, language in zip(*manifest.values(), strict=True)
    }
    return [
        LabelledClip(clip.clip_id, *about[clip.clip_id], clip.pose)
        for clip in materialized_clips(dataset, settings)
        if clip.clip_id in about
    ]


def video_settings(settings: "AnalysisConfig") -> Any:
    if settings.video_probes is None:
        raise ValueError("the analysis settings have no video_probes section")
    return settings.video_probes


def readout_corpus(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> Any:
    """The read-out clips of the pose-teacher comparison: same seeded subset, same split."""
    from signworld.experiment.studies.pose_teachers import PoseCorpus, probe_subset
    from signworld.metrics.probes import video_split

    corpus = PoseCorpus.load(labelled_clips(dataset, settings), settings.pose_teachers.min_score)
    probe, _, _ = probe_subset(corpus, video_split(corpus.videos), settings.pose_teachers)
    return probe


def video_encoder(name: str, settings: "AnalysisConfig", device: str) -> tuple[Any, Any]:
    """The frozen encoder and, for V-JEPA 2.1, its predictor and load report."""
    import torch

    from signworld.models.encoders.video_encoders import build_encoder

    video = video_settings(settings)
    if name not in video.encoders:
        raise ValueError(f"unknown encoder {name!r}; available: {', '.join(video.encoders)}")
    options = video.encoders[name]
    checkpoint = None
    if options.checkpoint is not None:
        checkpoint = settings.checkpoints[options.checkpoint].materialize(
            settings.models_root, settings.acquisition().http
        )
    encoder, extra = build_encoder(name, options, video.hub_repo, checkpoint)
    return encoder.to(torch.device(device)), extra


def stored_frames(clip: Any) -> Any:
    """The 64 selected frames of a stored test clip, (64, H, W, 3) uint8."""
    from signworld.data.pose.wholebody import PoseTrack
    from signworld.data.video import ClipReader

    track = PoseTrack.load(clip.pose)
    return ClipReader(_clip_video(clip)).frames(track.frame_indices)


def _clip_video(clip: Any) -> Path:
    return Path(clip.pose).parent.parent / "clips" / f"{clip.clip_id}.mp4"
