"""``signworld sources|fetch-metadata|fetch-media|probe-youtube|status``: the raw datasets."""

from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from signworld.cli.support import SourceArgument, reports_user_errors
from signworld.data.acquisition.config import AcquisitionConfig
from signworld.data.acquisition.ledger import read_ledgers
from signworld.data.acquisition.outcome import Status
from signworld.data.acquisition.probe import probe
from signworld.data.acquisition.sharding import Shard
from signworld.data.acquisition.sources import SOURCES, build_source
from signworld.data.acquisition.sources.base import DatasetSource
from signworld.experiment.collaudo.report import write_report
from signworld.experiment.collaudo.results import result_path

DEFAULT_CONFIG = Path("configs/acquisition.yaml")
EXIT_INCOMPLETE = 3
EXIT_REFUSED = 4

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        envvar="SIGNWORLD_CONFIG",
        exists=True,
        dir_okay=False,
        help="Acquisition settings (YAML).",
    ),
]


def _open(source: str, config: Path) -> DatasetSource[Any]:
    return build_source(source, AcquisitionConfig.from_yaml(config))


def sources() -> None:
    """List every dataset source and how it can be obtained."""
    for name, source in sorted(SOURCES.items()):
        typer.echo(f"{name:<14}{source.access:<13}{source.terms}")
        typer.echo(f"{'':<27}{source.homepage}")


@reports_user_errors
def fetch_metadata(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Download a source's annotation files and record their provenance."""
    _open(source, config).fetch_metadata()


@reports_user_errors
def fetch_media(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Fetch at most this many items (pilot runs).")
    ] = None,
) -> None:
    """Fetch the media items of one shard, resuming from the ledgers.

    Exits with 0 once every item of the shard is settled, with 4 if the host refused the run,
    and with 3 while items remain for any other reason.
    """
    report = _open(source, config).fetch_media(Shard(shard, num_shards), limit=limit)
    summary = ", ".join(f"{status}={count}" for status, count in sorted(report.counts.items()))
    reason = (
        " (stopped by refusals)"
        if report.refused
        else " (run budget spent)"
        if report.stopped_early
        else ""
    )
    typer.echo(
        f"{report.source} {report.shard.label}: {summary or 'nothing fetched'}; "
        f"{report.remaining} remaining{reason}"
    )
    if report.refused:
        raise typer.Exit(EXIT_REFUSED)
    if not report.complete:
        raise typer.Exit(EXIT_INCOMPLETE)


@reports_user_errors
def probe_youtube(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    videos: Annotated[
        list[str] | None,
        typer.Option(
            "--video",
            help="Video to check (repeatable); default: one fetched and one refused video.",
        ),
    ] = None,
) -> None:
    """Tell account refusals from network ones: one metadata request per video and variant."""
    dataset = _open(source, config)
    ledgers = read_ledgers(dataset.layout.ledgers)
    if not videos:
        videos = [
            next(key for key, entry in ledgers.items() if entry.status is state)
            for state in (Status.DONE, Status.BLOCKED)
        ]
    results = probe(dataset.config.youtube, videos)
    for result in results:
        rejected = " · cookies rejected" if result.cookies_rejected else ""
        typer.echo(f"{result.variant:<16}{result.video_id:<14}{result.status:<12}{rejected}")
        if result.status is not Status.DONE:
            typer.echo(f"{'':<30}{result.detail[:120]}")
    typer.echo(write_report(result_path("sonda-youtube", f"{dataset.name}.json"), "probe", results))


@reports_user_errors
def status(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Summarise how many media items are settled, failing, or never attempted."""
    dataset = _open(source, config)
    keys = dataset.media_keys()
    ledgers = read_ledgers(dataset.layout.ledgers)
    entries = [ledgers[key] for key in keys if key in ledgers]
    max_attempts = dataset.config.max_attempts
    counts = Counter(entry.status for entry in entries)
    settled = sum(entry.is_settled(max_attempts) for entry in entries)
    abandoned = sum(
        entry.status is Status.FAILED and entry.is_settled(max_attempts) for entry in entries
    )
    typer.echo(f"{dataset.name}: {len(keys)} media items")
    for state in Status:
        typer.echo(f"  {state:<22}{counts[state]}")
    typer.echo(f"  {f'failed {max_attempts}+ times':<22}{abandoned}")
    typer.echo(f"  {'to fetch':<22}{len(keys) - settled}")
