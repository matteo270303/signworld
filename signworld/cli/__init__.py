"""Command-line entry point: ``signworld <command>``."""

import logging
from typing import Annotated

import typer

from signworld.cli import acquisition
from signworld.cli.checks import check_app, testdata_app, text_app
from signworld.cli.corpus import manifest_app
from signworld.cli.experiment import experiment_app
from signworld.cli.train import train_app
from signworld.logger import configure_logging

app = typer.Typer(
    help="WorldSign data pipeline.", no_args_is_help=True, pretty_exceptions_enable=False
)

app.add_typer(manifest_app, name="manifest")
app.add_typer(check_app, name="check")
app.add_typer(text_app, name="text")
app.add_typer(testdata_app, name="testdata")
app.add_typer(experiment_app, name="experiment")
app.add_typer(train_app, name="train")

app.command()(acquisition.sources)
app.command("fetch-metadata")(acquisition.fetch_metadata)
app.command("fetch-media")(acquisition.fetch_media)
app.command("probe-youtube")(acquisition.probe_youtube)
app.command()(acquisition.status)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    configure_logging(logging.DEBUG if verbose else logging.INFO)
