"""Pieces shared by every command group of the command-line interface."""

import functools
from collections.abc import Callable
from typing import Annotated, Final

import typer

from .acquisition.sources import UnknownSourceError
from .acquisition.sources.base import AccessRequiredError

EXIT_USER_ERROR: Final = 2

SourceArgument = Annotated[str, typer.Argument(help="Dataset source; see `signworld sources`.")]


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
