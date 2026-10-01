"""Every ``signworld`` command line of the README names a real command and real options."""

import re
from pathlib import Path
from typing import Any

import pytest
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from signworld.cli import app

README = Path(__file__).resolve().parents[2] / "README.md"
COMMAND = re.compile(r"(?:uv run (?:python main\.py|signworld)|--no-python signworld) (.*)")


def command_lines() -> list[str]:
    text = README.read_text("utf-8").replace("\\\n", " ")
    return [
        match.group(1).split("#")[0] for match in map(COMMAND.search, text.splitlines()) if match
    ]


def resolve(words: list[str]) -> tuple[list[str], Any]:
    """The longest command path at the start of ``words``, and its command."""
    node: Any = get_command(app)
    path: list[str] = []
    for word in words:
        if not isinstance(node, TyperGroup) or word not in node.commands:
            break
        node = node.commands[word]
        path.append(word)
    return path, node


@pytest.mark.parametrize("line", command_lines())
def test_readme_command_exists(line: str) -> None:
    words = line.split()
    path, command = resolve(words)
    result = CliRunner().invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert not isinstance(command, TyperGroup) or "--help" in words, f"not a command: {line}"
    options = {name for parameter in command.params for name in parameter.opts} | {"--help"}
    used = {word.split("=")[0] for word in words if word.startswith("-")}
    assert used <= options, f"unknown options {used - options} in: {line}"
