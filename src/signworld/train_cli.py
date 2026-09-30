"""``signworld train``: the training index of stage 0 and the training runs (§4.10)."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from .cli_support import reports_user_errors

train_app = typer.Typer(help="Train WorldSign.", no_args_is_help=True)
logger = logging.getLogger(__name__)

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


@train_app.command("index")
@reports_user_errors
def index(  # noqa: PLR0913, PLR0917 (typer options)
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Clip manifest.")],
    materialized: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Root of the materialised clips.")
    ],
    embeddings: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="EmbeddingStore directory.")
    ],
    output: Annotated[Path, typer.Option(dir_okay=False, help="Index to write (Parquet).")],
    config: ConfigFiles,
    check_poses: Annotated[
        bool, typer.Option(help="Read every pose and drop those without shoulders.")
    ] = True,
    benchmark: Annotated[
        list[Path] | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Benchmark manifest whose validation and test clips are removed (§3.9); repeat.",
        ),
    ] = None,
) -> None:
    """Join manifest, materialised clips and captions into the training index, with splits."""
    from .checks.contamination import contamination_report  # noqa: PLC0415
    from .corpus.manifest import read_manifest  # noqa: PLC0415 (heavy imports on demand)
    from .corpus.materialize import MaterializedIndex  # noqa: PLC0415
    from .text.embedding import EmbeddingStore  # noqa: PLC0415
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.data import build_training_index, write_index  # noqa: PLC0415

    settings = load_config(*config).data
    corpus = read_manifest(manifest)
    excluded: frozenset[str] = frozenset()
    for path in benchmark or []:
        report = contamination_report(corpus, read_manifest(path))
        excluded |= report.excluded
        typer.echo(f"{path.name}: {len(report.excluded)} contaminated clips removed")
    table = build_training_index(
        corpus,
        MaterializedIndex(materialized).records(),
        EmbeddingStore(embeddings).clip_rows(),
        settings,
        check_poses=check_poses,
        excluded=excluded,
    )
    write_index(table, output)
    splits = table.column("split").value_counts().to_pylist()
    typer.echo(
        f"{output}: {table.num_rows} clips; "
        + ", ".join(f"{entry['values']}={entry['counts']}" for entry in splits)
    )


@train_app.command("run")
@reports_user_errors
def run(
    config: ConfigFiles,
    output: Annotated[Path, typer.Option(file_okay=False, help="Run directory.")],
) -> None:
    """Train, or resume, one run; launch with torchrun for several GPUs."""
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.distributed import Distributed  # noqa: PLC0415
    from .worldmodel.run import run as start  # noqa: PLC0415

    collective = Distributed.from_environment()
    try:
        state = start(load_config(*config), output, collective)
    finally:
        collective.shutdown()
    if collective.is_main:
        typer.echo(f"{output}: finished at step {state.step}, best at {state.best_step}")
