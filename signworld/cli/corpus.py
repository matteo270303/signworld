"""``signworld manifest build`` and ``train index|materialize``: the clip corpus.

The heavy dependencies (pyarrow, torch, the pose model) are imported inside the commands.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from signworld.cli.support import (
    ANALYSIS_CONFIG,
    AnalysisConfigOption,
    ConfigFiles,
    SourceArgument,
    load_analysis,
    open_source,
    read_source_manifest,
    reports_user_errors,
)

manifest_app = typer.Typer(help="Per-dataset clip manifests.", no_args_is_help=True)
logger = logging.getLogger(__name__)


@manifest_app.command("build")
@reports_user_errors
def build(source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """Write the clip manifest of one dataset from its annotations."""
    from signworld.data.corpus.builders import build_manifest

    settings = load_analysis(config)
    typer.echo(build_manifest(source, settings.acquisition()))


@reports_user_errors
def index(  # noqa: PLR0913, PLR0917 (typer options)
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Clip manifest.")],
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
    materialized: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Root of clips materialised by us."),
    ] = None,
    videos: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Flat folder of <clip>.mp4 (OpenASL)."),
    ] = None,
    poses: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Flat folder of <clip>.npz (OpenASL)."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Processes that read the poses.")] = 8,
) -> None:
    """Join manifest, materialised clips and captions into the training index, with splits."""
    from signworld.data.corpus.manifest import (
        read_manifest,
    )
    from signworld.data.corpus.materialize import MaterializedIndex
    from signworld.data.loaders import (
        build_training_index,
        records_from_files,
        write_index,
    )
    from signworld.data.text import EmbeddingStore
    from signworld.experiment.collaudo.contamination import contamination_report
    from signworld.experiment.train.config import load_config

    settings = load_config(*config).data
    corpus = read_manifest(manifest)
    excluded: frozenset[str] = frozenset()
    for path in benchmark or []:
        report = contamination_report(corpus, read_manifest(path))
        excluded |= report.excluded
        typer.echo(f"{path.name}: {len(report.excluded)} contaminated clips removed")
    if materialized is not None:
        records = MaterializedIndex(materialized).records()
    elif videos is not None and poses is not None:
        records = records_from_files(corpus, videos, poses)
    else:
        raise ValueError("give --materialized, or --videos and --poses")
    typer.echo(f"{len(records)} clips with video and pose")
    table = build_training_index(
        corpus,
        records,
        EmbeddingStore(embeddings).clip_rows(),
        settings,
        check_poses=check_poses,
        excluded=excluded,
        workers=workers,
    )
    write_index(table, output)
    splits = table.column("split").value_counts().to_pylist()
    typer.echo(
        f"{output}: {table.num_rows} clips; "
        + ", ".join(f"{entry['values']}={entry['counts']}" for entry in splits)
    )


@reports_user_errors
def materialize(
    source: Annotated[str, typer.Argument(help="Dataset source, e.g. youtube_sl25.")],
    output: Annotated[Path, typer.Option(file_okay=False, help="Root of the training clips.")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="analysis.yaml")
    ] = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the detector and pose model.")] = "cuda",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Stage 0 on the whole corpus: crop, 64 frames, pose, for every captioned clip (§3.6).

    The same pipeline as the test clips (``testdata build``), without the contiguous poses,
    written under ``output``, never next to the downloaded data. Resumes: clips already in the
    index are skipped; shards split the clips by a stable hash of their ID.
    """
    from signworld.data.acquisition.sharding import Shard
    from signworld.data.corpus.materialize import (
        ClipCut,
        ClipMaterializer,
        MaterializedIndex,
    )
    from signworld.data.pose.estimator import WholebodyEstimator

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    options = settings.test_data
    part = Shard(shard, num_shards)
    index = MaterializedIndex(output, part if num_shards > 1 else None)
    done = {record.clip_id for record in index.records()}
    columns = ["clip_id", "video_id", "start_s", "end_s", "video_available"]
    rows = read_source_manifest(dataset).select(columns).to_pylist()
    cuts = [
        ClipCut(row["clip_id"], row["video_id"], row["start_s"], row["end_s"])
        for row in rows
        if row["video_available"]
        and part.owns(row["clip_id"])
        and row["clip_id"] not in done
        and options.min_duration_s <= row["end_s"] - row["start_s"] <= options.max_duration_s
    ]
    materializer = ClipMaterializer(
        WholebodyEstimator(device=device),
        options.size,
        options.crop_margin,
        options.detection_frames,
        contiguous=False,
    )
    videos = dataset.layout.raw / "videos"
    skipped, failed = 0, 0
    for position, cut in enumerate(cuts, start=1):
        try:
            record = materializer.materialize(videos / f"{cut.video_id}.mp4", cut, output)
        except Exception:  # one unreadable clip must not end a run of millions
            logger.exception("%s: materialization failed", cut.clip_id)
            failed += 1
            continue
        if record is None:
            skipped += 1
            continue
        index.append(record)
        if position % 500 == 0:
            typer.echo(f"{position}/{len(cuts)} clips, {skipped} without a signer, {failed} failed")
    typer.echo(
        f"{source} {part.label}: {len(cuts)} clips; {skipped} without a signer, {failed} failed"
    )
