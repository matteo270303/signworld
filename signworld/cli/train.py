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
    from .checks.contamination import contamination_report  # noqa: PLC0415
    from .corpus.manifest import read_manifest  # noqa: PLC0415 (heavy imports on demand)
    from .corpus.materialize import MaterializedIndex  # noqa: PLC0415
    from .text.embedding import EmbeddingStore  # noqa: PLC0415
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.data import (  # noqa: PLC0415
        build_training_index,
        records_from_files,
        write_index,
    )

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


@train_app.command("materialize")
@reports_user_errors
def materialize(
    source: Annotated[str, typer.Argument(help="Dataset source, e.g. youtube_sl25.")],
    output: Annotated[Path, typer.Option(file_okay=False, help="Root of the training clips.")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="analysis.yaml")
    ] = Path("configs/analysis.yaml"),
    device: Annotated[str, typer.Option(help="Device of the detector and pose model.")] = "cuda",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Stage 0 on the whole corpus: crop, 64 frames, pose, for every captioned clip (§3.6).

    The same pipeline as the test clips (``testdata build``), without the contiguous poses,
    written under ``output``, never next to the downloaded data. Resumes: clips already in the
    index are skipped; shards split the clips by a stable hash of their ID.
    """
    from .acquisition.sharding import Shard  # noqa: PLC0415
    from .analysis_cli import _load, _manifest, _source  # noqa: PLC0415
    from .corpus.materialize import ClipCut, ClipMaterializer, MaterializedIndex  # noqa: PLC0415
    from .pose.estimator import WholebodyEstimator  # noqa: PLC0415

    settings = _load(config)
    dataset = _source(source, settings)
    options = settings.test_data
    part = Shard(shard, num_shards)
    index = MaterializedIndex(output, part if num_shards > 1 else None)
    done = {record.clip_id for record in index.records()}
    columns = ["clip_id", "video_id", "start_s", "end_s", "video_available"]
    rows = _manifest(dataset).select(columns).to_pylist()
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


@train_app.command("evaluate")
@reports_user_errors
def evaluate(  # noqa: PLR0913, PLR0917 (typer options)
    config: ConfigFiles,
    run_directory: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    index: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Clips to evaluate.")],
    output: Annotated[Path, typer.Option(dir_okay=False, help="Report to write (JSON).")],
    checkpoint: Annotated[str, typer.Option(help="best, final, or another name.")] = "best",
    split: Annotated[str | None, typer.Option(help="Only this split of the index.")] = None,
    clips: Annotated[int | None, typer.Option(min=1, help="A fixed subset of the clips.")] = None,
    masks: Annotated[int, typer.Option(min=1, help="Masks per clip for Ē_fis.")] = 8,
    gate: Annotated[bool, typer.Option(help="Read the gate (F4): OpenASL test clips.")] = False,
    ridge_baseline: Annotated[
        float | None, typer.Option(help="R@1 T2V of the ridge baseline of PC2, as a fraction.")
    ] = None,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Final evaluation of a checkpoint (§4.12.2-§4.12.4) on the clips of an index."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from .text.embedding import EmbeddingStore  # noqa: PLC0415
    from .worldmodel.checkpoint import CheckpointStore  # noqa: PLC0415
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.curriculum import trainable_names  # noqa: PLC0415
    from .worldmodel.data import (  # noqa: PLC0415
        ClipDataset,
        Collate,
        read_index,
        validation_subset,
    )
    from .worldmodel.evaluation import evaluate as measure  # noqa: PLC0415
    from .worldmodel.model import build_worldsign  # noqa: PLC0415
    from .worldmodel.run import STATISTICS, load_statistics  # noqa: PLC0415

    settings = load_config(*config)
    if settings.data.embeddings is None:
        raise ValueError("data.embeddings must name the caption embeddings of these clips")
    model = build_worldsign(settings)
    load_statistics(model, torch.load(run_directory / STATISTICS, weights_only=False))
    CheckpointStore(run_directory / "checkpoints", trainable_names(model)).load(
        checkpoint, model, None
    )
    table = read_index(index, split)
    known = pa.array(list(model.text.centering.languages))
    table = table.filter(pc.is_in(table.column("caption_language"), known))
    if clips is not None:
        table = validation_subset(table, clips)
    signs = table.column("sign_language").to_pylist()
    names = sorted({str(s) for s in signs})
    dataset = ClipDataset(
        table,
        EmbeddingStore(settings.data.embeddings).embeddings(),
        augmentation=None,
        box_threshold=settings.physical.box_threshold,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=Collate(model.text.centering.languages),
        num_workers=settings.data.workers,
    )
    target = torch.device(device)
    report = measure(
        model.to(target),
        loader,
        target,
        settings,
        torch.tensor([names.index(str(s)) for s in signs]),
        table.column("video_id").to_pylist(),
        masks=masks,
        ridge_baseline_r1=ridge_baseline,
        gate=gate,
    )
    report.write(output)
    found = report.measures
    typer.echo(
        f"{output}: T2V R@1 {found['t2v_r1']:.4f} "
        f"[{found['t2v_r1_low']:.4f}, {found['t2v_r1_high']:.4f}]"
        + (f"; gate: {report.gate}" if report.gate else "")
    )


@train_app.command("overfit")
@reports_user_errors
def overfit_command(
    config: ConfigFiles,
    output: Annotated[Path, typer.Option(dir_okay=False, help="Result to write (JSON).")],
    clips: Annotated[int, typer.Option(min=2)] = 64,
    steps: Annotated[int, typer.Option(min=1)] = 300,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Collaudo «overfitting controllato» (§4.13.1): a few real clips, every loss on."""
    import json  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from .text.embedding import EmbeddingStore  # noqa: PLC0415
    from .worldmodel.collaudo import overfit  # noqa: PLC0415
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.data import (  # noqa: PLC0415
        TRAIN,
        ClipDataset,
        Collate,
        read_index,
        validation_subset,
    )
    from .worldmodel.model import build_worldsign  # noqa: PLC0415
    from .worldmodel.run import fit_statistics  # noqa: PLC0415

    settings = load_config(*config)
    if settings.data.index is None or settings.data.embeddings is None:
        raise ValueError("data.index and data.embeddings must name the stage-0 outputs")
    model = build_worldsign(settings)
    embeddings = EmbeddingStore(settings.data.embeddings).embeddings()
    train = read_index(settings.data.index, TRAIN)
    fit_statistics(model, train, embeddings, settings.data.statistics_clips)
    subset = validation_subset(train, clips)
    dataset = ClipDataset(
        subset, embeddings, augmentation=None, box_threshold=settings.physical.box_threshold
    )
    target = torch.device(device)
    batch = Collate(model.text.centering.languages)([dataset[i] for i in range(len(dataset))])
    result = overfit(model.to(target), batch.to(target), steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    typer.echo(f"{output}: {'passed' if result.passed else 'FAILED'} {asdict(result)}")


@train_app.command("benchmark")
@reports_user_errors
def benchmark(
    config: ConfigFiles,
    output: Annotated[Path, typer.Option(file_okay=False, help="Scratch run directory.")],
    steps: Annotated[int, typer.Option(min=1)] = 200,
    peak_tflops: Annotated[float, typer.Option(help="Peak bf16 TFLOP/s of one GPU.")] = 989.0,
) -> None:
    """Collaudo «efficienza di calcolo» (§4.13.1, PC7): loader, step time, MFU, memory."""
    import json  # noqa: PLC0415

    from .worldmodel.collaudo import measure_efficiency  # noqa: PLC0415
    from .worldmodel.config import load_config  # noqa: PLC0415
    from .worldmodel.distributed import Distributed  # noqa: PLC0415
    from .worldmodel.run import prepare  # noqa: PLC0415

    collective = Distributed.from_environment()
    try:
        prepared = prepare(load_config(*config), output, collective)
        result = measure_efficiency(prepared.trainer, steps, peak_flops=peak_tflops * 1e12)
    finally:
        collective.shutdown()
    if collective.is_main:
        (output / "efficiency.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
        typer.echo(f"{output}: {result.as_dict()}")


@train_app.command("toyworld")
@reports_user_errors
def toyworld(
    output: Annotated[Path, typer.Option(file_okay=False, help="Root of the toy world.")],
    clips: Annotated[int, typer.Option(min=1)] = 20_000,
    shard: Annotated[int, typer.Option(min=0)] = 0,
    num_shards: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Collaudo «mondo giocattolo» (§4.13.1): synthetic clips in the materialised format."""
    from .worldmodel.toyworld import build  # noqa: PLC0415

    written = build(output, clips, shard=shard, shards=num_shards, seed=seed)
    typer.echo(f"{output}: {written} clips written by shard {shard}/{num_shards}")


@train_app.command("toyworld-embed")
@reports_user_errors
def toyworld_embed(
    output: Annotated[Path, typer.Option(exists=True, file_okay=False, help="The toy world.")],
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """EmbeddingGemma rows of the toy captions, with the pinned model and prompt (§4.4.4)."""
    from .text.embedding import (  # noqa: PLC0415
        EmbeddingSettings,
        SentenceTransformerEncoder,
        embed_manifest,
    )
    from .worldmodel.toyworld import manifest_table  # noqa: PLC0415

    encoder = SentenceTransformerEncoder(EmbeddingSettings(), device=device)
    store = embed_manifest(manifest_table(output), encoder, encoder.identity, output / "text")
    typer.echo(f"{store.directory}: {len(store.captions())} captions")
