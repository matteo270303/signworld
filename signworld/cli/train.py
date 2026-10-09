"""``signworld train``: the training index of stage 0 and the training runs (§4.10)."""

from pathlib import Path
from typing import Annotated

import typer

from signworld.cli.corpus import index, materialize
from signworld.cli.support import ConfigFiles, reports_user_errors

train_app = typer.Typer(help="Train WorldSign.", no_args_is_help=True)

# Stage 0 lives with the corpus commands; it is listed first, as it runs first.
train_app.command("index")(index)
train_app.command("materialize")(materialize)


@train_app.command("run")
@reports_user_errors
def run(
    config: ConfigFiles,
    output: Annotated[Path, typer.Option(file_okay=False, help="Run directory.")],
    allow_code_change: Annotated[
        bool,
        typer.Option(help="Resume although the code differs from the first launch (recorded)."),
    ] = False,
) -> None:
    """Train, or resume, one run; launch with torchrun for several GPUs."""
    from signworld.experiment.train.config import load_config
    from signworld.experiment.train.distributed import Distributed
    from signworld.experiment.train.run import run as start

    collective = Distributed.from_environment()
    try:
        state = start(load_config(*config), output, collective, allow_code_change=allow_code_change)
    finally:
        collective.shutdown()
    if collective.is_main:
        typer.echo(f"{output}: finished at step {state.step}, best at {state.best_step}")


@train_app.command("ridge-baseline")
@reports_user_errors
def ridge_baseline_command(  # noqa: PLR0913, PLR0917 (typer options)
    config: ConfigFiles,
    output: Annotated[Path, typer.Option(dir_okay=False, help="Report to write (JSON).")],
    split: Annotated[
        list[str], typer.Option(help="Splits to score; the first chooses the penalty.")
    ],
    device: Annotated[str, typer.Option(help="Device of the encoder.")] = "cuda",
    fit_clips: Annotated[int, typer.Option(min=1, help="Training clips the ridge fits.")] = 20_000,
    clips: Annotated[
        int | None, typer.Option(min=1, help="Clips of each split but validation (all if unset).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed of the bootstrap intervals.")] = 0,
) -> None:
    """PC2's ridge from frozen V-JEPA 2.1 features on the run's own splits (§4.12.1)."""
    import torch

    from signworld.experiment.evaluation.ridge import ridge_baseline, write_ridge_report
    from signworld.experiment.train.config import load_config

    report = ridge_baseline(
        load_config(*config),
        split,
        torch.device(device),
        fit_clips=fit_clips,
        clips=clips,
        seed=seed,
    )
    baseline = report["ridge_baseline"]
    typer.echo(write_ridge_report(report, output))
    typer.echo(
        f"diagnostics.ridge_baseline: {{t2v: {baseline['t2v']:.4f}, v2t: {baseline['v2t']:.4f}}}"
    )


@train_app.command("report")
@reports_user_errors
def report_command(
    run: Annotated[
        Path, typer.Option("--run", exists=True, file_okay=False, help="Run directory.")
    ],
    evaluation: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Its `train evaluate` JSON.")
    ] = None,
    ridge: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="The `train ridge-baseline` JSON."),
    ] = None,
    split: Annotated[str | None, typer.Option(help="The split the evaluation read.")] = None,
) -> None:
    """The run's report: identity, training, stops, evaluation and the triage of its alarms."""
    from signworld.experiment.evaluation.report import analyse_run, write_report

    for path in write_report(analyse_run(run, evaluation, ridge, split), run):
        typer.echo(path)


@train_app.command("compare")
@reports_user_errors
def compare_command(
    run: Annotated[list[Path], typer.Option(exists=True, file_okay=False, help="Run directory.")],
    evaluation: Annotated[
        list[Path],
        typer.Option(exists=True, dir_okay=False, help="Its evaluation JSON, in the same order."),
    ],
    output: Annotated[Path, typer.Option(file_okay=False, help="Directory to write.")],
    k: Annotated[int, typer.Option(min=1, help="R@k compared.")] = 1,
) -> None:
    """The ablation tables: planned contrasts with a paired bootstrap over the same queries."""
    from signworld.experiment.evaluation.compare import compare_runs, write_comparison

    if len(run) != len(evaluation):
        raise typer.BadParameter("one --evaluation per --run", param_hint="--evaluation")
    for path in write_comparison(compare_runs(list(zip(run, evaluation, strict=True)), k), output):
        typer.echo(path)


@train_app.command("fetch-models")
@reports_user_errors
def fetch_models(config: ConfigFiles) -> None:
    """Download the V-JEPA 2.1 weights if absent (run on a node with internet access).

    The hub code is never downloaded: it must already sit in ``encoder.hub_repo``.
    """
    from signworld.experiment.train.config import load_config
    from signworld.models.encoders.weights import VJEPA2_1_VITL_384_URL, ensure_encoder_files

    encoder = load_config(*config).encoder
    ensure_encoder_files(
        encoder.hub_repo,
        encoder.checkpoint,
        url=encoder.checkpoint_url or VJEPA2_1_VITL_384_URL,
        sha256=encoder.checkpoint_sha256,
    )
    typer.echo(f"{encoder.checkpoint}: present")


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
    ridge_t2v: Annotated[
        float | None, typer.Option(help="T2V R@1 of the ridge baseline of PC2, as a fraction.")
    ] = None,
    ridge_v2t: Annotated[
        float | None, typer.Option(help="V2T R@1 of the ridge baseline of PC2, as a fraction.")
    ] = None,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Final evaluation of a checkpoint (§4.12.2-§4.12.4) on the clips of an index."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import torch
    from torch.utils.data import DataLoader

    from signworld.data.loaders import (
        ClipDataset,
        Collate,
        read_index,
        validation_subset,
    )
    from signworld.data.text import EmbeddingStore
    from signworld.experiment.evaluation.worldsign import evaluate as measure
    from signworld.experiment.train.checkpoint import CheckpointStore
    from signworld.experiment.train.config import load_config
    from signworld.experiment.train.curriculum import trainable_names
    from signworld.experiment.train.run import (
        STATISTICS,
        false_negative_thresholds,
        load_statistics,
    )
    from signworld.metrics.directions import DIRECTIONS, Bidirectional
    from signworld.models.worldsign.model import build_worldsign

    if (ridge_t2v is None) != (ridge_v2t is None):
        raise typer.BadParameter(
            "give the ridge baseline in both directions, or in neither",
            param_hint="--ridge-t2v/--ridge-v2t",
        )
    ridge = None if ridge_t2v is None or ridge_v2t is None else Bidirectional(ridge_t2v, ridge_v2t)
    settings = load_config(*config)
    if settings.data.embeddings is None:
        raise ValueError("data.embeddings must name the caption embeddings of these clips")
    model = build_worldsign(settings, false_negatives=false_negative_thresholds(settings))
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
        channels=table.column("channel_id").to_pylist(),
        masks=masks,
        ridge_baseline=ridge,
        gate=gate,
    )
    report.write(output)
    found = report.measures
    recalls = " · ".join(
        f"{d.upper()} R@1 {found[f'{d}_r1']:.4f} "
        f"[{found[f'{d}_r1_low']:.4f}, {found[f'{d}_r1_high']:.4f}]"
        for d in DIRECTIONS
    )
    typer.echo(f"{output}: {recalls}" + (f"; gate: {report.gate}" if report.gate else ""))


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
    import json
    from dataclasses import asdict

    import torch

    from signworld.data.loaders import (
        TRAIN,
        ClipDataset,
        Collate,
        read_index,
        validation_subset,
    )
    from signworld.data.text import EmbeddingStore
    from signworld.experiment.collaudo.worldsign import overfit
    from signworld.experiment.train.config import load_config
    from signworld.experiment.train.run import false_negative_thresholds, fit_statistics
    from signworld.models.worldsign.model import build_worldsign

    settings = load_config(*config)
    if settings.data.index is None or settings.data.embeddings is None:
        raise ValueError("data.index and data.embeddings must name the stage-0 outputs")
    model = build_worldsign(settings, false_negatives=false_negative_thresholds(settings))
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
    import json

    from signworld.experiment.collaudo.worldsign import measure_efficiency
    from signworld.experiment.train.config import load_config
    from signworld.experiment.train.distributed import Distributed
    from signworld.experiment.train.run import prepare

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
    from signworld.data.toyworld import build

    written = build(output, clips, shard=shard, shards=num_shards, seed=seed)
    typer.echo(f"{output}: {written} clips written by shard {shard}/{num_shards}")


@train_app.command("toyworld-embed")
@reports_user_errors
def toyworld_embed(
    output: Annotated[Path, typer.Option(exists=True, file_okay=False, help="The toy world.")],
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """EmbeddingGemma rows of the toy captions, with the pinned model and prompt (§4.4.4)."""
    from signworld.data.text import (
        EmbeddingSettings,
        SentenceTransformerEncoder,
        embed_manifest,
    )
    from signworld.data.toyworld import manifest_table

    encoder = SentenceTransformerEncoder(EmbeddingSettings(), device=device)
    store = embed_manifest(manifest_table(output), encoder, encoder.identity, output / "text")
    typer.echo(f"{store.directory}: {len(store.captions())} captions")
