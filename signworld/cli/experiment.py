"""``signworld experiment``: design experiments that settle an open choice.

Their dependencies (torch, numpy) are imported inside the commands: the download jobs share
the entry point and must not pay that import on every start.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from signworld.cli.support import (
    ANALYSIS_CONFIG,
    AnalysisConfigOption,
    SourceArgument,
    labelled_clips,
    load_analysis,
    open_source,
    read_source_manifest,
    readout_corpus,
    reports_user_errors,
    testdata_root,
    video_encoder,
    video_settings,
)
from signworld.experiment.collaudo.results import result_path

if TYPE_CHECKING:
    from signworld.data.acquisition.sources.base import DatasetSource
    from signworld.experiment.collaudo.analysis import AnalysisConfig

experiment_app = typer.Typer(
    help="Design experiments that settle an open choice.", no_args_is_help=True
)


@experiment_app.command("pose-teachers")
@reports_user_errors
def pose_teachers(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the teachers' pre-training.")] = "cuda",
    checkpoint: Annotated[
        str, typer.Option(help="Uni-Sign checkpoint listed in the analysis settings.")
    ] = "unisign_csl_stage1",
    reuse_teachers: Annotated[
        bool, typer.Option(help="Load teachers already trained in the output folder.")
    ] = False,
) -> None:
    """§4.4.3: kinematic features, Uni-Sign, MAMP and S-JEPA as the physical-level target."""
    import torch

    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.pose_teachers import run

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    clips = labelled_clips(dataset, settings)
    path = settings.checkpoints[checkpoint].materialize(
        settings.models_root, settings.acquisition().http
    )
    root = testdata_root(dataset, settings) / "pose-teachers"
    report = run(
        clips,
        path,
        settings.pose_teachers,
        torch.device(device),
        root,
        progress=typer.echo,
        reuse=reuse_teachers,
    )
    typer.echo(f"channels per language: {report.channels_per_language}")
    for candidate in report.candidates:
        hands = [candidate.articulators[part] for part in ("left", "right")]
        typer.echo(
            f"{candidate.name:<9} hands R² position "
            f"{sum(h.r2_position for h in hands) / 2:.3f} · velocity "
            f"{sum(h.r2_velocity for h in hands) / 2:.3f} · channel "
            f"{candidate.channel_accuracy:.2f} (chance {candidate.channel_chance:.2f}) · "
            f"language {candidate.language_accuracy:.2f} "
            f"(chance {candidate.language_chance:.2f}) · "
            f"on unseen channels {candidate.language_accuracy_unseen_channels:.2f} "
            f"(chance {candidate.language_chance_unseen_channels:.2f})"
        )
    typer.echo(
        write_report(
            result_path("pc5-encoder-posa", f"{dataset.name}.json"), "pose-teachers", report
        )
    )


@experiment_app.command("pose-spectrum")
@reports_user_errors
def pose_spectrum(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device that encodes the clips.")] = "cuda",
) -> None:
    """§4.4.3: how many directions the S-JEPA latent uses, and its fixed whitening."""
    import torch

    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.pose_spectrum import measure, save_whitenings

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    root = testdata_root(dataset, settings) / "pose-teachers"
    weights = root / "sjepa.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: run `signworld experiment pose-teachers` first")
    report, whitenings = measure(
        labelled_clips(dataset, settings), weights, settings.pose_teachers, torch.device(device)
    )
    for part, result in report.articulators.items():
        chosen = next(t for t in result.truncations if t.k == result.chosen_k)
        typer.echo(
            f"{part:<9} erank {result.effective_rank:5.1f} · iso {result.isoscore:.3f} · "
            f"90 % in {result.directions_for_90}, 99 % in {result.directions_for_99} · "
            f"k = {result.chosen_k}: R² {chosen.r2_position:.3f} / {chosen.r2_velocity:.3f}, "
            f"iso after whitening {chosen.isoscore_whitened:.3f}"
        )
    typer.echo(
        save_whitenings(
            whitenings, result_path("spettro-posa", f"{dataset.name}.sjepa-whitening.npz")
        )
    )
    typer.echo(
        write_report(result_path("spettro-posa", f"{dataset.name}.json"), "pose-spectrum", report)
    )


def _records(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> dict[str, Any]:
    from signworld.data.corpus.materialize import MaterializedIndex

    return {r.clip_id: r for r in MaterializedIndex(testdata_root(dataset, settings)).records()}


@experiment_app.command("video-features")
@reports_user_errors
def video_features(
    source: SourceArgument,
    run: Annotated[str, typer.Option(help="Run listed under video_probes.runs.")],
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the encoder.")] = "cuda",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Frozen video features of the read-out clips for PC2, PC3, PC4 and the frame collaudo.

    A finished shard is not recomputed, so a failed submission can be resubmitted unchanged.
    """
    from signworld.data.acquisition.sharding import Shard
    from signworld.experiment.studies.video_probes import FrameSource, extract

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    video = video_settings(settings)
    options = video.runs[run]
    output = testdata_root(dataset, settings) / "video-features" / run
    path = output / f"shard-{shard:05d}-of-{num_shards:05d}.npz"
    if path.is_file():
        typer.echo(f"{path}: already extracted")
        return
    part = Shard(shard, num_shards)
    clips = readout_corpus(dataset, settings).clips[: options.clips]
    clips = [clip for clip in clips if part.owns(clip.clip_id)]
    encoder, _ = video_encoder(options.encoder, settings, device)
    frames = FrameSource(options, settings.test_data.size, dataset.layout.raw / "videos")
    typer.echo(f"{run}: {len(clips)} clips on shard {shard}/{num_shards}, {encoder.name}")
    features = extract(
        clips, _records(dataset, settings), frames, encoder, video.batch_size, typer.echo
    )
    typer.echo(features.save(path))


def _align(features: Any, corpus: Any) -> tuple[Any, Any]:
    """Features and corpus restricted to their common clips, in corpus order."""
    import numpy as np

    present = set(features.clip_ids)
    rows = np.array([i for i, c in enumerate(corpus.clips) if c.clip_id in present], dtype=int)
    subset = corpus.subset(rows)
    return features.select([c.clip_id for c in subset.clips]), subset


def _text_targets(dataset: "DatasetSource[Any]", corpus: Any) -> Any:
    import numpy as np

    from signworld.data.corpus.languages import written_language
    from signworld.data.text import EmbeddingStore
    from signworld.experiment.studies.video_probes import TextTargets
    from signworld.metrics.probes import video_split

    stores = sorted(dataset.layout.text_embeddings.glob("*/identity.json"))
    if len(stores) != 1:
        raise FileNotFoundError(f"{dataset.layout.text_embeddings}: expected one embedding set")
    store = EmbeddingStore(stores[0].parent)
    mapping = store.clip_rows().to_pydict()
    row_of = dict(zip(mapping["clip_id"], mapping["row"], strict=True))
    manifest = read_source_manifest(dataset).select(["clip_id", "caption_language"]).to_pydict()
    track_of = dict(zip(manifest["clip_id"], manifest["caption_language"], strict=True))
    ids = [clip.clip_id for clip in corpus.clips]
    languages = np.array([written_language(track_of[c]) if track_of[c] else "?" for c in ids])
    return TextTargets.of(
        np.array([row_of[c] for c in ids]),
        np.asarray(store.embeddings()),
        languages,
        ~video_split(corpus.videos),
    )


@experiment_app.command("video-probes")
@reports_user_errors
def video_probes(source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """PC2, PC3, PC4 and the selected-against-contiguous collaudo, from extracted features."""
    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.video_probes import ClipFeatures, run_scores

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    video = video_settings(settings)
    corpus = readout_corpus(dataset, settings)
    feature_root = testdata_root(dataset, settings) / "video-features"
    scores, aligned = {}, {}
    for name, options in video.runs.items():
        if options.frames != "selected":
            continue
        directory = feature_root / name
        if not directory.is_dir():
            typer.echo(f"{name}: no features, skipped")
            continue
        features, subset = _align(ClipFeatures.load(directory), corpus)
        targets = _text_targets(dataset, subset)
        result = run_scores(name, features, subset, targets, video.keypoint_clips)
        scores[name], aligned[name] = result, (features, subset)
        hands = " · ".join(
            f"{part} {s.r2_position:.3f}/{s.r2_velocity:.3f}" for part, s in result.hands.items()
        )
        typer.echo(
            f"{name}: {result.clips} clips; hands R² pos/vel {hands}; T2V R@1 "
            f"{result.text.t2v[1]:.3f} [{result.text.t2v_r1.low:.3f}, "
            f"{result.text.t2v_r1.high:.3f}] of {result.text.gallery}"
        )
        typer.echo(
            write_report(
                result_path("letture-video", f"{dataset.name}.{name}.json"),
                "video-probes",
                result,
            )
        )
    _video_decisions(dataset, settings, scores, aligned)
    if video.low_resolution_run in aligned:
        _frame_collaudo(
            dataset.name, settings, corpus, feature_root, aligned[video.low_resolution_run]
        )


def _video_decisions(
    dataset: "DatasetSource[Any]",
    settings: "AnalysisConfig",
    scores: dict[str, Any],
    aligned: dict[str, Any],
) -> None:
    """PC2, PC3 and PC4 from the read-outs of the selected-frame runs present."""
    import numpy as np

    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.video_probes import (
        baseline_report,
        choose_encoder,
        choose_resolution,
    )

    video = video_settings(settings)
    compared = [scores[run] for run in video.encoder_runs if run in scores]
    if compared:
        _, subset = aligned[compared[0].run]
        records = _records(dataset, settings)
        durations = np.array(
            [records[c.clip_id].frames / records[c.clip_id].fps for c in subset.clips]
        )
        baseline = baseline_report(
            durations,
            _text_targets(dataset, subset),
            subset.videos,
            {s.run: s.text for s in compared},
            video.seed,
        )
        typer.echo(
            f"PC2: chance R@1 {baseline.chance_r1:.4f}, "
            f"random {baseline.random_features.t2v[1]:.4f}, "
            f"duration {baseline.duration_only.t2v[1]:.4f}; floor {baseline.floor_r1:.4f}"
        )
        typer.echo(
            write_report(result_path("pc2-baseline", f"{dataset.name}.json"), "PC2", baseline)
        )
    if len(compared) >= 2:  # noqa: PLR2004
        choice = choose_encoder(compared, video.default_encoder_run)
        typer.echo(f"PC3: wins {choice.wins}; chosen {choice.chosen}")
        typer.echo(
            write_report(result_path("pc3-encoder-video", f"{dataset.name}.json"), "PC3", choice)
        )
    low, high = video.low_resolution_run, video.high_resolution_run
    if low in scores and high in scores:
        resolution = choose_resolution(
            scores[low],
            scores[high],
            video.runs[low].size,
            video.runs[high].size,
            video.min_hand_gain,
        )
        typer.echo(f"PC4: hand R² gain {resolution.gain:+.3f}; chosen {resolution.chosen}²")
        typer.echo(
            write_report(result_path("pc4-risoluzione", f"{dataset.name}.json"), "PC4", resolution)
        )


def _frame_collaudo(
    dataset_name: str,
    settings: "AnalysisConfig",
    corpus: Any,
    feature_root: Path,
    selected: tuple[Any, Any],
) -> None:
    """Selected against contiguous frames, on the clips both runs cover."""
    import numpy as np

    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.pose_teachers import PoseCorpus
    from signworld.experiment.studies.video_probes import ClipFeatures, frame_selection

    video = video_settings(settings)
    directory = feature_root / video.contiguous_run
    if not video.contiguous_run or not directory.is_dir():
        typer.echo(f"{video.contiguous_run}: no features, frame collaudo skipped")
        return
    contiguous = ClipFeatures.load(directory)
    extracted = set(contiguous.clip_ids)
    clips = [c for c in corpus.clips if c.clip_id in extracted]
    other = PoseCorpus.load(clips, settings.pose_teachers.min_score, "contiguous_")
    selected_features, selected_corpus = selected
    common = {c.clip_id for c in other.clips} & set(selected_features.clip_ids)

    def restricted(side: Any) -> Any:
        rows = [i for i, c in enumerate(side.clips) if c.clip_id in common]
        return side.subset(np.array(rows, dtype=int))

    mine, theirs = restricted(selected_corpus), restricted(other)
    ids = [c.clip_id for c in mine.clips]
    result = frame_selection(
        selected_features.select(ids),
        mine,
        contiguous.select(ids),
        theirs,
        video.frame_tolerance,
    )
    typer.echo(
        f"frames: selected {result.selected_r2:.3f} vs contiguous {result.contiguous_r2:.3f} "
        f"on {result.clips} clips; passed={result.passed}"
    )
    typer.echo(
        write_report(
            result_path("frame-selezionati-vs-contigui", f"{dataset_name}.json"),
            "A-frames",
            result,
        )
    )


@experiment_app.command("pose-isotropy")
@reports_user_errors
def pose_isotropy(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[
        str, typer.Option(help="Device of the teacher, the flows and k-NN.")
    ] = "cuda",
) -> None:
    """§4.4.3: whitening, RBIG, SINF and a flow on the S-JEPA latent, before and after."""
    import torch

    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.pose_isotropy import run

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    weights = testdata_root(dataset, settings) / "pose-teachers" / "sjepa.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: run `signworld experiment pose-teachers` first")
    report = run(
        labelled_clips(dataset, settings),
        weights,
        settings.pose_teachers,
        settings.pose_isotropy,
        torch.device(device),
        progress=typer.echo,
    )
    for result in report.configs:
        passed = [part for part, r in result.parts.items() if r.passed]
        typer.echo(
            f"{result.method:<9} {result.setting:<14} channel {result.channel_accuracy:.2f} "
            f"(chance {result.channel_chance:.2f}) · passes on {passed or 'none'}"
        )
    path = result_path("isotropizzazione-posa", f"{dataset.name}.json")
    typer.echo(write_report(path, "pose-isotropy", report))
