"""``signworld check``, ``text`` and ``testdata``: data checks and caption embeddings.

Their dependencies (torch, pyarrow, sentence-transformers) are imported inside the commands:
the download jobs share the entry point and must not pay that import on every start.
"""

import logging
from typing import TYPE_CHECKING, Annotated, Any

import typer

from signworld.cli.support import (
    ANALYSIS_CONFIG,
    AnalysisConfigOption,
    DeviceOption,
    SourceArgument,
    load_analysis,
    materialized_clips,
    open_source,
    read_source_manifest,
    readout_corpus,
    reports_user_errors,
    stored_frames,
    testdata_root,
    video_encoder,
    video_settings,
)
from signworld.experiment.collaudo.results import result_path

if TYPE_CHECKING:
    from signworld.data.acquisition.sources.base import DatasetSource
    from signworld.experiment.collaudo.analysis import AnalysisConfig

check_app = typer.Typer(help="Data checks of the collaudo (§4.13.1) and PC1.", no_args_is_help=True)
text_app = typer.Typer(help="Frozen caption embeddings (§4.4.4).", no_args_is_help=True)
testdata_app = typer.Typer(
    help="Clips cut, cropped and posed for the collaudo, kept apart from the datasets.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

MINOR_SECTION_FRACTION = 0.01

PromptOption = Annotated[
    str | None,
    typer.Option(help="Use another prompt than the configured one, to compare the two."),
]


@check_app.command("durations")
@reports_user_errors
def durations(source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """Caption duration distribution and the implied frame spacing, T/32."""
    from signworld.experiment.collaudo.durations import duration_report
    from signworld.experiment.collaudo.report import write_report

    dataset = open_source(source, load_analysis(config))
    report = duration_report(read_source_manifest(dataset))
    overall = report.overall
    typer.echo(
        f"{dataset.name}: {overall.clips} clips, median {overall.quantiles_s['p50']:.2f} s, "
        f"p99 {overall.quantiles_s['p99']:.2f} s, max {overall.max_s:.2f} s, "
        f"invalid {overall.invalid}"
    )
    typer.echo(
        write_report(result_path("durate-didascalie", f"{dataset.name}.json"), "durations", report)
    )


@check_app.command("contamination")
@reports_user_errors
def contamination(corpus: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """Corpus clips overlapping or repeating the benchmark's evaluation clips (§3.9, P2)."""
    from signworld.experiment.collaudo.contamination import contamination_report
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    options = settings.contamination
    dataset = open_source(corpus, settings)
    benchmark = open_source(options.benchmark, settings)
    report = contamination_report(
        read_source_manifest(dataset),
        read_source_manifest(benchmark),
        margin_s=options.margin_s,
        near_duplicate_ratio=options.near_duplicate_ratio,
    )
    exclusions = dataset.layout.manifest.with_name(f"excluded-{benchmark.name}.txt")
    exclusions.write_text("".join(f"{clip}\n" for clip in sorted(report.excluded)), "utf-8")
    typer.echo(
        f"{dataset.name} vs {benchmark.name}: {report.shared_videos} shared videos, "
        f"{report.overlapping} overlapping and {report.near_duplicate_captions} duplicated "
        f"clips, {len(report.excluded)} excluded of {report.corpus_clips}"
    )
    report_path = result_path(
        "contaminazione-benchmark", f"{dataset.name}.vs-{benchmark.name}.json"
    )
    typer.echo(write_report(report_path, "contamination", report))
    typer.echo(exclusions)


@check_app.command("split-duplicates")
@reports_user_errors
def split_duplicates(
    source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG
) -> None:
    """Same caption with the same duration in different splits (re-uploaded videos)."""
    from signworld.experiment.collaudo.contamination import duplicates_across_splits
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    groups = duplicates_across_splits(
        read_source_manifest(dataset), settings.contamination.duration_tolerance_s
    )
    typer.echo(f"{dataset.name}: {len(groups)} caption groups span more than one split")
    typer.echo(
        write_report(result_path("duplicati-fra-split", f"{dataset.name}.json"), "split", groups)
    )


@text_app.command("embed")
@reports_user_errors
def embed(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: DeviceOption = None,
    prompt: PromptOption = None,
) -> None:
    """Embed the unique captions of a manifest with the pinned model and prompt."""
    from signworld.data.text import SentenceTransformerEncoder, embed_manifest

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    encoder = SentenceTransformerEncoder(_embedding(settings, prompt), device=device)
    store = embed_manifest(
        read_source_manifest(dataset), encoder, encoder.identity, dataset.layout.text_embeddings
    )
    typer.echo(store.directory)


@text_app.command("verify")
@reports_user_errors
def verify(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: DeviceOption = None,
    prompt: PromptOption = None,
) -> None:
    """Collaudo of the prompt: same fingerprint, and 100 re-encoded captions match."""
    from signworld.data.text import EmbeddingStore, SentenceTransformerEncoder

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    encoder = SentenceTransformerEncoder(_embedding(settings, prompt), device=device)
    store = EmbeddingStore.for_identity(dataset.layout.text_embeddings, encoder.identity)
    if not store.directory.is_dir():
        raise FileNotFoundError(f"{store.directory}: run `signworld text embed {source}` first")
    if not _prompt_reproduced(dataset, store, encoder):
        raise typer.Exit(1)


@check_app.command("text-geometry")
@reports_user_errors
def text_geometry_check(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    fingerprint: Annotated[
        str, typer.Option(help="Prefix of the embedding fingerprint to analyse.")
    ] = "",
) -> None:
    """PC1: geometry of the caption targets at every MRL dimension, and the chosen d_MRL."""
    from signworld.data.text import EmbeddingStore

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    candidates = sorted(dataset.layout.text_embeddings.glob(f"{fingerprint}*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{dataset.layout.text_embeddings}: expected one embedding set matching "
            f"{fingerprint!r}, found {len(candidates)}; pass --fingerprint"
        )
    _text_geometry(dataset, EmbeddingStore(candidates[0]), settings)


@text_app.command("collaudo")
@reports_user_errors
def text_collaudo(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: DeviceOption = None,
) -> None:
    """A2 and PC1 in one run: embed if needed, check the prompt, then measure the geometry."""
    from signworld.data.text import EmbeddingStore, SentenceTransformerEncoder, embed_manifest

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    encoder = SentenceTransformerEncoder(settings.embedding, device=device)
    store = EmbeddingStore.for_identity(dataset.layout.text_embeddings, encoder.identity)
    if not (store.directory / store.IDENTITY_FILE).is_file():
        root = dataset.layout.text_embeddings
        store = embed_manifest(read_source_manifest(dataset), encoder, encoder.identity, root)
    reproduced = _prompt_reproduced(dataset, store, encoder)
    _text_geometry(dataset, store, settings)
    if not reproduced:
        raise typer.Exit(1)


def _prompt_reproduced(dataset: "DatasetSource[Any]", store: Any, encoder: Any) -> bool:
    """A2: same fingerprint, and re-encoded captions match the stored ones."""
    from signworld.data.text import verify_reproduction
    from signworld.experiment.collaudo.report import write_report

    report = verify_reproduction(store, encoder, encoder.identity)
    typer.echo(f"{dataset.name}: minimum cosine {report.min_cosine:.6f}, passed={report.passed}")
    typer.echo(
        write_report(result_path("prompt-embeddinggemma", f"{dataset.name}.json"), "A2", report)
    )
    return report.passed


def _text_geometry(dataset: "DatasetSource[Any]", store: Any, settings: "AnalysisConfig") -> None:
    """PC1 on one embedding set: print the table and write the report."""
    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.collaudo.text_geometry import text_geometry

    report = text_geometry(
        store.embeddings(), _row_languages(dataset, store), settings.text_geometry
    )
    typer.echo(f"{dataset.name}: {report.captions} captions, languages {report.languages}")
    for row in report.per_dimension:
        typer.echo(
            f"d={row.dimension:<4} erank={row.effective_rank:7.1f} iso={row.isoscore:.3f} "
            f"cos={row.mean_random_pair_cosine:+.3f} (before centring "
            f"{row.mean_random_pair_cosine_before_centering:+.3f}) sigreg={row.sigreg:.4f} "
            f"(gaussian {row.sigreg_gaussian_reference:.4f}) "
            f"language share {row.language_variance_fraction:.3f} "
            f"neighbours kept {row.neighbour_recall:.3f}"
        )
    typer.echo(f"chosen d_MRL = {report.chosen_dimension}")
    typer.echo(
        write_report(result_path("pc1-geometria-testo", f"{dataset.name}.json"), "PC1", report)
    )


@check_app.command("checkpoint")
@reports_user_errors
def checkpoint(
    name: Annotated[str, typer.Argument(help="Checkpoint listed in the analysis settings.")],
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
) -> None:
    """PC5/PC6: sections, parameter counts, shapes and normalisations of a checkpoint."""
    from signworld.data.acquisition.sources import UnknownSourceError
    from signworld.experiment.collaudo.checkpoint import inspect_checkpoint
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    if name not in settings.checkpoints:
        available = ", ".join(sorted(settings.checkpoints))
        raise UnknownSourceError(f"unknown checkpoint {name!r}; available: {available}")
    path = settings.checkpoints[name].materialize(settings.models_root, settings.acquisition().http)
    report = inspect_checkpoint(path)
    largest = max((summary.parameters for summary in report.sections.values()), default=0)
    minor = 0
    for section, summary in report.sections.items():
        if summary.parameters < MINOR_SECTION_FRACTION * largest:
            minor += 1
            continue
        typer.echo(
            f"{section}: {summary.tensors} tensors, {summary.parameters / 1e6:.1f} M parameters, "
            f"{len(summary.normalization_keys)} normalisation tensors"
        )
    if minor:
        typer.echo(f"({minor} sections under 1 % of the largest, e.g. optimizer state, omitted)")
    typer.echo(write_report(result_path("contenuto-checkpoint", f"{name}.json"), name, report))


def _embedding(settings: "AnalysisConfig", prompt: str | None) -> Any:
    """The configured embedding settings, with the prompt replaced when one is given."""
    if prompt is None:
        return settings.embedding
    return settings.embedding.model_copy(update={"prompt": prompt})


@testdata_app.command("build")
@reports_user_errors
def testdata_build(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the detector and pose model.")] = "cpu",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Cut a channel-stratified sample of clips, crop the signer, select frames, estimate poses.

    Resumes: clips already in the test index are skipped. Shards split the sample by a stable
    hash of the clip ID, so parallel jobs share no work and no file.
    """
    from signworld.data.acquisition.sharding import Shard
    from signworld.data.corpus.materialize import ClipMaterializer, MaterializedIndex, sample_cuts
    from signworld.data.pose.estimator import WholebodyEstimator

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    options = settings.test_data
    root = testdata_root(dataset, settings)
    part = Shard(shard, num_shards)
    index = MaterializedIndex(root, part if num_shards > 1 else None)
    done = {record.clip_id for record in index.records()}
    cuts = sample_cuts(
        read_source_manifest(dataset),
        options.clips,
        options.min_duration_s,
        options.max_duration_s,
        options.seed,
    )
    materializer = ClipMaterializer(
        WholebodyEstimator(device=device),
        options.size,
        options.crop_margin,
        options.detection_frames,
    )
    cuts = [cut for cut in cuts if part.owns(cut.clip_id)]
    videos = dataset.layout.raw / "videos"
    skipped, failed = 0, 0
    for position, cut in enumerate(cuts, start=1):
        if cut.clip_id in done:
            continue
        try:
            record = materializer.materialize(videos / f"{cut.video_id}.mp4", cut, root)
        except Exception:  # one unreadable clip must not end a run of thousands
            logger.exception("%s: materialization failed", cut.clip_id)
            failed += 1
            continue
        if record is None:
            skipped += 1
            continue
        index.append(record)
        if position % 50 == 0:
            typer.echo(f"{position}/{len(cuts)} clips, {skipped} without a signer, {failed} failed")
    typer.echo(
        f"{dataset.name}: {len(index.records())} clips under {root}; "
        f"{skipped} without a signer, {failed} failed"
    )


@check_app.command("unisign")
@reports_user_errors
def unisign(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    checkpoint: Annotated[
        str, typer.Option(help="Uni-Sign checkpoint listed in the analysis settings.")
    ] = "unisign_csl_stage1",
) -> None:
    """PC5: what Uni-Sign's frozen pose representation keeps, and where it falls short."""
    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.collaudo.unisign import PosedClip, unisign_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    manifest = (
        read_source_manifest(dataset).select(["clip_id", "video_id", "sign_language"]).to_pydict()
    )
    about = {
        clip: (video, language or "unknown")
        for clip, video, language in zip(*manifest.values(), strict=True)
    }
    clips = [
        PosedClip(clip.clip_id, *about[clip.clip_id], clip.pose)
        for clip in materialized_clips(dataset, settings)
        if clip.clip_id in about
    ]
    path = settings.checkpoints[checkpoint].materialize(
        settings.models_root, settings.acquisition().http
    )
    report = unisign_report(clips, path, settings.pose_checks.seed, progress=typer.echo)
    for part, result in report.articulators.items():
        typer.echo(
            f"{part:<9} R² released {result.r2_pretrained:.3f} · adapted BN "
            f"{result.r2_adapted_batch_norm:.3f} · random {result.r2_random_init:.3f} · "
            f"velocity {result.r2_velocity_pretrained:.3f} (random "
            f"{result.r2_velocity_random_init:.3f}) · contiguous "
            f"{result.r2_contiguous_own_probe:.3f}"
        )
    typer.echo(f"passed={report.passed} on {report.test_clips} held-out clips of {report.clips}")
    report_path = result_path("pc5-encoder-posa", f"{dataset.name}.unisign-{checkpoint}.json")
    typer.echo(write_report(report_path, "PC5", report))


@check_app.command("frame-selection")
@reports_user_errors
def frame_selection(source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """A3, first part: recomputing the frame selection reproduces the stored indices."""
    from signworld.experiment.collaudo.pose_alignment import frame_selection_report
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    options = settings.pose_checks
    report = frame_selection_report(materialized_clips(dataset, settings, options.selection_clips))
    typer.echo(f"{dataset.name}: {report.reproduced}/{report.clips} selections reproduced")
    typer.echo(write_report(result_path("selezione-frame", f"{dataset.name}.json"), "A3", report))
    if not report.passed:
        raise typer.Exit(1)


@check_app.command("pose-alignment")
@reports_user_errors
def pose_alignment(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the pose estimator.")] = "cpu",
) -> None:
    """A3: re-estimate the pose on decoded frames; compare with the stored pose and shifts."""
    from signworld.data.pose.estimator import WholebodyEstimator
    from signworld.experiment.collaudo.pose_alignment import pose_alignment_report
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    options = settings.pose_checks
    clips = materialized_clips(dataset, settings, options.alignment_clips)
    report = pose_alignment_report(clips, WholebodyEstimator(device=device), seed=options.seed)
    shifts = ", ".join(f"{s:+d}: {e:.4f}" for s, e in report.error_by_shift.items())
    typer.echo(
        f"{dataset.name}: hand error {report.hand_error_mean:.4f} of the frame width "
        f"(p95 {report.hand_error_p95:.4f}); by shift {shifts}; passed={report.passed}"
    )
    typer.echo(
        write_report(result_path("allineamento-video-posa", f"{dataset.name}.json"), "A3", report)
    )
    if not report.passed:
        raise typer.Exit(1)


@check_app.command("pose-quality")
@reports_user_errors
def pose_quality(source: SourceArgument, config: AnalysisConfigOption = ANALYSIS_CONFIG) -> None:
    """A4: shoulders, box coverage and normalised keypoints over every stored pose."""
    from signworld.experiment.collaudo.contact_sheet import contact_sheet
    from signworld.experiment.collaudo.pose_quality import pose_quality_report
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    options = settings.pose_checks
    everything = materialized_clips(dataset, settings)
    report = pose_quality_report(
        [clip.pose for clip in everything], options.workers, options.outside_limits
    )
    outside = ", ".join(f"{part} {share:.2%}" for part, share in report.outside_fraction.items())
    typer.echo(
        f"{dataset.name}: {report.clips} clips; frames without shoulders "
        f"{report.frames_without_shoulders_fraction:.2%}; boxes outside the frame: {outside}; "
        f"within 5 shoulder widths {report.within_five_shoulder_widths:.2%}; "
        f"passed={report.passed}"
    )
    for failure in report.failures:
        typer.echo(f"  failed: {failure}")
    typer.echo(write_report(result_path("qualita-posa", f"{dataset.name}.json"), "A4", report))
    sheets = result_path("qualita-posa", f"{dataset.name}.riquadri")
    sheets.mkdir(parents=True, exist_ok=True)
    for clip in everything[: options.contact_sheets]:
        contact_sheet(clip.video, clip.pose).convert("RGB").save(
            sheets / f"{clip.video.stem}.jpg", quality=85
        )
    typer.echo(sheets)
    if not report.passed:
        raise typer.Exit(1)


def _row_languages(dataset: "DatasetSource[Any]", store: Any) -> list[str]:
    """Written language of every embedding row, from the first clip that uses the caption."""
    from signworld.data.corpus.languages import written_language

    manifest = read_source_manifest(dataset).select(["clip_id", "caption_language"]).to_pydict()
    language_of = dict(zip(manifest["clip_id"], manifest["caption_language"], strict=True))
    rows = store.clip_rows().to_pydict()
    languages: dict[int, str] = {}
    for clip_id, row in zip(rows["clip_id"], rows["row"], strict=True):
        track = language_of.get(clip_id)
        languages.setdefault(row, written_language(track) if track else "unknown")
    return [languages[row] for row in range(len(languages))]


@check_app.command("video-reproduction")
@reports_user_errors
def video_reproduction_check(
    source: SourceArgument,
    encoder: Annotated[str, typer.Option(help="Encoder listed under video_probes.encoders.")],
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the encoder.")] = "cuda",
) -> None:
    """Collaudo: every weight loads, and our input path gives the official tokens."""
    from signworld.experiment.collaudo.model_reproduction import video_reproduction
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    video = video_settings(settings)
    frozen, extra = video_encoder(encoder, settings, device)
    load = extra[1] if extra is not None else {}
    clips = readout_corpus(dataset, settings).clips[: video.reference_clips]
    report = video_reproduction(frozen, [stored_frames(c) for c in clips], load)
    typer.echo(
        f"{encoder}: fp32 min cosine {report.fp32.min_cosine:.6f}, max diff "
        f"{report.fp32.max_abs_difference:.2e}; "
        f"BGR control {report.swapped_channels.min_cosine:.4f}; "
        f"bf16 mean cosine {report.bf16.mean_cosine:.5f}; passed={report.passed}"
    )
    path = result_path("riproduzione-pesi-video", f"{dataset.name}.{encoder}.json")
    typer.echo(write_report(path, "A1-video", report))
    if not report.passed:
        raise typer.Exit(1)


@check_app.command("pose-reproduction")
@reports_user_errors
def pose_reproduction_check(
    source: SourceArgument,
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the teacher.")] = "cuda",
) -> None:
    """Collaudo: the saved S-JEPA teacher reloads whole and reproduces its reference latents."""
    import torch

    from signworld.experiment.collaudo.model_reproduction import pose_reproduction
    from signworld.experiment.collaudo.report import write_report
    from signworld.experiment.studies.pose_teachers import teacher_shape
    from signworld.models.encoders.pose_teachers import SJEPATeacher

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    video = video_settings(settings)
    root = testdata_root(dataset, settings) / "pose-teachers"
    options = settings.pose_teachers
    teacher = SJEPATeacher(
        teacher_shape(options), momentum=(options.ema_start, 1.0), centre_rate=options.centre_rate
    )
    teacher.load_state_dict(torch.load(root / "sjepa.pt", map_location="cpu"), strict=True)
    teacher.to(torch.device(device)).eval()
    tokens = torch.from_numpy(readout_corpus(dataset, settings).tokens[: video.reference_clips])
    report = pose_reproduction(
        lambda t: teacher.encode(t.to(device)), tokens, root / "sjepa-reference.npz"
    )
    state = "reference written"
    if report.agreement is not None:
        state = f"min cosine {report.agreement.min_cosine:.6f}"
    typer.echo(f"S-JEPA: {state}; deterministic={report.deterministic}; passed={report.passed}")
    typer.echo(
        write_report(
            result_path("riproduzione-pesi-posa", f"{dataset.name}.json"), "A1-pose", report
        )
    )
    if not report.passed:
        raise typer.Exit(1)


@check_app.command("predictor")
@reports_user_errors
def predictor_check(
    source: SourceArgument,
    encoder: Annotated[str, typer.Option(help="A V-JEPA 2.1 encoder of video_probes.encoders.")],
    config: AnalysisConfigOption = ANALYSIS_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the model.")] = "cuda",
) -> None:
    """PC6: what the predictor holds, and the multi-level input reproducing it at zero."""
    import torch

    from signworld.experiment.collaudo.model_reproduction import predictor_report
    from signworld.experiment.collaudo.report import write_report

    settings = load_analysis(config)
    dataset = open_source(source, settings)
    frozen, extra = video_encoder(encoder, settings, device)
    if extra is None:
        raise ValueError(f"{encoder}: PC6 applies to V-JEPA 2.1 checkpoints only")
    predictor, load = extra
    predictor.to(torch.device(device)).eval()
    clip = readout_corpus(dataset, settings).clips[0]
    report = predictor_report(encoder, frozen, predictor, load, stored_frames(clip))
    typer.echo(
        f"{encoder}: levels {report.hierarchical_layers}, {report.per_level_norms} level norms "
        f"(drift from init {[round(d, 3) for d in report.level_norm_drift]}); "
        f"predictor {report.predictor_input} -> {report.predictor_output}, depth "
        f"{report.predictor_depth}, trained mask tokens {report.trained_mask_tokens}; "
        f"fusion difference {report.fusion_difference:.2e}; passed={report.passed}"
    )
    typer.echo(write_report(result_path("pc6-predictor", f"{encoder}.json"), "PC6", report))
    if not report.passed:
        raise typer.Exit(1)
