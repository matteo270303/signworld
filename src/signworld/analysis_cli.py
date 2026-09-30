"""Commands for manifests, data checks and caption embeddings.

Their dependencies (torch, pyarrow, sentence-transformers) are imported inside the commands:
the download jobs share the entry point and must not pay that import on every start.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from .cli_support import SourceArgument, reports_user_errors
from .collaudo import result_path

if TYPE_CHECKING:
    from .acquisition.sources.base import DatasetSource
    from .analysis import AnalysisConfig

manifest_app = typer.Typer(help="Per-dataset clip manifests.", no_args_is_help=True)
check_app = typer.Typer(help="Data checks of the collaudo (§4.13.1) and PC1.", no_args_is_help=True)
text_app = typer.Typer(help="Frozen caption embeddings (§4.4.4).", no_args_is_help=True)
testdata_app = typer.Typer(
    help="Clips cut, cropped and posed for the collaudo, kept apart from the datasets.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(
    help="Design experiments that settle an open choice.", no_args_is_help=True
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis.yaml")
MINOR_SECTION_FRACTION = 0.01

DeviceOption = Annotated[str | None, typer.Option(help="Torch device, e.g. cuda.")]
PromptOption = Annotated[
    str | None,
    typer.Option(help="Use another prompt than the configured one, to compare the two."),
]

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        envvar="SIGNWORLD_ANALYSIS_CONFIG",
        exists=True,
        dir_okay=False,
        help="Analysis settings (YAML).",
    ),
]


def _load(config: Path) -> "AnalysisConfig":
    import torch

    from .analysis import AnalysisConfig

    settings = AnalysisConfig.from_yaml(config)
    torch.set_num_threads(settings.threads)
    return settings


def _source(name: str, settings: "AnalysisConfig") -> "DatasetSource[Any]":
    from .acquisition.sources import build_source

    return build_source(name, settings.acquisition())


def _manifest(source: "DatasetSource[Any]") -> Any:
    from .corpus.manifest import read_manifest

    path = source.layout.manifest
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run `signworld manifest build {source.name}` first")
    return read_manifest(path)


@manifest_app.command("build")
@reports_user_errors
def build(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Write the clip manifest of one dataset from its annotations."""
    from .corpus.builders import build_manifest

    settings = _load(config)
    typer.echo(build_manifest(source, settings.acquisition()))


@check_app.command("durations")
@reports_user_errors
def durations(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Caption duration distribution and the implied frame spacing, T/32."""
    from .checks.durations import duration_report
    from .checks.report import write_report

    dataset = _source(source, _load(config))
    report = duration_report(_manifest(dataset))
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
def contamination(corpus: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Corpus clips overlapping or repeating the benchmark's evaluation clips (§3.9, P2)."""
    from .checks.contamination import contamination_report
    from .checks.report import write_report

    settings = _load(config)
    options = settings.contamination
    dataset = _source(corpus, settings)
    benchmark = _source(options.benchmark, settings)
    report = contamination_report(
        _manifest(dataset),
        _manifest(benchmark),
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
def split_duplicates(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Same caption with the same duration in different splits (re-uploaded videos)."""
    from .checks.contamination import duplicates_across_splits
    from .checks.report import write_report

    settings = _load(config)
    dataset = _source(source, settings)
    groups = duplicates_across_splits(
        _manifest(dataset), settings.contamination.duration_tolerance_s
    )
    typer.echo(f"{dataset.name}: {len(groups)} caption groups span more than one split")
    typer.echo(
        write_report(result_path("duplicati-fra-split", f"{dataset.name}.json"), "split", groups)
    )


@text_app.command("embed")
@reports_user_errors
def embed(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    device: DeviceOption = None,
    prompt: PromptOption = None,
) -> None:
    """Embed the unique captions of a manifest with the pinned model and prompt."""
    from .text.embedding import SentenceTransformerEncoder, embed_manifest

    settings = _load(config)
    dataset = _source(source, settings)
    encoder = SentenceTransformerEncoder(_embedding(settings, prompt), device=device)
    store = embed_manifest(
        _manifest(dataset), encoder, encoder.identity, dataset.layout.text_embeddings
    )
    typer.echo(store.directory)


@text_app.command("verify")
@reports_user_errors
def verify(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    device: DeviceOption = None,
    prompt: PromptOption = None,
) -> None:
    """Collaudo of the prompt: same fingerprint, and 100 re-encoded captions match."""
    from .text.embedding import EmbeddingStore, SentenceTransformerEncoder

    settings = _load(config)
    dataset = _source(source, settings)
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
    config: ConfigOption = DEFAULT_CONFIG,
    fingerprint: Annotated[
        str, typer.Option(help="Prefix of the embedding fingerprint to analyse.")
    ] = "",
) -> None:
    """PC1: geometry of the caption targets at every MRL dimension, and the chosen d_MRL."""
    from .text.embedding import EmbeddingStore

    settings = _load(config)
    dataset = _source(source, settings)
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
    config: ConfigOption = DEFAULT_CONFIG,
    device: DeviceOption = None,
) -> None:
    """A2 and PC1 in one run: embed if needed, check the prompt, then measure the geometry."""
    from .text.embedding import EmbeddingStore, SentenceTransformerEncoder, embed_manifest

    settings = _load(config)
    dataset = _source(source, settings)
    encoder = SentenceTransformerEncoder(settings.embedding, device=device)
    store = EmbeddingStore.for_identity(dataset.layout.text_embeddings, encoder.identity)
    if not (store.directory / store.IDENTITY_FILE).is_file():
        root = dataset.layout.text_embeddings
        store = embed_manifest(_manifest(dataset), encoder, encoder.identity, root)
    reproduced = _prompt_reproduced(dataset, store, encoder)
    _text_geometry(dataset, store, settings)
    if not reproduced:
        raise typer.Exit(1)


def _prompt_reproduced(dataset: "DatasetSource[Any]", store: Any, encoder: Any) -> bool:
    """A2: same fingerprint, and re-encoded captions match the stored ones."""
    from .checks.report import write_report
    from .text.embedding import verify_reproduction

    report = verify_reproduction(store, encoder, encoder.identity)
    typer.echo(f"{dataset.name}: minimum cosine {report.min_cosine:.6f}, passed={report.passed}")
    typer.echo(
        write_report(result_path("prompt-embeddinggemma", f"{dataset.name}.json"), "A2", report)
    )
    return report.passed


def _text_geometry(dataset: "DatasetSource[Any]", store: Any, settings: "AnalysisConfig") -> None:
    """PC1 on one embedding set: print the table and write the report."""
    from .checks.report import write_report
    from .checks.text_geometry import text_geometry

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
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """PC5/PC6: sections, parameter counts, shapes and normalisations of a checkpoint."""
    from .acquisition.sources import UnknownSourceError
    from .checks.checkpoint import inspect_checkpoint
    from .checks.report import write_report

    settings = _load(config)
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
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the detector and pose model.")] = "cpu",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Cut a channel-stratified sample of clips, crop the signer, select frames, estimate poses.

    Resumes: clips already in the test index are skipped. Shards split the sample by a stable
    hash of the clip ID, so parallel jobs share no work and no file.
    """
    from .acquisition.sharding import Shard
    from .corpus.materialize import ClipMaterializer, MaterializedIndex, sample_cuts
    from .pose.estimator import WholebodyEstimator

    settings = _load(config)
    dataset = _source(source, settings)
    options = settings.test_data
    root = _test_root(dataset, settings)
    part = Shard(shard, num_shards)
    index = MaterializedIndex(root, part if num_shards > 1 else None)
    done = {record.clip_id for record in index.records()}
    cuts = sample_cuts(
        _manifest(dataset),
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
    config: ConfigOption = DEFAULT_CONFIG,
    checkpoint: Annotated[
        str, typer.Option(help="Uni-Sign checkpoint listed in the analysis settings.")
    ] = "unisign_csl_stage1",
) -> None:
    """PC5: what Uni-Sign's frozen pose representation keeps, and where it falls short."""
    from .checks.report import write_report
    from .checks.unisign import PosedClip, unisign_report

    settings = _load(config)
    dataset = _source(source, settings)
    manifest = _manifest(dataset).select(["clip_id", "video_id", "sign_language"]).to_pydict()
    about = {
        clip: (video, language or "unknown")
        for clip, video, language in zip(*manifest.values(), strict=True)
    }
    clips = [
        PosedClip(clip.clip_id, *about[clip.clip_id], clip.pose)
        for clip in _materialized(dataset, settings)
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


def _test_root(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> Path:
    return settings.test_data.root / dataset.name


def _materialized(
    dataset: "DatasetSource[Any]", settings: "AnalysisConfig", count: int | None = None
) -> list[Any]:
    """A reproducible sample of the clips materialized under the test root."""
    import random

    from .checks.pose_alignment import MaterializedClip
    from .corpus.materialize import MaterializedIndex

    root = _test_root(dataset, settings)
    records = MaterializedIndex(root).records()
    if not records:
        raise FileNotFoundError(f"{root}: run `signworld testdata build {dataset.name}` first")
    clips = [MaterializedClip(r.clip_id, Path(r.video), Path(r.pose)) for r in records]
    random.Random(settings.pose_checks.seed).shuffle(clips)
    return clips if count is None else clips[:count]


@check_app.command("frame-selection")
@reports_user_errors
def frame_selection(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """A3, first part: recomputing the frame selection reproduces the stored indices."""
    from .checks.pose_alignment import frame_selection_report
    from .checks.report import write_report

    settings = _load(config)
    dataset = _source(source, settings)
    options = settings.pose_checks
    report = frame_selection_report(_materialized(dataset, settings, options.selection_clips))
    typer.echo(f"{dataset.name}: {report.reproduced}/{report.clips} selections reproduced")
    typer.echo(write_report(result_path("selezione-frame", f"{dataset.name}.json"), "A3", report))
    if not report.passed:
        raise typer.Exit(1)


@check_app.command("pose-alignment")
@reports_user_errors
def pose_alignment(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the pose estimator.")] = "cpu",
) -> None:
    """A3: re-estimate the pose on decoded frames; compare with the stored pose and shifts."""
    from .checks.pose_alignment import pose_alignment_report
    from .checks.report import write_report
    from .pose.estimator import WholebodyEstimator

    settings = _load(config)
    dataset = _source(source, settings)
    options = settings.pose_checks
    clips = _materialized(dataset, settings, options.alignment_clips)
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
def pose_quality(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """A4: shoulders, box coverage and normalised keypoints over every stored pose."""
    from .checks.contact_sheet import contact_sheet
    from .checks.pose_quality import pose_quality_report
    from .checks.report import write_report

    settings = _load(config)
    dataset = _source(source, settings)
    options = settings.pose_checks
    everything = _materialized(dataset, settings)
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
    from .corpus.languages import written_language

    manifest = _manifest(dataset).select(["clip_id", "caption_language"]).to_pydict()
    language_of = dict(zip(manifest["clip_id"], manifest["caption_language"], strict=True))
    rows = store.clip_rows().to_pydict()
    languages: dict[int, str] = {}
    for clip_id, row in zip(rows["clip_id"], rows["row"], strict=True):
        track = language_of.get(clip_id)
        languages.setdefault(row, written_language(track) if track else "unknown")
    return [languages[row] for row in range(len(languages))]


@experiment_app.command("pose-teachers")
@reports_user_errors
def pose_teachers(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
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

    from .checks.report import write_report
    from .experiments.pose_teachers import run

    settings = _load(config)
    dataset = _source(source, settings)
    clips = _labelled_clips(dataset, settings)
    path = settings.checkpoints[checkpoint].materialize(
        settings.models_root, settings.acquisition().http
    )
    root = _test_root(dataset, settings) / "pose-teachers"
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
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device that encodes the clips.")] = "cuda",
) -> None:
    """§4.4.3: how many directions the S-JEPA latent uses, and its fixed whitening."""
    import torch

    from .checks.report import write_report
    from .experiments.pose_spectrum import measure, save_whitenings

    settings = _load(config)
    dataset = _source(source, settings)
    root = _test_root(dataset, settings) / "pose-teachers"
    weights = root / "sjepa.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: run `signworld experiment pose-teachers` first")
    report, whitenings = measure(
        _labelled_clips(dataset, settings), weights, settings.pose_teachers, torch.device(device)
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


def _labelled_clips(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> list[Any]:
    """The materialized test clips with the video, channel and language the probes read."""
    from .experiments.pose_teachers import LabelledClip

    columns = ["clip_id", "video_id", "channel_id", "sign_language"]
    manifest = _manifest(dataset).select(columns).to_pydict()
    about = {
        clip: (video, channel or "unknown", language or "unknown")
        for clip, video, channel, language in zip(*manifest.values(), strict=True)
    }
    return [
        LabelledClip(clip.clip_id, *about[clip.clip_id], clip.pose)
        for clip in _materialized(dataset, settings)
        if clip.clip_id in about
    ]


# --------------------------------------------------------------------------- frozen video


def _video_settings(settings: "AnalysisConfig") -> Any:
    if settings.video_probes is None:
        raise ValueError("the analysis settings have no video_probes section")
    return settings.video_probes


def _readout_corpus(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> Any:
    """The read-out clips of the pose-teacher comparison: same seeded subset, same split."""
    from .experiments.pose_teachers import PoseCorpus, probe_subset
    from .metrics.probes import video_split

    corpus = PoseCorpus.load(_labelled_clips(dataset, settings), settings.pose_teachers.min_score)
    probe, _, _ = probe_subset(corpus, video_split(corpus.videos), settings.pose_teachers)
    return probe


def _records(dataset: "DatasetSource[Any]", settings: "AnalysisConfig") -> dict[str, Any]:
    from .corpus.materialize import MaterializedIndex

    return {r.clip_id: r for r in MaterializedIndex(_test_root(dataset, settings)).records()}


def _video_encoder(name: str, settings: "AnalysisConfig", device: str) -> tuple[Any, Any]:
    """The frozen encoder and, for V-JEPA 2.1, its predictor and load report."""
    import torch

    from .models.video_encoders import build_encoder

    video = _video_settings(settings)
    if name not in video.encoders:
        raise ValueError(f"unknown encoder {name!r}; available: {', '.join(video.encoders)}")
    options = video.encoders[name]
    checkpoint = None
    if options.checkpoint is not None:
        checkpoint = settings.checkpoints[options.checkpoint].materialize(
            settings.models_root, settings.acquisition().http
        )
    encoder, extra = build_encoder(name, options, video.hub_repo, checkpoint)
    return encoder.to(torch.device(device)), extra


def _stored_frames(clip: Any) -> Any:
    """The 64 selected frames of a stored test clip, (64, H, W, 3) uint8."""
    from .pose.wholebody import PoseTrack
    from .video import ClipReader

    track = PoseTrack.load(clip.pose)
    return ClipReader(_clip_video(clip)).frames(track.frame_indices)


def _clip_video(clip: Any) -> Path:
    return Path(clip.pose).parent.parent / "clips" / f"{clip.clip_id}.mp4"


@experiment_app.command("video-features")
@reports_user_errors
def video_features(
    source: SourceArgument,
    run: Annotated[str, typer.Option(help="Run listed under video_probes.runs.")],
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the encoder.")] = "cuda",
    shard: Annotated[int, typer.Option(min=0, help="Index of this shard.")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of shards.")] = 1,
) -> None:
    """Frozen video features of the read-out clips for PC2, PC3, PC4 and the frame collaudo.

    A finished shard is not recomputed, so a failed submission can be resubmitted unchanged.
    """
    from .acquisition.sharding import Shard
    from .experiments.video_probes import FrameSource, extract

    settings = _load(config)
    dataset = _source(source, settings)
    video = _video_settings(settings)
    options = video.runs[run]
    output = _test_root(dataset, settings) / "video-features" / run
    path = output / f"shard-{shard:05d}-of-{num_shards:05d}.npz"
    if path.is_file():
        typer.echo(f"{path}: already extracted")
        return
    part = Shard(shard, num_shards)
    clips = _readout_corpus(dataset, settings).clips[: options.clips]
    clips = [clip for clip in clips if part.owns(clip.clip_id)]
    encoder, _ = _video_encoder(options.encoder, settings, device)
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

    from .corpus.languages import written_language
    from .experiments.video_probes import TextTargets
    from .metrics.probes import video_split
    from .text.embedding import EmbeddingStore

    stores = sorted(dataset.layout.text_embeddings.glob("*/identity.json"))
    if len(stores) != 1:
        raise FileNotFoundError(f"{dataset.layout.text_embeddings}: expected one embedding set")
    store = EmbeddingStore(stores[0].parent)
    mapping = store.clip_rows().to_pydict()
    row_of = dict(zip(mapping["clip_id"], mapping["row"], strict=True))
    manifest = _manifest(dataset).select(["clip_id", "caption_language"]).to_pydict()
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
def video_probes(source: SourceArgument, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """PC2, PC3, PC4 and the selected-against-contiguous collaudo, from extracted features."""
    from .checks.report import write_report
    from .experiments.video_probes import ClipFeatures, run_scores

    settings = _load(config)
    dataset = _source(source, settings)
    video = _video_settings(settings)
    corpus = _readout_corpus(dataset, settings)
    feature_root = _test_root(dataset, settings) / "video-features"
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

    from .checks.report import write_report
    from .experiments.video_probes import baseline_report, choose_encoder, choose_resolution

    video = _video_settings(settings)
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

    from .checks.report import write_report
    from .experiments.pose_teachers import PoseCorpus
    from .experiments.video_probes import ClipFeatures, frame_selection

    video = _video_settings(settings)
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


@check_app.command("video-reproduction")
@reports_user_errors
def video_reproduction_check(
    source: SourceArgument,
    encoder: Annotated[str, typer.Option(help="Encoder listed under video_probes.encoders.")],
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the encoder.")] = "cuda",
) -> None:
    """Collaudo: every weight loads, and our input path gives the official tokens."""
    from .checks.model_reproduction import video_reproduction
    from .checks.report import write_report

    settings = _load(config)
    dataset = _source(source, settings)
    video = _video_settings(settings)
    frozen, extra = _video_encoder(encoder, settings, device)
    load = extra[1] if extra is not None else {}
    clips = _readout_corpus(dataset, settings).clips[: video.reference_clips]
    report = video_reproduction(frozen, [_stored_frames(c) for c in clips], load)
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
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the teacher.")] = "cuda",
) -> None:
    """Collaudo: the saved S-JEPA teacher reloads whole and reproduces its reference latents."""
    import torch

    from .checks.model_reproduction import pose_reproduction
    from .checks.report import write_report
    from .experiments.pose_teachers import teacher_shape
    from .models.pose_teachers import SJEPATeacher

    settings = _load(config)
    dataset = _source(source, settings)
    video = _video_settings(settings)
    root = _test_root(dataset, settings) / "pose-teachers"
    options = settings.pose_teachers
    teacher = SJEPATeacher(
        teacher_shape(options), momentum=(options.ema_start, 1.0), centre_rate=options.centre_rate
    )
    teacher.load_state_dict(torch.load(root / "sjepa.pt", map_location="cpu"), strict=True)
    teacher.to(torch.device(device)).eval()
    tokens = torch.from_numpy(_readout_corpus(dataset, settings).tokens[: video.reference_clips])
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
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option(help="Device of the model.")] = "cuda",
) -> None:
    """PC6: what the predictor holds, and the multi-level input reproducing it at zero."""
    import torch

    from .checks.model_reproduction import predictor_report
    from .checks.report import write_report

    settings = _load(config)
    dataset = _source(source, settings)
    frozen, extra = _video_encoder(encoder, settings, device)
    if extra is None:
        raise ValueError(f"{encoder}: PC6 applies to V-JEPA 2.1 checkpoints only")
    predictor, load = extra
    predictor.to(torch.device(device)).eval()
    clip = _readout_corpus(dataset, settings).clips[0]
    report = predictor_report(encoder, frozen, predictor, load, _stored_frames(clip))
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


@experiment_app.command("pose-isotropy")
@reports_user_errors
def pose_isotropy(
    source: SourceArgument,
    config: ConfigOption = DEFAULT_CONFIG,
    device: Annotated[
        str, typer.Option(help="Device of the teacher, the flows and k-NN.")
    ] = "cuda",
) -> None:
    """§4.4.3: whitening, RBIG, SINF and a flow on the S-JEPA latent, before and after."""
    import torch

    from .checks.report import write_report
    from .experiments.pose_isotropy import run

    settings = _load(config)
    dataset = _source(source, settings)
    weights = _test_root(dataset, settings) / "pose-teachers" / "sjepa.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: run `signworld experiment pose-teachers` first")
    report = run(
        _labelled_clips(dataset, settings),
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
