"""Assertions before launching (§4.13.2): if one fails, the run does not start.

Each assertion returns PASS, FAIL or SKIP with the measured detail; the report is written next
to the checkpoints. SKIP is for what cannot be checked yet (P2 without benchmark manifests) and
is logged loudly; it does not stop the run. P13, the single-batch overfit, trains on a copy of
the trainable state and restores it.

P7 compares the masks with V-JEPA's own generator (``src/masks/multiseq_multiblock3d``) on the
same grid: on 32 x 16 x 16 it hides 61 % of the tokens with the short masks and 81 % with the
long ones (30/9). The project document's "≈ 90 %" is V-JEPA's paper figure, which its code
does not produce.
"""

import hashlib
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor

from signworld.data.augmentation import ClipAugmenter
from signworld.data.corpus.manifest import read_manifest
from signworld.data.pose.tokens import JOINTS, STEPS, TOKEN_CHANNELS
from signworld.data.pose.wholebody import LEFT_SHOULDER, RIGHT_SHOULDER
from signworld.experiment.collaudo.contamination import contamination_report
from signworld.models.encoders.video_encoders import PATCH, TUBELET
from signworld.models.worldsign.masking import MultiBlockMasks, TokenGrid
from signworld.models.worldsign.model import StepRandomness, WorldSign, WorldSignBatch

from .budget import ModelBudget
from .config import MaskSpec, WorldSignConfig
from .curriculum import families
from .distributed import SINGLE, Distributed

logger = logging.getLogger(__name__)

TRAINABLE_CEILING: Final = 22_000_000
"""§3.4: the ceiling of trainable parameters."""
EFFECTIVE_BATCH: Final = 128
"""§4.14: the one effective batch of every run."""
V_JEPA_SPECS: Final = (
    MaskSpec(blocks=8, spatial_scale=0.15),
    MaskSpec(blocks=2, spatial_scale=0.7),
)


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Assertion:
    code: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    assertions: list[Assertion]

    @property
    def passed(self) -> bool:
        return all(a.status != Status.FAIL for a in self.assertions)

    def failures(self) -> list[Assertion]:
        return [a for a in self.assertions if a.status == Status.FAIL]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(a) for a in self.assertions]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _check(code: str, ok: bool, detail: str) -> Assertion:
    return Assertion(code, Status.PASS if ok else Status.FAIL, detail)


# ---------------------------------------------------------------- data


def p1_channels(train: pa.Table, held_out: Sequence[pa.Table], official: bool = False) -> Assertion:
    """No channel in common between training and the held-out splits."""
    if official:
        return Assertion("P1", Status.SKIP, "the benchmark's own splits, not ours by channel")
    seen = set(train.column("channel_id").to_pylist())
    shared = {c for table in held_out for c in table.column("channel_id").to_pylist()} & seen
    return _check("P1", not shared, f"{len(shared)} channels in both" if shared else "disjoint")


def p2_contamination(
    clip_ids: set[str], corpus: Path | None, benchmarks: Sequence[Path]
) -> Assertion:
    """No corpus clip overlaps or repeats a benchmark's validation or test clip (§3.9)."""
    if corpus is None or not benchmarks:
        return Assertion("P2", Status.SKIP, "no corpus or benchmark manifest configured")
    manifest = read_manifest(corpus)
    used = manifest.filter(
        pa.array([c in clip_ids for c in manifest.column("clip_id").to_pylist()])
    )
    leaked: set[str] = set()
    for benchmark in benchmarks:
        leaked |= contamination_report(used, read_manifest(benchmark)).excluded
    return _check(
        "P2",
        not leaked,
        f"{len(leaked)} contaminated clips in the index"
        if leaked
        else f"clean against {len(benchmarks)} benchmarks",
    )


# ---------------------------------------------------------------- model


def p3_budget(model: WorldSign) -> Assertion:
    """Every trainable parameter is in the budget of §4.8, and the total is under 22 M."""
    counted = ModelBudget.of(model).total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok = counted == trainable and trainable <= TRAINABLE_CEILING
    return _check(
        "P3",
        ok,
        f"{trainable:,} trainable, {counted:,} in the budget, ceiling {TRAINABLE_CEILING:,}",
    )


def p4_frozen(model: WorldSign) -> Assertion:
    """Nothing outside the curriculum's families can train."""
    listed = {id(p) for group in families(model).values() for p in group}
    stray = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in listed]
    return _check(
        "P4",
        not stray,
        f"trainable outside the families: {stray[:3]}" if stray else "frozen weights frozen",
    )


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def p5_checksums(
    files: dict[str, Path], expected: dict[str, str | None], record: Path
) -> Assertion:
    """Pre-trained weights and embeddings: the expected files, unchanged since the run began."""
    found = {name: sha256(path) for name, path in files.items()}
    wrong = [n for n, value in expected.items() if value is not None and found.get(n) != value]
    changed: list[str] = []
    if record.is_file():
        saved = json.loads(record.read_text())
        changed = [n for n, value in found.items() if saved.get(n) not in (None, value)]
    else:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(found, indent=2) + "\n")
    problems = [f"{n}: not the expected file" for n in wrong] + [
        f"{n}: changed since the run began" for n in changed
    ]
    return _check("P5", not problems, "; ".join(problems) or f"{len(found)} files verified")


def p6_masking(model: WorldSign, frames: Tensor, generator: torch.Generator) -> Assertion:
    """Masked tokens are removed from the encoder's input; context and target are disjoint."""
    problems = []
    grid = model.video.grid
    for mask in model.video.masks(len(frames), generator):
        context, target = set(mask.context[0].tolist()), set(mask.target[0].tolist())
        if context & target:
            problems.append("context and target overlap")
        if len(context) + len(target) != grid.size:
            problems.append("tokens neither visible nor masked")
        if model.video.has_physical_level:
            with torch.no_grad():
                levels = model.video.backbone.context_levels(frames, mask.context.to(frames.device))
            if levels[0].shape[1] != mask.context.shape[1]:
                problems.append(
                    f"encoder saw {levels[0].shape[1]} tokens, {mask.context.shape[1]} visible"
                )
    return _check("P6", not problems, "; ".join(problems) or "removed, disjoint, complete")


def meta_mask_ratio(hub: Path, spec: MaskSpec, grid: TokenGrid, draws: int = 200) -> float:
    """Mean hidden share of V-JEPA's own mask generator for ``spec`` on ``grid``."""
    sys.path.insert(0, str(hub))
    try:
        from src.masks.multiseq_multiblock3d import (  # type: ignore[import-not-found]  # noqa: PLC0415
            _MaskGenerator,
        )
    finally:
        sys.path.remove(str(hub))
    torch.manual_seed(0)
    generator = _MaskGenerator(
        crop_size=(grid.rows * PATCH, grid.columns * PATCH),
        num_frames=grid.steps * TUBELET,
        spatial_patch_size=(PATCH, PATCH),
        temporal_patch_size=TUBELET,
        spatial_pred_mask_scale=(spec.spatial_scale, spec.spatial_scale),
        temporal_pred_mask_scale=(spec.temporal_scale, spec.temporal_scale),
        aspect_ratio=spec.aspect_ratio,
        npred=spec.blocks,
    )
    shares = []
    for _ in range(draws):
        context, target = generator(1)
        shares.append(target[0].shape[-1] / (context[0].shape[-1] + target[0].shape[-1]))
    return float(np.mean(shares))


def p7_mask_statistics(
    specs: Sequence[MaskSpec], grid: TokenGrid, hub: Path, draws: int = 200
) -> Assertion:
    """V-JEPA's masks: 8 short and 2 long tubes, hiding what V-JEPA's own generator hides."""
    kinds = {(s.blocks, round(s.spatial_scale, 3)) for s in specs}
    expected = {(s.blocks, s.spatial_scale) for s in V_JEPA_SPECS}
    if kinds != expected or any(s.temporal_scale != 1.0 for s in specs):
        return _check(
            "P7", False, f"mask kinds {sorted(kinds)} are not V-JEPA's {sorted(expected)} as tubes"
        )
    generator = torch.Generator().manual_seed(0)
    details, ok = [], True
    for spec in specs:
        masks = MultiBlockMasks(spec, grid)
        ratio = float(np.mean([masks(1, generator).ratio for _ in range(draws)]))
        reference = meta_mask_ratio(hub, spec, grid, draws)
        ok &= abs(ratio - reference) < 0.03  # noqa: PLR2004 (sampling noise of 200 draws)
        details.append(f"{spec.blocks}x{spec.spatial_scale}: {ratio:.1%} (V-JEPA {reference:.1%})")
    return _check("P7", ok, "; ".join(details))


def p8_no_flip(augmenter: ClipAugmenter, draws: int = 1000) -> Assertion:
    """No view mirrors the clip: the crop scale stays positive."""
    generator = torch.Generator().manual_seed(0)
    mirrored = sum(augmenter.sample(generator).scale <= 0 for _ in range(draws))
    return _check("P8", mirrored == 0, f"{mirrored} mirrored views in {draws}")


def p9_pose_format(batch: WorldSignBatch) -> Assertion:
    """S-JEPA's input: 32 steps of two frames, 69 joints, x, y and presence per frame."""
    shape = tuple(batch.pose_tokens.shape[1:])
    presence = batch.pose_tokens[..., 2::3]
    ok = shape == (STEPS, len(JOINTS), TOKEN_CHANNELS) and bool(
        ((presence == 0) | (presence == 1)).all()
    )
    return _check(
        "P9",
        ok,
        f"tokens {shape}, presence binary: {bool(((presence == 0) | (presence == 1)).all())}",
    )


def p10_pose_normalised(batch: WorldSignBatch) -> Assertion:
    """Shoulder units: the shoulders' midpoint near 0 and their distance near 1."""
    left, right = JOINTS.index(LEFT_SHOULDER), JOINTS.index(RIGHT_SHOULDER)
    both = (batch.keypoint_weights[..., left] > 0) & (batch.keypoint_weights[..., right] > 0)
    if not bool(both.any()):
        return _check("P10", False, "no step with both shoulders")
    points = batch.keypoints[both]
    centre = ((points[:, left] + points[:, right]) / 2).norm(dim=-1).median().item()
    distance = (points[:, left] - points[:, right]).norm(dim=-1).median().item()
    ok = centre < 0.1 and abs(distance - 1.0) < 0.1  # noqa: PLR2004 (tolerances of §3.6)
    return _check(
        "P10", ok, f"median |midpoint| {centre:.3f}, median shoulder distance {distance:.3f}"
    )


def p11_text_centred(model: WorldSign, captions: Tensor, languages: Sequence[str]) -> Assertion:
    """After the centring, the caption embeddings average 0 in every language."""
    centering = model.text.centering
    device = centering.means.device
    centred = centering(captions.to(device), centering.index(languages).to(device))
    scale = centred.norm(dim=-1).mean().item()
    worst = 0.0
    for name in set(languages):
        rows = torch.tensor([language == name for language in languages], device=device)
        worst = max(worst, centred[rows].mean(dim=0).norm().item() / scale)
    return _check("P11", worst < 1e-3, f"largest language mean / row norm {worst:.2e}")  # noqa: PLR2004


def p12_geometry(augmenter: ClipAugmenter, size: int = 64) -> Assertion:
    """The view moves a keypoint exactly where it moves the pixel under it."""
    generator = torch.Generator().manual_seed(1)
    view = augmenter.sample(generator)
    frame = torch.zeros(1, size, size, 3, dtype=torch.uint8)
    point = torch.tensor([0.4, 0.55])
    x, y = int(point[0] * size), int(point[1] * size)
    frame[0, y - 1 : y + 2, x - 1 : x + 2] = 255
    moved = augmenter.frames(frame, view)[0].float().sum(dim=-1)
    rows, columns = torch.nonzero(moved > moved.max() / 2, as_tuple=True)
    found = torch.stack([columns.float().mean() + 0.5, rows.float().mean() + 0.5]) / size
    expected = augmenter.points((point + 0.5 / size)[None], view)[0]
    error = (found - expected).abs().max().item() * size
    return _check("P12", error < 1.5, f"pixel and keypoint apart by {error:.2f} pixels")  # noqa: PLR2004


def p13_overfit(
    model: WorldSign,
    batch: WorldSignBatch,
    steps: int,
    learning_rate: float = 1e-3,
    *,
    bf16: bool = False,
) -> Assertion:
    """One batch, SIGReg off: the predictive loss must fall; the model is restored afterwards.

    The forward runs under the training's autocast: in float32 the whole model on a GPU's
    batch does not fit in memory.
    """
    if steps == 0:
        return Assertion("P13", Status.SKIP, "done when the run began")
    trainable = [p for p in model.parameters() if p.requires_grad]
    saved = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    losses = []
    device = batch.frames.device
    try:
        for step in range(steps):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                terms = model.loss(batch, step, steps, StepRandomness.at(0, 0)).parts
            total = torch.stack([v for k, v in terms.items() if not k.startswith("sigreg")]).sum()
            optimizer.zero_grad(set_to_none=True)
            total.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(total))
    finally:
        with torch.no_grad():
            for parameter, value in zip(trainable, saved, strict=True):
                parameter.copy_(value)
        for parameter in trainable:
            parameter.grad = None
    ok = losses[-1] < 0.5 * losses[0]
    return _check("P13", ok, f"loss {losses[0]:.4f} -> {losses[-1]:.4f} in {steps} steps")


def p14_pose_isolated(
    model: WorldSign, batch: WorldSignBatch, generator: torch.Generator
) -> Assertion:
    """The pose never reaches the physical predictor: its output has no gradient wrt the boxes."""
    if not model.video.has_physical_level:
        return Assertion("P14", Status.SKIP, "no physical level (ESP-2)")
    boxes = batch.boxes.clone().requires_grad_(True)
    output = model.video.physical(batch.frames, boxes, batch.box_visible, 0, 1, generator)
    total = sum(p.masked.float().sum() + p.visible.float().sum() for p in output.predictions)
    (gradient,) = torch.autograd.grad(total, boxes, allow_unused=True)  # type: ignore[arg-type]
    leak = gradient is not None and bool(gradient.abs().sum() > 0)
    return _check(
        "P14",
        not leak,
        "the boxes only choose tokens" if not leak else "gradient flows from the pose",
    )


def p15_infonce(model: WorldSign) -> Assertion:
    """Arm C: pairs of different clips of one video never enter the denominator."""
    infonce = model.objective.infonce
    if infonce is None:
        return Assertion("P15", Status.SKIP, "not arm C")
    embeddings = torch.randn(3, 8, device=infonce.log_temperature.device)
    logits = infonce.logits(embeddings, embeddings, ["a", "a", "b"])
    ok = bool(torch.isinf(logits[0, 1]) and torch.isinf(logits[1, 0])) and bool(
        torch.isfinite(logits.diagonal()).all()
    )
    return _check(
        "P15", ok, "same-video pairs excluded" if ok else "same-video pairs in the denominator"
    )


def p16_batch(config: WorldSignConfig, collective: Distributed, per_gpu: int) -> Assertion:
    """SIGReg sees the whole effective batch: 128 clips over every GPU."""
    total = int(collective.all_sum(torch.tensor(float(per_gpu), device=collective.device)).item())
    ok = config.training.batch_size == EFFECTIVE_BATCH == total
    return _check(
        "P16",
        ok,
        f"{per_gpu} clips x {collective.world_size} GPUs = {total}; SIGReg on {total} per "
        f"modality, up to {total} x 32 per articulator",
    )


def run_preflight(  # noqa: PLR0913, PLR0917 (what the assertions look at)
    model: WorldSign,
    config: WorldSignConfig,
    splits: dict[str, pa.Table],
    batch: WorldSignBatch,
    captions: tuple[Tensor, list[str]],
    record: Path,
    collective: Distributed = SINGLE,
    *,
    overfit_steps: int = 30,
    checksum_files: dict[str, Path] | None = None,
) -> PreflightReport:
    """Every assertion of §4.13.2 on the model, the data and one batch of this GPU."""
    train = splits["train"]
    held_out = [table for name, table in splits.items() if name != "train"]
    augmenter = ClipAugmenter(config.augmentation)
    generator = torch.Generator().manual_seed(0)
    clip_ids = {c for table in splits.values() for c in table.column("clip_id").to_pylist()}
    files = checksum_files or {}
    expected = {
        "encoder": config.encoder.checkpoint_sha256,
        "pose_encoder": config.pose_encoder.checkpoint_sha256,
    }
    checks: list[Callable[[], Assertion]] = [
        lambda: p1_channels(train, held_out, config.data.split_source == "manifest"),
        lambda: p2_contamination(clip_ids, config.data.manifest, config.data.benchmarks),
        lambda: p3_budget(model),
        lambda: p4_frozen(model),
        lambda: p5_checksums(files, {k: v for k, v in expected.items() if k in files}, record),
        lambda: p6_masking(model, batch.frames[:1], generator),
        lambda: p7_mask_statistics(config.masking.specs, model.video.grid, config.encoder.hub_repo),
        lambda: p8_no_flip(augmenter),
        lambda: p9_pose_format(batch),
        lambda: p10_pose_normalised(batch),
        lambda: p11_text_centred(model, *captions),
        lambda: (
            p12_geometry(augmenter)
            if config.augmentation.enabled
            else Assertion("P12", Status.SKIP, "augmentation off: frames and keypoints untouched")
        ),
        lambda: p13_overfit(model, batch, overfit_steps, bf16=config.training.precision == "bf16"),
        lambda: p14_pose_isolated(model, batch, generator),
        lambda: p15_infonce(model),
        lambda: p16_batch(config, collective, len(batch.videos)),
    ]
    assertions = []
    for check in checks:
        result = check()
        assertions.append(result)
        log = logger.error if result.status == Status.FAIL else logger.info
        log("%s %s: %s", result.code, result.status.value, result.detail)
    return PreflightReport(assertions)


class PreflightError(RuntimeError):
    """An assertion of §4.13.2 failed: the run does not start."""

    def __init__(self, report: PreflightReport) -> None:
        failed = "; ".join(f"{a.code}: {a.detail}" for a in report.failures())
        super().__init__(f"preflight failed, the run does not start. {failed}")
        self.report = report


def enforce(report: PreflightReport) -> None:
    if not report.passed:
        raise PreflightError(report)
    for assertion in report.assertions:
        if assertion.status == Status.SKIP:
            logger.warning("%s skipped: %s", assertion.code, assertion.detail)
