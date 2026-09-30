"""The measurements of a set of clips, shared by validation and test (§4.12, §4.13).

``collect`` runs the model once per batch, as in training but without gradient, and keeps
what the measures read. ``split_measures`` derives from it:

* **retrieval** both ways (``validation.retrieval``): R@k, Precision@k, Recall@k, MRR, MedR,
  the R@1 tolerant to near-duplicate captions, R@1 by caption language, clip duration and
  caption length, the chance level, bootstrap intervals where asked;
* the **loss** of the objective and each of its terms, over the clips (``loss_*``);
* **E_sem by caption language**;
* the **physical read-outs** (gamma, R² overall, per articulator and per mask coverage, the
  dynamics against its baseline) and the **keypoint errors** against interpolation and
  constant velocity, averaged over the batches;
* **alignment and uniformity** of Wang and Isola [Lett. 40] on the unit sphere;
* the **geometry of ŷ and ẽ**: effective rank, IsoScore, condition number, SIGReg;
* the **noise test**, **hubness**, the **modality gap** and the **2x2 energy table**.

``model_measures`` adds the leak test and the attention of the queries on the first clips;
``pose_target_measures`` the isotropy and kinematic content of the pose target.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from ..metrics.geometry import centered, condition_number, effective_rank, isoscore
from ..metrics.retrieval import hubness
from ..pose.tokens import JOINT_ARTICULATOR
from .config import WorldSignConfig
from .distributed import SINGLE, Distributed
from .losses import uniformity
from .model import StepRandomness, WorldSign, WorldSignBatch
from .readings import (
    Spread,
    attention_readings,
    energy_table,
    keypoint_errors,
    modality_gap,
    physical_readings,
    ridge_r2,
    sigreg_ratio,
)
from .readout import membership
from .validation import RetrievalScores, retrieval, similarity

PARTS = ("body", "left", "right", "face")


@dataclass(frozen=True, slots=True)
class Collected:
    """What one pass over the clips produced, gathered from every GPU, on the CPU."""

    tensors: dict[str, Tensor]
    """Per clip: ``predicted`` (n, K, d), ``texts``, ``captions``, ``rows``, ``languages``,
    ``durations``, ``words``, ``e_sem``; with the physical pass ``e_fis``; with the noise test
    ``noise``; with the pose target ``latents``, ``keypoints``, ``weights``."""
    averages: dict[str, float]
    """``loss_*`` terms and the physical read-outs, averaged over the clips."""


def _noise(frames: Tensor, generator: torch.Generator) -> Tensor:
    noise = torch.randint(0, 256, frames.shape, generator=generator, dtype=torch.uint8)
    return noise.to(frames.device)


def _physical(model: WorldSign, clips: WorldSignBatch, record: dict[str, Any]) -> dict[str, float]:
    """The read-outs and keypoint errors of one batch's physical pass."""
    pose = model.pose
    if pose is None:
        return {}
    latent, confidence = record["latent"].float(), record["confidence"].float()
    predictions = record["physical"].predictions
    groups = torch.as_tensor(JOINT_ARTICULATOR, device=latent.device)
    found = physical_readings(predictions, latent, confidence)
    return found | keypoint_errors(
        predictions, latent, pose.decoders, clips.keypoints, clips.keypoint_weights, groups
    )


@torch.no_grad()
def collect(  # noqa: PLR0913 (the model, the clips, where, and what to keep)
    model: WorldSign,
    batches: Iterable[WorldSignBatch],
    device: torch.device,
    collective: Distributed = SINGLE,
    *,
    bf16: bool = True,
    physical: bool = True,
    noise: bool = True,
    pose_target: bool = False,
    seed: int = 0,
) -> Collected:
    """One pass of the model over ``batches`` with the quantities the measures read.

    Every GPU must pass the same number of batches of the same sizes: the objective's
    collectives (SIGReg, InfoNCE) run once per batch, as in training.
    """
    was_training = model.training
    model.eval()
    parts: dict[str, list[Tensor]] = defaultdict(list)
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    generator = torch.Generator().manual_seed(seed)
    physical = physical and model.pose is not None

    def add(name: str, value: float, weight: float) -> None:
        if not math.isnan(value):
            sums[name] += value * weight
            counts[name] += weight

    for number, batch in enumerate(batches):
        clips = batch.to(device).augmented()
        size = float(len(clips.videos))
        record: dict[str, Any] = {}
        randomness = StepRandomness.at(seed, number, collective.rank)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            terms = model.loss(clips, 1, 1, randomness, physical=physical, record=record)
            noisy = model.video.semantic(_noise(clips.frames, generator)) if noise else None
        add("loss_total", float(terms.total), size)
        for name, value in (terms.parts | terms.diagnostics).items():
            add(f"loss_{name}", float(value), size)
        parts["predicted"].append(record["predicted"].float().cpu())
        parts["texts"].append(record["target"].float().cpu())
        parts["captions"].append(clips.captions.float().cpu())
        parts["languages"].append(clips.languages.cpu())
        parts["e_sem"].append(terms.samples["e_sem"].cpu())
        for name, column in (
            ("rows", clips.caption_rows),
            ("durations", clips.durations),
            ("words", clips.caption_words),
        ):
            if column is not None:
                parts[name].append(column.cpu())
        if noisy is not None:
            parts["noise"].append(noisy.float().cpu())
        if physical:
            parts["e_fis"].append(terms.samples["e_fis"].cpu())
            for name, reading in _physical(model, clips, record).items():
                add(name, reading, size)
        if pose_target and model.pose is not None:
            parts["latents"].append(model.pose.targets(clips.pose_tokens).float().cpu())
            parts["keypoints"].append(clips.keypoints.float().cpu())
            parts["weights"].append(clips.keypoint_weights.float().cpu())
    model.train(was_training)
    local: dict[str, Any] = {
        "tensors": {k: torch.cat(v) for k, v in parts.items()},
        "sums": dict(sums),
        "counts": dict(counts),
    }
    gathered = collective.gather_objects([local])
    tensors = {k: torch.cat([g["tensors"][k] for g in gathered]) for k in local["tensors"]}
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    for g in gathered:
        for name, total in g["sums"].items():
            totals[name] += total
            weights[name] += g["counts"][name]
    averages = {k: totals[k] / weights[k] for k in totals if weights[k] > 0}
    return Collected(tensors, averages)


def _geometry(name: str, rows: Tensor) -> dict[str, float]:
    x = rows.float()
    return {
        f"{name}_effective_rank": effective_rank(centered(x)),
        f"{name}_isoscore": isoscore(x),
        f"{name}_condition_number": condition_number(x),
        f"{name}_sigreg": sigreg_ratio(x),
    }


def split_measures(
    collected: Collected,
    names: Sequence[str],
    config: WorldSignConfig,
    *,
    bootstrap: Sequence[str] = (),
    physical_prefix: str = "",
) -> tuple[RetrievalScores, dict[str, float]]:
    """Every measure the pass of ``collect`` allows, and the retrieval scores that decide.

    ``names`` are the caption languages in the order of their index. ``physical_prefix``
    goes before the physical read-outs, the keypoint errors and E_sem by language: the
    monitor reads the same quantities on training batches under their plain names.
    """
    t = collected.tensors
    scores = retrieval(
        t["predicted"],
        t["texts"],
        t["rows"],
        captions=t["captions"],
        duplicate_cosine=config.diagnostics.duplicate_cosine,
        languages=t["languages"],
        names=names,
        durations=t.get("durations"),
        words=t.get("words"),
        bootstrap=bootstrap,
    )
    out = scores.as_log()
    physical = {k: v for k, v in collected.averages.items() if not k.startswith("loss_")}
    out |= {k: v for k, v in collected.averages.items() if k.startswith("loss_")}
    if "dynamics_r2" in physical:
        physical["dynamics_margin"] = physical["dynamics_r2"] - physical["dynamics_baseline_r2"]
    baselines = [
        physical[k]
        for k in ("keypoint_error_interpolation", "keypoint_error_constant_velocity")
        if not math.isnan(physical.get(k, math.nan))
    ]
    if baselines and not math.isnan(physical.get("keypoint_error_model", math.nan)):
        physical["keypoint_margin"] = min(baselines) - physical["keypoint_error_model"]
    for index, name in enumerate(names):
        rows = t["languages"] == index
        if int(rows.sum()) > 0:
            physical[f"e_sem_{name}"] = float(t["e_sem"][rows].mean())
    out |= {physical_prefix + k: v for k, v in physical.items()}

    predicted, texts = t["predicted"], t["texts"]
    hypotheses = predicted.flatten(0, 1)
    best = predicted[
        torch.arange(len(texts)),
        functional.cosine_similarity(predicted, texts[:, None], dim=-1).argmax(dim=1),
    ]
    unit_video = functional.normalize(best, dim=-1)
    unit_text = functional.normalize(texts, dim=-1)
    out["alignment"] = float((unit_video - unit_text).pow(2).sum(-1).mean())
    out["uniformity_video"] = float(uniformity(hypotheses, config.losses.uniformity_t))
    out["uniformity_text"] = float(uniformity(texts, config.losses.uniformity_t))
    out |= _geometry("y", hypotheses) | _geometry("text", texts)

    if "noise" in t:
        noisy = retrieval(t["noise"], texts, t["rows"])
        out["noise_t2v_r1"] = noisy.t2v[1]
        out["noise_drop"] = 1 - noisy.t2v[1] / scores.t2v[1] if scores.t2v[1] > 0 else math.nan
    out["hubness"] = hubness(similarity(predicted, texts), min(10, len(texts)))
    out["modality_gap"] = modality_gap(predicted[:, 0], texts)
    if "e_fis" in t:
        semantic = 1 - functional.cosine_similarity(predicted.mean(1), texts, dim=-1)
        out |= energy_table(t["e_fis"], semantic)
    return scores, out


# ------------------------------------------------------------------ measures with their own pass


def leak_change(model: WorldSign, clips: WorldSignBatch, device: torch.device) -> float:
    """Largest change of the physical predictions when masked pixels come from another clip."""
    video = model.video
    physical = video.physical_predictor
    if physical is None or len(clips.frames) < 2:  # noqa: PLR2004
        return math.nan
    grid = video.grid
    mask = video.masks(len(clips.frames), torch.Generator().manual_seed(5))[-1].to(device)
    hidden = torch.zeros(grid.size, dtype=torch.bool, device=device)
    hidden[mask.target[0]] = True
    spatial = hidden.view(grid.steps, grid.rows, grid.columns)[0]
    patch = clips.frames.shape[2] // grid.rows
    pixels = spatial.repeat_interleave(patch, 0).repeat_interleave(patch, 1)[None, None, :, :, None]
    mixed = torch.where(pixels, clips.frames.roll(1, dims=0), clips.frames)
    outputs = [
        physical.tokens(video.backbone.context_levels(frames, mask.context), mask)
        for frames in (clips.frames, mixed)
    ]
    scale = outputs[0].float().abs().mean().clamp_min(1e-12)
    return float((outputs[0].float() - outputs[1].float()).abs().max() / scale)


def attention(model: WorldSign, clips: WorldSignBatch) -> dict[str, float]:
    """Entropy of the queries' attention and its share on the articulators' boxes."""
    video = model.video
    weights = video.semantic_predictor.query_attention(video.backbone.tokens(clips.frames))
    members = membership(clips.boxes, clips.box_visible, video.grid.rows, video.grid.columns)
    out = attention_readings(weights.float(), members.float())
    if out["articulator_area"] > 0:
        out["attention_on_boxes_ratio"] = out["attention_on_articulators"] / out["articulator_area"]
    return out


@torch.no_grad()
def model_measures(
    model: WorldSign, batch: WorldSignBatch, device: torch.device, clips: int
) -> dict[str, float]:
    """The leak test and the attention of the queries, on the first ``clips`` of ``batch``."""
    was_training = model.training
    model.eval()
    small = batch.take(clips).to(device).augmented()
    out = attention(model, small)
    if model.pose is not None:
        out["leak_change"] = leak_change(model, small, device)
    model.train(was_training)
    return out


def pose_target_measures(
    latent: Tensor, keypoints: Tensor, weights: Tensor
) -> dict[str, dict[str, float]]:
    """Per articulator: IsoScore, effective rank and SIGReg of the pose target ``s``, and the R²
    of a ridge from ``s`` to the keypoint positions and velocities (fit on the first half of
    the clips, read on the second).

    ``latent`` (clips, steps, 4, C), ``keypoints`` (clips, steps, 69, 2), ``weights``
    (clips, steps, 69). Keyed by measure, then by articulator name.
    """
    groups = torch.as_tensor(JOINT_ARTICULATOR)
    clips, steps = latent.shape[:2]
    fit = torch.arange(clips) < clips // 2
    velocity = keypoints[:, 1:] - keypoints[:, :-1]
    moving = weights[:, 1:] * weights[:, :-1]
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for index, name in enumerate(PARTS):
        rows = latent[:, :, index].flatten(0, 1)
        spread = Spread.of(rows)
        out["isoscore"][name] = spread.isoscore
        out["rank"][name] = spread.effective_rank
        out["sigreg"][name] = sigreg_ratio(rows)
        joints = groups == index
        out["r2_position"][name] = ridge_r2(
            rows,
            keypoints[:, :, joints].flatten(0, 1).flatten(1),
            weights[:, :, joints].mean(-1).flatten(),
            fit[:, None].expand(clips, steps).flatten(),
        )
        out["r2_velocity"][name] = ridge_r2(
            latent[:, 1:, index].flatten(0, 1),
            velocity[:, :, joints].flatten(0, 1).flatten(1),
            moving[:, :, joints].mean(-1).flatten(),
            fit[:, None].expand(clips, steps - 1).flatten(),
        )
    return dict(out)
