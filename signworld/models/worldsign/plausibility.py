"""Plausibility as violation of expectation (§4.12.4): Ē_fis of clips intact and manipulated.

Video and pose are manipulated together, a step (two frames) at a time: time reversed, steps
skipped, steps frozen; controls: the pose shifted in time against the video (it must rise if
the predictor uses the video), the colour changed (it must stay within the variability between
masks). The same masks serve intact and manipulated clips. The document asks for the
manipulations before the frame selection, which needs the pose estimated again on the
manipulated video: acting on the 64 selected frames is the approximation that needs no new
pose.
"""

import dataclasses
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from signworld.experiment.train.distributed import SINGLE, Distributed
from signworld.loss.worldsign import physical_energy_per_clip

from .model import StepRandomness, WorldSign, WorldSignBatch
from .pose_branch import articulator_confidence

Manipulation = Callable[[WorldSignBatch], WorldSignBatch]


def _steps(
    batch: WorldSignBatch, order: Tensor, *, video: bool = True, pose: bool = True
) -> WorldSignBatch:
    """The clip rebuilt from its steps in ``order`` (32 step indices), video and/or pose."""
    changes: dict[str, Any] = {}
    if video:
        frames = torch.stack([order * 2, order * 2 + 1], dim=1).flatten().to(batch.frames.device)
        changes["frames"] = batch.frames[:, frames]
    if pose:
        index = order.to(batch.pose_tokens.device)
        for name in ("pose_tokens", "keypoints", "keypoint_weights", "boxes", "box_visible"):
            changes[name] = getattr(batch, name)[:, index]
    return dataclasses.replace(batch, **changes)


def reverse_time(batch: WorldSignBatch) -> WorldSignBatch:
    """The arrow of time: frames, pose and boxes played backwards."""
    steps = batch.pose_tokens.shape[1]
    reversed_batch = _steps(batch, torch.arange(steps - 1, -1, -1))
    tokens = reversed_batch.pose_tokens
    swapped = torch.cat([tokens[..., 3:], tokens[..., :3]], dim=-1)  # the two frames of a step
    frames = batch.frames.flip(1)
    return dataclasses.replace(reversed_batch, pose_tokens=swapped, frames=frames)


def skip_steps(batch: WorldSignBatch, start: int = 12, length: int = 8) -> WorldSignBatch:
    """Continuity: ``length`` steps cut out, the end held to keep 32 steps."""
    steps = batch.pose_tokens.shape[1]
    kept = [i for i in range(steps) if not start <= i < start + length]
    order = torch.tensor(kept + [kept[-1]] * (steps - len(kept)))
    return _steps(batch, order)


def freeze_steps(batch: WorldSignBatch, start: int = 12, length: int = 8) -> WorldSignBatch:
    """Inertia: one step repeated ``length`` times, then the motion resumes where it was."""
    steps = batch.pose_tokens.shape[1]
    order = torch.tensor([start if start <= i < start + length else i for i in range(steps)])
    return _steps(batch, order)


def shift_pose(batch: WorldSignBatch, shift: int = 4) -> WorldSignBatch:
    """Control: the pose target shifted in time against the video (boxes follow the video)."""
    steps = batch.pose_tokens.shape[1]
    order = (torch.arange(steps) + shift) % steps
    shifted = _steps(batch, order, video=False)
    return dataclasses.replace(shifted, boxes=batch.boxes, box_visible=batch.box_visible)


def change_colour(batch: WorldSignBatch, factor: float = 1.4) -> WorldSignBatch:
    """Negative control: brighter frames, nothing physical changed."""
    frames = (batch.frames.float() * factor).clamp(0, 255).to(torch.uint8)
    return dataclasses.replace(batch, frames=frames)


MANIPULATIONS: dict[str, tuple[Manipulation, str]] = {
    "time_reversed": (reverse_time, "rise"),
    "steps_skipped": (skip_steps, "rise"),
    "steps_frozen": (freeze_steps, "rise"),
    "pose_shifted": (shift_pose, "rise"),
    "colour_changed": (change_colour, "flat"),
}
"""Name → (manipulation, expected effect on Ē_fis) of §4.12.4."""


@torch.no_grad()
def plausibility(model: WorldSign, batch: WorldSignBatch, masks: int) -> Tensor:
    """(masks, batch) E_fis of every clip under ``masks`` fixed random masks of each kind."""
    pose = model.pose
    if pose is None:
        raise RuntimeError("no physical level")
    latent = pose.targets(batch.pose_tokens)
    confidence = articulator_confidence(batch.keypoint_weights)
    energies = []
    for draw in range(masks):
        randomness = StepRandomness.at(1234, draw)  # the same masks for intact and manipulated
        output = model.video.physical(
            batch.frames, batch.boxes, batch.box_visible, 0, 1, randomness.masks
        )
        energies.append(
            physical_energy_per_clip(output.predictions, latent, confidence, output.context_lambda)
        )
    return torch.stack(energies).float().cpu()


@dataclass(frozen=True, slots=True)
class PlausibilityResult:
    name: str
    expected: str
    mean_increase: float
    """Mean over clips of Ē_fis(manipulated) - Ē_fis(intact)."""
    share_increased: float
    mask_variability: float
    """Mean over clips of the standard error of Ē_fis between masks: the noise floor."""
    passed: bool


def plausibility_tests(
    model: WorldSign,
    batches: Iterable[WorldSignBatch],
    device: torch.device,
    masks: int = 8,
    collective: Distributed = SINGLE,
) -> list[PlausibilityResult]:
    """Every manipulation of §4.12.4 against the intact clips, with the same masks.

    Each GPU measures its own clips; the energies of every GPU are gathered before the means.
    """
    was_training = model.training
    model.eval()
    intact_parts: list[Tensor] = []
    manipulated: dict[str, list[Tensor]] = {name: [] for name in MANIPULATIONS}
    for batch in batches:
        clips = batch.to(device).augmented()
        intact_parts.append(plausibility(model, clips, masks))
        for name, (manipulate, _) in MANIPULATIONS.items():
            manipulated[name].append(plausibility(model, manipulate(clips), masks))
    model.train(was_training)
    local = {"intact": torch.cat(intact_parts, dim=1)} | {
        name: torch.cat(parts, dim=1) for name, parts in manipulated.items()
    }
    gathered = collective.gather_objects([local])
    joined = {k: torch.cat([g[k] for g in gathered], dim=1) for k in local}
    intact = joined["intact"]
    floor = float((intact.std(0) / math.sqrt(masks)).mean()) if masks > 1 else 0.0
    results = []
    for name, (_, expected) in MANIPULATIONS.items():
        energy = joined[name]
        increase = energy.mean(0) - intact.mean(0)
        mean = float(increase.mean())
        passed = mean > floor if expected == "rise" else abs(mean) <= 2 * floor
        results.append(
            PlausibilityResult(
                name, expected, mean, float((increase > 0).float().mean()), floor, passed
            )
        )
    return results


def as_measures(results: Iterable[PlausibilityResult]) -> dict[str, float]:
    """The results as flat measures for the metrics log."""
    out: dict[str, float] = {}
    for result in results:
        out[f"plausibility_{result.name}_increase"] = result.mean_increase
        out[f"plausibility_{result.name}_share"] = result.share_increased
        out[f"plausibility_{result.name}_passed"] = float(result.passed)
        out["plausibility_mask_variability"] = result.mask_variability
    return out
