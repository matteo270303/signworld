"""WorldSign: three levels on the video, pose and text branches (§4.2, gerarchia §2-§3).

``WorldSign.loss`` runs the passes the stage of the curriculum asks for (gerarchia §6):

* **level 0, the pose**: the pose encoder on the clean sequence and on a view, with
  ``L_inv``, ``L_anchor`` and ``SIGReg_posa`` (posa §4);
* **level 1, the physical level**: the masked clip predicts the pose target, ``E_fis``
  against ``sg(s)``: the video cannot move its target;
* **level 2, the semantic level**: the whole clip predicts the caption embedding from
  ``sg(Enc(x))`` (``VideoBranch.semantic``), with the arm's terms.

The terms combine as §4.5.7, ``(1 - λ)·(predictive) + λ·(SIGReg)``; with the gradient stopped
between the levels each term trains only its own level. In ESP-2 there is no pose and no
physical level.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.experiment.train.config import WorldSignConfig
from signworld.experiment.train.distributed import SINGLE, Distributed
from signworld.loss.worldsign import (
    LossTerms,
    Objective,
    invariance,
    physical_energy,
    physical_energy_per_clip,
)

from .masking import TokenGrid
from .pose_branch import PoseBranch, step_confidence
from .text_branch import TextBranch
from .video_branch import VideoBranch, build_video_branch


@dataclass(frozen=True, slots=True)
class WorldSignBatch:
    """One batch of clips, as the data pipeline delivers it."""

    frames: Tensor
    """(batch, 64, H, W, 3) uint8: the selected frames."""
    pose_tokens: Tensor
    """(batch, 32, 69, 6): S-JEPA's input, ``PoseSequence.tokens``."""
    keypoints: Tensor
    """(batch, 32, 69, 2) p̂: each joint's mean position over the two frames of a step."""
    keypoint_weights: Tensor
    """(batch, 32, 69) c_{t,j}: weight of each joint in L_anchor; averaged, c̄ of E_fis."""
    boxes: Tensor
    """(batch, 32, 4, 4) articulator box of each step, in frame fractions."""
    box_visible: Tensor
    """(batch, 32, 4) bool."""
    captions: Tensor
    """(batch, 768) pre-computed EmbeddingGemma rows."""
    languages: Tensor
    """(batch,) long: each caption's language, as ``LanguageCentering.index`` numbers it."""
    videos: Sequence[str]
    """Source video of each clip: InfoNCE does not use clips of one video as negatives."""
    caption_rows: Tensor | None = None
    """(batch,) long: each clip's caption row; validation counts equal captions as matches."""
    views: Tensor | None = None
    """(batch, 6) augmentation still to apply to ``frames``; None once applied or if none."""
    clip_ids: Sequence[str] | None = None
    durations: Tensor | None = None
    """(batch,) seconds of each clip, for the R@1 by duration band."""
    caption_words: Tensor | None = None
    """(batch,) words of each caption, for the R@1 by caption length."""

    def augmented(self) -> "WorldSignBatch":
        """The frames moved by their views (on the frames' device), which are then spent."""
        if self.views is None:
            return self
        # augmentation imports config only
        from signworld.data.augmentation import apply_views  # noqa: PLC0415

        return dataclasses.replace(self, frames=apply_views(self.frames, self.views), views=None)

    def take(self, count: int) -> "WorldSignBatch":
        """The first ``count`` clips."""
        taken: dict[str, Any] = {
            f.name: getattr(self, f.name)[:count]
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }
        taken["videos"] = list(self.videos[:count])
        if self.clip_ids is not None:
            taken["clip_ids"] = list(self.clip_ids[:count])
        return dataclasses.replace(self, **taken)

    def to(self, device: torch.device) -> "WorldSignBatch":
        moved = {
            f.name: getattr(self, f.name).to(device, non_blocking=True)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }
        return dataclasses.replace(self, **moved)


@dataclass(frozen=True, slots=True)
class StepRandomness:
    """The random draws of one step, reproducible from the seed, the step and the rank."""

    masks: torch.Generator
    """This GPU's own: every GPU draws different masks, as in V-JEPA."""
    directions: torch.Generator
    """The same on every GPU: SIGReg's slices must agree for its statistic to be one."""
    views: torch.Generator
    """This GPU's own: the drawn views of the pose's invariance term."""

    @classmethod
    def at(cls, seed: int, step: int, rank: int = 0) -> "StepRandomness":
        def generator(*key: int) -> torch.Generator:
            state = int(np.random.SeedSequence([seed, *key]).generate_state(1)[0])
            return torch.Generator().manual_seed(state)

        return cls(
            masks=generator(step, rank, 0),
            directions=generator(step, 1),
            views=generator(step, rank, 2),
        )


class WorldSign(nn.Module):
    def __init__(
        self,
        video: VideoBranch,
        pose: PoseBranch | None,
        text: TextBranch,
        objective: Objective,
    ) -> None:
        super().__init__()
        if video.has_physical_level != (pose is not None):
            raise ValueError("the physical predictor and the pose branch go together (ESP-2)")
        self.video = video
        self.pose = pose
        self.text = text
        self.objective = objective

    @property
    def has_physical_level(self) -> bool:
        return self.pose is not None

    def _pose(self) -> PoseBranch:
        if self.pose is None:
            raise RuntimeError("the physical level is disabled (ESP-2)")
        return self.pose

    def pose_terms(
        self,
        batch: WorldSignBatch,
        randomness: StepRandomness,
        record: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Level 0: ``L_inv``, ``L_anchor`` and ``SIGReg_posa``, and the target ``sg(s)``.

        The clean sequence gives ``s``, the anchor's input and the target; three draws of
        nuisances give the other views of ``L_inv`` and ``SIGReg_posa`` (posa §4).
        """
        pose = self._pose()
        views = pose.encode_views(batch.pose_tokens, randomness.views)
        latent = views[0]
        confidence = step_confidence(batch.keypoint_weights)
        present = confidence > 0
        terms = {
            "inv_posa": invariance(views, present),
            "anchor": pose.anchor(latent, batch.keypoints, batch.keypoint_weights),
            "sigreg_posa": self.objective.pose_sigreg(views, present, randomness.directions),
        }
        target = latent.detach()
        if record is not None:
            record |= {"latent": target, "confidence": confidence}
        return terms, target

    def physical_terms(
        self,
        batch: WorldSignBatch,
        target: Tensor,
        step: int,
        total_steps: int,
        randomness: StepRandomness,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Level 1: ``E_fis`` of the masked clip against the pose target, already stopped."""
        grid: TokenGrid = self.video.grid
        if target.shape[1] != grid.steps:
            raise ValueError(f"{target.shape[1]} pose steps for {grid.steps} video steps")
        if target.requires_grad:
            raise ValueError("the pose target must reach E_fis with its gradient stopped")
        confidence = step_confidence(batch.keypoint_weights)
        output = self.video.physical(
            batch.frames, batch.boxes, batch.box_visible, step, total_steps, randomness.masks
        )
        if record is not None:
            record["physical"] = output
        energy = physical_energy(output.predictions, target, confidence, output.context_lambda)
        return {"e_fis": energy}

    def semantic_terms(
        self,
        batch: WorldSignBatch,
        randomness: StepRandomness,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Level 2: the arm's L_pred_sem and SIGReg_sem, from ŷ of the whole clip and ẽ."""
        predicted = self.video.semantic(batch.frames, record)
        target = self.text(batch.captions, batch.languages)
        if record is not None:
            record |= {"predicted": predicted, "target": target}
        return self.objective.semantic_terms(predicted, target, batch.videos, randomness.directions)

    def loss(  # noqa: PLR0913 (the step, its randomness, the passes and a record)
        self,
        batch: WorldSignBatch,
        step: int,
        total_steps: int,
        randomness: StepRandomness,
        *,
        pose: bool = True,
        physical: bool = True,
        semantic: bool = True,
        record: dict[str, Any] | None = None,
    ) -> LossTerms:
        """The objective of one step over the levels the stage runs.

        ``pose`` and ``physical`` are ignored when the physical level is off (ESP-2). The
        physical level needs the pose target: with ``physical`` but not ``pose`` the pose
        terms are measured as diagnostics, outside the total. ``record``, if given, receives
        the intermediate tensors the diagnostics read: the pose target, the confidences, the
        physical read-outs, ŷ, ẽ and the query outputs.
        """
        terms: dict[str, Tensor] = {}
        diagnostics: dict[str, Tensor] = {}
        seen: dict[str, Any] = record if record is not None else {}
        if self.has_physical_level and (pose or physical):
            found, target = self.pose_terms(batch, randomness, seen)
            if pose:
                terms.update(found)
            else:
                diagnostics.update({name: value.detach() for name, value in found.items()})
            if physical:
                terms.update(
                    self.physical_terms(batch, target, step, total_steps, randomness, seen)
                )
        if semantic:
            terms.update(self.semantic_terms(batch, randomness, seen))
        if not terms:
            raise ValueError("no pass selected")
        combined = self.objective.combine(terms)
        return LossTerms(combined.total, combined.parts, diagnostics, self._per_clip(seen))

    @staticmethod
    @torch.no_grad()
    def _per_clip(seen: dict[str, Any]) -> dict[str, Tensor]:
        """Each clip's energies: E_sem through its best hypothesis, E_fis over the step's masks."""
        out: dict[str, Tensor] = {}
        if "predicted" in seen:
            cosine = functional.cosine_similarity(
                seen["predicted"], seen["target"][:, None], dim=-1
            )
            out["e_sem"] = (1 - cosine.amax(dim=1)).float()
        if "physical" in seen:
            output = seen["physical"]
            out["e_fis"] = physical_energy_per_clip(
                output.predictions, seen["latent"], seen["confidence"], output.context_lambda
            ).float()
        return out

    forward = loss
    """DDP synchronises the gradients of what ``forward`` computes: the whole step."""


def assemble_worldsign(
    video: VideoBranch,
    pose: PoseBranch | None,
    config: WorldSignConfig,
    collective: Distributed = SINGLE,
) -> WorldSign:
    """The model around already-built branches (also used by the tests, with tiny ones)."""
    objective = Objective(
        config.losses, config.semantic, physical=config.physical.enabled, collective=collective
    )
    return WorldSign(video, pose, TextBranch(config.text), objective)


def build_worldsign(
    config: WorldSignConfig, grid: TokenGrid | None = None, collective: Distributed = SINGLE
) -> WorldSign:
    """V-JEPA 2.1 loaded from its checkpoint and adapted, the pose encoder built from scratch.

    The language means of the text branch and the keypoint variance of the anchor are fitted
    afterwards, on the training clips (``text.centering.fit``, ``pose.fit``).
    """
    video = build_video_branch(config, grid)
    pose = (
        PoseBranch.from_settings(config.pose_encoder, video.grid.steps)
        if config.physical.enabled
        else None
    )
    return assemble_worldsign(video, pose, config, collective)
